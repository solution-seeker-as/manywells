"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The ManyWells side of the table fluids' build (specs/sampling.md, SMP-112 and SMP-116): which wells of a dataset take
a table, with the draws that table_build.py makes their fluids from (DIR/wells.json), and each well's GOR axis from
its life, with its fluid's values in it (DIR/axes.json). The build runs in four steps, two here and two in NeqSim's
environment (table_build.py, run by its path):

1. write_table_wells: the table wells among draw indices 0 to n - 1, and their draws and grids (DIR/wells.json);
2. table_build.py DIR: each well's fluid (DIR/fluids/kK.json);
3. write_axes: each fluid's GOR axis (DIR/axes.json);
4. table_build.py DIR, again: each well's table set and manifest (DIR/kK.csv, DIR/kK.json), which tables._read_table
   reads.

Each step resumes: a well already in a file keeps its record, so a dataset that needs more draw indices extends its
tables by running the four steps again with a larger n.

DIR/wells.json holds `kind` ('sol' or 'nsol'), `seed`, `wells_range` ([0, n]: the draw indices it covers) and `wells`,
one record per table well, in draw order: `well` (its draw index), `fluid_class`, `coverage_tail`, the drawn `api`,
`sg_gas`, `p_sat`, `p_i`, `p_r` (at the drawn stage), `T_r`, `T_s`, `p_sep` and `T_sep`, `R_si` (the drawn initial
GOR), `composition_seed`, the grid's `T_axis` and `p_top` (table_build._well_grid) and `source` (table_build._fluid_source).
DIR/axes.json holds, by draw index as a string, `gors` (Sm3/Sm3, ascending), `p_r_range` (bar) and `gor_range`
(Sm3/Sm3), the reservoir pressures and producing GORs that the axis covers before the GOR's drift, and the fluid's
`R_i`; or `error`, the traceback of a well whose axis could not be made, which table_build.py records as the well's
failure.
"""

import json
import math
import traceback
from pathlib import Path

from manywells.sampling.v2.draws import draw_well
from manywells.sampling.v2.life import Life, _sol2_date
from manywells.sampling.v2.reservoir import _GOR_NODES, _OILS, _producing_gor
from manywells.sampling.v2.samples import _P_R_DRIFT, _RATIO_DRIFT
from manywells.sampling.v2.table_build import _fluid_source, _well_grid
from manywells.sampling.v2.tables import _TABLE_CLASSES
from manywells.sampling.wells import rng_for

_GOR_STEP = 1.25  # The largest ratio of neighbouring GORs on a table set's axis  # spec: SMP-116
_MARGIN = 1e-6  # The axis's margin beyond the GORs it covers, for the CSV's rounding to ten significant digits
_RECORD = ('well', 'fluid_class', 'coverage_tail', 'api', 'sg_gas', 'p_sat', 'p_i', 'p_r', 'T_r', 'T_s', 'p_sep')
_RECORD += ('T_sep',)


def _table_well(seed: int, k: int) -> dict | None:  # spec: SMP-112
    """
    Well k's record of wells.json (the module's docstring) if its fluid class takes a table, else None. The
    `composition` generator, rng_for(seed, k, 'v2', 'composition'), draws the composition's seed, then the source.

    :param seed: The dataset's seed
    :param k: The well's draw index
    """
    w, _ = draw_well(seed, k)
    if w['fluid_class'] not in _TABLE_CLASSES:
        return None
    r = rng_for(seed, k, 'v2', 'composition')  # spec: SMP-109
    comp_seed, u_source, u_key = int(r.integers(2**63)), r.random(), r.random()
    T_axis, p_top = _well_grid(w['T_s'], w['T_r'], w['p_i'])  # spec: SMP-116
    rec = {key: w[key] for key in _RECORD}
    rec.update(R_si=w['R_i'], composition_seed=comp_seed, T_axis=[float(T) for T in T_axis], p_top=float(p_top))
    rec['source'] = _fluid_source(w['fluid_class'], u_source, u_key)
    return rec


def _table_wells(kind: str, seed: int, k0: int, k1: int) -> list:  # spec: SMP-112
    """The records of the table wells among draw indices k0 to k1 - 1 of a dataset (_table_well), in draw order."""
    if kind not in ('sol', 'nsol'):
        raise ValueError(f"kind must be 'sol' or 'nsol', got {kind!r}")
    return [rec for k in range(k0, k1) if (rec := _table_well(seed, k)) is not None]


def write_table_wells(tables_dir, kind: str, seed: int, n: int) -> dict:  # spec: SMP-116
    """
    Step 1 of the build: DIR/wells.json with the table wells among draw indices 0 to n - 1. A file already there for
    the same dataset and seed keeps its records and is extended to n; one for another dataset or seed is refused.

    :param tables_dir: The build's folder, made if missing
    :param kind: 'sol' or 'nsol'
    :param seed: The dataset's seed
    :param n: The draw indices to cover, 0 to n - 1
    :return: The file's content
    """
    tables_dir = Path(tables_dir)
    path = tables_dir / 'wells.json'
    wells, k0 = [], 0
    if path.exists():
        spec = json.loads(path.read_text())
        if (spec['kind'], spec['seed']) != (kind, seed):
            raise ValueError(f'{path} holds the table wells of {spec["kind"]}, seed {spec["seed"]}, not {kind}, {seed}')
        if spec['wells_range'][0] != 0:
            raise ValueError(f'{path} does not start at draw index 0')
        wells, k0 = spec['wells'], spec['wells_range'][1]
    spec = dict(kind=kind, seed=seed, wells_range=[0, max(n, k0)], wells=wells + _table_wells(kind, seed, k0, n))
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, spec)
    return spec


def _gor_axis(kind: str, seed: int, k: int, hook: dict) -> dict:  # spec: SMP-116
    """
    A table well's GOR axis (PVT-TAB-7): GORs at most GOR_STEP apart from the least to the largest producing GOR that
    its samples reach, with the fluid's values in the well. sol-2: the reservoir pressure within its drift of its
    date's (SMP-96); nsol-2: the life's trend from its start to its end (SMP-85). Each GOR's drift (SMP-96,
    SMP-100) widens the range at both ends. An oil's GOR is linear in p_r/p_sat between GOR_NODES, so its extremes over
    a range of pressures lie at the range's ends or at a node.

    :param kind: 'sol' or 'nsol'
    :param seed: The dataset's seed
    :param k: The well's draw index
    :param hook: The fluid's values that replace the fluid block's draws (table_build._build_fluid's `hook`)
    :return: A dict of `gors`, `p_r_range`, `gor_range` and `R_i` (the module's docstring)
    """
    life = Life(seed, k, fluid=hook)
    w = life.w0
    if kind == 'sol':  # The date's reservoir pressure, as sol2_well takes it, with its drift
        _, t = _sol2_date(life)
        p_r = life.p_r(min(t, life.T_life))
        p_lo, p_hi = (1 - _P_R_DRIFT) * p_r, (1 + _P_R_DRIFT) * p_r
    else:  # The life's trend, from its start to its end
        p_lo, p_hi = life.p_r(life.T_life), life.p_r(0.0)
    p = [p_lo, p_hi, w['p_sat']]
    if w['fluid_type'] in _OILS:
        p += list(_GOR_NODES * w['p_sat'])
    g = [_producing_gor(w, x) for x in p if p_lo <= x <= p_hi]
    g_lo = min(g) * (1 - _RATIO_DRIFT) * (1 - _MARGIN)
    g_hi = max(g) * (1 + _RATIO_DRIFT) * (1 + _MARGIN)
    n = max(1, math.ceil(math.log(g_hi / g_lo) / math.log(_GOR_STEP) - 1e-9))
    gors = [g_lo * (g_hi / g_lo) ** (i / n) for i in range(n + 1)]
    return dict(gors=gors, p_r_range=[p_lo, p_hi], gor_range=[min(g), max(g)], R_i=hook['R_i'])


def write_axes(tables_dir) -> dict:  # spec: SMP-116
    """
    Step 3 of the build: DIR/axes.json, the GOR axis of every well of DIR/wells.json whose fluid DIR/fluids/kK.json
    was built. A well already in the file keeps its axis; a well whose axis fails gets its traceback as `error`.

    :return: The file's content
    """
    tables_dir = Path(tables_dir)
    spec = json.loads((tables_dir / 'wells.json').read_text())
    path = tables_dir / 'axes.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    for w in spec['wells']:
        k = w['well']
        fluid_path = tables_dir / 'fluids' / f'k{k:06d}.json'
        if str(k) in out or not fluid_path.exists():
            continue
        fluid = json.loads(fluid_path.read_text())
        if fluid['status'] == 'ok':
            try:
                out[str(k)] = _gor_axis(spec['kind'], spec['seed'], k, fluid['hook'])
            except MemoryError:
                raise  # The process's, not the well's: nothing is recorded, and a rerun makes the axis
            except Exception:  # noqa: BLE001 - the well's tables are not built, and its manifest says why
                out[str(k)] = dict(error=traceback.format_exc()[-1500:])
    _write_json(path, out)
    return out


def _write_json(path, obj):
    """obj as JSON at path, written to a temporary name and renamed, so that a stopped write leaves no file."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)
