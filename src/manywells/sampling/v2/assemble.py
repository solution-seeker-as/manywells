"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The assembly of a v2 dataset from its run's results (specs/sampling.md, SMP-108 and SMP-110;
plans/sampler-redesign/datasets-draft.md, Section 2): the wells up to the n-th accepted, in draw order, give the
tables `wells` (the accepted wells, numbered by `ID` and split into train and test), `samples` (their operating
points), `geometry` (their surveys) and `pvt` (their tables), and the logs `drawn_wells` (every drawn well) and
`redraws` (every slot's attempt without an operating point, and every attempt of a dropped well), written by
datasets.v2 as the dataset's folder and its companion folder of logs. The attempts' search counts and times stay with
the run: the chunks hold them, and the assembly can write them by chunk as the run's attempt log.

The columns follow datasets-draft.md's Section 2 with the changes of decisions D-122 to D-125
(plans/sampler-redesign/decisions.md): no columns of the CasADi retry, `lift_start_rule` in both datasets, the counts
of a well's attempts by reason, the columns that record a table fluid's build, and the inputs that vary in nsol-2 in
`samples` only.
"""

import collections
import datetime
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.configurations import V2
from manywells.datasets.v2 import (
    DATASETS,
    FEATURES,
    LOGS,
    NSOL_INPUTS,
    NSOL_TIME,
    REASONS,
    SAMPLE_INPUTS,
    TABLES,
    WEEKS_PER_YEAR,
    code_record,
    lock_hash,
    platform_record,
    typed,
    write_tables,
)
from manywells.geometry import WellGeometry
from manywells.sampling.v2 import generate
from manywells.sampling.v2.life import WLR_MAX
from manywells.sampling.v2.mapping import N_CELLS
from manywells.sampling.v2.reservoir import OILS
from manywells.sampling.v2.run import read_results
from manywells.sampling.v2.tables import TABLE_CLASSES
from manywells.sampling.wells import rng_for

_MSCF = 28.3168  # Sm3 per Mscf
_LIFT = {  # A lift design's columns, by its draws' keys
    'injection_md': 'injection_md',
    'injection_tvd': 'z_inj',
    'T_lg': 'T_lg',
    'w_lg_max': 'w_lg_max',
    'glr_cap': 'glr_cap',
    'p_inj': 'p_inj',
    'dp_valve': 'dp_valve',
    'packer_offset': 'packer_offset',
    'psi_lg': 'psi_lg',
}
_GEOMETRY = {  # The survey's parameters, by the draws' keys
    'kickoff_tvd': 'geo_kickoff',
    'build_rate': 'geo_build_rate',
    'build_rate_2': 'geo_build_rate_2',
    'hold_inclination': 'geo_theta_hold',
    'drop_rate': 'geo_drop_rate',
    'end_inclination': 'geo_theta_end',
    'end_section_tvd': 'geo_end_section_tvd',
    'heel_inclination': 'geo_theta_heel',
    'tangent_md': 'geo_tangent_md',
    'drain_length': 'drain_length',
}
_TABLE_FLUID = (  # A table fluid's columns: its composition, and what its build records
    'comp_w1 comp_w2 comp_w3 z_c7p M_c7p alpha_c7p C_f f_int c7_mol gor_moved api_drawn sg_gas_drawn R_si_drawn '
    'p_sat_drawn offset_api offset_sg_gas offset_gor field_api field_sg_gas field_gor n_gors table_nodes'
).split()
_LIFE = (  # An nsol-2 well's life (life.Life.parameters), the end that the scan gives, and its plan's p_change
    'T_life T_end t_plateau decline_D decline_b gamma_p r_p_start r_p_end F_bt wlr_bt water_start water_end t_sep_1 '
    't_sep_2 r_s_mid r_s_late p_change lift_start'
).split()
_GENERATION = ('n_redraws', 'n_redraws_no_op', 'n_redraws_solver') + tuple(f'n_redraws_{r}' for r in REASONS)


def well_columns(kind: str) -> list:  # spec: SMP-110
    """The columns of a dataset's `wells` table, in their order."""
    sol = kind == 'sol'
    cols = ['ID', 'draw_index', 'split', 'configuration', 'n_cells']
    cols += 'coverage_tail well_type fluid_type fluid_class class_rule drive'.split()
    cols += ['life_stage', 'life_state'] if sol else []
    cols += 'pressure_class saturation geometry_class geometry_family pvt_model fluid_source saturation_kind'.split()
    cols += 'saturation_rule inflow tubing choke_model chk_profile lift lift_start_rule'.split()
    cols += 'z water_depth L md_bottom max_inclination'.split() + list(_GEOMETRY)
    cols += 'salinity rho_w p_h overpressure_ratio p_i T_s geothermal_gradient T_r p_sep T_sep'.split()
    cols += ['p_s_nom', 'r_s'] if sol else []
    cols += 'dp_flowline api sg_gas rho_o rho_g p_sat R_si pvt_table_id'.split() + _TABLE_FLUID
    cols += ['life_date', 'p_r_nom', 'gor_nom', 'wlr_nom', 'wgr_nom'] if sol else []
    cols += ['kgko_weight', 'gor_factor_cond'] + ([] if sol else _LIFE)
    cols += 'permeability net_pay skin ln_re_rw drawdown_design anisotropy productivity_factor J non_darcy_D'.split()
    cols += ['k_l', 'w_l_max', 'forchheimer_a', 'forchheimer_b'] if sol else []
    cols += 'design_rate floor best_rate best_rate_w_lg D'.split()
    cols += ['roughness'] if sol else ['roughness_start', 'roughness_end']
    cols += 'bean_ratio C_D K_c'.split() + list(_LIFT) + (['h'] if sol else [])
    cols += 'U_t k_e well_age cp_o cp_w cp_g flags'.split() + list(_GENERATION)
    return cols


def drawn_well_columns(kind: str) -> list:  # spec: SMP-108
    """The columns of a dataset's `drawn_wells` log: every column of `wells`, then its outcome and counts."""
    more = 'outcome not_producible_reason error n_attempts n_several_stable best_rate_outcomes'.split()
    more += ['scan_best_rate', 'scan_outcome'] if kind == 'nsol' else []
    return well_columns(kind) + more + ['seconds', 'cpu_seconds', 'table_seconds']


def sample_columns(kind: str) -> list:  # spec: SMP-110
    """The columns of a dataset's `samples` table, in their order."""
    time = ['WEEKS', *NSOL_TIME.values()] if kind == 'nsol' else []
    inputs = list(SAMPLE_INPUTS.values()) + (list(NSOL_INPUTS.values()) if kind == 'nsol' else [])
    return ['ID', 'slot', 'attempt', *time, *inputs, *FEATURES]


def redraw_columns(kind: str) -> list:  # spec: SMP-108
    """The columns of a dataset's `redraws` log: the attempt, its inputs, its outcome and the sizes of its root set."""
    inputs = sample_columns(kind)[3 : len(sample_columns(kind)) - len(FEATURES)]
    return ['draw_index', 'ID', 'slot', 'attempt', *inputs, 'outcome', 'reason', 'n_roots', 'n_stable']


# --- A well's rows ----------------------------------------------------------------------------------------------------
def max_inclination(stations, D) -> float:
    """The largest cell inclination from vertical on the model's grid (degrees)."""
    geo = WellGeometry.from_survey(stations[0], stations[1], N_CELLS, D=D)
    return math.degrees(math.acos(min(1.0, float(np.min(np.asarray(geo.cos_incl))))))


def well_row(res: dict, kind: str) -> dict:  # spec: SMP-108, SMP-110
    """
    A drawn well's row of `drawn_wells` from its result (generate's module docstring): its draws, labels and table,
    its outcome and the counts of its attempts, without `ID` and `split`, which the assembly gives the accepted wells.
    A crashed well has its draw index, outcome, error and times only.
    """
    w, info, tab = res['well'], res['info'], res['table'] or {}
    times = dict(seconds=res['seconds'], cpu_seconds=res.get('cpu_seconds', math.nan))
    if w is None:
        row = dict(draw_index=info['draw_index'], configuration=V2, n_cells=N_CELLS, **_counts(res))
        row.update(outcome=info['outcome'], error=info.get('error'), best_rate_outcomes=[], **times)
        return row | (dict(scan_best_rate=[], scan_outcome=[]) if kind == 'nsol' else {})

    def g(key):
        v = w.get(key, math.nan)
        return math.nan if v is None else v

    oil, k, stations = w['fluid_type'] in OILS, w['well'], res['stations']
    has_table = tab.get('status') == 'ok'
    row = dict(draw_index=k, configuration=V2, n_cells=N_CELLS)
    row.update({key: w[key] for key in ('coverage_tail', 'well_type', 'fluid_type', 'fluid_class', 'class_rule')})
    row.update({key: w[key] for key in ('drive', 'pressure_class', 'saturation', 'geometry_class', 'geometry_family')})
    row.update(pvt_model='table' if w['fluid_class'] in TABLE_CLASSES else 'black_oil')
    row.update({key: tab.get(key) for key in ('saturation_kind', 'saturation_rule')}, fluid_source=tab.get('source'))
    row.update({key: w[key] for key in ('inflow', 'tubing', 'choke_model', 'chk_profile')})
    row.update(z=w['z'], water_depth=w['d_w'], L=w['L'], md_bottom=stations[0][-1])
    row.update(max_inclination=max_inclination(stations, w['D']), **{col: g(key) for col, key in _GEOMETRY.items()})
    row.update({key: w[key] for key in ('salinity', 'rho_w', 'p_h')})
    row.update(overpressure_ratio=(w['p_i'] - w['p_h']) / (w['p_lith'] - w['p_h']), p_i=w['p_i'], T_s=w['T_s'])
    row.update(geothermal_gradient=w['G'], T_r=w['T_r'], p_sep=w['p_sep'], T_sep=w['T_sep'], dp_flowline=w['dp_fl'])
    row.update({key: w[key] for key in ('api', 'sg_gas', 'rho_o', 'rho_g')}, p_sat=g('p_sat'), R_si=w['R_i'])
    row.update(pvt_table_id=f'k{k:06d}' if has_table else None)
    row.update(kgko_weight=g('w_kgko') if oil else math.nan)
    row.update(gor_factor_cond=g('c_k') if w['fluid_class'] == 'gas_condensate' else math.nan)
    row.update(permeability=w['perm'], net_pay=w['h_net'], skin=w['skin'], ln_re_rw=w['ln_re_rw'])
    row.update(drawdown_design=w['delta_d'], anisotropy=g('I_ani'), productivity_factor=w['f_geo'])
    row.update(J=g('J') if oil else math.nan, non_darcy_D=math.nan if oil else w['D_nondarcy'] / _MSCF)
    row.update(design_rate=w['q_l_ref'] if oil else w['q_g_ref'], floor=generate.FLOOR[w['well_type']])
    row.update(best_rate=info['best_rate'], best_rate_w_lg=info['best_rate_w_lg'])
    row.update(D=w['D'], bean_ratio=w['beta'], C_D=w['c_d'], K_c=w['K_c'])
    design = res['nsol']['design'] if kind == 'nsol' else (w if w['lift'] == 'gas_lift' else None)
    row.update({col: design[key] if design is not None else math.nan for col, key in _LIFT.items()})
    row.update(U_t=w['U_t'], k_e=w['k_e'], well_age=w['age_years'] * 365, cp_o=w['cp_o'], cp_w=w['cp_w'])
    row.update(cp_g=w['cp_g'], flags=w['flags'])
    if kind == 'sol':
        stage = w['life_stage']
        late = 'depleted' if w['drive'] == 'depletion' else 'watered_out'
        row.update(life_stage=stage, life_state=stage if stage != 'late' else late, life_date=w['date'])
        row.update(lift=w['lift'], lift_start_rule=w['lift_start_rule'], p_s_nom=w['p_s'], r_s=w['r_s'])
        row.update(p_r_nom=w['p_r'], gor_nom=w['R_go'], wlr_nom=w['wlr'], wgr_nom=math.nan if oil else w['wgr'])
        row.update({key: g(key) for key in ('k_l', 'w_l_max', 'forchheimer_a', 'forchheimer_b')})
        row.update(roughness=w['roughness'], h=w['h'])
    else:
        life = {key: v for key, v in res['nsol'].items() if key != 'design'}
        row.update(life, lift_start=math.nan if life['lift_start'] is None else life['lift_start'])
        row.update(lift='gas_lift' if life['lift_start'] is not None else 'natural')
        row.update(T_end=info['T_end'], p_change=info['p_change'])
    if 'hook' in tab:  # The table's fluid: its composition and what it replaced (SMP-113 to SMP-115)
        comp, drawn = tab['composition'], tab['drawn']
        weights = list((comp.get('template_weights') or {}).values()) or [math.nan] * 3
        offsets, field = comp.get('offsets') or {}, tab.get('field') or {}
        row.update({f'comp_w{i + 1}': x for i, x in enumerate(weights)})
        row.update(
            z_c7p=comp.get('z7', math.nan), M_c7p=comp.get('M7', math.nan), alpha_c7p=comp.get('alpha', math.nan)
        )
        row.update(C_f=comp.get('Cf', math.nan), f_int=comp.get('intermediates_factor', math.nan))
        row.update(api_drawn=drawn['api'], sg_gas_drawn=drawn['sg_gas'], R_si_drawn=drawn['R_si'])
        row.update(p_sat_drawn=drawn['p_sat'], gor_moved=tab.get('gor_moved') or math.nan, c7_mol=tab.get('c7_mol'))
        row.update({f'offset_{key}': offsets.get(key, math.nan) for key in ('api', 'sg_gas', 'gor')})
        row.update({f'field_{key}': field.get(key, math.nan) for key in ('api', 'sg_gas', 'gor')})
        row.update(n_gors=len(tab.get('gors', [])), table_nodes=tab.get('nodes', math.nan))
    row.update(_counts(res), outcome=info['outcome'], not_producible_reason=info['not_producible_reason'])
    row.update(error=info['error'], best_rate_outcomes=list(info['best_rate_outcomes']), **times)
    row['table_seconds'] = sum(tab.get('seconds', {}).get(key, 0.0) for key in ('fluid', 'tables')) if tab else math.nan
    if kind == 'nsol':
        row.update(scan_best_rate=[s['best'] for s in info['scan']], scan_outcome=[s['status'] for s in info['scan']])
    return row


def _counts(res) -> dict:  # spec: SMP-108
    """A well's counts of its slots' attempts: all, the redraws, those without an operating point by outcome and by
    reason, and its samples whose root set had several stable roots."""
    slots = [r for r in res['log'] if r['kind'] == 'slot']
    failed = [r for r in slots if r['outcome'] != 'operating_point']
    out = dict(n_attempts=len(slots), n_redraws=sum(r['attempt'] > 0 for r in slots))
    out.update(n_redraws_no_op=sum(r['outcome'] == 'no_operating_point' for r in failed))
    out.update(n_redraws_solver=sum(r['outcome'] == 'solver_failure' for r in failed))
    out.update({f'n_redraws_{reason}': sum(r['reason'] == reason for r in failed) for reason in REASONS})
    out['n_several_stable'] = sum(bool(r['several_stable']) for r in res['samples'])
    return out


def _attempt_rows(recs, kind) -> pd.DataFrame:
    """Attempts' records as rows, their inputs under the samples' names."""
    df = pd.DataFrame(recs)
    if kind == 'nsol' and len(df):
        df['WEEKS'] = df['t'] * WEEKS_PER_YEAR
    return df.rename(columns=SAMPLE_INPUTS | (NSOL_INPUTS | NSOL_TIME if kind == 'nsol' else {}))


def pvt_rows(table: dict, table_id: str) -> pd.DataFrame:  # spec: SMP-110
    """
    A table well's table set as rows of `pvt`: the nodes of its CSV file (tables.read_table's manifest names it), with
    the header's densities and text in every row, so that datasets.v2.pvt_from_rows rebuilds PVTTable.read_csv's tables
    from them, to the bit. A single table's GOR, from its header, is in every row's `gor`.
    """
    path = Path(table['tables_dir']) / table['csv']
    header = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].partition(':')
            header[key.strip()] = value.strip()
    df = pd.read_csv(path, comment='#', float_precision='round_trip')
    if 'gor' not in df:
        df.insert(0, 'gor', float(header['gor']) if 'gor' in header else math.nan)
    df.insert(0, 'pvt_table_id', table_id)
    df['rho_o'], df['rho_g'] = float(header['rho_o']), float(header['rho_g'])
    for key in ('name', 'source', 'separation'):
        df[key] = header.get(key)
    return df


def split(wells: pd.DataFrame, seed: int) -> np.ndarray:  # spec: SMP-110
    """
    The wells' split: 20% `test` and 80% `train`, stratified by fluid type, well type, geometry class and the tail.
    Each well, in the order of the rows (by ID), draws one number from rng_for(seed, 'v2', 'split'); within each
    stratum the wells are ranked by it, and ranks 5, 10, 15, ... are `test`.
    """
    r = rng_for(seed, 'v2', 'split')  # spec: SMP-109
    strata = ['fluid_type', 'well_type', 'geometry_class', 'coverage_tail']
    df = wells[strata].astype(str).assign(key=r.random(len(wells)))
    rank = df.groupby(strata)['key'].rank(method='first')
    return np.where(rank.astype(int) % 5 == 0, 'test', 'train')


# --- The dataset ------------------------------------------------------------------------------------------------------
def assemble(results, kind: str, seed: int, n_wells: int, attempts=None) -> dict:  # spec: SMP-108, SMP-110
    """
    A dataset's tables and logs from its run's results, in draw order: the wells up to the n-th accepted are drawn,
    the first n accepted are the dataset's, numbered by `ID` from 0 in draw order.

    :param results: The wells' results in draw order (run.read_results), from draw index 0
    :param kind: 'sol' or 'nsol'
    :param seed: The dataset's seed
    :param n_wells: The accepted wells of the dataset
    :param attempts: None, or a callable that takes the attempt log of each hundred wells as a DataFrame (every
        record of their logs, its inputs under the samples' names, with `ID` and `well_outcome`), in draw order
    :return: A dict of the typed tables (`wells`, `samples`, `geometry`, `pvt`, `drawn_wells`, `redraws`) and
        `counts`: the wells drawn and accepted, their outcomes and the slots' attempts by outcome, and the CPU time
    :raises ValueError: if the results hold fewer than n_wells accepted wells
    """
    drawn, samples, geometry, pvt, redraws, chunk = [], [], [], [], [], []
    attempt_outcomes, cpu, n_id = collections.Counter(), 0.0, 0
    for res in results:
        if n_id >= n_wells:
            break
        row = well_row(res, kind)
        accepted = row['outcome'] == 'accepted'
        ID = n_id if accepted else None
        n_id += accepted
        row['ID'] = ID
        drawn.append(row)
        cpu += row['cpu_seconds']
        slots = [rec for rec in res['log'] if rec['kind'] == 'slot']
        attempt_outcomes.update(rec['outcome'] for rec in slots)
        kept = [rec for rec in slots if rec['outcome'] != 'operating_point' or not accepted]
        if kept:
            redraws.append(_attempt_rows(kept, kind).assign(ID=ID))
        if accepted:
            samples.append(_attempt_rows(res['samples'], kind).assign(ID=ID))
            md, tvd = res['stations']
            geometry.append(pd.DataFrame(dict(ID=ID, station=np.arange(len(md)), md=md, tvd=tvd)))
            if row['pvt_table_id'] is not None:
                pvt.append(pvt_rows(res['table'], row['pvt_table_id']))
        if attempts is not None:
            chunk.append(_attempt_rows(res['log'], kind).assign(ID=ID, well_outcome=row['outcome']))
            if len(drawn) % 100 == 0:
                attempts(pd.concat(chunk, ignore_index=True))
                chunk = []
    if n_id < n_wells:
        raise ValueError(f'the results hold {n_id} accepted wells, not {n_wells}')
    if attempts is not None and chunk:
        attempts(pd.concat(chunk, ignore_index=True))
    drawn = pd.DataFrame(drawn).reindex(columns=drawn_well_columns(kind))
    is_well = drawn['ID'].notna()
    splits = np.full(len(drawn), None, dtype=object)
    splits[is_well.to_numpy()] = split(drawn[is_well], seed)
    drawn['split'] = splits
    wells = drawn.loc[is_well, well_columns(kind)].astype({'ID': 'int32'}).reset_index(drop=True)
    pvt_cols = 'pvt_table_id gor p T R_so B_o Z mu_o mu_g sigma rho_o rho_g name source separation'.split()
    out = dict(
        wells=wells,
        samples=_frame(samples, sample_columns(kind)).sort_values(['ID', 'slot'], ignore_index=True),
        geometry=_frame(geometry, ['ID', 'station', 'md', 'tvd']),
        pvt=_frame(pvt, pvt_cols),
        drawn_wells=drawn,
        redraws=_frame(redraws, redraw_columns(kind)),
    )
    out = {name: typed(df, name) for name, df in out.items()}
    outcomes = drawn['outcome'].value_counts().to_dict()
    out['counts'] = dict(
        wells_drawn=len(drawn),
        wells_accepted=int(is_well.sum()),
        outcomes={key: int(v) for key, v in outcomes.items() if v},
        attempts=dict(attempt_outcomes),
        cpu_seconds=cpu,
    )
    return out


def _frame(parts, columns) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).reindex(columns=columns)


def records(kind: str, seed: int, n_wells: int, settings: dict) -> dict:  # spec: SMP-110
    """The records of a dataset (SMP-110): its seed, the configuration, the grid, the backend, the limits and the
    floor, the code that ran and assembled it, the lock file's hash and the platform."""
    return dict(
        dataset=DATASETS[kind],
        kind=kind,
        seed=seed,
        n_wells=n_wells,
        n_slots=settings['n_slots'],
        configuration=V2,
        n_cells=N_CELLS,
        backend='rust',
        retry_backend=None,
        limits=settings['limits'],
        producibility=dict(
            floor=settings['floor'],
            margin=settings['floor_margin'],
            lift_fractions=generate._BEST_RATE_LIFT,
            scan_points=generate._N_SCAN if kind == 'nsol' else None,
            wlr_max=WLR_MAX,
        ),
        split=dict(test_share=0.2, strata=['fluid_type', 'well_type', 'geometry_class', 'coverage_tail']),
        chunk_size=settings['chunk_size'],
        tables=settings['tables'],
        code=settings['code'],
        assembled_by=code_record(),
        lock_sha256=lock_hash(),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        **platform_record(),
    )


def write_dataset(run_dir, out, n_wells: int, allow_dirty: bool = False, attempts: bool = True) -> dict:
    """
    A dataset from a complete run (run.run): its folder OUT/<dataset> with the four tables and metadata.json, and the
    companion folder OUT/<dataset>-logs with the two logs and its metadata.json (SMP-108), each file's row count and
    SHA-256 in the metadata. The run's attempt log goes to RUN/attempts/ as one parquet file per hundred wells.

    :param run_dir: The run's folder
    :param out: The folder to write the dataset and its logs into
    :param n_wells: The accepted wells of the dataset
    :param allow_dirty: Write a dataset from code with changes or outside a git checkout, as for a pilot. A dataset
        to publish is written from a clean working tree, so that its commit reproduces it (SMP-110)
    :param attempts: Whether to write the run's attempt log
    :return: The dataset's metadata
    """
    run_dir, out = Path(run_dir), Path(out)
    settings = json.loads((run_dir / 'settings.json').read_text())
    kind, seed = settings['kind'], settings['seed']
    meta = records(kind, seed, n_wells, settings)
    for code in (meta['code'], meta['assembled_by']):
        if (code['dirty'] is not False or code['commit'] is None) and not allow_dirty:
            raise ValueError(
                f'the code {code["version"]} has changes or no commit: a published dataset needs both clean'
            )
    writer = None
    if attempts:
        (run_dir / 'attempts').mkdir(exist_ok=True)
        for old in (run_dir / 'attempts').glob('part-*.parquet'):
            old.unlink()
        part = itertools.count()

        def writer(df):
            df.to_parquet(run_dir / 'attempts' / f'part-{next(part):06d}.parquet', index=False)

    tables = assemble(read_results(run_dir), kind, seed, n_wells, attempts=writer)
    meta['counts'] = tables.pop('counts')
    name = DATASETS[kind]
    data = write_tables(out / name, {key: tables[key] for key in TABLES}, meta)
    write_tables(out / f'{name}-logs', {key: tables[key] for key in LOGS}, dict(meta, companion_of=name))
    return data
