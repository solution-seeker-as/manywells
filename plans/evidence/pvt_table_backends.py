"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

PVT tables on both backends (feature spec specs/features/018-pvt-tables.md, acceptance 9): the operating points
found with a table against the black oil, and the cost per case, with PVTTable as implemented (specs/model/pvt/
table.md). pvt_tables.py measured the prototype on the CasADi backend; these are its cases.

Parts:

  wells  the 263 cases of 88 wells of develop's sampler (seed 2026), each at its nominal operating point and two
         sampled ones (the cases of jt_wells.py and pvt_tables.py wells), solved cold with develop's black oil and
         with a table of it on the medium grid (10 bar by 10 K, from 1 to 470 bar and 270 to 430 K), on each backend:
         PBH, PWH and TWH, whether a point lies off the grid, and the time to make the table, to build the well and
         to find its roots
  eos    the cases of the first 88 wells whose p_r and T_r synthetic_eos_oil.csv covers, with the fluid replaced by
         the table's, and by develop's black oil at its stock-tank densities and GOR, capped at the table's bubble
         point at T_r, on each backend

Each case runs on one process, with one BLAS thread; the processes share the machine. From the repository root:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_table_backends.py wells 88 \\
        pvt_table_backends_wells.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_table_backends.py eos 88 \\
        pvt_table_backends_eos.csv
    uv run python plans/evidence/pvt_table_backends.py summary pvt_table_backends_wells.csv
    uv run python plans/evidence/pvt_table_backends.py eos-summary pvt_table_backends_eos.csv
"""

import pathlib
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import casadi as ca
import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.pvt.table import PVTTable
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026
PROCESSES = 12
BACKENDS = ('casadi', 'rust')
P_GRID = np.unique(np.r_[1.0, np.arange(10.0, 470.0 + 1e-9, 10.0)])   # The medium grid of pvt_tables.py
T_GRID = np.arange(270.0, 430.0 + 1e-9, 10.0)
EOS = pathlib.Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'pvt' / 'synthetic_eos_oil.csv'


def cases(well):
    """The well's cases: (k, wp, bc), the nominal operating point and two sampled ones, as jt_wells.py's."""
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            yield k, well_properties(draw, fractions, DEVELOP), bc
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue


def tabulate(fluid, p, T):
    """The fluid's R_so, B_o and Z at the nodes of the grid p x T, as a table at its GOR."""
    ps, Ts = ca.SX.sym('p'), ca.SX.sym('T')
    f = ca.Function('pvt', [ps, Ts], [fluid.rs(ps, Ts), fluid.bo(ps, Ts), fluid.z_factor(ps, Ts)])
    P, TT = np.meshgrid(p, T, indexing='ij')
    out = f.map(P.size)(P.ravel(), TT.ravel())
    return PVTTable(p=p, T=T, rho_o=fluid.rho_o, rho_g=fluid.rho_g, gor=fluid.gor,
                    **{k: np.asarray(v, float).reshape(P.shape) for k, v in zip(('R_so', 'B_o', 'Z'), out)})


def solve(wp, bc, backend):
    """The operating point's outputs, or the error's name, with the time to build the well and to find its roots."""
    t0 = time.perf_counter()
    try:
        sim = SSDFSimulator(wp, backend=backend)
        t1 = time.perf_counter()
        op = sim.simulate(bc)
    except SimError as e:
        return dict(err=type(e).__name__, build=np.nan, search=time.perf_counter() - t0)
    s = op.state
    return dict(PBH=s[0, 0], PWH=s[-1, 0], TWH=s[-1, 6], off_table=op.off_table, build=t1 - t0,
                search=time.perf_counter() - t1)


def solve_variants(rec, wp, bc, variants):
    for name, fluid in variants.items():
        for backend in BACKENDS:
            out = solve(replace(wp, fluid=fluid), bc, backend)
            rec.update({f'{key}_{name}_{backend}': v for key, v in out.items()})
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# Part 1: develop's black oil and its table
# ---------------------------------------------------------------------------------------------------------------------

def well_cases(well):
    rows = []
    for k, wp, bc in cases(well):
        t0 = time.perf_counter()
        table = tabulate(wp.fluid, P_GRID, T_GRID)
        rec = dict(well=well, k=k, gor=wp.fluid.gor, wlr=wp.fluid.wlr, p_r=bc.p_r, T_r=bc.T_r, w_lg=bc.w_lg,
                   table=time.perf_counter() - t0)
        rows.append(solve_variants(rec, wp, bc, {'bo': wp.fluid, 'table': replace(wp.fluid, pvt=table)}))
    return rows


def found(df, name, backend):
    return df[f'PBH_{name}_{backend}'].notna()


def compare(df, a, b, backends):
    """Quantiles and the largest |difference| of PBH, PWH and TWH, a minus b, where both have an operating point."""
    (na, ba), (nb, bb) = (a, backends[0]), (b, backends[1])
    ok = df[found(df, na, ba) & found(df, nb, bb)]
    d = pd.DataFrame({x: ok[f'{x}_{na}_{ba}'] - ok[f'{x}_{nb}_{bb}'] for x in ('PBH', 'PWH', 'TWH')})
    print(f'\n{na} on {ba} minus {nb} on {bb}, {len(ok)} cases (bar, bar, K):')
    print(d.quantile([0, 0.1, 0.5, 0.9, 1]).round(4).to_string())
    print('largest |difference|: ' + ', '.join(f'{c} {d[c].abs().max():.2e}' for c in d))


def times(df, names):
    t = pd.DataFrame({f'{n} {b}': df[f'build_{n}_{b}'].fillna(0) + df[f'search_{n}_{b}']
                      for n in names for b in BACKENDS})
    print('\nTime per case, to build the well and find its roots on one process (s):')
    print(pd.DataFrame({'median': t.median(), 'total': t.sum()}).round(3).to_string())
    for b in BACKENDS:
        ratio = t[f'{names[1]} {b}'] / t[f'{names[0]} {b}']
        print(f'{names[1]} / {names[0]} on {b}: median ratio {ratio.median():.2f}, total ratio '
              f'{t[f"{names[1]} {b}"].sum() / t[f"{names[0]} {b}"].sum():.2f}')
    print('casadi / rust: ' + ', '.join(f'{n} {t[f"{n} casadi"].sum() / t[f"{n} rust"].sum():.1f}' for n in names))


def operating_points(df, names):
    print(f'{len(df)} cases. Operating points:')
    for n in names:
        print(f'  {n}: ' + ', '.join(f'{b} {found(df, n, b).sum()}' for b in BACKENDS))
    for b in BACKENDS:
        a, t = found(df, names[0], b), found(df, names[1], b)
        print(f'  {b}: {names[1]} lost {(a & ~t).sum()} of {names[0]}\'s, and found {(t & ~a).sum()} more')
    for n in names:
        c, r = found(df, n, 'casadi'), found(df, n, 'rust')
        print(f'  {n}: rust only {(r & ~c).sum()}, casadi only {(c & ~r).sum()}')
    for n in names:
        if f'off_table_{n}_casadi' in df:
            print(f'  {n}: operating points with a point off the table: '
                  + ', '.join(f'{b} {df[f"off_table_{n}_{b}"].fillna(False).astype(bool).sum()}' for b in BACKENDS))


def summary(df):
    names = ('bo', 'table')
    operating_points(df, names)
    for b in BACKENDS:
        compare(df, 'table', 'bo', (b, b))
    for n in names:
        compare(df, n, n, ('rust', 'casadi'))
    times(df, names)
    print(f'\nTime to make a table (s): median {df.table.median():.3f}, largest {df.table.max():.3f}')


def wells(n_wells, out):
    with Pool(PROCESSES) as pool:
        df = pd.DataFrame([r for rs in pool.map(well_cases, range(n_wells), chunksize=1) for r in rs])
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    summary(df)


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: the synthetic equation-of-state fluid
# ---------------------------------------------------------------------------------------------------------------------

def bubble_points(path):
    """{T (K): p_b (bar)} from the table's header line 'p_bubble: 280=109.268, ...  (T in K = p in bar)'."""
    with open(path) as fh:
        line = next(line for line in fh if line.startswith('# p_bubble:'))
    pairs = line.partition(':')[2].split('(')[0].split(',')
    return dict(sorted((float(t), float(p)) for t, p in (kv.split('=') for kv in pairs)))


def eos_cases(well):
    eos, p_b = PVTTable.read_csv(EOS), bubble_points(EOS)
    rows = []
    for k, wp, bc in cases(well):
        if not eos.covers(bc.p_r, bc.T_r):  # The simulator refuses the rest
            continue
        cap = float(np.interp(bc.T_r, list(p_b), list(p_b.values())))
        bo = replace(wp.fluid.pvt, rho_o=eos.rho_o, rho_g=eos.rho_g, p_bubble=cap)
        rec = dict(well=well, k=k, wlr=wp.fluid.wlr, p_r=bc.p_r, T_r=bc.T_r, w_lg=bc.w_lg, p_b=cap)
        variants = {'bo_cap': replace(wp.fluid, pvt=bo, gor=eos.gor), 'eos': replace(wp.fluid, pvt=eos, gor=eos.gor)}
        rows.append(solve_variants(rec, wp, bc, variants))
    return rows


def eos_summary(df):
    names = ('bo_cap', 'eos')
    operating_points(df, names)
    for b in BACKENDS:
        compare(df, 'eos', 'bo_cap', (b, b))
    for n in names:
        compare(df, n, n, ('rust', 'casadi'))
    times(df, names)


def eos_wells(n_wells, out):
    with Pool(PROCESSES) as pool:
        df = pd.DataFrame([r for rs in pool.map(eos_cases, range(n_wells), chunksize=1) for r in rs])
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    eos_summary(df)


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    if sys.argv[1] == 'wells':
        wells(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'summary':
        summary(pd.read_csv(sys.argv[2]))
    elif sys.argv[1] == 'eos':
        eos_wells(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'eos-summary':
        eos_summary(pd.read_csv(sys.argv[2]))
