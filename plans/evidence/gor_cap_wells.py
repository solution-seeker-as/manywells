"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The black oil's cap at the stream's gas-oil ratio (PVT-OIL-15, with PVT-OIL-13's cap in Sm3/Sm3; issue #16) on wells
drawn by develop's sampler. Each well (seed 2026) is solved at its nominal operating point and at two sampled ones, in
develop's default configuration, with the model of the tree the script runs in. Run it before and after the change
and compare the two:

  run       solve the wells and write one row per case: whether an operating point was found, PBH, PWH, TWH, the
            reservoir liquid rate, the bottomhole liquid density, and the black oil's uncapped solution gas-oil ratio
            at the solved bottomhole (BlackOil's own rs), which says whether the oil is undersaturated there
  compare   join two runs on (well, k) and print the operating points found in each, and the quantiles of the
            changes in the cases where the oil is undersaturated at the bottomhole and in the others

From the repository root, one BLAS thread per process, the environment of the tree to measure:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/gor_cap_wells.py run 500 rust.csv --backend rust
    uv run python plans/evidence/gor_cap_wells.py compare before.csv after.csv
"""

import argparse
import time
from functools import partial
from multiprocessing import Pool

import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026


def well_cases(well, backend):
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'gor_cap')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        fl = wp.fluid
        rec = dict(
            well=well, k=k, wlr=fl.wlr, gor=fl.gor, api=fl.api, w_lg=bc.w_lg, p_r=bc.p_r, p_s=bc.p_s, T_r=bc.T_r, u=bc.u
        )
        t0 = time.time()
        try:
            op = SSDFSimulator(wp, backend=backend).simulate(bc)
            x = op.state
            rec.update(
                PBH=x[0, 0],
                PWH=x[-1, 0],
                TWH=x[-1, 6],
                w_res=op.w_res,
                rho_l_bh=x[0, 5],
                Rs_bh=float(fl.pvt.rs(x[0, 0], x[0, 6])),
            )
        except SimError as e:
            rec['err'] = type(e).__name__
        rec['seconds'] = time.time() - t0
        rows.append(rec)
    return rows


def run(n_wells, out, backend, processes):
    with Pool(processes) as pool:
        rows = [r for rs in pool.map(partial(well_cases, backend=backend), range(n_wells), chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(
        f'{len(df)} cases from {n_wells} wells ({backend}), operating point in {df.PBH.notna().sum()}, '
        f'{df.seconds.sum():.0f} s in total'
    )


def changes(df):
    return pd.DataFrame(
        {
            'rho_l bh (%)': 100 * (df.rho_l_bh_a / df.rho_l_bh_b - 1),
            'PBH (bar)': df.PBH_a - df.PBH_b,
            'PWH (bar)': df.PWH_a - df.PWH_b,
            'TWH (K)': df.TWH_a - df.TWH_b,
            'w_res (%)': 100 * (df.w_res_a / df.w_res_b - 1),
        }
    )


def compare(before, after):
    q = [0, 0.1, 0.5, 0.9, 1]
    b, a = pd.read_csv(before), pd.read_csv(after)
    df = b.merge(a, on=['well', 'k'], suffixes=('_b', '_a'))
    found_b, found_a = df.PBH_b.notna(), df.PBH_a.notna()
    print(
        f'{len(df)} cases; operating point before {found_b.sum()}, after {found_a.sum()}; '
        f'lost {(found_b & ~found_a).sum()}, gained {(~found_b & found_a).sum()}'
    )
    for name, rows in (('lost', df[found_b & ~found_a]), ('gained', df[~found_b & found_a])):
        if len(rows):
            print(f'  {name}: ' + ', '.join(f'({r.well}, {r.k})' for r in rows.itertuples()))
    both = df[found_b & found_a]
    under = both.Rs_bh_b > both.gor_b  # undersaturated at the bottomhole, by the solved state before
    for name, rows in (
        ('undersaturated at the bottomhole', both[under]),
        ('saturated at the bottomhole', both[~under]),
    ):
        print(
            f'\n{len(rows)} cases {name}, with an operating point in both. Quantiles of the changes, after minus before:'
        )
        print(changes(rows).quantile(q).round(4).to_string())
    print(
        f'\nGOR of the undersaturated cases: {both[under].gor_b.min():.1f} to {both[under].gor_b.max():.1f} Sm3/Sm3, '
        f'median {both[under].gor_b.median():.1f}; their R_so at the bottomhole up to '
        f'{(both[under].Rs_bh_b / both[under].gor_b).max():.2f} times the GOR'
    )
    print(
        f'Seconds per case, median (total): before {df.seconds_b.median():.3f} ({df.seconds_b.sum():.0f}), '
        f'after {df.seconds_a.median():.3f} ({df.seconds_a.sum():.0f})'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('n_wells', type=int)
    r.add_argument('out')
    r.add_argument('--backend', choices=('casadi', 'rust'), default='rust')
    r.add_argument('--processes', type=int, default=8)
    c = sub.add_parser('compare')
    c.add_argument('before')
    c.add_argument('after')
    args = parser.parse_args()
    pd.set_option('display.width', 200)
    if args.cmd == 'run':
        run(args.n_wells, args.out, args.backend, args.processes)
    else:
        compare(args.before, args.after)


if __name__ == '__main__':
    main()
