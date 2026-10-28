"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The liquid density with black oil and water (PVT-MIX-6, issue #8) on wells drawn by develop's sampler. Each well
(seed 2026) is solved at its nominal operating point and at two sampled ones, in develop's default configuration,
with the model of the tree the script runs in. Run it before and after a change to PVT-MIX-6 and compare the two:

  run       solve the wells and write one row per case: whether an operating point was found, PBH, PWH, TWH, the
            reservoir liquid rate, and at the bottomhole the solved liquid density, B_o, and the liquid density by
            the old (stock-tank volume) and the new (in-situ volume) mixing, both from the solved p and T
  compare   join two runs on (well, k) and print the operating points found in each and the quantiles of the changes

From the repository root, one BLAS thread per process:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/liquid_density_wells.py run 500 rust.csv --backend rust
    uv run python plans/evidence/liquid_density_wells.py compare before.csv after.csv
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


def bottomhole(fluid, p, T):
    """B_o and the liquid density by the old and the new mixing at (p, T), from the fluid's R_so and B_o."""
    wlr = fluid.wlr
    Rs, Bo = float(fluid.rs(p, T)), float(fluid.bo(p, T))
    rho_lo = (fluid.rho_o + Rs * fluid.rho_g) / Bo
    old = wlr * fluid.rho_w + (1 - wlr) * rho_lo
    new = ((1 - wlr) * (fluid.rho_o + Rs * fluid.rho_g) + wlr * fluid.rho_w) / ((1 - wlr) * Bo + wlr)
    return dict(Bo_bh=Bo, Rs_bh=Rs, rho_l_old_bh=old, rho_l_new_bh=new)


def well_cases(well, backend):
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'liquid_density')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        fl = wp.fluid
        rec = dict(well=well, k=k, wlr=fl.wlr, gor=fl.gor, api=fl.api, w_lg=bc.w_lg, p_r=bc.p_r, p_s=bc.p_s,
                   T_r=bc.T_r, u=bc.u)
        t0 = time.time()
        try:
            op = SSDFSimulator(wp, backend=backend).simulate(bc)
            x = op.state
            rec.update(PBH=x[0, 0], PWH=x[-1, 0], TWH=x[-1, 6], w_res=op.w_res, rho_l_bh=x[0, 5])
            rec.update(bottomhole(fl, x[0, 0], x[0, 6]))
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
    print(f'{len(df)} cases from {n_wells} wells ({backend}), operating point in {df.PBH.notna().sum()}, '
          f'{df.seconds.sum():.0f} s in total')


def compare(before, after):
    q = [0, 0.1, 0.5, 0.9, 1]
    b, a = pd.read_csv(before), pd.read_csv(after)
    df = b.merge(a, on=['well', 'k'], suffixes=('_b', '_a'))
    found_b, found_a = df.PBH_b.notna(), df.PBH_a.notna()
    print(f'{len(df)} cases; operating point before {found_b.sum()}, after {found_a.sum()}; '
          f'lost {(found_b & ~found_a).sum()}, gained {(~found_b & found_a).sum()}')
    water = df[(df.wlr_b > 0) & found_b & found_a]
    print(f'{len(water)} cases with water and an operating point in both')
    ratio = water.rho_l_new_bh_b / water.rho_l_old_bh_b - 1
    diff = pd.DataFrame({
        'rho_l bh new/old - 1 (%)': 100 * ratio,
        'PBH (bar)': water.PBH_a - water.PBH_b,
        'PWH (bar)': water.PWH_a - water.PWH_b,
        'TWH (K)': water.TWH_a - water.TWH_b,
        'w_res (%)': 100 * (water.w_res_a / water.w_res_b - 1),
    })
    print('Quantiles of the changes, after minus before:')
    print(diff.quantile(q).round(3).to_string())
    print(f'wlr from {water.wlr_b.min():.3f} to {water.wlr_b.max():.3f}, median {water.wlr_b.median():.3f}; '
          f'B_o at the bottomhole from {water.Bo_bh_b.min():.3f} to {water.Bo_bh_b.max():.3f}')
    print(f'Seconds per case, median: before {df.seconds_b.median():.3f}, after {df.seconds_a.median():.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('n_wells', type=int)
    r.add_argument('out')
    r.add_argument('--backend', choices=('casadi', 'rust'), default='rust')
    r.add_argument('--processes', type=int, default=12)
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
