"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The implemented feature 020 (specs/features/020-profile-scale.md, Acceptance, item 7) on the wells and operating
points of five_regime_default.py (develop's sampler, seed 2026, 88 wells at their nominal operating point and two
sampled ones), in develop's default configuration with the profile parameter scaled (SLIP-18):

  1    the five regimes' own C_0, develop's default
  0.5  C_0 halfway to 1: 1.1 where Table 1 has 1.2, CAL-6's prior median
  0    C_0 = 1, no slip in the profile; the drift velocities as before

Per case, backend and scale it records whether there is an operating point, the number of roots, PBH, PWH, TWH, the
reservoir liquid rate and the root search's time, on one process. The Rust core solves every case; the CasADi
backend the cases of the first --casadi-wells wells.

From the repository root (about 15 minutes with --casadi-wells 30):

    uv run python plans/evidence/profile_scale_wells.py profile_scale_wells.csv --casadi-wells 30
"""

import argparse
import time
from collections import Counter
from dataclasses import replace

import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026
SCALES = {'1': 1.0, '0.5': 0.5, '0': 0.0}


def cases(n_wells):
    """The cases of five_regime_default.py: each well at its nominal operating point and two sampled ones."""
    for well in range(n_wells):
        draw = sample_well(SEED, well)
        rng = rng_for(SEED, well, 'jt')
        for k in range(3):
            bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
            try:
                yield well, k, bc, well_properties(draw, fractions, DEVELOP)
            except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
                continue


def solve(wp, bc, backend):
    t0 = time.perf_counter()
    try:
        rs = SSDFSimulator(wp, backend=backend).root_set(bc)
    except SimError:
        rs = None
    out = {'seconds': time.perf_counter() - t0, 'roots': len(rs.roots) if rs else 0}
    op = rs.operating_point if rs else None
    if op is not None:
        X = op.state
        out.update(PBH=X[0, 0], PWH=X[-1, 0], TWH=X[-1, 6], w_res=op.w_res, alpha_mean=float(X[:, 3].mean()))
    return out


def summary(df):
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    pd.set_option('display.width', 250)
    for backend, d in df.groupby('backend'):
        print(f'== {backend}: {len(d)} cases')
        for v in SCALES:
            s = d[f'seconds_{v}']
            print(
                f'  profile_scale {v}: operating point in {d[f"PBH_{v}"].notna().sum()}, roots per case '
                f'{dict(sorted(Counter(d[f"roots_{v}"]).items()))}, {s.median():.2f} s median, {s.sum():.1f} s total'
            )
        for v in ('0.5', '0'):
            both = d.dropna(subset=['PBH_1', f'PBH_{v}'])
            lost = d[d['PBH_1'].notna() & d[f'PBH_{v}'].isna()]
            gained = d[d['PBH_1'].isna() & d[f'PBH_{v}'].notna()]
            print(
                f'  {v} against 1: {len(both)} with an operating point in both; lost: {list(zip(lost.well, lost.k))};'
                f' gained: {list(zip(gained.well, gained.k))}'
            )
            diff = pd.DataFrame(
                {
                    'PBH': both[f'PBH_{v}'] - both['PBH_1'],
                    'PWH': both[f'PWH_{v}'] - both['PWH_1'],
                    'TWH': both[f'TWH_{v}'] - both['TWH_1'],
                    'rate (rel)': both[f'w_res_{v}'] / both['w_res_1'] - 1,
                    'mean alpha': both[f'alpha_mean_{v}'] - both['alpha_mean_1'],
                }
            )
            print(diff.quantile(q).round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--wells', type=int, default=88)
    parser.add_argument('--casadi-wells', type=int, default=30)
    args = parser.parse_args()
    rows = []
    for well, k, bc, wp in cases(args.wells):
        for backend in ('rust', 'casadi') if well < args.casadi_wells else ('rust',):
            rec = dict(backend=backend, well=well, k=k, f_g=wp.fluid.f_g, w_lg=bc.w_lg)
            for v, scale in SCALES.items():
                rec.update(
                    {
                        f'{key}_{v}': val
                        for key, val in solve(
                            replace(wp, slip=replace(wp.slip, profile_scale=scale)), bc, backend
                        ).items()
                    }
                )
            rows.append(rec)
            print(
                f'{backend:>6} well {well:2d} k {k}: PBH '
                + ' -> '.join(f'{rec.get(f"PBH_{v}", float("nan")):8.3f}' for v in SCALES),
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary(df)


if __name__ == '__main__':
    main()
