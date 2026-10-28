"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The implemented feature 019 (specs/features/019-five-flow-regimes.md) on the wells and operating points of
dak_jt_default.py (develop's sampler, seed 2026, 88 wells at their nominal operating point and two sampled ones), in
two variants of develop's default configuration:

  fitted   v1.0.0's fitted three-regime classifier: develop's default before 019
  hasan    the five regimes from the rules of Hasan et al.'s hierarchy: develop's default after 019

Per case, backend and variant it records whether there is an operating point, the number of roots, PBH, PWH, TWH,
the reservoir liquid rate, the share of the operating point's grid points in each regime, and the root search's
time, on one process, so that the times compare. The Rust core solves every case; the CasADi backend the cases of
the first --casadi-wells wells.

From the repository root (about 15 minutes with --casadi-wells 30):

    uv run python plans/evidence/five_regime_default.py five_regime_default.csv --casadi-wells 30
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
from manywells.slip import FITTED_REGIMES, HASAN_REGIMES, SlipModel

SEED = 2026
VARIANTS = {'fitted': SlipModel(regime_model='fitted'), 'hasan': SlipModel(regime_model='hasan')}
LABELS = sorted(set(FITTED_REGIMES) | set(HASAN_REGIMES))


def cases(n_wells):
    """The cases of dak_jt_default.py: each well at its nominal operating point and two sampled ones."""
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
        shares = Counter(op.flow_regime)
        out.update(PBH=X[0, 0], PWH=X[-1, 0], TWH=X[-1, 6], w_res=op.w_res,
                   **{f'share_{r}': shares[r] / len(op.flow_regime) for r in LABELS})
    return out


def summary(df):
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    pd.set_option('display.width', 250)
    for backend, d in df.groupby('backend'):
        print(f'== {backend}: {len(d)} cases')
        for v in VARIANTS:
            s = d[f'seconds_{v}']
            print(f'  {v}: operating point in {d[f"PBH_{v}"].notna().sum()}, roots per case '
                  f'{dict(sorted(Counter(d[f"roots_{v}"]).items()))}, {s.median():.2f} s median, {s.sum():.1f} s total')
        both = d.dropna(subset=[f'PBH_{v}' for v in VARIANTS])
        lost = d[d['PBH_fitted'].notna() & d['PBH_hasan'].isna()]
        gained = d[d['PBH_fitted'].isna() & d['PBH_hasan'].notna()]
        print(f'  {len(both)} with an operating point in both; lost with hasan: {list(zip(lost.well, lost.k))}; '
              f'gained: {list(zip(gained.well, gained.k))}')
        diff = pd.DataFrame({'PBH': both['PBH_hasan'] - both['PBH_fitted'], 'PWH': both['PWH_hasan'] - both['PWH_fitted'],
                             'TWH': both['TWH_hasan'] - both['TWH_fitted'],
                             'rate (rel)': both['w_res_hasan'] / both['w_res_fitted'] - 1})
        print(diff.quantile(q).round(4).to_string())
        for v in VARIANTS:
            shares = {r: d[f'share_{r}_{v}'].mean() for r in LABELS if d[f'share_{r}_{v}'].sum() > 0}
            print(f'  {v} regimes along the operating points: ' + ', '.join(f'{r} {s:.1%}' for r, s in shares.items()))


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
            for v, slip in VARIANTS.items():
                rec.update({f'{key}_{v}': val for key, val in solve(replace(wp, slip=slip), bc, backend).items()})
            rows.append(rec)
            print(f"{backend:>6} well {well:2d} k {k}: PBH {rec.get('PBH_fitted', float('nan')):8.3f} -> "
                  f"{rec.get('PBH_hasan', float('nan')):8.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary(df)


if __name__ == '__main__':
    main()
