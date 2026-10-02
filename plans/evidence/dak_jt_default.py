"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The implemented feature 016 (specs/features/016-joule-thomson.md) on the wells and operating points of
jt_wells.py (develop's sampler, seed 2026, 88 wells at their nominal operating point and two sampled ones), solved
by the Rust core in four variants of develop's default configuration:

  old        Papay's z-factor, no Joule-Thomson term: develop's default before 016
  dak        the Dranchuk-Abou-Kassem gas law (PVT-GAS-11), no Joule-Thomson term
  papay_jt   Papay's z-factor with the Joule-Thomson term (DAK's factor at Papay's density), as jt_wells.py's dak_sub
  new        DAK with the Joule-Thomson term: develop's default after 016

Per case and variant it records PBH, PWH, TWH, the minimum over points i >= 1 of T_i - T_a,i, min T - T_s, the
search's time and the core's work counters (march.rs, Counts), on one process, so that the times compare.

From the repository root (about 5 minutes):

    uv run python plans/evidence/dak_jt_default.py dak_jt_default.csv
"""

import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026
VARIANTS = {'old': ('papay', False), 'dak': ('dak', False), 'papay_jt': ('papay', True), 'new': ('dak', True)}
COUNTS = ('marches', 'states', 'temperature_solves', 'chord_fallbacks', 'step_outs', 'lower_step_outs',
          'temperature_minima', 'temperature_failures', 'non_finite', 'edge_refinements')


def cases(n_wells):
    for well in range(n_wells):
        draw = sample_well(SEED, well)
        rng = rng_for(SEED, well, 'jt')
        for k in range(3):
            bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
            try:
                yield well, k, draw, bc, well_properties(draw, fractions, DEVELOP)
            except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
                continue


def solve(wp, bc):
    sim = SSDFSimulator(wp, backend='rust')
    t0 = time.perf_counter()
    try:
        op = sim.simulate(bc)
    except SimError:
        op = None
    out = {'seconds': time.perf_counter() - t0, **{c: sim._roots.counts.get(c, 0) for c in COUNTS}}
    if op is not None:
        X = op.state
        T_a = np.array([wp.thermal.ambient_temperature(f, bc.T_r, bc.T_s) for f in wp.geometry.tvd_frac])
        out.update(PBH=X[0, 0], PWH=X[-1, 0], TWH=X[-1, 6], min_T_minus_Ta=float(np.min(X[1:, 6] - T_a[1:])),
                   min_T_minus_Ts=float(np.min(X[:, 6]) - bc.T_s))
    return out


def summary(df):
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    print(f'{len(df)} cases')
    for v in VARIANTS:
        print(f'  {v}: operating point in {df[f"TWH_{v}"].notna().sum()}')
    ok = df.dropna(subset=[f'TWH_{v}' for v in VARIANTS])
    print(f'{len(ok)} with an operating point in every variant')
    diff = pd.DataFrame({f'{k} {a}-{b}': ok[f'{k}_{a}'] - ok[f'{k}_{b}']
                         for a, b in (('new', 'old'), ('dak', 'old'), ('new', 'dak'), ('papay_jt', 'old'))
                         for k in ('TWH', 'PWH', 'PBH')})
    pd.set_option('display.width', 250)
    print(diff.quantile(q).round(2).to_string())
    for v in VARIANTS:
        c = ok[f'min_T_minus_Ta_{v}']
        print(f'{v}: colder than ambient somewhere in {(c < 0).sum()} (min {c.min():.2f} K); '
              f'min T - T_s {ok[f"min_T_minus_Ts_{v}"].min():.2f} K')
    print('Work, every case:')
    for v in VARIANTS:
        s = df[f'seconds_{v}']
        print(f'  {v}: {s.median() * 1e3:.1f} ms median, {s.sum():.1f} s total; '
              + ', '.join(f'{c} {int(df[f"{c}_{v}"].sum())}' for c in COUNTS))


def main(out):
    rows = []
    for well, k, draw, bc, wp in cases(88):
        rec = dict(well=well, k=k, f_g=wp.fluid.f_g, w_lg=bc.w_lg, p_r=bc.p_r)
        for v, (z, jt) in VARIANTS.items():
            w = replace(wp, fluid=replace(wp.fluid, z_factor_model=z), thermal=replace(wp.thermal, joule_thomson=jt))
            rec.update({f'{key}_{v}': val for key, val in solve(w, bc).items()})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary(df)


if __name__ == '__main__':
    main(sys.argv[1])
