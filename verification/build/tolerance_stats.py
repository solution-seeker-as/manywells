"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Measurements behind the tolerance proposal (v2 plan, Step 2 item 5), on the committed case set.
Runs in the develop environment after label_cases.py:

    uv run python verification/build/tolerance_stats.py

Reports quantiles of: how far apart v1's solves of one root land from different starts (the
floor for tol_x); the smallest distance between two roots of a case (the ceiling for tol_x); the
reference roots' |normalized dR/dp0| (against the indeterminate threshold); invariant margins at
the reference roots; and the observed convergence orders of v1's solutions.
"""

from pathlib import Path

import numpy as np

from manywells_verify.cases import read_cases, read_roots
from manywells_verify.checks import Tolerances, ambient_temperature, check_convergence, grid, mass_rates, state_distance

DATA = Path(__file__).resolve().parents[1] / 'data'


def quantiles(name, values, unit=''):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if not len(v):
        print(f'{name:50s} (no values)')
        return
    q = np.quantile(v, [0, 0.5, 0.9, 0.99, 1])
    print(f'{name:50s} n={len(v):4d}  min {q[0]:.1e}  median {q[1]:.1e}  p90 {q[2]:.1e}  p99 {q[3]:.1e}  max {q[4]:.1e} {unit}')


def main():
    cases = read_cases(DATA / 'cases.parquet')
    reference = read_roots(DATA / 'reference_roots.parquet')
    v1 = read_roots(DATA / 'v1_cold.parquet')

    spread, separation, slopes = [], [], []
    p_rise, flux, below_ambient, wellhead_margin, choke_margin = [], [], [], [], []
    for cid, case in cases.items():
        refs = reference.get(cid, [])
        spread += [r.info['spread'] for r in refs if r.info['spread'] > 0]
        slopes += [abs(r.info['slope']) for r in refs]
        separation += [state_distance(a.x, b.x, case) for i, a in enumerate(refs) for b in refs[i + 1:]]
        for r in refs:
            X = grid(r.x, case.n_cells)
            w_g, w_l = mass_rates(X, case)
            flux.append(max(np.abs(w_g - w_g[0]).max(), np.abs(w_l - w_l[0]).max()) / (w_g[0] + w_l[0]))
            p_rise.append(np.diff(X[:, 0]).max())
            below_ambient.append((ambient_temperature(case) - X[:, 6]).max())
            wellhead_margin.append(X[-1, 0] - case.params['p_s'])
            choke_margin.append(abs(case.params['cpr'] * X[-1, 0] - case.params['p_s']))

    print(f'{len(cases)} cases, {sum(len(v) for v in reference.values())} reference roots\n')
    quantiles('one root, different v1 starts: distance apart', spread)
    quantiles('distinct roots of a case: smallest distance', separation)
    quantiles('reference roots: |normalized dR/dp0|', slopes)
    quantiles('largest pressure rise between points', p_rise, 'bar')
    quantiles('phase mass-rate variation / total rate', flux)
    quantiles('largest temperature below ambient', below_ambient, 'K')
    quantiles('wellhead pressure above p_s', wellhead_margin, 'bar')
    quantiles('|cpr p_N - p_s|', choke_margin, 'bar')

    groups = {}
    for cid, case in cases.items():
        if case.group:
            op = [r for r in v1.get(cid, []) if r.operating_point]
            groups.setdefault(case.group, []).append((case, op[0] if op else None))
    print(f'\nConvergence groups (v1 on the stable root): {len(groups)}')
    for g, members in groups.items():
        c = check_convergence(members, Tolerances(order=(-np.inf, np.inf)))
        print(f'  {g}: {c.status} {c.detail}')


if __name__ == '__main__':
    main()
