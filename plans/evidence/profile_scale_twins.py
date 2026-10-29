"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Calibration twins for feature 020 (specs/features/020-profile-scale.md, Acceptance, item 7): the scale of the profile
parameter, profile_scale (SLIP-18), as a free parameter (CAL-5, CAL-6). The wells are the twin study's of 017
(develop's sampler, seed 2026, 20 cells, at their nominal operating point), all synthetic.

    recovery      every instrumentation, 20 rows, with K_c, w_l_max, roughness, h and profile_scale free and z*
                  drawn from the five priors (N(0, 1) clipped to +-2.5): the error of each parameter in prior
                  standard deviations, the share of fits that converge and end closer to z* than the medians, and
                  the correlation across fits of the errors of profile_scale and roughness
    misspecified  the first wells, with the truth's roughness at its prior median and its profile_scale drawn:
                  20 rows at choke positions 0.2-0.5, calibrated three ways,
                    friction  roughness free, profile_scale held at 1 (develop's model before 020)
                    scale     profile_scale free, roughness held at its median (the truth's)
                    both      both free
                  each with K_c, w_l_max and h free too; the fitted values, and the RMS of the scaled residuals on
                  10 held-out rows at choke positions 0.7-1.0, higher rates than the calibration saw

From the repository root (about an hour on 8 threads):

    uv run python plans/evidence/profile_scale_twins.py profile_scale_twins.json --wells 30 --detail 8
"""

import argparse
import json
import time
import traceback

import numpy as np

from manywells.calibration import CalibrationError, apply, calibrate, evaluate, synthetic_data
from manywells.calibration.fit import priors_for
from manywells.calibration.synthetic import INSTRUMENTATIONS
from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

FREE = ['K_c', 'w_l_max', 'roughness', 'h', 'profile_scale']
N_ROWS, N_HOLDOUT = 20, 10
LOW, HIGH = (0.2, 0.5), (0.7, 1.0)      # choke positions of the misspecified fits' rows and of their held-out rows
WAYS = {'friction': ['K_c', 'w_l_max', 'roughness', 'h'], 'scale': ['K_c', 'w_l_max', 'h', 'profile_scale'],
        'both': FREE}


def rms(table) -> float:
    res = table[[c for c in table if c.endswith('_res')]].to_numpy().ravel()
    res = res[~np.isnan(res)]
    return float(np.sqrt(np.mean(res ** 2))) if len(res) else float('nan')


def fit(wp, bc, truth_wp, free, priors, seed, workers, instrumentation='full', u=(0.2, 1.0), holdout_u=(0.2, 1.0)):
    """Calibrate wp on N_ROWS noisy rows of truth_wp; a record with the fit and the held-out RMS."""
    t = time.perf_counter()
    data = synthetic_data(truth_wp, bc, {}, N_ROWS, seed=seed, instrumentation=instrumentation, workers=workers, u=u)
    result = calibrate(wp, data, bc, free, priors={p.name: p for p in priors if p.name in free}, workers=workers)
    holdout = synthetic_data(truth_wp, bc, {}, N_HOLDOUT, seed=seed + 1000, workers=workers, u=holdout_u)
    return {'rows': len(data), 'success': bool(result.success), 'solves': result.n_solves,
            'seconds': time.perf_counter() - t, 'z': result.z, 'values': result.values, 'rms': result.rms(),
            'holdout_rms': rms(evaluate(result.well, holdout, bc, workers=workers))}


def run_well(seed, well, n_cells, n_detail, workers):
    draw = sample_well(seed, well)
    wp = well_properties(draw, configuration=DEVELOP, n_cells=n_cells)
    bc = nominal_conditions(draw)
    rec = {'well': well, 'trajectory': draw.trajectory[0], 'records': []}
    try:
        SSDFSimulator(wp, backend='rust').simulate(bc)
    except (SimError, ValueError) as e:
        rec['skipped'] = f'no operating point at the nominal conditions: {e}'
        return rec
    priors = priors_for(wp, FREE)
    z_true = np.clip(rng_for(seed, well, 'twin-020').normal(size=len(FREE)), -2.5, 2.5)
    rec['z_true'] = dict(zip(FREE, z_true.tolist()))
    truth_wp = apply(wp, {p.name: p.value(z) for p, z in zip(priors, z_true)})

    def add(part, label, f):
        try:
            r = f()
        except (CalibrationError, SimError, ValueError) as e:
            r = {'error': f'{type(e).__name__}: {e}'}
        except Exception as e:  # A failure is a finding of the study, not a reason to stop it
            r = {'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}
        rec['records'].append({'part': part, 'label': label, **r})
        print(f'well {well:3d} {draw.trajectory[0]:9s} {part:12s} {label:16s} '
              + (r['error'][:80] if 'error' in r else
                 ' '.join(f'{n} {r["z"][n]:+.2f}' for n in r['z'])
                 + f' holdout {r["holdout_rms"]:.2f} {r["seconds"]:.0f}s'), flush=True)

    s = 100 * well
    for k, inst in enumerate(INSTRUMENTATIONS):
        add('recovery', inst, lambda: fit(wp, bc, truth_wp, FREE, priors, s + k, workers, instrumentation=inst))
    if well < n_detail:
        # The truth's roughness at its median, its profile_scale drawn; the other parameters as drawn
        mis = dict(zip(FREE, z_true))
        mis['roughness'] = 0.0
        rec['z_misspecified'] = {n: float(z) for n, z in mis.items()}
        mis_wp = apply(wp, {p.name: p.value(mis[p.name]) for p in priors})
        for way, free in WAYS.items():
            # What is not free keeps the calibrated well's value: roughness at its median, profile_scale at 1
            calib_wp = apply(wp, {p.name: p.median for p in priors if p.name in free or p.name == 'roughness'})
            calib_wp = apply(calib_wp, {'profile_scale': 1.0} if 'profile_scale' not in free else {})
            add('misspecified', way, lambda: fit(calib_wp, bc, mis_wp, free, priors, s + 30, workers, u=LOW,
                                                 holdout_u=HIGH))
    return rec


def summary(records):
    lines = []
    fits = [(w, r) for w in records if 'records' in w for r in w['records'] if 'error' not in r]
    errors = [(w['well'], r['part'], r['label'], r['error'][:100]) for w in records if 'records' in w
              for r in w['records'] if 'error' in r]
    lines += ['## Recovery (20 rows, five parameters free): error in prior standard deviations, median and largest',
              '', '| Instrumentation | fits | converged | closer than the medians | '
              + ' | '.join(f'`{n}`' for n in FREE) + ' | corr(`profile_scale`, `roughness`) errors | held-out RMS |',
              '|---|---|---|---|' + '---|' * len(FREE) + '---|---|']
    for inst in INSTRUMENTATIONS:
        rs = [(w, r) for w, r in fits if r['part'] == 'recovery' and r['label'] == inst]
        if not rs:
            continue
        e = {n: np.array([r['z'][n] - w['z_true'][n] for w, r in rs]) for n in FREE}
        closer = np.mean([np.linalg.norm([r['z'][n] - w['z_true'][n] for n in FREE])
                          < np.linalg.norm([w['z_true'][n] for n in FREE]) for w, r in rs])
        corr = np.corrcoef(e['profile_scale'], e['roughness'])[0, 1]
        lines.append(f'| `{inst}` | {len(rs)} | {np.mean([r["success"] for _, r in rs]):.0%} | {closer:.0%} | '
                     + ' | '.join(f'{np.median(np.abs(e[n])):.2f}, {np.max(np.abs(e[n])):.2f}' for n in FREE)
                     + f' | {corr:+.2f} | {np.median([r["holdout_rms"] for _, r in rs]):.2f} |')
    lines += ['', '## Misspecified: the truth\'s roughness at its median, profile_scale drawn; rows at u 0.2-0.5, '
              'held out at u 0.7-1.0', '',
              '| Well | z* `profile_scale` | friction: z `roughness`, held-out RMS | scale: z `profile_scale`, held-out RMS '
              '| both: z `roughness`, z `profile_scale`, held-out RMS |', '|---|---|---|---|---|']
    for w in records:
        if 'z_misspecified' not in w:
            continue
        by = {r['label']: r for r in w['records'] if r['part'] == 'misspecified' and 'error' not in r}
        cell = {
            'friction': lambda r: f'{r["z"]["roughness"]:+.2f}, {r["holdout_rms"]:.2f}',
            'scale': lambda r: f'{r["z"]["profile_scale"]:+.2f}, {r["holdout_rms"]:.2f}',
            'both': lambda r: f'{r["z"]["roughness"]:+.2f}, {r["z"]["profile_scale"]:+.2f}, {r["holdout_rms"]:.2f}'}
        lines.append(f'| {w["well"]} | {w["z_misspecified"]["profile_scale"]:+.2f} | '
                     + ' | '.join(cell[k](by[k]) if k in by else 'failed' for k in WAYS) + ' |')
    if errors:
        lines += ['', '## Failed fits', ''] + [f'- well {w}, {p}, {lab}: {e}' for w, p, lab, e in errors]
    skipped = [w for w in records if 'skipped' in w]
    if skipped:
        lines += ['', 'Skipped: ' + ', '.join(f'well {w["well"]} ({w["skipped"][:60]})' for w in skipped)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='JSON file for every record')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--wells', type=int, default=30)
    parser.add_argument('--detail', type=int, default=8, help='wells with the misspecified fits')
    parser.add_argument('--n-cells', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8, help="threads for the Rust core's solves")
    args = parser.parse_args()
    records = []
    for well in range(args.wells):
        records.append(run_well(args.seed, well, args.n_cells, args.detail, args.workers))
        with open(args.out, 'w') as f:
            json.dump(records, f, indent=1, default=float)
    print(summary(records))


if __name__ == '__main__':
    main()
