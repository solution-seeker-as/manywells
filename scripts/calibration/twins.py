"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The twin study of the calibration (plans/calibration-plan.md, Step C6; specs/calibration.md, Checks), for the
measurements of specs/features/017-calibration.md. Run from the project root:

    uv run python -m scripts.calibration.twins twins.json --wells 30

Wells are drawn with manywells.sampling (develop's model), each with parameters theta* drawn from the priors, and
calibrated from the priors' medians on synthetic data (calibration.synthetic):

    recovery         every instrumentation, 20 rows: the error in prior standard deviations, whether the fit got
                     closer than the medians, and the held-out rows' RMS of scaled residuals
    consistency      full instrumentation, 10, 40 and 160 rows and the noise scaled by 1 and 1/4 (the first wells)
    misspecified     data from a model the calibration lacks (the first wells): the bias and the held-out RMS
    backends         the Rust core and the CasADi backend on small twins (the first wells)

Every record is written to the JSON file, and the tables are printed as Markdown.
"""

import argparse
import json
import time
import traceback
from dataclasses import replace

import numpy as np

from manywells.calibration import CalibrationError, Noise, apply, calibrate, evaluate, synthetic_data
from manywells.calibration.fit import priors_for
from manywells.calibration.synthetic import INSTRUMENTATIONS
from manywells.choke import CHOKE_PROFILES, BernoulliChokeModel
from manywells.configurations import DEVELOP
from manywells.friction import FixedFrictionFactor
from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator
from manywells.slip import SlipModel

FREE = ['K_c', 'w_l_max', 'roughness', 'h']
SHARED = ['K_c', 'w_l_max', 'h']       # The parameters every misspecified fit has, for its bias
N_ROWS, N_HOLDOUT = 20, 10


def scaled(noise: Noise, factor: float) -> Noise:
    return Noise(PBH=noise.PBH * factor, PWH=noise.PWH * factor, TWH=noise.TWH * factor,
                 rate_test=noise.rate_test * factor, rate_mpfm=noise.rate_mpfm * factor)


def holdout_rms(well, wp_truth, bc, seed, noise=None):
    """The RMS of the scaled residuals of the calibrated well on N_HOLDOUT new noisy rows of the truth."""
    holdout = synthetic_data(wp_truth, bc, {}, N_HOLDOUT, seed=seed, noise=noise)
    table = evaluate(well, holdout, bc)
    res = table[[c for c in table if c.endswith('_res')]].to_numpy().ravel()
    res = res[~np.isnan(res)]
    return float(np.sqrt(np.mean(res ** 2))) if len(res) else float('nan')


def fit(wp, bc, data_wp, truth, free, priors, seed, instrumentation='full', n_rows=N_ROWS, noise=None,
        backend='rust', holdout=True):
    """One calibration of wp on synthetic data of data_wp at the values truth, on a backend; a record of it. The data
    come from the Rust core whatever the backend, so that both backends calibrate the same rows."""
    t = time.perf_counter()
    data = synthetic_data(data_wp, bc, truth, n_rows, seed=seed, instrumentation=instrumentation, noise=noise)
    result = calibrate(wp, data, bc, free, priors={p.name: p for p in priors if p.name in free}, backend=backend)
    rec = {'rows': len(data), 'success': result.success, 'solves': result.n_solves,
           'seconds': time.perf_counter() - t, 'z': result.z, 'rms': result.rms(),
           'moved_start': any(v != 0 for v in result.start.values())}
    if holdout:
        rec['holdout_rms'] = holdout_rms(result.well, apply(data_wp, truth), bc, seed + 1000, noise)
    return rec


def misspecified_variants(wp):
    """(name, the well the data come from, the well the calibration has, its free parameters)."""
    profile = CHOKE_PROFILES[(CHOKE_PROFILES.index(wp.choke.chk_profile) + 1) % len(CHOKE_PROFILES)]
    slip = wp.slip
    return [
        ('fixed f_D for roughness friction', wp, replace(wp, friction=FixedFrictionFactor(f_D=0.02)),
         ['K_c', 'w_l_max', 'f_D', 'h']),
        ('Bernoulli for Simpson choke', wp,
         replace(wp, choke=BernoulliChokeModel(K_c=wp.choke.K_c, chk_profile=wp.choke.chk_profile)), FREE),
        (f'choke profile {profile} for {wp.choke.chk_profile}', wp,
         replace(wp, choke=replace(wp.choke, chk_profile=profile)), FREE),
        ('slip C_0 10% low', replace(wp, slip=SlipModel(C_0_annular=1.1 * slip.C_0_annular,
                                                        C_0_slug=1.1 * slip.C_0_slug,
                                                        C_0_bubbly=1.1 * slip.C_0_bubbly)), wp, FREE),
        ('energy terms left out', wp,
         replace(wp, thermal=replace(wp.thermal, frictional_heating=False, gravity_term=False, joule_thomson=False)),
         FREE),
    ]


def run_well(seed, well, n_cells, parts, n_detail):
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
    z_true = np.clip(rng_for(seed, well, 'twin').normal(size=len(FREE)), -2.5, 2.5)
    truth = {p.name: p.value(z) for p, z in zip(priors, z_true)}
    rec['z_true'] = dict(zip(FREE, z_true.tolist()))
    truth_wp = apply(wp, truth)

    def add(part, label, f):
        try:
            r = f()
        except (CalibrationError, SimError, ValueError) as e:
            r = {'error': f'{type(e).__name__}: {e}'}
        except Exception as e:  # A failure is a finding of the study, not a reason to stop it
            r = {'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}
        rec['records'].append({'part': part, 'label': label, **r})
        print(f'well {well:3d} {draw.trajectory[0]:9s} {part:12s} {label:40s} '
              + (r['error'][:80] if 'error' in r else
                 f"err {np.linalg.norm([r['z'][n] - rec['z_true'].get(n, 0) for n in r['z'] if n in rec['z_true']]):.3f} "
                 f"holdout {r.get('holdout_rms', float('nan')):.2f} {r['seconds']:.1f}s"), flush=True)

    s = 100 * well
    if 'recovery' in parts:
        for k, inst in enumerate(INSTRUMENTATIONS):
            add('recovery', inst, lambda: fit(wp, bc, wp, truth, FREE, priors, s + k, instrumentation=inst))
    detail = well < n_detail
    if 'consistency' in parts and detail:
        for n in (10, 40, 160):
            add('consistency', f'{n} rows', lambda: fit(wp, bc, wp, truth, FREE, priors, s + 10 + n, n_rows=n))
        add('consistency', 'noise / 4', lambda: fit(wp, bc, wp, truth, FREE, priors, s + 20,
                                                    noise=scaled(Noise(), 0.25)))
    if 'misspecified' in parts and detail:
        for name, data_wp, calib_wp, free in misspecified_variants(truth_wp):
            calib_wp = apply(calib_wp, {p.name: p.median for p in priors if p.name in free})
            mpriors = priors_for(calib_wp, free, {p.name: p for p in priors if p.name in free})
            add('misspecified', name, lambda: fit(calib_wp, bc, data_wp, {}, free, mpriors, s + 30))
    if 'backends' in parts and detail:
        small = well_properties(draw, configuration=DEVELOP, n_cells=10)
        sp = priors_for(small, ['K_c', 'h'])
        st = {p.name: p.value(z) for p, z in zip(sp, z_true[[0, 3]])}
        for backend in ('rust', 'casadi'):
            add('backends', backend, lambda: fit(small, bc, small, st, ['K_c', 'h'], sp, s + 40, n_rows=4,
                                                 backend=backend, holdout=False))
    return rec


def summary(records):
    """The Markdown tables of the study."""
    lines = []
    rec = [(w, r) for w in records if 'records' in w for r in w['records']]

    def err(w, r, n):
        return abs(r['z'][n] - w['z_true'][n])

    lines += ['### Recovery (20 rows, noisy)', '',
              '| Instrumentation | fits | converged | started off the medians | closer than the medians '
              '| median error (K_c, w_l_max, roughness, h) | largest error | held-out RMS (median) |',
              '|---|---|---|---|---|---|---|---|']
    for inst in INSTRUMENTATIONS:
        rs = [(w, r) for w, r in rec if r['part'] == 'recovery' and r['label'] == inst and 'z' in r]
        failed = sum(1 for w, r in rec if r['part'] == 'recovery' and r['label'] == inst and 'error' in r)
        if not rs:
            continue
        closer = np.mean([np.linalg.norm([err(w, r, n) for n in FREE]) < np.linalg.norm(list(w['z_true'].values()))
                          for w, r in rs])
        med = [np.median([err(w, r, n) for w, r in rs]) for n in FREE]
        big = [np.max([err(w, r, n) for w, r in rs]) for n in FREE]
        moved = sum(bool(r.get('moved_start')) for _, r in rs)
        lines.append(f'| {inst} | {len(rs)} (+{failed} failed) | {np.mean([r["success"] for _, r in rs]):.0%} '
                     f'| {moved} | {closer:.0%} | {", ".join(f"{m:.2f}" for m in med)} | {", ".join(f"{b:.2f}" for b in big)} '
                     f'| {np.median([r["holdout_rms"] for _, r in rs]):.2f} |')
    lines += ['', '### Consistency (full instrumentation)', '',
              '| Data | fits | median error (K_c, w_l_max, roughness, h) |', '|---|---|---|']
    for label in ('10 rows', '40 rows', '160 rows', 'noise / 4'):
        rs = [(w, r) for w, r in rec if r['part'] == 'consistency' and r['label'] == label and 'z' in r]
        if rs:
            lines.append(f'| {label} | {len(rs)} | '
                         + ', '.join(f'{np.median([err(w, r, n) for w, r in rs]):.3f}' for n in FREE) + ' |')
    lines += ['', '### Misspecified twins', '',
              '| Model error | fits | median signed bias (K_c, w_l_max, h) | largest bias | held-out RMS (median, largest) |',
              '|---|---|---|---|---|']
    def group(label):  # The choke-profile variants differ by well; one row for them all
        return 'another choke profile' if label.startswith('choke profile') else label

    labels = sorted({group(r['label']) for _, r in rec if r['part'] == 'misspecified'})
    for label in labels:
        rs = [(w, r) for w, r in rec if r['part'] == 'misspecified' and group(r['label']) == label and 'z' in r]
        if rs:
            bias = {n: [r['z'][n] - w['z_true'][n] for w, r in rs] for n in SHARED}
            held = [r['holdout_rms'] for _, r in rs]
            lines.append(f'| {label} | {len(rs)} | '
                         + ', '.join(f'{np.median(bias[n]):+.2f}' for n in SHARED) + ' | '
                         + ', '.join(f'{max(bias[n], key=abs):+.2f}' for n in SHARED)
                         + f' | {np.median(held):.2f}, {np.max(held):.2f} |')
    pairs = {}
    for w, r in rec:
        if r['part'] == 'backends':
            pairs.setdefault(w['well'], {})[r['label']] = r
    if pairs:
        lines += ['', '### Backends (the Rust core\'s data, 4 rows, K_c and h free)', '',
                  '| Well | largest difference in z | Rust (s) | CasADi (s) |', '|---|---|---|---|']
        for well, p in sorted(pairs.items()):
            if len(p) != 2:
                continue
            rust, casadi = p['rust'], p['casadi']
            if 'z' in rust and 'z' in casadi:
                d = max(abs(rust['z'][n] - casadi['z'][n]) for n in rust['z'])
                lines.append(f'| {well} | {d:.2e} | {rust["seconds"]:.1f} | {casadi["seconds"]:.1f} |')
            else:
                failed = '; '.join(f'{k}: {r["error"][:60]}' for k, r in p.items() if 'error' in r)
                lines.append(f'| {well} | {failed} | | |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='JSON file for every record')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--wells', type=int, default=30)
    parser.add_argument('--detail', type=int, default=8, help='wells with the consistency, misspecified and '
                                                              'backend parts')
    parser.add_argument('--n-cells', type=int, default=20)
    parser.add_argument('--parts', default='recovery,consistency,misspecified,backends')
    args = parser.parse_args()
    parts = set(args.parts.split(','))
    records = []
    for well in range(args.wells):
        records.append(run_well(args.seed, well, args.n_cells, parts, args.detail))
        with open(args.out, 'w') as f:
            json.dump({'seed': args.seed, 'n_cells': args.n_cells, 'wells': records}, f, indent=1, default=float)
    print(summary(records))


if __name__ == '__main__':
    main()
