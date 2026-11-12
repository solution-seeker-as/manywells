"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 12 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The measurements of the review before the v2 datasets' full generation
(plans/sampler-redesign/preflight-measurements.md), on a run's chunks (manywells.sampling.v2.run) and on a folder of
table fluids (table_wells.py, table_build.py). Every number is ManyWells' own, from synthetic wells.

    python plans/evidence/preflight_checks.py outcomes RUN     # the wells' outcomes by class, tail and lift rule
    python plans/evidence/preflight_checks.py dead RUN         # nsol-2: where the dead dates and open slots lie
    python plans/evidence/preflight_checks.py tables TABLES    # the table fluids: failures, repair, GOR against class
    python plans/evidence/preflight_checks.py wells RUN        # the kept wells: tail p_s, tubing flag, lift capacity

RUN is a run's folder (with chunks/), TABLES a folder of tables (with wells.json). From the project root, capped:
(ulimit -v 6000000; OMP_NUM_THREADS=1 uv run python plans/evidence/preflight_checks.py ...).
"""

import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

LIMITS = ('slot_limit', 'dead_date', 'redraw_limit', 'solver_failure_limit', 'table_failed')
CLASS_GOR = {'volatile_oil': (356.0, 588.0), 'gas_condensate': (588.0, 8900.0)}  # Sm3/Sm3, SMP-55
DRY_GAS_GOR = 1e6 / 2.8  # The GOR at SMP-55's dry-gas yield
OILS = ('black_oil', 'volatile_oil')


def results(run_dir) -> list:
    """The results of a run's wells, in draw order (run._read_results, without the settings' check)."""
    out = []
    for path in sorted((Path(run_dir) / 'chunks').glob('chunk-*.pkl')):
        with open(path, 'rb') as f:
            pickle.load(f)
            out += pickle.load(f)
    return out


def share(n, total) -> str:
    return f'{n} of {total} ({100 * n / total:.1f}%)' if total else '0 of 0'


def slot_log(res) -> dict:
    """A well's slot attempts, by slot."""
    slots = collections.defaultdict(list)
    for rec in res['log']:
        if rec['kind'] == 'slot':
            slots[rec['slot']].append(rec)
    return slots


def lift_rule(res) -> str:
    """The rule that starts a well's lift, or why it has none."""
    if res['nsol'] is None:
        return res['well'].get('lift_start_rule') or 'natural'
    ns = res['nsol']
    return ns['lift_start_rule'] or ('natural, design unused' if ns['design'] else 'natural, no design')


def outcomes(run_dir):
    res = results(run_dir)
    count = collections.Counter(r['info']['outcome'] for r in res)
    print(f'{len(res)} wells drawn: {dict(count)}; accepted {share(count["accepted"], len(res))}')
    groups = collections.defaultdict(collections.Counter)
    for r in res:
        w = r['well'] or {}
        groups['class', w.get('fluid_class')][r['info']['outcome']] += 1
        groups['tail', w.get('coverage_tail')][r['info']['outcome']] += 1
    for key in sorted(groups, key=str):
        c = groups[key]
        n, dropped = sum(c.values()), sum(c[o] for o in LIMITS)
        print(
            f'  {str(key):34s} drawn {n:5d}; dropped at the limits {share(dropped, n)}',
            {o: c[o] for o in LIMITS if c[o]},
        )
    # The wells that reached their slots, by the rule that starts their lift
    by_rule = collections.defaultdict(collections.Counter)
    slots_outcomes = ('accepted', 'slot_limit', 'dead_date', 'redraw_limit', 'solver_failure_limit')
    for r in res:
        if r['info']['outcome'] in slots_outcomes:
            oil = 'oil' if r['well']['fluid_class'] in OILS else 'gas'
            by_rule[oil, lift_rule(r)][r['info']['outcome']] += 1
    print('wells that reached their slots, by lift rule:')
    for key in sorted(by_rule):
        c = by_rule[key]
        n = sum(c.values())
        print(f'  {str(key):46s} dropped {share(n - c["accepted"], n)}', {o: c[o] for o in slots_outcomes[1:] if c[o]})
    # Solver failures, by class
    attempts, with_failure = collections.defaultdict(collections.Counter), collections.defaultdict(set)
    for r in res:
        cls = (r['well'] or {}).get('fluid_class')
        for rec in r['log']:
            attempts[cls][rec['outcome']] += 1
            if rec['outcome'] == 'solver_failure':
                with_failure[cls].add(r['info']['draw_index'])
    for cls in sorted(attempts, key=str):
        a = attempts[cls]
        n = sum(a.values())
        print(
            f'  attempts {str(cls):16s} {n:8d}; solver failures {100 * a["solver_failure"] / n:.3f}%, in {len(with_failure[cls])} wells'
        )
    sec = np.array([r['seconds'] for r in res])
    print(
        f'  seconds per well: median {np.median(sec):.1f}, mean {sec.mean():.1f}, P99 {np.percentile(sec, 99):.0f}, max {sec.max():.0f}'
    )
    dead = collections.Counter(d['action'] for r in res for d in r['info'].get('dead_dates', ()))
    if dead:
        acc = [r for r in res if r['info']['outcome'] == 'accepted' and r['info'].get('dead_dates')]
        cut = np.array([1 - r['info']['T_end'] / r['info']['T_scan'] for r in acc])
        print(
            f'  dead dates that changed a life: {dict(dead)}, in {len(acc)} accepted wells; lives cut: {(cut > 0).sum()}',
            end='',
        )
        print(
            f", by a median of {np.median(cut[cut > 0]):.3f} and at most {cut.max():.3f} of the scan's life"
            if (cut > 0).any()
            else ''
        )


def dead(run_dir):
    """nsol-2: the wells dropped as dead_date or at the redraw limit, by where their open slots lie."""
    kinds = collections.Counter()
    for r in results(run_dir):
        outcome = r['info']['outcome']
        if outcome not in ('dead_date', 'redraw_limit') or r['nsol'] is None:
            continue
        ns, T_end = r['nsol'], r['info']['T_end']
        slots = slot_log(r)
        got = {s['slot'] for s in r['samples']}
        open_ = sorted(j for j in slots if j not in got)
        first = slots[open_[0]][0]
        t = [slots[j][0]['t'] / T_end for j in open_]
        if not first['lift_on'] and ns['lift_start'] is not None and first['t'] < ns['lift_start']:
            where = 'natural, before its lift starts'
        elif not first['lift_on']:
            where = 'natural, no lift to come'
        else:
            where = 'lifted'
        where += ', at the end of the life' if max(t) > 0.97 else ''
        kinds[outcome, where] += 1
        full = sum(len(slots[j]) >= 100 for j in open_)
        print(
            f'{r["info"]["draw_index"]:6d} {outcome:13s} {r["well"]["fluid_class"]:15s} rule {str(ns["lift_start_rule"]):20s}'
            f' {len(open_):3d} open slots at {min(t):.2f} to {max(t):.2f} of the life, {full} with 100 attempts; {where}'
        )
    for key, n in sorted(kinds.items()):
        print(n, key)


def tables(tables_dir):
    tables_dir = Path(tables_dir)
    spec = json.loads((tables_dir / 'wells.json').read_text())
    total, failed, errors = collections.Counter(), collections.Counter(), collections.Counter()
    repair, gors = [], collections.defaultdict(list)
    for w in spec['wells']:
        man = json.loads((tables_dir / f'k{w["well"]:06d}.json').read_text())
        key = (w['fluid_class'], w['coverage_tail'])
        total[key] += 1
        total[w['fluid_class']] += 1
        if man['status'] == 'ok':
            gors[w['fluid_class'], man['saturation_rule']].append(man['hook']['R_i'])
            continue
        failed[key] += 1
        failed[w['fluid_class']] += 1
        last = str(man.get('error')).strip().splitlines()[-1]
        errors[w['fluid_class'], man.get('step'), re.sub(r'[-+]?\d+\.?\d*(e[-+]?\d+)?', '#', last)[:90]] += 1
        m = re.search(r'p/Z falls at (\d+) of (\d+) pressures', last)
        if m:
            repair.append(int(m[1]) / int(m[2]))
    n_draws = spec['wells_range'][1]
    print(f'{spec["kind"]}, seed {spec["seed"]}: {len(spec["wells"])} table wells among {n_draws} draws')
    for key in sorted(total, key=str):
        print(f'  {str(key):34s} failed {share(failed[key], total[key])}')
    for key, n in errors.most_common():
        print(f'  {n:5d} {key}')
    if repair:
        r = np.array(repair)
        print(
            f'  repair failures: share of the pressures dropped at the GOR that failed, median {np.median(r):.3f}, max {r.max():.3f};',
            end='',
        )
        print(' within 7.5%, 10%, 15%:', [int((r <= x).sum()) for x in (0.075, 0.10, 0.15)], 'of', len(r))
    for cls, (lo, hi) in CLASS_GOR.items():
        every = np.array([g for (c, _), v in gors.items() if c == cls for g in v])
        print(
            f'  {cls}: {len(every)} tables; initial GOR below the class range {share(int((every < lo * 0.999).sum()), len(every))}',
            end='',
        )
        print(
            f', above it {share(int((every > hi * 1.001).sum()), len(every))}; above the dry-gas limit {int((every > DRY_GAS_GOR).sum())}; max {every.max():.3g}'
        )
        for rule in ('matched', 'gor_moved', 'equilibrium_phase'):
            v = np.array(gors.get((cls, rule), []))
            if len(v):
                out = int(((v < lo * 0.999) | (v > hi * 1.001)).sum())
                print(
                    f'      {rule:18s} {len(v):5d} tables, P5 {np.percentile(v, 5):.4g}, P50 {np.median(v):.4g}, P95 {np.percentile(v, 95):.4g}; outside the range {out}'
                )


def wells(run_dir):
    """The kept wells: the sol-2 tail's nominal downstream pressure, the tubing flag, the lift capacity's floor, and
    nsol-2's wells with one lifted sample."""
    res = results(run_dir)
    acc = [r for r in res if r['info']['outcome'] == 'accepted']
    sol = acc[0]['nsol'] is None
    for tail in ('core', 'tail'):
        drawn = [r['well'] for r in res if r['well'] and r['well']['coverage_tail'] == tail]
        flagged = sum('design_rate_above_tubing_capacity' in (w.get('flags') or '') for w in drawn)
        print(f'{tail}: design_rate_above_tubing_capacity on {share(flagged, len(drawn))} of the drawn wells')
        if sol:
            dry = [r['well'] for r in acc if r['well']['coverage_tail'] == tail and r['well']['well_type'] != 'subsea']
            p = np.array([w['p_s'] - w['dp_fl'] for w in dry])
            print(
                f'  kept platform and onshore wells: p_s_nom at 1.5 bar {share(int((np.abs(p - 1.5) < 1e-9).sum()), len(p))}; median {np.median(p):.1f}, max {p.max():.1f} bar'
            )
    if sol:
        cap = np.array([r['well']['w_lg_max'] for r in acc if r['well']['lift'] == 'gas_lift'])
    else:
        cap = np.array([r['nsol']['design']['w_lg_max'] for r in acc if r['nsol']['lift_start'] is not None])
        one = sum(sum(bool(s['lift_on']) for s in r['samples']) == 1 for r in acc)
        print(f'kept wells with one lifted sample: {share(one, len(acc))}')
    print(f'kept lifted wells with w_lg_max at 0.1 kg/s: {share(int((np.abs(cap - 0.1) < 1e-12).sum()), len(cap))}')


if __name__ == '__main__':
    {'outcomes': outcomes, 'dead': dead, 'tables': tables, 'wells': wells}[sys.argv[1]](sys.argv[2])
