"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Reference root sets (v2 plan, Step 2 item 4) from `make_cases.py solve`. Runs in the develop
environment:

    uv run python verification/build/label_cases.py

Every root is a v1.0.0 solution. A root is accepted if Ipopt reported Solve_Succeeded and it
passes the Invariants check; a case with any other v1 solution is not settled, since that
solution may be a root the rule cannot judge (for example Feasible_Point_Found at a fold). It was found by method A if a v1 start (default guess, the dataset
generator's start, an interpolated coarser root, or a cellwise guess) reached it, and by method B
if v1 reached it from a Rust root. Its label is the sign of its normalized stability slope, and
indeterminate when the magnitude is at most LABEL_MIN. A case is settled when both methods give
the same accepted roots, no label is indeterminate and no solution was rejected. Settled cases are written to
verification/data/ (committed): cases.parquet, reference_roots.parquet, and v1's own solutions
as candidates, v1_cold.parquet (default guess) and v1_dataset.parquet (the dataset generator's
start, where there is one). The other cases go to data/build/disagreements.md for Bjarne.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from manywells_verify.cases import Case, Root, Variant, write_cases, write_roots
from manywells_verify.checks import Tolerances, check_invariants

VERIFICATION = Path(__file__).resolve().parents[1]
BUILD, OUT = VERIFICATION / 'data' / 'build', VERIFICATION / 'data'
LABEL_MIN = 1e-3          # |normalized dR/dp0| at or below which a label is indeterminate


def case_from(d: dict) -> Case:
    return Case(d['case_id'], d['params'], d['n_cells'], Variant(*d['variant']), source=d['source'],
                group=d['group'], meta={'config_id': d['config_id'], 'seed': d['seed'], 'note': d['note']})


def label_of(slope: float) -> str:
    return 'indeterminate' if abs(slope) <= LABEL_MIN else ('unstable' if slope > 0 else 'stable')


def method(starts) -> set:
    return {'B' if s.startswith('rust') else 'A' for s in starts}


def main():
    cases = json.loads((BUILD / 'cases.json').read_text())
    runs = json.loads((BUILD / 'v1_runs.json').read_text())
    with np.load(BUILD / 'v1_arrays.npz') as f:
        arrays = dict(f)

    settled, reference, cold, dataset, lines = [], {}, {}, {}, []
    for d in cases:
        cid, case, r = d['case_id'], case_from(d), runs[d['case_id']]
        roots = []
        for root in r['roots']:
            x = arrays[f'{cid}.{root["key"]}']
            ok = root['return_status'] == 'Solve_Succeeded' and not check_invariants(x, case, None, Tolerances()).failed
            if ok:
                roots.append(dict(root, x=x, label=label_of(root['slope']), methods=method(root['starts'])))
        a = [x for x in roots if 'A' in x['methods']]
        b = [x for x in roots if 'B' in x['methods']]
        indeterminate = [x for x in roots if x['label'] == 'indeterminate']
        rejected = [x for x in r['roots'] if not any(x['key'] == y['key'] for y in roots)]
        if len(a) == len(b) == len(roots) and not indeterminate and not rejected:
            settled.append(case)
            reference[cid] = [Root(x['x'], x['label'], info={'slope': x['slope'], 'spread': x['spread'], 'starts': ', '.join(x['starts'])})
                              for x in roots]
        else:
            why = []
            if not len(a) == len(b) == len(roots):
                why.append('v1 starts: ' + ', '.join(f'{x["p0"]:.3f}' for x in a) + ' bar; Rust roots re-solved: '
                           + ', '.join(f'{x["p0"]:.3f}' for x in b) + ' bar')
            if indeterminate:
                why.append(f'{len(indeterminate)} indeterminate label(s)')
            if rejected:
                why.append('rejected v1 solution(s): ' + ', '.join(
                    f'{x["p0"]:.3f} bar ({x["return_status"]}, slope {x["slope"]:+.1e})' for x in rejected))
            lines.append(f'- `{cid}` ({d["source"]}; {d["note"] or "-"}): ' + '; '.join(why) + '. Roots: '
                         + ', '.join(f'{x["p0"]:.3f} bar {x["label"]} (slope {x["slope"]:+.1e}, from {", ".join(x["starts"])})'
                                     for x in roots))

        # v1's own solutions as candidates. Convergence members follow the stable root on every grid.
        if d['source'] == 'convergence':
            if f'{cid}.interp' in arrays:
                cold[cid] = [Root(arrays[f'{cid}.interp'], operating_point=True, info={'v1_start': 'interp'})]
            else:
                stable = [x for x in roots if x['label'] == 'stable' and 'A' in x['methods']]
                if stable:
                    cold[cid] = [Root(stable[0]['x'], operating_point=True, info={'v1_start': stable[0]['starts'][0]})]
        elif f'{cid}.cold' in arrays:
            cold[cid] = [Root(arrays[f'{cid}.cold'], operating_point=True, info={'v1_start': 'cold'})]
        for name in ('warm', 'dataset'):
            if f'{cid}.{name}' in arrays:
                dataset[cid] = [Root(arrays[f'{cid}.{name}'], operating_point=True, info={'v1_start': name})]

    # A convergence group is kept only if all three members are settled
    groups = Counter(c.group for c in settled if c.group)
    settled = [c for c in settled if not c.group or groups[c.group] == 3]
    ids = {c.case_id for c in settled}

    OUT.mkdir(parents=True, exist_ok=True)
    write_cases(settled, OUT / 'cases.parquet')
    write_roots({k: v for k, v in reference.items() if k in ids}, OUT / 'reference_roots.parquet')
    write_roots({k: v for k, v in cold.items() if k in ids}, OUT / 'v1_cold.parquet')
    write_roots({k: v for k, v in dataset.items() if k in ids}, OUT / 'v1_dataset.parquet')
    (BUILD / 'disagreements.md').write_text(
        f'# Cases left out ({len(lines)})\n\nThe two root searches disagree, a label is indeterminate '
        f'(|normalized dR/dp0| <= {LABEL_MIN:g}), or a v1 solution was rejected (no Solve_Succeeded, or it '
        'fails Invariants). These cases are not in the case set (Bjarne, 2026-09-30).\n\n' + '\n'.join(lines) + '\n')

    counts = Counter((c.source, len(reference[c.case_id])) for c in settled)
    print(f'{len(settled)} settled cases, {len(lines)} for adjudication')
    for (source, n), k in sorted(counts.items()):
        print(f'  {source:28s} {n} root(s): {k}')


if __name__ == '__main__':
    main()
