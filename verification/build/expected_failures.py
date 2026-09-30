"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The expected-failure list for v1.0.0 (v2 plan, Step 2 item 6): verification/expected_failures.csv.

Runs the verifier on v1's default-guess solutions (verification/data/v1_cold.parquet) and lists an
Operating point failure as expected only when v1's own outcome explains it: its default guess
reached the unstable trickle root, or it failed. These are the known v1 defect (no root selection)
and v1's 95.6% success rate. Every other failure is left unexpected, as a finding for Bjarne.
Runs in the develop environment after label_cases.py:

    uv run python verification/build/expected_failures.py
"""

import csv
import json
from pathlib import Path

from manywells_verify.cases import read_cases, read_roots
from manywells_verify.report import verify

VERIFICATION = Path(__file__).resolve().parents[1]
DATA, BUILD = VERIFICATION / 'data', VERIFICATION / 'data' / 'build'


def main():
    cases = read_cases(DATA / 'cases.parquet')
    report = verify(cases, read_roots(DATA / 'v1_cold.parquet'), read_roots(DATA / 'reference_roots.parquet'),
                    name='v1.0.0')
    runs = json.loads((BUILD / 'v1_runs.json').read_text())

    rows, other = [], []
    for case_id, check, detail in report.unexpected_failures():
        cold = runs.get(case_id, {}).get('cold', {})
        if check == 'operating_point' and cases[case_id].source != 'convergence':
            if cold.get('status') == 'failed':
                rows.append({'case_id': case_id, 'check': check, 'reason': f'v1 default guess fails ({cold.get("stage")})'})
                continue
            if cold.get('slope', 0) > 0:
                rows.append({'case_id': case_id, 'check': check,
                             'reason': f'v1 default guess reaches the trickle root (p0 = {cold["p0"]:.3f} bar)'})
                continue
        other.append((case_id, check, detail.detail))

    with open(VERIFICATION / 'expected_failures.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'check', 'reason'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['case_id']))
    print(f'{len(rows)} expected failures written; {len(other)} other failures (findings for Bjarne):')
    for case_id, check, detail in other:
        print(f'  {case_id} {check}: {detail}')


if __name__ == '__main__':
    main()
