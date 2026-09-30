"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Command line: manywells-verify CANDIDATE --data DIR [--expected-failures CSV] [--json OUT]

DIR holds cases.parquet and reference_roots.parquet. CANDIDATE is a parquet file in the root
file format (manywells_verify.cases). Prints the Markdown report and exits with status 1 if
there are unexpected failures.
"""

import argparse
import json
import sys
from pathlib import Path

from manywells_verify.cases import read_cases, read_roots
from manywells_verify.checks import Tolerances
from manywells_verify.report import read_expected_failures, verify
from manywells_verify.residual import ResidualGraph


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='manywells-verify', description='Check candidate roots against the '
                                     'frozen ManyWells residual graph and a case set with reference root sets.')
    parser.add_argument('candidate', type=Path, help='candidate roots (parquet, root file format)')
    parser.add_argument('--data', type=Path, required=True, help='folder with cases.parquet and reference_roots.parquet')
    parser.add_argument('--expected-failures', type=Path, help='CSV with columns case_id, check, reason')
    parser.add_argument('--json', type=Path, help='write the full report here as JSON')
    parser.add_argument('--name', help='candidate name in the report (default: the file name)')
    args = parser.parse_args(argv)

    cases = read_cases(args.data / 'cases.parquet')
    reference = read_roots(args.data / 'reference_roots.parquet')
    candidate = read_roots(args.candidate)
    unknown = sorted(set(candidate) - set(cases))
    if unknown:
        parser.error(f'{len(unknown)} candidate case ids are not in the case set, e.g. {unknown[0]}')
    models = {c.model for c in cases.values()}
    if len(models) != 1:
        parser.error(f'the case set mixes models {sorted(models)}; verify one model at a time')

    graph = ResidualGraph(models.pop())
    report = verify(graph, cases, candidate, reference, Tolerances(),
                    read_expected_failures(args.expected_failures), args.name or args.candidate.stem)
    print(report.to_markdown())
    if args.json:
        args.json.write_text(json.dumps(report.to_json(), indent=1, default=float) + '\n')
    return 1 if report.unexpected_failures() else 0


if __name__ == '__main__':
    sys.exit(main())
