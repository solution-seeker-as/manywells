"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Run the checks over a case set and report: a one-screen Markdown summary and a full JSON report.

A failure is expected if its (case_id, check) pair is on the expected-failure list, a CSV with
columns case_id, check, reason; for Convergence the case_id is the group id. The verdict is
PASS when there are no unexpected failures.
"""

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from manywells_verify.checks import Check, Tolerances, check_convergence, verify_case

CHECKS = ('residuals', 'invariants', 'stability', 'operating_point', 'root_set')
TITLES = {'residuals': 'Residuals', 'invariants': 'Invariants', 'stability': 'Stability',
          'operating_point': 'Operating point', 'root_set': 'Root set', 'convergence': 'Convergence (groups)'}
STATUSES = ('pass', 'fail', 'expected', 'indeterminate', 'n/a')


def read_expected_failures(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, newline='') as f:
        return {(row['case_id'], row['check']): row['reason'] for row in csv.DictReader(f)}


@dataclass
class Report:
    candidate: str
    model: str
    tolerances: Tolerances
    results: list                                    # CaseResult per case
    convergence: dict                                # group id -> Check
    expected: dict = field(default_factory=dict)     # (case_id or group, check) -> reason

    def _entries(self):
        """(id, check name, Check) for every check of every case and group."""
        for res in self.results:
            for name in CHECKS:
                yield res.case.case_id, name, res.checks[name]
        for group, check in self.convergence.items():
            yield group, 'convergence', check

    def status(self, key, name, check: Check) -> str:
        return 'expected' if check.failed and (key, name) in self.expected else check.status

    def unexpected_failures(self):
        return [(k, n, c) for k, n, c in self._entries() if self.status(k, n, c) == 'fail']

    def expected_now_passing(self):
        seen = {(k, n): c for k, n, c in self._entries()}
        return [(k, n) for (k, n) in self.expected if (k, n) in seen and not seen[k, n].failed]

    def stable_root_rate(self):
        """Of the cases with one stable reference root, how many the candidate returns as the operating point."""
        eligible = [r for r in self.results if r.n_stable_reference == 1]
        hits = sum(r.checks['operating_point'].status == 'pass' for r in eligible)
        return hits, len(eligible)

    def counts(self):
        table = defaultdict(Counter)
        for k, n, c in self._entries():
            table[n][self.status(k, n, c)] += 1
        return table

    def to_markdown(self, max_items: int = 15) -> str:
        failures = self.unexpected_failures()
        n_expected = sum(1 for k, n, c in self._entries() if self.status(k, n, c) == 'expected')
        hits, eligible = self.stable_root_rate()
        rate = f'{100 * hits / eligible:.1f}%' if eligible else 'n/a'
        verdict = 'PASS' if not failures else 'FAIL'
        lines = [f'## manywells-verify: {self.candidate} against ManyWells {self.model}', '',
                 f'**Verdict: {verdict}.** {len(failures)} unexpected failures, {n_expected} expected failures, '
                 f'{len(self.results)} cases, {len(self.convergence)} convergence groups.', '',
                 f'**Stable-root rate: {rate}** ({hits} of {eligible} cases with a stable reference root).', '',
                 '| Check | Pass | Fail | Expected fail | Indeterminate | n/a |',
                 '|---|--:|--:|--:|--:|--:|']
        counts = self.counts()
        for name in CHECKS + ('convergence',):
            c = counts[name]
            lines.append(f'| {TITLES[name]} | ' + ' | '.join(str(c[s]) for s in STATUSES) + ' |')
        worst = max((r for r in self.results if r.checks['residuals'].value is not None),
                    key=lambda r: r.checks['residuals'].value, default=None)
        t = self.tolerances
        lines += ['', (f'Worst scaled residual {worst.checks["residuals"].value:.1e} ({worst.case.case_id}). '
                       if worst else '') + f'Tolerances: tol_r = {t.tol_r:g}, tol_x = {t.tol_x:g}.']

        def section(title, items):
            if not items:
                return
            lines.extend(['', f'### {title} ({len(items)})'])
            lines.extend(f'- {item}' for item in items[:max_items])
            if len(items) > max_items:
                lines.append(f'- ... and {len(items) - max_items} more in the JSON report')

        section('Unexpected failures', [f'`{k}` {TITLES[n]}: {c.detail}' for k, n, c in failures])
        section('Expected failures now passing', [f'`{k}` {TITLES[n]}' for k, n in self.expected_now_passing()])
        section('Findings for review', [f'`{r.case.case_id}`: {f}' for r in self.results for f in r.findings])
        return '\n'.join(lines) + '\n'

    def to_json(self) -> dict:
        def check_dict(key, name, c):
            return {'status': self.status(key, name, c), 'detail': c.detail, 'value': c.value}
        hits, eligible = self.stable_root_rate()
        return {
            'candidate': self.candidate, 'model': self.model, 'tolerances': asdict(self.tolerances),
            'verdict': 'PASS' if not self.unexpected_failures() else 'FAIL',
            'stable_root_rate': {'hits': hits, 'eligible': eligible},
            'cases': [{'case_id': r.case.case_id, 'source': r.case.source, 'group': r.case.group,
                       'checks': {n: check_dict(r.case.case_id, n, r.checks[n]) for n in CHECKS},
                       'roots': [{'index': rr.index, 'label': rr.label, 'slope': rr.slope,
                                  'checks': {n: asdict(c) for n, c in rr.checks.items()}} for rr in r.roots],
                       'findings': r.findings} for r in self.results],
            'convergence': {g: check_dict(g, 'convergence', c) for g, c in self.convergence.items()},
        }


def verify(graph, cases: dict, candidate: dict, reference: dict, tol: Tolerances = Tolerances(),
           expected: dict | None = None, name: str = 'candidate') -> Report:
    """
    Check a candidate's roots on every case. Every case has a reference root set; a case with
    no reference rows has an empty one (no root). A case with no candidate rows means that the
    candidate reported no root.
    """
    results = [verify_case(graph, case, candidate.get(cid, []), reference.get(cid, []), tol)
               for cid, case in cases.items()]

    groups = defaultdict(list)
    for res in results:
        if res.case.group:
            op = [candidate[res.case.case_id][rr.index] for rr in res.roots
                  if rr.valid and candidate[res.case.case_id][rr.index].operating_point]
            groups[res.case.group].append((res.case, op[0] if op else None))
    convergence = {g: check_convergence(members, tol) for g, members in sorted(groups.items())}
    models = {c.model for c in cases.values()}
    return Report(name, ', '.join(sorted(models)), tol, results, convergence, expected or {})
