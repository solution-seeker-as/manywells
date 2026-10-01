"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Rust core against the CasADi backend on more wells than the tests take (tests/backend_cases.py), or the
time each backend takes, for the measurements of specs/features/015-rust-develop-model.md. Run from the project root:

    uv run python -m scripts.verification.compare_backends comparison.json --per-configuration 20
    uv run python -m scripts.verification.compare_backends timing.json --set verifier --backends rust --processes 1
    uv run python -m scripts.verification.compare_backends timing.json --set verifier --overlay "frictional heating"

--set comparison takes the comparison set: sampled wells, each with one overlay that switches an option on or off
(backend_cases.COMPARISON). --set verifier takes the verifier's case set in the v1.0.0 configuration
(specs/verification.md), with --overlay applied to every case. With both backends, each case's root sets are
compared by the tests' pass rule; with one, the backend is timed only. The tables are printed as Markdown, and every
case's comparison is written to the JSON file. Timings are build and search per case on one process each; use
--processes 1 for timings that go into a feature spec.
"""

import argparse
import functools
import json
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from manywells.simulator import SSDFSimulator
from tests.backend_cases import OVERLAYS, Comparison, compare_case, comparison_set

DATA = Path('verification/data')


@dataclass(frozen=True)
class VerifierCase:
    """A case of the verifier's case set, in the v1.0.0 configuration, with an overlay (or none)."""
    case_id: str
    overlay: str = ''

    @property
    def name(self) -> str:
        return self.case_id + (f'+{self.overlay}' if self.overlay else '')

    def inputs(self):
        from scripts.verification.develop_candidate import well_of
        wp, bc = well_of(_cases()[self.case_id])
        return OVERLAYS[self.overlay](wp, bc) if self.overlay else (wp, bc)

    @property
    def group(self) -> str:
        return 'verifier' + (f'+{self.overlay}' if self.overlay else '')


@functools.cache
def _cases():
    from manywells_verify.cases import read_cases
    return read_cases(DATA / 'cases.parquet')


def time_case(case, backend) -> Comparison:
    """One backend's roots for a case, timed (build and search)."""
    c = Comparison(case=case.name)
    try:
        wp, bc = case.inputs()
        t0 = time.perf_counter()
        sim = SSDFSimulator(wp, backend=backend)
        rs = sim.root_set(bc)
        c.seconds = {backend: time.perf_counter() - t0}
        setattr(c, backend, [(r.p_0, r.label) for r in rs.roots])
        if backend == 'rust':
            c.marches, c.counts = rs.search[0].iterations, dict(sim._roots.counts)
    except Exception as e:
        c.error = f'{type(e).__name__}: {e}'
    return c


def run(case, backends):
    c = compare_case(case) if len(backends) == 2 else time_case(case, backends[0])
    return c, case.group


def agreement_table(results) -> str:
    rows = ['| Configuration | Cases | CasADi roots | Core roots | Missed | Labels differ | Core-only (zero the rows) '
            '| Several α | Other branch | Largest row difference | Errors |', '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for conf in dict.fromkeys(k for _, k in results):
        cs = [c for c, k in results if k == conf]
        core_only = [z for c in cs for _, z in c.core_only]
        rows.append(f'| {conf} | {len(cs)} | {sum(len(c.casadi) for c in cs)} | {sum(len(c.rust) for c in cs)} '
                    f'| {sum(len(c.missed) for c in cs)} | {sum(len(c.labels) for c in cs)} '
                    f'| {len(core_only)} ({sum(core_only)}) | {sum(c.several_alpha for c in cs)} '
                    f'| {sum(bool(c.other_branch) for c in cs)} '
                    f'| {max((c.row_rel for c in cs), default=0):.1e} | {sum(bool(c.error) for c in cs)} |')
    return '\n'.join(rows)


def timing_table(results, backends) -> str:
    cs = [c for c, _ in results if not c.error]
    t = {b: np.array([c.seconds[b] for c in cs]) for b in backends}
    rows = ['| Backend | Cases | Median (s) | p90 (s) | Total (s) | Median marches |', '|---|--:|--:|--:|--:|--:|']
    for b in backends:
        marches = np.median([c.marches for c in cs]) if b == 'rust' and cs else float('nan')
        rows.append(f'| {b} | {len(cs)} | {np.median(t[b]):.4f} | {np.percentile(t[b], 90):.4f} | {t[b].sum():.2f} '
                    f'| {marches:.0f} |')
    if len(backends) == 2:
        rows.append(f'\nCasADi / core: {np.median(t["casadi"]) / np.median(t["rust"]):.1f}× at the median, '
                    f'{t["casadi"].sum() / t["rust"].sum():.1f}× in total; per case at least '
                    f'{np.min(t["casadi"] / t["rust"]):.1f}×')
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('out', type=Path, help='JSON file for every case\'s comparison')
    parser.add_argument('--set', choices=('comparison', 'verifier'), default='comparison')
    parser.add_argument('--per-configuration', type=int, default=20, help='wells per configuration (comparison)')
    parser.add_argument('--configurations', nargs='*', help='only these configurations of the comparison set')
    parser.add_argument('--overlay', default='', choices=('',) + tuple(OVERLAYS), help='overlay (verifier)')
    parser.add_argument('--backends', nargs='+', choices=('casadi', 'rust'), default=['casadi', 'rust'])
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    args = parser.parse_args()

    if args.set == 'comparison':
        cases = comparison_set(args.per_configuration, configurations=args.configurations)
    else:
        cases = [VerifierCase(cid, args.overlay) for cid in _cases()]
    backends = sorted(set(args.backends))
    t0 = time.perf_counter()
    with multiprocessing.Pool(args.processes) as pool:
        results = pool.map(functools.partial(run, backends=backends), cases, chunksize=1)
    print(f'{len(cases)} cases in {time.perf_counter() - t0:.1f} s on {args.processes} processes\n')
    if len(backends) == 2:
        print(agreement_table(results) + '\n')
        failed = [c for c, _ in results if not c.ok]
        print(f'{len(results) - len(failed)} of {len(results)} cases pass the rule of test_backend_comparison.py'
              + ''.join(f'\n  {c.case}: missed {c.missed}, labels {c.labels}, core-only {c.core_only}, '
                        f'rows {c.row_rel:.1e} {c.row_at} {c.error.splitlines()[0] if c.error else ""}' for c in failed)
              + ''.join(f'\n  {c.case}: other branch at {c.other_branch}' for c, _ in results if c.other_branch)
              + '\n')
    print(timing_table(results, backends))
    args.out.write_text(json.dumps([{**asdict(c), 'configuration': k} for c, k in results], indent=1, default=str))


if __name__ == '__main__':
    main()
