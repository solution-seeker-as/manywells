"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the case and root files, the report and the command line, on the v1.0.0 fixtures.
A "v1-like" candidate reports one unlabelled root per case, as v1.0.0 does; for well 977 it
reports the trickle root, which v1's default guess reaches.
"""

import json

import numpy as np
import pytest

from manywells_verify.cases import Root, read_cases, read_roots, write_cases, write_roots
from manywells_verify.cli import main
from manywells_verify.report import read_expected_failures, verify


def v1_like(fixtures):
    candidate = {}
    for cid, (case, roots) in fixtures.items():
        if roots:
            pick = roots[-1] if cid == 'sol1-977' else roots[0]   # trickle root for 977
            candidate[cid] = [Root(pick.x, operating_point=True)]
    return candidate


@pytest.fixture
def data_dir(tmp_path, fixtures):
    write_cases([case for case, _ in fixtures.values()], tmp_path / 'cases.parquet')
    write_roots({cid: roots for cid, (_, roots) in fixtures.items()}, tmp_path / 'reference_roots.parquet')
    write_roots(v1_like(fixtures), tmp_path / 'v1.parquet')
    return tmp_path


def test_files_round_trip(data_dir, fixtures):
    cases = read_cases(data_dir / 'cases.parquet')
    reference = read_roots(data_dir / 'reference_roots.parquet')
    assert set(cases) == set(fixtures)
    for cid, (case, roots) in fixtures.items():
        assert cases[cid].params == pytest.approx(case.params)
        assert cases[cid].variant == case.variant and cases[cid].group == case.group
        assert [r.label for r in reference.get(cid, [])] == [r.label for r in roots]
        for a, b in zip(reference.get(cid, []), roots):
            np.testing.assert_array_equal(a.x, b.x)
    assert 'sol1-977-past-fold' not in reference         # a case with no root has no rows


def test_report_counts_and_rate(fixtures):
    report = verify({c.case_id: c for c, _ in fixtures.values()}, v1_like(fixtures),
                    {cid: roots for cid, (_, roots) in fixtures.items()}, name='v1-like')
    failures = report.unexpected_failures()
    assert [(k, n) for k, n, _ in failures] == [('sol1-977', 'operating_point')]
    assert report.stable_root_rate() == (4, 5)
    assert report.convergence['conv-one-root'].status == 'pass'
    text = report.to_markdown()
    assert '**Verdict: FAIL.**' in text and '**Stable-root rate: 80.0%**' in text
    assert len(text.splitlines()) < 40
    json.dumps(report.to_json(), default=float)


def test_expected_failure_gives_pass(fixtures, tmp_path):
    path = tmp_path / 'expected.csv'
    path.write_text('case_id,check,reason\nsol1-977,operating_point,v1 returns the trickle root\n'
                    'sol1-one-root,invariants,stale entry\n')
    expected = read_expected_failures(path)
    report = verify({c.case_id: c for c, _ in fixtures.values()}, v1_like(fixtures),
                    {cid: roots for cid, (_, roots) in fixtures.items()}, expected=expected)
    assert report.unexpected_failures() == []
    assert report.expected_now_passing() == [('sol1-one-root', 'invariants')]
    assert '**Verdict: PASS.**' in report.to_markdown()


def test_cli_exit_status(data_dir, tmp_path, capsys):
    assert main([str(data_dir / 'v1.parquet'), '--data', str(data_dir), '--json', str(tmp_path / 'r.json')]) == 1
    assert 'Unexpected failures (1)' in capsys.readouterr().out
    assert json.loads((tmp_path / 'r.json').read_text())['verdict'] == 'FAIL'

    expected = tmp_path / 'expected.csv'
    expected.write_text('case_id,check,reason\nsol1-977,operating_point,v1 returns the trickle root\n')
    assert main([str(data_dir / 'v1.parquet'), '--data', str(data_dir), '--expected-failures', str(expected)]) == 0
