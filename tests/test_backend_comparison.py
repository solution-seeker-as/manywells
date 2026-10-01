"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Rust core against the CasADi backend in every configuration (plans/manywells-v2-plan.md, Step 9;
specs/features/015-rust-develop-model.md). develop's model has no reference root sets, so the backends are checked
against each other, strongest first:

- the rows at the same state, on the configuration matrix (backend_cases.matrix): the same IDs, and the same values
  to rounding;
- the core's march, which solves every row but the choke row, zeroes the CasADi rows;
- the root sets on the comparison set (slow): every CasADi root found by the core with its label, and every root
  only the core finds a root of the CasADi rows (recorded, not failed); where the slip law has several void
  fractions, the rows only.
"""

import multiprocessing
import os

import numpy as np
import pytest

from manywells.discretization import build_system
from manywells.solvers.rust import RustRootFinder

from .backend_cases import (COMPARISON, FEATURES, OVERLAYS, ROOT_ROW, ROW_REL, casadi_rows, compare_case,
                            comparison_set, features, matrix, report, row_difference, trial_state)

# Step 9 ports develop's model one feature spec at a time: the features whose options the core refuses so far. A well
# that uses one of them must be refused, and once a feature is ported its number must go from here.
NOT_YET_PORTED = {'011'}

MATRIX = matrix()


def refused(wp) -> bool:
    if features(wp) & NOT_YET_PORTED:
        with pytest.raises(ValueError, match='Rust core cannot solve'):
            RustRootFinder(wp)
        return True
    return False


@pytest.mark.parametrize('configuration', MATRIX, ids=lambda c: c.name)
def test_rows_agree(configuration):
    """The core's rows are the CasADi system's at the same state, with the same IDs in the same order."""
    wp, bc = configuration.inputs()
    if refused(wp):
        return
    X = trial_state(wp, bc)
    ids, rust = RustRootFinder(wp).rows(bc, X)
    casadi_ids, casadi = casadi_rows(build_system(wp), bc, X)
    assert list(ids) == list(casadi_ids)
    rel, k, eq_id = row_difference(ids, rust, casadi)
    assert rel <= ROW_REL, f'{eq_id} (row {k}): rust {rust[k]!r}, casadi {casadi[k]!r}'


@pytest.mark.parametrize('configuration', MATRIX, ids=lambda c: c.name)
def test_the_cores_march_zeroes_the_casadi_rows(configuration):
    """The core's march from p_0 solves every row of the system but the choke row; the CasADi rows agree."""
    wp, bc = configuration.inputs()
    if refused(wp):
        return
    finder, system = RustRootFinder(wp), build_system(wp)
    for fraction in (0.7, 0.85, 0.5, 0.95, 0.3):
        x, failed, below = finder.march(bc, bc.p_s + fraction * (bc.p_r - bc.p_s))
        if failed or below:
            continue
        ids, rows = casadi_rows(system, bc, x)
        rows = rows[np.array(ids) != 'CHK-1']
        assert np.max(np.abs(rows)) < ROOT_ROW, f'p_0 at {fraction} of the drawdown: {np.max(np.abs(rows))}'
        return
    pytest.fail('no march reached the wellhead')


def test_the_matrix_switches_every_option():
    """Each feature's options are on in some configuration of the matrix, and every overlay is used, so a ported
    option is compared on rows; a new option (Step 10) must be added to the matrix."""
    used = set().union(*(features(c.inputs()[0]) for c in MATRIX))
    assert used == set(FEATURES) - {'005'}  # 005 is the fluid's inputs, which every well uses
    overlays = {o for c in MATRIX for o in c.overlays} | {o for os in COMPARISON.values() for o in os if o}
    assert overlays == set(OVERLAYS)
    assert any(c.inputs()[0].fluid.wlr > 0 for c in MATRIX)
    assert any(c.inputs()[1].T_lg not in (None, c.inputs()[1].T_r) for c in MATRIX)


@pytest.fixture(scope='module')
def comparisons():
    cases = [c for c in comparison_set() if not features(c.inputs()[0]) & NOT_YET_PORTED]
    with multiprocessing.Pool(min(len(cases), os.cpu_count() or 1)) as pool:
        return pool.map(compare_case, cases, chunksize=1)


@pytest.mark.slow
def test_the_backends_agree_on_the_comparison_set(comparisons):
    report(comparisons)
    failed = [c for c in comparisons if not c.ok]
    assert not failed, '\n'.join(
        f'{c.case}: missed {c.missed}, labels differ at {c.labels}, core-only {c.core_only}, rows {c.row_rel:.1e} '
        f'({c.row_at}){" " + c.error if c.error else ""}' for c in failed)
    with_roots = sum(bool(c.casadi) for c in comparisons)
    assert with_roots >= 0.8 * len(comparisons), f'only {with_roots} of {len(comparisons)} cases have a CasADi root'
