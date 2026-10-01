"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the root search (manywells.solvers.roots) and the operating point (manywells.solution).

The library cannot import the verifier, so it keeps copies of the verifier's state distance and thresholds
(specs/architecture.md, Solvers); the tests import both and check that the copies agree.
"""

import pathlib
import re

import numpy as np
import pytest

from manywells_verify import checks
from manywells_verify.cases import Case
from manywells.simulator import BoundaryConditions
from manywells.solution import Root, RootSet, select_operating_point
from manywells.solvers import roots

BC = BoundaryConditions(p_r=250.0, p_s=30.0, T_r=370.0, T_s=280.0)


def test_thresholds_equal_the_verifiers():
    assert roots.TOL_X == checks.Tolerances().tol_x
    assert roots.V_FLOOR == checks.V_FLOOR
    # label_min is applied when the reference is built, so it lives in the build script (specs/verification.md)
    build = (pathlib.Path(__file__).parents[1] / 'verification' / 'build' / 'label_cases.py').read_text()
    assert roots.LABEL_MIN == float(re.search(r'^LABEL_MIN = (\S+)', build, re.M).group(1))


def test_state_distance_equals_the_verifiers():
    rng = np.random.default_rng(1)
    n_cells = 10
    case = Case('c', {'p_r': BC.p_r, 'p_s': BC.p_s, 'T_r': BC.T_r, 'T_s': BC.T_s}, n_cells)
    for _ in range(20):
        ref = rng.uniform(0.01, 2.0, 7 * (n_cells + 1)) * np.tile([100, 5, 2, 0.5, 50, 800, 330], n_cells + 1)
        x = ref * (1 + rng.normal(0, 1e-3, ref.size))
        assert roots.state_distance(x, ref, BC) == checks.state_distance(x, ref, case)


def test_label_of():
    assert roots.label_of(-5.0) == 'stable'
    assert roots.label_of(5.0) == 'unstable'
    assert roots.label_of(roots.LABEL_MIN) == 'indeterminate'
    assert roots.label_of(-roots.LABEL_MIN / 2) == 'indeterminate'


def test_admissible():
    """SOL-1: pressures in [p_s, p_r], alpha in [0, 1], positive velocities and densities."""
    good = np.tile([100.0, 5.0, 2.0, 0.5, 50.0, 800.0, 330.0], 3)
    assert roots.admissible(good, BC)
    for k, value in ((0, BC.p_s - 1e-6), (0, BC.p_r + 1e-6), (3, 1.0 + 1e-9), (2, 0.0), (4, -1.0), (6 + 1, np.nan)):
        bad = good.copy()
        bad[k] = value
        assert not roots.admissible(bad, BC)


def root(p_0, label):
    return Root(x=[p_0] + [1.0] * 6, label=label, slope=-1.0 if label == 'stable' else 1.0, choked=False,
                flow_regime=('bubbly',), w_res=1.0, w_g_res=0.1)


def test_operating_point_is_the_stable_root():
    """SOL-4"""
    rs = RootSet.of([root(180.0, 'unstable'), root(120.0, 'stable')])
    assert [r.p_0 for r in rs.roots] == [120.0, 180.0]
    assert rs.operating_point.p_0 == 120.0 and not rs.several_stable


def test_no_stable_root_no_operating_point():
    """SOL-5"""
    assert select_operating_point([root(180.0, 'unstable')]) == (None, False)
    assert select_operating_point([]) == (None, False)


def test_several_stable_roots():
    """SOL-6: the stable root with the lowest p_0, and the case is flagged."""
    op, several = select_operating_point([root(151.4, 'stable'), root(150.5, 'unstable'), root(149.5, 'stable')])
    assert op.p_0 == 149.5 and several


def test_starts():
    starts = roots.RootFinder.starts(BC)
    assert starts[0] == ('default', BC.p_r - 0.05 * (BC.p_r - BC.p_s))
    fractions = [(p_0 - BC.p_s) / (BC.p_r - BC.p_s) for _, p_0 in starts[1:]]
    assert fractions == pytest.approx(roots.START_FRACTIONS)
    assert len({round(p, 9) for _, p in starts}) == len(starts)  # no start twice
