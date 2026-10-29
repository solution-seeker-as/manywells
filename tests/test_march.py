"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 16 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the initial-guess march (manywells.solvers.march).
"""

import casadi as ca
import numpy as np
import pytest

from manywells.discretization import DIM_X, build_system
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.slip import SlipModel
from manywells.solvers.march import MarchFailed, Marcher, _PointSolver


def point_solver(rows):
    x, p = ca.SX.sym('x'), ca.SX.sym('p')
    return _PointSolver('test', x, p, rows(x, p), [0.0], [np.inf])


def test_newton_solves_a_point():
    assert point_solver(lambda x, p: x**2 - p)(np.array([3.0]), np.array([4.0])) == pytest.approx([2.0])


def test_a_point_with_nan_rows_goes_to_ipopt():
    """CasADi's Newton returns the guess as solved where the rows are NaN there (issue #7); Ipopt solves the point."""
    assert point_solver(lambda x, p: ca.sqrt(x) - p)(np.array([-1.0]), np.array([1.0])) == pytest.approx([1.0])


def test_a_point_with_nan_rows_and_no_root_fails():
    with pytest.raises(MarchFailed):
        point_solver(lambda x, p: ca.sqrt(x) + p)(np.array([-1.0]), np.array([1.0]))


@pytest.fixture(scope='module')
def default_well():
    # The default well with the fitted classifier, develop's default when issue #7 was found
    system = build_system(WellProperties(slip=SlipModel(regime_model='fitted')))
    bc = BoundaryConditions(u=0.5)
    return Marcher(system), system.params(bc), bc


def test_a_march_that_runs_out_of_pressure_fails(default_well):
    """From p_0 = 65 bar, Newton stepped to p < 0 at point 42, and the march returned that state (issue #7)."""
    march, params, _ = default_well
    with pytest.raises(MarchFailed, match='point 42'):
        march(params, 65.0)


def test_a_march_reaches_the_wellhead(default_well):
    march, params, bc = default_well
    x = march(params, bc.p_s + 0.85 * (bc.p_r - bc.p_s)).reshape(-1, DIM_X)
    assert np.all(np.isfinite(x))
    assert np.all((bc.p_s <= x[:, 0]) & (x[:, 0] <= x[0, 0]))
    assert np.all(np.diff(x[:, 0]) < 0)
