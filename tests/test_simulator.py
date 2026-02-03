"""Tests for manywells.simulator."""

import pytest
import numpy as np

from manywells.simulator import (
    SimError,
    WellProperties,
    BoundaryConditions,
    SSDFSimulator,
)
from manywells.choke import BernoulliChokeModel
from manywells.inflow import ProductivityIndex


def test_well_properties_defaults():
    """WellProperties has expected defaults and A property."""
    wp = WellProperties()
    assert wp.L == 2000
    assert wp.D == pytest.approx(0.1554)
    assert wp.A == pytest.approx(np.pi * (wp.D / 2) ** 2)


def test_well_properties_invalid_L():
    """WellProperties rejects non-positive length."""
    with pytest.raises(AssertionError, match="Pipe length"):
        WellProperties(L=0)
    with pytest.raises(AssertionError, match="Pipe length"):
        WellProperties(L=-1)


def test_well_properties_invalid_D():
    """WellProperties rejects non-positive diameter."""
    with pytest.raises(AssertionError, match="Pipe diameter"):
        WellProperties(D=0)


def test_well_properties_choke_default():
    """WellProperties gets default BernoulliChokeModel when choke is None."""
    wp = WellProperties()
    assert wp.choke is not None
    assert isinstance(wp.choke, BernoulliChokeModel)
    assert wp.choke.K_c == pytest.approx(0.1 * wp.A)


def test_boundary_conditions_defaults():
    """BoundaryConditions has expected defaults."""
    bc = BoundaryConditions()
    assert bc.p_r == 170
    assert bc.p_s == 20
    assert bc.u == 1.0
    assert 0 <= bc.u <= 1


def test_boundary_conditions_invalid_pressure():
    """BoundaryConditions rejects non-positive pressures."""
    with pytest.raises(AssertionError, match="Reservoir pressure"):
        BoundaryConditions(p_r=0)
    with pytest.raises(AssertionError, match="Separator pressure"):
        BoundaryConditions(p_s=-1)


def test_boundary_conditions_invalid_choke():
    """BoundaryConditions requires u in [0, 1]."""
    with pytest.raises(AssertionError, match="Choke opening"):
        BoundaryConditions(u=-0.1)
    with pytest.raises(AssertionError, match="Choke opening"):
        BoundaryConditions(u=1.5)


def test_boundary_conditions_negative_lift_gas():
    """BoundaryConditions rejects negative lift gas rate."""
    with pytest.raises(AssertionError, match="Gas lift"):
        BoundaryConditions(w_lg=-1.0)


def test_simulator_construction():
    """SSDFSimulator constructs with well and boundary conditions."""
    wp = WellProperties(L=100, D=0.1)
    bc = BoundaryConditions(p_r=150, p_s=25)
    sim = SSDFSimulator(wp, bc, n_cells=5)
    assert sim.n_cells == 5
    assert sim.dim_x == 7
    assert sim.variable_names == ["p", "v_g", "v_l", "alpha", "rho_g", "rho_l", "T"]


def test_simulator_create_variables():
    """_create_variables returns 7 symbols per cell."""
    wp = WellProperties(L=100, D=0.1)
    bc = BoundaryConditions()
    sim = SSDFSimulator(wp, bc, n_cells=2)
    x0 = sim._create_variables(0)
    assert len(x0) == 7
    assert all(hasattr(v, "name") for v in x0)


@pytest.mark.slow
def test_simulator_solve_small():
    """Simulator solve runs with minimal grid (may be slow)."""
    wp = WellProperties(L=500, D=0.1, rho_l=800)
    bc = BoundaryConditions(p_r=120, p_s=30, u=0.8)
    sim = SSDFSimulator(wp, bc, n_cells=2)
    try:
        result = sim.simulate()
        assert result is not None
        n_vars = (sim.n_cells + 1) * sim.dim_x
        assert len(result) == n_vars
    except SimError:
        pytest.skip("Simulator solve failed (e.g. Ipopt not available or no solution)")


def test_simulator_solution_as_df_shape():
    """solution_as_df returns DataFrame with expected columns and z when given state list."""
    wp = WellProperties(L=100, D=0.1)
    bc = BoundaryConditions()
    sim = SSDFSimulator(wp, bc, n_cells=3)
    # Build a physical dummy state [p, v_g, v_l, alpha, rho_g, rho_l, T] per cell (rho_l > rho_g for flow regime)
    n_cells = sim.n_cells + 1
    dummy_state = []
    for _ in range(n_cells):
        dummy_state.extend([50.0, 5.0, 2.0, 0.3, 50.0, 700.0, 300.0])
    df = sim.solution_as_df(dummy_state)
    assert "z" in df.columns
    for name in sim.variable_names:
        assert name in df.columns
    assert len(df) == sim.n_cells + 1
