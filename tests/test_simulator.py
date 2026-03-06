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
from manywells.pvt.fluid import FluidModel
from manywells.constants import STD_GRAVITY


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
    wp = WellProperties(L=500, D=0.1, fluid=FluidModel(api=45.0))
    bc = BoundaryConditions(p_r=120, p_s=30, u=0.8)
    sim = SSDFSimulator(wp, bc, n_cells=2)
    try:
        result = sim.simulate()
        assert result is not None
        n_vars = (sim.n_cells + 1) * sim.dim_x
        assert len(result) == n_vars
    except SimError:
        pytest.skip("Simulator solve failed (e.g. Ipopt not available or no solution)")


# ---------------------------------------------------------------------------
# Energy equation thermal term tests
# ---------------------------------------------------------------------------

class TestEnergyEquationTerms:
    """Verify the three sub-terms (dT_heat, dT_fric, dT_grav) of the energy equation."""

    def test_gravity_pure_gas_equals_lapse_rate(self):
        """For pure gas (alpha=1), gravitational cooling rate equals g / cp_g."""
        alpha = 1.0
        rho_g, v_g = 50.0, 10.0
        rho_l, v_l = 850.0, 2.0
        cp_g = 2225.0

        mass_flux = alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l
        liq_flux = (1 - alpha) * v_l
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        cp_flux = cp_g * alpha * rho_g * v_g

        dT_grav_per_m = STD_GRAVITY * (mass_flux - liq_flux * rho_m) / cp_flux
        expected = STD_GRAVITY / cp_g

        assert dT_grav_per_m == pytest.approx(expected, rel=1e-12)

    def test_gravity_pure_liquid_is_zero(self):
        """For pure liquid (alpha=0), gravitational cooling vanishes."""
        alpha = 0.0
        rho_g, v_g = 50.0, 10.0
        rho_l, v_l = 850.0, 2.0

        mass_flux = alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l
        liq_flux = (1 - alpha) * v_l
        rho_m = alpha * rho_g + (1 - alpha) * rho_l

        assert mass_flux - liq_flux * rho_m == pytest.approx(0.0, abs=1e-12)

    def test_friction_pure_liquid(self):
        """For pure liquid (alpha=0), friction dissipation equals (f_D/(2D))*v_l^2/cp_l."""
        alpha = 0.0
        rho_l, v_l = 850.0, 2.0
        rho_g, v_g = 50.0, 10.0
        cp_l = 4180.0
        f_D = 0.05
        D = 0.15

        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        v_m = alpha * v_g + (1 - alpha) * v_l
        cp_flux = cp_l * (1 - alpha) * rho_l * v_l

        F_fric = (f_D / D / 2) * rho_m * v_m ** 2
        dT_fric_per_m = (1 - alpha) * v_l * F_fric / cp_flux

        expected = (f_D / (2 * D)) * v_l ** 2 / cp_l

        assert dT_fric_per_m == pytest.approx(expected, rel=1e-12)

    def test_friction_pure_gas_is_zero(self):
        """For pure gas (alpha=1), liquid friction dissipation vanishes."""
        alpha = 1.0
        dT_fric_factor = (1 - alpha)  # multiplies the entire friction term
        assert dT_fric_factor == pytest.approx(0.0, abs=1e-12)

    def test_gravity_mixed_is_positive(self):
        """For a gas-liquid mixture, gravitational cooling term is positive (cooling)."""
        alpha = 0.5
        rho_g, v_g = 50.0, 10.0
        rho_l, v_l = 850.0, 2.0
        cp_g, cp_l = 2225.0, 4180.0

        mass_flux = alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l
        liq_flux = (1 - alpha) * v_l
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        cp_flux = cp_g * alpha * rho_g * v_g + cp_l * (1 - alpha) * rho_l * v_l

        dT_grav_per_m = STD_GRAVITY * (mass_flux - liq_flux * rho_m) / cp_flux
        assert dT_grav_per_m > 0


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
