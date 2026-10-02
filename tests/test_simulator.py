"""Tests for manywells.simulator."""

import dataclasses

import pytest
import numpy as np

from manywells.geometry import WellGeometry
from manywells.simulator import (
    NoOperatingPoint,
    SimError,
    WellProperties,
    BoundaryConditions,
    SSDFSimulator,
)
from manywells.choke import BernoulliChokeModel
from manywells.discretization import DIM_X
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.fluid import FluidModel
from manywells.solution import Root, RootSet


def test_well_properties_defaults():
    """WellProperties has expected defaults via its geometry."""
    wp = WellProperties()
    geo = wp.geometry
    assert geo.L == 2000
    assert geo.D == pytest.approx(0.1554)
    assert geo.A == pytest.approx(np.pi * (geo.D / 2) ** 2)


def test_well_properties_choke_default():
    """The default choke is set by WellProperties itself, using geometry.A, not by the simulator."""
    wp = WellProperties()
    assert isinstance(wp.choke, BernoulliChokeModel)
    assert wp.choke.K_c == pytest.approx(0.1 * wp.geometry.A)


def test_inputs_are_frozen():
    """The simulator never writes to its inputs, and a typo in a field name fails (plans/improvements.md, 1.3)."""
    wp, bc = WellProperties(), BoundaryConditions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        wp.choke = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        wp.L = 3000.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        bc.u = 0.5
    assert dataclasses.replace(bc, u=0.5).u == 0.5


def test_well_properties_rejects_wrong_component():
    with pytest.raises(ValueError, match="friction must be a FrictionModel"):
        WellProperties(friction=0.03)


def test_boundary_conditions_defaults():
    """BoundaryConditions has expected defaults."""
    bc = BoundaryConditions()
    assert bc.p_r == 170
    assert bc.p_s == 20
    assert bc.u == 1.0
    assert 0 <= bc.u <= 1


def test_boundary_conditions_invalid_pressure():
    """BoundaryConditions rejects non-positive pressures."""
    with pytest.raises(ValueError, match="Reservoir pressure"):
        BoundaryConditions(p_r=0)
    with pytest.raises(ValueError, match="Separator pressure"):
        BoundaryConditions(p_s=-1)


def test_boundary_conditions_invalid_choke():
    """BoundaryConditions requires u in [0, 1]."""
    with pytest.raises(ValueError, match="Choke opening"):
        BoundaryConditions(u=-0.1)
    with pytest.raises(ValueError, match="Choke opening"):
        BoundaryConditions(u=1.5)


def test_boundary_conditions_negative_lift_gas():
    """BoundaryConditions rejects negative lift gas rate."""
    with pytest.raises(ValueError, match="Gas lift"):
        BoundaryConditions(w_lg=-1.0)


def test_simulator_construction():
    """SSDFSimulator builds the well's system once: 7 unknowns and 7 rows per grid point."""
    geo = WellGeometry.vertical(100, 5, D=0.1)
    sim = SSDFSimulator(WellProperties(geometry=geo))
    assert sim.n_cells == 5
    assert sim.dim_x == 7
    assert sim.variable_names == ["p", "v_g", "v_l", "alpha", "rho_g", "rho_l", "T"]
    assert sim.system.n_x == 7 * 6
    assert len(sim.system.row_ids) == sim.system.n_x
    assert sim.system.residual.size1_out(0) == sim.system.n_x


def test_row_order():
    """Rows at each point in the order of DISC-11: the bottomhole's six, seven per point, and CHK-1 at the top."""
    sim = SSDFSimulator(WellProperties(geometry=WellGeometry.vertical(100, 2, D=0.1)))
    ids = sim.system.row_ids
    closures = ('SLIP-1', 'PVT-GAS-11', 'PVT-MIX-6')  # the default fluid: real gas by DAK, black oil
    assert ids[:6] == ('INF-6', 'INF-7', 'THM-5') + closures
    assert ids[6:13] == ('DISC-7', 'DISC-8', 'DISC-9', 'DISC-10') + closures
    assert ids[13:] == ('DISC-7', 'DISC-8', 'DISC-9', 'DISC-10', 'CHK-1') + closures


def test_deprecated_two_argument_constructor():
    """SSDFSimulator(wp, bc) with simulate() still works, with a warning, and returns the state as a flat list."""
    geo = WellGeometry.vertical(2000, 10)
    wp = WellProperties(geometry=geo, fluid=FluidModel(rho_o=density_from_api(35.0), oil_model='dead_oil'))
    bc = BoundaryConditions(p_r=200, p_s=20, u=0.8)
    with pytest.warns(DeprecationWarning):
        sim = SSDFSimulator(wp, bc)
    x = sim.simulate()
    assert isinstance(x, list) and len(x) == DIM_X * (geo.n_cells + 1)
    assert np.allclose(x, SSDFSimulator(wp).simulate(bc).x)


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_simulator_solve_small():
    """Simulator solve runs with minimal grid."""
    geo = WellGeometry.vertical(500, 2, D=0.1)
    wp = WellProperties(geometry=geo, fluid=FluidModel(rho_o=density_from_api(45.0), oil_model='dead_oil'))
    op = SSDFSimulator(wp).simulate(BoundaryConditions(p_r=120, p_s=30, u=0.8))
    assert isinstance(op, Root)
    assert op.label == 'stable'
    assert len(op.x) == (geo.n_cells + 1) * DIM_X


@pytest.mark.slow
def test_mass_conservation():
    """Total mass flux (gas + liquid) is constant across all cells with black oil, where gas dissolves."""
    fl = FluidModel(
        rho_o=density_from_api(35),
        rho_g=gas_density_from_sg(0.65),
        wlr=0.0,
        p_bubble=250.0,
    )
    geo = WellGeometry.vertical(2000, 5, D=0.1554)
    wp = WellProperties(geometry=geo, fluid=fl)
    X = SSDFSimulator(wp).simulate(BoundaryConditions(p_r=200, p_s=30, u=0.8)).state

    A = geo.A
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    w_m = A * alpha * rho_g * v_g + A * (1 - alpha) * rho_l * v_l
    np.testing.assert_allclose(w_m, w_m[0], rtol=1e-6)


@pytest.mark.slow
def test_simulator_solve_l_shaped():
    """Simulator converges for an L-shaped well (2000 m vertical, gradual build, 1000 m horizontal)."""
    R = 250.0
    md_survey = [0.0, 2000.0]
    tvd_survey = [0.0, 2000.0]
    for t in np.linspace(0, np.pi / 2, 6)[1:]:
        md_survey.append(2000.0 + R * t)
        tvd_survey.append(2000.0 + R * np.sin(t))
    md_survey.append(md_survey[-1] + 1000.0)
    tvd_survey.append(tvd_survey[-1])

    geo = WellGeometry.from_survey(md_survey=md_survey, tvd_survey=tvd_survey, n_cells=50)
    wp = WellProperties(geometry=geo, fluid=FluidModel(rho_o=density_from_api(35.0), oil_model='dead_oil'))
    sim = SSDFSimulator(wp)
    op = sim.simulate(BoundaryConditions(p_r=200, p_s=20, u=0.8))
    df = sim.solution_as_df(op)
    assert df["p"].iloc[0] > df["p"].iloc[-1]  # pressure decreases bottom to top
    assert all(df["alpha"] >= 0) and all(df["alpha"] <= 1)
    assert "md" in df.columns and "tvd" in df.columns


@pytest.mark.slow
def test_simulator_solve_with_gas_lift():
    """The lift gas is carried in the gas phase: w_g = f_g / (1 - f_g) * w_l + w_lg for dead oil."""
    geo = WellGeometry.vertical(2000, 20, D=0.1554)
    fl = FluidModel(rho_o=density_from_api(35.0), oil_model='dead_oil')
    wp = WellProperties(geometry=geo, fluid=fl)
    w_lg = 1.0
    op = SSDFSimulator(wp).simulate(BoundaryConditions(p_r=200, p_s=20, u=0.8, w_lg=w_lg, T_lg=300.0))

    p, v_g, v_l, alpha, rho_g, rho_l, T = op.state[-1]  # wellhead state
    w_g = geo.A * alpha * rho_g * v_g
    w_l = geo.A * (1 - alpha) * rho_l * v_l
    assert w_g - fl.f_g / (1 - fl.f_g) * w_l == pytest.approx(w_lg, rel=1e-6)
    assert op.w_res == pytest.approx(w_l, rel=1e-6)
    assert op.w_g_res == pytest.approx(fl.f_g / (1 - fl.f_g) * w_l, rel=1e-6)


@pytest.mark.slow
def test_root_set_of_a_two_root_well():
    """A well whose static liquid column cannot reach the separator has a stable and an unstable root (SOL-7)."""
    geo = WellGeometry.vertical(2000, 50)
    fl = FluidModel(rho_o=900.0, gor=50.0, oil_model='dead_oil')
    wp = WellProperties(geometry=geo, fluid=fl)
    bc = BoundaryConditions(p_r=170, p_s=20, u=0.8)
    assert bc.p_r - bc.p_s < fl.rho_l * 9.80665 * geo.L / 1e5
    rs = SSDFSimulator(wp).root_set(bc)
    assert [r.label for r in rs.roots] == ['stable', 'unstable']  # sorted by p_0: the unstable root is the trickle root
    assert rs.operating_point is rs.roots[0]
    assert not rs.several_stable
    assert rs.search and all(a.outcome for a in rs.search)


@pytest.mark.slow
def test_no_operating_point():
    """Below the fold the well cannot flow: simulate raises NoOperatingPoint, a SimError, with the root set."""
    geo = WellGeometry.vertical(2000, 20)
    wp = WellProperties(geometry=geo, fluid=FluidModel(rho_o=900.0, gor=50.0, oil_model='dead_oil'))
    bc = BoundaryConditions(p_r=60, p_s=20, u=0.8)
    with pytest.raises(NoOperatingPoint) as e:
        SSDFSimulator(wp).simulate(bc)
    assert isinstance(e.value, SimError)
    assert isinstance(e.value.root_set, RootSet)
    assert e.value.root_set.operating_point is None


def test_simulator_solution_as_df_shape():
    """solution_as_df returns DataFrame with expected columns including md/tvd."""
    geo = WellGeometry.vertical(100, 3, D=0.1)
    sim = SSDFSimulator(WellProperties(geometry=geo))
    # Build a physical dummy state [p, v_g, v_l, alpha, rho_g, rho_l, T] per cell (rho_l > rho_g for flow regime)
    n_nodes = sim.n_cells + 1
    dummy_state = []
    for _ in range(n_nodes):
        dummy_state.extend([50.0, 5.0, 2.0, 0.3, 50.0, 700.0, 300.0])
    df = sim.solution_as_df(dummy_state)
    assert "md" in df.columns
    assert "tvd" in df.columns
    for name in sim.variable_names:
        assert name in df.columns
    assert len(df) == sim.n_cells + 1
    assert set(df['flow-regime']) <= {'annular', 'slug-churn', 'bubbly'}
