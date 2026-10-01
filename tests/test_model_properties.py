"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Property and spot checks of develop's full model, which has no reference root sets (plans/manywells-v2-plan.md,
"New model versions", and Step 7 item 5). The wells exercise develop's options: deviated and L-shaped trajectories,
black oil with dissolved gas, real gas, friction from roughness, frictional heating and the gravity term, and
lift gas colder than the reservoir.

Checked at every root: the verifier's Invariants that do not assume dead oil (total mass rate constant instead of
each phase's); spot checks of the relations that need no closure (the inflow rows at the bottom, the choke row at
the top, the phase rates of the fluid model at every point, non-negative friction, heat flowing from the fluid to
colder surroundings); and the stability property of two-root wells. Convergence at first order on N, 2N and 4N.
"""

import dataclasses

import numpy as np
import pytest

from manywells.discretization import DIM_X, PointState
from manywells.geometry import WellGeometry
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties

pytestmark = pytest.mark.slow

TOL_RATE = 1e-6        # relative: mass rates
TOL_P = 1e-6           # bar: a pressure rise between neighbouring points
TOL_T = 1e-6           # K


def deviated(n_cells, tvd=2500.0, kickoff=0.3, theta_deg=45.0):
    """Vertical to the kickoff depth, then straight at theta to the bottomhole's true vertical depth."""
    k = kickoff * tvd
    md = k + (tvd - k) / np.cos(np.radians(theta_deg))
    return WellGeometry.from_survey([0.0, k, md], [0.0, k, tvd], n_cells=n_cells)


def l_shaped(n_cells, tvd=2200.0, horizontal=800.0, radius=300.0):
    """Vertical, a build of the given radius to horizontal, then a horizontal section."""
    md, z = [0.0, tvd - radius], [0.0, tvd - radius]
    for t in np.linspace(0, np.pi / 2, 8)[1:]:
        md.append(tvd - radius + radius * t)
        z.append(tvd - radius + radius * np.sin(t))
    md.append(md[-1] + horizontal)
    z.append(z[-1])
    return WellGeometry.from_survey(md, z, n_cells=n_cells)


BLACK_OIL = FluidModel(rho_o=density_from_api(35.0), rho_g=gas_density_from_sg(0.7), gor=120.0, wlr=0.3)
WELLS = {
    'deviated black oil': (lambda n: WellProperties(geometry=deviated(n), fluid=BLACK_OIL),
                           BoundaryConditions(p_r=260.0, p_s=25.0, T_r=370.0, T_s=280.0, u=0.6)),
    'L-shaped black oil': (lambda n: WellProperties(geometry=l_shaped(n), fluid=BLACK_OIL),
                           BoundaryConditions(p_r=240.0, p_s=20.0, T_r=365.0, T_s=280.0, u=0.7)),
    'deviated, cold lift gas': (lambda n: WellProperties(geometry=deviated(n, tvd=2000.0), fluid=dataclasses.replace(
                                    BLACK_OIL, gor=40.0, wlr=0.5)),
                                BoundaryConditions(p_r=200.0, p_s=20.0, T_r=360.0, T_s=280.0, u=0.8, w_lg=2.0, T_lg=300.0)),
    'vertical dead oil, real gas': (lambda n: WellProperties(geometry=WellGeometry.vertical(2000.0, n),
                                                             fluid=FluidModel(rho_o=880.0, gor=60.0, oil_model='dead_oil')),
                                    BoundaryConditions(p_r=210.0, p_s=20.0, T_r=360.0, T_s=280.0, u=0.5)),
}


@pytest.fixture(scope='module')
def solved():
    """Every well's root set at 100 cells, and its simulator."""
    out = {}
    for name, (make, bc) in WELLS.items():
        sim = SSDFSimulator(make(100))
        out[name] = (sim, bc, sim.root_set(bc))
    return out


def points(sim, x):
    return [PointState.of(row) for row in np.reshape(x, (-1, DIM_X))]


@pytest.mark.parametrize('name', WELLS)
def test_well_has_a_stable_operating_point(solved, name):
    sim, bc, rs = solved[name]
    assert rs.operating_point is not None, [a.outcome for a in rs.search]
    assert rs.operating_point.label == 'stable'


@pytest.mark.parametrize('name', WELLS)
def test_invariants(solved, name):
    """The verifier's Invariants, with the total mass rate constant instead of each phase's (with dissolved gas)."""
    sim, bc, rs = solved[name]
    A = sim.wp.geometry.A
    for root in rs.roots:
        X = root.state
        p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
        assert np.all((0 <= alpha) & (alpha <= 1))
        assert np.all(v_g > 0) and np.all(v_l > 0) and np.all(rho_g > 0) and np.all(rho_l > 0)
        w_m = A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
        assert np.max(np.abs(w_m - w_m[0])) <= TOL_RATE * w_m[0]
        assert np.max(np.diff(p)) <= TOL_P, 'pressure rises along the well'
        assert p[-1] > bc.p_s and p[0] < bc.p_r
        assert root.choked == (bc.p_s <= sim.wp.choke.cpr * p[-1])


@pytest.mark.parametrize('name', WELLS)
def test_spot_checks(solved, name):
    """Relations that need no closure: the inflow rows (INF-6, INF-7), the choke row (CHK-1), and at every point
    the phase rates of the fluid model (DISC-7, DISC-8) and non-negative friction (FRIC-1)."""
    sim, bc, rs = solved[name]
    wp = sim.wp
    A, D, fluid = wp.geometry.A, wp.geometry.D, wp.fluid
    for root in rs.roots:
        S = points(sim, root.x)
        w_res = float(wp.inflow.liquid_mass_flow_rate(S[0].p, bc.p_r))
        assert root.w_res == pytest.approx(w_res, rel=1e-12)
        for s in S:
            w_g, w_l = (float(v) for v in fluid.phase_rates(s.p, s.T, w_res, bc.w_lg))
            assert A * s.gas_flux == pytest.approx(w_g, rel=TOL_RATE)
            assert A * s.liquid_flux == pytest.approx(w_l, rel=TOL_RATE)
            assert float(wp.friction.pressure_gradient(s, fluid, D)) >= 0
        w_m = A * (S[-1].gas_flux + S[-1].liquid_flux)
        assert w_m == pytest.approx(float(wp.choke.mass_flow_rate(bc.u, bc.p_s, S[-1], A)), rel=TOL_RATE)


@pytest.mark.parametrize('name', [n for n in WELLS if WELLS[n][1].w_lg == 0])
def test_heat_flows_from_the_fluid_to_the_surroundings(solved, name):
    """Without lift gas the fluid enters at T_r, the ambient temperature at the bottomhole, and is warmer than its
    surroundings everywhere above: the heat loss carries heat outwards, against frictional heating and gravity."""
    sim, bc, rs = solved[name]
    geo = sim.wp.geometry
    T_a = np.array([sim.wp.thermal.ambient_temperature(f, bc.T_r, bc.T_s) for f in geo.tvd_frac])
    for root in rs.roots:
        T = root.state[:, 6]
        assert T[0] == pytest.approx(bc.T_r, abs=TOL_T)
        assert np.all(T[1:] - T_a[1:] > 0)


def test_cold_lift_gas_cools_the_inflow(solved):
    """THM-5: lift gas colder than the reservoir lowers the bottomhole temperature, but not below T_lg."""
    sim, bc, rs = solved['deviated, cold lift gas']
    T_0 = rs.operating_point.state[0, 6]
    assert bc.T_lg < T_0 < bc.T_r


def test_gas_comes_out_of_solution_up_the_well(solved):
    """BAL-10: with black oil the free-gas rate grows as the pressure falls towards the wellhead."""
    sim, bc, rs = solved['deviated black oil']
    X = rs.operating_point.state
    w_g = sim.wp.geometry.A * X[:, 3] * X[:, 4] * X[:, 1]
    assert np.all(np.diff(w_g) > 0)


def test_two_root_well_stability():
    """A two-root well has one stable and one unstable root, the unstable one at higher p_0 (plan, "New model
    versions"), here on develop's full model in a deviated well."""
    fl = FluidModel(rho_o=density_from_api(25.0), rho_g=gas_density_from_sg(0.7), gor=60.0, wlr=0.6)
    wp = WellProperties(geometry=deviated(60, tvd=2200.0), fluid=fl)
    bc = BoundaryConditions(p_r=225.0, p_s=25.0, T_r=360.0, T_s=280.0, u=0.6)
    assert bc.p_r - bc.p_s < fl.rho_l * 9.80665 * 2200.0 / 1e5  # the static column cannot reach the separator (SOL-7)
    rs = SSDFSimulator(wp).root_set(bc)
    assert [r.label for r in rs.roots] == ['stable', 'unstable'], [(r.p_0, r.label) for r in rs.roots]
    assert rs.operating_point is rs.roots[0]
    assert rs.roots[0].slope < 0 < rs.roots[1].slope


@pytest.mark.parametrize('name', ['deviated black oil', 'L-shaped black oil'])
def test_convergence(name):
    """Outputs change at first order as the grid is refined (implicit Euler), within the verifier's order bounds."""
    make, bc = WELLS[name]
    outputs = []
    for n in (50, 100, 200):
        op = SSDFSimulator(make(n)).simulate(bc)
        X = op.state
        outputs.append(np.array([X[0, 0], X[-1, 0], X[-1, 6], op.w_res]))  # PBH, PWH, TWH, liquid rate
    d1, d2 = np.abs(outputs[0] - outputs[1]), np.abs(outputs[1] - outputs[2])
    used = d2 > 1e-9 * np.abs(outputs[2])
    order = np.log2(d1[used] / d2[used])
    assert used.any() and np.all((0.8 <= order) & (order <= 1.25)), order
