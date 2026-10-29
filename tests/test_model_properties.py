"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Property and spot checks of develop's full model, which has no reference root sets (plans/manywells-v2-plan.md,
"New model versions", and Step 7 item 5). The wells exercise develop's options: deviated and L-shaped trajectories,
black oil with dissolved gas, real gas by the Dranchuk-Abou-Kassem equation of state, friction from roughness,
frictional heating, the gravity term and Joule-Thomson cooling, and lift gas colder than the reservoir, injected at
the bottomhole or at a valve above it (specs/features/021-lift-gas-injection-depth.md).

Checked at every root: the verifier's Invariants that do not assume dead oil (total mass rate constant instead of
each phase's, below the injection point and from it up); spot checks of the relations that need no closure (the
inflow rows at the bottom, the choke row at the top, the phase rates of the fluid model at every point, non-negative
friction, heat flowing from the fluid to colder surroundings, the energy balance across the injection point); and the
stability property of two-root wells. Convergence at first order on N, 2N and 4N.
Every check runs on both backends, the CasADi system and the Rust core (specs/features/015-rust-develop-model.md), and
the two must find the same roots.
"""

import dataclasses

import numpy as np
import pytest

from manywells.discretization import DIM_X, PointState, lifts
from manywells.geometry import WellGeometry
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.dead_oil import DeadOil
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.thermal import ThermalModel

from .backend_cases import compare_root_sets

pytestmark = pytest.mark.slow

TOL_RATE = 1e-6  # relative: mass rates
TOL_P = 1e-6  # bar: a pressure rise between neighbouring points
TOL_T = 1e-6  # K
GAS_FLOOR = 1e-3  # kg/s: the free gas of PVT-OIL-13 where all the reservoir gas is dissolved, SMO-1's 5e-4


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


def with_valve(geometry, share, **kw):
    """A well whose lift gas enters at a valve at share of the measured depth from the surface (INF-9)."""
    return WellProperties(geometry=geometry, injection_md=share * geometry.L, **kw)


BLACK_OIL = FluidModel(pvt=BlackOil(rho_o=density_from_api(35.0), rho_g=gas_density_from_sg(0.7)), gor=120.0, wlr=0.3)
VALVE_WELLS = ('deviated, cold lift gas at a valve halfway', 'L-shaped, lift gas at a valve at three quarters')
WELLS = {
    'deviated black oil': (
        lambda n: WellProperties(geometry=deviated(n), fluid=BLACK_OIL),
        BoundaryConditions(p_r=260.0, p_s=25.0, T_r=370.0, T_s=280.0, u=0.6),
    ),
    'L-shaped black oil': (
        lambda n: WellProperties(geometry=l_shaped(n), fluid=BLACK_OIL),
        BoundaryConditions(p_r=240.0, p_s=20.0, T_r=365.0, T_s=280.0, u=0.7),
    ),
    'deviated, cold lift gas': (
        lambda n: WellProperties(
            geometry=deviated(n, tvd=2000.0), fluid=dataclasses.replace(BLACK_OIL, gor=40.0, wlr=0.5)
        ),
        BoundaryConditions(p_r=200.0, p_s=20.0, T_r=360.0, T_s=280.0, u=0.8, w_lg=2.0, T_lg=300.0),
    ),
    'vertical dead oil, real gas': (
        lambda n: WellProperties(
            geometry=WellGeometry.vertical(2000.0, n), fluid=FluidModel(pvt=DeadOil(rho_o=880.0), gor=60.0)
        ),
        BoundaryConditions(p_r=210.0, p_s=20.0, T_r=360.0, T_s=280.0, u=0.5),
    ),
    VALVE_WELLS[0]: (
        lambda n: with_valve(deviated(n, tvd=2000.0), 0.5, fluid=dataclasses.replace(BLACK_OIL, gor=40.0, wlr=0.5)),
        BoundaryConditions(p_r=200.0, p_s=20.0, T_r=360.0, T_s=280.0, u=0.8, w_lg=2.0, T_lg=300.0),
    ),
    # The lift gas at the ambient temperature at the valve (T_lg None), in the well's build section
    VALVE_WELLS[1]: (
        lambda n: with_valve(l_shaped(n), 0.75, fluid=BLACK_OIL),
        BoundaryConditions(p_r=240.0, p_s=20.0, T_r=365.0, T_s=280.0, u=0.7, w_lg=1.5),
    ),
}


BACKENDS = ('casadi', 'rust')


@pytest.fixture(scope='module')
def solved():
    """Every well's root set at 100 cells, and its simulator, with each backend."""
    out = {}
    for name, (make, bc) in WELLS.items():
        for backend in BACKENDS:
            sim = SSDFSimulator(make(100), backend=backend)
            out[name, backend] = (sim, bc, sim.root_set(bc))
    return out


def points(sim, x):
    return [PointState.of(row) for row in np.reshape(x, (-1, DIM_X))]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', WELLS)
def test_well_has_a_stable_operating_point(solved, name, backend):
    sim, bc, rs = solved[name, backend]
    assert rs.operating_point is not None, [a.outcome for a in rs.search]
    assert rs.operating_point.label == 'stable'


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', WELLS)
def test_invariants(solved, name, backend):
    """The verifier's Invariants, with the total mass rate constant instead of each phase's (with dissolved gas):
    w_res + w_g_res below the injection point k, and w_lg more from point k up (INF-9). From point k up the lift gas
    keeps PVT-OIL-13's smoothing inactive, and the total is constant to TOL_RATE; below it, where all the reservoir gas
    can be dissolved, the free gas is SMO-1's floor, and the total is w_res + w_g_res to GAS_FLOOR."""
    sim, bc, rs = solved[name, backend]
    A, k = sim.wp.geometry.A, sim.wp.injection_point
    for root in rs.roots:
        X = root.state
        p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
        assert np.all((0 <= alpha) & (alpha <= 1))
        assert np.all(v_g > 0) and np.all(v_l > 0) and np.all(rho_g > 0) and np.all(rho_l > 0)
        w_m = A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
        assert np.max(np.abs(w_m[k:] - w_m[k])) <= TOL_RATE * w_m[k]
        w_in = root.w_res + root.w_g_res
        assert np.max(np.abs(w_m[:k] - w_in), initial=0.0) <= GAS_FLOOR
        assert w_m[k] == pytest.approx(w_in + bc.w_lg, abs=GAS_FLOOR)
        assert np.max(np.diff(p)) <= TOL_P, 'pressure rises along the well'
        assert p[-1] > bc.p_s and p[0] < bc.p_r
        assert root.choked == (bc.p_s <= sim.wp.choke.cpr * p[-1])


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', WELLS)
def test_every_root_is_subsonic_with_rising_energy_rows(solved, name, backend):
    """SOL-8 and SOL-9 at every root of both backends, by the CasADi system's slopes."""
    sim, bc, rs = solved[name, backend]
    system = solved[name, 'casadi'][0].system
    params = system.params(bc)
    for root in rs.roots:
        assert system.subsonic(root.x, params) and system.energy_rows_rise(root.x, params)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', WELLS)
def test_spot_checks(solved, name, backend):
    """Relations that need no closure: the inflow rows (INF-6, INF-7), the choke row (CHK-1), and at every point
    the phase rates of the fluid model (DISC-7, DISC-8) and non-negative friction (FRIC-1)."""
    sim, bc, rs = solved[name, backend]
    wp = sim.wp
    A, D, fluid, lift = wp.geometry.A, wp.geometry.D, wp.fluid, lifts(wp)
    for root in rs.roots:
        S = points(sim, root.x)
        w_res = float(wp.inflow.liquid_mass_flow_rate(S[0].p, bc.p_r))
        assert root.w_res == pytest.approx(w_res, rel=1e-12)
        for i, s in enumerate(S):
            w_g, w_l = (float(v) for v in fluid.phase_rates(s.p, s.T, w_res, lift[i] * bc.w_lg))  # INF-9
            assert A * s.gas_flux == pytest.approx(w_g, rel=TOL_RATE)
            assert A * s.liquid_flux == pytest.approx(w_l, rel=TOL_RATE)
            assert float(wp.friction.pressure_gradient(s, fluid, D)) >= 0
        w_m = A * (S[-1].gas_flux + S[-1].liquid_flux)
        assert w_m == pytest.approx(float(wp.choke.mass_flow_rate(bc.u, bc.p_s, S[-1], A)), rel=TOL_RATE)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', [n for n in WELLS if WELLS[n][1].w_lg == 0])
def test_heat_flows_from_the_fluid_to_the_surroundings(solved, name, backend):
    """Without lift gas the fluid enters at T_r, the ambient temperature at the bottomhole, and is warmer than its
    surroundings everywhere above: the heat loss carries heat outwards, against frictional heating and gravity. The
    Joule-Thomson term (THM-8) can cool the fluid of a gas-rich well below its surroundings
    (specs/features/016-joule-thomson.md); these wells carry too little gas for that."""
    sim, bc, rs = solved[name, backend]
    geo = sim.wp.geometry
    T_a = np.array([sim.wp.thermal.ambient_temperature(f, bc.T_r, bc.T_s) for f in geo.tvd_frac])
    for root in rs.roots:
        T = root.state[:, 6]
        assert T[0] == pytest.approx(bc.T_r, abs=TOL_T)
        assert np.all(T[1:] - T_a[1:] > 0)


GAS_RICH = (
    lambda n, joule_thomson: WellProperties(
        geometry=WellGeometry.vertical(3000.0, n),
        fluid=FluidModel(
            pvt=BlackOil(rho_o=density_from_api(40.0), rho_g=gas_density_from_sg(0.65)), gor=800.0, wlr=0.1
        ),
        thermal=ThermalModel(joule_thomson=joule_thomson),
    ),
    BoundaryConditions(p_r=300.0, p_s=20.0, T_r=380.0, T_s=280.0, u=0.5),
)


@pytest.mark.parametrize('backend', BACKENDS)
def test_joule_thomson_cools_a_gas_rich_well(backend):
    """THM-8: in a gas-rich well (a gas mass fraction of 0.4) the real gas cools as it expands, so the wellhead is
    several kelvin colder with the term than without it, from the same bottomhole temperature (8 K on 2026-10-02)."""
    make, bc = GAS_RICH
    on, off = (SSDFSimulator(make(100, jt), backend=backend).simulate(bc).state for jt in (True, False))
    assert on[0, 6] == pytest.approx(bc.T_r, abs=TOL_T) and off[0, 6] == pytest.approx(bc.T_r, abs=TOL_T)
    assert on[-1, 6] < off[-1, 6] - 5.0


@pytest.mark.parametrize('backend', BACKENDS)
def test_cold_lift_gas_cools_the_inflow(solved, backend):
    """THM-5: lift gas colder than the reservoir lowers the bottomhole temperature, but not below T_lg."""
    sim, bc, rs = solved['deviated, cold lift gas', backend]
    T_0 = rs.operating_point.state[0, 6]
    assert bc.T_lg < T_0 < bc.T_r


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', VALVE_WELLS)
def test_energy_balance_across_the_injection_point(solved, name, backend):
    """THM-9, DISC-12: at the injection point k the heat-capacity flux of the state, C_k T_k, is that of the flow
    from below at the end of cell k's step, T~_k = T_{k-1} + ΔMD (dT/dMD)_k, plus the lift gas's at T_lg, with the
    rates of the state. The point's temperature lies between the two, and the bottomhole is at T_r (THM-3)."""
    sim, bc, rs = solved[name, backend]
    wp, k = sim.wp, sim.wp.injection_point
    geo, fluid, A = wp.geometry, wp.fluid, wp.geometry.A
    T_lg = bc.T_lg if bc.T_lg is not None else wp.thermal.ambient_temperature(geo.tvd_frac[k], bc.T_r, bc.T_s)
    assert 0 < k < geo.n_cells
    for root in rs.roots:
        S = points(sim, root.x)
        s, prev = S[k], S[k - 1]
        T_a = wp.thermal.ambient_temperature(geo.tvd_frac[k], bc.T_r, bc.T_s)
        F = float(wp.friction.pressure_gradient(s, fluid, geo.D))
        dT = float(wp.thermal.temperature_gradient(s, fluid, T_a, F, 0.0, geo.cos_incl[k - 1], geo.D))
        T_tilde = prev.T + geo.delta_md[k - 1] * dT
        C_k = A * (fluid.cp_g * s.gas_flux + fluid.cp_l * s.liquid_flux)
        C_lg = fluid.cp_g * bc.w_lg
        assert C_k * s.T == pytest.approx((C_k - C_lg) * T_tilde + C_lg * T_lg, rel=1e-9)
        assert min(T_tilde, T_lg) < s.T < max(T_tilde, T_lg)
        assert S[0].T == pytest.approx(bc.T_r, abs=TOL_T)


@pytest.mark.parametrize('backend', BACKENDS)
def test_cold_lift_gas_at_a_valve_cools_the_flow_from_there_up(solved, backend):
    """INF-9, THM-9: lift gas colder than the flow cools it at the valve, not at the bottomhole, so the temperature
    falls more steeply over the injection cell than over the cells next to it."""
    sim, bc, rs = solved[VALVE_WELLS[0], backend]
    k, T = sim.wp.injection_point, rs.operating_point.state[:, 6]
    steps = np.diff(T)  # steps[i - 1] is cell i's
    assert steps[k - 1] < min(steps[k - 2], steps[k]) - 1.0


@pytest.mark.parametrize('backend', BACKENDS)
def test_gas_comes_out_of_solution_up_the_well(solved, backend):
    """BAL-10: with black oil the free-gas rate grows as the pressure falls towards the wellhead."""
    sim, bc, rs = solved['deviated black oil', backend]
    X = rs.operating_point.state
    w_g = sim.wp.geometry.A * X[:, 3] * X[:, 4] * X[:, 1]
    assert np.all(np.diff(w_g) > 0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_two_root_well_stability(backend):
    """A two-root well has one stable and one unstable root, the unstable one at higher p_0 (plan, "New model
    versions"), here on develop's full model in a deviated well."""
    fl = FluidModel(pvt=BlackOil(rho_o=density_from_api(25.0), rho_g=gas_density_from_sg(0.7)), gor=60.0, wlr=0.6)
    wp = WellProperties(geometry=deviated(60, tvd=2200.0), fluid=fl)
    bc = BoundaryConditions(p_r=225.0, p_s=29.0, T_r=360.0, T_s=280.0, u=0.6)
    rho_static = float(fl.liquid_density(bc.p_r, bc.T_r) + fl.liquid_density(bc.p_s, bc.T_s)) / 2  # in situ (PVT-MIX-6)
    assert bc.p_r - bc.p_s < rho_static * 9.80665 * 2200.0 / 1e5  # the static column cannot reach the separator (SOL-7)
    rs = SSDFSimulator(wp, backend=backend).root_set(bc)
    assert [r.label for r in rs.roots] == ['stable', 'unstable'], [(r.p_0, r.label) for r in rs.roots]
    assert rs.operating_point is rs.roots[0]
    assert rs.roots[0].slope < 0 < rs.roots[1].slope


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('share', (0.5, 0.75))
def test_two_root_well_stability_with_a_valve(backend, share):
    """The two-root well of test_two_root_well_stability, with p_r 1 bar lower and 5 g/s of lift gas at a valve at
    half and at three quarters of the measured depth: still one stable and one unstable root, the unstable one at higher
    p_0. More lift gas makes the well flow with one root, as gas lift is meant to."""
    fl = FluidModel(pvt=BlackOil(rho_o=density_from_api(25.0), rho_g=gas_density_from_sg(0.7)), gor=60.0, wlr=0.6)
    wp = with_valve(deviated(60, tvd=2200.0), share, fluid=fl)
    bc = BoundaryConditions(p_r=224.0, p_s=29.0, T_r=360.0, T_s=280.0, u=0.6, w_lg=0.005)
    rs = SSDFSimulator(wp, backend=backend).root_set(bc)
    assert [r.label for r in rs.roots] == ['stable', 'unstable'], [(r.p_0, r.label) for r in rs.roots]
    assert rs.operating_point is rs.roots[0]
    assert rs.roots[0].slope < 0 < rs.roots[1].slope


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', ['deviated black oil', 'L-shaped black oil'] + list(VALVE_WELLS))
def test_convergence(name, backend):
    """Outputs change at first order as the grid is refined (implicit Euler), within the verifier's order bounds. The
    valve wells' grids have a point at the valve (INF-9) at each refinement."""
    make, bc = WELLS[name]
    outputs = []
    for n in (40, 80, 160) if name in VALVE_WELLS else (50, 100, 200):
        op = SSDFSimulator(make(n), backend=backend).simulate(bc)
        X = op.state
        outputs.append(np.array([X[0, 0], X[-1, 0], X[-1, 6], op.w_res]))  # PBH, PWH, TWH, liquid rate
    d1, d2 = np.abs(outputs[0] - outputs[1]), np.abs(outputs[1] - outputs[2])
    used = d2 > 1e-9 * np.abs(outputs[2])
    order = np.log2(d1[used] / d2[used])
    assert used.any() and np.all((0.8 <= order) & (order <= 1.25)), order


@pytest.mark.parametrize('name', WELLS)
def test_the_backends_find_the_same_roots(solved, name):
    """The Rust core finds every root of the CasADi backend with its label, and every root it finds is one of the CasADi
    system's, by the pass rule of test_backend_comparison.py."""
    casadi, bc, rs_casadi = solved[name, 'casadi']
    rust, _, rs_rust = solved[name, 'rust']
    c = compare_root_sets(
        casadi.wp, bc, casadi.system, rs_casadi.roots, rs_rust.roots, lambda x: rust._roots.rows(bc, x)[1]
    )
    assert c.ok, c
