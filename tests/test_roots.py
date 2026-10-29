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
    assert roots.label_of(np.nan) == 'indeterminate'
    assert roots.label_of(-np.inf) == 'indeterminate'


def test_a_singular_reduced_jacobian_gives_no_slope():
    """SOL-3 needs the reduced Jacobian nonsingular; where it is not, the slope is NaN, not an error that loses the
    roots found so far."""
    import casadi as ca
    from manywells.discretization import build_system
    from manywells.geometry import WellGeometry
    from manywells.simulator import WellProperties
    system = build_system(WellProperties(geometry=WellGeometry.vertical(100, 3, D=0.1)))
    slope = roots.StabilitySlope(system)
    slope._jac = lambda x, params: ca.DM(np.zeros((system.n_x, system.n_x)))
    s = slope(np.ones(system.n_x), system.params(BoundaryConditions()))
    assert np.isnan(s) and roots.label_of(s) == 'indeterminate'


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


@pytest.mark.slow
def test_a_state_with_a_supersonic_cell_is_not_a_root():
    """
    SOL-8: at v1.0.0+chen#6 of the comparison set (tests/backend_cases.py), Ipopt from one start reaches a state with
    every row zero whose last cell falls from 106 to 10.3 bar, past its sonic point. The search rejects it; the other
    starts give the one root, at 264.00 bar, which is subsonic in every cell and which the Rust core also finds.
    """
    from manywells.simulator import SSDFSimulator
    from .backend_cases import Case
    wp, bc = Case('v1.0.0', 'chen', 6).inputs()
    sim = SSDFSimulator(wp)
    rs = sim.root_set(bc)
    assert [round(r.p_0, 2) for r in rs.roots] == [264.0]
    assert 'not admissible (a cell is supersonic)' in [a.outcome for a in rs.search]
    assert sim.system.subsonic(rs.roots[0].x, sim.system.params(bc))
    assert [round(r.p_0, 2) for r in SSDFSimulator(wp, backend='rust').root_set(bc).roots] == [264.0]


def test_the_energy_slope_with_heat_loss_alone():
    """SOL-9: with heat loss alone (THM-1) the energy row rises in T_i by 1 + delta_md 4h / (D C) along the point's
    mass rows and closures, which hold the heat-capacity flux C at the previous point's."""
    from manywells.configurations import v1_well
    from manywells.discretization import build_system
    from manywells.inflow import Vogel
    from manywells.choke import SimpsonChokeModel
    wp = v1_well(L=2000.0, D=0.127, rho_l=880.0, R_s=420.0, cp_g=2225.0, cp_l=3000.0, f_D=0.03, h=25.0, f_g=0.15,
                 inflow=Vogel(60.0), choke=SimpsonChokeModel(K_c=0.0015, chk_profile='linear'), n_cells=10)
    bc = BoundaryConditions(p_r=220.0, p_s=20.0, u=0.6)
    system = build_system(wp)
    params = system.params(bc)
    x = np.asarray(system.bottom_guess(180.0, params), dtype=float).ravel()
    X = np.vstack([x] * 2)
    X[1, 0] -= 5.0                                  # any state of the next point: the slope holds along the curve
    geo = wp.geometry
    tau = float(system.energy_slope(X[1], X[0], float(system.reservoir_rate(X[0, 0], params)), params,
                                    geo.delta_md[0], geo.cos_incl[0], geo.tvd_frac[1]))
    p, v_g, v_l, alpha, rho_g, rho_l, T = X[0]
    C = wp.fluid.cp_g * alpha * rho_g * v_g + wp.fluid.cp_l * (1 - alpha) * rho_l * v_l
    assert tau == pytest.approx(1 + geo.delta_md[0] * 4 * wp.thermal.h / (geo.D * C), rel=1e-10)


@pytest.mark.parametrize('T_r, T_s, T_lg', ((370.0, 280.0, None), (300.0, 310.0, None), (340.0, 277.15, 380.0),
                                             (340.0, 277.15, 260.0)))
def test_the_temperature_bound(T_r, T_s, T_lg):
    """The CasADi backend bounds T below by min(T_r, T_s, T_lg), or by the Joule-Thomson floor where that is lower,
    and above by max(T_r, T_s, T_lg) + 50 K: frictional heating, hot lift gas and a warm surface put roots above
    T_r + 1 (issue #13)."""
    from dataclasses import replace
    from manywells.discretization import JT_T_PR_MIN, T_MARGIN, build_system
    from manywells.geometry import WellGeometry
    from manywells.simulator import WellProperties
    from manywells.thermal import ThermalModel
    bc = BoundaryConditions(p_r=250.0, p_s=30.0, T_r=T_r, T_s=T_s, T_lg=T_lg, w_lg=1.0)
    wp = WellProperties(geometry=WellGeometry.vertical(length=2000.0, n_cells=4, D=0.1),
                        thermal=ThermalModel(joule_thomson=False))
    for well, floor in ((wp, min(T for T in (T_r, T_s, T_lg) if T is not None)),
                        (replace(wp, thermal=ThermalModel()), JT_T_PR_MIN * wp.fluid.pseudo_critical[1])):
        lb, ub = build_system(well).bounds(bc)
        assert T_MARGIN == 50.0
        assert np.all(ub[6::7] == max(T for T in (T_r, T_s, T_lg) if T is not None) + T_MARGIN)
        assert np.all(lb[6::7] == min(floor, min(T for T in (T_r, T_s, T_lg) if T is not None)))


def _hot_well(case):
    """The Rust core's operating point lies above T_r + 1 K, the CasADi backend's old upper temperature bound: from
    frictional heating in 3 inch tubing at a high rate (issue #13's example), or from lift gas at 380 K."""
    from dataclasses import replace
    from manywells.inflow import ProductivityIndex
    from manywells.pvt.fluid import FluidModel
    from manywells.simulator import WellProperties
    from manywells.thermal import ThermalModel
    wp = WellProperties()
    if case == 'frictional heating':
        return (replace(wp, fluid=FluidModel(gor=50.0), thermal=ThermalModel(h=10.0), inflow=ProductivityIndex(k_l=2.0),
                        geometry=replace(wp.geometry, D=3 * 0.0254)),
                BoundaryConditions(p_r=300.0, p_s=20.0, u=1.0))
    return wp, BoundaryConditions(p_r=200.0, p_s=20.0, T_r=340.0, T_lg=380.0, w_lg=2.0)


@pytest.mark.slow
@pytest.mark.parametrize('case', ('frictional heating', 'hot lift gas'))
def test_the_search_finds_operating_points_warmer_than_the_reservoir(case):
    """Both backends find the operating point, warmer than T_r + 1 K somewhere; with that upper bound the CasADi
    search found no root (issue #13)."""
    from manywells.simulator import SSDFSimulator
    wp, bc = _hot_well(case)
    core = SSDFSimulator(wp, backend='rust').simulate(bc)
    op = SSDFSimulator(wp).simulate(bc)
    assert op.state[:, 6].max() > bc.T_r + 1
    assert op.state[0, 0] == pytest.approx(core.state[0, 0], abs=1e-6)


def _all_liquid_well(n_cells):
    """Well 87 of develop's sampler (seed 2026) at its nominal point, with gor = 40: R_so at (p_r, T_r) is 141 Sm³/Sm³,
    so all the gas is dissolved at the bottomhole (issue #10). With the fitted classifier, develop's default when the
    issue was found: with the five regimes (019) point 0 solves from the bottom guess without the retry."""
    from dataclasses import replace
    from manywells.configurations import DEVELOP
    from manywells.sampling.conditions import nominal_conditions
    from manywells.sampling.wells import sample_well, well_properties
    draw = sample_well(2026, 87)
    from manywells.slip import SlipModel
    wp = replace(well_properties(draw, draw.fractions, DEVELOP, n_cells=n_cells), slip=SlipModel(regime_model='fitted'))
    return wp, replace(wp, fluid=replace(wp.fluid, gor=40.0)), nominal_conditions(draw)


def test_no_slip_guess():
    """Point 0's retry: the no-slip void fraction of the guess's volume rates, at most the guess's own, with the
    velocities that carry the same rates; None where it would be the guess itself, or nothing flows (issue #10)."""
    from manywells.solvers.march import ALPHA_MIN, no_slip_guess
    guess = np.array([300.0, 0.004, 4.0, 0.5, 200.0, 700.0, 380.0])   # q_g = 0.002, q_l = 2 (m/s)
    retry = no_slip_guess(guess)
    assert retry[3] == pytest.approx(0.002 / 2.002, rel=1e-12)
    assert retry[3] * retry[1] == pytest.approx(0.002, rel=1e-12)
    assert (1 - retry[3]) * retry[2] == pytest.approx(2.0, rel=1e-12)
    assert np.array_equal(np.delete(retry, [1, 2, 3]), np.delete(guess, [1, 2, 3]))
    assert no_slip_guess(np.array([300.0, 0.0, 4.0, 0.5, 200.0, 700.0, 380.0]))[3] == ALPHA_MIN
    assert no_slip_guess(np.array([300.0, 4.0, 4.0, 0.5, 200.0, 700.0, 380.0])) is None   # no-slip 0.5: no retry
    assert no_slip_guess(np.array([300.0, 0.0, 0.0, 0.5, 200.0, 700.0, 380.0])) is None   # no flow


def test_the_march_retries_point_0_where_the_bottomhole_is_all_liquid():
    """Point 0 fails from the bottom guess's void fraction of 0.5 where all the gas is dissolved at the bottomhole,
    and the march retries it from the no-slip void fraction (issue #10)."""
    from manywells.discretization import build_system
    from manywells.solvers.march import Marcher, MarchFailed, no_slip_guess
    _, wp, bc = _all_liquid_well(n_cells=10)
    system = build_system(wp)
    params, p_0 = system.params(bc), 322.0
    march = Marcher(system)
    guess = np.asarray(system.bottom_guess(p_0, params), dtype=float).ravel()
    with pytest.raises(MarchFailed):
        march._bottom(guess[1:], np.concatenate([[p_0], params]))
    assert no_slip_guess(guess)[3] < 1e-2
    x = march(params, p_0)
    assert x[3] < 1e-2  # all liquid at the bottomhole, but for the smoothed phase rates' gas


@pytest.mark.slow
def test_the_search_finds_an_operating_point_with_an_all_liquid_bottomhole():
    """With all the gas dissolved at the bottomhole, the CasADi search finds the Rust core's operating point; without
    the march's retry of point 0 it found none (issue #10)."""
    from manywells.simulator import SSDFSimulator
    _, wp, bc = _all_liquid_well(n_cells=100)
    core = SSDFSimulator(wp, backend='rust').simulate(bc)
    op = SSDFSimulator(wp).simulate(bc)
    assert op.state[0, 3] < 1e-2  # all liquid at the bottomhole, but for the smoothed phase rates' gas
    assert op.state[0, 0] == pytest.approx(core.state[0, 0], abs=1e-6)
