"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Rust core as the simulator's backend (manywells.solvers.rust, rust/). The verifier checks it on the case set
in the v1.0.0 configuration (scripts/verification/develop_candidate.py --backend rust); these tests check it against
the CasADi backend, which is independent code, on v1.0.0's wells: the same roots, and every CasADi row zero at the
core's roots. The core's rows are checked against v1.0.0's row vectors and its components against the vector tables
in test_spec_vectors.py, and both against the CasADi backend in every configuration in test_backend_comparison.py.
"""

from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from dataclasses import replace

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import v1_well
from manywells.inflow import FixedFlowRate, InflowModel, ProductivityIndex, Vogel
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.simulator import BoundaryConditions, NoOperatingPoint, SSDFSimulator, WellProperties
from manywells.slip import SlipModel
from manywells.solvers.roots import TOL_X, state_distance

DATA = Path(__file__).resolve().parents[1] / 'verification' / 'data'


def well_977():
    """Well 977 of manywells-sol-1 (the public config) at u = 0.5. v1.0.0 has its stable root at p_0 = 169.00 bar and
    the trickle root at 208.02 bar (plans/evidence/README.md); the port put the stable root at 169.13 bar."""
    wp = v1_well(L=2113.295350541874, D=0.1016, rho_l=989.4990880705913, R_s=414.3821670270195, cp_g=2225,
                 cp_l=3962.1236794293, f_D=0.0272533546952829, h=32.95930831622784, f_g=0.2090894400617591,
                 inflow=Vogel(w_l_max=23.864413094825), choke=SimpsonChokeModel(K_c=0.0016739098450736,
                                                                                chk_profile='concave'), n_cells=100)
    bc = BoundaryConditions(p_r=210.7407624097664, p_s=69.43649798000607, T_r=351.5488605162562, T_s=277.15, u=0.5)
    return wp, bc


# The three wells of the port's own tests (rust_implementation, manywells_rs/tests/test_simulator.py), with the
# number of roots and their bottomhole pressures there (to 2%), highest first
PORT_WELLS = {
    'pi_bernoulli_u0.5': (
        dict(L=2000, D=0.1554, rho_l=850, R_s=518.3, cp_g=2225, cp_l=4180, f_D=0.05, h=20.0, f_g=0.1379,
             inflow=ProductivityIndex(k_l=0.5), choke=BernoulliChokeModel(K_c=0.0018966705911591126)),
        dict(p_r=170, p_s=20, T_r=373.15, T_s=277.15, u=0.5, w_lg=0.0), [169.732375, 130.565907]),
    'vogel_simpson_u0.63': (
        dict(L=1721.9603339481268, D=0.1397, rho_l=930.743487963018, R_s=363.4427150793591, cp_g=2225,
             cp_l=3156.504075437806, f_D=0.05, h=12.67595099338305, f_g=0.2599255592113066,
             inflow=Vogel(w_l_max=106.717070619435), choke=SimpsonChokeModel(K_c=0.0014075302875897,
                                                                             chk_profile='convex')),
        dict(p_r=115.18137045160027, p_s=18.19728485632093, T_r=339.8088100184438, T_s=277.15, u=0.6323478195871421,
             w_lg=4.802525514813793), [109.94]),
    'no_solution': (
        dict(L=3464.592636891714, D=0.0761999999999999, rho_l=972.9309195918396, R_s=355.44652560268554, cp_g=2225,
             cp_l=3828.209929393251, f_D=0.05, h=22.77347807279142, f_g=0.2052894249538803,
             inflow=Vogel(w_l_max=24.759037189495423), choke=SimpsonChokeModel(K_c=0.0009286106405242)),
        dict(p_r=222.35772124368995, p_s=42.0459274200113, T_r=392.0877791067514, T_s=277.15, u=0.3594533898306623,
             w_lg=0.0), []),
}


def port_well(name):
    well, bc, _ = PORT_WELLS[name]
    return v1_well(**well, n_cells=100), BoundaryConditions(**bc)


WELLS = {'977': well_977, **{name: (lambda name=name: port_well(name)) for name in PORT_WELLS}}


def test_well_977_has_v1s_roots():
    wp, bc = well_977()
    rs = SSDFSimulator(wp, backend='rust').root_set(bc)
    assert [r.label for r in rs.roots] == ['stable', 'unstable']
    assert [r.p_0 for r in rs.roots] == pytest.approx([169.0004, 208.0218], abs=1e-3)
    assert rs.operating_point is rs.roots[0]


@pytest.mark.parametrize('name', PORT_WELLS)
def test_the_ports_wells(name):
    wp, bc = port_well(name)
    sim = SSDFSimulator(wp, backend='rust')
    rs = sim.root_set(bc)
    expected = PORT_WELLS[name][2]
    assert sorted((r.p_0 for r in rs.roots), reverse=True) == pytest.approx(expected, rel=0.02)
    if not expected:
        with pytest.raises(NoOperatingPoint):
            sim.simulate(bc)


@pytest.mark.parametrize('name', ['977', 'pi_bernoulli_u0.5'])
def test_casadi_rows_vanish_at_the_cores_roots(name):
    """Every row of the CasADi system, which is independent code, is zero at each root the core finds."""
    wp, bc = WELLS[name]()
    roots = SSDFSimulator(wp, backend='rust').root_set(bc).roots
    assert len(roots) == 2
    casadi = SSDFSimulator(wp)
    params = casadi.system.params(bc)
    ids = np.array(casadi.system.row_ids)
    for r in roots:
        rows = np.asarray(casadi.system.residual(r.x, params)).ravel()
        p, v_g, v_l, alpha, rho_g, rho_l, T = r.state[-1]
        w_m = wp.geometry.A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
        assert np.max(np.abs(rows[ids != 'CHK-1'])) < 1e-8
        assert abs(rows[ids == 'CHK-1'][0]) < 1e-6 * w_m  # p_0's resolution times dR/dp_0, steep at a trickle root


class LinearInflow(InflowModel):
    """A user's inflow model: the core cannot run Python code, so the CasADi backend solves such a well."""

    def liquid_mass_flow_rate(self, p, p_r):
        return 0.5 * (p_r - p)


def test_the_rust_backend_refuses_what_it_does_not_cover():
    wp = well_977()[0]
    with pytest.raises(ValueError, match='inflow: LinearInflow is not one of the core'):
        SSDFSimulator(replace(wp, inflow=LinearInflow()), backend='rust')
    with pytest.raises(ValueError, match='C_0 >= 1'):
        SSDFSimulator(replace(wp, slip=SlipModel(C_0_annular=0.9)), backend='rust')
    with pytest.raises(ValueError, match='Rust core cannot solve'):
        SSDFSimulator(replace(wp, inflow=FixedFlowRate(w_l_const=10.0)), backend='rust')  # Step 9: not ported yet
    with pytest.raises(ValueError, match='backend'):
        SSDFSimulator(wp, backend='fortran')


def test_the_slip_law_brackets_the_void_fraction():
    """
    The core solves the slip law for α by Brent on [0, 1]: h(α) = α (C_0 j_m + v_inf) - j_g needs h(0) < 0 < h(1),
    which holds because C_0 >= 1 and v_inf >= 0 for every mix of the regimes and every inclination in [0, 1]. Checked
    on random states, with the slip constants the core accepts.
    """
    rng = np.random.default_rng(20261002)
    h = void_fraction_residual(SlipModel())
    n = 10_000
    j_g, j_l = 10 ** rng.uniform(-3, 1.5, n), 10 ** rng.uniform(-3, 1, n)
    rho_l = rng.uniform(600, 1050, n)
    rho_g = rho_l * 10 ** rng.uniform(-4, -0.05, n)
    sigma, D = rng.uniform(0.005, 0.04, n), rng.uniform(0.05, 0.2, n)
    cos_incl = np.concatenate([[0.0, 1.0], rng.uniform(0, 1, n - 2)])
    args = [j_g, j_l, rho_g, rho_l, sigma, D, cos_incl]
    at = lambda alpha: np.asarray(h.map(n)(np.full((1, n), alpha), *(a.reshape(1, -1) for a in args))).ravel()
    assert np.all(at(1e-12) < 0) and np.all(at(1 - 1e-12) > 0)  # α = 0 and 1 divide by zero in v_g or v_l


def test_solution_as_df_with_the_rust_backend():
    wp, bc = well_977()
    sim = SSDFSimulator(wp, backend='rust')
    op = sim.simulate(bc)
    df = sim.solution_as_df(op)
    assert len(df) == wp.geometry.n_cells + 1
    assert tuple(df['flow-regime']) == op.flow_regime == tuple(sim.solution_as_df(op.x.tolist())['flow-regime'])


@pytest.mark.slow
@pytest.mark.parametrize('name', WELLS)
def test_rust_and_casadi_find_the_same_roots(name):
    wp, bc = WELLS[name]()
    rust = SSDFSimulator(wp, backend='rust').root_set(bc).roots
    casadi = SSDFSimulator(wp).root_set(bc).roots
    assert len(rust) == len(casadi)
    for r, c in zip(rust, casadi):
        assert r.label == c.label
        assert r.choked == c.choked
        assert r.flow_regime == c.flow_regime
        assert state_distance(r.x, c.x, bc) <= TOL_X


def case_set():
    from manywells_verify.cases import read_cases
    from scripts.verification.develop_candidate import well_of
    return {cid: well_of(case) for cid, case in read_cases(DATA / 'cases.parquet').items()}


@pytest.mark.slow
def test_the_slope_has_the_sign_of_the_bracket_on_the_case_set():
    """The label comes from a central difference of R; the core also reports which way R crossed zero in the
    bracket the root came from. With one crossing per bracket the two agree."""
    from manywells.solvers.rust import RustRootFinder, operating_point
    n = 0
    for cid, (wp, bc) in case_set().items():
        found, _ = RustRootFinder(wp).core.root_set(operating_point(bc))
        for r in found:
            assert (r.slope > 0) == r.rising, (cid, r.x[0])
            n += 1
    assert n >= 200


def void_fraction_residual(slip=SlipModel()):
    """h(α) = α (C_0 j_m + v_inf) - j_g, the slip row times -α at fixed superficial velocities, as a CasADi function
    of (α, j_g, j_l, rho_g, rho_l, sigma, D, cos_incl) on develop's slip model."""
    alpha, j_g, j_l, rho_g, rho_l, sigma, D, cos_incl = (ca.SX.sym(n) for n in ('alpha', 'j_g', 'j_l', 'rho_g',
                                                                                'rho_l', 'sigma', 'D', 'cos_incl'))
    C_0, v_inf = slip.identify_parameters(j_g / alpha, j_l / (1 - alpha), alpha, rho_g, rho_l, sigma, D, cos_incl)
    return ca.Function('h', [alpha, j_g, j_l, rho_g, rho_l, sigma, D, cos_incl],
                       [alpha * (C_0 * (j_g + j_l) + v_inf) - j_g])


@pytest.mark.slow
def test_the_slip_row_has_one_root_in_alpha_at_every_reference_point():
    """
    A property of the model that the core's void-fraction solve relies on to return the reference's α: at fixed
    superficial velocities, h(α) changes sign once on (0, 1), at every point of every reference root. If this ever
    fails, it is a question about the model, not the solver.
    """
    import pandas as pd
    from manywells_verify.cases import read_cases
    cases = read_cases(DATA / 'cases.parquet')
    ref = pd.read_parquet(DATA / 'reference_roots.parquet')
    X = np.vstack([np.reshape(x, (-1, 7)) for x in ref.x])
    D = np.concatenate([np.full(len(x) // 7, cases[cid].params['D']) for cid, x in zip(ref.case_id, ref.x)])
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    sigma = np.array([dead_oil_surface_tension(r, t) for r, t in zip(rho_l, T)], dtype=float)
    inputs = np.vstack([alpha * v_g, (1 - alpha) * v_l, rho_g, rho_l, sigma, D, np.ones_like(D)])
    grid = np.linspace(1e-9, 1 - 1e-9, 201)
    h = void_fraction_residual().map(len(grid) * X.shape[0])
    args = [np.repeat(grid, X.shape[0])] + list(np.tile(inputs, len(grid)))
    values = np.asarray(h(*(a.reshape(1, -1) for a in args))).reshape(len(grid), -1)
    sign_changes = np.sum(np.diff(np.sign(values), axis=0) != 0, axis=0)
    assert np.all(sign_changes == 1), f'{np.sum(sign_changes != 1)} points with {set(sign_changes[sign_changes != 1])}'
