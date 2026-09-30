"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Checks develop's implementation against the test vectors in specs/model/, which come from v1.0.0
(specs/model/README.md, "Test vectors").

A vector whose equation has no implementation on develop is skipped with the reason. A vector that
develop is known not to reproduce is a strict expected failure, so it fails once develop does
reproduce it and the entry has to go; Step 7 of plans/manywells-v2-plan.md resolves each one.
"""

import math

import casadi as ca
import numpy as np
import pytest

from manywells.ca_functions import ca_max_approx, ca_min_approx, ca_softmax
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt import LiquidProperties, api_from_density, liquid_mix
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.pvt.fluid import FluidModel
from manywells.pvt.gas import gas_density
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.slip import SlipModel, classify_flow_regime
from manywells.units import P_REF, T_REF

from .spec_parse import row_vectors, vector_tables

REL, ABS = 1e-12, 1e-15      # component vectors: the same arithmetic, so agreement to rounding
ROW_REL = 1e-10              # row vectors: relative to the row's size, which is far from zero


def num(x):
    return float(np.asarray(ca.DM(x)).item()) if isinstance(x, (ca.DM, ca.SX)) else float(x)


def vec(x):
    return tuple(np.asarray(ca.DM(x)).ravel())


def choke(model, K_c, profile):
    return model(K_c=K_c, chk_profile=profile)


# develop's function for each vector table, keyed by the table's heading
ADAPTERS = {
    'SMO-1': lambda x, y, eps: ca_max_approx(x, y, eps),
    'SMO-2': lambda x, y, eps: ca_min_approx(x, y, eps),
    'SMO-3': lambda y1, y2, y3: vec(ca_softmax(ca.DM([y1, y2, y3]))),
    'CHK-2, CHK-3': lambda K_c, profile, u, p_in, p_out, rho, Phi:
        choke(SimpsonChokeModel, K_c, profile).choke_equation(u, p_in, p_out, rho, Phi),
    'CHK-4': lambda gamma: SimpsonChokeModel.critical_pressure_ratio(gamma),
    'CHK-5 (multiplier)': SimpsonChokeModel.simpson_multiplier,
    'CHK-5 (rate)': lambda K_c, profile, u, p_in, p_out, x_g, rho_g, rho_l:
        choke(SimpsonChokeModel, K_c, profile).mass_flow_rate(u, p_in, p_out, x_g, rho_g, rho_l),
    'CHK-6': lambda K_c, profile, u, p_in, p_out, rho_m:
        choke(BernoulliChokeModel, K_c, profile).mass_flow_rate(u, p_in, p_out, rho_m),
    'CHK-7': lambda u: choke(SimpsonChokeModel, 1.0, 'linear').choke_opening(u),
    'CHK-8': lambda u: choke(SimpsonChokeModel, 1.0, 'sigmoid').choke_opening(u),
    'CHK-9': lambda u: choke(SimpsonChokeModel, 1.0, 'convex').choke_opening(u),
    'CHK-10': lambda u: choke(SimpsonChokeModel, 1.0, 'concave').choke_opening(u),
    'CHK-12': lambda p_in, p_out: bool(choke(SimpsonChokeModel, 1.0, 'linear').is_choked(p_in, p_out)),
    'INF-1': lambda w_l_max, p, p_r: Vogel(w_l_max).liquid_mass_flow_rate(p, p_r),
    'INF-2': lambda k_l, p, p_r: ProductivityIndex(k_l).liquid_mass_flow_rate(p, p_r),
    'SLIP-2, SLIP-3': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, D, **_:
        SlipModel().identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl=1.0),
    'SLIP-4': lambda rho_g, rho_l, sigma, **_: SlipModel.harmathy_rise_velocity(rho_g, rho_l, sigma),
    'SLIP-5': SlipModel.taylor_rise_velocity,
    'SLIP-6, SLIP-7': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, **_:
        vec(classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl=1.0)),
    'SLIP-8': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, **_:
        SlipModel().flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl=1.0),
    'PVT-GAS-1': lambda p, T, R_s: gas_density(R_s, p * 1e5, T),
    'PVT-GAS-2': lambda R_s: gas_density(R_s),
    'PVT-OIL-2': api_from_density,
    'PVT-OIL-3': dead_oil_surface_tension,
    'PVT-MIX-2, PVT-MIX-3, PVT-MIX-4': lambda rho_o, cp_o, rho_w, cp_w, x_o: (lambda m: (m.rho, m.cp))(
        liquid_mix(LiquidProperties('oil', rho_o, cp_o), LiquidProperties('water', rho_w, cp_w), x_o)),
}

NOT_ON_DEVELOP = {
    'INF-3': "develop's FixedFlowRate fixes the liquid rate only and takes the gas from the fluid model",
    'INF-4': "develop computes the reservoir gas rate inline in SSDFSimulator._gas_and_liquid_flow_rate",
}

KNOWN_DEVIATIONS = {
    'SLIP-2, SLIP-3': 'develop multiplies the Taylor velocity by sqrt(cos_incl + 1e-9) * (1 + sin_incl)**1.2, '
                      'which is 1 + 5e-10 in a vertical well (plans/develop_model_changes.md)',
}


def marks(key, skips, deviations):
    if key in skips:
        return [pytest.mark.skip(reason=skips[key])]
    if key in deviations:
        return [pytest.mark.xfail(strict=True, reason=deviations[key])]
    return []


def vector_params():
    return [pytest.param(table, id=table.heading, marks=marks(table.heading, NOT_ON_DEVELOP, KNOWN_DEVIATIONS))
            for table in vector_tables()]


def check_outputs(table, k, expected, got):
    got = got if isinstance(got, tuple) else (got,)
    assert len(got) == len(table.outputs)
    for name, value in zip(table.outputs, got):
        want = expected[name]
        if isinstance(want, (bool, str)):
            assert value == want, f'row {k}, {name}: {value!r} != {want!r}'
        else:
            assert math.isclose(num(value), want, rel_tol=REL, abs_tol=ABS), f'row {k}, {name}: {num(value)!r} != {want!r}'


@pytest.mark.parametrize('table', vector_params())
def test_component_vectors(table):
    if table.heading not in ADAPTERS:
        pytest.fail(f'no adapter for vector table {table.heading!r}: add one, or list it in NOT_ON_DEVELOP')
    for k, (inputs, expected) in enumerate(table.rows):
        check_outputs(table, k, expected, ADAPTERS[table.heading](**inputs))


def test_every_table_is_handled():
    headings = {t.heading for t in vector_tables()}
    assert headings, 'no vector tables found'
    assert set(KNOWN_DEVIATIONS) <= set(ADAPTERS)
    assert headings <= set(ADAPTERS) | set(NOT_ON_DEVELOP), headings - set(ADAPTERS) - set(NOT_ON_DEVELOP)


# Row vectors (specs/model/discretization.md), on develop's nearest configuration to v1.0.0 today:
# a vertical well, fixed f_D, dead oil with wlr = 0 and rho_o = rho_l, ideal gas. Step 7 replaces this
# with the v1-compatibility configuration, which must reproduce every row.

ROW_IDS = {'first': ['INF-6', 'INF-7', 'THM-3'], 'cell': ['DISC-2', 'DISC-3', 'DISC-4', 'DISC-5'],
           'last': ['CHK-1'], 'closure': ['SLIP-1', 'PVT-GAS-1', 'PVT-MIX-1']}

ROWS_NOT_COMPARABLE = {
    'DISC-2': 'develop replaces flux continuity by A alpha rho_g v_g = w_g(p, T) at every point',
    'DISC-3': 'develop replaces flux continuity by A (1 - alpha) rho_l v_l = w_l(p, T) at every point',
}

ROW_DEVIATIONS = {
    'INF-6': 'the dissolved-gas path uses smin(R_so ..., w_g) and smax(..., 0), which are not exact at R_so = 0',
    'INF-7': 'the dissolved-gas path uses smin(R_so ..., w_g), which is not exact at R_so = 0',
    'DISC-5': 'develop always adds frictional heating and a gravity term to the energy row',
    'SLIP-1': 'develop takes sigma from rho_o, not from the state rho_l, and adds 1e-9 to cos_incl '
              'in the Taylor deviation factor',
}


def develop_simulator(w):
    """develop's simulator in its nearest configuration to v1.0.0 for a well of v1_rows.json."""
    rho_g_sc = P_REF / (w['R_s'] * T_REF)
    fluid = FluidModel(rho_o=w['rho_l'], rho_g=rho_g_sc, wlr=0.0, gor=w['f_g'] * w['rho_l'] / ((1 - w['f_g']) * rho_g_sc),
                       oil_model='dead_oil', ideal_gas=True, cp_g=w['cp_g'], cp_o=w['cp_l'])
    inflow = Vogel(w['w_l_max']) if w['inflow'] == 'vogel' else ProductivityIndex(w['k_l'])
    model = SimpsonChokeModel if w['choke'] == 'simpson' else BernoulliChokeModel
    wp = WellProperties(geometry=WellGeometry.vertical(length=w['L'], n_cells=w['n_cells'], D=w['D']), fluid=fluid,
                        f_D=w['f_D'], h=w['h'], inflow=inflow, choke=model(K_c=w['K_c'], chk_profile=w['profile']))
    bc = BoundaryConditions(p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], u=w['u'], w_lg=w['w_lg'])
    return SSDFSimulator(wp, bc)


def develop_rows(sim, X):
    """develop's rows at every point, in v1.0.0's canonical form, as {ID: [value per point]}."""
    out = {}
    n = sim.n_cells
    for i in range(n + 1):
        x = list(X[i])
        if i == 0:
            vals, ids = sim._left_boundary_eqs(x), list(ROW_IDS['first'])
        else:
            vals, ids = sim._differential_equations(x, list(X[i - 1]), i), list(ROW_IDS['cell'])
            if i == n:
                vals, ids = vals + sim._right_boundary_eqs(x), ids + ROW_IDS['last']
        vals, ids = vals + sim._closure_relations(x, i), ids + ROW_IDS['closure']
        for eq_id, v in zip(ids, vals):
            v = num(v)
            if eq_id == 'PVT-GAS-1':  # develop: rho_g - c p / (R_s T); v1.0.0: p - rho_g R_s T / c
                v = -v * sim.wp.fluid.R_s * x[6] / 1e5
            out.setdefault(eq_id, []).append(v)
    return out


def row_params():
    params = []
    for well in row_vectors()['wells']:
        expected = {}
        for point in well['rows']:
            for eq_id, value in point:
                expected.setdefault(eq_id, []).append(value)
        params += [pytest.param(well, eq_id, values, id=f'{well["name"]} {eq_id}',
                                marks=marks(eq_id, ROWS_NOT_COMPARABLE, ROW_DEVIATIONS))
                   for eq_id, values in expected.items()]
    return params


@pytest.mark.parametrize('well, eq_id, expected', row_params())
def test_row_vector(well, eq_id, expected):
    sim = develop_simulator(well['params'])
    got = develop_rows(sim, np.array(well['x']))[eq_id]
    np.testing.assert_allclose(got, expected, rtol=ROW_REL, atol=0)


def test_row_vectors_follow_row_order():
    """The rows at each point are in DISC-6's order (specs/model/discretization.md)."""
    for well in row_vectors()['wells']:
        n = well['params']['n_cells']
        for i, point in enumerate(well['rows']):
            ids = [eq_id for eq_id, _ in point]
            first = ROW_IDS['first'] if i == 0 else ROW_IDS['cell'] + (ROW_IDS['last'] if i == n else [])
            assert ids == first + ROW_IDS['closure'], (well['name'], i, ids)
