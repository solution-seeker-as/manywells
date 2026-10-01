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
reproduce it and the entry has to go.

The row vectors are checked on develop's v1.0.0 configuration (manywells.configurations), which must
reproduce v1.0.0's rows as functions of the state (specs/model/discretization.md, Interface).
"""

import math

import casadi as ca
import numpy as np
import pytest

from manywells.ca_functions import ca_max_approx, ca_min_approx, ca_softmax
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import v1_fluid, v1_well
from manywells.discretization import PointState, build_system
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt import LiquidProperties, api_from_density, liquid_mix
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.pvt.gas import gas_density
from manywells.simulator import BoundaryConditions
from manywells.slip import SlipModel, classify_flow_regime

from .spec_parse import ROOT, row_vectors, vector_tables


def _develop_tables():
    """The adapters of the develop vectors: the generator's own calls (specs/tools/make_develop_vectors.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_develop_vectors', ROOT / 'specs' / 'tools' / 'make_develop_vectors.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {t.heading: t.fn for t in module.TABLES}


DEVELOP_ADAPTERS = _develop_tables()

REL, ABS = 1e-12, 1e-15      # component vectors: the same arithmetic, so agreement to rounding
ROW_REL = 1e-10              # row vectors: relative to the row's size, which is far from zero


def num(x):
    return float(np.asarray(ca.DM(x)).item()) if isinstance(x, (ca.DM, ca.SX)) else float(x)


def vec(x):
    return tuple(np.asarray(ca.DM(x)).ravel())


def choke(model, K_c, profile):
    return model(K_c=K_c, chk_profile=profile)


A = 0.0127  # Pipe cross-section (m²) of the choke adapters; it cancels in x_g


def choke_state(p_in, rho_g=50.0, rho_l=800.0, x_g=0.5, rho_m=None):
    """A wellhead state at p_in: with gas mass fraction x_g of the flow, or with mixture density rho_m."""
    if rho_m is not None:
        return PointState(p=p_in, v_g=1.0, v_l=1.0, alpha=0.0, rho_g=rho_g, rho_l=rho_m, T=300.0)  # rho_m = rho_l
    v_g = x_g / (1 - x_g) * rho_l / rho_g  # alpha = 1/2, v_l = 1
    return PointState(p=p_in, v_g=v_g, v_l=1.0, alpha=0.5, rho_g=rho_g, rho_l=rho_l, T=300.0)


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
        choke(SimpsonChokeModel, K_c, profile).mass_flow_rate(u, p_out, choke_state(p_in, rho_g, rho_l, x_g=x_g), A),
    'CHK-6': lambda K_c, profile, u, p_in, p_out, rho_m:
        choke(BernoulliChokeModel, K_c, profile).mass_flow_rate(u, p_out, choke_state(p_in, rho_m=rho_m), A),
    'CHK-7': lambda u: choke(SimpsonChokeModel, 1.0, 'linear').choke_opening(u),
    'CHK-8': lambda u: choke(SimpsonChokeModel, 1.0, 'sigmoid').choke_opening(u),
    'CHK-9': lambda u: choke(SimpsonChokeModel, 1.0, 'convex').choke_opening(u),
    'CHK-10': lambda u: choke(SimpsonChokeModel, 1.0, 'concave').choke_opening(u),
    'CHK-12': lambda p_in, p_out: bool(choke(SimpsonChokeModel, 1.0, 'linear').is_choked(p_in, p_out)),
    'INF-1': lambda w_l_max, p, p_r: Vogel(w_l_max).liquid_mass_flow_rate(p, p_r),
    'INF-2': lambda k_l, p, p_r: ProductivityIndex(k_l).liquid_mass_flow_rate(p, p_r),
    'INF-4': lambda f_g, w_l: v1_fluid(rho_l=850.0, R_s=420.0, cp_g=2225.0, cp_l=2000.0, f_g=f_g).reservoir_gas_rate(w_l),
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
    'INF-3': "develop's FixedFlowRate fixes the liquid rate only and takes the gas from the fluid model (INF-8)",
}

KNOWN_DEVIATIONS = {}


def marks(key, skips, deviations):
    if key in skips:
        return [pytest.mark.skip(reason=skips[key])]
    if key in deviations:
        return [pytest.mark.xfail(strict=True, reason=deviations[key])]
    return []


def vector_params():
    return [pytest.param(table, id=f'{table.source}: {table.heading}',
                         marks=marks(table.heading, NOT_ON_DEVELOP, KNOWN_DEVIATIONS) if table.source == 'v1.0.0' else [])
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
    adapters = ADAPTERS if table.source == 'v1.0.0' else DEVELOP_ADAPTERS
    if table.heading not in adapters:
        pytest.fail(f'no adapter for vector table {table.heading!r}: add one, or list it in NOT_ON_DEVELOP')
    for k, (inputs, expected) in enumerate(table.rows):
        check_outputs(table, k, expected, adapters[table.heading](**inputs))


def test_every_table_is_handled():
    tables = vector_tables()
    v1 = {t.heading for t in tables if t.source == 'v1.0.0'}
    develop = {t.heading for t in tables if t.source == 'develop'}
    assert v1 and develop, 'no vector tables found'
    assert set(KNOWN_DEVIATIONS) <= set(ADAPTERS)
    assert v1 <= set(ADAPTERS) | set(NOT_ON_DEVELOP), v1 - set(ADAPTERS) - set(NOT_ON_DEVELOP)
    assert develop == set(DEVELOP_ADAPTERS), develop ^ set(DEVELOP_ADAPTERS)  # the develop blocks are up to date


# Row vectors (specs/model/discretization.md), on develop's v1.0.0 configuration. develop's rows that generalize
# v1.0.0's carry their own IDs (DISC-11); in this configuration they are v1.0.0's rows as functions of the state.

ROW_IDS = {'first': ['INF-6', 'INF-7', 'THM-3'], 'cell': ['DISC-2', 'DISC-3', 'DISC-4', 'DISC-5'],
           'last': ['CHK-1'], 'closure': ['SLIP-1', 'PVT-GAS-1', 'PVT-MIX-1']}

DEVELOP_ROW = {'DISC-2': 'DISC-7', 'DISC-3': 'DISC-8', 'DISC-4': 'DISC-9', 'DISC-5': 'DISC-10'}

ROWS_NOT_COMPARABLE = {}

ROW_DEVIATIONS = {}


def v1_case(w):
    """develop's system in the v1.0.0 configuration, and the parameters, for a well of v1_rows.json."""
    inflow = Vogel(w['w_l_max']) if w['inflow'] == 'vogel' else ProductivityIndex(w['k_l'])
    model = SimpsonChokeModel if w['choke'] == 'simpson' else BernoulliChokeModel
    wp = v1_well(L=w['L'], D=w['D'], rho_l=w['rho_l'], R_s=w['R_s'], cp_g=w['cp_g'], cp_l=w['cp_l'], f_D=w['f_D'],
                 h=w['h'], f_g=w['f_g'], inflow=inflow, choke=model(K_c=w['K_c'], chk_profile=w['profile']),
                 n_cells=w['n_cells'])
    bc = BoundaryConditions(p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], u=w['u'], w_lg=w['w_lg'])
    system = build_system(wp)
    return system, system.params(bc)


def develop_rows(system, params, X):
    """develop's rows at every point, as {ID: [value per point]}, with develop's IDs."""
    r = np.asarray(system.residual(np.ravel(X), params)).ravel()
    out = {}
    for eq_id, v in zip(system.row_ids, r):
        out.setdefault(eq_id, []).append(float(v))
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
    system, params = v1_case(well['params'])
    got = develop_rows(system, params, np.array(well['x']))[DEVELOP_ROW.get(eq_id, eq_id)]
    np.testing.assert_allclose(got, expected, rtol=ROW_REL, atol=0)


def test_v1_configuration_rows_follow_the_v1_row_order():
    """In the v1.0.0 configuration, develop's rows are v1.0.0's, point by point in DISC-6's order."""
    for well in row_vectors()['wells']:
        system, _ = v1_case(well['params'])
        v1_ids = [eq_id for point in well['rows'] for eq_id, _ in point]
        assert list(system.row_ids) == [DEVELOP_ROW.get(i, i) for i in v1_ids]


def test_row_vectors_follow_row_order():
    """The rows at each point are in DISC-6's order (specs/model/discretization.md)."""
    for well in row_vectors()['wells']:
        n = well['params']['n_cells']
        for i, point in enumerate(well['rows']):
            ids = [eq_id for eq_id, _ in point]
            first = ROW_IDS['first'] if i == 0 else ROW_IDS['cell'] + (ROW_IDS['last'] if i == n else [])
            assert ids == first + ROW_IDS['closure'], (well['name'], i, ids)
