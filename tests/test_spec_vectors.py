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
reproduce v1.0.0's rows as functions of the state (specs/model/discretization.md, Interface), with each backend:
the CasADi system and the Rust core's rows.

Both backends implement each equation (specs/goals.md), so the Rust core is checked against the same component
vectors, through a test-only binding that calls its function by name on a well built from the table's inputs.
"""

import math
from dataclasses import replace

import casadi as ca
import numpy as np
import pytest

from manywells.ca_functions import ca_max_approx, ca_min_approx, ca_softmax
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import v1_fluid, v1_well
from manywells.discretization import PointState, build_system
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt import LiquidProperties, api_from_density, density_from_api, gas_density_from_sg, liquid_mix
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.pvt.fluid import FluidModel
from manywells.pvt.gas import gas_density
from manywells.simulator import BoundaryConditions
from manywells.slip import SlipModel, classify_flow_regime
from manywells.thermal import ThermalModel
from manywells.units import CF_BAR

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


# The Rust core's functions for the same tables. Each adapter builds a core well from the base well of the row
# vectors (W1) with the table's components, through the same conversion as the simulator (manywells.solvers.rust),
# and with every option the output does not depend on as in v1.0.0, so that a table needs only its own feature.

REGIMES = ('annular', 'slug-churn', 'bubbly')


def _base_well():
    return v1_inputs(row_vectors()['wells'][0]['params'])[0]


def rust(name, *args, **components):
    """The core's function `name` at args, on the base well with the given components."""
    from manywells.solvers.rust import core_well
    out = core_well(replace(_base_well(), **components))._component(name, [float(a) for a in args])
    return out[0] if len(out) == 1 else tuple(out)


def v1_choke(model, K_c, profile):
    return dict(choke=model(K_c=K_c, chk_profile=profile))


def choke_rates(s):
    """(w_g, w_l, alpha, rho_g, rho_l) of a wellhead state, the arguments of the core's choke rate."""
    return A * s.alpha * s.rho_g * s.v_g, A * (1 - s.alpha) * s.rho_l * s.v_l, s.alpha, s.rho_g, s.rho_l


def oil(api, sg_gas, gor=150.0, wlr=0.0, **kw):
    """Black oil as the develop tables' fluid, with an ideal gas and the 'liquid' surface tension unless given."""
    kw = {'ideal_gas': True, 'surface_tension_model': 'liquid', **kw}
    return FluidModel(rho_o=density_from_api(api), rho_g=gas_density_from_sg(sg_gas), gor=gor, wlr=wlr, **kw)


def dead_fluid(**kw):
    """A dead oil, ideal gas and 'liquid' surface tension, as in v1.0.0, with the given fields."""
    return FluidModel(**{'oil_model': 'dead_oil', 'ideal_gas': True, 'surface_tension_model': 'liquid', **kw})


def thermal(**on):
    return ThermalModel(**{'h': 0.0, 'frictional_heating': False, 'gravity_term': False, 'lift_gas_mixing': False,
                           **on})


def thermal_term(alpha, rho_g, v_g, rho_l, v_l, F, cos_incl, cp_g, cp_l, **on):
    """The temperature gradient with h = 0 and one term on, as make_develop_vectors.thermal_term."""
    return rust('temperature_gradient', 100.0, v_g, v_l, alpha, rho_g, rho_l, 350.0, 350.0, F, cos_incl,
                fluid=dead_fluid(cp_g=cp_g, cp_o=cp_l), thermal=thermal(**on),
                geometry=WellGeometry.vertical(length=1000.0, n_cells=10, D=0.1))


RUST_ADAPTERS = {
    # v1.0.0 tables
    'SMO-1': lambda x, y, eps: rust('max_approx', x, y, eps),
    'SMO-2': lambda x, y, eps: rust('min_approx', x, y, eps),
    'SMO-3': lambda y1, y2, y3: rust('softmax', y1, y2, y3),
    'CHK-2, CHK-3': lambda K_c, profile, u, p_in, p_out, rho, Phi:
        rust('choke_equation', u, p_in, p_out, rho, Phi, **v1_choke(SimpsonChokeModel, K_c, profile)),
    'CHK-4': lambda gamma: rust('critical_pressure_ratio', gamma),
    'CHK-5 (multiplier)': lambda x_g, rho_g, rho_l: rust('simpson_multiplier', x_g, rho_g, rho_l),
    'CHK-5 (rate)': lambda K_c, profile, u, p_in, p_out, x_g, rho_g, rho_l:
        rust('choke_rate', u, p_in, p_out, *choke_rates(choke_state(p_in, rho_g, rho_l, x_g=x_g)),
             **v1_choke(SimpsonChokeModel, K_c, profile)),
    'CHK-6': lambda K_c, profile, u, p_in, p_out, rho_m:
        rust('choke_rate', u, p_in, p_out, *choke_rates(choke_state(p_in, rho_m=rho_m)),
             **v1_choke(BernoulliChokeModel, K_c, profile)),
    'CHK-7': lambda u: rust('choke_opening', u, **v1_choke(SimpsonChokeModel, 1.0, 'linear')),
    'CHK-8': lambda u: rust('choke_opening', u, **v1_choke(SimpsonChokeModel, 1.0, 'sigmoid')),
    'CHK-9': lambda u: rust('choke_opening', u, **v1_choke(SimpsonChokeModel, 1.0, 'convex')),
    'CHK-10': lambda u: rust('choke_opening', u, **v1_choke(SimpsonChokeModel, 1.0, 'concave')),
    'CHK-12': lambda p_in, p_out: bool(rust('is_choked', p_in, p_out)),
    'INF-1': lambda w_l_max, p, p_r: rust('liquid_rate', p, p_r, inflow=Vogel(w_l_max)),
    'INF-2': lambda k_l, p, p_r: rust('liquid_rate', p, p_r, inflow=ProductivityIndex(k_l)),
    'INF-4': lambda f_g, w_l: rust('reservoir_gas_rate', w_l,
                                   fluid=v1_fluid(rho_l=850.0, R_s=420.0, cp_g=2225.0, cp_l=2000.0, f_g=f_g)),
    'SLIP-2, SLIP-3': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, D, **_:
        rust('slip_parameters', v_g, v_l, alpha, rho_g, rho_l, sigma, D),
    'SLIP-4': lambda rho_g, rho_l, sigma, **_: rust('harmathy_rise_velocity', rho_g, rho_l, sigma),
    'SLIP-5': lambda rho_g, rho_l, D: rust('taylor_rise_velocity', rho_g, rho_l, D),
    'SLIP-6, SLIP-7': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, **_:
        rust('regime_probabilities', v_g, v_l, alpha, rho_g, rho_l, sigma),
    'SLIP-8': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, **_:
        REGIMES[int(rust('regime', v_g, v_l, alpha, rho_g, rho_l, sigma))],
    'PVT-GAS-1': lambda p, T, R_s: rust('ideal_gas_density', p, T, R_s),
    'PVT-OIL-2': lambda rho: rust('api_from_density', rho),
    'PVT-OIL-3': lambda rho, T: rust('dead_oil_surface_tension', rho, T),
    # develop tables
    'THM-4': lambda tvd_frac, T_r, T_s: rust('ambient_temperature', tvd_frac, T_r, T_s),
    'THM-5': lambda w_res, w_lg, T_r, T_lg, f_g, cp_g, cp_l:
        rust('inflow_temperature', w_res, w_lg, T_r, T_lg, thermal=thermal(lift_gas_mixing=True),
             fluid=v1_fluid(rho_l=850.0, R_s=420.0, cp_g=cp_g, cp_l=cp_l, f_g=f_g)),
    'THM-6': lambda F, **kw: thermal_term(F=F, cos_incl=1.0, frictional_heating=True, **kw),
    'THM-7': lambda cos_incl, **kw: -thermal_term(F=0.0, cos_incl=cos_incl, gravity_term=True, **kw),
    'SLIP-10, SLIP-11': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl:
        rust('slip_parameters', v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl),
    'SLIP-11 (classifier)': lambda v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl:
        rust('regime_probabilities', v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl),
    'FRIC-3': lambda p, T, alpha, v_g, v_l, rho_g, rho_l, D, roughness:
        rust('friction', p, v_g, v_l, alpha, rho_g, rho_l, T, fluid=replace(FluidModel(), ideal_gas=True,
                                                                             surface_tension_model='liquid'),
             friction=RoughnessFriction(roughness=roughness), geometry=WellGeometry.vertical(1000.0, 10, D=D)),
    'FRIC-4': lambda Re, eps_D: rust('chen_friction_factor', Re, eps_D),
    'FRIC-5': lambda Re, eps_D: rust('haaland_friction_factor', Re, eps_D),
    'FRIC-6': lambda Re, eps_D: rust('friction_factor_of_re', Re, eps_D, friction=RoughnessFriction()),
    'PVT-GAS-3': lambda p, T, rho_g_sc: rust('gas_density', p, T, fluid=dead_fluid(rho_g=rho_g_sc, ideal_gas=False)),
    'PVT-GAS-4, PVT-GAS-5': lambda p, T, sg_gas: (
        rust('sutton_pseudo_critical', sg_gas)[0] / CF_BAR, rust('sutton_pseudo_critical', sg_gas)[1],
        rust('z_factor', p, T, fluid=dead_fluid(rho_g=gas_density_from_sg(sg_gas), ideal_gas=False))),
    'PVT-GAS-6': lambda rho_g_sc: rust('gas_parameters', fluid=dead_fluid(rho_g=rho_g_sc)),
    'PVT-GAS-7': lambda T, rho_g, M_g: rust('gas_viscosity', T, rho_g, M_g),
    'PVT-OIL-5': lambda api, sg_gas, p_sep, T_sep: rust('separator_gravity', api, sg_gas, p_sep, T_sep),
    'PVT-OIL-6, PVT-OIL-8': lambda api, sg_gas, p, T: (rust('rs', p, T, fluid=oil(api, sg_gas)),
                                                       rust('bo', p, T, fluid=oil(api, sg_gas))),
    'PVT-OIL-7': lambda api, sg_gas, p_bubble, p, T: rust('rs', p, T, fluid=oil(api, sg_gas, p_bubble=p_bubble)),
    'PVT-OIL-9': lambda api, sg_gas, p, T: rust('liquid_density', p, T, fluid=oil(api, sg_gas)),
    'PVT-OIL-10': lambda api, T: rust('dead_oil_viscosity', api, T),
    'PVT-OIL-11': lambda mu_od, R_so_scf: rust('live_oil_viscosity', mu_od, R_so_scf),
    'PVT-OIL-12': lambda sigma_od, R_so_scf: rust('live_oil_surface_tension', sigma_od, R_so_scf),
    'PVT-OIL-13': lambda api, sg_gas, gor, wlr, p, T, w_res, w_lg:
        rust('phase_rates', p, T, w_res, w_lg, fluid=oil(api, sg_gas, gor, wlr)),
    'PVT-WAT-3': lambda T: rust('water_viscosity', T),
    'PVT-MIX-6': lambda api, sg_gas, gor, wlr, p, T: rust('liquid_density', p, T, fluid=oil(api, sg_gas, gor, wlr)),
    'PVT-MIX-7': lambda api, sg_gas, p, T, rho_l:
        rust('surface_tension', p, T, rho_l, fluid=oil(api, sg_gas, surface_tension_model='oil')),
    'PVT-MIX-8': lambda api, sg_gas, wlr, p, T: rust('liquid_viscosity', p, T, fluid=oil(api, sg_gas, wlr=wlr)),
    'PVT-MIX-9': lambda mu_l, mu_g, alpha, rho_l, rho_g: rust('mixture_viscosity', mu_l, mu_g, alpha, rho_l, rho_g),
    'PVT-MIX-10': lambda rho_o, rho_g_sc, rho_w, gor, wlr, cp_o, cp_w:
        rust('fluid_parameters', fluid=dead_fluid(rho_o=rho_o, rho_g=rho_g_sc, rho_w=rho_w, gor=gor, wlr=wlr,
                                                  cp_o=cp_o, cp_w=cp_w)),
}

# Equations no row of the system uses: the sampler's liquid mixing, conversions between the inputs, and water's
# formation volume factor; the core has no function for them (specs/features/015-rust-develop-model.md)
RUST_NOT_IN_CORE = {
    'PVT-GAS-2': 'a conversion between inputs (R_s and the gas density at standard conditions), in Python only',
    'PVT-MIX-2, PVT-MIX-3, PVT-MIX-4': "the sampler's liquid mixing (SMP-12), in Python only",
    'PVT-WAT-2': 'water is incompressible in every configuration: no row uses its formation volume factor',
    'INF-3': NOT_ON_DEVELOP['INF-3'],
}

# Step 9 ports develop's model to the core one feature spec at a time: the tables whose functions it has not got yet
RUST_NOT_PORTED = {
    'SMO-2': '007', 'THM-5': '010', 'SLIP-10, SLIP-11': '002',
    'SLIP-11 (classifier)': '002', 'FRIC-3': '004', 'FRIC-4': '004', 'FRIC-5': '004', 'FRIC-6': '004',
    'PVT-GAS-3': '006', 'PVT-GAS-4, PVT-GAS-5': '006', 'PVT-GAS-7': '004', 'PVT-OIL-5': '007',
    'PVT-OIL-6, PVT-OIL-8': '007', 'PVT-OIL-7': '007', 'PVT-OIL-9': '007', 'PVT-OIL-10': '004', 'PVT-OIL-11': '004',
    'PVT-OIL-12': '003', 'PVT-OIL-13': '008', 'PVT-WAT-3': '004', 'PVT-MIX-6': '007', 'PVT-MIX-7': '003',
    'PVT-MIX-8': '004', 'PVT-MIX-9': '004',
}


def rust_vector_params():
    out = []
    for table in vector_tables():
        if table.heading in RUST_NOT_IN_CORE:
            continue
        marks = ([pytest.mark.xfail(strict=True, raises=ValueError,
                                    reason=f'not in the core until feature {RUST_NOT_PORTED[table.heading]}')]
                 if table.heading in RUST_NOT_PORTED else [])
        out.append(pytest.param(table, id=f'{table.source}: {table.heading}', marks=marks))
    return out


@pytest.mark.parametrize('table', rust_vector_params())
def test_rust_component_vectors(table):
    for k, (inputs, expected) in enumerate(table.rows):
        check_outputs(table, k, expected, RUST_ADAPTERS[table.heading](**inputs))


def test_every_table_has_a_rust_adapter():
    headings = {t.heading for t in vector_tables()}
    assert not set(RUST_ADAPTERS) & set(RUST_NOT_IN_CORE)
    assert headings == set(RUST_ADAPTERS) | set(RUST_NOT_IN_CORE), headings ^ (set(RUST_ADAPTERS) | set(RUST_NOT_IN_CORE))
    assert set(RUST_NOT_PORTED) <= set(RUST_ADAPTERS)


# Row vectors (specs/model/discretization.md), on develop's v1.0.0 configuration. develop's rows that generalize
# v1.0.0's carry their own IDs (DISC-11); in this configuration they are v1.0.0's rows as functions of the state.

ROW_IDS = {'first': ['INF-6', 'INF-7', 'THM-3'], 'cell': ['DISC-2', 'DISC-3', 'DISC-4', 'DISC-5'],
           'last': ['CHK-1'], 'closure': ['SLIP-1', 'PVT-GAS-1', 'PVT-MIX-1']}

DEVELOP_ROW = {'DISC-2': 'DISC-7', 'DISC-3': 'DISC-8', 'DISC-4': 'DISC-9', 'DISC-5': 'DISC-10'}

ROWS_NOT_COMPARABLE = {}

ROW_DEVIATIONS = {}


def v1_inputs(w):
    """develop's inputs in the v1.0.0 configuration for a well of v1_rows.json."""
    inflow = Vogel(w['w_l_max']) if w['inflow'] == 'vogel' else ProductivityIndex(w['k_l'])
    model = SimpsonChokeModel if w['choke'] == 'simpson' else BernoulliChokeModel
    wp = v1_well(L=w['L'], D=w['D'], rho_l=w['rho_l'], R_s=w['R_s'], cp_g=w['cp_g'], cp_l=w['cp_l'], f_D=w['f_D'],
                 h=w['h'], f_g=w['f_g'], inflow=inflow, choke=model(K_c=w['K_c'], chk_profile=w['profile']),
                 n_cells=w['n_cells'])
    bc = BoundaryConditions(p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], u=w['u'], w_lg=w['w_lg'])
    return wp, bc


def v1_case(w):
    """develop's system in the v1.0.0 configuration, and the parameters, for a well of v1_rows.json."""
    wp, bc = v1_inputs(w)
    system = build_system(wp)
    return system, system.params(bc)


def rust_rows(w, X):
    """The Rust core's rows at every point, as {ID: [value per point]}, with develop's IDs (DISC-11)."""
    from manywells.solvers.rust import RustRootFinder
    wp, bc = v1_inputs(w)
    ids, values = RustRootFinder(wp).rows(bc, X)
    out = {}
    for eq_id, v in zip(ids, values):
        out.setdefault(eq_id, []).append(float(v))
    return out


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


@pytest.mark.parametrize('well, eq_id, expected', row_params())
def test_rust_row_vector(well, eq_id, expected):
    got = rust_rows(well['params'], np.array(well['x']))[DEVELOP_ROW.get(eq_id, eq_id)]
    np.testing.assert_allclose(got, expected, rtol=ROW_REL, atol=0)


def test_rust_rows_follow_the_v1_row_order():
    """In the v1.0.0 configuration, the Rust core's rows are v1.0.0's, point by point in DISC-6's order, with the IDs
    of the CasADi system (DISC-11)."""
    from manywells.solvers.rust import RustRootFinder
    for well in row_vectors()['wells']:
        wp, bc = v1_inputs(well['params'])
        ids, _ = RustRootFinder(wp).rows(bc, np.array(well['x']))
        v1_ids = [eq_id for point in well['rows'] for eq_id, _ in point]
        assert list(ids) == [DEVELOP_ROW.get(i, i) for i in v1_ids] == list(build_system(wp).row_ids)


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
