"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the v2 sampler's reservoir closed forms (manywells.sampling.v2.reservoir; specs/sampling.md, SMP-70 to
SMP-73), the inflow's coefficients at a reservoir state with the Forchheimer fit (inflow; SMP-117) and the map from a
drawn well to the model's inputs in the v2.0.0 configuration (mapping; SMP-111). One slow test solves a few wells.
"""

import math
import pathlib

import numpy as np
import pytest

from manywells.choke import SimpsonChokeModel
from manywells.configurations import V2, V2_SLIP, check, differences
from manywells.friction import RoughnessFriction, chen_friction_factor
from manywells.inflow import Forchheimer, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.table import PVTTable
from manywells.sampling.v2.draws import _P_ATM, _black_oil
from manywells.sampling.v2.inflow import (
    _COEF_KEYS,
    _pseudo_pressure,
    _wellstream_factor,
    _fit_pressure_squared,
    _gas_inflow,
    _inflow_at,
)
from manywells.sampling.v2.mapping import _N_CELLS, WellBuilder
from manywells.sampling.v2.reservoir import (
    _GOR_NODES,
    _R_W,
    _drain_index,
    _forchheimer,
    _gas_rate,
    _j_radial,
    _joshi_ratio,
    _liner_gas,
    _liner_liquid,
    _nondarcy,
    _producing_gor,
    _slant_skin,
)
from manywells.simulator import NoOperatingPoint, SSDFSimulator
from manywells.units import P_REF, T_REF

from .sampling_v2_wells import drawn_wells, is_oil, nominal_inputs

FT, PSI, STB, MSCF = 0.3048, 0.0689476, 0.158987, 28.3168  # m, bar, m3 and Sm3 per field unit
TABLE_SET = pathlib.Path(__file__).parent / 'data' / 'pvt' / 'synthetic_eos_oil_gor.csv'


@pytest.fixture(scope='session')
def wells():
    return drawn_wells()


@pytest.fixture(scope='session')
def gas_wells(wells):
    return [w for w, _ in wells if not is_oil(w)]


# --- The producing GOR (SMP-70) --------------------------------------------------------------------------------------
def test_producing_gor_of_an_oil(wells):
    """R_si at and above the saturation pressure; below it the material balance's curve: a dip while the freed gas
    is below its critical saturation, then a rise; and a more mobile gas (the tail's blend) gives the higher GOR."""
    w = next(w for w, _ in wells if w['fluid_class'] == 'black_oil' and w['coverage_tail'] == 'core')
    p_sat, R_i = w['p_sat'], w['R_i']
    assert _producing_gor(w, p_sat) == _producing_gor(w, 1.3 * p_sat) == R_i
    factor = [_producing_gor(w, x * p_sat) / R_i for x in (0.999, 0.95, 0.9, 0.5, 0.3)]
    assert factor[0] == pytest.approx(1, abs=5e-3)  # Continuous at the saturation pressure
    assert factor[2] < factor[1] < factor[0] < 1 and factor[4] > factor[3] > 1.5
    mobile = dict(w, w_kgko=1.0)  # Table 10.2's k_g/k_o
    assert _producing_gor(mobile, 0.5 * p_sat) > _producing_gor(w, 0.5 * p_sat)
    assert _producing_gor(mobile, p_sat) == R_i
    # The curve is linear in p_r / p_sat between its nodes, from 0.05 to 1
    assert (
        _GOR_NODES[0] == 0.05 and _GOR_NODES[-1] == 1.0 and len(_GOR_NODES) == 381 and np.all(np.diff(_GOR_NODES) > 0)
    )
    lo, hi = _GOR_NODES[200] * p_sat, _GOR_NODES[201] * p_sat
    middle = 0.5 * (_producing_gor(w, lo) + _producing_gor(w, hi))
    assert _producing_gor(w, 0.5 * (lo + hi)) == pytest.approx(middle, rel=1e-12)


def test_producing_gor_of_gas_classes(gas_wells):
    cond = next(w for w in gas_wells if w['fluid_class'] == 'gas_condensate')
    p_sat, R_i, c_k = cond['p_sat'], cond['R_i'], cond['c_k']
    assert _producing_gor(cond, p_sat) == _producing_gor(cond, 1.2 * p_sat) == R_i
    assert _producing_gor(cond, 0.6 * p_sat) == pytest.approx(R_i * (1 + 0.4 * c_k))
    wet = next(w for w in gas_wells if w['fluid_class'] == 'wet_gas')
    assert _producing_gor(wet, 0.3 * wet['p_r']) == wet['R_i']


# --- Productivity (SMP-72) and the radial gas coefficients (SMP-71, SMP-73) ------------------------------------------
def test_j_radial_in_field_units():
    """Darcy's radial flow in SI against the oilfield form, J = k h / (141.2 mu B (ln(r_e/r_w) - 0.75 + s)) STB/(d psi)
    with k in md, h in ft and mu in cp."""
    J = _j_radial(100.0, 10.0, 1e-3, 1.2, 7.25, 0.75)
    assert J == pytest.approx(100 * (10 / FT) / (141.2 * 1.0 * 1.2 * 8.0) * STB / PSI, rel=1e-3)
    assert _j_radial(100.0, 10.0, 2e-3, 1.2, 7.25, 0.75) == pytest.approx(J / 2)


def test_joshi_ratio_against_a_worked_example():
    """Economides et al. (1994), Example 2-10: a 2000 ft drain in 53 ft of pay with I_ani = 3, k_H = 8.2 md,
    r_eH = 2980 ft, mu = 1.7 cp and B = 1.1 gives 1634 STB/d at a drawdown of 2151 psi. The example's wellbore radius
    is 0.328 ft where the sampler's is 0.354 ft, which changes the rate by less than 1%."""
    L_d, h, k, mu, B = 2000 * FT, 53 * FT, 8.2, 1.7e-3, 1.1
    ln_re = math.log(2980 * FT / _R_W)
    ratio = _joshi_ratio(L_d, h, 3.0, 0.0, ln_re)
    q = ratio * _j_radial(k, h, mu, B, ln_re - 0.75, 0.0) * (5651 - 3500) * PSI / STB
    assert q == pytest.approx(1634, rel=0.01)
    assert _joshi_ratio(L_d, h, 3.0, 5.0, ln_re) > ratio  # Skin hurts a drain less than a vertical well
    # A long drain's drainage length grows with it, r_eH = max(r_e, L_d / 1.6): its ellipse keeps its shape, and in
    # thin pay the index tends to 2 pi k h / (mu B ln 2.53)
    assert _joshi_ratio(2400.0, h, 3.0, 0.0, 7.5) > _joshi_ratio(1200.0, h, 3.0, 0.0, 7.5) > 1
    a = 0.5 * math.sqrt(0.5 + math.sqrt(0.25 + 1.25**4))  # The ellipse's half-axis over L_d, at 2 r_eH / L_d = 1.25
    limit = (7.5 - 0.75) / math.log((a + math.sqrt(a * a - 0.25)) / 0.5)
    assert _joshi_ratio(2000.0, 1e-6, 1.0, 0.0, 7.5) == pytest.approx(limit, rel=1e-6)
    assert math.log((a + math.sqrt(a * a - 0.25)) / 0.5) == pytest.approx(math.log(2.53), rel=1e-3)


def test_slant_skin_against_economides_table_5_1():
    """SMP-72 (D-113): Cinco-Ley et al.'s skin of a slanted well, fully completed, at h_D = 100, against Economides et
    al.'s (1994) Table 5-1, p. 91, within 3%; it grows with the angle and the pay, and anisotropy lessens it."""
    table = {15: -0.128, 30: -0.517, 45: -1.178, 60: -2.149, 75: -3.577}
    for theta, s in table.items():
        assert _slant_skin(theta, 100 * _R_W, 1.0) == pytest.approx(s, rel=0.0301)
    assert _slant_skin(0.0, 15.0, 2.0) == 0.0
    assert _slant_skin(45.0, 50.0, 1.0) < _slant_skin(45.0, 10.0, 1.0) < _slant_skin(30.0, 10.0, 1.0) < 0
    assert _slant_skin(45.0, 10.0, 3.0) > _slant_skin(45.0, 10.0 * 3, 1.0)  # The pay's thickness in h_D, not the angle
    t, h_D = math.degrees(math.atan(math.tan(math.radians(60.0)) / 2.0)), 10.0 * 2.0 / _R_W
    assert _slant_skin(60.0, 10.0, 2.0) == pytest.approx(
        -((t / 41) ** 2.06) - (t / 56) ** 1.865 * math.log10(h_D / 100)
    )


def test_liner_friction():
    """The liner's coefficients against the pipe-flow forms: Chen's factor (FRIC-4) times 1.5 for the perforations,
    f rho v2 / (2 D) for a liquid, with the inflow's acceleration, and d(p2)/dx = f G2 Z R T / (D M) for a gas."""
    D, rho, mu, B, eps, q = 0.1243, 780.0, 1.2e-3, 1.25, 3e-5, 2500.0
    v = q * B / 86400 / (math.pi * D**2 / 4)  # m/s
    f_D = 1.5 * float(chen_friction_factor(rho * v * D / mu, eps / D))
    c = _liner_liquid(D, rho, mu, B, eps)
    assert float(c(q, 1e12)) * q**2 * 1e5 == pytest.approx(f_D * rho * v**2 / (2 * D), rel=1e-6)  # Pa/m
    assert float(c(q, 500.0) - c(q, 1e12)) * q**2 * 1e5 * 500 / 4 == pytest.approx(2 / 3 * rho * v**2, rel=1e-6)
    sg, Z, T, mu_g, q_g = 0.65, 0.9, 380.0, 2e-5, 5e5
    M = sg * 0.02897
    G = q_g / 86400 * P_REF * M / (8.314 * T_REF) / (math.pi * D**2 / 4)  # kg/(m2 s)
    f_D = 1.5 * float(chen_friction_factor(G * D / mu_g, eps / D))
    c_g = _liner_gas(D, sg, Z, T, mu_g, eps)
    assert float(c_g(q_g)) * q_g**2 * 1e10 == pytest.approx(f_D * G**2 * Z * 8.314 * T / (D * M), rel=1e-6)  # Pa2/m


def test_drain_index():
    """Dikken's efficiency in closed form, eta = (1 + Pi / 1.5)^(-1/3) with Pi = c J_H^2 u L_d: 1 without friction,
    1 - Pi / 4.5 for a little; and at a cap on the rate, the index at the drawdown that gives the capped rate."""
    J_H, L_d, u = 40.0, 1500.0, 30.0

    def const(c):
        return lambda q, L: c

    assert _drain_index(J_H, L_d, const(0.0), u, 1e9) == pytest.approx(J_H, rel=1e-14)
    for c in (1e-12, 1e-9, 1e-7, 1e-5):
        Pi = c * J_H**2 * u * L_d
        assert _drain_index(J_H, L_d, const(c), u, 1e9) == pytest.approx(J_H * (1 + Pi / 1.5) ** (-1 / 3), rel=1e-12)
    Pi = 1e-12 * J_H**2 * u * L_d
    assert _drain_index(J_H, L_d, const(1e-12), u, 1e9) == pytest.approx(J_H * (1 - Pi / 4.5), rel=1e-8)
    c = 1e-7
    free = _drain_index(J_H, L_d, const(c), u, 1e9) * u  # The heel's rate at the drawdown u
    capped = _drain_index(J_H, L_d, const(c), u, free / 2)
    u_cap = free / 2 / capped  # The drawdown at the capped rate
    assert u_cap < u and capped == pytest.approx(J_H * (1 + c * J_H**2 * L_d / 1.5 * u_cap) ** (-1 / 3), rel=1e-9)
    assert free / u < capped < J_H  # Less drawdown, less friction


def test_radial_forchheimer_in_field_units():
    """Economides et al. (1994), Eqs. 4-32 and 4-33: p_r^2 - p_wf^2 = 1424 mu Z T / (k h) (ln(0.472 r_e/r_w) + s +
    D q) q, with q in Mscf/d, pressures in psi, T in R, k in md, h in ft and mu in cp. Its standard conditions are
    60 F and 14.7 psia, a third of a percent from the model's."""
    k, h, mu, Z, T, ln_term, s, D = 5.0, 20.0, 2.2e-5, 0.92, 390.0, 7.4, 2.0, 4e-4
    a, b = _forchheimer(k, h, mu, Z, T, ln_term, s, D)
    a_field = 1424 * (mu * 1e3) * Z * (T * 1.8) * (ln_term + s) / (k * h / FT)  # psi2 per Mscf/d
    assert a == pytest.approx(a_field * PSI**2 / MSCF, rel=5e-3)
    assert b == pytest.approx(a * D / MSCF / (ln_term + s), rel=1e-12)  # The non-Darcy flow as a skin D q
    assert 86400 / MSCF == pytest.approx(3051, rel=1e-4)  # 1 d/Mscf in s/Sm3
    q = _gas_rate(a, b, 150.0**2 - 90.0**2)
    assert q > 0 and a * q + b * q**2 == pytest.approx(150.0**2 - 90.0**2, rel=1e-12)


def test_nondarcy_estimate_in_field_units():
    """Economides et al. (1994), Eq. 4-34, from SI inputs: D = 6e-5 gamma_g k^-0.1 h / (mu r_w h_perf^2) d/Mscf with
    k in md, the lengths in ft and mu in cp."""
    sg, k, h, mu, h_perf = 0.65, 10.0, 30 * FT, 2e-5, 12 * FT
    assert _nondarcy(sg, k, h, mu, h_perf) == pytest.approx(6e-5 * sg * k**-0.1 * 30 / (0.02 * (_R_W / FT) * 12**2))
    assert _nondarcy(sg, k, h, mu, h) == pytest.approx(_nondarcy(sg, k, h, mu, h_perf) * (12 / 30) ** 2)


# --- The Forchheimer fit (SMP-117) -----------------------------------------------------------------------------------
def test_fit_reproduces_a_pressure_squared_law():
    p_r, a, b = 240.0, 3e-3, 2e-9
    a_fit, b_fit = _fit_pressure_squared(lambda p_wf: _gas_rate(a, b, p_r**2 - p_wf**2), p_r)
    assert (a_fit, b_fit) == pytest.approx((a, b), rel=1e-9)
    a_fit, b_fit = _fit_pressure_squared(lambda p_wf: (p_r**2 - p_wf**2) / a, p_r)  # Darcy flow alone
    assert a_fit == pytest.approx(a, rel=1e-9) and abs(b_fit) < 1e-9 * a / _gas_rate(a, b, p_r**2)


def test_fit_is_least_squares_with_a_floor_on_b():
    """Least squares of (p_r^2 - p_wf^2)/q against q at the seven points; and where that line's slope is negative,
    b = 0 and a is the mean of (p_r^2 - p_wf^2)/q."""
    p_r = 300.0
    points = [f * p_r for f in (0.95, 0.9, 0.8, 0.6, 0.4, 0.2)] + [_P_ATM]
    dp2 = np.array([p_r**2 - p**2 for p in points])
    for n, steeper in ((0.8, True), (1.25, False)):  # q ~ (p_r^2 - p_wf^2)^n: a curve that no a and b follow
        q = 1e3 * (dp2 / 1e4) ** n
        a, b = _fit_pressure_squared(lambda p_wf: 1e3 * ((p_r**2 - p_wf**2) / 1e4) ** n, p_r)
        y = dp2 / q
        if steeper:  # The free fit: its residuals are orthogonal to 1 and q
            residual = y - a - b * q
            assert b > 0 and abs(residual.sum()) < 1e-9 * y.sum() and abs(residual @ q) < 1e-9 * (y @ q)
        else:
            assert b == 0.0 and a == pytest.approx(y.mean(), rel=1e-12)


def reservoir_gas(w, R_go):
    """A black-oil gas well's reservoir gas (SMP-117): the black-oil gas at the wellstream's density."""
    c = _wellstream_factor(w['api'], R_go)
    rho_ws = (w['rho_g'] + w['rho_o'] / R_go) / c
    return BlackOil(rho_o=w['rho_o'], rho_g=rho_ws, gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep']), c


def radial_coefficients(w, gas, p_r):
    """SMP-73's a and b at a state, with the well's geometry factors."""
    T_r = w['T_r']
    Z, mu_g = float(gas.z_factor(p_r, T_r)), float(gas.gas_viscosity(p_r, T_r, gas.gas_density(p_r, T_r)))
    a, b = _forchheimer(w['perm'], w['h_net'], mu_g, Z, T_r, w['ln_re_rw'] - 0.75, w['skin'], w['D_nondarcy'])
    return a * w['a_ratio'], b * w['b_ratio']


def test_wellstream_factor():
    """Whitson and Brule (2000), Eqs. 3.73b and 3.73c: the condensate's gas equivalent, in Sm3/Sm3."""
    api, R_go = 50.0, 2000.0
    gamma_o, M_o = 141.5 / (131.5 + api), 6084 / (api - 5.9)
    assert _wellstream_factor(api, R_go) == pytest.approx(1 + 0.178108 * 133_000 * gamma_o / M_o / R_go)
    assert 1.05 < _wellstream_factor(api, R_go) < 1.1  # A rich condensate's liquid adds 5% to 10% to the gas


def test_gas_inflow_at_low_pressure_is_the_pressure_squared_law(gas_wells):
    """Where mu_g Z is nearly constant, at low pressure, the pseudo-pressure's inflow is SMP-73's law in the
    pressure squared, so the fitted model gives its rates."""
    p_r = 15.0
    for w in gas_wells[:15]:
        pvt = _black_oil(w)
        fit = Forchheimer(**{k[12:]: v for k, v in _gas_inflow(w, pvt, p_r, w['R_go']).items()})
        gas, c = reservoir_gas(w, w['R_go'])
        a, b = radial_coefficients(w, gas, p_r)
        law = Forchheimer.from_gas_rate(c * a, c * c * b, w['rho_g'])  # For the separator gas, in the model's units
        for f in (0.95, 0.8, 0.5, 0.2):
            rate = float(fit.gas_mass_flow_rate(f * p_r, p_r))
            assert rate == pytest.approx(float(law.gas_mass_flow_rate(f * p_r, p_r)), rel=0.02)


def test_gas_inflow_follows_the_pseudo_pressure(gas_wells):
    """The fitted model's rates against the pseudo-pressure's inflow, from a drawdown of 20% to the absolute open
    flow: close where the fit keeps b, and within the known limit of the form where b = 0 (validity.md, note 12)."""
    kept, floored = [], []
    for w in gas_wells[:30]:
        p_r, T_r, R_go, pvt = w['p_r'], w['T_r'], w['R_go'], _black_oil(w)
        coef = _gas_inflow(w, pvt, p_r, R_go)
        assert math.isfinite(coef['forchheimer_a']) and coef['forchheimer_a'] > 0 and coef['forchheimer_b'] >= 0
        fit = Forchheimer(a=coef['forchheimer_a'], b=coef['forchheimer_b'])
        gas, c = reservoir_gas(w, R_go)
        a, b = radial_coefficients(w, gas, p_r)
        p, m, muz_r = _pseudo_pressure(gas, p_r, T_r)
        assert len(p) == 200 and p[0] == 1.0 and p[-1] == p_r and m[0] == 0.0
        errors = []
        for p_wf in [f * p_r for f in (0.8, 0.6, 0.4, 0.2)] + [_P_ATM]:
            q_ws = _gas_rate(a, b, muz_r * (m[-1] - np.interp(p_wf, p, m)))  # The wellstream, Sm3/d
            q_sep = float(fit.gas_mass_flow_rate(p_wf, p_r)) * 86400 / w['rho_g']  # The model's separator gas
            errors.append(abs(c * q_sep / q_ws - 1))
        (floored if coef['forchheimer_b'] == 0 else kept).append(max(errors))
    assert len(kept) >= 5 and len(floored) >= 5
    assert max(kept) < 0.15 and np.median(kept) < 0.02
    assert max(floored) < 0.5 and np.median(floored) < 0.15


def test_gas_inflow_scales_with_the_radial_coefficients(gas_wells):
    """Half the k h with twice the non-Darcy coefficient doubles the radial a and quadruples b, so every rate
    halves, and the fitted a doubles and b quadruples; with the black oil's gas and with a table's (PVT-TAB-4)."""
    w = dict(next(w for w in gas_wells if w['fluid_class'] == 'gas_condensate'), T_r=360.0)
    weaker = dict(w, perm=w['perm'] / 2, D_nondarcy=2 * w['D_nondarcy'])
    table = PVTTable.read_csv(TABLE_SET)
    for pvt, p_r, R_go in ((_black_oil(w), 0.5 * w['p_r'], w['R_go']), (table, 150.0, 120.0)):
        base, half = _gas_inflow(w, pvt, p_r, R_go), _gas_inflow(weaker, pvt, p_r, R_go)
        assert half['forchheimer_a'] == pytest.approx(2 * base['forchheimer_a'], rel=1e-9)
        assert half['forchheimer_b'] == pytest.approx(4 * base['forchheimer_b'], rel=1e-9, abs=1e-300)
        assert base['forchheimer_a'] > 0


def test_inflow_at_the_drawn_state(wells):
    """An oil's coefficient at its drawn state is the draw's, Vogel's w_l_max or the index k_l by its inflow model;
    a gas well's are the fit's; the others are NaN."""
    seen = set()
    for w, _ in wells[:40]:
        coef = _inflow_at(w, _black_oil(w), w['p_r'], w['R_go'], w['wlr'])
        assert tuple(coef) == _COEF_KEYS
        used = {'vogel': ['w_l_max'], 'productivity_index': ['k_l'], 'forchheimer': ['forchheimer_a', 'forchheimer_b']}
        assert [key for key in _COEF_KEYS if not math.isnan(coef[key])] == used[w['inflow']]
        if is_oil(w):
            key = used[w['inflow']][0]
            assert coef[key] == pytest.approx(w[key], rel=1e-9)
        else:
            fit = _gas_inflow(w, _black_oil(w), w['p_r'], w['R_go'])
            assert {key: coef[key] for key in fit} == fit and set(fit) == {'forchheimer_a', 'forchheimer_b'}
        seen.add(w['inflow'])
    assert seen == {'vogel', 'productivity_index', 'forchheimer'}


def test_inflow_at_with_a_table(wells):
    """A volatile oil's index takes its table's oil viscosity and B_o at the state's GOR (SMP-72)."""
    w = dict(next(w for w, _ in wells if w['fluid_class'] == 'volatile_oil' and w['inflow'] == 'vogel'), T_r=360.0)
    table = PVTTable.read_csv(TABLE_SET)
    p_r, R_go, wlr = 180.0, 110.0, 0.2
    at = table.at_gor(R_go)
    mu_o, B_o = float(at.oil_viscosity(p_r, 360.0)), float(at.bo(p_r, 360.0))
    J = _j_radial(w['perm'], w['h_net'], mu_o, B_o, w['ln_re_rw'] - 0.75, w['skin'])
    k_l = J * w['f_geo'] * ((1 - wlr) * table.rho_o + wlr * w['rho_w']) / 86400
    assert _inflow_at(w, table, p_r, R_go, wlr)['w_l_max'] == pytest.approx(k_l * p_r / 1.8, rel=1e-9)
    assert _inflow_at(dict(w, inflow='productivity_index'), table, p_r, R_go, wlr)['k_l'] == pytest.approx(
        k_l, rel=1e-9
    )


# --- The mapping (SMP-111) -------------------------------------------------------------------------------------------
def test_every_drawn_well_maps_to_a_v2_well(wells):
    """v2: checks, Bounds: every well at its nominal state passes configurations.check(wp, 'v2.0.0'), with every
    input from its draws as SMP-111's table says."""
    for w, (md, tvd) in wells:
        inp = nominal_inputs(w)
        wp, bc = WellBuilder(w, (md, tvd), design=w).case(inp)
        check(wp, V2)
        assert differences(wp, V2) == []
        geometry, fluid = wp.geometry, wp.fluid
        assert geometry.n_cells == _N_CELLS == 100 and geometry.D == w['D']
        assert geometry.md[0] == pytest.approx(md[-1]) and geometry.tvd[0] == pytest.approx(w['L'])
        assert min(geometry.cos_incl) >= math.cos(math.radians(75)) - 1e-9
        black = BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])
        assert fluid.pvt == black
        assert (fluid.rho_w, fluid.gor, fluid.wlr) == (w['rho_w'], w['R_go'], w['wlr'])
        assert (fluid.cp_g, fluid.cp_o, fluid.cp_w) == (w['cp_g'], w['cp_o'], w['cp_w'])
        assert fluid.surface_tension_model == 'oil'
        assert wp.friction == RoughnessFriction(roughness=w['roughness'], correlation='chen')
        assert wp.thermal.h == w['h'] and wp.slip is V2_SLIP
        assert wp.choke == SimpsonChokeModel(K_c=w['K_c'], chk_profile=w['chk_profile'])
        if w['inflow'] == 'vogel':
            assert wp.inflow == Vogel(w_l_max=inp['w_l_max'])
        elif w['inflow'] == 'productivity_index':
            assert wp.inflow == ProductivityIndex(k_l=inp['k_l'])
        else:
            assert wp.inflow == Forchheimer(a=inp['forchheimer_a'], b=inp['forchheimer_b'])
        lifted = w['lift'] == 'gas_lift'
        assert wp.injection_md == (w['injection_md'] if lifted else None)
        assert bc.T_lg == (w['T_lg'] if lifted else None)
        assert (bc.p_r, bc.p_s, bc.T_r, bc.T_s, bc.u, bc.w_lg) == (w['p_r'], w['p_s'], w['T_r'], w['T_s'], 1.0, 0.0)


def test_an_attempts_lift_and_pvt(wells):
    """The valve and the lift gas are there while lift is installed: with lift off the well has no injection point
    and no lift gas, whatever the attempt's rate; a design from another draw gives the valve; and a table well's PVT
    model is its table set."""
    w, stations = next((w, st) for w, st in wells if w['lift'] == 'natural' and w['inflow'] == 'vogel')
    design = dict(injection_md=0.6 * w['md'], T_lg=320.0)
    builder = WellBuilder(w, stations, design=design)
    inp = dict(nominal_inputs(w), u=0.4, w_lg=0.7)
    wp, bc = builder.case(dict(inp, lift_on=False))
    assert wp.injection_md is None and bc.T_lg is None and bc.w_lg == 0.0 and bc.u == 0.4
    wp, bc = builder.case(dict(inp, lift_on=True))
    assert wp.injection_md == design['injection_md'] and bc.T_lg == 320.0 and bc.w_lg == 0.7
    check(wp, V2)
    assert wp.geometry is builder.geometry and wp.choke is builder.choke  # The fixed parts are built once
    table = PVTTable.read_csv(TABLE_SET)
    wp, _ = WellBuilder(w, stations, pvt=table).case(dict(inp, lift_on=False, gor=120.0))
    assert wp.fluid.pvt is table and wp.fluid.gor == 120.0
    with pytest.raises(ValueError):  # No coefficient: the productivity index refuses NaN
        builder.case(dict(inp, w_l_max=math.nan, lift_on=False))


@pytest.mark.slow
def test_mapped_core_wells_have_an_operating_point(wells):
    """The mapping end to end, on the Rust core: most of the first core wells flow at their nominal state with the
    choke fully open, a lifted well at half its lift-gas capacity, between the boundary pressures."""
    core = [(w, st) for w, st in wells if w['coverage_tail'] == 'core'][:12]
    flowing = 0
    for w, stations in core:
        inp = dict(nominal_inputs(w), w_lg=0.5 * w.get('w_lg_max', 0.0))
        wp, bc = WellBuilder(w, stations, design=w).case(inp)
        try:
            op = SSDFSimulator(wp, backend='rust').simulate(bc)
        except NoOperatingPoint:
            continue
        flowing += 1
        x = np.asarray(op.x).reshape(-1, 7)
        p, T = x[:, 0], x[:, 6]
        assert bc.p_s < p[-1] < p[0] < bc.p_r and np.all(np.diff(p) < 0)
        assert T[0] == pytest.approx(bc.T_r) and T[-1] < bc.T_r
    assert flowing >= 9
