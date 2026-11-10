"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for a v2 well's life (manywells.sampling.v2.life; specs/sampling.md, SMP-83 to SMP-92): the trends' parameters
and their start and end states by drive and class, the trends' shapes, the well's properties along the life, the
start of gas lift, the stage of a date, and a sol-2 well as its life at its date. No test solves a well.
"""

import copy
import math

import numpy as np
import pytest

from manywells.configurations import V2, check
from manywells.pvt.black_oil import BlackOil
from manywells.sampling.v2 import life as life_module
from manywells.sampling.v2.draws import P_S_MAX, STAGES, black_oil, draw_well, stream
from manywells.sampling.v2.inflow import COEF_KEYS, inflow_at
from manywells.sampling.v2.life import WLR_MAX, Life, _cumulative, lift_on, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.reservoir import producing_gor
from manywells.sampling.wells import rng_for

from .sampling_v2_wells import N_MANY_LIVES, SEED, drawn_lives, drawn_sol2_wells, drawn_wells, is_oil

LIFT_KEYS = 'z_inj injection_md z_inj_frac p_inj dp_valve packer_offset T_lg phi_lg T_lg_below_ambient glr_cap w_lg_max'
LIFT_KEYS = LIFT_KEYS.split()
AT_THE_DATE = (  # The keys of a sol-2 well that hold its life's state at its date (SMP-92)
    'life_stage date p_r r_p R_go wlr wgr ogr saturation below_hydrostatic p_s r_s h roughness lift lift_start_rule'
).split()
USED = {'vogel': ['w_l_max'], 'productivity_index': ['k_l'], 'forchheimer': ['forchheimer_a', 'forchheimer_b']}


@pytest.fixture(scope='session')
def lives():
    return drawn_lives()


@pytest.fixture(scope='session')
def sol2_wells():
    return drawn_sol2_wells()


def between(x, lo, hi, tol=1e-9):
    return lo - tol * max(1.0, abs(lo)) <= x <= hi + tol * max(1.0, abs(hi))


def is_tail(lf) -> bool:
    return lf.w0['coverage_tail'] == 'tail'


def has_support(lf) -> bool:
    return lf.w0['drive'] == 'pressure_support'


def dates(lf, n=41) -> list:
    return [float(t) for t in np.linspace(0.0, lf.T_life, n)]


def needs_lift(lf, t) -> bool:
    """SMP-78's static check at the trends' state at t: whether the no-slip column at the mean of the reservoir's
    and the wellhead's conditions outweighs the reservoir pressure less the nominal downstream pressure."""
    w0 = lf.w0
    p_r, p_s = lf.p_r(t), lf.p_s_nom(t)
    T_wh = w0['T_s'] + 0.6 * (w0['T_r'] - w0['T_s'])
    T_m = 0.5 * (w0['T_r'] + T_wh)
    vol, mass = stream(black_oil(w0), producing_gor(w0, p_r), lf.water(t), w0['rho_w'], 0.5 * (p_r + p_s), T_m)
    return p_r < p_s + mass / vol * 9.80665 * w0['L'] / 1e5


def check_life(lf):
    """The parameters' ranges, the start and the end by drive and class, and the trends between them (SMP-83 to
    SMP-91) that no life breaks."""
    w0, prm = lf.w0, lf.parameters()
    oil, tail, support, T = is_oil(w0), is_tail(lf), has_support(lf), lf.T_life
    assert lf.oil == oil and w0['life_stage'] == 'initial' and w0['well'] == lf.k

    # The life span and the clock (SMP-84)
    b, D, t_pl = prm['decline_b'], prm['decline_D'], prm['t_plateau']
    assert prm['T_life'] == T and between(T, 3, 30) and between(t_pl, 0, 0.25 * T)
    assert between(b, *((0.3, 0.5) if support else (0, 0.4))) and D > 0
    q_end = (1 + b * D * (T - t_pl)) ** (-1 / b) if b > 1e-6 else math.exp(-D * (T - t_pl))
    assert between(q_end, 0.03, 0.3)  # Arps's rate at the end of the life over the plateau's
    ts = dates(lf)
    F = [lf.F(t) for t in ts]
    assert F[0] == 0 and F[-1] == 1 and np.all(np.diff(F) > 0)
    assert lf.F(t_pl / 2) == pytest.approx(lf.F(t_pl) / 2, rel=1e-12)  # A plateau of unit rate
    assert np.all(np.diff(F, 2) < 1e-12)  # The rate never rises

    # The start and the end (SMP-83)
    r0, r1, a, z = prm['r_p_start'], prm['r_p_end'], prm['water_start'], prm['water_end']
    water = 'wlr' if oil else 'wgr'
    if tail:  # From a core initial state to its tail state
        assert between(r0, 0.97 if support else 0.85, 1) and r1 == min(w0['r_p'], r0) and z == w0[water]
        assert (a == 0 or between(a, 0, 0.1)) if oil else between(a, 2, 20)
    else:
        assert r0 == w0['r_p'] and a == w0[water]
        if support:  # The late draw's pressure, and the water at its cap
            assert between(r1, 0.70, 1) and r1 <= r0 and z == (0.98 if oil else 10000)
        else:  # The reservoir exhausted, with the late draw's water
            assert between(r1, 0.15, 0.3 if oil else 0.2) and between(z, *((0.05, 0.4) if oil else (5, 80)))

    # The reservoir pressure (SMP-85)
    p_i = w0['p_i']
    p_r = [lf.p_r(t) for t in ts]
    assert p_r[0] == pytest.approx(r0 * p_i, rel=1e-9) and p_r[-1] == pytest.approx(r1 * p_i, rel=1e-9)
    assert np.all(np.diff(p_r) <= 1e-9 * p_i) and between(prm['gamma_p'], 1, 2)
    if oil or support:
        t = ts[17]
        assert lf.p_r(t) == pytest.approx(p_i * (r1 + (r0 - r1) * (1 - lf.F(t)) ** prm['gamma_p']), rel=1e-12)

    # The water (SMP-86)
    F_bt, wlr_bt = prm['F_bt'], prm['wlr_bt']
    assert between(F_bt, *((0.05, 0.4) if support else (0.2, 0.7)))
    assert between(wlr_bt, *((0.05, 0.3) if support else (0.02, 0.1)))
    wat = [lf.water(t) for t in ts]
    assert wat[0] == a and wat[-1] == pytest.approx(max(a, z), rel=1e-9) and np.all(np.diff(wat) >= -1e-12 * z)
    assert all(v == a for v, f in zip(wat, F) if f < F_bt)
    assert not oil or max(wat) <= WLR_MAX

    # The GOR (SMP-87): the start's above the saturation pressure, and SMP-70's below it
    R_i, p_sat = w0['R_i'], w0['p_sat']
    gor = [producing_gor(w0, p) for p in p_r]
    if p_sat is None:
        assert all(R == R_i for R in gor)
    else:
        assert all(R == R_i for R, p in zip(gor, p_r) if p >= p_sat)
        if not oil:
            assert np.all(np.diff(gor) >= -1e-9 * R_i)

    # The separator (SMP-89)
    t_1, t_2, r_mid, r_late = prm['t_sep_1'], prm['t_sep_2'], prm['r_s_mid'], prm['r_s_late']
    assert 0.25 * T <= t_1 <= t_2 <= 0.9 * T and between(r_mid, 0.5, 1) and between(r_late, 0.1, 0.6)
    steps = [(0.0, 1.0), (0.999 * t_1, 1.0), (t_1, r_mid), (0.5 * (t_1 + t_2), r_mid), (t_2, min(r_mid, r_late))]
    for t, r_s in steps + [(T, min(r_mid, r_late))]:
        assert lf.r_s(t) == r_s
        if tail:
            assert lf.p_s_nom(t) == max((w0['p_s'] - w0['dp_fl']) * r_s + w0['dp_fl'], 1.5)
        else:
            assert lf.p_s_nom(t) == min(max(r_s * w0['p_sep'] + w0['dp_fl'], 1.5), P_S_MAX[w0['well_type']])
    p_s = [lf.p_s_nom(t) for t in ts]
    assert p_s[0] == pytest.approx(w0['p_s'], rel=1e-12) and np.all(np.diff(p_s) <= 0) and min(p_s) >= 1.5

    # The properties that age (SMP-88)
    eps = [lf.roughness(t) for t in ts]
    assert eps[0] == prm['roughness_start'] == w0['roughness'] and np.all(np.diff(eps) >= 0)
    assert eps[-1] == pytest.approx(prm['roughness_end'], rel=1e-12) and between(eps[-1], eps[0], 1.5e-4)
    half = math.sqrt(eps[0] * prm['roughness_end'])
    assert lf.roughness(T / 2) == pytest.approx(half, rel=1e-12)  # Log-linear in t
    h = [lf.h(t) for t in ts]
    assert h[0] == pytest.approx(w0['h'], rel=1e-9)
    assert all(v == w0['h'] for v in h) if tail else np.all(np.diff(h) < 0) and h[-1] > 0
    if not tail:  # SMP-81's coefficient at the age t_a + t
        t_D = 1.03e-6 * (w0['age_years'] + ts[9]) * 365 * 86400 / (0.108 if w0['D'] <= 0.101 else 0.156) ** 2
        T_D = math.log(math.exp(-0.2 * t_D) + (1.5 - 0.3719 * math.exp(-t_D)) * math.sqrt(t_D))
        assert h[9] == pytest.approx(w0['U_t'] * w0['k_e'] / (w0['k_e'] + w0['D'] / 2 * w0['U_t'] * T_D), rel=1e-12)

    # The stage of a date (SMP-91)
    assert [lf.stage(t) for t in (0.0, 0.999, 1.0, 0.499 * T, 0.5 * T, T)] == ['initial'] * 2 + ['mid_life'] * 2 + [
        'late'
    ] * 2

    # Gas lift (SMP-90)
    start, rule, design = lf.lift_start, lf.lift_start_rule, lf.design
    assert rule in ('from_start', 'static_check', 'drawn_time', None) and (rule is None) == (start is None)
    assert (w0['lift'] == 'gas_lift') == (rule == 'from_start') == (design is w0)
    assert design is not None if oil else rule in ('from_start', 'drawn_time', None)
    if design is None:
        assert rule is None
    else:
        assert design['lift'] == 'gas_lift' and all(key in design for key in LIFT_KEYS)
    if oil and rule != 'from_start':  # The static check's first failure on its 41 dates, if before a drawn date
        failed = next((t for t in ts if needs_lift(lf, t)), None)
        if rule == 'static_check':
            assert start == failed
        elif rule == 'drawn_time':
            assert between(start, 0.2 * T, 0.9 * T) and (failed is None or failed >= start)
        else:
            assert failed is None
    elif rule == 'drawn_time':
        assert between(start, 0.2 * T, 0.9 * T)


def check_state(lf, t, eps=(1.0, 1.0)):
    """A date's state (SMP-88, SMP-100): the trends' values, the drift on the GOR and the water only, and the
    inflow's coefficients at the trends' state."""
    w0, st = lf.w0, lf.state(t, eps)
    p_r, water = lf.p_r(t), lf.water(t)
    R_go = producing_gor(w0, p_r)
    assert list(st) == ['t', 'p_r', 'gor', 'wlr', 'wgr', 'h', 'roughness', 'p_s_nom', 'life_stage', *COEF_KEYS]
    assert (st['t'], st['p_r'], st['gor']) == (t, p_r, R_go * eps[0])
    assert (st['h'], st['roughness'], st['p_s_nom'], st['life_stage']) == (
        lf.h(t),
        lf.roughness(t),
        lf.p_s_nom(t),
        lf.stage(t),
    )
    if lf.oil:
        assert st['wlr'] == min(WLR_MAX, water * eps[1]) and math.isnan(st['wgr'])
        wlr_trend = min(WLR_MAX, water)
    else:
        assert st['wgr'] == water * eps[1] and st['wlr'] == st['wgr'] / (st['wgr'] + 1e6 / st['gor'])
        wlr_trend = water / (water + 1e6 / R_go)
    assert 0 <= st['wlr'] < 1
    coef = inflow_at(w0, black_oil(w0), p_r, R_go, wlr_trend)  # Without the drift
    assert [key for key in COEF_KEYS if not math.isnan(st[key])] == USED[w0['inflow']]
    assert all(st[key] == coef[key] and st[key] >= 0 for key in USED[w0['inflow']])
    return st


def check_sol2_well(lf, w, drawn_stage):
    """A sol-2 well (SMP-92): its life at one date in its drawn stage, with the rest as its initial-stage draw."""
    w0, T, t, stage = lf.w0, lf.T_life, w['date'], w['life_stage']
    assert stage == drawn_stage and w['well'] == lf.k
    if is_tail(lf):
        assert t == T  # The end of its life, which holds its tail state
        assert w['p_r'] == pytest.approx(min(w0['r_p'], lf.parameters()['r_p_start']) * w0['p_i'], rel=1e-9)
    else:
        lo, hi = {'initial': (0, 1), 'mid_life': (1, T / 2), 'late': (T / 2, T)}[stage]
        assert lo <= t <= hi and lf.stage(t) == stage
    st = lf.state(t)
    assert (w['p_r'], w['R_go'], w['wlr'], w['p_s']) == (st['p_r'], st['gor'], st['wlr'], st['p_s_nom'])
    assert (w['h'], w['roughness'], w['r_s'], w['r_p']) == (st['h'], st['roughness'], lf.r_s(t), w['p_r'] / w0['p_i'])
    assert all(w[key] == st[key] or (math.isnan(w[key]) and math.isnan(st[key])) for key in COEF_KEYS)
    assert w['below_hydrostatic'] == (w['p_r'] < w0['p_h'])
    if lf.oil:
        assert 'wgr' not in w and w['ogr'] is None
    else:
        assert w['wgr'] == st['wgr'] and w['ogr'] == 1e6 / w['R_go']
    if w0['p_sat'] is None:
        assert w['saturation'] == 'single_phase_gas'
    else:
        assert w['saturation'] == ('undersaturated' if w['p_r'] > w0['p_sat'] else 'saturated')
    lifted = lift_on(t, lf.lift_start)  # The lift at the date (SMP-90)
    assert w['lift'] == ('gas_lift' if lifted else 'natural')
    assert w['lift_start_rule'] == (lf.lift_start_rule if lifted else None)
    for key in LIFT_KEYS:
        assert w[key] == lf.design[key] if lifted else key not in w
    fixed = [key for key in w0 if key not in AT_THE_DATE + LIFT_KEYS + list(COEF_KEYS)]  # Fixed for the life (SMP-88)
    assert all(w[key] == w0[key] for key in fixed) and len(fixed) > 60
    assert set(w) == set(w0) | {'date', 'lift_start_rule', *COEF_KEYS} | set(LIFT_KEYS if lifted else [])
    # v2: checks, Bounds: the well at its nominal state is a well of the v2.0.0 configuration
    inp = dict(st, u=1.0, w_lg=0.0, lift_on=lifted, p_s=w['p_s'])
    wp, bc = WellBuilder(w0, lf.stations, design=lf.design).case(inp)
    check(wp, V2)
    assert (bc.p_r, bc.p_s, wp.injection_md) == (w['p_r'], w['p_s'], w['injection_md'] if lifted else None)


# --- The life --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('part', ['core', 'tail'])
def test_no_life_breaks_a_bound(lives, part):
    """SMP-83 to SMP-91, on core and tail wells apart."""
    drawn = [lf for lf in lives if lf.w0['coverage_tail'] == part]
    assert len(drawn) >= 10
    for lf in drawn:
        check_life(lf)


def test_every_kind_of_life_is_drawn(lives):
    """The lives of the tests hold every drive of the oils and of the gas classes, core and tail, and every rule of
    the lift's start, so that the bounds above check them."""
    kinds = {(lf.w0['coverage_tail'], lf.oil, lf.w0['drive']) for lf in lives}
    assert len(kinds) == 8
    assert {lf.lift_start_rule for lf in lives} == {'from_start', 'static_check', 'drawn_time', None}
    assert any(lf.design is None for lf in lives)
    assert any(lf.design is not None and lf.lift_start is None for lf in lives)  # An oil well that may be lifted


def test_the_evolution_generators_numbers(lives):
    """SMP-83 to SMP-90, SMP-109: the numbers of a well's `evolution` generator in their order, each drawn for
    every well: the life span, the plateau, the decline's end rate and exponent, gamma, the breakthrough's recovery
    and WLR, the separator's two steps, the lift's drawn date, a tail well's start (the pressure, whether it is dry,
    its WLR and its WGR) and a core well's end pressure under depletion."""
    seen = set()
    for lf in lives:
        prm, T = lf.parameters(), lf.T_life
        oil, tail, support = lf.oil, is_tail(lf), has_support(lf)
        u = rng_for(SEED, lf.k, 'v2', 'evolution').random(15)

        def uniform(i, lo, hi, u=u):
            return pytest.approx(lo + (hi - lo) * u[i], rel=1e-12)

        def log_uniform(i, lo, hi, u=u):
            return pytest.approx(lo * (hi / lo) ** u[i], rel=1e-12)

        assert T == log_uniform(0, 3, 30) and prm['t_plateau'] / T == uniform(1, 0, 0.25)
        b, tau = prm['decline_b'], T - prm['t_plateau']
        assert b == (uniform(3, 0.3, 0.5) if support else uniform(3, 0, 0.4))
        assert b < 1e-6 or (1 + b * prm['decline_D'] * tau) ** (-1 / b) == log_uniform(2, 0.03, 0.3)
        assert prm['gamma_p'] == uniform(4, 1, 2)
        assert prm['F_bt'] == (uniform(5, 0.05, 0.4) if support else uniform(5, 0.2, 0.7))
        assert prm['wlr_bt'] == (uniform(6, 0.05, 0.3) if support else uniform(6, 0.02, 0.1))
        steps = sorted(0.25 + 0.65 * x for x in u[7:9])
        assert [prm['t_sep_1'] / T, prm['t_sep_2'] / T] == pytest.approx(steps, rel=1e-12)
        if lf.lift_start_rule == 'drawn_time':
            assert lf.lift_start / T == uniform(9, 0.2, 0.9)
        if tail:
            assert prm['r_p_start'] == uniform(10, 0.97 if support else 0.85, 1)
            if oil:
                assert prm['water_start'] == (0 if u[11] < 0.5 else uniform(12, 0, 0.1))
            else:
                assert prm['water_start'] == log_uniform(13, 2, 20)
        elif not support:
            assert prm['r_p_end'] == uniform(14, 0.15, 0.3 if oil else 0.2)
        seen.add((tail, oil, support, lf.lift_start_rule == 'drawn_time', tail and oil and u[11] < 0.5))
    assert len(seen) >= 12 and {dry for *_, dry in seen} == {True, False}


def test_roughness_never_falls(lives):
    """SMP-88: where the late draw's roughness is below the initial draw's, the roughness stays."""
    lf = next(lf for lf in lives if not is_tail(lf))
    smoother = copy.copy(lf)
    smoother._wL = dict(lf._wL, roughness=0.5 * lf.w0['roughness'])
    assert smoother.parameters()['roughness_end'] == lf.w0['roughness']
    assert all(smoother.roughness(t) == lf.w0['roughness'] for t in dates(lf, 5))
    assert lf.roughness(lf.T_life) > lf.w0['roughness']


def test_arps_cumulative_production():
    """SMP-84: N(t) is the integral of a unit rate to the plateau's end and of Arps's rate (1 + b D tau)^(-1/b)
    after it, exponential where b < 1e-6 (Ahmed 2006, Eqs. 16-1 to 16-4)."""
    t_pl, D = 2.0, 0.4
    assert _cumulative(1.5, t_pl, D, 0.3) == 1.5 and _cumulative(t_pl, t_pl, D, 0.3) == t_pl
    for b in (0.0, 0.1, 0.5):
        for t in (2.5, 6.0, 20.0):
            tau, dt = t - t_pl, 1e-5
            rate = math.exp(-D * tau) if b == 0 else (1 + b * D * tau) ** (-1 / b)
            slope = (_cumulative(t + dt, t_pl, D, b) - _cumulative(t - dt, t_pl, D, b)) / (2 * dt)
            assert slope == pytest.approx(rate, rel=1e-7)
    assert _cumulative(9.0, t_pl, D, 2e-6) == pytest.approx(_cumulative(9.0, t_pl, D, 0.0), rel=1e-5)
    assert _cumulative(9.0, t_pl, D, 0.0) == pytest.approx(t_pl + (1 - math.exp(-D * 7.0)) / D, rel=1e-14)


def test_a_gas_wells_depletion_follows_p_over_z(lives):
    """SMP-85: for a gas or condensate well under depletion, p_r / Z(p_r) is linear in F between the start's and
    the end's, with the black oil's DAK Z at T_r (Ahmed 2006, Eq. 13-8)."""
    gas = [lf for lf in lives if not lf.oil and not has_support(lf)]
    assert len(gas) >= 5
    for lf in gas:
        w0, prm, bo = lf.w0, lf.parameters(), black_oil(lf.w0)

        def p_over_z(p):
            return p / float(bo.z_factor(p, w0['T_r']))

        start, end = p_over_z(prm['r_p_start'] * w0['p_i']), p_over_z(prm['r_p_end'] * w0['p_i'])
        for t in dates(lf, 9):
            F = lf.F(t)
            assert p_over_z(lf.p_r(t)) == pytest.approx((1 - F) * start + F * end, rel=1e-4)


def test_an_oils_water_jumps_at_breakthrough(lives):
    """SMP-86: an oil's WLR stays at the start's until F_bt, jumps to WLR_bt, at most the end's, and then rises
    with its logit linear in F; a gas class's WGR rises log-linearly in F from F_bt."""
    seen = 0
    for lf in lives:
        prm = lf.parameters()
        a, z, F_bt = prm['water_start'], prm['water_end'], prm['F_bt']
        if z <= a:
            assert all(lf.water(t) == a for t in dates(lf, 9))  # An end that is not above the start
            continue
        after = [t for t in dates(lf, 201) if lf.F(t) >= F_bt]
        t_bt, t_mid = after[0], after[len(after) // 2]
        x = (lf.F(t_mid) - F_bt) / (1 - F_bt)
        if lf.oil:
            jump = min(prm['wlr_bt'], z)
            if jump <= a:
                continue
            seen += 1
            assert lf.water(t_bt) >= jump * (1 - 1e-12) and jump > a  # The jump, to the logit's rounding
            logit = math.log(jump / (1 - jump)) + x * (math.log(z / (1 - z)) - math.log(jump / (1 - jump)))
            assert lf.water(t_mid) == pytest.approx(1 / (1 + math.exp(-logit)), rel=1e-12)
        else:
            assert lf.water(t_mid) == pytest.approx(a * (z / a) ** x, rel=1e-12)
    assert seen >= 20


def test_a_dates_state(lives):
    """SMP-88, SMP-100: the state of a date holds the trends' values, the date's drift on the GOR and the water
    only, and the inflow's coefficients of the trends' state, whatever the drift; the start of a core well is its
    initial-stage draw."""
    for lf in lives[:40]:
        w0, T = lf.w0, lf.T_life
        start = check_state(lf, 0.0)
        for t, eps in ((0.37 * T, (0.96, 1.04)), (0.37 * T, (1.0, 1.0)), (T, (1.05, 0.95)), (T, (0.97, 1.04))):
            check_state(lf, t, eps)
        assert lf.state(math.nextafter(T, math.inf)) == lf.state(T)  # A date that rounding has put past the end
        if not is_tail(lf):
            assert start['p_r'] == pytest.approx(w0['p_r'], rel=1e-12) and start['wlr'] == pytest.approx(w0['wlr'])
            assert start['gor'] == pytest.approx(w0['R_go'], rel=1e-9)
            if lf.oil:  # The index of the initial draw, at its state
                key = USED[w0['inflow']][0]
                assert start[key] == pytest.approx(w0[key], rel=1e-9)


def test_the_coefficients_take_the_pvt_model(lives):
    """SMP-72, SMP-117: a table fluid's coefficients are made with its PVT model, and its draws with the table's
    values in place of the fluid block's (SMP-115)."""
    lf = next(lf for lf in lives if lf.w0['fluid_class'] == 'volatile_oil')
    w0 = lf.w0
    fluid = dict(p_sat=0.9 * w0['p_sat'], R_i=1.1 * w0['R_i'], rho_o=w0['rho_o'] + 5, rho_g=1.04 * w0['rho_g'])
    pvt = BlackOil(rho_o=fluid['rho_o'] + 10, rho_g=fluid['rho_g'], gas_law='dak', p_sep=w0['p_sep'], T_sep=310.0)
    table = Life(SEED, lf.k, fluid=fluid, pvt=pvt)
    assert table.pvt is pvt and lf.pvt == black_oil(w0)
    assert all(table.w0[key] == value for key, value in fluid.items())
    assert table.T_life == lf.T_life and table.stations == lf.stations  # No draw moves
    st = table.state(0.4 * table.T_life)
    assert st['gor'] == producing_gor(table.w0, st['p_r'])
    coef = inflow_at(table.w0, pvt, st['p_r'], st['gor'], st['wlr'])
    key = USED[table.w0['inflow']][0]
    assert st[key] == coef[key] != inflow_at(table.w0, black_oil(table.w0), st['p_r'], st['gor'], st['wlr'])[key]


def test_lift_design_is_the_late_draws(lives):
    """SMP-90: a well lifted later takes the late-stage draw's design, made as if lifted where that draw leaves an
    oil well natural; a gas well that the late draw leaves natural has none."""
    later = [lf for lf in lives if lf.w0['lift'] == 'natural'][:20]
    for lf in later:
        late = draw_well(SEED, lf.k, 'late')[0]
        if late['lift'] == 'gas_lift':
            assert lf.design == late and lf.lift_start is not None
            assert lf.lift_start <= 0.9 * lf.T_life  # At the drawn date, or at the static check's failure before it
        elif lf.oil:
            assert lf.design == draw_well(SEED, lf.k, 'late', force_lift=True)[0]
            assert lf.lift_start_rule in ('static_check', None)
        else:
            assert lf.design is None and lf.lift_start is None
    assert {lf.oil for lf in later} == {True, False}


def test_lift_on():
    """SMP-90: lift is installed from its start, a date at the start that rounding has put below it included."""
    assert not lift_on(5.0, None) and not lift_on(0.0, None)
    assert lift_on(0.0, 0.0) and lift_on(3.0, 3.0) and lift_on(3.1, 3.0) and not lift_on(2.9, 3.0)
    assert lift_on(0.3 - 1e-16, 0.1 + 0.2)


# --- sol-2 -----------------------------------------------------------------------------------------------------------
def test_a_sol2_well_is_its_life_at_a_date(lives, sol2_wells):
    """SMP-92, and SMP-83's one life for both datasets."""
    drawn = drawn_wells()
    for lf, w in zip(lives, sol2_wells):
        check_sol2_well(lf, w, drawn[lf.k][0]['life_stage'])
    core = [w for w in sol2_wells if w['coverage_tail'] == 'core']
    assert {w['life_stage'] for w in core} == set(STAGES)
    assert {w['lift_start_rule'] for w in sol2_wells} == {'from_start', 'static_check', 'drawn_time', None}


def test_a_sol2_wells_date_is_uniform_in_its_stage(lives, sol2_wells):
    """SMP-92: the date's place in its stage's interval is the one number of the date's generator."""
    for lf, w in zip(lives, sol2_wells):
        if is_tail(lf):
            continue
        T = lf.T_life
        lo, hi = {'initial': (0, 1), 'mid_life': (1, T / 2), 'late': (T / 2, T)}[w['life_stage']]
        u = rng_for(SEED, lf.k, 'v2', 'date').random()
        assert w['date'] == pytest.approx(lo + (hi - lo) * u, rel=1e-12)


def test_a_sol2_well_lifted_by_the_filter(lives, sol2_wells):
    """SMP-103's lift of a natural oil well that cannot flow: the life's design, labelled `no_natural_flow`, and
    nothing else changed; no other well takes it."""
    natural = [(lf, w) for lf, w in zip(lives, sol2_wells) if w['lift'] == 'natural']
    oils = [(lf, w) for lf, w in natural if lf.oil]
    assert len(oils) >= 10
    for lf, w in oils:
        lifted = sol2_well(lf, no_natural_flow=True)
        assert (lifted['lift'], lifted['lift_start_rule']) == ('gas_lift', 'no_natural_flow')
        assert all(lifted[key] == lf.design[key] for key in LIFT_KEYS)
        assert {key for key in lifted if lifted[key] != w.get(key) and lifted[key] == lifted[key]} == {
            'lift',
            'lift_start_rule',
            *LIFT_KEYS,
        }
    gas = next(lf for lf, _ in natural if not lf.oil)
    lifted_already = next(lf for lf, w in zip(lives, sol2_wells) if w['lift'] == 'gas_lift')
    for lf in (gas, lifted_already):
        with pytest.raises(ValueError, match='natural oil well'):
            sol2_well(lf, no_natural_flow=True)


# --- Reproducibility -------------------------------------------------------------------------------------------------
def test_a_rerun_gives_the_same_life(lives, sol2_wells):
    """SMP-109: a life depends on the seed and the well's index only, whatever was drawn before it."""
    for k in (23, 4):
        lf = Life(SEED, k)
        t = 0.6 * lf.T_life
        assert lf.parameters() == lives[k].parameters() and lf.w0 == lives[k].w0 and lf.stations == lives[k].stations
        assert (lf.lift_start, lf.lift_start_rule, lf.design) == (
            lives[k].lift_start,
            lives[k].lift_start_rule,
            lives[k].design,
        )
        assert lf.state(t) == lives[k].state(t)
        again = sol2_well(lf)
        assert all(again[key] == v or v != v for key, v in sol2_wells[k].items()) and set(again) == set(sol2_wells[k])
    assert Life(SEED + 1, 4).T_life != lives[4].T_life


def test_the_lifes_generators(lives, sol2_wells, monkeypatch):
    """SMP-109: the life's parameters come from the well's `evolution` generator and a sol-2 well's date from its
    `date` generator, so that with the evolution drawn from another seed the well's draws and its date's number
    stay."""
    k = 9
    asked = []

    def other_evolution(seed, *parts):
        asked.append(parts)
        return rng_for(seed + 1 if parts[-1] == 'evolution' else seed, *parts)

    monkeypatch.setattr(life_module, 'rng_for', other_evolution)
    lf = Life(SEED, k)
    assert asked == [(k, 'v2', 'evolution')]
    assert lf.w0 == lives[k].w0 and lf.T_life != lives[k].T_life
    w = sol2_well(lf)
    assert asked[1:] == [(k, 'v2', 'labels'), (k, 'v2', 'date')]
    assert w['life_stage'] == sol2_wells[k]['life_stage']
    if not is_tail(lf):
        lo, hi = {'initial': (0, 1), 'mid_life': (1, lf.T_life / 2), 'late': (lf.T_life / 2, lf.T_life)}[
            w['life_stage']
        ]
        assert w['date'] == pytest.approx(lo + (hi - lo) * rng_for(SEED, k, 'v2', 'date').random(), rel=1e-12)


@pytest.mark.slow
def test_bounds_on_many_lives():
    """The same checks on 300 lives and their sol-2 wells, which reach the rare branches; and the shares of the
    stages among the core sol-2 wells, which are the labels' (SMP-56)."""
    many, drawn = drawn_lives(N_MANY_LIVES), drawn_wells()
    stages = []
    for lf, w in zip(many, drawn_sol2_wells(N_MANY_LIVES)):
        check_life(lf)
        check_state(lf, 0.8 * lf.T_life, (1.03, 0.97))
        check_sol2_well(lf, w, drawn[lf.k][0]['life_stage'])
        if not is_tail(lf):
            stages.append(w['life_stage'])
    for stage, p in zip(STAGES, (0.25, 0.40, 0.35)):
        n, hits = len(stages), stages.count(stage)
        assert abs(hits - n * p) <= 4 * math.sqrt(n * p * (1 - p)) + 0.5, stage
