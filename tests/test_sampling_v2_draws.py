"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the v2 well draws (manywells.sampling.v2.draws and survey; specs/sampling.md, SMP-51 to SMP-82 and its
section v2: checks): the bounds that no draw breaks, on core and tail wells apart, the labels' shares and the draws'
marginals within their sampling error, and that the draws are reproducible. No test solves a well.
"""

import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import rankdata

from manywells.geometry import WellGeometry
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.sampling.v2 import draws
from manywells.sampling.v2.draws import P_S_MAX, STAGES, TUBING, black_oil, draw_well, stream
from manywells.sampling.v2.survey import FAMILIES, survey
from manywells.sampling.wells import rng_for, sample_well

from .sampling_v2_wells import N_MANY, SEED, counting, drawn_wells, is_oil

BLOCKS = 'labels depth geometry water pressure temperature separator fluid stage inflow completion choke lift heat'
LABELS = {  # SMP-52, the labels that the well draws set
    'coverage_tail': {'core', 'tail'},
    'well_type': {'platform', 'subsea', 'onshore'},
    'fluid_type': {'black_oil', 'volatile_oil', 'gas_condensate', 'gas'},
    'fluid_class': {'black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas'},
    'class_rule': {'mccain_1990'},
    'drive': {'depletion', 'pressure_support'},
    'life_stage': {'initial', 'mid_life', 'late'},
    'pressure_class': {'under', 'normal', 'over'},
    'saturation': {'undersaturated', 'saturated', 'single_phase_gas'},
    'geometry_class': {'vertical', 'slightly_deviated', 'deviated', 'highly_deviated'},
    'geometry_family': {'vertical', 'build_and_hold', 's_shaped', 'high_angle'},
    'inflow': {'vogel', 'productivity_index', 'forchheimer'},
    'tubing': {'2-7/8', '3-1/2', '4-1/2', '5-1/2', '7', '9-5/8'},
    'choke_model': {'simpson'},
    'chk_profile': {'linear', 'sigmoid', 'convex', 'concave'},
    'lift': {'natural', 'gas_lift'},
}
FLAGS = {
    'geometry_redrawn',
    'geometry_to_vertical',
    'p_i_cap_binds',
    'G_bounds_conflict',
    'G_raised_for_G_over_L',
    'T_r_cap_binds',
    'p_b_lowered_to_black_oil_limit',
    'design_rate_above_tubing_capacity',
    'bean_capped_at_4_in',
    'lift_not_installed_strong_well',
    'injection_at_packer',
    'injection_raised_to_0.3L',
    'lift_capacity_at_bound',
}
# The draws' ranges, as specs/sampling.md states them
API = {'black_oil': (16, 45), 'volatile_oil': (40, 50), 'gas_condensate': (42, 58), 'wet_gas': (45, 58)}
API['dry_gas'] = API['wet_gas']
GAS_GRAVITY = {'black_oil': (0.58, 0.95), 'volatile_oil': (0.62, 1.0), 'gas_condensate': (0.60, 0.90)}
GAS_GRAVITY['wet_gas'] = GAS_GRAVITY['dry_gas'] = (0.571, 0.75)
R_SI = {'black_oil': (0, 356), 'volatile_oil': (356, 588), 'gas_condensate': (588, 8900)}
R_SI.update(wet_gas=(1e6 / 112, 1e6 / 2.8), dry_gas=(1e6 / 2.8, 1e6 / 0.5))
R_P = {  # (stage, drive): p_r / p_i of a core well; late under depletion by fluid
    ('initial', 'depletion'): (0.85, 1.0),
    ('initial', 'pressure_support'): (0.97, 1.0),
    ('mid_life', 'depletion'): (0.55, 0.92),
    ('mid_life', 'pressure_support'): (0.85, 1.0),
    ('late', 'pressure_support'): (0.70, 1.0),
}
WLR = {
    ('initial', 'depletion'): (0, 0.1),
    ('initial', 'pressure_support'): (0, 0.1),
    ('mid_life', 'depletion'): (0, 0.3),
    ('mid_life', 'pressure_support'): (0.1, 0.6),
    ('late', 'depletion'): (0.05, 0.4),
    ('late', 'pressure_support'): (0.6, 0.98),
}
WGR = {
    ('initial', 'depletion'): (2, 20),
    ('initial', 'pressure_support'): (2, 20),
    ('mid_life', 'depletion'): (5, 50),
    ('mid_life', 'pressure_support'): (10, 150),
    ('late', 'depletion'): (5, 80),
    ('late', 'pressure_support'): (50, 500),
}
PERM = {'platform': (0.5, 3000), 'subsea': (0.5, 3000), 'onshore': (0.05, 1000)}
ROUGHNESS = {'initial': (1.5e-6, 3e-5), 'mid_life': (1e-5, 6e-5), 'late': (2e-5, 1.5e-4)}
AGE = {'initial': (30 / 365, 1), 'mid_life': (1, 8), 'late': (5, 25)}  # years
P_INJ = {'platform': (48, 85), 'subsea': (130, 210), 'onshore': (48, 85)}
THETA = {'slightly_deviated': (5, 20), 'deviated': (20, 60), 'highly_deviated': (60, 75)}
FAMILIES_OF = {
    'vertical': ('vertical',),
    'slightly_deviated': ('build_and_hold',),
    'deviated': ('build_and_hold', 's_shaped'),
    'highly_deviated': ('build_and_hold', 'high_angle'),
}


@pytest.fixture(scope='session')
def wells():
    return drawn_wells()


def between(x, lo, hi, tol=1e-9):
    return lo - tol * max(1.0, abs(lo)) <= x <= hi + tol * max(1.0, abs(hi))


def check_well(w, md, tvd):
    """The bounds and relations of specs/sampling.md that no drawn well breaks (v2: checks, Bounds)."""
    tail, wt, fc = w['coverage_tail'] == 'tail', w['well_type'], w['fluid_class']
    oil, stage, drive = is_oil(w), w['life_stage'], w['drive']
    z, d_w, L, p_i, T_r, T_s = w['z'], w['d_w'], w['L'], w['p_i'], w['T_r'], w['T_s']

    # Labels (SMP-52)
    for label, values in LABELS.items():
        assert w[label] in values, label
    assert fc == w['fluid_type'] or (w['fluid_type'] == 'gas' and fc in ('wet_gas', 'dry_gas'))
    assert set(filter(None, w['flags'].split(';'))) <= FLAGS

    # Depth and survey (SMP-57 to SMP-61): every cell within 75 degrees of vertical, MD at most 9342 m
    assert between(z, 600, 7000 if tail else 6500)
    d_w_range = {'platform': (15, 2000), 'subsea': (50 if tail else 80, 3000), 'onshore': (0, 0)}[wt]
    assert between(d_w, *d_w_range)
    assert L == (z + d_w if wt == 'platform' else z)
    assert md[0] == tvd[0] == 0.0 and tvd[-1] == pytest.approx(L, rel=1e-12) and md[-1] == w['md'] <= 9342
    assert np.all(np.diff(md) > 0) and np.all(np.diff(tvd) > -1e-9)
    geometry = WellGeometry.from_survey(md, tvd, 100, D=w['D'])
    assert min(geometry.cos_incl) >= math.cos(math.radians(75)) - 1e-9
    gcl, family = w['geometry_class'], w['geometry_family']
    assert family in FAMILIES_OF[gcl]
    if family == 'vertical':
        assert w['theta_max'] == 0 and md == tvd
    elif family == 's_shaped':
        assert between(w['theta_max'], 15, 45, 1e-6) and w['theta_end'] <= 35 + 1e-6
    else:  # The hold's inclination (build_and_hold) or the heel's (high_angle) is the class's
        assert between(w['theta_max'], *THETA[gcl], 1e-6) and w['theta_end'] == pytest.approx(w['theta_max'], abs=1e-9)

    # Water and initial pressure (SMP-62, SMP-63): p_i at most 950 bar, lambda at most 0.85
    sal = w['salinity']
    assert between(sal, 0.002, 0.25) and w['rho_w'] == pytest.approx(999.1 + 692 * sal + 311 * sal**2, rel=1e-12)
    p_h, p_lith = w['p_h'], w['p_lith']
    assert p_h == pytest.approx(1.01325 + 9.80665 * (1025 * d_w + w['rho_w'] * z) / 1e5, rel=1e-12)
    assert p_lith == pytest.approx(1.01325 + 9.80665 * 1025 * d_w / 1e5 + 0.226 * z, rel=1e-12)
    assert p_i <= 950 + 1e-9
    lam = (p_i - p_h) / (p_lith - p_h)
    if w['pressure_class'] == 'under':
        assert between(p_i, 0.8 * p_h, p_h)
    else:
        assert between(lam, 0, 0.8 if tail else 0.85) and (w['pressure_class'] == 'over') == (w['lam'] >= 0.25)
        assert lam == pytest.approx(w['lam'], abs=1e-9)

    # Temperatures (SMP-64, SMP-65): T_r at most 450 K, and at least 10 K/km over the modelled well
    T_s_range = (272, 297) if wt != 'onshore' else ((258, 308) if tail else (268, 303))
    assert between(T_s, *T_s_range) and between(w['G'], 12, 55)
    assert T_r == pytest.approx(T_s + w['G'] * z / 1000, rel=1e-12) and T_r <= 450 + 1e-9
    assert (T_r - T_s) / L * 1000 >= 10 - 1e-9

    # Separator and the nominal downstream pressure (SMP-66)
    p_sep_range, T_sep_range = {
        'platform': ((5, 85), (300, 340)),
        'subsea': ((5, 85), (290, 320)),
        'onshore': ((2, 70), (285, 320)),
    }[wt]
    assert between(w['p_sep'], *p_sep_range) and between(w['T_sep'], *T_sep_range)
    dp_fl = w['dp_fl']
    assert between(dp_fl, 5 + 0.03 * d_w, 30 + 0.07 * d_w) if wt == 'subsea' else dp_fl == 0
    assert between(w['r_s'], *{'initial': (1, 1), 'mid_life': (0.5, 1), 'late': (0.1, 0.6)}[stage])
    assert between(w['p_s'], 1.5, P_S_MAX[wt]) and P_S_MAX[wt] == (250 if wt == 'subsea' else 120)
    if not tail:
        assert w['p_s'] == min(max(w['r_s'] * w['p_sep'] + dp_fl, 1.5), P_S_MAX[wt])

    # Fluid (SMP-67, SMP-68): API from 16 to 58, gas gravity at least 0.571, p_sat at most p_i, a black oil's
    # R_si at most 356 Sm3/Sm3
    assert between(w['api'], *API[fc]) and between(w['sg_gas'], *GAS_GRAVITY[fc])
    assert w['rho_o'] == density_from_api(w['api']) and w['rho_g'] == gas_density_from_sg(w['sg_gas'])
    assert between(w['R_i'], *R_SI[fc])
    if fc in ('wet_gas', 'dry_gas'):
        assert w['p_sat'] is None and w['saturation'] == 'single_phase_gas'
    else:
        assert 0 < w['p_sat'] <= p_i + 1e-9
        assert w['saturation'] == ('undersaturated' if w['p_r'] > w['p_sat'] else 'saturated')
    if fc == 'black_oil':
        assert w['R_i'] == pytest.approx(black_oil(w).rs(w['p_sat'], T_r), rel=1e-9)

    # Reservoir state (SMP-69, SMP-70): an oil's WLR at most 0.98
    r_p_range = (0.15, 1.0) if tail else R_P.get((stage, drive), (0.3, 0.6) if oil else (0.2, 0.55))
    assert between(w['r_p'], *r_p_range) and w['p_r'] == w['r_p'] * p_i
    assert w['below_hydrostatic'] == (w['p_r'] < p_h)
    if oil:
        assert between(w['wlr'], *((0, 0.98) if tail else WLR[stage, drive]))
        assert w['R_go'] > 0 and (w['R_go'] == w['R_i']) == (w['p_r'] >= w['p_sat'])
        assert w['ogr'] is None and 'wgr' not in w and between(w['w_kgko'], 0, 1 if tail else 0)
    else:
        assert between(w['wgr'], *((1, 10000) if tail else WGR[stage, drive]))
        assert w['ogr'] == 1e6 / w['R_go'] and w['wlr'] == pytest.approx(w['wgr'] / (w['wgr'] + w['ogr']), rel=1e-12)
        if fc == 'gas_condensate':
            assert between(w['c_k'], 0.5, 3) and between(w['R_go'], w['R_i'], w['R_i'] * (1 + w['c_k']))
        else:
            assert w['c_k'] is None and w['R_go'] == w['R_i']

    # Reservoir and inflow (SMP-71 to SMP-74)
    k_lo, k_hi = PERM[wt]
    assert between(w['perm'], *((k_lo, k_hi) if oil else (k_lo / 14, k_hi / 14)))
    assert between(w['h_net'], 3, 100) and between(w['skin'], -3, 15) and between(w['delta_d'], 0.1, 0.4)
    assert between(w['ln_re_rw'], *((8.5, 9.5) if family == 'high_angle' else (7, 9)))
    if family == 'high_angle':  # A drain: Joshi's ratio, times an oil drain's friction efficiency
        assert between(w['drain_length'], 300, 2500) and between(w['I_ani'], 1, 3) and w['J_H_over_J_V'] > 0
        assert w['f_geo'] == pytest.approx(w['J_H_over_J_V'] * w.get('eta_drain', 1.0), rel=1e-12)
    else:  # The slant's factor on the radial index
        assert 'drain_length' not in w and 'eta_drain' not in w and 'b_friction_share' not in w
        slant = 1.0 if family == 'vertical' else math.cos(math.radians(w['theta_end'])) ** -0.5
        assert w['f_geo'] == pytest.approx(slant, rel=1e-12)
    if oil:
        rho_lsc = (1 - w['wlr']) * w['rho_o'] + w['wlr'] * w['rho_w']
        assert w['inflow'] == ('vogel' if w['p_sat'] >= 0.8 * w['p_r'] and w['wlr'] < 0.6 else 'productivity_index')
        assert w['J'] > 0 and w['k_l'] == pytest.approx(w['J'] * rho_lsc / 86400, rel=1e-12)
        assert w['w_l_max'] == pytest.approx(w['k_l'] * w['p_r'] / 1.8, rel=1e-12)
        assert w['q_l_ref'] == pytest.approx(w['J'] * w['delta_d'] * w['p_r'], rel=1e-12)
        assert 0 < w.get('eta_drain', 1.0) <= 1
    else:
        assert w['inflow'] == 'forchheimer' and 'k_l' not in w and 'w_l_max' not in w
        assert w['a_gas'] > 0 and w['b_gas'] > 0 and w['a_ratio'] > 0 and w['b_ratio'] > 0
        assert between(w['D_nondarcy'], 1e-5, 2e-3) and between(w['perf_share'], 0.3, 1)
        assert 0 < w['q_g_ref'] <= w['AOF'] and 0 < w.get('b_friction_share', 0.5) < 1
        assert w['a_ratio'] == pytest.approx(1 / w['f_geo'], rel=1e-12)
        if family == 'high_angle':  # The non-Darcy loss per completed length, and the liner's friction on top
            per_length = (w['I_ani'] * w['h_net'] / w['drain_length']) ** 2
            assert w['b_ratio'] * (1 - w['b_friction_share']) == pytest.approx(per_length, rel=1e-6)
        else:
            assert w['b_ratio'] == pytest.approx(1 / w['f_geo'], rel=1e-12)
        assert w['q_l_ref'] == pytest.approx(w['q_g_ref'] * (w['ogr'] + w['wgr']) / 1e6, rel=1e-12)

    # Tubing and roughness (SMP-75, SMP-76)
    assert w['D'] == TUBING[w['tubing']] and (not oil or w['tubing'] != '9-5/8')
    assert between(w['c_e'], 100, 150) and w['q_design'] > 0 and w['q_cap'] > 0
    assert between(w['roughness'], *((1.5e-6, 1.5e-4) if tail else ROUGHNESS[stage]))

    # Choke (SMP-77)
    assert between(w['beta'], 0.15, 0.75 if tail else 0.6) and between(w['c_d'], 0.75, 0.9)
    assert w['d_bean'] == pytest.approx(w['beta'] * w['D'], rel=1e-12) and w['d_bean'] <= 0.1016
    assert w['K_c'] == pytest.approx(w['c_d'] * math.pi * w['d_bean'] ** 2 / 4, rel=1e-12)

    # Gas lift (SMP-78 to SMP-80): w_lg_max within 0.1 to 3.5 kg/s
    lifted = w['lift'] == 'gas_lift'
    assert lifted == ('injection_md' in w) == ('w_lg_max' in w) and (lifted or not w['needs_lift'])
    if lifted:
        H = w['z_inj']
        assert between(w['packer_offset'], 30, 150) and between(H, 0.3 * L, L - w['packer_offset'])
        assert w['injection_md'] == float(np.interp(H, tvd, md)) and H - 1e-9 <= w['injection_md'] < md[-1]
        assert between(w['p_inj'], *P_INJ[wt]) and between(w['dp_valve'], 3.4, 13.8) and between(w['phi_lg'], 0, 0.3)
        T_a = T_s + (T_r - T_s) * H / L
        assert w['T_lg'] == pytest.approx(T_a - w['phi_lg'] * (T_a - T_s), rel=1e-12) and T_s <= w['T_lg'] <= T_a
        assert between(w['w_lg_max'], 0.1, 3.5)
        if 'lift_capacity_at_bound' not in w['flags']:
            assert between(w['glr_cap'], 30, 1000)

    # Heat (SMP-81, SMP-82)
    assert between(w['U_t'], 11, 85) and between(w['k_e'], 1.5, 2.6) and between(w['age_years'], *AGE[stage])
    assert between(w['h'], 2, 40) if tail else 0 < w['h'] < w['U_t']
    S = 1000 * sal
    assert w['cp_w'] == pytest.approx(4190 - 5.62 * S + 0.0076 * S**2, rel=1e-12)
    t_F = (T_s + 0.8 * (T_r - T_s) - 273.15) * 1.8 + 32
    gamma_o = 141.5 / (131.5 + w['api'])
    assert w['cp_o'] == pytest.approx(4186.8 * (0.388 + 0.00045 * t_F) / math.sqrt(gamma_o), rel=1e-12)
    factor = (0.554 / w['sg_gas']) ** 0.4
    assert between(w['cp_g'], 2236 * factor, 3549 * factor)  # Methane's table, times the factor for heavier gases


# --- Bounds ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('part', ['core', 'tail'])
def test_no_draw_breaks_a_bound(wells, part):
    """v2: checks, Bounds: on core and tail wells apart."""
    drawn = [(w, st) for w, st in wells if w['coverage_tail'] == part]
    assert len(drawn) > 30
    for w, (md, tvd) in drawn:
        check_well(w, md, tvd)


def test_every_branch_is_drawn(wells):
    """The wells of the tests hold every label value and every family, so that the bounds above check them."""
    for label, values in LABELS.items():
        assert {w[label] for w, _ in wells} == values, label
    assert {w['lift'] for w, _ in wells if not is_oil(w)} == {'natural', 'gas_lift'}
    assert any(w['geometry_family'] == 'high_angle' and is_oil(w) for w, _ in wells)
    assert any(w['geometry_family'] == 'high_angle' and not is_oil(w) for w, _ in wells)


def test_lift_by_need_and_stage(wells):
    """SMP-78: an oil well whose static column cannot reach the surface is lifted, and a core gas well is not
    lifted at the start of its life."""
    ws = [w for w, _ in wells]
    assert any(w['needs_lift'] for w in ws) and all(w['lift'] == 'gas_lift' for w in ws if w['needs_lift'])
    assert not any(w['needs_lift'] for w in ws if not is_oil(w))
    starting = [w for w in ws if w['coverage_tail'] == 'core' and not is_oil(w) and w['life_stage'] == 'initial']
    assert starting and all(w['lift'] == 'natural' for w in starting)


class Extreme:
    """A stand-in for a block's generator that draws the top, or the bottom, of every range."""

    def __init__(self, top):
        self.top = top

    def random(self):
        return 1 - 1e-12 if self.top else 0.0

    def uniform(self, lo, hi):
        return hi if self.top else lo


@pytest.mark.parametrize('part', ['core', 'tail'])
def test_truncations_of_pressure_and_temperature(part):
    """SMP-63, SMP-65: the blocks at the ends of their draws, where the truncations bind: p_i at 950 bar, lambda at
    0.85 (0.8 in the tail), an underpressured reservoir at 0.8 of hydrostatic, T_r at 450 K, and the gradient over
    the modelled well at 10 K/km."""
    top, bottom = Extreme(True), Extreme(False)
    deep = dict(coverage_tail=part, z=6000.0, d_w=1500.0, rho_w=1100.0)  # p_h near 800 bar: the 950 bar cap binds
    x = SimpleNamespace(flags=[])
    draws._pressure(top, deep, x)
    assert deep['p_i'] == pytest.approx(950) and deep['lam'] < 0.25 and x.flags == ['p_i_cap_binds']
    shallow = dict(coverage_tail=part, z=2500.0, d_w=100.0, rho_w=1040.0)
    draws._pressure(top, shallow, SimpleNamespace(flags=[]))
    lam = (shallow['p_i'] - shallow['p_h']) / (shallow['p_lith'] - shallow['p_h'])
    assert lam == pytest.approx(0.8 if part == 'tail' else 0.85) and shallow['pressure_class'] == 'over'
    draws._pressure(bottom, shallow, SimpleNamespace(flags=[]))  # The first number decides: underpressured
    assert shallow['p_i'] == 0.8 * shallow['p_h'] and shallow['pressure_class'] == 'under' and shallow['lam'] == 0

    riser = dict(coverage_tail=part, well_type='platform', z=700.0, d_w=1900.0, L=2600.0)  # A long riser
    x = SimpleNamespace(flags=[])
    draws._temperature(bottom, riser, x)
    assert riser['G'] == pytest.approx(10 * 2600 / 700) and riser['G_eff'] == pytest.approx(10)
    assert riser['T_s'] == pytest.approx(272 + 7 * math.exp(-1900 / 400)) and 'G_raised_for_G_over_L' in x.flags
    hot = dict(coverage_tail=part, well_type='subsea', z=6400.0, d_w=300.0, L=6400.0)
    x = SimpleNamespace(flags=[])
    draws._temperature(top, hot, x)
    assert hot['T_r'] == pytest.approx(450) and hot['G'] < 55 and x.flags == ['T_r_cap_binds']
    onshore = dict(coverage_tail=part, well_type='onshore', z=1500.0, d_w=0.0, L=1500.0)
    draws._temperature(top, onshore, SimpleNamespace(flags=[]))
    assert onshore['T_s'] == (308 if part == 'tail' else 303) and onshore['G'] == pytest.approx(55)


@pytest.mark.xfail(
    strict=True,
    reason='SMP-63, an open finding of R10: where the hydrostatic pressure alone is above 950 bar, a deep reservoir '
    'under deep water (about 1 draw in 10 000, most of them tail subsea wells), lambda_max is floored at 0.001 and '
    'p_i exceeds the cap that the rule and v2: checks state',
)
def test_initial_pressure_cap_where_the_hydrostatic_pressure_exceeds_it():
    w = dict(coverage_tail='tail', z=6900.0, d_w=2300.0, rho_w=1150.0)
    draws._pressure(Extreme(True), w, SimpleNamespace(flags=[]))
    assert w['p_h'] > 950
    assert w['p_i'] <= 950


# --- Shares and marginals --------------------------------------------------------------------------------------------
def share_failures(wells):
    """The label shares of SMP-53 to SMP-59, SMP-63 and SMP-77 that lie more than 4 standard errors off."""
    ws = [w for w, _ in wells]
    core = [w for w in ws if w['coverage_tail'] == 'core']
    tail = [w for w in ws if w['coverage_tail'] == 'tail']
    checks = [('coverage tail', ws, lambda w: w['coverage_tail'] == 'tail', 0.20)]

    def categorical(name, subset, key, shares):
        checks.extend((f'{name} {key} {v}', subset, lambda w, v=v: w[key] == v, p) for v, p in shares.items())

    categorical('core', core, 'well_type', dict(platform=0.55, subsea=0.35, onshore=0.10))
    categorical('core', core, 'fluid_type', dict(black_oil=0.55, volatile_oil=0.10, gas_condensate=0.15, gas=0.20))
    categorical('core', core, 'life_stage', dict(initial=0.25, mid_life=0.40, late=0.35))
    categorical('tail', tail, 'well_type', dict.fromkeys(LABELS['well_type'], 1 / 3))
    categorical('tail', tail, 'fluid_type', dict.fromkeys(LABELS['fluid_type'], 1 / 4))
    categorical('tail', tail, 'life_stage', dict.fromkeys(LABELS['life_stage'], 1 / 3))
    categorical('all', ws, 'geometry_class', dict.fromkeys(LABELS['geometry_class'], 1 / 4))
    categorical('all', ws, 'chk_profile', dict.fromkeys(LABELS['chk_profile'], 1 / 4))

    def support(w):
        return w['drive'] == 'pressure_support'

    def dry(w):
        return w['fluid_class'] == 'dry_gas'

    deviated = [w for w in ws if w['geometry_class'] == 'deviated']
    highly_deviated = [w for w in ws if w['geometry_class'] == 'highly_deviated']
    platforms = [w for w in core if w['well_type'] == 'platform']
    checks += [
        ('core dry gas', [w for w in core if w['fluid_type'] == 'gas'], dry, 0.07),
        ('tail dry gas', [w for w in tail if w['fluid_type'] == 'gas'], dry, 0.25),
        ('core oils with support', [w for w in core if is_oil(w)], support, 0.7),
        ('core gases with support', [w for w in core if not is_oil(w)], support, 0.4),
        ('tail with support', tail, support, 0.5),
        ('core underpressured', core, lambda w: w['pressure_class'] == 'under', 0.04),
        ('tail underpressured', tail, lambda w: w['pressure_class'] == 'under', 0.10),
        ('S-shaped of the deviated', deviated, lambda w: w['geometry_family'] == 's_shaped', 1 / 3),
        ('drains of the highly deviated', highly_deviated, lambda w: w['geometry_family'] == 'high_angle', 1 / 2),
        ('deep-water platforms', platforms, lambda w: w['d_w'] > 400, 0.15),
    ]
    failures = []
    for name, subset, hit, p in checks:
        n, hits = len(subset), sum(map(hit, subset))
        if abs(hits - n * p) > 4 * math.sqrt(n * p * (1 - p)) + 0.5:
            failures.append(f'{name}: {hits} of {n}, expected share {p:.3f}')
    return failures


def truncated_normal(mu, sd, lo, hi):
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    return lambda x: (ndtr((x - mu) / sd) - a) / (b - a)


def truncated_lognormal(median, s, lo, hi):
    cdf = truncated_normal(math.log(median), s, math.log(lo), math.log(hi))
    return lambda x: cdf(np.log(x))


def uniform(lo, hi):
    return lambda x: (x - lo) / (hi - lo)


def log_uniform(lo, hi):
    return lambda x: (np.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))


def marginal_failures(wells):
    """The draws whose sample lies further from its distribution than Kolmogorov's statistic allows at the 0.1%
    level: a few draws of each kind, core and tail."""
    ws = [w for w, _ in wells]
    core = [w for w in ws if w['coverage_tail'] == 'core']
    tail = [w for w in ws if w['coverage_tail'] == 'tail']
    black = [w for w in core if w['fluid_class'] == 'black_oil']
    offshore = [w for w in core if w['well_type'] != 'onshore']
    platform_oils = [w for w in core if w['well_type'] == 'platform' and is_oil(w)]
    checks = [
        ('z, core black oils', black, 'z', truncated_lognormal(2674, 0.317, 600, 6500)),  # SMP-57
        ('salinity, core', core, 'salinity', truncated_lognormal(0.06, 0.6, 0.002, 0.25)),  # SMP-62
        ('p_sep, core offshore', offshore, 'p_sep', truncated_lognormal(35, 0.45, 5, 85)),  # SMP-66
        ('API, core black oils', black, 'api', truncated_normal(33, 4.5, 16, 45)),  # SMP-67
        ('gas gravity, core black oils', black, 'sg_gas', truncated_normal(0.70, 0.07, 0.58, 0.95)),
        ('permeability, core platform oils', platform_oils, 'perm', truncated_lognormal(15, 1.2, 0.5, 3000)),  # SMP-71
        ('net pay, core', core, 'h_net', truncated_lognormal(15, 0.8, 3, 100)),
        ('skin, core', core, 'skin', truncated_normal(2, 3, -3, 15)),
        ('design drawdown', ws, 'delta_d', uniform(0.1, 0.4)),
        ('C of the erosional velocity', ws, 'c_e', uniform(100, 150)),  # SMP-75
        ('discharge coefficient', ws, 'c_d', uniform(0.75, 0.9)),  # SMP-77
        ('U_t', ws, 'U_t', log_uniform(11, 85)),  # SMP-81
        ('k_e', ws, 'k_e', uniform(1.5, 2.6)),
        ('z, tail', tail, 'z', log_uniform(600, 7000)),
        ('salinity, tail', tail, 'salinity', log_uniform(0.002, 0.25)),
        ('skin, tail', tail, 'skin', uniform(-3, 15)),
        ('p_r / p_i, tail', tail, 'r_p', uniform(0.15, 1.0)),  # SMP-69
        ('h, tail', tail, 'h', uniform(2, 40)),  # SMP-81
    ]
    failures = []
    for name, subset, key, cdf in checks:
        x = np.sort([w[key] for w in subset])
        n, F = len(x), cdf(x)
        i = np.arange(1, n + 1)
        D = max(np.max(i / n - F), np.max(F - (i - 1) / n))
        if n < 30 or D * math.sqrt(n) > 1.95:
            failures.append(f'{name}: n = {n}, sqrt(n) D = {D * math.sqrt(n):.2f}')
    return failures


def test_label_shares(wells):
    """v2: checks, Shares: each label's share within its sampling error of the rule's."""
    assert share_failures(wells) == []


def test_marginals(wells):
    """v2: checks, Marginals: a few draws' distributions within their sampling error of the rule's."""
    assert marginal_failures(wells) == []


@pytest.mark.slow
def test_shares_marginals_and_bounds_on_many_wells():
    """The same checks on 3000 wells, whose sampling errors are a third as wide, and which reach the rare branches."""
    many = drawn_wells(N_MANY)
    for w, (md, tvd) in many:
        check_well(w, md, tvd)
    assert share_failures(many) == []
    assert marginal_failures(many) == []
    assert not any('geometry_to_vertical' in w['flags'] for w, _ in many)  # SMP-60: no well reaches the fallback


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def test_rank_correlations_of_core_black_oils(wells):
    """v2: checks, Rank correlations: a check of realism, not a bound, so only their signs are tested. Depth carries
    the pressure and the temperature, and the initial GOR follows the pressure, the depth and the API gravity."""
    black = [w for w, _ in wells if w['coverage_tail'] == 'core' and w['fluid_class'] == 'black_oil']
    assert len(black) > 100
    for a, b in (('z', 'p_i'), ('z', 'T_r'), ('R_i', 'p_i'), ('R_i', 'T_r'), ('R_i', 'z'), ('api', 'R_i')):
        assert spearman([w[a] for w in black], [w[b] for w in black]) > 0, (a, b)


# --- Reproducibility -------------------------------------------------------------------------------------------------
def test_a_rerun_gives_the_same_draws(wells):
    """SMP-109: a well's draws depend on the seed and its index only, whatever was drawn before it."""
    for k in (41, 7, 199):
        assert draw_well(SEED, k) == wells[k]
    assert draw_well(SEED + 1, 7)[0] != wells[7][0]


def test_a_well_drawn_alone_in_another_process_is_the_same(wells):
    """SMP-109: the same draws in a process that draws no other well, a saturated oil with its material balance."""
    k = next(w['well'] for w, _ in wells if is_oil(w) and w['p_r'] < w['p_sat'])
    code = f'import json; from manywells.sampling.v2.draws import draw_well; print(json.dumps(draw_well({SEED}, {k})))'
    env = dict(os.environ, PYTHONHASHSEED='20261110')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, env=env).stdout
    assert json.loads(out) == json.loads(json.dumps(wells[k]))


def test_each_block_has_its_own_generator(wells, monkeypatch):
    """SMP-51, SMP-109: the blocks' generators, in their order, each seeded by (seed, k, 'v2', block), so that a
    change to one block's draws moves no other block's: with the depth block drawn from another seed, the other
    blocks' own numbers stay."""
    k = next(w['well'] for w, _ in wells if w['coverage_tail'] == 'core' and not is_oil(w))
    base = wells[k][0]
    asked = []

    def other_depth(seed, *parts):
        asked.append(parts)
        return rng_for(seed + 1 if parts[-1] == 'depth' else seed, *parts)

    monkeypatch.setattr(draws, 'rng_for', other_depth)
    w, _ = draw_well(SEED, k)
    assert all(parts[:2] == (k, 'v2') and len(parts) == 3 for parts in asked)
    blocks = [parts[2] for parts in asked]
    assert blocks == BLOCKS.replace('inflow', 'inflow nondarcy').split()  # A gas well's blocks
    assert w['z'] != base['z'] and w['p_i'] != base['p_i']  # The depth, and what follows from it
    unmoved = 'coverage_tail well_type fluid_class drive life_stage geometry_class salinity rho_w p_sep T_sep api'
    unmoved += ' sg_gas perm h_net skin delta_d perf_share c_e c_d chk_profile U_t k_e'
    for key in unmoved.split():
        assert w[key] == base[key], key


BLOCK_NUMBERS = dict(  # The numbers that each block takes of its generator, for every well (SMP-51)
    labels=6,
    depth=3,
    geometry=2 + 20 * 6,  # The class and the family, and 20 surveys of six numbers
    water=1,
    pressure=2,
    temperature=3,
    separator=7,
    fluid=5,
    stage=14,
    inflow=7,
    nondarcy=2,  # A gas well's block
    completion=3,
    choke=3,
    lift=7,
    heat=4,
)


def test_every_block_takes_the_same_numbers_for_every_well(wells, monkeypatch):
    """SMP-51 (D-108): each block takes the same count of its generator's numbers, call by call, for every well,
    whatever its branches: core and tail, every well type, fluid class, drive, stage and geometry family, a survey
    redrawn or not, saturated or not, lifted or natural. Only gas wells have the `nondarcy` block."""
    made = counting(monkeypatch, draws)
    cases = [(w0['well'], None, False) for w0, _ in wells]
    cases += [(k, stage, force) for k in range(30) for stage in STAGES for force in (False, True)]
    calls, seen = {}, set()
    for k, stage, force in cases:
        del made[:]
        w, _ = draw_well(SEED, k, stage, force_lift=force)
        blocks = [parts[2] for parts, _ in made]
        gas = not is_oil(w)
        assert blocks == BLOCKS.replace('inflow', 'inflow nondarcy' if gas else 'inflow').split()
        for (_, _, block), counts in made:
            assert calls.setdefault(block, counts) == counts, (k, block)
        seen |= {w['coverage_tail'], w['well_type'], w['fluid_class'], w['drive'], w['life_stage'], w['lift']}
        seen |= {w['geometry_family'], w['pressure_class'], w['saturation'], *filter(None, w['flags'].split(';'))}
    assert {block: sum(counts) for block, counts in calls.items()} == BLOCK_NUMBERS
    assert all(count == 1 for counts in calls.values() for count in counts)  # A call takes one number
    branches = set().union(*(LABELS[key] for key in ('coverage_tail', 'well_type', 'fluid_class', 'drive')))
    branches |= set().union(*(LABELS[key] for key in ('life_stage', 'lift', 'geometry_family', 'pressure_class')))
    assert seen >= branches | LABELS['saturation'] | {'geometry_redrawn', 'lift_not_installed_strong_well'}


def test_a_branchs_constant_moves_no_later_draw(wells, monkeypatch):
    """SMP-51 (D-108): where a constant that decides a branch changes, the block's later numbers are as they were.
    The dry-gas share (`labels`): the drive and the stage stay. The saturated shares (`fluid`): the initial GOR's
    number stays. The lift's need and shares (`lift`): a lifted well's design stays."""
    base = [w for w, _ in wells[:120]]

    def redrawn():
        return [draw_well(SEED, w['well'])[0] for w in base]

    with monkeypatch.context() as m:
        m.setattr(draws, '_P_DRY_GAS', 0.9)
        other = redrawn()
    assert any(a['fluid_class'] != b['fluid_class'] for a, b in zip(base, other))
    for key in ('coverage_tail', 'well_type', 'fluid_type', 'drive', 'life_stage'):
        assert [w[key] for w in other] == [w[key] for w in base], key

    with monkeypatch.context() as m:
        m.setattr(draws, '_P_SATURATED', dict.fromkeys(draws._P_SATURATED, 1.0))
        other = redrawn()
    drawn_gor = [i for i, w in enumerate(base) if w['fluid_class'] != 'black_oil']
    assert any(other[i]['p_sat'] != base[i]['p_sat'] for i in drawn_gor) and len(drawn_gor) > 30
    assert all(w['p_sat'] is None or w['p_sat'] == w['p_i'] or 'p_b_lowered' in w['flags'] for w in other)
    for key in ('api', 'sg_gas'):
        assert [w[key] for w in other] == [w[key] for w in base], key
    assert [other[i]['R_i'] for i in drawn_gor] == [base[i]['R_i'] for i in drawn_gor]

    with monkeypatch.context() as m:  # No well needs lift, and the draw lifts every well
        m.setattr(draws, '_needs_lift', lambda *args: False)
        m.setattr(draws, '_P_LIFT_OIL', dict.fromkeys(draws._P_LIFT_OIL, (1.0, 1.0, 1.0)))
        m.setattr(draws, '_P_LIFT_GAS', (1.0, 1.0, 1.0))
        other = redrawn()
    lifted = [(a, b) for a, b in zip(base, other) if a['lift'] == b['lift'] == 'gas_lift']
    assert sum(a['needs_lift'] for a, _ in lifted) >= 3 and sum(not a['needs_lift'] for a, _ in lifted) >= 3
    assert sum(b['lift'] == 'gas_lift' for b in other) > sum(a['lift'] == 'gas_lift' for a in base)
    for key in ('packer_offset', 'p_inj', 'dp_valve', 'phi_lg'):
        assert [b[key] for _, b in lifted] == [a[key] for a, _ in lifted], key


def test_v2_generators_are_not_v1s():
    """SMP-109: every v2 key holds 'v2', and drawing a v2 well leaves v1's draws (SMP-31) as they were."""
    before = sample_well(SEED, 3)
    draw_well(SEED, 3)
    assert sample_well(SEED, 3) == before
    assert rng_for(SEED, 3, 'v2', 'labels').random() != rng_for(SEED, 3, 'well').random()


STAGE_FREE = (  # The draws that do not depend on the life stage
    'coverage_tail well_type fluid_type fluid_class class_rule drive z d_w L geometry_class geometry_family md md_tvd '
    'theta_max theta_end salinity rho_w p_h p_lith lam p_i pressure_class grad_i T_s G T_r G_eff p_sep T_sep dp_fl api '
    'sg_gas rho_o rho_g R_i p_sat w_kgko c_k perm h_net skin ln_re_rw delta_d drain_length I_ani J_H_over_J_V '
    'perf_share c_e c_d choke_model chk_profile U_t k_e cp_o cp_w'
).split()
STAGE_FREE_OIL = 'q_design D tubing q_cap beta d_bean K_c'.split()  # An oil's design point is its initial state


def test_a_stage_override_changes_only_what_depends_on_the_stage(wells):
    """SMP-51, The same well at another stage: the same generators, so the survey and every draw that does not
    depend on the stage stay, and the well at each stage keeps the bounds. A tail well's draws are its tail forms,
    the same at every stage but for the separator's factor and the well's age."""
    for w0, st0 in wells[:60]:
        k, tail = w0['well'], w0['coverage_tail'] == 'tail'
        assert draw_well(SEED, k, w0['life_stage']) == (w0, st0)  # Its drawn stage, set again
        for stage in STAGES:
            w, st = draw_well(SEED, k, stage)
            assert w['life_stage'] == stage and st == st0
            check_well(w, *st)
            keys = [key for key in w0 if key.startswith('geo_')] + STAGE_FREE + (STAGE_FREE_OIL if is_oil(w0) else [])
            if tail:
                keys = [key for key in w0 if key not in ('life_stage', 'r_s', 'age_years')]
                assert set(w) == set(w0)
            for key in keys:
                assert w.get(key) == w0.get(key), (k, stage, key)
    with pytest.raises(ValueError, match='stage'):
        draw_well(SEED, 0, 'mid-life')


def test_force_lift_gives_a_design_and_moves_no_other_draw(wells):
    """SMP-90's design of a well whose lift starts later: the lift block's design, and no other draw changed."""
    natural = [w for w, _ in wells if w['lift'] == 'natural'][:25]
    for w0 in natural:
        w, st = draw_well(SEED, w0['well'], force_lift=True)
        assert w['lift'] == 'gas_lift' and 0.1 <= w['w_lg_max'] <= 3.5 and 0 < w['injection_md'] < w['md']
        check_well(w, *st)
        assert all(w[key] == w0[key] for key in w0 if key not in ('lift', 'flags'))


def test_a_table_fluid_replaces_the_fluid_blocks_values(wells):
    """SMP-115: a table's saturation pressure, initial GOR and densities replace the fluid block's after its draws,
    so that no draw moves and the later blocks follow the table; its API and gas gravity replace the drawn ones
    where the fluid has its own."""
    w0, st0 = next((w, st) for w, st in wells if w['fluid_class'] == 'gas_condensate' and w['coverage_tail'] == 'core')
    k = w0['well']
    table = dict(p_sat=0.9 * w0['p_sat'], R_i=1.2 * w0['R_i'], rho_o=w0['rho_o'] + 10, rho_g=1.05 * w0['rho_g'])
    w, st = draw_well(SEED, k, fluid=table)
    assert st == st0 and all(w[key] == value for key, value in table.items())
    assert (w['api'], w['sg_gas']) == (w0['api'], w0['sg_gas'])
    own_numbers = 'z p_i T_r p_sep T_sep p_s r_p p_r wgr c_k perm h_net skin delta_d perf_share c_e c_d U_t k_e cp_o'
    for key in own_numbers.split():
        assert w[key] == w0[key], key
    assert w['R_go'] == table['R_i'] * (1 + w['c_k'] * max(0.0, 1 - w['p_r'] / table['p_sat'])) != w0['R_go']
    assert w['ogr'] == 1e6 / w['R_go'] and w['cp_g'] == w0['cp_g']
    w, _ = draw_well(SEED, k, fluid=dict(table, api=50.0, sg_gas=0.8))
    assert (w['api'], w['sg_gas']) == (50.0, 0.8) and w['cp_o'] != w0['cp_o'] and w['cp_g'] != w0['cp_g']
    oil0 = next(w for w, _ in wells if w['fluid_class'] == 'volatile_oil')
    table = dict(p_sat=1.0, R_i=oil0['R_i'], rho_o=oil0['rho_o'], rho_g=oil0['rho_g'])  # Two phases nowhere
    w, _ = draw_well(SEED, oil0['well'], fluid=table)
    assert w['saturation'] == 'undersaturated' and w['R_go'] == w['R_i']


# --- The survey and the stream ---------------------------------------------------------------------------------------
def test_survey_vertical_and_refusals():
    rng = np.random.default_rng(1)
    assert survey(rng, 'vertical', 1234.0) == ([0.0, 1234.0], [0.0, 1234.0], {})
    assert survey(rng, 'build_and_hold', 300.0, (60.0, 75.0)) is None  # No room for the build above 300 m
    with pytest.raises(ValueError, match='family'):
        survey(rng, 'l_shaped', 2000.0)
    assert set(FAMILIES) == LABELS['geometry_family']


@pytest.mark.parametrize('family', ['build_and_hold', 's_shaped', 'high_angle'])
def test_survey_shapes(family):
    """SMP-60: each family's sections, their ranges, the kickoff's range and the least hold."""
    theta = {'build_and_hold': (20.0, 60.0), 's_shaped': None, 'high_angle': (60.0, 75.0)}[family]
    rng = np.random.default_rng(5)
    fitted = 0
    for _ in range(150):
        L, d_mud = rng.uniform(1200, 6000), float(rng.choice([0.0, 400.0]))
        res = survey(rng, family, L, theta, d_mud)
        if res is None:
            continue
        fitted += 1
        md, tvd, prm = res
        z, K = L - d_mud, prm['kickoff']
        incl = np.degrees(np.arccos(np.clip(np.diff(tvd) / np.diff(md), -1, 1)))
        assert md[0] == tvd[0] == 0.0 and md[1] == tvd[1] == K and tvd[-1] == pytest.approx(L, abs=1e-6)
        assert np.all(np.diff(md) > 0) and np.all(np.diff(tvd) > 0)
        assert between(K, d_mud + 0.05 * z, d_mud + 0.75 * z) and between(prm['build_rate'], 0.3, 3.6)
        th = prm['theta_hold']
        if family == 'build_and_hold':
            assert between(th, *theta) and incl.max() == pytest.approx(th, abs=1e-6)
            assert len(md) == 3 + int(th)  # The wellhead, the kickoff, a station per degree of the build, the bottom
            assert tvd[-1] - tvd[-2] >= 50 - 1e-6
        elif family == 's_shaped':
            assert between(th, 15, 45) and incl.max() == pytest.approx(th, abs=1e-6)
            assert prm['theta_end'] <= min(35, th) and incl[-1] == pytest.approx(prm['theta_end'], abs=1e-6)
            assert between(prm['drop_rate'], 0.2, 2.2) and prm['end_section_tvd'] <= min(400, L / 4)
            assert tvd[-1] - tvd[-2] == pytest.approx(prm['end_section_tvd'], abs=1e-6)
        else:
            assert between(prm['theta_heel'], *theta) and th <= prm['theta_heel'] and between(th, 10, 65)
            assert incl.max() == incl[-1] == pytest.approx(prm['theta_heel'], abs=1e-6)
            assert between(prm['build_rate_2'], 1.5, 7) and between(prm['tangent_md'], 50, 300)
            assert md[-1] - md[-2] == pytest.approx(prm['tangent_md'], abs=1e-6)
    assert fitted > 100


def test_no_slip_stream():
    """The stream per Sm3 of stock-tank liquid: its mass, and its volume with and without free gas."""
    bo = BlackOil(rho_o=850.0, rho_g=1.0, gas_law='dak', p_sep=20.0, T_sep=310.0)
    gor, wlr, rho_w, T = 100.0, 0.25, 1050.0, 350.0
    oil = bo.at_gor(gor)
    vol, mass = stream(bo, gor, wlr, rho_w, 400.0, T)  # All the gas in solution
    assert mass == 0.75 * (850.0 + 100.0 * 1.0) + 0.25 * 1050.0
    assert oil.rs(400.0, T) == pytest.approx(gor) and vol == pytest.approx(0.75 * oil.bo(400.0, T) + 0.25)
    vol_low, mass_low = stream(bo, gor, wlr, rho_w, 10.0, T)
    free = 0.75 * (gor - oil.rs(10.0, T)) * 1.0 / oil.gas_density(10.0, T)
    assert mass_low == mass and vol_low == pytest.approx(0.75 * oil.bo(10.0, T) + 0.25 + free) and vol_low > 2 * vol
