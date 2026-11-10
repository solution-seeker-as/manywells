"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The v2 well draws (specs/sampling.md, SMP-51 to SMP-82): a well's labels, depths, survey, fluid, reservoir state at a
life stage, inflow, completion, choke, gas lift and heat transfer, drawn in blocks. Each block has a generator of its
own, rng_for(seed, k, 'v2', block) (SMP-109), and may use the blocks above it. A block takes the same numbers of its
generator in the same order for every well: a number that the well's branch does not use is drawn and left unused, so
that a change of one rule moves no other draw (SMP-51). Nothing here solves a well.

draw_well gives a well's draws as a dict, with its survey. The dict holds the labels (SMP-52), the drawn numbers and
the values derived from them, in the model's units (m, bar, K, kg/s, kg/m3, Sm3/Sm3); rates are Sm3/d. Its keys are
listed at draw_well.
"""

import math
from types import SimpleNamespace

import numpy as np

from manywells.choke import CHOKE_PROFILES
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.sampling.v2._distributions import logistic, logu, pick, tlognorm, tnorm, unused
from manywells.sampling.v2.reservoir import (
    OILS,
    drain_index,
    forchheimer,
    gas_rate,
    j_radial,
    joshi_ratio,
    liner_gas,
    liner_liquid,
    nondarcy,
    producing_gor,
    slant_skin,
)
from manywells.sampling.v2.survey import N_NUMBERS, survey
from manywells.sampling.wells import rng_for
from manywells.units import P_REF, STD_GRAVITY

STAGES = ('initial', 'mid_life', 'late')  # spec: SMP-56
P_ATM = P_REF / 1e5  # bar
P_S_MIN = 1.5  # bar, the floor of the nominal pressure downstream of the choke  # spec: SMP-66
P_S_MAX = {'platform': 120.0, 'subsea': 250.0, 'onshore': 120.0}  # bar, its cap  # spec: SMP-66
TUBING = {  # Inner diameter (m) by nominal size (in)  # spec: SMP-75
    '2-7/8': 0.0620,
    '3-1/2': 0.0760,
    '4-1/2': 0.1005,
    '5-1/2': 0.1243,
    '7': 0.1571,
    '9-5/8': 0.2205,
}

_TAIL_SHARE = 0.20  # spec: SMP-53
_WELL_TYPES, _P_WELL_TYPE = ('platform', 'subsea', 'onshore'), (0.55, 0.35, 0.10)  # spec: SMP-54
_FLUID_TYPES, _P_FLUID = ('black_oil', 'volatile_oil', 'gas_condensate', 'gas'), (0.55, 0.10, 0.15, 0.20)
_P_DRY_GAS = 0.07  # Share of dry gas among the wet and dry gas wells  # spec: SMP-55
_P_STAGE = (0.25, 0.40, 0.35)  # spec: SMP-56
_DEPTH = {  # Median (m below the mudline) and log standard deviation  # spec: SMP-57
    'black_oil': (2674, 0.317),
    'volatile_oil': (3159, 0.315),
    'gas_condensate': (3424, 0.278),
    'wet_gas': (2419, 0.422),
    'dry_gas': (1141, 0.586),
}
_P_DEEP_TREE = 0.10  # Share of deep-water dry trees among a core's platform wells  # spec: SMP-58
_GEOMETRY_CLASSES = ('vertical', 'slightly_deviated', 'deviated', 'highly_deviated')  # spec: SMP-59
_P_GEOMETRY = (0.25, 0.25, 0.25, 0.25)
_THETA_CLASS = {'slightly_deviated': (5.0, 20.0), 'deviated': (20.0, 60.0), 'highly_deviated': (60.0, 75.0)}
_P_S_SHAPED = 1 / 3  # Share of the deviated class drawn as S-shaped wells
_P_DRAIN = 0.5  # Share of the highly deviated class drawn as drains to the heel
_MD_MAX = 9342.0  # m  # spec: SMP-60
_GEOMETRY_REDRAWS = 20  # spec: SMP-60
_RHO_SEA = 1025.0  # kg/m3  # spec: SMP-63
_LITHO = 0.226  # bar/m, the lithostatic gradient below the mudline  # spec: SMP-63
_P_MAX, _T_MAX = 950.0, 450.0  # bar, K  # spec: SMP-63, SMP-65
_G_EFF_MIN = 10.0  # K/km over the modelled well  # spec: SMP-65
_RISER = (0.03, 0.07)  # bar/m of water depth, a subsea well's flowline and riser  # spec: SMP-66
_API = {'volatile_oil': (40, 50), 'gas_condensate': (42, 58), 'wet_gas': (45, 58), 'dry_gas': (45, 58)}
_GAS_GRAVITY = {  # Mean, standard deviation and bounds  # spec: SMP-67
    'black_oil': (0.70, 0.07, 0.58, 0.95),
    'volatile_oil': (0.75, 0.06, 0.62, 1.0),
    'gas_condensate': (0.72, 0.05, 0.60, 0.90),
    'wet_gas': (0.62, 0.035, 0.571, 0.75),
    'dry_gas': (0.62, 0.035, 0.571, 0.75),
}
_P_SATURATED = {'black_oil': 0.20, 'volatile_oil': 0.40, 'gas_condensate': 0.30}  # spec: SMP-68
_R_BLACK_OIL = 356.0  # Sm3/Sm3, the black oils' largest initial GOR  # spec: SMP-55, SMP-68
_R_P = {  # p_r / p_i by stage and drive  # spec: SMP-69
    ('initial', 'depletion'): (0.85, 1.0),
    ('initial', 'pressure_support'): (0.97, 1.0),
    ('mid_life', 'depletion'): (0.55, 0.92),
    ('mid_life', 'pressure_support'): (0.85, 1.0),
    ('late', 'pressure_support'): (0.70, 1.0),
}
_R_P_DEPLETED = {True: (0.3, 0.6), False: (0.2, 0.55)}  # Late under depletion: oils, gas classes  # spec: SMP-69
_WGR = {  # Sm3 of water per 1e6 Sm3 of gas, by stage and drive  # spec: SMP-69
    ('initial', 'pressure_support'): (2, 20),
    ('initial', 'depletion'): (2, 20),
    ('mid_life', 'pressure_support'): (10, 150),
    ('mid_life', 'depletion'): (5, 50),
    ('late', 'pressure_support'): (50, 500),
    ('late', 'depletion'): (5, 80),
}
_PERM = {  # mD: median, log standard deviation and bounds by well type  # spec: SMP-71
    'platform': (15, 1.2, 0.5, 3000),
    'subsea': (60, 1.2, 0.5, 3000),
    'onshore': (3, 1.5, 0.05, 1000),
}
_GAS_K_FACTOR = 1 / 14  # The gas classes' effective permeability over the well type's  # spec: SMP-71
_DRAIN_LENGTH = (300.0, 2500.0)  # m  # spec: SMP-71
_PERF_SHARE = (0.3, 1.0)  # The perforated share of the net pay  # spec: SMP-71
_D_ERROR, _D_BOUNDS = 0.7, (1e-5, 2e-3)  # The non-Darcy estimate's log error, and its bounds (d/Mscf)  # spec: SMP-71
_LINER_MAX = 0.1571  # m, the largest liner of a gas well's drain (7 in)  # spec: SMP-73
_SLANT_FLOOR = 2.0  # The floor on ln(r_e/r_w) - 0.75 + s + s_theta of a slanted well  # spec: SMP-72
_ROUGHNESS = {'initial': (1.5e-6, 3e-5), 'mid_life': (1e-5, 6e-5), 'late': (2e-5, 1.5e-4)}  # m  # spec: SMP-76
_BEAN_MAX = 0.1016  # m, the fully open bean's largest diameter (4 in)  # spec: SMP-77
_P_LIFT_OIL = {'offshore': (0.20, 0.45, 0.70), 'onshore': (0.15, 0.25, 0.35)}  # By stage  # spec: SMP-78
_P_LIFT_GAS = (0.0, 0.05, 0.15)  # spec: SMP-78
_P_INJ = {'platform': (48.0, 85.0), 'subsea': (130.0, 210.0), 'onshore': (48.0, 85.0)}  # bar  # spec: SMP-79
_GILBERT_H = 12192.0  # m: Gilbert's 40,000 ft (Golan and Whitson 1991, Eq. 5.17)  # spec: SMP-79
_PSI_LG = (0.0, 0.75)  # The lift gas's share of the way from the ambient at the valve to T_r  # spec: SMP-79
_W_LG_MAX = (0.1, 3.5)  # kg/s, the lift-gas capacity's bounds  # spec: SMP-80
_AGE = {'initial': (30, 365), 'mid_life': (365, 8 * 365), 'late': (5 * 365, 25 * 365)}  # days  # spec: SMP-81
_CH4_CP_P, _CH4_CP_T = (1.0, 101.0, 201.0, 401.0), (300.0, 380.0)  # bar, K  # spec: SMP-82
_CH4_CP = ((2236.0, 3011.0, 3549.0, 3336.0), (2466.0, 2801.0, 3079.0, 3234.0))  # Methane's c_p, J/(kg K), [T][p]


def black_oil(w) -> BlackOil:
    """The black oil of a well's draws, as the draws and a black-oil well's mapping use it (SMP-111)."""
    return BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])


def stream(bo, gor, wlr, rho_w, p, T) -> tuple:
    """
    The no-slip stream at (p, T): its in-situ volume (m3) and mass (kg) per Sm3 of stock-tank liquid.

    :param bo: The well's black oil
    :param gor: Gas-oil ratio (Sm3/Sm3)
    :param wlr: Water-liquid ratio at standard conditions
    :param rho_w: Water density at standard conditions (kg/m3)
    :param p: Pressure (bar)
    :param T: Temperature (K)
    """
    b = bo.at_gor(gor)
    x_o = 1 - wlr
    free = max(gor - b.rs(p, T), 0.0) * bo.rho_g / b.gas_density(p, T)
    return x_o * b.bo(p, T) + wlr + x_o * free, x_o * (bo.rho_o + gor * bo.rho_g) + wlr * rho_w


def draw_well(seed: int, k: int, stage: str | None = None, force_lift: bool = False, fluid: dict | None = None):
    """
    The draws of well k of a dataset with seed `seed`: a dict of its labels, draws and derived values, and its survey.

    :param seed: The dataset's seed
    :param k: The well's draw index
    :param stage: A life stage (one of STAGES) that replaces the drawn one after the labels block has drawn it. Every
        generator draws the same numbers, so this is the same well at another stage of its life (SMP-51).
    :param force_lift: Give the well its lift design even where the lift block leaves it natural, as a well whose
        lift starts later in its life needs (SMP-90). It changes no other draw.
    :param fluid: A table fluid's values (SMP-115): its saturation pressure at T_r `p_sat` (bar), initial GOR `R_i`
        (Sm3/Sm3) and stock-tank densities `rho_o` and `rho_g` (kg/m3), and `api` and `sg_gas` where the fluid is not
        matched to the drawn ones. They replace the fluid block's after its draws, so that no draw moves and every
        later block follows the table.
    :return: (w, (md, tvd)): the well's draws, and its survey stations from the wellhead (m)

    The keys of w, by block (a key in brackets is there only for the wells it applies to):

    - `well`: k. `flags`: the truncations and redraws that applied, joined by ';'.
    - labels: `coverage_tail`, `well_type`, `fluid_type`, `fluid_class`, `class_rule`, `drive`, `life_stage`.
    - depth: `z`, `d_w`, `L`.
    - geometry: `geometry_class`, `geometry_family`, `md` (the bottom's), `md_tvd`, `theta_max`, `theta_end` (the
      survey's largest and last inclination, degrees) and the family's parameters `geo_*` (survey.survey).
    - water: `salinity`, `rho_w`. pressure: `p_h`, `p_lith`, `lam`, `p_i`, `pressure_class`, `grad_i`.
    - temperature: `T_s`, `G`, `T_r`, `G_eff` (K/km over L). separator: `p_sep`, `T_sep`, `r_s`, `dp_fl`, `p_s` (the
      nominal pressure downstream of the choke).
    - fluid: `api`, `sg_gas`, `rho_o`, `rho_g`, `R_i`, `p_sat` (None for wet and dry gas).
    - stage: `r_p`, `p_r`, `wlr`, `R_go`, `saturation`, `below_hydrostatic`, `ogr` and [`wgr`] (gas classes, else
      None and absent), [`w_kgko`] (oils), `c_k` (condensates, else None).
    - inflow: `perm`, `h_net`, `skin`, `ln_re_rw`, `delta_d`, [`drain_length`, `I_ani`, `J_H_over_J_V`] (drains),
      [`I_ani`, `s_theta`] (slanted wells), `f_geo`, `inflow`, `q_l_ref`; oils: `mu_o`, `B_o`, `J`, `k_l`, `w_l_max`, [`eta_drain`]; gas classes:
      `perf_share`, `D_eq434`, `D_nondarcy` (d/Mscf), `a_gas`, `b_gas` (bar2 per Sm3/d and its square), `a_ratio`,
      `b_ratio` (over a vertical well's), `AOF`, `q_g_ref`, [`b_friction_share`].
    - completion: `c_e`, `q_design`, `D`, `tubing`, `q_cap`, `roughness`.
    - choke: `beta`, `d_bean`, `c_d`, `K_c`, `choke_model`, `chk_profile`.
    - lift: `lift`, `needs_lift`, and for a lifted well `z_inj`, `injection_md`, `z_inj_frac`, `p_inj`, `dp_valve`,
      `packer_offset`, `T_lg`, `psi_lg` (its share of the way from the ambient to T_r), `glr_cap`, `w_lg_max`.
    - heat: `h`, `age_years`, `U_t`, `k_e`, `cp_o`, `cp_w`, `cp_g`.
    """
    if stage is not None and stage not in STAGES:
        raise ValueError(f'stage must be one of {STAGES} or None, got {stage!r}')
    w = {'well': k}
    x = SimpleNamespace(flags=[])  # Working values that the well's draws do not record

    def gen(block):  # spec: SMP-109
        return rng_for(seed, k, 'v2', block)

    # spec: SMP-51. The blocks, in their order
    _labels(gen('labels'), w, stage)
    _depth(gen('depth'), w)
    _geometry(gen('geometry'), w, x)
    _water(gen('water'), w)
    _pressure(gen('pressure'), w, x)
    _temperature(gen('temperature'), w, x)
    _separator(gen('separator'), w)
    _fluid(gen('fluid'), w, x, fluid)
    _stage(gen('stage'), w)
    _inflow(gen, w, x)
    _completion(gen('completion'), w, x)
    _inflow_at_stage(w, x)
    _choke(gen('choke'), w, x)
    _lift(gen('lift'), w, x, force_lift)
    _heat(gen('heat'), w)
    w['flags'] = ';'.join(x.flags)
    return w, x.stations


def _is_tail(w) -> bool:
    return w['coverage_tail'] == 'tail'


def _labels(r, w, stage):  # spec: SMP-52
    tail = bool(r.random() < _TAIL_SHARE)  # spec: SMP-53
    wt = pick(r, _WELL_TYPES, (1, 1, 1) if tail else _P_WELL_TYPE)  # spec: SMP-54
    ft = pick(r, _FLUID_TYPES, (1, 1, 1, 1) if tail else _P_FLUID)  # spec: SMP-55
    u_dry = r.random()  # spec: SMP-51. Drawn for every well, whatever its fluid type
    fc = ft if ft != 'gas' else ('dry_gas' if u_dry < (0.25 if tail else _P_DRY_GAS) else 'wet_gas')
    p_support = 0.5 if tail else (0.7 if ft in OILS else 0.4)  # spec: SMP-56
    drive = 'pressure_support' if r.random() < p_support else 'depletion'
    drawn = pick(r, STAGES, (1, 1, 1) if tail else _P_STAGE)
    w.update(coverage_tail='tail' if tail else 'core', well_type=wt, fluid_type=ft, fluid_class=fc)
    w.update(class_rule='mccain_1990', drive=drive, life_stage=drawn if stage is None else stage)


def _depth(r, w):
    tail, wt = _is_tail(w), w['well_type']
    z = logu(r, 600, 7000) if tail else tlognorm(r, *_DEPTH[w['fluid_class']], 600, 6500)  # spec: SMP-57
    u_deep = r.random()  # spec: SMP-51. A deep-water dry tree's number, drawn for every well
    if wt == 'platform':  # spec: SMP-58
        if tail:
            d_w = logu(r, 5, 2000)
        else:
            d_w = logu(r, 400, 2000) if u_deep < _P_DEEP_TREE else tlognorm(r, 90, 0.6, 5, 400)
    elif wt == 'subsea':
        d_w = logu(r, 50, 3000) if tail else logu(r, 80, 3000)
    else:
        unused(r)  # The water depth's number
        d_w = 0.0
    w.update(z=z, d_w=d_w, L=z + d_w if wt == 'platform' else z)


def _geometry(r, w, x):
    L = w['L']
    d_mud = w['d_w'] if w['well_type'] == 'platform' else 0.0
    gcl = pick(r, _GEOMETRY_CLASSES, _P_GEOMETRY)  # spec: SMP-59
    sub = r.random()
    family = {
        'vertical': 'vertical',
        'slightly_deviated': 'build_and_hold',
        'deviated': 's_shaped' if sub < _P_S_SHAPED else 'build_and_hold',
        'highly_deviated': 'high_angle' if sub < _P_DRAIN else 'build_and_hold',
    }[gcl]
    for attempt in range(_GEOMETRY_REDRAWS):  # spec: SMP-60
        res = survey(r, family, L, _THETA_CLASS.get(gcl), d_mud)
        if res is not None and res[0][-1] <= _MD_MAX:
            md, tvd, prm = res
            break
        x.flags.append('geometry_redrawn')
    else:
        gcl, family, (md, tvd, prm) = 'vertical', 'vertical', ([0.0, L], [0.0, L], {})
        x.flags.append('geometry_to_vertical')
    unused(r, (_GEOMETRY_REDRAWS - 1 - attempt) * N_NUMBERS)  # spec: SMP-51. The numbers of the redraws not made
    incl = [  # spec: SMP-61
        math.degrees(math.acos(min(1.0, (tvd[i] - tvd[i - 1]) / (md[i] - md[i - 1])))) for i in range(1, len(md))
    ]
    w.update(geometry_class=gcl, geometry_family=family, md=md[-1], md_tvd=md[-1] / L)
    w.update(theta_max=max(incl), theta_end=incl[-1])
    w.update({f'geo_{key}': val for key, val in prm.items()})
    x.stations = (md, tvd)


def _water(r, w):  # spec: SMP-62
    sal = logu(r, 0.002, 0.25) if _is_tail(w) else tlognorm(r, 0.06, 0.6, 0.002, 0.25)  # Mass fraction of solids
    w.update(salinity=sal, rho_w=999.1 + 692.0 * sal + 311.0 * sal**2)


def _pressure(r, w, x):  # spec: SMP-63
    tail, z, d_w, rho_w = _is_tail(w), w['z'], w['d_w'], w['rho_w']
    p_h = P_ATM + (_RHO_SEA * d_w + rho_w * z) * STD_GRAVITY / 1e5
    p_lith = P_ATM + _RHO_SEA * d_w * STD_GRAVITY / 1e5 + _LITHO * z
    lam_cap = max(1e-3, min(0.85, (_P_MAX - p_h) / (p_lith - p_h)))
    mu = -math.log(1 / (0.02 + 0.45 * logistic((z / 1000 - 3.8) / 0.9)) - 1)  # The logit of the median
    if r.random() < (0.10 if tail else 0.04):
        p_i, lam, pclass = p_h * r.uniform(0.8, 1.0), 0.0, 'under'
    else:
        if tail:
            lam = r.uniform(0.0, max(0.0, min(0.8, lam_cap)))
        else:
            lam = logistic(tnorm(r, mu, 0.9, -12.0, math.log(lam_cap / (1 - lam_cap))))
        p_i = p_h + lam * (p_lith - p_h)
        pclass = 'over' if lam >= 0.25 else 'normal'
    if p_h > _P_MAX:  # The hydrostatic pressure alone passes the cap: the reservoir is held at it, underpressured
        p_i, lam, pclass = _P_MAX, 0.0, 'under'
    if lam_cap < logistic(mu + 1.645 * 0.9):
        x.flags.append('p_i_cap_binds')
    w.update(p_h=p_h, p_lith=p_lith, lam=lam, p_i=p_i, pressure_class=pclass, grad_i=(p_i - P_ATM) / (z + d_w))


def _temperature(r, w, x):
    tail, z, d_w, L = _is_tail(w), w['z'], w['d_w'], w['L']
    if w['well_type'] != 'onshore':  # spec: SMP-64
        t_deep, t_shelf = r.uniform(272, 278), r.uniform(279, 297)
        T_s = t_deep + (t_shelf - t_deep) * math.exp(-d_w / 400)
    else:
        T_s = r.uniform(258, 308) if tail else r.uniform(268, 303)
        unused(r)  # spec: SMP-51. The seabed's second number
    g_lo = max(12.0, _G_EFF_MIN * L / z)  # spec: SMP-65
    g_hi = min(55.0, (_T_MAX - T_s) / (z / 1000))
    if g_lo > g_hi:
        g_lo = g_hi
        x.flags.append('G_bounds_conflict')
    g50 = 20 + 10 * math.exp(-max(d_w - 100, 0) / 400)
    if g_lo > g50 * math.exp(-1.645 * 0.22):
        x.flags.append('G_raised_for_G_over_L')
    if g_hi < g50 * math.exp(1.645 * 0.22):
        x.flags.append('T_r_cap_binds')
    G = r.uniform(g_lo, g_hi) if tail else tlognorm(r, min(max(g50, g_lo), g_hi), 0.22, g_lo, g_hi)
    T_r = T_s + G * z / 1000
    w.update(T_s=T_s, G=G, T_r=T_r, G_eff=(T_r - T_s) / L * 1000)


def _separator(r, w):  # spec: SMP-66
    wt = w['well_type']
    if wt == 'onshore':
        p_sep, T_sep = tlognorm(r, 15, 0.6, 2, 70), r.uniform(285, 320)
    else:
        p_sep, T_sep = tlognorm(r, 35, 0.45, 5, 85), r.uniform(300, 340) if wt == 'platform' else r.uniform(290, 340)
    r_mid, r_late = r.uniform(0.5, 1.0), r.uniform(0.1, 0.6)
    r_s = {'initial': 1.0, 'mid_life': r_mid, 'late': r_late}[w['life_stage']]
    # spec: SMP-51. The flowline's two numbers and the tail's pressure are drawn for every well
    dp_subsea = r.uniform(5, 30) + r.uniform(*_RISER) * w['d_w']  # Flowline and riser
    dp_fl = dp_subsea if wt == 'subsea' else 0.0
    p_cap = P_S_MAX[wt]
    p_s_tail = logu(r, P_S_MIN, 120)
    if _is_tail(w):
        p_s = min(p_s_tail + dp_fl, p_cap)
    else:
        p_s = min(max(r_s * p_sep + dp_fl, P_S_MIN), p_cap)
    w.update(p_sep=p_sep, T_sep=T_sep, p_s=p_s, r_s=r_s, dp_fl=dp_fl)


def _rs_inverse(bo, rs_target, T, p_hi):
    """The pressure (bar) at which the black oil's uncapped R_so at T is rs_target: bisection on 1 bar to p_hi."""
    lo, hi = 1.0, p_hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bo.rs(mid, T) < rs_target else (lo, mid)
    return 0.5 * (lo + hi)


def _fluid(r, w, x, fluid):
    tail, fc, p_i, T_r = _is_tail(w), w['fluid_class'], w['p_i'], w['T_r']
    # spec: SMP-67
    if fc == 'black_oil':
        api = r.uniform(16, 45) if tail else tnorm(r, 33, 4.5, 16, 45)
    else:
        api = r.uniform(*_API[fc])
    mean, sd, lo, hi = _GAS_GRAVITY[fc]
    sg = r.uniform(lo, hi) if tail else tnorm(r, mean, sd, lo, hi)
    w.update(rho_o=density_from_api(api), rho_g=gas_density_from_sg(sg))
    bo = black_oil(w)
    # spec: SMP-68
    p_sat = None
    # spec: SMP-51. Whether the fluid is saturated, and its undersaturation, are drawn for every well
    u_sat = r.random()
    u = r.uniform(0.0, 0.85) if tail else tlognorm(r, 0.10 if fc == 'black_oil' else 0.05, 1.2, 0.005, 0.85)
    if fc in _P_SATURATED:
        if u_sat < _P_SATURATED[fc]:
            u = 0.0  # Saturated at discovery
        p_sat = p_i * (1 - u)
    if fc == 'black_oil':
        unused(r)  # spec: SMP-51. The other classes' number of the initial GOR
        R_i = bo.rs(p_sat, T_r)
        if R_i > _R_BLACK_OIL:
            p_sat, R_i = _rs_inverse(bo, _R_BLACK_OIL, T_r, p_sat), _R_BLACK_OIL
            x.flags.append('p_b_lowered_to_black_oil_limit')
    elif fc == 'volatile_oil':
        R_i = logu(r, 356, 588)
    elif fc == 'gas_condensate':
        R_i = logu(r, 588, 8900) if tail else tlognorm(r, 2000 * math.exp(-0.3 * (w['z'] / 1000 - 3.4)), 0.6, 588, 8900)
    elif fc == 'wet_gas':
        R_i = 1e6 / (logu(r, 2.8, 112) if tail else tlognorm(r, 30, 1.2, 2.8, 112))
    else:
        R_i = 1e6 / logu(r, 0.5, 2.8)
    if fluid is not None:  # spec: SMP-115. A table's saturation pressure, initial GOR and densities
        p_sat, R_i = fluid['p_sat'], fluid['R_i']
        api, sg = fluid.get('api', api), fluid.get('sg_gas', sg)
        w.update(rho_o=fluid['rho_o'], rho_g=fluid['rho_g'])
        bo = black_oil(w)
    w.update(api=api, sg_gas=sg, R_i=R_i, p_sat=p_sat)
    x.bo = bo


def _stage(r, w):
    tail, fc, drive, stage = _is_tail(w), w['fluid_class'], w['drive'], w['life_stage']
    oil = w['fluid_type'] in OILS
    # spec: SMP-69
    # spec: SMP-51. Every number below is drawn for every well: the tail's forms, an oil's and a gas class's
    r_p = r.uniform(*_R_P[stage, drive]) if (stage, drive) in _R_P else r.uniform(*_R_P_DEPLETED[oil])
    r_p_tail = r.uniform(0.15, 1.0)
    if tail:
        r_p = r_p_tail
    p_r = r_p * w['p_i']
    ogr = c_k = None
    dry_support, wlr_support = r.random() < 0.5, r.uniform(0, 0.1)
    dry_depletion, wlr_depletion = r.random() < 0.5, r.uniform(0, 0.1)
    wlr = {
        ('initial', 'pressure_support'): 0.0 if dry_support else wlr_support,
        ('initial', 'depletion'): 0.0 if dry_depletion else wlr_depletion,
        ('mid_life', 'pressure_support'): r.uniform(0.1, 0.6),
        ('mid_life', 'depletion'): r.uniform(0.0, 0.3),
        ('late', 'pressure_support'): r.uniform(0.6, 0.98),
        ('late', 'depletion'): r.uniform(0.05, 0.4),
    }[stage, drive]
    wlr_tail, u_kgko = r.uniform(0, 0.98), r.random()
    wgr = logu(r, 1, 10000) if tail else logu(r, *_WGR[stage, drive])
    u_ck = r.uniform(0.5, 3.0)
    if oil:
        if tail:
            wlr = wlr_tail
        w['w_kgko'] = u_kgko if tail else 0.0  # spec: SMP-70. Fig. 9.16 in the core, the book's range in the tail
    else:
        if fc == 'gas_condensate':
            c_k = u_ck  # spec: SMP-70
        w['wgr'] = wgr
    w['c_k'] = c_k
    R_go = producing_gor(w, p_r)
    if not oil:
        ogr = 1e6 / R_go
        wlr = wgr / (wgr + ogr)
    w.update(r_p=r_p, p_r=p_r, wlr=wlr, R_go=R_go, ogr=ogr, below_hydrostatic=p_r < w['p_h'])
    if fc in ('wet_gas', 'dry_gas'):  # spec: SMP-68
        w['saturation'] = 'single_phase_gas'
    else:
        w['saturation'] = 'undersaturated' if p_r > w['p_sat'] else 'saturated'


def _on_geometry(w, x, a, b):  # spec: SMP-73
    """A vertical well's Forchheimer coefficients for the well's geometry: a over the geometry's factor, and for a
    drain b scaled by (I_ani h / L_d)^2, the non-Darcy loss of the rate per completed length; else b as it is, a
    slanted well's non-Darcy loss being in its D, from its perforated length along the well (SMP-71)."""
    if w['geometry_family'] == 'high_angle':
        return a / x.f_geo, b * (w['I_ani'] * w['h_net'] / w['drain_length']) ** 2
    return a / x.f_geo, b


def _inflow(gen, w, x):
    """The inflow block, and a gas well's nondarcy block: the reservoir's parameters, and the inflow at the stage's
    state before a drain's friction, which needs the tubing."""
    r = gen('inflow')
    tail, family, bo = _is_tail(w), w['geometry_family'], x.bo
    oil = w['fluid_type'] in OILS
    p_r, T_r = w['p_r'], w['T_r']
    # spec: SMP-71
    k_med, k_s, k_lo, k_hi = _PERM[w['well_type']]
    if not oil:  # A gas well's effective permeability, carrying the operating limits the model lacks
        k_med, k_lo, k_hi = (_GAS_K_FACTOR * v for v in (k_med, k_lo, k_hi))
    perm = logu(r, k_lo, k_hi) if tail else tlognorm(r, k_med, k_s, k_lo, k_hi)
    h_net = logu(r, 3, 100) if tail else tlognorm(r, 15, 0.8, 3, 100)
    skin = r.uniform(-3, 15) if tail else tnorm(r, 2, 3, -3, 15)
    ln_re = r.uniform(8.5, 9.5) if family == 'high_angle' else r.uniform(7, 9)
    delta_d = r.uniform(0.1, 0.4)  # Design drawdown, a fraction of p_r
    L_d, I_ani = r.uniform(*_DRAIN_LENGTH), logu(r, 1, 3)  # A drain's length and anisotropy, drawn for every well
    # spec: SMP-72. The geometry's factor on the radial index
    if family == 'high_angle':
        x.f_geo = joshi_ratio(L_d, h_net, I_ani, skin, ln_re)
        w.update(drain_length=L_d, I_ani=I_ani, J_H_over_J_V=x.f_geo)
    elif family in ('build_and_hold', 's_shaped'):  # Cinco-Ley's slant skin in the radial index, with a floor
        s_theta = slant_skin(w['theta_end'], h_net, I_ani)
        x.f_geo = (ln_re - 0.75 + skin) / max(ln_re - 0.75 + skin + s_theta, _SLANT_FLOOR)
        w.update(I_ani=I_ani, s_theta=s_theta)
    else:
        x.f_geo = 1.0
    w.update(perm=perm, h_net=h_net, skin=skin, delta_d=delta_d, ln_re_rw=ln_re)
    if oil:
        b_now = bo.at_gor(w['R_go'])
        x.mu_o, x.B_o = b_now.oil_viscosity(p_r, T_r), b_now.bo(p_r, T_r)
        x.J_V = j_radial(perm, h_net, x.mu_o, x.B_o, ln_re - 0.75, skin)
        x.J = x.J_V * x.f_geo
    else:
        x.Z = bo.z_factor(p_r, T_r)
        x.mu_g = bo.gas_viscosity(p_r, T_r, bo.gas_density(p_r, T_r))
        # spec: SMP-71. Eq. 4-34 at a perforated share of the net pay, times a log-normal error factor, truncated
        # A slanted or vertical well's perforated length along the well, theta_end being 0 in a vertical one
        r_nd = gen('nondarcy')
        f_perf = r_nd.uniform(*_PERF_SHARE)
        along = 1.0 if family == 'high_angle' else 1 / math.cos(math.radians(w['theta_end']))
        estimate = nondarcy(w['sg_gas'], perm, h_net, x.mu_g, f_perf * h_net * along)
        x.D_nd = tlognorm(r_nd, estimate, _D_ERROR, *_D_BOUNDS)
        w.update(perf_share=f_perf, D_eq434=nondarcy(w['sg_gas'], perm, h_net, x.mu_g, h_net))
        x.a_v, x.b_v = forchheimer(perm, h_net, x.mu_g, x.Z, T_r, ln_re - 0.75, skin, x.D_nd)  # spec: SMP-73
        x.a, x.b = _on_geometry(w, x, x.a_v, x.b_v)


def _completion(r, w, x):
    tail, stage, bo = _is_tail(w), w['life_stage'], x.bo
    oil = w['fluid_type'] in OILS
    p_i, T_r, R_i, delta_d = w['p_i'], w['T_r'], w['R_i'], w['delta_d']
    ln_term, skin = w['ln_re_rw'] - 0.75, w['skin']
    # spec: SMP-75. The tubing from the design rate at completion, the initial state
    c_e = r.uniform(100, 150)
    p_wh = 1.5 * w['p_sep'] + 5 + w['dp_fl']
    T_wh = w['T_s'] + 0.6 * (T_r - w['T_s'])
    if oil:
        wlr_d = 0.0
        b_i = bo.at_gor(R_i)
        J_i = j_radial(w['perm'], w['h_net'], b_i.oil_viscosity(p_i, T_r), b_i.bo(p_i, T_r), ln_term, skin) * x.f_geo
        q_d = J_i * delta_d * p_i
    else:
        wlr_d = 10 / (10 + 1e6 / R_i)
        a_i, b_i = _on_geometry(
            w, x, *forchheimer(w['perm'], w['h_net'], x.mu_g, bo.z_factor(p_i, T_r), T_r, ln_term, skin, x.D_nd)
        )
        q_d = gas_rate(a_i, b_i, p_i**2 - ((1 - delta_d) * p_i) ** 2) * (1e6 / R_i + 10) / 1e6
    vol, mass = stream(bo, R_i, wlr_d, w['rho_w'], p_wh, T_wh)
    v_e = 1.22 * c_e / math.sqrt(mass / vol)  # Erosional velocity (m/s)
    names = list(TUBING) if not oil else list(TUBING)[:-1]
    sizes = [TUBING[name] for name in names]
    fits = [D for D in sizes if q_d * vol / 86400 / (math.pi * D * D / 4) <= v_e]
    u_up = r.random()  # Drawn for every well, so that the flag moves no later draw
    if fits:
        i = sizes.index(fits[0])
        if u_up < (0.5 if tail else 0.3) and i + 1 < len(sizes):
            i += 1
    else:
        i = len(sizes) - 1
        x.flags.append('design_rate_above_tubing_capacity')
    D = sizes[i]
    q_cap = v_e * math.pi * D * D / 4 * 86400 / vol  # Sm3/d of liquid at the erosional velocity
    eps = logu(r, 1.5e-6, 1.5e-4) if tail else logu(r, *_ROUGHNESS[stage])  # spec: SMP-76
    w.update(c_e=c_e, q_design=q_d, D=D, tubing=names[i], q_cap=q_cap, roughness=eps)


def _inflow_at_stage(w, x):
    """The inflow at the stage's state with a drain's friction along its liner, taken as the tubing, the inflow
    model and the design rates. No draws."""
    p_r, T_r, wlr, delta_d = w['p_r'], w['T_r'], w['wlr'], w['delta_d']
    D, eps, q_cap = w['D'], w['roughness'], w['q_cap']
    drain = w['geometry_family'] == 'high_angle'
    if w['fluid_type'] in OILS:
        J = x.J
        if drain:  # spec: SMP-72. Dikken's model in closed form
            B_l = (1 - wlr) * x.B_o + wlr
            rho_l = ((1 - wlr) * (w['rho_o'] + w['R_go'] * w['rho_g']) + wlr * w['rho_w']) / B_l
            c_lin = liner_liquid(D, min(max(rho_l, 400.0), 1100.0), (1 - wlr) * x.mu_o + wlr * 0.5e-3, B_l, eps)
            J_eff = drain_index(J, w['drain_length'], c_lin, delta_d * p_r, q_cap)
            w['eta_drain'] = J_eff / J
            J = J_eff
        rho_lsc = (1 - wlr) * w['rho_o'] + wlr * w['rho_w']
        k_l = J * rho_lsc / 86400  # spec: SMP-111
        vogel = w['p_sat'] >= 0.8 * p_r and wlr < 0.6  # spec: SMP-74
        w.update(mu_o=x.mu_o, B_o=x.B_o, J=J, f_geo=J / x.J_V, k_l=k_l, w_l_max=k_l * p_r / 1.8)
        w.update(inflow='vogel' if vogel else 'productivity_index', q_l_ref=J * delta_d * p_r)
    else:
        a, b = x.a, x.b
        u_d = p_r**2 - ((1 - delta_d) * p_r) ** 2
        if drain:  # spec: SMP-73. The friction's first-order term c L_d / 4 in b, c at the heel's rate
            c_lin = liner_gas(min(D, _LINER_MAX), w['sg_gas'], x.Z, T_r, x.mu_g, eps)
            q_cap_g = q_cap * 1e6 / (1e6 / w['R_i'] + 10)
            b_f = 0.0
            for _ in range(4):
                q = min(gas_rate(a, b + b_f, u_d), q_cap_g)
                b_f = float(c_lin(q)) * w['drain_length'] / 4
            w['b_friction_share'] = b_f / (b + b_f)
            b += b_f
        q_g_ref = gas_rate(a, b, u_d)
        w.update(a_gas=a, b_gas=b, a_ratio=a / x.a_v, b_ratio=b / x.b_v, D_nondarcy=x.D_nd)
        w.update(AOF=gas_rate(a, b, p_r**2 - P_ATM**2), q_g_ref=q_g_ref, f_geo=x.f_geo)
        w.update(inflow='forchheimer', q_l_ref=q_g_ref * (w['ogr'] + w['wgr']) / 1e6)  # spec: SMP-74


def _choke(r, w, x):  # spec: SMP-77
    D = w['D']
    beta = r.uniform(0.15, 0.75) if _is_tail(w) else r.uniform(0.15, 0.6)
    d_bean = min(beta * D, _BEAN_MAX)
    if d_bean < beta * D:
        x.flags.append('bean_capped_at_4_in')
    c_d = r.uniform(0.75, 0.9)
    w.update(beta=d_bean / D, d_bean=d_bean, c_d=c_d, K_c=c_d * math.pi * d_bean**2 / 4)
    w.update(choke_model='simpson', chk_profile=pick(r, CHOKE_PROFILES, (1,) * 4))


def _needs_lift(w, bo, p_r, p_s, R_go, wlr) -> bool:  # spec: SMP-78
    """The static check of an oil well at a reservoir state: whether its no-slip column, at the mean of the
    reservoir's and the wellhead's conditions, is too heavy for p_r to lift it to the surface against p_s."""
    T_r, T_s = w['T_r'], w['T_s']
    T_wh = T_s + 0.6 * (T_r - T_s)
    p_m, T_m = 0.5 * (p_r + p_s), 0.5 * (T_r + T_wh)
    vol_m, mass_m = stream(bo, R_go, wlr, w['rho_w'], p_m, T_m)
    return p_r < p_s + mass_m / vol_m * STD_GRAVITY * w['L'] / 1e5


def _lift(r, w, x, force_lift):
    tail, wt, bo = _is_tail(w), w['well_type'], x.bo
    oil = w['fluid_type'] in OILS
    L, p_r, p_s, T_r, T_s = w['L'], w['p_r'], w['p_s'], w['T_r'], w['T_s']
    R_go, wlr, rho_w = w['R_go'], w['wlr'], w['rho_w']
    md, tvd = x.stations
    # spec: SMP-78
    i_st = STAGES.index(w['life_stage'])
    need = False
    if oil:
        need = _needs_lift(w, bo, p_r, p_s, R_go, wlr)
        p_lift = 0.5 if tail else _P_LIFT_OIL['offshore' if wt != 'onshore' else 'onshore'][i_st]
    else:
        p_lift = 0.25 if tail else _P_LIFT_GAS[i_st]
    # spec: SMP-51. The lift's number and the design's are drawn for every well, lifted or not
    u_lift = r.random()
    dz_pk = r.uniform(30, 150)
    p_inj, dp_v = r.uniform(*_P_INJ[wt]), r.uniform(3.4, 13.8)  # At the top: the surface or the tree
    delta_lift = r.uniform(0.3, 0.7)  # A lifted well's design drawdown
    psi, u_glr = r.uniform(*_PSI_LG), r.random()
    lift = need or u_lift < p_lift or force_lift
    q_l_now = min(w['q_l_ref'], w['q_cap'])
    if lift:  # spec: SMP-79
        p_wf_d = (1 - delta_lift) * p_r
        vol_b, mass_b = stream(bo, R_go, wlr, rho_w, p_wf_d, T_r)  # The flowing column below the valve, no slip
        G_l = mass_b / vol_b * STD_GRAVITY / 1e5
        den = G_l - p_inj / _GILBERT_H
        H_bal = (p_inj - dp_v - p_wf_d + G_l * L) / den if den > 0 else L
        if H_bal < 0.3 * L and not need and not tail and not force_lift:  # spec: SMP-78
            lift = False  # A strong well: the operator does not install gas lift yet
            x.flags.append('lift_not_installed_strong_well')
    w.update(lift='gas_lift' if lift else 'natural', needs_lift=need)
    if lift:
        H = min(max(H_bal, 0.3 * L), L - dz_pk)
        if H != H_bal:
            x.flags.append('injection_at_packer' if H == L - dz_pk else 'injection_raised_to_0.3L')
        T_a = T_s + (T_r - T_s) * H / L
        # spec: SMP-80. GLR_cap log-uniform over the part of 30 to 1000 Sm3/Sm3 whose capacity lies within the bounds
        kq = w['rho_g'] * q_l_now / 86400  # kg/s per unit of GLR
        lo, hi = max(30.0, _W_LG_MAX[0] / kq), min(1000.0, _W_LG_MAX[1] / kq)
        if lo <= hi:
            glr = lo * (hi / lo) ** u_glr
            w_lg_max = kq * glr
        else:
            w_lg_max = _W_LG_MAX[0] if 1000.0 * kq < _W_LG_MAX[0] else _W_LG_MAX[1]
            glr = w_lg_max / kq
            x.flags.append('lift_capacity_at_bound')
        w.update(z_inj=H, injection_md=float(np.interp(H, tvd, md)), z_inj_frac=H / L, p_inj=p_inj, dp_valve=dp_v)
        w.update(T_lg=T_a + psi * (T_r - T_a), packer_offset=dz_pk, psi_lg=psi)  # At most T_r
        w.update(glr_cap=glr, w_lg_max=w_lg_max)


def _cp_methane(p, T):
    """Methane's c_p (J/(kg K)) at (p, T): bilinear in NIST's values, clamped to their ranges."""
    p, T = min(max(p, _CH4_CP_P[0]), _CH4_CP_P[-1]), min(max(T, _CH4_CP_T[0]), _CH4_CP_T[-1])
    f = (T - _CH4_CP_T[0]) / (_CH4_CP_T[1] - _CH4_CP_T[0])
    return (1 - f) * np.interp(p, _CH4_CP_P, _CH4_CP[0]) + f * np.interp(p, _CH4_CP_P, _CH4_CP[1])


def _heat_transfer(D, U_t, k_e, age) -> float:  # spec: SMP-81
    """The heat-transfer coefficient h (W/(m2 K)) of a well with the tubing diameter D (m), the tubing-to-formation
    coefficient U_t (W/(m2 K)) and the formation's conductivity k_e (W/(m K)), after `age` seconds on production."""
    r_hole = 0.108 if D <= 0.101 else 0.156
    t_d = 1.03e-6 * age / r_hole**2
    T_D = math.log(math.exp(-0.2 * t_d) + (1.5 - 0.3719 * math.exp(-t_d)) * math.sqrt(t_d))
    return U_t * k_e / (k_e + D / 2 * U_t * T_D)


def _heat(r, w):
    D, T_s, T_r = w['D'], w['T_s'], w['T_r']
    # spec: SMP-81
    age = logu(r, *_AGE[w['life_stage']]) * 86400
    U_t, k_e = logu(r, 11, 85), r.uniform(1.5, 2.6)
    h = _heat_transfer(D, U_t, k_e, age)
    h_tail = r.uniform(2, 40)  # spec: SMP-51. Drawn for every well
    if _is_tail(w):
        h = h_tail
    # spec: SMP-82
    T_rep = T_s + 0.8 * (T_r - T_s)
    t_f = (T_rep - 273.15) * 1.8 + 32
    cp_o = (0.388 + 0.00045 * t_f) / math.sqrt(141.5 / (131.5 + w['api'])) * 4186.8
    S = 1000 * w['salinity']
    cp_w = 4190 - 5.62 * S + 0.0076 * S**2
    cp_g = float(_cp_methane(0.5 * (w['p_r'] + w['p_s']), T_rep) * (0.554 / w['sg_gas']) ** 0.4)
    w.update(h=h, age_years=age / 86400 / 365, U_t=U_t, k_e=k_e, cp_o=cp_o, cp_w=cp_w, cp_g=cp_g)
