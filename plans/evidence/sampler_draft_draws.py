"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Draws wells from the draft well draws of the sampler redesign (plans/sampler-redesign/sampling-draft.md, Step R3)
without solving them, and prints the labels' shares, the marginals, a few rank correlations, the shares behind
Bjarne's goals, and every truncation and bound of the draft. Each well is mapped to the v2.0.0 configuration with the
draft's interim mapping (the black oil for every fluid class, a productivity index for gas wells; sampling-draft.md,
Mapping) and checked with configurations.differences, without building its system. R6 extends it to solve the wells.

The geometry shares are placeholders (GEOMETRY_SHARES_PLACEHOLDER): the draft leaves them pending the geometry
evidence. Every draw comes from numpy's default_rng, seeded by (seed, well, 'v2', block) through
manywells.sampling.wells.rng_for, one generator per block of the draft.

From the repository root:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 600 .venv/bin/python plans/evidence/sampler_draft_draws.py 2000)

    --seed N     the dataset seed (default 2026)
    --csv OUT    write one row per well
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from manywells.choke import CHOKE_PROFILES, SimpsonChokeModel
from manywells.configurations import V2, V2_SLIP, differences
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.sampling.wells import rng_for
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.thermal import ThermalModel
from manywells.units import P_REF, STD_GRAVITY, T_REF

# --- Shares and constants of the draft (sampling-draft.md) ---------------------------------------------------------
TAIL_SHARE = 0.15  # Coverage tail, D-6
WELL_TYPES = ('platform', 'subsea', 'onshore')
P_WELL_TYPE = (0.55, 0.35, 0.10)  # D-5
FLUID_TYPES = ('black_oil', 'volatile_oil', 'gas_condensate', 'gas')
P_FLUID = (0.55, 0.10, 0.15, 0.20)  # D-4
P_DRY_GAS = 0.07  # share of dry gas among wet-or-dry gas wells, D-4
OILS = ('black_oil', 'volatile_oil')
STAGES = ('initial', 'mid_life', 'late')
P_STAGE = (0.25, 0.40, 0.35)  # D-13
GEOMETRIES = ('vertical', 'build_and_hold', 's_shaped', 'high_angle')
GEOMETRY_SHARES_PLACEHOLDER = {'offshore': (0.40, 0.30, 0.10, 0.20), 'onshore': (0.70, 0.20, 0.05, 0.05)}  # pending
THETA_MAX = 75.0  # Inclination limit (degrees from vertical), D-30
MD_MAX = 9342.0  # Hasan et al.'s largest measured depth (m), D-33
P_MAX, T_MAX = 950.0, 450.0  # bar, K, D-33
G_EFF_MIN = 10.0  # K/km over the modelled well, D-10
P_ATM = P_REF / 1e5
RHO_SEA = 1025.0  # kg/m3
LITHO = 0.226  # bar/m, lithostatic gradient below the mudline (1 psi/ft)
GILBERT_H = 12192.0  # m: Gilbert's 40,000 ft (Golan and Whitson 1991, Eq. 5.17)
DEPTH = {  # median (m below the mudline) and log-sd, from BOEM's P5/P50/P95 by wells (sources-reservoir-fluids, §1)
    'black_oil': (2674, 0.317),
    'volatile_oil': (3159, 0.315),
    'gas_condensate': (3424, 0.278),
    'wet_gas': (2419, 0.422),
    'dry_gas': (1141, 0.586),
}
TUBING_ID = (0.0620, 0.0760, 0.1005, 0.1243, 0.1571, 0.2205)  # m: 2-7/8, 3-1/2, 4-1/2, 5-1/2, 7 and 9-5/8 in
P_LIFT_OIL = {'offshore': (0.20, 0.45, 0.70), 'onshore': (0.15, 0.25, 0.35)}  # by stage, D-23
P_LIFT_GAS = (0.0, 0.05, 0.15)
CH4_CP_P, CH4_CP_T = (1.0, 101.0, 201.0, 401.0), (300.0, 380.0)  # NIST methane (sources-reservoir-fluids, §10)
CH4_CP = ((2236.0, 3011.0, 3549.0, 3336.0), (2466.0, 2801.0, 3079.0, 3234.0))  # J/(kg K), [T][p]


# --- Distributions -------------------------------------------------------------------------------------------------
def tnorm(rng, mu, sd, lo, hi):
    """Normal(mu, sd) truncated to [lo, hi], by the inverse CDF of one uniform draw."""
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    u = min(max(rng.uniform(a, b), 1e-15), 1 - 1e-15)
    return float(min(max(mu + sd * ndtri(u), lo), hi))


def tlognorm(rng, median, s, lo, hi):
    return math.exp(tnorm(rng, math.log(median), s, math.log(lo), math.log(hi)))


def logu(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def pick(rng, names, p):
    return str(names[rng.choice(len(names), p=np.asarray(p) / np.sum(p))])


def logistic(x):
    return 1 / (1 + math.exp(-x))


# --- Geometry (sampling-draft.md, Geometry) ------------------------------------------------------------------------
def arc(st, th1, rate):
    """Append survey stations along a circular arc from the last station's inclination to th1 (degrees), at a build
    or drop rate (degrees per 30 m), one station per degree."""
    md0, tvd0, th0 = st[-1]
    kappa = math.radians(rate) / 30.0
    n = max(1, int(abs(th1 - th0)))
    for th in np.linspace(th0, th1, n + 1)[1:]:
        md = md0 + abs(math.radians(th - th0)) / kappa
        tvd = tvd0 + abs(math.sin(math.radians(th)) - math.sin(math.radians(th0))) / kappa
        st.append((md, tvd, th))


def hold_to(st, tvd_end):
    md0, tvd0, th = st[-1]
    if tvd_end > tvd0:
        st.append((md0 + (tvd_end - tvd0) / math.cos(math.radians(th)), tvd_end, th))


def arc_tvd(th0, th1, rate):
    return abs(math.sin(math.radians(th1)) - math.sin(math.radians(th0))) * 30.0 / math.radians(rate)


def trajectory(rng, kind, L, tail):
    """(md, tvd) stations from the surface to the bottom of the modelled well at TVD L, and the family's parameters."""
    prm = {}
    if kind == 'vertical':
        return [0.0, L], [0.0, L], prm
    st = [(0.0, 0.0, 0.0)]
    if kind == 'build_and_hold':
        th = rng.uniform(5, 70) if tail else tnorm(rng, 35, 13, 5, 65)
        b = rng.uniform(1.5, 3.0)
        while 0.9 * L - 50 < arc_tvd(0, th, b):  # raise the build rate, then lower the inclination, until it fits
            if b < 5.0:
                b = min(5.0, b * 1.2)
            else:
                th *= 0.9
        k_max = min(0.7 * L, L - 50 - arc_tvd(0, th, b))
        K = rng.uniform(0.1 * L, max(0.1 * L, k_max))
        st.append((K, K, 0.0))
        arc(st, th, b)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, build_rate=b)
    elif kind == 's_shaped':
        th, b, d = rng.uniform(15, 45), rng.uniform(1.5, 3.0), rng.uniform(1.0, 2.5)
        th_e, tail_tvd = rng.uniform(0, 10), min(rng.uniform(100, 400), 0.25 * L)
        while 0.9 * L - 150 - tail_tvd < arc_tvd(0, th, b) + arc_tvd(th, min(th_e, th), d) and th > 5:
            th *= 0.9
        th_e = min(th_e, th)
        if 0.9 * L - 100 - tail_tvd < arc_tvd(0, th, b) + arc_tvd(th, th_e, d):
            return None  # the shape does not fit the well's depth
        k_max = min(0.6 * L, L - 100 - tail_tvd - arc_tvd(0, th, b) - arc_tvd(th, th_e, d))
        K = rng.uniform(0.1 * L, max(0.1 * L, k_max))
        st.append((K, K, 0.0))
        arc(st, th, b)
        hold_to(st, L - tail_tvd - arc_tvd(th, th_e, d))
        arc(st, th_e, d)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_end=th_e, build_rate=b, drop_rate=d)
    elif kind == 'high_angle':
        th = rng.uniform(35, 65)
        th_heel = rng.uniform(max(60.0, th), THETA_MAX)
        b1, b2 = rng.uniform(1.5, 3.0), rng.uniform(2.0, 4.0)
        tan_md = rng.uniform(50, 300)
        need = arc_tvd(0, th, b1) + arc_tvd(th, th_heel, b2) + tan_md * math.cos(math.radians(th_heel))
        while 0.9 * L - 100 < need and th > 20:
            th *= 0.9
            need = arc_tvd(0, th, b1) + arc_tvd(th, th_heel, b2) + tan_md * math.cos(math.radians(th_heel))
        if 0.95 * L - 100 < need:
            return None
        K = rng.uniform(0.05 * L, max(0.05 * L, min(0.4 * L, L - 100 - need)))
        st.append((K, K, 0.0))
        arc(st, th, b1)
        hold_to(st, L - arc_tvd(th, th_heel, b2) - tan_md * math.cos(math.radians(th_heel)))
        arc(st, th_heel, b2)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_heel=th_heel, build_rate=b1)
    md = [s[0] for s in st]
    tvd = [s[1] for s in st]
    keep = [0] + [i for i in range(1, len(md)) if md[i] > md[i - 1] + 1e-6]
    return [md[i] for i in keep], [tvd[i] for i in keep], prm


# --- Fluid helpers ---------------------------------------------------------------------------------------------------
def rs_inverse(bo, rs_target, T, p_hi):
    """The pressure (bar) at which the black oil's uncapped R_so at T is rs_target (bisection on 1 bar to p_hi)."""
    lo, hi = 1.0, p_hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bo.rs(mid, T) < rs_target else (lo, mid)
    return 0.5 * (lo + hi)


def stream(bo, gor, wlr, rho_w, p, T):
    """In-situ volume (m3) and mass (kg) of the stream per Sm3 of stock-tank liquid at (p, T), without slip."""
    b = bo.at_gor(gor)
    x_o = 1 - wlr
    free = max(gor - b.rs(p, T), 0.0) * bo.rho_g / b.gas_density(p, T)
    return x_o * b.bo(p, T) + wlr + x_o * free, x_o * (bo.rho_o + gor * bo.rho_g) + wlr * rho_w


def j_radial(k_md, h, mu, B, ln_term, s):
    """Pseudo-steady radial productivity index, Sm3/(d bar) of stock-tank liquid."""
    return 2 * math.pi * k_md * 9.869233e-16 * h / (mu * B * (ln_term + s)) * 86400 * 1e5


def j_joshi(k_md, h, mu, B, L_d, I_ani, r_e, r_w, s):
    """Joshi's horizontal-well productivity index (Economides et al. 1994, Eqs. 2-46 to 2-48), Sm3/(d bar)."""
    a = L_d / 2 * math.sqrt(0.5 + math.sqrt(0.25 + (2 * r_e / L_d) ** 4))
    t1 = math.log((a + math.sqrt(a * a - (L_d / 2) ** 2)) / (L_d / 2))
    t2 = I_ani * h / L_d * (math.log(I_ani * h / (r_w * (I_ani + 1))) + s)
    return 2 * math.pi * k_md * 9.869233e-16 * h / (mu * B * (t1 + t2)) * 86400 * 1e5


def forchheimer(k_md, h, mu, Z, T, ln_term, s, D_dmscf):
    """a (bar2 per Sm3/d) and b (bar2 per (Sm3/d)2) of p_r^2 - p_wf^2 = a q + b q^2 (Economides et al. 1994,
    Eq. 4-32 to 4-33), from the radial gas-flow equation in SI units and the non-Darcy coefficient D in d/Mscf."""
    a_si = P_REF * T * mu * Z * (ln_term + s) / (math.pi * k_md * 9.869233e-16 * h * T_REF)  # Pa2 per Sm3/s
    b_si = a_si * D_dmscf * 86400 / 28.3168 / (ln_term + s)  # Pa2 per (Sm3/s)2
    return a_si / 1e10 / 86400, b_si / 1e10 / 86400**2


def gas_rate(a, b, dp2):
    return (-a + math.sqrt(a * a + 4 * b * dp2)) / (2 * b)


def cp_methane(p, T):
    p, T = min(max(p, CH4_CP_P[0]), CH4_CP_P[-1]), min(max(T, CH4_CP_T[0]), CH4_CP_T[-1])
    w = (T - CH4_CP_T[0]) / (CH4_CP_T[1] - CH4_CP_T[0])
    return (1 - w) * np.interp(p, CH4_CP_P, CH4_CP[0]) + w * np.interp(p, CH4_CP_P, CH4_CP[1])


# --- The draws -------------------------------------------------------------------------------------------------------
def draw_well(seed, k):
    """One well of the draft, as a dict of its draws, labels and derived quantities, and its survey stations."""
    w = {'well': k}
    flags = []

    def gen(block):
        return rng_for(seed, k, 'v2', block)

    # Labels
    r = gen('labels')
    tail = bool(r.random() < TAIL_SHARE)
    wt = pick(r, WELL_TYPES, (1, 1, 1) if tail else P_WELL_TYPE)
    ft = pick(r, FLUID_TYPES, (1, 1, 1, 1) if tail else P_FLUID)
    fc = ft if ft != 'gas' else ('dry_gas' if r.random() < (0.25 if tail else P_DRY_GAS) else 'wet_gas')
    oil = ft in OILS
    drive = 'pressure_support' if r.random() < (0.5 if tail else (0.7 if oil else 0.4)) else 'depletion'
    stage = pick(r, STAGES, (1, 1, 1) if tail else P_STAGE)
    offshore = wt != 'onshore'
    w.update(tail=tail, well_type=wt, fluid_type=ft, fluid_class=fc, drive=drive, stage=stage)

    # Depth and water depth
    r = gen('depth')
    z = logu(r, 600, 7000) if tail else tlognorm(r, *DEPTH[fc], 600, 6500)
    if wt == 'platform':
        if tail:
            d_w = logu(r, 15, 2000)
        else:
            d_w = logu(r, 400, 2000) if r.random() < 0.15 else tlognorm(r, 90, 0.6, 15, 400)
    elif wt == 'subsea':
        d_w = logu(r, 50, 3000) if tail else tlognorm(r, 350, 0.8, 80, 3000)
    else:
        d_w = 0.0
    L = z + d_w if wt == 'platform' else z
    w.update(z=z, d_w=d_w, L=L)

    # Geometry
    r = gen('geometry')
    shares = (1, 1, 1, 1) if tail else GEOMETRY_SHARES_PLACEHOLDER['offshore' if offshore else 'onshore']
    gc = pick(r, GEOMETRIES, shares)
    for attempt in range(20):
        res = trajectory(r, gc, L, tail)
        if res is not None and res[0][-1] <= MD_MAX:
            md_st, tvd_st, gprm = res
            break
        flags.append('geometry_redrawn')
    else:
        gc, (md_st, tvd_st, gprm) = 'vertical', trajectory(r, 'vertical', L, tail)
        flags.append('geometry_to_vertical')
    incl = [
        math.degrees(math.acos(min(1.0, (tvd_st[i] - tvd_st[i - 1]) / (md_st[i] - md_st[i - 1]))))
        for i in range(1, len(md_st))
    ]
    w.update(geometry_class=gc, md=md_st[-1], md_tvd=md_st[-1] / L, theta_max=max(incl), theta_end=incl[-1])
    w.update({f'geo_{key}': val for key, val in gprm.items()})

    # Formation water
    r = gen('water')
    sal = logu(r, 0.005, 0.25) if tail else tlognorm(r, 0.08, 0.6, 0.01, 0.25)  # mass fraction of solids
    rho_w = 999.1 + 692.0 * sal + 311.0 * sal**2
    w.update(salinity=sal, rho_w=rho_w)

    # Initial reservoir pressure
    r = gen('pressure')
    p_h = P_ATM + (RHO_SEA * d_w + rho_w * z) * STD_GRAVITY / 1e5
    p_lith = P_ATM + RHO_SEA * d_w * STD_GRAVITY / 1e5 + LITHO * z
    lam_cap = max(1e-3, min(0.85, (P_MAX - p_h) / (p_lith - p_h)))
    mu = -math.log(1 / (0.02 + 0.45 * logistic((z / 1000 - 3.8) / 0.9)) - 1)  # logit of the median
    if r.random() < (0.10 if tail else 0.04):
        p_i, lam, pclass = p_h * r.uniform(0.8, 1.0), 0.0, 'under'
    else:
        if tail:
            lam = r.uniform(0.0, max(0.0, min(0.8, lam_cap)))
        else:
            lam = logistic(tnorm(r, mu, 0.9, -12.0, math.log(lam_cap / (1 - lam_cap))))
        p_i = p_h + lam * (p_lith - p_h)
        pclass = 'over' if lam >= 0.25 else 'normal'
    if lam_cap < logistic(mu + 1.645 * 0.9):
        flags.append('p_i_cap_binds')
    w.update(p_h=p_h, p_lith=p_lith, lam=lam, p_i=p_i, pressure_class=pclass, grad_i=(p_i - P_ATM) / (z + d_w))

    # Temperature
    r = gen('temperature')
    if offshore:
        t_deep, t_shelf = r.uniform(272, 278), r.uniform(279, 297)
        T_s = t_deep + (t_shelf - t_deep) * math.exp(-d_w / 400)
    else:
        T_s = r.uniform(258, 308) if tail else r.uniform(268, 303)
    g_lo = max(12.0, G_EFF_MIN * L / z)
    g_hi = min(55.0, (T_MAX - T_s) / (z / 1000))
    if g_lo > g_hi:
        g_lo = g_hi
        flags.append('G_bounds_conflict')
    g50 = 20 + 10 * math.exp(-max(d_w - 100, 0) / 400)
    if g_lo > g50 * math.exp(-1.645 * 0.22):
        flags.append('G_raised_for_G_over_L')
    if g_hi < g50 * math.exp(1.645 * 0.22):
        flags.append('T_r_cap_binds')
    G = r.uniform(g_lo, g_hi) if tail else tlognorm(r, min(max(g50, g_lo), g_hi), 0.22, g_lo, g_hi)
    T_r = T_s + G * z / 1000
    w.update(T_s=T_s, G=G, T_r=T_r, G_eff=(T_r - T_s) / L * 1000)

    # Separator and the nominal downstream pressure
    r = gen('separator')
    if wt == 'onshore':
        p_sep, T_sep = tlognorm(r, 15, 0.6, 2, 70), r.uniform(285, 320)
    else:
        p_sep, T_sep = tlognorm(r, 35, 0.45, 5, 85), r.uniform(300, 340) if wt == 'platform' else r.uniform(290, 320)
    r_s = {'initial': 1.0, 'mid_life': r.uniform(0.5, 1.0), 'late': r.uniform(0.1, 0.6)}[stage]
    dp_fl = r.uniform(5, 30) + 0.02 * d_w if wt == 'subsea' else 0.0
    p_s = min(logu(r, 1.5, 120) + dp_fl, 120) if tail else min(max(r_s * p_sep + dp_fl, 1.5), 120)
    w.update(p_sep=p_sep, T_sep=T_sep, p_s=p_s)

    # Fluid
    r = gen('fluid')
    sat_p = None
    if fc == 'black_oil':
        api = rng_uniform_or(r, tail, 16, 45, lambda: tnorm(r, 33, 4.5, 16, 45))
        sg = rng_uniform_or(r, tail, 0.58, 0.95, lambda: tnorm(r, 0.66, 0.06, 0.58, 0.95))
    elif fc == 'volatile_oil':
        api = r.uniform(40, 50)
        sg = rng_uniform_or(r, tail, 0.62, 1.0, lambda: tnorm(r, 0.75, 0.06, 0.62, 1.0))
    elif fc == 'gas_condensate':
        api = r.uniform(42, 58)
        sg = rng_uniform_or(r, tail, 0.60, 0.90, lambda: tnorm(r, 0.72, 0.05, 0.60, 0.90))
    else:
        api = r.uniform(45, 58)
        sg = rng_uniform_or(r, tail, 0.571, 0.75, lambda: tnorm(r, 0.62, 0.035, 0.571, 0.75))
    rho_o, rho_g = density_from_api(api), gas_density_from_sg(sg)
    bo = BlackOil(rho_o=rho_o, rho_g=rho_g, gas_law='dak', p_sep=p_sep, T_sep=T_sep)
    if oil or fc == 'gas_condensate':
        p_sat_share = {'black_oil': 0.20, 'volatile_oil': 0.40, 'gas_condensate': 0.30}[fc]
        if r.random() < p_sat_share:
            x = 0.0
        else:
            x = r.uniform(0.0, 0.85) if tail else tlognorm(r, 0.10 if fc == 'black_oil' else 0.05, 1.2, 0.005, 0.85)
        sat_p = p_i * (1 - x)
    if fc == 'black_oil':
        R_i = bo.rs(sat_p, T_r)
        if R_i > 356:
            sat_p, R_i = rs_inverse(bo, 356, T_r, sat_p), 356.0
            flags.append('p_b_lowered_to_black_oil_limit')
    elif fc == 'volatile_oil':
        R_i = logu(r, 356, 588)
    elif fc == 'gas_condensate':
        R_i = tlognorm(r, 2000 * math.exp(-0.3 * (z / 1000 - 3.4)), 0.6, 588, 8900) if not tail else logu(r, 588, 8900)
    elif fc == 'wet_gas':
        R_i = 1e6 / (logu(r, 2.8, 112) if tail else tlognorm(r, 30, 1.2, 2.8, 112))
    else:
        R_i = 1e6 / logu(r, 0.5, 2.8)
    w.update(api=api, sg_gas=sg, rho_o=rho_o, rho_g=rho_g, R_i=R_i, p_sat=sat_p)

    # Life stage: the reservoir's state now
    r = gen('stage')
    if stage == 'initial':
        r_p = r.uniform(0.97, 1.0)
    elif stage == 'mid_life':
        r_p = r.uniform(0.85, 1.0) if drive == 'pressure_support' else r.uniform(0.55, 0.92)
    elif drive == 'pressure_support':
        r_p = r.uniform(0.70, 1.0)
    else:
        r_p = r.uniform(0.3, 0.6) if oil else r.uniform(0.2, 0.55)
    if tail:
        r_p = r.uniform(0.15, 1.0)
    p_r = r_p * p_i
    if oil:
        wlr = {
            ('initial', 'pressure_support'): 0.0 if r.random() < 0.5 else r.uniform(0, 0.1),
            ('initial', 'depletion'): 0.0 if r.random() < 0.5 else r.uniform(0, 0.1),
            ('mid_life', 'pressure_support'): r.uniform(0.1, 0.6),
            ('mid_life', 'depletion'): r.uniform(0.0, 0.3),
            ('late', 'pressure_support'): r.uniform(0.6, 0.95),
            ('late', 'depletion'): r.uniform(0.05, 0.4),
        }[(stage, drive)]
        if tail:
            wlr = r.uniform(0, 0.98)
        R_go = R_i * min(10.0, 1 + logu(r, 2, 15) * max(0.0, 1 - p_r / sat_p))
        ogr = None
    else:
        wgr = {
            ('initial', 'pressure_support'): (2, 20),
            ('initial', 'depletion'): (2, 20),
            ('mid_life', 'pressure_support'): (10, 150),
            ('mid_life', 'depletion'): (5, 50),
            ('late', 'pressure_support'): (50, 500),
            ('late', 'depletion'): (5, 80),
        }[(stage, drive)]
        wgr = logu(r, 1, 1000) if tail else logu(r, *wgr)  # Sm3 water per 1e6 Sm3 gas
        R_go = R_i
        if fc == 'gas_condensate':
            R_go = R_i * (1 + r.uniform(0.5, 3.0) * max(0.0, 1 - p_r / sat_p))
        ogr = 1e6 / R_go
        wlr = wgr / (wgr + ogr)
    w.update(r_p=r_p, p_r=p_r, wlr=wlr, R_go=R_go, ogr=ogr, below_hydrostatic=p_r < p_h)
    w['saturation'] = (
        'single_phase_gas' if fc in ('wet_gas', 'dry_gas') else ('undersaturated' if p_r > sat_p else 'saturated')
    )

    # Inflow
    r = gen('inflow')
    k_med, k_lo, k_hi = (50, 0.5, 3000) if offshore else (10, 0.05, 1000)
    perm = logu(r, k_lo, k_hi) if tail else tlognorm(r, k_med, 1.5, k_lo, k_hi)
    h_net = logu(r, 3, 100) if tail else tlognorm(r, 15, 0.8, 3, 100)
    skin = r.uniform(-3, 15) if tail else tnorm(r, 2, 3, -3, 15)
    ln_re = r.uniform(8.5, 9.5) if gc == 'high_angle' else r.uniform(7, 9)
    r_w = 0.108
    delta_d = r.uniform(0.1, 0.4)  # design drawdown, a fraction of p_r
    w.update(perm=perm, h_net=h_net, skin=skin, delta_d=delta_d)
    rho_lsc = (1 - wlr) * rho_o + wlr * rho_w
    if oil:
        b_now = bo.at_gor(R_go)
        mu, B = b_now.oil_viscosity(p_r, T_r), b_now.bo(p_r, T_r)
        J = j_radial(perm, h_net, mu, B, ln_re - 0.75, skin)
        f_geo = 1.0
        if gc == 'high_angle':
            L_d, I_ani = r.uniform(300, min(1000, 1.6 * r_w * math.exp(ln_re))), logu(r, 1, 3)
            f_geo = j_joshi(perm, h_net, mu, B, L_d, I_ani, r_w * math.exp(ln_re), r_w, skin) / J
            if f_geo > 6.0:
                f_geo = 6.0
                flags.append('J_H_over_J_V_capped_at_6')
            w.update(drain_length=L_d, I_ani=I_ani)
        elif gc in ('build_and_hold', 's_shaped'):
            f_geo = math.cos(math.radians(incl[-1])) ** -0.5
        J *= f_geo
        k_l = J * rho_lsc / 86400
        vogel = sat_p >= 0.8 * p_r and wlr < 0.6
        inflow = 'vogel' if vogel else 'productivity_index'
        q_l_ref = J * delta_d * p_r
        w.update(mu_o=mu, B_o=B, J=J, f_geo=f_geo, k_l=k_l, w_l_max=k_l * p_r / 1.8, inflow=inflow, q_l_ref=q_l_ref)
    else:
        Z = bo.z_factor(p_r, T_r)
        mu_g = bo.gas_viscosity(p_r, T_r, bo.gas_density(p_r, T_r))
        D_nd = logu(r, 2e-4, 2e-3)
        f_geo = 1.0
        if gc == 'high_angle':
            f_geo = r.uniform(1.5, 4.0)
        elif gc in ('build_and_hold', 's_shaped'):
            f_geo = math.cos(math.radians(incl[-1])) ** -0.5
        a, b = forchheimer(perm, h_net * f_geo, mu_g, Z, T_r, ln_re - 0.75, skin, D_nd)
        aof = gas_rate(a, b, p_r**2 - P_ATM**2)
        q_g_ref = gas_rate(a, b, p_r**2 - ((1 - delta_d) * p_r) ** 2)
        q_l_ref = q_g_ref * (ogr + wgr) / 1e6
        k_l = q_l_ref * rho_lsc / 86400 / (delta_d * p_r)  # interim productivity index (Mapping)
        w.update(a_gas=a, b_gas=b, D_nondarcy=D_nd, AOF=aof, q_g_ref=q_g_ref, f_geo=f_geo, k_l=k_l)
        w.update(inflow='gas_rate', q_l_ref=q_l_ref, wgr=wgr)

    # Completion: tubing from the design rate at completion (the initial state), and roughness
    r = gen('completion')
    c_e = r.uniform(100, 150)
    p_wh = 1.5 * p_sep + 5 + (0.02 * d_w if wt == 'subsea' else 0.0)
    T_wh = T_s + 0.6 * (T_r - T_s)
    if oil:
        wlr_d, gor_d = 0.0, R_i
        b_i = bo.at_gor(R_i)
        J_i = j_radial(perm, h_net, b_i.oil_viscosity(p_i, T_r), b_i.bo(p_i, T_r), ln_re - 0.75, skin) * f_geo
        q_d = J_i * delta_d * p_i
    else:
        wlr_d, gor_d = 10 / (10 + 1e6 / R_i), R_i
        a_i, b_i_ = forchheimer(perm, h_net * f_geo, mu_g, bo.z_factor(p_i, T_r), T_r, ln_re - 0.75, skin, D_nd)
        q_d = gas_rate(a_i, b_i_, p_i**2 - ((1 - delta_d) * p_i) ** 2) * (1e6 / R_i + 10) / 1e6
    vol, mass = stream(bo, gor_d, wlr_d, rho_w, p_wh, T_wh)
    v_e = 1.22 * c_e / math.sqrt(mass / vol)
    sizes = TUBING_ID if not oil else TUBING_ID[:-1]
    fits = [D for D in sizes if q_d * vol / 86400 / (math.pi * D * D / 4) <= v_e]
    if fits:
        i = sizes.index(fits[0])
        if r.random() < (0.5 if tail else 0.3) and i + 1 < len(sizes):
            i += 1
        D = sizes[i]
    else:
        D = sizes[-1]
        flags.append('design_rate_above_tubing_capacity')
    q_cap = v_e * math.pi * D * D / 4 * 86400 / vol  # Sm3/d of liquid at the erosional velocity
    eps = {
        'initial': (1.5e-6, 3e-5),
        'mid_life': (1e-5, 6e-5),
        'late': (2e-5, 1.5e-4),
    }[stage]
    eps = logu(r, 1.5e-6, 1.5e-4) if tail else logu(r, *eps)
    w.update(c_e=c_e, q_design=q_d, D=D, q_cap=q_cap, roughness=eps)

    # Choke
    r = gen('choke')
    beta = r.uniform(0.2, 0.75) if tail else r.uniform(0.25, 0.6)
    c_d = r.uniform(0.75, 0.9)
    w.update(
        beta=beta, c_d=c_d, K_c=c_d * beta**2 * math.pi * D * D / 4, choke_profile=pick(r, CHOKE_PROFILES, (1,) * 4)
    )

    # Gas lift
    r = gen('lift')
    i_st = STAGES.index(stage)
    need = False
    if oil:
        p_m, T_m = 0.5 * (p_r + p_s), 0.5 * (T_r + T_wh)
        vol_m, mass_m = stream(bo, R_go, wlr, rho_w, p_m, T_m)
        need = p_r < p_s + mass_m / vol_m * STD_GRAVITY * L / 1e5
        p_lift = 0.5 if tail else P_LIFT_OIL['offshore' if offshore else 'onshore'][i_st]
    else:
        p_lift = 0.25 if tail else P_LIFT_GAS[i_st]
    lift = need or r.random() < p_lift
    q_l_now = min(q_l_ref, q_cap)
    if lift:
        dz_pk = r.uniform(30, 150)
        p_inj, dp_v = r.uniform(48, 76), r.uniform(3.4, 13.8)
        p_wf_d = (1 - r.uniform(0.3, 0.7)) * p_r  # a lifted well's design drawdown
        vol_b, mass_b = stream(bo, R_go, wlr, rho_w, p_wf_d, T_r)  # the flowing column below the valve, no slip
        G_l = mass_b / vol_b * STD_GRAVITY / 1e5
        den = G_l - p_inj / GILBERT_H
        H_bal = (p_inj - dp_v - p_wf_d + G_l * L) / den if den > 0 else L
        if H_bal < 0.3 * L and not need and not tail:
            lift = False  # a strong well: the operator does not install gas lift yet
            flags.append('lift_not_installed_strong_well')
    w.update(lift='gas_lift' if lift else 'natural', needs_lift=need)
    if lift:
        H = min(max(H_bal, 0.3 * L), L - dz_pk)
        if H != H_bal:
            flags.append('injection_at_packer' if H == L - dz_pk else 'injection_raised_to_0.3L')
        md_inj = float(np.interp(H, tvd_st, md_st))
        T_a = T_s + (T_r - T_s) * H / L
        phi = r.uniform(0, 0.3)
        glr = logu(r, 25, 1000) if tail else logu(r, 50, 400)
        w_lg_max = min(max(rho_g * glr * q_l_now / 86400, 0.02), 2.0)
        w.update(z_inj=H, md_inj=md_inj, z_inj_frac=H / L, p_inj=p_inj, dp_valve=dp_v, T_lg=T_a - phi * (T_a - T_s))
        w.update(T_lg_below_ambient=phi * (T_a - T_s), glr_cap=glr, w_lg_max=w_lg_max)

    # Heat transfer and heat capacities
    r = gen('heat')
    age = {'initial': (30, 365), 'mid_life': (365, 8 * 365), 'late': (5 * 365, 25 * 365)}[stage]
    age = logu(r, *age) * 86400
    U_t, k_e = logu(r, 11, 85), r.uniform(1.5, 2.6)
    r_hole = 0.108 if D <= 0.101 else 0.156
    t_d = 1.03e-6 * age / r_hole**2
    T_D = math.log(math.exp(-0.2 * t_d) + (1.5 - 0.3719 * math.exp(-t_d)) * math.sqrt(t_d))
    h = U_t * k_e / (k_e + D / 2 * U_t * T_D)
    if tail:
        h = r.uniform(2, 40)
    T_rep = T_s + 0.8 * (T_r - T_s)
    t_f = (T_rep - 273.15) * 1.8 + 32
    cp_o = (0.388 + 0.00045 * t_f) / math.sqrt(141.5 / (131.5 + api)) * 4186.8
    S = 1000 * sal
    cp_w = 4190 - 5.62 * S + 0.0076 * S**2
    cp_g = float(cp_methane(0.5 * (p_r + p_s), T_rep) * (0.554 / sg) ** 0.4)
    w.update(h=h, age_years=age / 86400 / 365, U_t=U_t, k_e=k_e, cp_o=cp_o, cp_w=cp_w, cp_g=cp_g)

    w['flags'] = ';'.join(flags)
    return w, (md_st, tvd_st)


def rng_uniform_or(r, tail, lo, hi, core):
    return r.uniform(lo, hi) if tail else core()


# --- The interim mapping to v2.0.0 (sampling-draft.md, Mapping) -------------------------------------------------------
def well_properties(w, stations):
    md_st, tvd_st = stations
    pvt = BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])
    fluid = FluidModel(
        pvt=pvt, rho_w=w['rho_w'], gor=w['R_go'], wlr=w['wlr'], cp_g=w['cp_g'], cp_o=w['cp_o'], cp_w=w['cp_w']
    )
    inflow = Vogel(w_l_max=w['w_l_max']) if w['inflow'] == 'vogel' else ProductivityIndex(k_l=w['k_l'])
    md_inj = w.get('md_inj')
    wp = WellProperties(
        geometry=WellGeometry.from_survey(md_st, tvd_st, 100, D=w['D']),
        fluid=fluid,
        friction=RoughnessFriction(roughness=w['roughness']),
        thermal=ThermalModel(h=w['h']),
        slip=V2_SLIP,
        inflow=inflow,
        choke=SimpsonChokeModel(K_c=w['K_c'], chk_profile=w['choke_profile']),
        injection_md=None if md_inj is None or math.isnan(md_inj) else md_inj,
    )
    T_lg = w.get('T_lg')
    BoundaryConditions(
        p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], T_lg=None if T_lg is None else T_lg, u=1.0, w_lg=0.0
    )
    cos = np.asarray(wp.geometry.cos_incl)
    return wp, math.degrees(math.acos(min(1.0, float(cos.min()))))


# --- Report ------------------------------------------------------------------------------------------------------------
def q(s, ps=(5, 50, 95)):
    s = pd.Series(s).dropna()
    return ' / '.join(f'{v:.4g}' for v in np.percentile(s, ps)) + f'  (n={len(s)})' if len(s) else '-'


def spearman(x, y):
    x, y = pd.Series(x), pd.Series(y)
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def report(df):
    n = len(df)
    core = df[~df['tail']]
    print(f'{n} wells; coverage tail {df["tail"].mean():.1%}\n')
    print('Labels (share of all wells; core in brackets)')
    for col in (
        'well_type',
        'fluid_class',
        'drive',
        'stage',
        'geometry_class',
        'saturation',
        'inflow',
        'lift',
        'pressure_class',
        'choke_profile',
    ):
        a, c = df[col].value_counts(normalize=True), core[col].value_counts(normalize=True)
        print(f'  {col}: ' + ', '.join(f'{k} {a[k]:.1%} ({c.get(k, 0):.1%})' for k in a.index))
    print('\nMarginals, core wells: P5 / P50 / P95')
    rows = [
        ('z below mudline (m)', core['z']),
        ('water depth, platform (m)', core.loc[core.well_type == 'platform', 'd_w']),
        ('water depth, subsea (m)', core.loc[core.well_type == 'subsea', 'd_w']),
        ('L (m)', core['L']),
        ('MD (m)', core['md']),
        ('MD/TVD, deviated families', core.loc[core.geometry_class != 'vertical', 'md_tvd']),
        ('max inclination, deviated (deg)', core.loc[core.geometry_class != 'vertical', 'theta_max']),
        ('salinity (mass fraction)', core['salinity']),
        ('rho_w (kg/m3)', core['rho_w']),
        ('p_i (bar)', core['p_i']),
        ('p_i gradient below sea level (bar/m)', core['grad_i']),
        ('lambda', core['lam']),
        ('T_s offshore (K)', core.loc[core.well_type != 'onshore', 'T_s']),
        ('G (K/km)', core['G']),
        ('G over L (K/km)', core['G_eff']),
        ('T_r (K)', core['T_r']),
        ('p_sep (bar)', core['p_sep']),
        ('p_s nominal (bar)', core['p_s']),
        ('API, black oil', core.loc[core.fluid_class == 'black_oil', 'api']),
        ('gas gravity, black oil', core.loc[core.fluid_class == 'black_oil', 'sg_gas']),
        ('R_si, black oil (Sm3/Sm3)', core.loc[core.fluid_class == 'black_oil', 'R_i']),
        ('p_i - p_b, black oil (bar)', (core['p_i'] - core['p_sat'])[core.fluid_class == 'black_oil']),
        ('R_go, gas condensate (Sm3/Sm3)', core.loc[core.fluid_class == 'gas_condensate', 'R_go']),
        ('OGR, wet gas (Sm3/1e6 Sm3)', core.loc[core.fluid_class == 'wet_gas', 'ogr']),
        ('p_r/p_i', core['r_p']),
        ('water cut, oils', core.loc[core.fluid_type.isin(OILS), 'wlr']),
        ('WGR, gas classes (Sm3/1e6 Sm3)', core['wgr']),
        ('permeability (mD)', core['perm']),
        ('J, oils (Sm3/(d bar))', core['J']),
        ('J_H/J_V, high-angle oils', core.loc[core.geometry_class == 'high_angle', 'f_geo']),
        ('q_l_ref, oils (Sm3/d)', core.loc[core.fluid_type.isin(OILS), 'q_l_ref']),
        ('AOF, gas classes (Sm3/d)', core['AOF']),
        ('q_g_ref, gas classes (Sm3/d)', core['q_g_ref']),
        ('D (mm)', core['D'] * 1000),
        ('roughness (um)', core['roughness'] * 1e6),
        ('K_c/A', core['K_c'] / (math.pi * core['D'] ** 2 / 4)),
        ('injection TVD / L', core['z_inj_frac']),
        ('T_lg below the ambient at the valve (K)', core['T_lg_below_ambient']),
        ('w_lg capacity (kg/s)', core['w_lg_max']),
        ('h (W/(m2 K))', core['h']),
        ('cp_o (J/(kg K))', core['cp_o']),
        ('cp_w (J/(kg K))', core['cp_w']),
        ('cp_g (J/(kg K))', core['cp_g']),
    ]
    for name, s in rows:
        print(f'  {name:42s} {q(s)}')
    bo_ = core[core.fluid_class == 'black_oil']
    print('\nSpearman rank correlations, core black oils (GOM by wells in brackets, sources-reservoir-fluids §1, §7)')
    for name, x, y, ref in (
        ('depth - p_i', bo_['z'], bo_['p_i'], '0.95'),
        ('depth - T_r', bo_['z'], bo_['T_r'], '0.82'),
        ('R_si - p_i', bo_['R_i'], bo_['p_i'], '0.62'),
        ('R_si - T_r', bo_['R_i'], bo_['T_r'], '0.56'),
        ('R_si - depth', bo_['R_i'], bo_['z'], '0.60'),
        ('API - R_si', bo_['api'], bo_['R_i'], '0.24'),
    ):
        print(f'  {name:14s} {spearman(x, y):+.2f}  ({ref})')
    gc_ = core[core.fluid_class == 'gas_condensate']
    print(f'  condensate R_go - depth {spearman(gc_["R_go"], gc_["z"]):+.2f}  (yield - depth +0.69, so GOR negative)')
    print("\nBjarne's goals (all wells; core in brackets)")

    def share(mask_all, mask_core):
        return f'{mask_all.mean():.1%} ({mask_core.mean():.1%})'

    print(
        f'  1. p_r below hydrostatic: {share(df.below_hydrostatic, core.below_hydrostatic)}; '
        f'below 0.9 of it: {share(df.p_r < 0.9 * df.p_h, core.p_r < 0.9 * core.p_h)}'
    )
    for st in STAGES:
        for dr in ('depletion', 'pressure_support'):
            m = (core.stage == st) & (core.drive == dr)
            print(f'       {st}, {dr}: {core.below_hydrostatic[m].mean():.0%} of {m.sum()} core wells')
    print(f'  2. p_s below 10 bar: {share(df.p_s < 10, core.p_s < 10)}')
    for wt in WELL_TYPES:
        print(f'       {wt}: {(core.loc[core.well_type == wt, "p_s"] < 10).mean():.1%} of core {wt} wells')
    over = df.grad_i > 0.1357
    print(f'  3. initial gradient above 0.6 psi/ft (0.1357 bar/m): {share(over, core.grad_i > 0.1357)}')
    hi_q = df['q_l_ref'] > df['q_l_ref'].quantile(0.75)
    print(f'       of these, top-quartile design liquid rate: {(over & hi_q).sum() / max(over.sum(), 1):.1%}')
    lift = df.lift == 'gas_lift'
    oils_all = df.fluid_type.isin(OILS)
    print(
        f'  5. gas lift: oils {lift[oils_all].mean():.1%}, gas classes {lift[~oils_all].mean():.1%}; '
        f'oils that need lift by the static check {df.needs_lift[oils_all].mean():.1%}'
    )
    rq = pd.qcut(df.loc[oils_all, 'R_go'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print(
        '       lifted share of oils by R_go quartile: '
        + ', '.join(f'{k} {lift[oils_all][rq == k].mean():.0%}' for k in ['Q1', 'Q2', 'Q3', 'Q4'])
    )
    oe = np.where(df.fluid_type.isin(OILS), df.q_l_ref * (1 - df.wlr), df.q_g_ref / 1000)
    floor = np.where(df.well_type == 'subsea', 8.0, 1.6)
    print(
        f'  design rate below the economic floor (oil equivalent, before any solve): {(oe < floor).mean():.1%}; '
        f'below half of it: {(oe < 0.5 * floor).mean():.1%}'
    )
    lim = df['flags'].str.contains('design_rate_above_tubing_capacity')
    print(
        "  design rate above the largest tubing's erosional capacity: "
        + ', '.join(f'{ft} {lim[df.fluid_type == ft].mean():.0%}' for ft in FLUID_TYPES)
    )
    print(f'  T_r above 395 K (122 C): {share(df.T_r > 395, core.T_r > 395)} (Nadeau et al. 2023: 12.9% of reservoirs)')
    print('\nTruncations and redraws (wells)')
    flags = df['flags'].str.split(';').explode()
    for k, v in flags[flags != ''].value_counts().items():
        print(f'  {k}: {v} ({v / n:.1%})')
    print('\nBounds of the draft (violations should be 0)')
    checks = {
        'inclination above 75 deg (grid)': df.theta_grid > THETA_MAX + 1e-6,
        'MD above 9342 m': df.md > MD_MAX,
        'p_i above 950 bar': df.p_i > P_MAX + 1e-6,
        'lambda above 0.85': (df.p_i - df.p_h) > 0.85 * (df.p_lith - df.p_h) + 1e-6,
        'T_r above 450 K': df.T_r > T_MAX + 1e-6,
        'G over L below 10 K/km': df.G_eff < G_EFF_MIN - 1e-6,
        'saturation pressure above p_i': df.p_sat > df.p_i + 1e-6,
        'black-oil R_si outside 1 to 356': (df.fluid_class == 'black_oil') & ~df.R_i.between(1, 356.0001),
        'API outside 16 to 58': ~df.api.between(16, 58),
        'gas gravity below 0.571': df.sg_gas < 0.571,
        'oil water cut above 0.98': df.fluid_type.isin(OILS) & (df.wlr > 0.98),
        'configurations.check fails': df.check != '',
    }
    for name, m in checks.items():
        print(f'  {name}: {int(m.sum())}')
    print('\nValidity ranges left (specs/model/validity.md), share of all wells; the model extrapolates')
    for name, m in {
        'p_r above 363 bar (Beggs-Robinson)': df.p_r > 363,
        'p_r above 552 bar (Lee-Gonzalez-Eakin)': df.p_r > 552,
        'T_r above 419 K (Beggs-Robinson)': df.T_r > 419,
        'T_r above 444 K (Lee-Gonzalez-Eakin)': df.T_r > 444,
        'R_go above 534 Sm3/Sm3, mapped to the black oil': df.R_go > 534,
        'R_go above 1000 Sm3/Sm3 (Hasan et al.)': df.R_go > 1000,
    }.items():
        print(f'  {name}: {m.mean():.1%}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--csv')
    args = ap.parse_args()
    rows = []
    for k in range(args.n):
        w, stations = draw_well(args.seed, k)
        try:
            wp, theta_grid = well_properties(w, stations)
            w['check'] = '; '.join(differences(wp, V2))
            w['theta_grid'] = theta_grid
        except ValueError as e:
            w['check'], w['theta_grid'] = f'ValueError: {e}', np.nan
        rows.append(w)
    df = pd.DataFrame(rows)
    report(df)
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
