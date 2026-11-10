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

The geometry is Bjarne's ruling of 2026-11-02 (D-9, D-57): an even mix of four classes, each drawn by one of R3's
families. R6 revised the permeability's median (D-100) and the tail's share (D-103). The rework implements the
checkpoint's rulings of 2026-11-06 (decisions.md): the shapes uniform over the pooled surveys' ranges (D-104), the
brine (D-11), the black oils' gas gravity (D-16), the initial depletion and the late water cut (D-18), the producing
GOR from the material balance (D-19, solution_gas_drive.py), the permeability by well type (D-100), a drain's index
from Joshi's on a growing drainage length times Dikken's friction efficiency (D-21, drain_friction.py), the subsea
flowline-and-riser loss (D-15), the bean (D-27), the injection pressure by well type (D-29) and the lift-gas capacity
(D-31); this script at commit 8e6c175 gives R6's draws. Every draw comes from numpy's default_rng, seeded by (seed,
well, 'v2', block) through manywells.sampling.wells.rng_for, one generator per block of the draft. Since D2's ruling (2026-11-06) a
gas well's non-Darcy coefficient is D-62's estimate from Economides et al.'s Eq. 4-34 and its permeability the well
type's law times GAS_K_FACTOR; NONDARCY = 'r3' restores R3's draw (D-23) for comparison (gas_rates_d62.py).

From the repository root:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 600 .venv/bin/python plans/evidence/sampler_draft_draws.py 2000)

    --seed N           the dataset seed (default 2026)
    --csv OUT          write one row per well
    --nondarcy r3|d62  the non-Darcy coefficient's draw (default NONDARCY)
"""

import argparse
import functools
import math

import numpy as np
import pandas as pd
from drain_friction import closed_form, gas_grad, liquid_grad, ratio
from scipy.special import ndtr, ndtri
from solution_gas_drive import X_GRID, black_oil_props, kgko_blend, muskat

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
TAIL_SHARE = 0.20  # Coverage tail: D-6's 0.15 drawn, raised by D-103 so that about 15% of the accepted wells are tail
WELL_TYPES = ('platform', 'subsea', 'onshore')
P_WELL_TYPE = (0.55, 0.35, 0.10)  # D-5
FLUID_TYPES = ('black_oil', 'volatile_oil', 'gas_condensate', 'gas')
P_FLUID = (0.55, 0.10, 0.15, 0.20)  # D-4
P_DRY_GAS = 0.07  # share of dry gas among wet-or-dry gas wells, D-4
OILS = ('black_oil', 'volatile_oil')
STAGES = ('initial', 'mid_life', 'late')
P_STAGE = (0.25, 0.40, 0.35)  # D-13
GEOMETRIES = ('vertical', 'build_and_hold', 's_shaped', 'high_angle')  # The families that draw a survey
GEOMETRY_CLASSES = ('vertical', 'slightly_deviated', 'deviated', 'highly_deviated')
P_GEOMETRY = (0.25, 0.25, 0.25, 0.25)  # Bjarne, 2026-11-02 (D-9): an even mix, for every well type, core and tail
THETA_CLASS = {'slightly_deviated': (5.0, 20.0), 'deviated': (20.0, 60.0), 'highly_deviated': (60.0, 75.0)}
P_S_SHAPED = 1 / 3  # Share of the deviated class drawn as S-shaped wells, D-57
P_DRAIN = 0.5  # Share of the highly deviated class drawn as drains to the heel (high_angle), D-9
KICKOFF = (0.05, 0.75)  # Kickoff's share of the TVD below the mudline, D-104
BUILD = (0.3, 3.6)  # Build rate (degrees per 30 m), D-104
DROP, S_BOTTOM = (0.2, 2.2), (0.0, 35.0)  # S wells' drop rate (degrees per 30 m) and bottom inclination, D-104
DRAIN_TANGENT, LANDING = (10.0, 65.0), (1.5, 7.0)  # Drains' tangent before landing (deg) and landing build, D-104
DRAIN_LENGTH = (300.0, 2500.0)  # m, D-104
THETA_MAX = 75.0  # Inclination limit (degrees from vertical), D-30
MD_MAX = 9342.0  # Hasan et al.'s largest measured depth (m), D-33
P_MAX, T_MAX = 950.0, 450.0  # bar, K, D-33
G_EFF_MIN = 10.0  # K/km over the modelled well, D-10
P_ATM = P_REF / 1e5
RHO_SEA = 1025.0  # kg/m3
LITHO = 0.226  # bar/m, lithostatic gradient below the mudline (1 psi/ft)
GILBERT_H = 12192.0  # m: Gilbert's 40,000 ft (Golan and Whitson 1991, Eq. 5.17)
DEPTH = {  # median (m below the mudline) and log-sd, fitted to BOEM's P5/P50/P95 by wells [own analysis of the BOEM sands atlas]
    'black_oil': (2674, 0.317),
    'volatile_oil': (3159, 0.315),
    'gas_condensate': (3424, 0.278),
    'wet_gas': (2419, 0.422),
    'dry_gas': (1141, 0.586),
}
TUBING_ID = (0.0620, 0.0760, 0.1005, 0.1243, 0.1571, 0.2205)  # m: 2-7/8, 3-1/2, 4-1/2, 5-1/2, 7 and 9-5/8 in
P_LIFT_OIL = {'offshore': (0.20, 0.45, 0.70), 'onshore': (0.15, 0.25, 0.35)}  # by stage, D-23
P_LIFT_GAS = (0.0, 0.05, 0.15)
PERM = {  # mD: median, log-sd and bounds by well type, D-20 as revised by D-100 (by type, set on the rework's pilot)
    'platform': (15, 1.2, 0.5, 3000),
    'subsea': (60, 1.2, 0.5, 3000),
    'onshore': (3, 1.5, 0.05, 1000),
}
NONDARCY = 'd62'  # the gas wells' non-Darcy coefficient: 'd62', D-62's Eq. 4-34 as ruled (D2); 'r3', D-23's LogU
GAS_K_FACTOR = 1 / 14  # the gas classes' permeability: the well type's law (PERM) times this, bounds included, D2
D_R3 = (2e-4, 2e-3)  # d/Mscf, D-23
PERF_SHARE = (0.3, 1.0)  # the perforated share of the net pay, h_perf / h, D-62
D_ERROR = 0.7  # log-sd of Eq. 4-34's error factor, median 1 ("may be off by 100%", Guo et al. 2008), D-62
D_BOUNDS = (1e-5, 2e-3)  # d/Mscf: D-62's truncation, R3's range widened down to 1e-5
R_W = 0.108  # m, the wellbore radius (8.5 in hole), sampling-draft.md, Productivity
P_S_MAX = {'platform': 120.0, 'subsea': 250.0, 'onshore': 120.0}  # bar, the nominal downstream pressure's cap, D-15
RISER = (0.03, 0.07)  # bar/m, a subsea well's flowline-and-riser loss per metre of water depth, D-15
BEAN_MAX = 0.1016  # m, the fully open bean's largest diameter (4 in), D-27
P_INJ = {'platform': (48.0, 85.0), 'subsea': (130.0, 210.0), 'onshore': (48.0, 85.0)}  # bar, at the top, D-29
W_LG_MAX = (0.1, 3.5)  # kg/s, the lift-gas capacity's bounds, D-31
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


def raise_rate(rate, top, need, room):
    """Raise a build or drop rate by steps of 20%, to at most top, until the TVD need(rate) fits in room."""
    while need(rate) > room and rate < top:
        rate = min(top, 1.2 * rate)
    return rate


def trajectory(rng, kind, L, theta, z0=0.0):
    """(md, tvd) stations from the surface to the bottom of the modelled well at TVD L, and the family's parameters.
    theta is the class's range of the hold inclination (build_and_hold) or of the heel's (high_angle), D-9. z0 is the
    mudline's TVD in the modelled well (a platform well's water depth, else 0). Each shape parameter is uniform over
    the pooled regions' range, the kickoff 0.05 to 0.75 of the depth below the mudline (D-104). A build or drop too
    gentle for the well's depth is raised towards its range's top; a shape that still does not fit returns None, and
    the caller redraws it."""
    prm = {}
    if kind == 'vertical':
        return [0.0, L], [0.0, L], prm
    z = L - z0
    room = L - z0 - KICKOFF[0] * z  # the TVD below the shallowest kickoff

    def kickoff(below):
        """Uniform over the part of the kickoff's range that leaves the TVD below for the rest of the shape."""
        f_max = min(KICKOFF[1], (L - below - z0) / z)
        return None if f_max < KICKOFF[0] else z0 + z * rng.uniform(KICKOFF[0], f_max)

    st = [(0.0, 0.0, 0.0)]
    if kind == 'build_and_hold':
        th, b = rng.uniform(*theta), rng.uniform(*BUILD)
        b = raise_rate(b, BUILD[1], lambda x: arc_tvd(0, th, x), room - 50)
        K = kickoff(50 + arc_tvd(0, th, b))
        if K is None:
            return None
        st.append((K, K, 0.0))
        arc(st, th, b)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, build_rate=b)
    elif kind == 's_shaped':
        th, b, d = rng.uniform(15, 45), rng.uniform(*BUILD), rng.uniform(*DROP)
        th_e, tail_tvd = min(rng.uniform(*S_BOTTOM), th), min(rng.uniform(100, 400), 0.25 * L)
        left = room - 100 - tail_tvd
        b = raise_rate(b, BUILD[1], lambda x: arc_tvd(0, th, x) + arc_tvd(th, th_e, d), left)
        d = raise_rate(d, DROP[1], lambda x: arc_tvd(0, th, b) + arc_tvd(th, th_e, x), left)
        K = kickoff(100 + tail_tvd + arc_tvd(0, th, b) + arc_tvd(th, th_e, d))
        if K is None:
            return None
        st.append((K, K, 0.0))
        arc(st, th, b)
        hold_to(st, L - tail_tvd - arc_tvd(th, th_e, d))
        arc(st, th_e, d)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_end=th_e, build_rate=b, drop_rate=d, end_section_tvd=tail_tvd)
    elif kind == 'high_angle':
        th, th_heel = rng.uniform(*DRAIN_TANGENT), rng.uniform(*theta)
        th = min(th, th_heel)  # the tangent is never steeper than the heel
        b1, b2, tan_md = rng.uniform(*BUILD), rng.uniform(*LANDING), rng.uniform(50, 300)

        def below(x1, x2):
            return arc_tvd(0, th, x1) + arc_tvd(th, th_heel, x2) + tan_md * math.cos(math.radians(th_heel))

        b1 = raise_rate(b1, BUILD[1], lambda x: below(x, b2), room - 100)
        b2 = raise_rate(b2, LANDING[1], lambda x: below(b1, x), room - 100)
        K = kickoff(100 + below(b1, b2))
        if K is None:
            return None
        st.append((K, K, 0.0))
        arc(st, th, b1)
        hold_to(st, L - arc_tvd(th, th_heel, b2) - tan_md * math.cos(math.radians(th_heel)))
        arc(st, th_heel, b2)
        hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_heel=th_heel, build_rate=b1, build_rate_2=b2, tangent_md=tan_md)
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


@functools.lru_cache(maxsize=4096)
def gor_curve(rho_o, rho_g, p_sep, T_sep, T_r, p_sat, w_kgko):
    """R/R_si against p_r/p_sat, both ascending, under solution-gas drive: Muskat's material balance on the well's
    black oil with the book's k_g/k_o blended at w_kgko, Fig. 9.16 at 0 (D-19; solution_gas_drive.py)."""
    well = pd.DataFrame([dict(rho_o=rho_o, rho_g=rho_g, p_sep=p_sep, T_sep=T_sep, T_r=T_r, p_sat=p_sat)])
    a = black_oil_props(well)
    _, R, _ = muskat(a['p'], a['Rs'], a['Bo'], a['Bg'], a['mu_r'], a['dRs'], a['dBo'], a['dBg'], kgko_blend(w_kgko))
    return X_GRID[::2][::-1], R[0][::-1] / a['Rs'][0, 0]


def gor_factor(w, p_r):
    """The producing GOR over R_si at reservoir pressure p_r for an oil well w: 1 at or above p_sat, else D-19's."""
    if p_r >= w['p_sat']:
        return 1.0
    keys = ('rho_o', 'rho_g', 'p_sep', 'T_sep', 'T_r', 'p_sat', 'w_kgko')
    x, F = gor_curve(*(float(w[k]) for k in keys))
    return float(np.interp(p_r / w['p_sat'], x, F))


def stream(bo, gor, wlr, rho_w, p, T):
    """In-situ volume (m3) and mass (kg) of the stream per Sm3 of stock-tank liquid at (p, T), without slip."""
    b = bo.at_gor(gor)
    x_o = 1 - wlr
    free = max(gor - b.rs(p, T), 0.0) * bo.rho_g / b.gas_density(p, T)
    return x_o * b.bo(p, T) + wlr + x_o * free, x_o * (bo.rho_o + gor * bo.rho_g) + wlr * rho_w


def j_radial(k_md, h, mu, B, ln_term, s):
    """Pseudo-steady radial productivity index, Sm3/(d bar) of stock-tank liquid."""
    return 2 * math.pi * k_md * 9.869233e-16 * h / (mu * B * (ln_term + s)) * 86400 * 1e5


def forchheimer(k_md, h, mu, Z, T, ln_term, s, D_dmscf):
    """a (bar2 per Sm3/d) and b (bar2 per (Sm3/d)2) of p_r^2 - p_wf^2 = a q + b q^2 (Economides et al. 1994,
    Eq. 4-32 to 4-33), from the radial gas-flow equation in SI units and the non-Darcy coefficient D in d/Mscf."""
    a_si = P_REF * T * mu * Z * (ln_term + s) / (math.pi * k_md * 9.869233e-16 * h * T_REF)  # Pa2 per Sm3/s
    b_si = a_si * D_dmscf * 86400 / 28.3168 / (ln_term + s)  # Pa2 per (Sm3/s)2
    return a_si / 1e10 / 86400, b_si / 1e10 / 86400**2


def nondarcy_eq434(sg, k_md, h, mu, h_perf):
    """Economides et al.'s estimate of the non-Darcy coefficient (1994, Eq. 4-34), D = 6e-5 gamma_g k^-0.1 h /
    (mu_g r_w h_perf^2) in field units, d/Mscf: k in md, the net pay h and the perforated interval h_perf (m) and R_W
    in ft, mu (Pa s) in cP."""
    ft = 0.3048
    return 6e-5 * sg * k_md**-0.1 * (h / ft) / (mu * 1e3 * (R_W / ft) * (h_perf / ft) ** 2)


def gas_rate(a, b, dp2):
    return (-a + math.sqrt(a * a + 4 * b * dp2)) / (2 * b)


def cp_methane(p, T):
    p, T = min(max(p, CH4_CP_P[0]), CH4_CP_P[-1]), min(max(T, CH4_CP_T[0]), CH4_CP_T[-1])
    w = (T - CH4_CP_T[0]) / (CH4_CP_T[1] - CH4_CP_T[0])
    return (1 - w) * np.interp(p, CH4_CP_P, CH4_CP[0]) + w * np.interp(p, CH4_CP_P, CH4_CP[1])


# --- The draws -------------------------------------------------------------------------------------------------------
def draw_well(seed, k, stage_override=None, force_lift=False, fluid=None):
    """One well of the draft, as a dict of its draws, labels and derived quantities, and its survey stations.
    stage_override sets the life stage after it is drawn, so that every generator draws the same numbers: the same
    well at another stage of its life, as R4's nsol-2 evolution uses it (sampler_draft_samples.py). force_lift gives
    the well its lift design even where the assignment leaves it natural (an nsol-2 well whose lift starts by D-49's
    rules 1 or 2); it changes no other draw. fluid is a volatile oil's or condensate's built table
    (table_fluids.py; fluids-draft.md): its p_sat at T_r and initial GOR R_i, and its rho_o and rho_g (the model's,
    single-stage, D-63), and where the fluid is not matched to the drawn API and gas gravity (a published example,
    an equilibrium phase), its api and sg_gas. They replace the fluid block's after its draws (D-65), so that no draw
    moves and every later block follows the table."""
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
    if stage_override is not None:
        stage = stage_override
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

    # Geometry: the class (Bjarne, 2026-11-02; D-9, D-57), then the family that draws the survey
    r = gen('geometry')
    gcl = pick(r, GEOMETRY_CLASSES, P_GEOMETRY)
    sub = r.random()
    gc = {
        'vertical': 'vertical',
        'slightly_deviated': 'build_and_hold',
        'deviated': 's_shaped' if sub < P_S_SHAPED else 'build_and_hold',
        'highly_deviated': 'high_angle' if sub < P_DRAIN else 'build_and_hold',
    }[gcl]
    for attempt in range(20):
        res = trajectory(r, gc, L, THETA_CLASS.get(gcl), d_w if wt == 'platform' else 0.0)
        if res is not None and res[0][-1] <= MD_MAX:
            md_st, tvd_st, gprm = res
            break
        flags.append('geometry_redrawn')
    else:
        gcl, gc, (md_st, tvd_st, gprm) = 'vertical', 'vertical', trajectory(r, 'vertical', L, None)
        flags.append('geometry_to_vertical')
    incl = [
        math.degrees(math.acos(min(1.0, (tvd_st[i] - tvd_st[i - 1]) / (md_st[i] - md_st[i - 1]))))
        for i in range(1, len(md_st))
    ]
    w.update(geometry_class=gcl, geometry_family=gc, md=md_st[-1], md_tvd=md_st[-1] / L)
    w.update(theta_max=max(incl), theta_end=incl[-1])
    w.update({f'geo_{key}': val for key, val in gprm.items()})

    # Formation water
    r = gen('water')
    sal = logu(r, 0.002, 0.25) if tail else tlognorm(r, 0.06, 0.6, 0.002, 0.25)  # mass fraction of solids, D-11
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
    dp_fl = r.uniform(5, 30) + r.uniform(*RISER) * d_w if wt == 'subsea' else 0.0  # flowline and riser, D-15
    p_cap = P_S_MAX[wt]
    p_s = min(logu(r, 1.5, 120) + dp_fl, p_cap) if tail else min(max(r_s * p_sep + dp_fl, 1.5), p_cap)
    w.update(p_sep=p_sep, T_sep=T_sep, p_s=p_s, r_s=r_s, dp_fl=dp_fl)

    # Fluid
    r = gen('fluid')
    sat_p = None
    if fc == 'black_oil':
        api = rng_uniform_or(r, tail, 16, 45, lambda: tnorm(r, 33, 4.5, 16, 45))
        sg = rng_uniform_or(r, tail, 0.58, 0.95, lambda: tnorm(r, 0.70, 0.07, 0.58, 0.95))  # D-16
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
    if fluid is not None:  # a built table's saturation pressure, initial GOR and densities (D-63, D-65, D-66)
        sat_p, R_i = fluid['p_sat'], fluid['R_i']
        api, sg = fluid.get('api', api), fluid.get('sg_gas', sg)
        rho_o, rho_g = fluid['rho_o'], fluid['rho_g']
        bo = BlackOil(rho_o=rho_o, rho_g=rho_g, gas_law='dak', p_sep=p_sep, T_sep=T_sep)
    w.update(api=api, sg_gas=sg, rho_o=rho_o, rho_g=rho_g, R_i=R_i, p_sat=sat_p)

    # Life stage: the reservoir's state now
    r = gen('stage')
    if stage == 'initial':
        r_p = r.uniform(0.85, 1.0) if drive == 'depletion' else r.uniform(0.97, 1.0)  # D-18
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
            ('late', 'pressure_support'): r.uniform(0.6, 0.98),  # D-18: the tail towards 0.98
            ('late', 'depletion'): r.uniform(0.05, 0.4),
        }[(stage, drive)]
        if tail:
            wlr = r.uniform(0, 0.98)
        u_kgko = r.random()
        w['w_kgko'] = u_kgko if tail else 0.0  # k_g/k_o: Fig. 9.16 in the core, the book's range in the tail, D-19
        R_go = R_i * gor_factor(w, p_r)
        ogr, c_k = None, None
    else:
        wgr = {
            ('initial', 'pressure_support'): (2, 20),
            ('initial', 'depletion'): (2, 20),
            ('mid_life', 'pressure_support'): (10, 150),
            ('mid_life', 'depletion'): (5, 50),
            ('late', 'pressure_support'): (50, 500),
            ('late', 'depletion'): (5, 80),
        }[(stage, drive)]
        wgr = logu(r, 1, 10000) if tail else logu(r, *wgr)  # Sm3 water per 1e6 Sm3 gas; the tail's top is D-107's end
        R_go, c_k = R_i, None
        if fc == 'gas_condensate':
            c_k = r.uniform(0.5, 3.0)
            R_go = R_i * (1 + c_k * max(0.0, 1 - p_r / sat_p))
        ogr = 1e6 / R_go
        wlr = wgr / (wgr + ogr)
    w.update(r_p=r_p, p_r=p_r, wlr=wlr, R_go=R_go, ogr=ogr, below_hydrostatic=p_r < p_h, c_k=c_k)
    w['saturation'] = (
        'single_phase_gas' if fc in ('wet_gas', 'dry_gas') else ('undersaturated' if p_r > sat_p else 'saturated')
    )

    # Inflow, before a drain's friction
    r = gen('inflow')
    k_med, k_s, k_lo, k_hi = PERM[wt]
    if not oil:  # a gas well's effective permeability, carrying the operating limits the model lacks (D2, D-100)
        k_med, k_lo, k_hi = (GAS_K_FACTOR * x for x in (k_med, k_lo, k_hi))
    perm = logu(r, k_lo, k_hi) if tail else tlognorm(r, k_med, k_s, k_lo, k_hi)
    h_net = logu(r, 3, 100) if tail else tlognorm(r, 15, 0.8, 3, 100)
    skin = r.uniform(-3, 15) if tail else tnorm(r, 2, 3, -3, 15)
    ln_re = r.uniform(8.5, 9.5) if gc == 'high_angle' else r.uniform(7, 9)
    delta_d = r.uniform(0.1, 0.4)  # design drawdown, a fraction of p_r
    L_d, I_ani = r.uniform(*DRAIN_LENGTH), logu(r, 1, 3)  # a drain's length and anisotropy, drawn for every well
    drain = gc == 'high_angle'
    if drain:  # Joshi's index on a drainage length that grows with the drain, r_eH = max(r_e, L / 1.6), D-21
        f_geo = float(ratio(L_d, h_net, I_ani, skin, ln_re, 'B'))
        w.update(drain_length=L_d, I_ani=I_ani, J_H_over_J_V=f_geo)
    elif gc in ('build_and_hold', 's_shaped'):
        f_geo = math.cos(math.radians(incl[-1])) ** -0.5  # the slant's placeholder, D-21
    else:
        f_geo = 1.0
    w.update(perm=perm, h_net=h_net, skin=skin, delta_d=delta_d, ln_re_rw=ln_re)

    def on_geometry(a, b):
        """A vertical well's Forchheimer coefficients for the geometry: a over J_H/J_V, and for a drain b scaled by
        (I_ani h / L)^2, the non-Darcy loss of the rate per completed length (D-21); the slant's factor on k h."""
        return a / f_geo, (b * (I_ani * h_net / L_d) ** 2 if drain else b / f_geo)

    if oil:
        b_now = bo.at_gor(R_go)
        mu, B = b_now.oil_viscosity(p_r, T_r), b_now.bo(p_r, T_r)
        J_V = j_radial(perm, h_net, mu, B, ln_re - 0.75, skin)
        J = J_V * f_geo
    else:
        Z = bo.z_factor(p_r, T_r)
        mu_g = bo.gas_viscosity(p_r, T_r, bo.gas_density(p_r, T_r))
        D_nd = logu(r, *D_R3)  # D-23; drawn under either option, so that the inflow block's draws stay as they were
        D_434 = nondarcy_eq434(sg, perm, h_net, mu_g, h_net)  # Eq. 4-34 perforated over the net pay, for the record
        if NONDARCY == 'd62':
            # D-62 as ruled: Eq. 4-34 at a perforated share, times a log-normal error factor, truncated; core and tail
            # alike, from their own generator so that no other draw moves. mu_g is a's, so b = a D / (ln + s) holds
            # no viscosity, as the inertial term should not. A drain scales b_V as any other (D-21, on_geometry)
            r_nd = gen('nondarcy')
            f_perf = r_nd.uniform(*PERF_SHARE)
            D_nd = tlognorm(r_nd, nondarcy_eq434(sg, perm, h_net, mu_g, f_perf * h_net), D_ERROR, *D_BOUNDS)
            w['perf_share'] = f_perf
        elif NONDARCY != 'r3':
            raise ValueError(f'NONDARCY must be r3 or d62, got {NONDARCY!r}')
        w['D_eq434'] = D_434
        a_v, b_v = forchheimer(perm, h_net, mu_g, Z, T_r, ln_re - 0.75, skin, D_nd)  # the vertical well's
        a, b = on_geometry(a_v, b_v)

    # Completion: tubing from the design rate at completion (the initial state), and roughness
    r = gen('completion')
    c_e = r.uniform(100, 150)
    p_wh = 1.5 * p_sep + 5 + dp_fl
    T_wh = T_s + 0.6 * (T_r - T_s)
    if oil:
        wlr_d, gor_d = 0.0, R_i
        b_i = bo.at_gor(R_i)
        J_i = j_radial(perm, h_net, b_i.oil_viscosity(p_i, T_r), b_i.bo(p_i, T_r), ln_re - 0.75, skin) * f_geo
        q_d = J_i * delta_d * p_i
    else:
        wlr_d, gor_d = 10 / (10 + 1e6 / R_i), R_i
        a_i, b_i_ = on_geometry(*forchheimer(perm, h_net, mu_g, bo.z_factor(p_i, T_r), T_r, ln_re - 0.75, skin, D_nd))
        q_d = gas_rate(a_i, b_i_, p_i**2 - ((1 - delta_d) * p_i) ** 2) * (1e6 / R_i + 10) / 1e6
    vol, mass = stream(bo, gor_d, wlr_d, rho_w, p_wh, T_wh)
    v_e = 1.22 * c_e / math.sqrt(mass / vol)
    sizes = TUBING_ID if not oil else TUBING_ID[:-1]
    fits = [D for D in sizes if q_d * vol / 86400 / (math.pi * D * D / 4) <= v_e]
    u_up = r.random()  # drawn for every well, so that the flag moves no later draw
    if fits:
        i = sizes.index(fits[0])
        if u_up < (0.5 if tail else 0.3) and i + 1 < len(sizes):
            i += 1
        D = sizes[i]
    else:
        D = sizes[-1]
        flags.append('design_rate_above_tubing_capacity')  # a check, D-100
    q_cap = v_e * math.pi * D * D / 4 * 86400 / vol  # Sm3/d of liquid at the erosional velocity
    eps = {
        'initial': (1.5e-6, 3e-5),
        'mid_life': (1e-5, 6e-5),
        'late': (2e-5, 1.5e-4),
    }[stage]
    eps = logu(r, 1.5e-6, 1.5e-4) if tail else logu(r, *eps)
    w.update(c_e=c_e, q_design=q_d, D=D, q_cap=q_cap, roughness=eps)

    # A drain's friction along its liner, taken as the tubing: Dikken's model in closed form (D-21, drain_friction.py)
    rho_lsc = (1 - wlr) * rho_o + wlr * rho_w
    if oil:
        if drain:
            B_l = (1 - wlr) * B + wlr
            rho_l = min(max(((1 - wlr) * (rho_o + R_go * rho_g) + wlr * rho_w) / B_l, 400.0), 1100.0)
            _, c_heel = liquid_grad(D, rho_l, (1 - wlr) * mu + wlr * 0.5e-3, B_l, eps=eps)
            J_eff = float(closed_form(J, L_d, c_heel, delta_d * p_r, q_cap))
            w['eta_drain'] = J_eff / J
            J = J_eff
        k_l = J * rho_lsc / 86400
        vogel = sat_p >= 0.8 * p_r and wlr < 0.6
        inflow = 'vogel' if vogel else 'productivity_index'
        q_l_ref = J * delta_d * p_r
        w.update(mu_o=mu, B_o=B, J=J, f_geo=J / J_V, k_l=k_l, w_l_max=k_l * p_r / 1.8, inflow=inflow, q_l_ref=q_l_ref)
    else:
        u_d = p_r**2 - ((1 - delta_d) * p_r) ** 2
        if drain:  # the friction's first-order term c L / 4 in b, c at the heel's rate (drain_friction.py)
            _, c_heel = gas_grad(min(D, 0.1571), sg, Z, T_r, mu_g, eps=eps)
            q_cap_g = q_cap * 1e6 / (1e6 / R_i + 10)
            b_f = 0.0
            for _ in range(4):
                q = min(gas_rate(a, b + b_f, u_d), q_cap_g)
                b_f = float(c_heel(q)) * L_d / 4
            w['b_friction_share'] = b_f / (b + b_f)
            b += b_f
        aof = gas_rate(a, b, p_r**2 - P_ATM**2)
        q_g_ref = gas_rate(a, b, u_d)
        q_l_ref = q_g_ref * (ogr + wgr) / 1e6
        k_l = q_l_ref * rho_lsc / 86400 / (delta_d * p_r)  # interim productivity index (Mapping)
        w.update(a_gas=a, b_gas=b, a_ratio=a / a_v, b_ratio=b / b_v, D_nondarcy=D_nd, AOF=aof, q_g_ref=q_g_ref)
        w.update(f_geo=f_geo, k_l=k_l)
        w.update(inflow='forchheimer', q_l_ref=q_l_ref, wgr=wgr)  # feature 022's label (D-60); R3's 'gas_rate'

    # Choke
    r = gen('choke')
    beta = r.uniform(0.15, 0.75) if tail else r.uniform(0.15, 0.6)  # D-27
    d_bean = min(beta * D, BEAN_MAX)
    if d_bean < beta * D:
        flags.append('bean_capped_at_4_in')
    c_d = r.uniform(0.75, 0.9)
    w.update(
        beta=d_bean / D,
        d_bean=d_bean,
        c_d=c_d,
        K_c=c_d * math.pi * d_bean**2 / 4,
        choke_profile=pick(r, CHOKE_PROFILES, (1,) * 4),
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
    lift = need or r.random() < p_lift or force_lift
    q_l_now = min(q_l_ref, q_cap)
    if lift:
        dz_pk = r.uniform(30, 150)
        p_inj, dp_v = r.uniform(*P_INJ[wt]), r.uniform(3.4, 13.8)  # at the top: the surface or the tree, D-29
        p_wf_d = (1 - r.uniform(0.3, 0.7)) * p_r  # a lifted well's design drawdown
        vol_b, mass_b = stream(bo, R_go, wlr, rho_w, p_wf_d, T_r)  # the flowing column below the valve, no slip
        G_l = mass_b / vol_b * STD_GRAVITY / 1e5
        den = G_l - p_inj / GILBERT_H
        H_bal = (p_inj - dp_v - p_wf_d + G_l * L) / den if den > 0 else L
        if H_bal < 0.3 * L and not need and not tail and not force_lift:
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
        # D-31 as ruled: GLR_cap log-uniform over the part of 30 to 1000 Sm3/Sm3 whose capacity lies within W_LG_MAX
        # (truncated, not clipped); a well too weak for the floor even at 1000 takes the floor. Core and tail alike
        kq = rho_g * q_l_now / 86400  # kg/s per unit of GLR
        lo, hi, u_glr = max(30.0, W_LG_MAX[0] / kq), min(1000.0, W_LG_MAX[1] / kq), r.random()
        if lo <= hi:
            glr = lo * (hi / lo) ** u_glr
            w_lg_max = kq * glr
        else:
            w_lg_max = W_LG_MAX[0] if 1000.0 * kq < W_LG_MAX[0] else W_LG_MAX[1]
            glr = w_lg_max / kq
            flags.append('lift_capacity_at_bound')
        w.update(z_inj=H, md_inj=md_inj, z_inj_frac=H / L, p_inj=p_inj, dp_valve=dp_v, T_lg=T_a - phi * (T_a - T_s))
        w.update(packer_offset=dz_pk, phi_lg=phi)
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
        'geometry_family',
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
        (
            'kickoff below the mudline / depth below it',
            (core['geo_kickoff'] - core['d_w'].where(core.well_type == 'platform', 0)) / core['z'],
        ),
        ('build rate (deg/30 m)', core['geo_build_rate']),
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
        ('p_s nominal, subsea (bar)', core.loc[core.well_type == 'subsea', 'p_s']),
        (
            'p_s nominal, subsea in water over 1000 m (bar)',
            core.loc[(core.well_type == 'subsea') & (core.d_w > 1000), 'p_s'],
        ),
        ('flowline and riser loss, subsea (bar)', core.loc[core.well_type == 'subsea', 'dp_fl']),
        ('API, black oil', core.loc[core.fluid_class == 'black_oil', 'api']),
        ('gas gravity, black oil', core.loc[core.fluid_class == 'black_oil', 'sg_gas']),
        ('R_si, black oil (Sm3/Sm3)', core.loc[core.fluid_class == 'black_oil', 'R_i']),
        ('p_i - p_b, black oil (bar)', (core['p_i'] - core['p_sat'])[core.fluid_class == 'black_oil']),
        ('R_go, gas condensate (Sm3/Sm3)', core.loc[core.fluid_class == 'gas_condensate', 'R_go']),
        ('OGR, wet gas (Sm3/1e6 Sm3)', core.loc[core.fluid_class == 'wet_gas', 'ogr']),
        ('p_r/p_i', core['r_p']),
        ('water cut, oils', core.loc[core.fluid_type.isin(OILS), 'wlr']),
        ('WGR, gas classes (Sm3/1e6 Sm3)', core['wgr']),
        ('permeability, platform (mD)', core.loc[core.well_type == 'platform', 'perm']),
        ('permeability, subsea (mD)', core.loc[core.well_type == 'subsea', 'perm']),
        ('permeability, onshore (mD)', core.loc[core.well_type == 'onshore', 'perm']),
        ('J, oils (Sm3/(d bar))', core['J']),
        ('J_eff/J_V, drains (oils)', core.loc[core.geometry_family == 'high_angle', 'f_geo']),
        ('friction efficiency J_eff/J_H, drains (oils)', core['eta_drain'] if 'eta_drain' in core else []),
        ("friction's share of b, drains (gas)", core['b_friction_share'] if 'b_friction_share' in core else []),
        ('drain length (m)', core['drain_length']),
        ('q_l_ref, oils (Sm3/d)', core.loc[core.fluid_type.isin(OILS), 'q_l_ref']),
        (
            'q_l_ref, platform oils (Sm3/d)',
            core.loc[core.fluid_type.isin(OILS) & (core.well_type == 'platform'), 'q_l_ref'],
        ),
        (
            'q_l_ref, subsea oils (Sm3/d)',
            core.loc[core.fluid_type.isin(OILS) & (core.well_type == 'subsea'), 'q_l_ref'],
        ),
        (
            'R_go/R_si, saturated core black oils',
            (core['R_go'] / core['R_i'])[(core.fluid_class == 'black_oil') & (core.saturation == 'saturated')],
        ),
        ('AOF, gas classes (Sm3/d)', core['AOF']),
        ('q_g_ref, gas classes (Sm3/d)', core['q_g_ref']),
        ('D (mm)', core['D'] * 1000),
        ('roughness (um)', core['roughness'] * 1e6),
        ('K_c/A', core['K_c'] / (math.pi * core['D'] ** 2 / 4)),
        ('bean diameter (mm)', core['d_bean'] * 1000),
        ('injection pressure, subsea (bar)', core.loc[core.well_type == 'subsea', 'p_inj']),
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
    # the Gulf's correlations by wells: an analysis of BOEM's sands atlas, kept outside the repo
    print(
        '\nSpearman rank correlations, core black oils (compare with the GOM by wells, [own analysis of the BOEM sands atlas])'
    )
    for name, x, y in (
        ('depth - p_i', bo_['z'], bo_['p_i']),
        ('depth - T_r', bo_['z'], bo_['T_r']),
        ('R_si - p_i', bo_['R_i'], bo_['p_i']),
        ('R_si - T_r', bo_['R_i'], bo_['T_r']),
        ('R_si - depth', bo_['R_i'], bo_['z']),
        ('API - R_si', bo_['api'], bo_['R_i']),
    ):
        print(f'  {name:14s} {spearman(x, y):+.2f}')
    gc_ = core[core.fluid_class == 'gas_condensate']
    print(
        f"  condensate R_go - depth {spearman(gc_['R_go'], gc_['z']):+.2f}  (the GOM's yield rises with depth, so GOR negative)"
    )
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
    floor = np.where(df.well_type == 'subsea', 20.0, 1.6)
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
    global NONDARCY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--csv')
    ap.add_argument('--nondarcy', choices=('r3', 'd62'), default=NONDARCY)
    args = ap.parse_args()
    NONDARCY = args.nondarcy
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
