"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 06 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

What a long horizontal drain adds to a well's productivity once the friction along the drain is coupled to the
inflow, for the sampler redesign's D-21 (Joshi's index for drains of 300 to 1000 m, capped at 6 times the vertical
index) and checkpoint card B1's drain length (plans/sampler-redesign/decisions.md and checkpoint.md). ManyWells models
a drain well to its heel only (D-9, D-10), so the inflow model stands for the whole drain: the quantity is the drain's
effective index at the heel, J_eff = q_heel / (p_r - p_heel), over the vertical index J_V that the sampler draws.

The model is Dikken's (1990), as Joshi (1991, Ch. 10, Eqs. 10-19 to 10-38, pp. 401-405) gives it. The drain runs from
the toe (x = 0) to the heel (x = L). The reservoir feeds it uniformly per unit length, dq/dx = j_s (p_r - p_w(x)),
with j_s = J_H(L) / L and J_H Joshi's index as Economides et al. (1994, Eqs. 2-46 to 2-48, p. 33) give it, with the
sampler's skin term (I_ani h / L) s: a drain's skin loss goes with its rate per unit length (Joshi 1991, p. 103). The
liner carries single-phase oil, steady and isothermal: -dp_w/dx = f rho v^2 / (2 D) + d(rho v^2)/dx, the second term
the acceleration of the inflow, with Chen's (1979) friction factor (FRIC-4; 64/Re below Re = 2300) at roughness eps
times a perforation factor (continuously perforated liner: 1.4 to 1.7 times smooth pipe, Brekke and Lien 1994,
p. 205). A shooting method on the toe's drawdown (bisection on its logarithm, RK4 march to the heel) meets the heel's
drawdown or rate. A gas drain takes the same equations in p^2: the reservoir term Forchheimer's per unit length,
p_r^2 - p_w^2 = q'/j_s + b' q'^2 with b' = b_V (I_ani h)^2 (the vertical well's b over (L / (I_ani h))^2, since the
non-Darcy loss goes with the square of the rate per unit of completed length), and d(p^2)/dx = -f G^2 Z R T / (D M)
for a mass flux G, acceleration neglected.

With f constant (fully rough; Joshi's Eq. 10-30 with alpha' = 0) and no acceleration, du/dx = c q^2, and the drain's
efficiency eta = J_eff / J_H depends on one number, Pi = c J_H^2 u_h L for a heel drawdown u_h. The script fits
eta = (1 + Pi / 1.5)^(-1/3): 1 - Pi/4.5 for small Pi against the exact 1 - Pi/4, and the infinitely long drain's
q_max = (3 j_s u_h^2 / (2 c))^(1/3) for large Pi (from Joshi's Eq. 10-31), so that the productive length is about
(1.5 / (c j_s^2 u_h))^(1/3). The closed form, with c from f at the heel's rate, is tested against the full model.

Two drainage geometries give j_s for long drains, and they bound the answer:
  Joshi-B  Joshi's ellipse with r_eH = max(r_e, L / 1.6), so that L / 2 < 0.9 r_eH holds (Economides et al., Eq. 2-48);
           J_V keeps the drawn r_e. The drainage area grows with the drain in both directions, so J_H saturates at
           about 2 pi k h / (mu B ln 2.5) for long drains: a property of the geometry, not of friction.
  box      a line sink centred in a slab of pay h between constant-pressure lateral boundaries at +-x_e, after the
           anisotropy transform (Joshi 1991, Eqs. 3-15 to 3-19, p. 83): j_s = 2 pi k_h / (mu B (pi x_e / h
           + I ln(I h / (pi r_w (1 + I))) + I s)), with x_e = r_e / 2, an own choice close to Joshi's index at 1000 m
           for the median well. Fixed spacing between parallel drains: J_H grows with L until friction stops it.
Joshi's Eq. 4-1 and Fig. 4-2 (pp. 115-116) put the loss from an off-centre drain under 3% for L / h >= 10 anywhere
within the middle 90% of the pay, so the drain is centred here.

Checks against the books (printed first):
  1. Joshi's Table 3-2 (Joshi 1991, p. 89) and Example 3-1 (pp. 78-79) with his Eqs. 3-10, 3-11 and 3-20, and
     Economides et al.'s Example 2-10 (Eq. 2-51, p. 34: 1634 STB/d) with Eqs. 2-46 to 2-48.
  2. Dikken's example as Joshi's Example 10-5 (pp. 405-407): k = 1000 md, h = 65.6 ft, mu = 0.5 cp, rho = 0.7 g/cc, the
     per-length indices and drawdowns of the book's table, f = 0.3164 Re^-alpha'; against Figs. 10-8 and 10-9, read
     by eye to about +-10 m3/d.
  3. Brekke and Lien (1994), Troll's 800 m drain in a 15.24 cm liner (pp. 205-208): the liner's pressure drop at
     7479 Sm3/d with uniform inflow (Fig. 10) at their 0.025 mm roughness, and the conventional liner's drawdown at its
     toe and heel at 5000 Sm3/d (Fig. 11) with j_s from Fig. 10 and the perforation factor. The paper gives no oil
     properties beyond GOR 69 Sm3/Sm3 and 30% water cut (Fig. 3); the in-situ liquid's 850 kg/m3, 1.2 mPa s and
     B = 1.14 are own estimates (Standing's B_o at the GOR and 70 C).

Then a sweep over the sampler's ranges (sampling-draft.md, Productivity and Tubing), at the design drawdown with the
heel rate capped at the liner's erosional velocity (D-25's rule, C = 125, at the heel's density); the share of drains
above the cap over the draft's distributions; and, with --draws, the high-angle wells of R6's draws: their own k, h,
I_ani, skin, r_e, mu_o, B_o, design drawdown, tubing (as the liner) and tubing capacity (as the cap on the heel rate),
with drains redrawn to U(300, 2500) m. For gas drains it also tests a Forchheimer law for the drain at the heel,
a_H = a_V J_V / J_H and b_H = b_V (I_ani h / L)^2 + c L / 4 (the friction's first-order term, c at the heel's rate),
against the coupled model. Single-phase oil: below the bubble point, free gas in the liner lowers the density but
raises the volume rate, so rho v^2 = G^2 / rho and the friction grow, and a two-phase multiplier adds to it; water
changes little. Gas drains carry G^2 / rho at a far larger j_s and are the most friction-limited.

From the repository root:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 600 .venv/bin/python plans/evidence/drain_friction.py)

    --draws CSV  the per-well rows of R6's draws, as sampler_draft_draws.py --csv writes them (2000 wells, seed 2026)
    --seed N     the seed of the Monte Carlo draws and of the redrawn drain lengths (default 2026)

Sources: Joshi, S.D. (1991), Horizontal well technology, PennWell. Economides, M.J., Hill, A.D. and Ehlig-Economides, C.
(1994), Petroleum production systems, Prentice Hall. Dikken, B.J. (1990), Pressure drop in horizontal wells and its
effect on production performance, JPT 42(11), 1426-1433 (through Joshi). Brekke, K. and Lien, S.C. (1994), New, simple
completion methods for horizontal wells improve production performance in high-permeability thin oil zones, SPE
Drilling & Completion 9(3), 205-209. Chen, N.H. (1979), Ind. Eng. Chem. Fundam. 18(3), 296-297. Papay (1968) for Z;
Lee, Gonzalez and Eakin (1966), JPT 18(8), 997-1000, for the gas viscosity.
"""

import argparse

import numpy as np
from scipy.special import ndtr, ndtri

R_GAS, M_AIR, P_SC, T_SC = 8.314, 0.02897, 1.01325e5, 288.15
MD = 9.869233e-16  # m2 per mD
DAY = 86400.0
R_W = 0.108  # m, 8.5 in hole (sampling-draft.md, Productivity)
LINERS = (0.062, 0.076, 0.1005, 0.1243, 0.1571)  # m, D-25's inner diameters, 2-7/8 to 7 in
EPS = 30e-6  # m, liner roughness, about the draws' median (sampling-draft.md, Roughness)
PERF = 1.5  # friction of a continuously perforated liner over plain pipe: 1.4 to 1.7 (Brekke and Lien 1994, p. 205)
LENGTHS = (300, 500, 1000, 1500, 2000, 2500, 3000)  # m
GRID = np.arange(300.0, 4001.0, 50.0)  # m, for the length beyond which 500 m more adds under 10%
N_STEPS = 300  # RK4 steps along the drain


# --- Reservoir side --------------------------------------------------------------------------------------------------
def joshi_den(L, h, ani, r_eh, s, r_w=R_W, form='economides'):
    """The bracket of Joshi's index: Economides et al. (1994) Eqs. 2-46 to 2-48 with the skin as (I_ani h / L) s, or
    Joshi's (1991) Eqs. 3-10, 3-11 and 3-20, (beta h / L) ln(beta h / (2 r_w)), without skin."""
    a = L / 2 * np.sqrt(0.5 + np.sqrt(0.25 + (2 * r_eh / L) ** 4))
    t1 = np.log((a + np.sqrt(a * a - (L / 2) ** 2)) / (L / 2))
    if form == 'economides':
        return t1 + ani * h / L * (np.log(ani * h / (r_w * (ani + 1))) + s)
    return t1 + ani * h / L * np.log(ani * h / (2 * r_w))


def box_den_per_m(h, ani, x_e, s, r_w=R_W):
    """The bracket of the per-length index of a line sink centred in a slab between lateral constant-pressure
    boundaries at +-x_e, after the anisotropy transform: j_s = 2 pi k_h / (mu B * this)."""
    return np.pi * x_e / h + ani * (np.log(ani * h / (np.pi * r_w * (1 + ani))) + s)


def pi_scale(k_md, mu, B):
    """2 pi k / (mu B) in Sm3/(d bar) per metre, for a bracket of 1."""
    return 2 * np.pi * k_md * MD / (mu * B) * DAY * 1e5


def ratio(L, h, ani, s, ln_re, geometry='B'):
    """J_H / J_V without friction. Joshi's index with r_eH = r_e ('A') or r_eH = max(r_e, L / 1.6) ('B'), or the box;
    J_V pseudo-steady with the drawn r_e and skin."""
    r_e = R_W * np.exp(ln_re)
    if geometry == 'box':
        return L / h * (ln_re - 0.75 + s) / box_den_per_m(h, ani, r_e / 2, s)
    r_eh = r_e if geometry == 'A' else np.maximum(r_e, L / 1.6)
    return (ln_re - 0.75 + s) / joshi_den(L, h, ani, r_eh, s)


# --- Liner side --------------------------------------------------------------------------------------------------------
def chen(re, eps_d):
    """Darcy friction factor: 64/Re below Re = 2300, Chen (1979) above 4000, linear in Re between."""
    re = np.maximum(re, 1e-12)
    rt = np.maximum(re, 2300.0)
    a = eps_d / 3.7065 - 5.0452 / rt * np.log10(eps_d**1.1098 / 2.8257 + 5.8506 / rt**0.8981)
    f_t = (-2 * np.log10(a)) ** -2
    w = np.clip((re - 2300.0) / 1700.0, 0.0, 1.0)
    return np.where(re < 2300.0, 64 / re, (1 - w) * 64 / re + w * f_t)


def liquid_grad(D, rho, mu, B, eps=EPS, perf=PERF, alpha=None, accel=True):
    """du/dx in bar/m for a liquid at q Sm3/d with inflow dq Sm3/(d m); alpha gives Dikken's f = 0.3164 Re^-alpha.
    Also returns c(q), the coefficient of q^2 at the heel for the closed form, the acceleration folded in."""
    cv = B / (DAY * np.pi * D**2 / 4)  # m/s per Sm3/d

    def fric(q):
        re = rho * q * cv * D / mu
        return 0.3164 * np.maximum(re, 1.0) ** -alpha if alpha is not None else perf * chen(re, eps / D)

    def grad(q, dq):
        v = q * cv
        g = fric(q) * rho * v * v / (2 * D)
        if accel:
            g = g + 2 * rho * v * cv * dq
        return g / 1e5

    def c_heel(q, L):  # the heel's rho v^2 less its mean over a uniform inflow is 2/3 of it: c L / 4 = 2/3 rho cv^2
        return (fric(q) * rho * cv * cv / (2 * D) + (8 / 3 * rho * cv * cv / L if accel else 0.0)) / 1e5

    return grad, c_heel


def gas_grad(D, sg, Z, T, mu, eps=EPS, perf=PERF):
    """d(p^2)/dx in bar^2/m for gas at q Sm3/d: f G^2 Z R T / (D M), G the mass flux; and c(q) as above."""
    M = sg * M_AIR
    cg = P_SC * M / (R_GAS * T_SC) / (DAY * np.pi * D**2 / 4)  # kg/(m2 s) per Sm3/d

    def c_heel(q, L=None):
        return perf * chen(q * cg * D / mu, eps / D) * cg * cg * Z * R_GAS * T / (D * M) / 1e10

    def grad(q, dq):
        return c_heel(q) * q * q

    return grad, c_heel


def march(t, L, j, b2, u_ref, grad, n):
    """RK4 from the toe, where q = 0 and the drawdown is u_ref exp(t), to the heel; returns the heel's q and u. The
    inflow per metre solves u = q'/j + b2 q'^2 (b2 = 0: Darcy)."""
    dx = L / n
    q, u = np.zeros_like(t), u_ref * np.exp(t)
    big = 1e3 * u_ref
    a = 1 / j

    def rhs(q, u):
        dq = 2 * u / (a + np.sqrt(a * a + 4 * b2 * u))
        return dq, grad(q, dq)

    for _ in range(n):
        k1 = rhs(q, u)
        k2 = rhs(q + 0.5 * dx * k1[0], u + 0.5 * dx * k1[1])
        k3 = rhs(q + 0.5 * dx * k2[0], u + 0.5 * dx * k2[1])
        k4 = rhs(q + dx * k3[0], u + dx * k3[1])
        q = np.minimum(q + dx / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), big * j * L)
        u = np.minimum(u + dx / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), big)
    return q, u


def solve(L, j, grad, u_heel=None, q_heel=None, b2=0.0, n=N_STEPS, iters=48):
    """Shoot on the toe's drawdown to meet the heel's drawdown u_heel or rate q_heel; returns the heel's q and u and
    the toe's u. Both are monotone in the toe's drawdown, so bisection on its logarithm over 40 e-folds converges."""
    L, j, b2 = np.broadcast_arrays(np.asarray(L, float), np.asarray(j, float), np.asarray(b2, float))
    target = np.broadcast_to(np.asarray(u_heel if q_heel is None else q_heel, float), L.shape)
    u_ref = target if q_heel is None else target / (j * L)
    lo, hi = np.full(L.shape, -40.0), np.zeros(L.shape)
    if q_heel is not None:  # the toe's drawdown lies below the mean, about q / (j L): bracket a little above it
        hi = hi + 0.1 + np.log1p(4 * b2 * j * j * u_ref)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q, u = march(mid, L, j, b2, u_ref, grad, n)
        over = (u if q_heel is None else q) > target
        hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
    t = 0.5 * (lo + hi)
    return *march(t, L, j, b2, u_ref, grad, n), u_ref * np.exp(t)


def operating_point(L, j, grad, u_design, q_cap, b2=0.0):
    """The heel's q and u at the design drawdown, or at the cap on the rate where the design drawdown exceeds it, and
    whether the cap binds."""
    q, u, _ = solve(L, j, grad, u_heel=u_design, b2=b2)
    capped = q > q_cap
    if capped.any():
        qc, uc, _ = solve(L, j, grad, q_heel=np.minimum(q_cap, q), b2=b2)
        q, u = np.where(capped, qc, q), np.where(capped, uc, u)
    return q, u, capped


def eta_closed(Pi):
    return (1 + Pi / 1.5) ** (-1 / 3)


def cf_drawdown(J_H, L, c_heel, q):
    """The heel drawdown that the closed form needs for a heel rate q: q = J_H u (1 + k u)^(-1/3), k = c J_H^2 L / 1.5,
    grows with u."""
    k = c_heel(q, L) * J_H**2 * L / 1.5
    lo, hi = np.log(q / J_H), np.log(q / J_H) + 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        u = np.exp(mid)
        over = J_H * u * (1 + k * u) ** (-1 / 3) > q
        hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
    return np.exp(0.5 * (lo + hi))


def closed_form(J_H, L, c_heel, u_design, q_cap):
    """J_eff from eta = (1 + Pi / 1.5)^(-1/3), Pi = c J_H^2 u L with c at the heel's rate, at the design drawdown or
    at the cap on the rate."""
    q = J_H * u_design
    for _ in range(6):
        q = J_H * u_design * eta_closed(c_heel(q, L) * J_H**2 * u_design * L)
    qc = np.minimum(q, q_cap)
    return np.where(q > q_cap, qc / cf_drawdown(J_H, L, c_heel, qc), q / u_design)


def erosional_cap(D, rho, rho_sc, c_e=125.0):
    """Sm3/d at the erosional velocity 1.22 c_e / sqrt(rho) m/s (D-25's rule) at the heel's in-situ density rho, for a
    stream of rho_sc kg per Sm3 at standard conditions."""
    return 1.22 * c_e / np.sqrt(rho) * np.pi * D**2 / 4 * rho / rho_sc * DAY


# --- Gas properties ----------------------------------------------------------------------------------------------------
def z_papay(p_bar, T, sg):
    tpc = (169.2 + 349.5 * sg - 74.0 * sg**2) / 1.8
    ppc = (756.8 - 131.0 * sg - 3.6 * sg**2) * 0.0689476
    ppr, tpr = p_bar / ppc, T / tpc
    return 1 - 3.52 * ppr / 10 ** (0.9813 * tpr) + 0.274 * ppr**2 / 10 ** (0.8157 * tpr)


def mu_lge(p_bar, T, sg):
    """Gas viscosity in Pa s (Lee, Gonzalez and Eakin 1966)."""
    M = sg * M_AIR * 1000
    rho = p_bar * 1e5 * M / 1000 / (z_papay(p_bar, T, sg) * R_GAS * T) / 1000  # g/cm3
    T_r = T * 1.8
    K = (9.4 + 0.02 * M) * T_r**1.5 / (209 + 19 * M + T_r)
    X = 3.5 + 986 / T_r + 0.01 * M
    return 1e-4 * K * np.exp(X * rho ** (2.4 - 0.2 * X)) * 1e-3


def forchheimer_ab(k, h, s, ln_re, p_r, T, sg, D_nd):
    """The draft's vertical gas well: a in bar2 per Sm3/d and b in bar2 per (Sm3/d)^2 (sampler_draft_draws.py)."""
    den = ln_re - 0.75 + s
    a = P_SC * T * mu_lge(p_r, T, sg) * z_papay(p_r, T, sg) * den / (np.pi * k * MD * h * T_SC) / 1e10 / DAY
    return a, a * D_nd / 28.3168 / den


def gas_rate(a, b, u):
    return 2 * u / (a + np.sqrt(a * a + 4 * b * u))


# --- Draws -------------------------------------------------------------------------------------------------------------
def tnorm(rng, mu, sd, lo, hi, n):
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    return np.clip(mu + sd * ndtri(rng.uniform(a, b, n)), lo, hi)


def logu(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def draft_draws(rng, n, tail_share=0.2):
    """The drain's geometry draws of sampling-draft.md, Productivity (core and the coverage tail, D-103)."""
    tail = rng.uniform(size=n) < tail_share
    h = np.where(tail, logu(rng, 3, 100, n), np.exp(tnorm(rng, np.log(15), 0.8, np.log(3), np.log(100), n)))
    s = np.where(tail, rng.uniform(-3, 15, n), tnorm(rng, 2, 3, -3, 15, n))
    return h, logu(rng, 1, 3, n), s, rng.uniform(8.5, 9.5, n)


def pct(x, ps=(5, 50, 95)):
    return ' / '.join(f'{v:.2f}' for v in np.percentile(x, ps))


# --- Checks ------------------------------------------------------------------------------------------------------------
FT, RB, PSI = 0.3048, 0.158987, 0.0689476


def check_joshi():
    print("Check 1: Joshi's index against the books")
    r_e, r_w = 1489 * FT, 0.365 * FT  # Table 3-2: 160-acre spacing for both wells
    rows = [(25, 2000, 1.0, 7.28), (25, 500, 0.1, 2.58), (100, 1000, 0.1, 2.24), (100, 2000, 0.5, 5.66)]
    rows += [(400, 2000, 1.0, 3.52), (50, 1000, 0.5, 3.94)]
    for h, L, kvkh, book in rows:
        den = joshi_den(L * FT, h * FT, kvkh**-0.5, r_e, 0.0, r_w, form='joshi')
        print(f'  Table 3-2, h {h} ft, L {L} ft, kv/kh {kvkh}: J_h/J_v {np.log(r_e / r_w) / den:.2f} (book {book})')
    den = joshi_den(1000 * FT, 160 * FT, 1.0, 1053 * FT, 0.0, r_w, form='joshi')
    print(f'  Example 3-1: J_h/J_v {np.log(745 / 0.365) / den:.2f} (book 44 / 13 = 3.4 from rounded indices)')
    den = joshi_den(2000 * FT, 53 * FT, 3.0, 2980 * FT, 0.0, 0.328 * FT)
    q = 8.2 * 53 * (5651 - 3500) / (141.2 * 1.1 * 1.7 * den)
    print(f'  Economides et al. Example 2-10 (Eq. 2-51): q {q:.0f} STB/d (book 1634)')


def check_dikken():
    print("Check 2: Dikken's example (Joshi 1991, Example 10-5, Figs. 10-8 and 10-9), m3/d at reservoir conditions")
    sizes = {'4-1/2': (4.98, 0.5808, 6.19), '5-1/2': (6.22, 0.5845, 6.16), '7': (7.86, 0.5895, 6.10)}
    sizes['9-5/8'] = (11.03, 0.5996, 6.00)
    fig8 = {'4-1/2': 395, '5-1/2': 555, '7': 805, '9-5/8': 1380}  # read at 2000 m
    L = np.array([200.0, 400.0, 800.0, 1200.0, 2000.0])
    for name, (rw_cm, js_field, dp_psi) in sizes.items():
        D, js, u = 2 * rw_cm / 100, js_field * RB / (PSI * FT), dp_psi * PSI
        grad, c_heel = liquid_grad(D, 700.0, 5e-4, 1.0, alpha=0.0, accel=False)
        q, _, _ = solve(L, js, grad, u_heel=u, n=800)
        q_max = (3 * js * u**2 / (2 * c_heel(1.0, 1.0))) ** (1 / 3)
        rates = ', '.join(f'{int(x)} m {y:.0f}' for x, y in zip(L, q))
        print(f"  {name} in, alpha' = 0: {rates}; asymptote {q_max:.0f} (Fig. 10-8 at 2000 m: about {fig8[name]})")
    rw_cm, js_field, dp_psi = sizes['4-1/2']
    D, js, u = 2 * rw_cm / 100, js_field * RB / (PSI * FT), dp_psi * PSI
    for alpha, book in {0.0: 365, 0.15: 570, 0.25: 795}.items():
        out = []
        for mu in (5e-4, 5e-3):
            grad, _ = liquid_grad(D, 700.0, mu, 1.0, alpha=alpha, accel=False)
            q, _, _ = solve(np.array([400.0, 1000.0, 4000.0]), js, grad, u_heel=u, n=800)
            out.append(f'mu in Re {mu * 1e3:.1f} mPa s: 400 m {q[0]:.0f}, 1000 m {q[1]:.0f}, 4000 m {q[2]:.0f}')
        print(f"  4-1/2 in, alpha' = {alpha}: {'; '.join(out)} (Fig. 10-9: plateau about {book})")
    grad, _ = liquid_grad(D, 700.0, 5e-4, 1.0, alpha=0.25, accel=False)
    q_fine, _, _ = solve(np.array([2000.0]), js, grad, u_heel=u, n=6000)
    q_used, _, _ = solve(np.array([2000.0]), js, grad, u_heel=u)
    print(f"  RK4 with {N_STEPS} steps against 6000, alpha' = 0.25 at 2000 m: {q_used[0]:.2f} against {q_fine[0]:.2f}")
    Ls = np.logspace(-1, 2.2, 60)  # c = j = u = 1, so Pi = L^3
    q, _, _ = solve(Ls, np.ones_like(Ls), lambda q, dq: q * q, u_heel=1.0, n=1000)
    err = np.abs(eta_closed(Ls**3) / (q / Ls) - 1)
    print(f'  closed form eta = (1 + Pi/1.5)^(-1/3) against the march, Pi from 1e-3 to 6e6: within {err.max():.1%}')


def check_brekke_lien():
    print('Check 3: Brekke and Lien (1994), Troll, 800 m of 15.24 cm liner, Figs. 10 and 11')
    D, L, rho, mu, B = 0.1524, 800.0, 850.0, 1.2e-3, 1.14
    x = np.linspace(0, L, 2001)
    for eps in (25e-6, 0.0):
        grad, _ = liquid_grad(D, rho, mu, B, eps, 1.0)
        g = grad(7479 * x / L, np.full_like(x, 7479 / L))
        dp = np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(x))
        print(f'  liner drop at 7479 Sm3/d, uniform inflow, eps {eps * 1e3:.3f} mm: {dp:.2f} bar (Fig. 10: about 3.45)')
    js = 7479 / (L * (15625 - 15515) / 100)  # Sm3/(d bar m) from Fig. 10's flat drawdown of 1.10 bar
    print(f'  j_s from Fig. 10: {js:.1f} Sm3/(d bar m) (the paper: 6 to 12)')
    for perf in (1.4, 1.55, 1.7):
        grad, _ = liquid_grad(D, rho, mu, B, 25e-6, perf)
        _, uh, ut = solve(np.array([L]), js, grad, q_heel=5000.0, n=800)
        print(
            f'  conventional liner at 5000 Sm3/d, perforation factor {perf}: drawdown at the toe {ut[0]:.2f} bar, '
            f'at the heel {uh[0]:.2f} bar (Fig. 11: about 0.40 and 1.85)'
        )
    J = pi_scale(8500, mu, B) * 22 / joshi_den(L, 22.0, np.sqrt(8500 / 1500), 1500 / np.sqrt(np.pi), 0.0)
    print(f"  Joshi's index for the same well (8500 md, kv 1500 md, h 22 m, 1.5 km square): j_s {J / L:.1f}")


# --- Sweep -------------------------------------------------------------------------------------------------------------
def oil_batch(cases, L, geometry):
    """J_H/J_V without friction, J_eff/J_V with it, and the heel rate, for each case (a dict of scalars) and length,
    with I_ani 1.7, skin 2 and ln(r_e/r_w) 9 unless the case says otherwise; arrays of shape (cases, lengths)."""
    L = np.asarray(L, float)
    shape = (len(cases), len(L))

    def col(key, default):
        return np.broadcast_to(np.array([[c.get(key, default)] for c in cases], float), shape).ravel()

    k, h, ani, s, ln_re = col('k', 14), col('h', 16), col('ani', 1.7), col('s', 2.0), col('ln_re', 9.0)
    mu, B, rho, dd, D = col('mu', 5e-4), col('B', 1.45), col('rho', 700.0), col('dd', 70.0), col('D', 0.1005)
    perf, cap, Ls = col('perf', PERF), col('cap', 0.0), np.broadcast_to(L, shape).ravel()
    cap = np.where(cap > 0, cap, erosional_cap(D, rho, rho * B))
    J_V = pi_scale(k, mu, B) * h / (ln_re - 0.75 + s)
    J_H = J_V * ratio(Ls, h, ani, s, ln_re, geometry)
    grad, _ = liquid_grad(D, rho, mu, B, perf=perf)
    q, u, _ = operating_point(Ls, J_H / Ls, grad, dd, cap)
    return (J_H / J_V).reshape(shape), (q / u / J_V).reshape(shape), q.reshape(shape)


def gas_batch(cases, L):
    """The same for gas drains (Joshi-B; p_r 300 bar, 360 K, gravity 0.68, drawdown 25% of p_r, D 6e-4 d/Mscf): J_H/J_V,
    J_eff/J_V (in p^2), the heel rate and the vertical well's rate at the design drawdown, under the same cap."""
    L = np.asarray(L, float)
    shape = (len(cases), len(L))

    def col(key, default):
        return np.broadcast_to(np.array([[c.get(key, default)] for c in cases], float), shape).ravel()

    k, h, ani, s, ln_re, D = (
        col('k', 14),
        col('h', 16),
        col('ani', 1.7),
        col('s', 2.0),
        col('ln_re', 9.0),
        col('D', 0.1571),
    )
    p_r, T, sg, Ls = 300.0, 360.0, 0.68, np.broadcast_to(L, shape).ravel()
    a, b = forchheimer_ab(k, h, s, ln_re, p_r, T, sg, 6e-4)
    J_H = ratio(Ls, h, ani, s, ln_re, 'B') / a
    u_d = p_r**2 - (0.75 * p_r) ** 2
    Z, M = z_papay(p_r, T, sg), sg * M_AIR
    rho_heel = 0.75 * p_r * 1e5 * M / (Z * R_GAS * T)
    cap = erosional_cap(D, rho_heel, P_SC * M / (R_GAS * T_SC))
    grad, _ = gas_grad(D, sg, Z, T, mu_lge(p_r, T, sg))
    q, u, _ = operating_point(Ls, J_H / Ls, grad, u_d, cap, b2=b * (ani * h) ** 2)
    return (
        (J_H * a).reshape(shape),
        (q / u * a).reshape(shape),
        q.reshape(shape),
        np.minimum(gas_rate(a, b, u_d), cap).reshape(shape),
    )


def l10(J):
    """The shortest length on GRID beyond which a further 500 m adds under 10% to J."""
    for i, Li in enumerate(GRID):
        k = np.searchsorted(GRID, Li + 500)
        if k < len(GRID) and J[k] < 1.1 * J[i]:
            return f'{Li:.0f}'
    return f'>{GRID[-1] - 500:.0f}'


def sweep():
    print('\nSweep: J_H/J_V without friction | with friction, by drain length (m); oil at the design drawdown')
    print("(70 bar), the heel rate capped at the liner's erosional velocity (C = 125: 2725 Sm3/d in 100 mm, 6660 in")
    print('157 mm); I_ani 1.7, skin 2, ln(r_e/r_w) 9 (r_e 875 m),')
    print(
        f'mu_o 0.5 mPa s, B_o 1.45, 700 kg/m3; perforated liner (Chen at {EPS * 1e6:.0f} um, x{PERF}); Joshi-B or box.'
    )
    print('L_10: the length beyond which 500 m more adds under 10% to J, without friction / with it.')
    print(f'  {"case":34s}' + ''.join(f'{L:>14d}' for L in LENGTHS) + f'{"L_10":>15s}')
    oils = {
        'k 14 mD (median)': dict(k=14),
        'k 250 mD (P95)': dict(k=250),
        'k 3000 mD (edge)': dict(k=3000),
        'k 1000, h 5 m, 1 mPa s, 20 bar': dict(k=1000, h=5, mu=1e-3, dd=20.0),
    }
    cases = [dict(c, D=D) for c in oils.values() for D in (0.1005, 0.1571)]
    names = [f'{n}, {D * 1000:.0f} mm' for n in oils for D in (0.1005, 0.1571)]
    for geometry in ('B', 'box'):
        r0, r1, q = oil_batch(cases, LENGTHS, geometry)
        g0, g1, _ = oil_batch(cases, GRID, geometry)
        print(f'  {"Joshi-B" if geometry == "B" else "box, x_e = r_e/2"}:')
        for i, name in enumerate(names):
            cells = ''.join(f'{a:6.1f} |{b:6.1f}' for a, b in zip(r0[i], r1[i]))
            print(f'  {name:34s}{cells}{l10(g0[i]):>8s} / {l10(g1[i])}')
            print(f'  {"  heel rate, Sm3/d":34s}' + ''.join(f'{x:14.0f}' for x in q[i]))
    print('  gas, Joshi-B (p_r 300 bar, 360 K, gravity 0.68, design drawdown 25% of p_r, the vertical well with')
    print("  D = 6e-4 d/Mscf), heel rate capped at the liner's erosional velocity at 225 bar; J in p^2:")
    gases = {'gas, k 14 mD': dict(k=14), 'gas, k 250 mD': dict(k=250)}
    cases = [dict(c, D=D) for c in gases.values() for D in (0.1243, 0.1571)]
    names = [f'{n}, {D * 1000:.0f} mm' for n in gases for D in (0.1243, 0.1571)]
    r0, r1, q, q_v = gas_batch(cases, LENGTHS)
    g0, g1, _, _ = gas_batch(cases, GRID)
    for i, name in enumerate(names):
        cells = ''.join(f'{a:6.1f} |{b:6.1f}' for a, b in zip(r0[i], r1[i]))
        print(f'  {name:34s}{cells}{l10(g0[i]):>8s} / {l10(g1[i])}')
        print(f'  {"  heel rate, 1e6 Sm3/d":34s}' + ''.join(f'{x / 1e6:14.2f}' for x in q[i]))
        label = "  over the vertical well's (same cap)"
        print(f'  {label:34s}' + ''.join(f'{x:14.1f}' for x in q[i] / q_v[i]))

    print('\nFriction alone: J_eff/J_H at 2500 m (Joshi-B; the median well otherwise) by permeability and liner')
    print('  ' + ' ' * 14 + ''.join(f'{D * 1000:>10.0f} mm' for D in LINERS))
    ks = (1, 15, 100, 500, 1500, 3000)
    cases = [dict(k=k, D=D) for k in ks for D in LINERS]
    r0, r1, _ = oil_batch(cases, [2500.0], 'B')
    eta = (r1 / r0).reshape(len(ks), len(LINERS))
    for k, row in zip(ks, eta):
        print(f'  k {k:5d} mD   ' + ''.join(f'{x:13.2f}' for x in row))
    print('  and by viscosity, pay, anisotropy, drawdown and completion at k 500 mD, 124 mm, 2500 m:')
    sens = {
        'mu 0.2 mPa s': dict(mu=0.2e-3),
        'mu 5 mPa s': dict(mu=5e-3),
        'h 5 m': dict(h=5),
        'h 50 m': dict(h=50),
        'I_ani 1': dict(ani=1.0),
        'I_ani 3': dict(ani=3.0),
        'drawdown 20 bar': dict(dd=20.0),
        'drawdown 120 bar': dict(dd=120.0),
        'plain pipe (x1.0)': dict(perf=1.0),
        'no cap on the rate': dict(cap=1e9),
    }
    r0, r1, q = oil_batch([dict(c, k=500, D=0.1243) for c in sens.values()], [2500.0], 'B')
    for i, name in enumerate(sens):
        print(
            f'    {name:20s} J_eff/J_H {r1[i, 0] / r0[i, 0]:.2f}, J_H/J_V {r0[i, 0]:.1f}, heel rate {q[i, 0]:.0f} Sm3/d'
        )
    print('  the heel drawdown over the mean that friction costs per km of drain at 5 m/s at the heel, uniform inflow')
    print('  (c L / 4 per km at 700 kg/m3, 0.5 mPa s, x1.5):', end='')
    for D in LINERS:
        _, c_heel = liquid_grad(D, 700.0, 5e-4, 1.0, accel=False)
        q5 = 5.0 * np.pi * D**2 / 4 * DAY
        print(f' {D * 1000:.0f} mm {c_heel(q5, 1000.0) * q5**2 * 1000 / 4:.1f} bar;', end='')
    print()


def cap_frequency(rng):
    print("\nJoshi's J_H/J_V without friction over the draft's h, I_ani, skin and r_e (Monte Carlo, core and tail at")
    print('0.8 / 0.2): P5 / P50 / P95 and the shares above 6, 8, 10 and 15')
    n = 200000
    h, ani, s, ln_re = draft_draws(rng, n)
    r_e = R_W * np.exp(ln_re)
    variants = {
        'draft: L ~ U(300, min(1000, 1.6 r_e)), r_eH = r_e': ('A', np.minimum(1000, 1.6 * r_e)),
        'L ~ U(300, min(2500, 1.6 r_e)), r_eH = r_e': ('A', np.minimum(2500, 1.6 * r_e)),
        'L ~ U(300, 2500), r_eH = max(r_e, L/1.6)': ('B', np.full(n, 2500.0)),
        'L ~ U(300, 2500), box with x_e = r_e/2': ('box', np.full(n, 2500.0)),
    }
    for name, (geometry, top) in variants.items():
        r = ratio(rng.uniform(300, top), h, ani, s, ln_re, geometry)
        shares = ', '.join(f'{np.mean(r > c):.0%}' for c in (6, 8, 10, 15))
        print(f'  {name:50s} {pct(r)}; {shares}')
    L = rng.uniform(300, 2500, n)
    r = ratio(L, h, ani, s, ln_re, 'B')
    print('  Joshi-B to 2500 m by skin and by length: share of wells, P50 and share above 6')
    for lo, hi in ((-3, 0), (0, 4), (4, 8), (8, 15.1)):
        m = (s >= lo) & (s < hi)
        print(f'    skin [{lo}, {hi:.0f}): {m.mean():.0%}, P50 {np.median(r[m]):.1f}, above 6 {np.mean(r[m] > 6):.0%}')
    for lo, hi in ((300, 1000), (1000, 1750), (1750, 2500)):
        m = (L >= lo) & (L < hi)
        print(f'    L [{lo}, {hi}) m: P50 {np.median(r[m]):.1f}, above 6 {np.mean(r[m] > 6):.0%}')


def draws(path, rng, reps=20):
    import pandas as pd

    d = pd.read_csv(path)
    ha = d[d.geometry_family == 'high_angle']
    oil = ha[ha.fluid_type.isin(['black_oil', 'volatile_oil'])]
    gas = ha[~ha.fluid_type.isin(['black_oil', 'volatile_oil'])]
    print(f"\nR6's draws ({path.split('/')[-1]}): {len(oil)} high-angle oil wells, {len(gas)} gas and condensate wells")
    raw = ratio(*(oil[c].to_numpy() for c in ('drain_length', 'h_net', 'I_ani', 'skin', 'ln_re_rw')), 'A')
    print(f'  oils as drawn (300 to 1000 m): uncapped J_H/J_V {pct(raw)}, at the cap of 6: {np.mean(raw > 6):.0%}')

    def rep(frame, col):
        return np.repeat(frame[col].to_numpy(), reps)

    k, h, ani, s, ln_re = (rep(oil, c) for c in ('perm', 'h_net', 'I_ani', 'skin', 'ln_re_rw'))
    mu, B, wlr, p_r = (rep(oil, c) for c in ('mu_o', 'B_o', 'wlr', 'p_r'))
    B_l = (1 - wlr) * B + wlr
    rho = ((1 - wlr) * (rep(oil, 'rho_o') + rep(oil, 'R_go') * rep(oil, 'rho_g')) + wlr * rep(oil, 'rho_w')) / B_l
    rho = np.clip(rho, 400, 1100)
    mu_l = (1 - wlr) * mu + wlr * 0.5e-3  # water's 0.5 mPa s: an own estimate
    D, cap, u_d = rep(oil, 'D'), rep(oil, 'q_cap'), rep(oil, 'delta_d') * p_r
    J_V = np.repeat((oil.J / oil.f_geo).to_numpy(), reps)
    L = rng.uniform(300, 2500, len(k))
    print(f'  drains redrawn to U(300, 2500) m, {reps} per well; liner = the drawn tubing; heel rate capped at the')
    print(f"  tubing's erosional capacity (q_cap); Chen at {EPS * 1e6:.0f} um, x{PERF}; P5 / P50 / P95")
    for geometry in ('B', 'box'):
        J_H = J_V * ratio(L, h, ani, s, ln_re, geometry)
        grad, c_heel = liquid_grad(D, rho, mu_l, B_l)
        q, u, capped = operating_point(L, J_H / L, grad, u_d, cap)
        eta, r1 = q / u / J_H, q / u / J_V
        J_cf = closed_form(J_H, L, c_heel, u_d, cap)
        name = 'Joshi-B' if geometry == 'B' else 'box'
        r0 = J_H / J_V
        print(f'  {name}: J_H/J_V {pct(r0)}, above 6 {np.mean(r0 > 6):.0%}, 10 {np.mean(r0 > 10):.0%}')
        print(
            f'    with friction: J_eff/J_V {pct(r1)}, above 6 {np.mean(r1 > 6):.0%}, 10 {np.mean(r1 > 10):.0%}; '
            f'friction takes over 10% of J_H in {np.mean(eta < 0.9):.1%}, over 30% in {np.mean(eta < 0.7):.1%}; '
            f'rate capped in {capped.mean():.0%}'
        )
        for lo, hi in ((0, 100), (100, 500), (500, 1e5)):
            m = (k >= lo) & (k < hi)
            print(
                f'      k [{lo:.0f}, {hi:.0f}) mD: {m.mean():.0%} of drains, J_eff/J_H P5 / P50 {pct(eta[m], (5, 50))}'
            )
        err = np.abs(J_cf / (q / u) - 1)
        print(f'    closed form against the march: |error| P50 {np.median(err):.2%}, P99 {np.percentile(err, 99):.1%}')

    n = len(gas) * reps
    p_r, T, sg, delta = (rep(gas, c) for c in ('p_r', 'T_r', 'sg_gas', 'delta_d'))
    h, s, ln_re, I_g = rep(gas, 'h_net'), rep(gas, 'skin'), rep(gas, 'ln_re_rw'), logu(rng, 1, 3, n)
    a_v, b_v = rep(gas, 'a_gas') * rep(gas, 'f_geo'), rep(gas, 'b_gas') * rep(gas, 'f_geo')  # the vertical well
    Z, mu_g = z_papay(p_r, T, sg), np.clip(mu_lge(p_r, T, sg), 1e-5, 1e-4)
    q_cap = rep(gas, 'q_cap') * 1e6 / (1e6 / rep(gas, 'R_i') + 10)  # the tubing's cap, as gas at the design WGR
    L = rng.uniform(300, 2500, n)
    r0 = ratio(L, h, I_g, s, ln_re, 'B')
    J_H, u_d = r0 / a_v, p_r**2 - ((1 - delta) * p_r) ** 2
    b2 = b_v * (I_g * h) ** 2
    grad, c_heel = gas_grad(rep(gas, 'D').clip(max=0.1571), sg, Z, T, mu_g)
    q, u, capped = operating_point(L, J_H / L, grad, u_d, q_cap, b2=b2)
    q_free, _, _ = solve(L, J_H / L, grad, u_heel=u_d, b2=b2)  # at the design drawdown, no cap
    eta = q / u / J_H
    q_v = gas_rate(a_v, b_v, u_d)
    f_eq = (a_v * q + b_v * q * q) / u  # the factor on k h that the draft's vertical law needs at the drain's rate
    b_H = b_v * (I_g * h / L) ** 2  # the drain's own non-Darcy term
    u_cf = cf_drawdown(J_H, L, c_heel, q) + b_H * q * q  # the closed form's heel drawdown at the drain's rate
    b_fit = b_H + c_heel(q, L) * L / 4  # Forchheimer for the drain: a = a_v / ratio, this b
    print(f'  gas and condensate drains, Joshi-B (I_ani drawn as for oil; liner = tubing up to 157 mm), {n} drains:')
    print(f"    J_H/J_V (Darcy, the same as oil's) {pct(r0)}; J_eff/J_H at the operating point {pct(eta)}")
    print(
        f'    friction takes over 10% of J_H in {np.mean(eta < 0.9):.0%}, over 30% in {np.mean(eta < 0.7):.0%}; rate capped'
    )
    print(f'    in {capped.mean():.0%}; drain over vertical rate at the design drawdown, uncapped: {pct(q_free / q_v)}')
    print(f"    the vertical well's non-Darcy share of its drawdown there: {pct(b_v * q_v**2 / u_d)}")
    print(f"    factor on k h that the draft's law (a and b both divided) needs at the drain's rate: {pct(f_eq)};")
    print(f'    within U(1.5, 4): {np.mean((f_eq >= 1.5) & (f_eq <= 4)):.0%}')
    err = np.abs(u_cf / u - 1)
    print(
        f"    closed form plus b_v (I h/L)^2 q^2, heel drawdown at the drain's rate: |error| P50 {np.median(err):.2%}, "
        f'P99 {np.percentile(err, 99):.1%}'
    )
    u_fit = a_v / r0 * q + b_fit * q * q
    err = np.abs(u_fit / u - 1)
    print(
        f'    Forchheimer a_v/ratio, b_v (I h/L)^2 + c L/4 at the operating rate: drawdown |error| P50 {np.median(err):.1%},'
    )
    print(
        f'    P95 {np.percentile(err, 95):.0%}; where friction takes under 30%: P95 {np.percentile(err[eta > 0.7], 95):.0%}'
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--draws', help="R6's per-well draws (CSV)")
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    check_joshi()
    check_dikken()
    check_brekke_lien()
    sweep()
    cap_frequency(rng)
    if args.draws:
        draws(args.draws, rng)


if __name__ == '__main__':
    main()
