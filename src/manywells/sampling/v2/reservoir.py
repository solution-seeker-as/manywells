"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The reservoir's closed forms that the v2 draws evaluate on floats (specs/sampling.md): the producing gas-oil ratio
under solution-gas drive (SMP-70), an oil well's radial productivity index, a slanted well's skin and a drain's index
(SMP-72), and a gas well's radial Forchheimer coefficients with their non-Darcy coefficient (SMP-71, SMP-73).

Sources: Craft and Hawkins (1991), Applied petroleum reservoir engineering, 2nd ed., Ch. 9 and 10; Economides, Hill
and Ehlig-Economides (1994), Petroleum production systems, with Cinco-Ley, Ramey and Miller's (1975) slant skin in
Section 5-4; Joshi (1991), Horizontal well technology, Ch. 10 (Dikken's
model); Brekke and Lien (1994), SPE Drilling & Completion 9(3); Chen (1979), Ind. Eng. Chem. Fundam. 18(3). The
measurements behind the closed forms are plans/evidence/solution_gas_drive.py's and drain_friction.py's.
"""

import functools
import math

import casadi as ca
import numpy as np

from manywells.pvt.black_oil import BlackOil
from manywells.units import P_REF, T_REF

_OILS = ('black_oil', 'volatile_oil')  # The fluid types with an oil's inflow and material balance
_R_W = 0.108  # Wellbore radius (m), an 8.5 in hole  # spec: SMP-71
_MD = 9.869233e-16  # m2 per mD
_DAY = 86400.0  # s
_MSCF = 28.3168  # Sm3 per Mscf
_R_GAS, _M_AIR = 8.314, 0.02897  # J/(mol K), kg/mol
_PERFORATED = 1.5  # Friction of a continuously perforated liner over plain pipe (Brekke and Lien 1994, p. 205)


# --- The producing GOR under solution-gas drive ----------------------------------------------------------------------
# Craft and Hawkins (1991): Fig. 9.16 (p. 373), k_g/k_o against the total liquid saturation, at a connate water of
# 0.20; and Table 10.2 (p. 404), the upper gas saturation of each segment with A0 and A1 of
# log10(k_g/k_o) = A0 + A1 S_g, at a connate water of 0.37
_FIG_9_16 = np.array(
    [
        [0.301, 34.7],
        [0.351, 12.85],
        [0.506, 1.99],
        [0.702, 0.195],
        [0.842, 0.0351],
        [0.893, 0.0149],
        [0.903, 0.00852],
        [0.914, 0.00291],
        [0.921, 0.00098],
    ]
)
_TABLE_10_2 = np.array(
    [[0.0430, -4.7981, 54.6520], [0.0810, -3.8450, 32.0170], [0.1090, -2.6308, 16.9890], [0.3500, -2.1119, 12.2410]]
)
_S_W_FIG, _S_W_TABLE = 0.20, 0.37
_FIG_X, _FIG_Y = _FIG_9_16[:, 0], np.log10(_FIG_9_16[:, 1])
_X = np.linspace(1.0, 0.05, 2 * 380 + 1)  # p/p_sat from the saturation pressure down, with RK4's midpoints
_GOR_NODES = _X[::2][::-1].copy()  # p_r/p_sat at the nodes of an oil's GOR curve, ascending: linear between them
_GOR_NODES.setflags(write=False)


def _kgko_fig_9_16(sg):
    """Fig. 9.16's k_g/k_o at the gas saturation sg of the hydrocarbon pore space: log-linear between the figure's
    points and beyond its end ones."""
    S_L = _S_W_FIG + (1 - _S_W_FIG) * (1 - np.asarray(sg, dtype=float))
    x, y = _FIG_X, _FIG_Y
    out = np.interp(S_L, x, y)
    lo, hi = S_L < x[0], S_L > x[-1]
    out = np.where(lo, y[0] + (S_L - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), out)
    out = np.where(hi, y[-1] + (S_L - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), out)
    return 10.0**out


def _kgko_table_10_2(sg):
    """Table 10.2's k_g/k_o at the gas saturation sg of the hydrocarbon pore space; the last segment beyond its end."""
    S_g = np.maximum(np.asarray(sg, dtype=float) * (1 - _S_W_TABLE), 0.0)
    idx = np.minimum(np.searchsorted(_TABLE_10_2[:, 0], S_g, side='left'), len(_TABLE_10_2) - 1)
    return 10.0 ** (_TABLE_10_2[idx, 1] + _TABLE_10_2[idx, 2] * S_g)


def _kgko(sg, w):
    """The book's laboratory curves blended in log10 k_g/k_o: Fig. 9.16 at w = 0 (the least mobile gas) to Table
    10.2 at w = 1 (the most)."""
    return 10.0 ** ((1 - w) * np.log10(_kgko_fig_9_16(sg)) + w * np.log10(_kgko_table_10_2(sg)))


@functools.lru_cache(maxsize=4096)
def _gor_curve(rho_o, rho_g, p_sep, T_sep, T_r, p_sat, w_kgko):  # spec: SMP-70
    """
    R/R_s(p_sat) against p_r/p_sat, both ascending, under solution-gas drive: Muskat's material balance (Craft and
    Hawkins 1991, Eqs. 9.33 to 9.35) in the hydrocarbon pore space, integrated by RK4 in pressure from the saturation
    pressure down to 0.05 of it, on the black oil BlackOil(rho_o, rho_g, 'dak', p_sep, T_sep) at T_r, with k_g/k_o
    blended at w_kgko.
    """
    oil = BlackOil(rho_o=rho_o, rho_g=rho_g, gas_law='dak', p_sep=p_sep, T_sep=T_sep)
    M, rel = len(_X), 1e-4
    p = _X * p_sat
    P = ca.DM(np.concatenate([p, p * (1 + rel), p * (1 - rel)]))  # The grid, and its central differences

    def arr(v):
        return np.array(v).ravel()

    rs, bo, rho = arr(oil.rs(P, T_r)), arr(oil.bo(P, T_r)), arr(oil.gas_density(P, T_r))
    bg = oil.rho_g / rho  # B_g: the gas density at standard conditions over the reservoir's
    mu_o = arr(oil.oil_viscosity(ca.DM(p), T_r))
    mu_g = arr(oil.gas_viscosity(ca.DM(p), T_r, ca.DM(rho[:M])))
    dp = 2 * rel * p
    (Rs, dRs), (Bo, dBo), (Bg, dBg) = ((v[:M], (v[M : 2 * M] - v[2 * M :]) / dp) for v in (rs, bo, bg))
    mu_r = mu_o / mu_g
    X, Y, Z = Bg / Bo * dRs, dBo / Bo, dBg / Bg  # Eq. 9.35

    def f(i, s):  # Eq. 9.34: the oil saturation's change with pressure
        km = _kgko(1 - s, w_kgko) * mu_r[i]
        return (s * X[i] + s * km * Y[i] - (1 - s) * Z[i]) / (1 + km)

    n = (M - 1) // 2
    s = np.ones(1)  # The oil saturation of the hydrocarbon pore space, 1 at the saturation pressure
    out = np.empty(n + 1)
    out[0] = s[0]
    for k in range(n):
        i = 2 * k
        h = p[i + 2] - p[i]
        k1 = f(i, s)
        k2 = f(i + 1, s + h / 2 * k1)
        k3 = f(i + 1, s + h / 2 * k2)
        k4 = f(i + 2, s + h * k3)
        s = np.clip(s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
        out[k + 1] = s[0]
    e = slice(0, None, 2)
    R = Rs[e] + _kgko(1 - out, w_kgko) * mu_r[e] * Bo[e] / Bg[e]  # Eq. 9.33
    return _GOR_NODES, R[::-1] / Rs[0]


def _producing_gor(w, p_r) -> float:  # spec: SMP-70
    """
    The producing gas-oil ratio (Sm3/Sm3) of a drawn well at reservoir pressure p_r (bar).

    Oils: R_si at or above the saturation pressure, and below it R_si times the material balance's factor at
    p_r/p_sat, linear between GOR_NODES. Gas condensates: R_si (1 + c_k max(0, 1 - p_r/p_sat)). Wet and dry gas: R_si.

    :param w: The well's draws (draws.draw_well): its fluid, separator, T_r, p_sat, R_i, and w_kgko or c_k
    """
    if w['fluid_type'] in _OILS:
        if p_r >= w['p_sat']:
            return w['R_i'] * 1.0
        keys = ('rho_o', 'rho_g', 'p_sep', 'T_sep', 'T_r', 'p_sat', 'w_kgko')
        x, F = _gor_curve(*(float(w[k]) for k in keys))
        return w['R_i'] * float(np.interp(p_r / w['p_sat'], x, F))
    if w['fluid_class'] == 'gas_condensate':
        return w['R_i'] * (1 + w['c_k'] * max(0.0, 1 - p_r / w['p_sat']))
    return w['R_i']


# --- Oil wells' productivity -----------------------------------------------------------------------------------------
def _j_radial(k_md, h, mu, B, ln_term, s) -> float:  # spec: SMP-72
    """
    Pseudo-steady radial productivity index, Sm3/(d bar) of stock-tank liquid.

    :param k_md: Permeability (mD)
    :param h: Net pay (m)
    :param mu: Oil viscosity (Pa s)
    :param B: Oil formation volume factor (m3/Sm3)
    :param ln_term: ln(r_e/r_w) - 0.75
    :param s: Skin
    """
    return 2 * math.pi * k_md * _MD * h / (mu * B * (ln_term + s)) * _DAY * 1e5


def _slant_skin(theta, h, I_ani) -> float:  # spec: SMP-72
    """
    Cinco-Ley, Ramey and Miller's (1975) skin of a fully completed slanted well, by the fit to their tables
    (Economides et al. 1994, Section 5-4, Tables 5-1 and 5-2): -(t/41)^2.06 - (t/56)^1.865 log10(h_D/100), with
    t = arctan(tan(theta) / I_ani) in degrees and h_D = h I_ani / r_w. Valid to 75 degrees.

    :param theta: The well's inclination through the pay (degrees)
    :param h: Net pay (m)
    :param I_ani: Anisotropy, sqrt(k_h/k_v)
    """
    t = math.degrees(math.atan(math.tan(math.radians(theta)) / I_ani))
    return -((t / 41) ** 2.06) - (t / 56) ** 1.865 * math.log10(h * I_ani / _R_W / 100)


def _joshi_ratio(L_d, h, I_ani, s, ln_re) -> float:  # spec: SMP-72
    """
    J_H/J_V of a horizontal drain without friction: Joshi's index (Economides et al. 1994, Eqs. 2-46 to 2-48, with
    the skin as (I_ani h / L_d) s) on the drainage length r_eH = max(r_e, L_d / 1.6), over the pseudo-steady radial
    index with the drawn r_e and skin.

    :param L_d: Drain length (m)
    :param h: Net pay (m)
    :param I_ani: Anisotropy, sqrt(k_h/k_v)
    :param s: Skin
    :param ln_re: ln(r_e/r_w)
    """
    r_eh = np.maximum(_R_W * np.exp(ln_re), L_d / 1.6)
    a = L_d / 2 * np.sqrt(0.5 + np.sqrt(0.25 + (2 * r_eh / L_d) ** 4))
    t1 = np.log((a + np.sqrt(a * a - (L_d / 2) ** 2)) / (L_d / 2))
    return float((ln_re - 0.75 + s) / (t1 + I_ani * h / L_d * (np.log(I_ani * h / (_R_W * (I_ani + 1))) + s)))


def _chen(re, eps_d):
    """Darcy friction factor: 64/Re below Re = 2300, Chen (1979) above 4000, linear in Re between."""
    re = np.maximum(re, 1e-12)
    rt = np.maximum(re, 2300.0)
    a = eps_d / 3.7065 - 5.0452 / rt * np.log10(eps_d**1.1098 / 2.8257 + 5.8506 / rt**0.8981)
    f_t = (-2 * np.log10(a)) ** -2
    w = np.clip((re - 2300.0) / 1700.0, 0.0, 1.0)
    return np.where(re < 2300.0, 64 / re, (1 - w) * 64 / re + w * f_t)


def _liner_liquid(D, rho, mu, B, eps):  # spec: SMP-72
    """
    A liquid's friction and acceleration along a drain's perforated liner: the function c(q, L_d), in bar per
    (Sm3/d)2 per metre, with which the drawdown grows along the liner as c q2, at the heel's rate q (Sm3/d).

    :param D: Liner's inner diameter (m)
    :param rho: In-situ liquid density (kg/m3)
    :param mu: In-situ liquid viscosity (Pa s)
    :param B: Formation volume factor of the liquid (m3/Sm3)
    :param eps: Wall roughness (m)
    """
    cv = B / (_DAY * np.pi * D**2 / 4)  # m/s per Sm3/d

    def c(q, L_d):  # The heel's rho v2 less its mean over a uniform inflow is 2/3 of it: c L / 4 = 2/3 rho cv2
        f_D = _PERFORATED * _chen(rho * q * cv * D / mu, eps / D)
        return (f_D * rho * cv * cv / (2 * D) + (8 / 3 * rho * cv * cv / L_d)) / 1e5

    return c


def _liner_gas(D, sg, Z, T, mu, eps):  # spec: SMP-73
    """
    A gas's friction along a drain's perforated liner: the function c(q), in bar2 per (Sm3/d)2 per metre, with which
    p2 falls along the liner as c q2, at the heel's rate q (Sm3/d).

    :param D: Liner's inner diameter (m)
    :param sg: Gas gravity (air = 1)
    :param Z: Compressibility factor of the gas
    :param T: Temperature (K)
    :param mu: Gas viscosity (Pa s)
    :param eps: Wall roughness (m)
    """
    M = sg * _M_AIR
    cg = P_REF * M / (_R_GAS * T_REF) / (_DAY * np.pi * D**2 / 4)  # kg/(m2 s) per Sm3/d

    def c(q):
        return _PERFORATED * _chen(q * cg * D / mu, eps / D) * cg * cg * Z * _R_GAS * T / (D * M) / 1e10

    return c


def _eta(Pi):
    """Dikken's friction efficiency J_eff/J_H in closed form."""
    return (1 + Pi / 1.5) ** (-1 / 3)


def _drawdown(J_H, L_d, c, q):
    """The heel's drawdown that the closed form needs for a heel rate q: q = J_H u (1 + k u)^(-1/3), with
    k = c J_H^2 L_d / 1.5, grows with u."""
    k = c(q, L_d) * J_H**2 * L_d / 1.5
    lo, hi = np.log(q / J_H), np.log(q / J_H) + 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        u = np.exp(mid)
        over = J_H * u * (1 + k * u) ** (-1 / 3) > q
        hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
    return np.exp(0.5 * (lo + hi))


def _drain_index(J_H, L_d, c, u, q_cap) -> float:  # spec: SMP-72
    """
    A drain's productivity index at the heel with the friction along its liner, Sm3/(d bar): J_H times Dikken's
    efficiency in closed form, eta = (1 + Pi / 1.5)^(-1/3) with Pi = c J_H^2 u L_d and c at the heel's rate, at the
    drawdown u or at the cap on the rate.

    :param J_H: The drain's index without friction, Sm3/(d bar)
    :param L_d: Drain length (m)
    :param c: The liner's coefficient (_liner_liquid)
    :param u: Heel drawdown (bar)
    :param q_cap: Cap on the heel's rate (Sm3/d)
    """
    q = J_H * u
    for _ in range(6):
        q = J_H * u * _eta(c(q, L_d) * J_H**2 * u * L_d)
    qc = np.minimum(q, q_cap)
    return float(np.where(q > q_cap, qc / _drawdown(J_H, L_d, c, qc), q / u))


# --- Gas wells' radial coefficients ----------------------------------------------------------------------------------
def _forchheimer(k_md, h, mu, Z, T, ln_term, s, D_nd) -> tuple:  # spec: SMP-73
    """
    a (bar2 per Sm3/d) and b (bar2 per (Sm3/d)2) of a vertical gas well's p_r^2 - p_wf^2 = a q + b q^2, with q the
    standard gas rate (Economides et al. 1994, Eqs. 4-32 and 4-33): the radial gas-flow equation in SI units, the
    non-Darcy flow as a skin D q.

    :param k_md: Permeability (mD)
    :param h: Net pay (m)
    :param mu: Gas viscosity (Pa s)
    :param Z: Compressibility factor of the gas
    :param T: Reservoir temperature (K)
    :param ln_term: ln(r_e/r_w) - 0.75
    :param s: Skin
    :param D_nd: Non-Darcy coefficient (d/Mscf)
    """
    a_si = P_REF * T * mu * Z * (ln_term + s) / (math.pi * k_md * _MD * h * T_REF)  # Pa2 per Sm3/s
    b_si = a_si * D_nd * _DAY / _MSCF / (ln_term + s)  # Pa2 per (Sm3/s)2
    return a_si / 1e10 / _DAY, b_si / 1e10 / _DAY**2


def _nondarcy(sg, k_md, h, mu, h_perf) -> float:  # spec: SMP-71
    """
    Economides et al.'s estimate of the non-Darcy coefficient (1994, Eq. 4-34), d/Mscf:
    D = 6e-5 gamma_g k^-0.1 h / (mu_g r_w h_perf^2) in field units.

    :param sg: Gas gravity (air = 1)
    :param k_md: Permeability (mD)
    :param h: Net pay (m)
    :param mu: Gas viscosity (Pa s)
    :param h_perf: Perforated interval (m)
    """
    ft = 0.3048
    return 6e-5 * sg * k_md**-0.1 * (h / ft) / (mu * 1e3 * (_R_W / ft) * (h_perf / ft) ** 2)


def _gas_rate(a, b, dp2) -> float:
    """The gas rate q (Sm3/d) of a q + b q^2 = dp2 (bar2), with b > 0."""
    return (-a + math.sqrt(a * a + 4 * b * dp2)) / (2 * b)
