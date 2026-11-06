"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 06 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The producing gas-oil ratio of a black oil under solution-gas drive, by the material balance, for the sampler
redesign's checkpoint (C1; D-19 and D-44 in plans/sampler-redesign/decisions.md). Below the bubble point the gas that
leaves the oil stays in the rock until its saturation lets it flow; then it flows faster than the oil, and the
producing GOR rises, peaks and falls as the reservoir empties. The script computes that curve, R/R_si against
p_r/p_b, with the model's own black oil, and compares it with the draft's linear factor of 2 to 15.

Method: Craft and Hawkins (1991), Applied petroleum reservoir engineering, 2nd ed. revised by R. E. Terry, Prentice
Hall, Ch. 9, Sect. 3.3, under its assumptions (p. 362): a volumetric reservoir, uniform saturations and pressure, no
gravity segregation, no water influx or production.
- Muskat: the oil saturation's change with pressure, Eq. 9.34 (with the groups X, Y, Z of Eq. 9.35), the producing
  GOR R = R_so + (k_g/k_o)(mu_o B_o)/(mu_g B_g), Eq. 9.33, and the recovery N_p/N = 1 - S_o B_oi / ((1 - S_w) B_o),
  Eq. 9.57. The book steps Eq. 9.36 by hand (Eqs. 9.37, 9.38); here it is integrated by RK4 in pressure.
- Schilthuis: the material balance with the two-phase factor B_t, Eq. 9.41 (as written on pp. 392-393), the liquid
  saturation, Eq. 9.40, and an average GOR over each step, Eqs. 9.43 to 9.46, solved for each step's oil by a root
  finder where the book uses the secant method, Eq. 9.47. This is the book's program (Table 10.1, pp. 399-403).

Checks against the book:
1. Example 9.1 (pp. 370-375): fluid data of Table 9.4, k_g/k_o of Fig. 9.16 (read off the figure: the dots of both
   branches), S_wi = 0.20, N = 56 MM STB, first step from 2500 to 2300 psia. The book gets N_p = 939,500 STB
   (Muskat, S_o = 0.7682), 935,200 (Schilthuis, N_p/N = 0.0167) and 939,300 (Tarner) at 2300 psia.
2. The history match of Ch. 10 (Sect. 3, pp. 394-405): the fluid regressions of Table 10.1, the k_g/k_o segments of
   Table 10.2 (log10 k_g/k_o = A0 + A1 S_g), S_w = 0.37, 5 psi steps from 620 psia; Table 10.3 lists N_p/N and the
   instantaneous GOR down to 425 psia. The listing's low-pressure branch (below 200 psia) inverts BG(1) where BG(I) is
   meant; the printed table does not reach it, and this script uses BG(I).
3. A field, for comparison only: the Gloyd-Mitchell zone of the Rodessa field, a dissolved-gas-drive reservoir
   (Ch. 5, Sect. 5, pp. 163-169, Table 5.5 and Fig. 5.6): average daily GOR and pressure by month, against the model's
   black oil made like its oil (42.8 API, 627 scf/STB, p. 164; T_r and the separator assumed).

The sampler's black oils: the model's PVT (src/manywells/pvt/), as the v2.0.0 configuration and the draft sampler
build it, BlackOil(rho_o, rho_g, gas_law='dak', p_sep, T_sep): Vazquez-Beggs R_s and B_o (PVT-OIL-6, PVT-OIL-8),
Beggs-Robinson live-oil viscosity (PVT-OIL-10, PVT-OIL-11), the DAK gas density (PVT-GAS-11), so B_g = rho_g,sc /
rho_g(p, T), and Lee-Gonzalez-Eakin gas viscosity (PVT-GAS-7), all at the reservoir temperature, with the bubble point
at the drawn saturation pressure (R_si = R_so(p_sat, T_r), D-17). k_g/k_o, which the model does not have:
- the book's three laboratory curves, each with its own connate water: Table 10.2 (S_w = 0.37), Fig. 9.16
  (S_w = 0.20) and the Mile Six Pool's k_o/k_g (Table 9.2, p. 357, S_w = 0.35, gas saturation as a fraction of the
  hydrocarbon pore space; zero at its critical gas saturation of 5%, p. 356, and taken linear from there to the first
  tabulated value at 10%);
- the book's curves blended in log10 k_g/k_o, from Fig. 9.16 (w = 0, the least mobile gas) to Table 10.2 (w = 1,
  the most; Table 9.2 lies close to it): one parameter that spans the book's laboratory curves (own construction);
- a Corey sensitivity in the hydrocarbon pore space, n_g = n_o = 2 (and 3), residual oil 0.3 of it, end point 1 (own
  assumptions, not from the book), whose critical gas saturation spans the book's range, a few percent to 20% of the
  pore space (p. 163), stated at S_w = 0.25.
In the hydrocarbon pore space (s = S_o / (1 - S_w)) Eq. 9.34 holds with S_w = 0, so S_w enters only through the
k_g/k_o curve. The script reports R/R_si against p/p_b and against the recovery, the peak, a three-parameter family
fitted to the curves, the drawn states below the saturation pressure against the draft's factor, and a proposed draw
(w ~ U(0, 1) per well, the factor from the material balance at the state's p_r/p_sat).

    .venv/bin/python plans/evidence/solution_gas_drive.py                     # book checks and a grid of black oils
    .venv/bin/python plans/evidence/solution_gas_drive.py --draws DRAWS.csv   # and the R6 pilot's drawn black oils

DRAWS.csv is the R6 pilot's synthetic draws (columns well, tail, fluid_class, drive, stage, rho_o, rho_g, p_sep,
T_sep, T_r, p_sat, R_i, p_r, R_go), kept outside the repo.
"""

import argparse

import casadi as ca
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares

from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.units import CF_PSI, CF_RS

PSI_PER_BAR = 1e5 / CF_PSI

# ----------------------------------------------------------------------------------------------------------------
# The book's data (Craft and Hawkins 1991)
# ----------------------------------------------------------------------------------------------------------------

# Table 9.4, p. 370: p (psia), B_o (bbl/STB), R_so (scf/STB), B_g (bbl/scf), mu_o (cp), mu_g (cp)
TABLE_9_4 = np.array(
    [
        [2500, 1.498, 721, 0.001048, 0.488, 0.0170],
        [2300, 1.463, 669, 0.001155, 0.539, 0.0166],
        [2100, 1.429, 617, 0.001280, 0.595, 0.0162],
        [1900, 1.395, 565, 0.001440, 0.658, 0.0158],
        [1700, 1.361, 513, 0.001634, 0.726, 0.0154],
        [1500, 1.327, 461, 0.001884, 0.802, 0.0150],
        [1300, 1.292, 409, 0.002206, 0.887, 0.0146],
        [1100, 1.258, 357, 0.002654, 0.981, 0.0142],
        [900, 1.224, 305, 0.003300, 1.085, 0.0138],
        [700, 1.190, 253, 0.004315, 1.199, 0.0134],
        [500, 1.156, 201, 0.006163, 1.324, 0.0130],
        [300, 1.121, 149, 0.010469, 1.464, 0.0126],
        [100, 1.087, 97, 0.032032, 1.617, 0.0122],
    ]
)
# Fig. 9.16, p. 373: k_g/k_o against the total liquid saturation S_L, the dots of both branches read off the figure
# (the right-hand branch on the left scale, the left-hand branch on the right scale, 100 times it)
FIG_9_16 = np.array(
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
EX_9_1 = dict(N=56e6, S_wi=0.20, p_i=2500.0, p_1=2300.0)  # Example 9.1, p. 370
EX_9_1_BOOK = dict(muskat_So=0.7682, muskat_Np=939_500, schilthuis_NpN=0.0167, schilthuis_Np=935_200, tarner_Np=939_300)
EX_9_1_BOOK_SLOPES = dict(X=(0.000182, 0.000205), Y=(0.003277, 0.003795), Z=(0.000454, 0.000500))  # p. 371

# Table 9.2, p. 357, Mile Six Pool: gas saturation (fraction of the hydrocarbon pore space) and k_o/k_g
TABLE_9_2 = np.array(
    [
        [0.05, np.inf],
        [0.10, 38],
        [0.15, 8.80],
        [0.20, 3.10],
        [0.25, 1.40],
        [0.30, 0.72],
        [0.35, 0.364],
        [0.40, 0.210],
        [0.45, 0.118],
        [0.50, 0.072],
        [0.55, 0.024],
    ]
)

# Table 10.2, p. 404: upper gas saturation of each segment, A0, A1 in log10(k_g/k_o) = A0 + A1 S_g
TABLE_10_2 = np.array(
    [[0.0430, -4.7981, 54.6520], [0.0810, -3.8450, 32.0170], [0.1090, -2.6308, 16.9890], [0.3500, -2.1119, 12.2410]]
)
CH10 = dict(p_i=620.0, dp=5.0, S_w=0.37)  # Table 10.1, p. 399
# Table 10.3, p. 405: p (psia), N_p/N, instantaneous GOR (scf/STB)
TABLE_10_3 = np.array(
    [
        [620, 0.000000, 198],
        [615, 0.003574, 198],
        [605, 0.010883, 196],
        [595, 0.018410, 195],
        [585, 0.026131, 196],
        [575, 0.033988, 200],
        [565, 0.041858, 209],
        [555, 0.049524, 228],
        [545, 0.056676, 264],
        [535, 0.063061, 311],
        [525, 0.068856, 353],
        [515, 0.074112, 405],
        [505, 0.078835, 465],
        [495, 0.083060, 535],
        [485, 0.086836, 611],
        [475, 0.090217, 695],
        [465, 0.093254, 786],
        [455, 0.095994, 882],
        [445, 0.098478, 983],
        [435, 0.100740, 1089],
        [425, 0.102811, 1200],
    ]
)

# Table 5.5, pp. 166-167, Gloyd-Mitchell zone: average pressure (psig) and average daily GOR (scf/STB) in the months
# with a measured or assumed pressure; initial GOR 625 scf/STB at 2700 psig (p. 164), bubble point near 2200 psig
# (Fig. 5.6, p. 168)
GLOYD_MITCHELL = np.array(
    [
        [2700, 625],
        [2490, 1000],
        [2280, 1200],
        [2070, 1700],
        [1860, 1900],
        [1650, 3200],
        [1250, 5300],
        [1115, 5900],
        [1000, 6800],
        [900, 7500],
        [825, 7600],
        [740, 7700],
        [725, 7800],
        [565, 7500],
        [500, 7300],
        [405, 6300],
        [310, 5700],
        [280, 4900],
        [225, 3600],
    ]
)
GLOYD_MITCHELL_PB = 2200.0  # psig, Fig. 5.6


# ----------------------------------------------------------------------------------------------------------------
# k_g/k_o as a function of the gas saturation in the hydrocarbon pore space, sg = S_g / (1 - S_w)
# ----------------------------------------------------------------------------------------------------------------


def kgko_table_10_2(sg, S_w=CH10['S_w']):
    """Table 10.2's segments (p. 404), at S_g = sg (1 - S_w); the last segment beyond its end."""
    S_g = np.maximum(np.asarray(sg, dtype=float) * (1 - S_w), 0.0)
    idx = np.minimum(np.searchsorted(TABLE_10_2[:, 0], S_g, side='left'), len(TABLE_10_2) - 1)
    return 10.0 ** (TABLE_10_2[idx, 1] + TABLE_10_2[idx, 2] * S_g)


def kgko_fig_9_16(sg, S_w=EX_9_1['S_wi']):
    """Fig. 9.16 (p. 373), log-linear between the dots and beyond the end ones, at S_L = S_w + (1 - S_w)(1 - sg)."""
    S_L = S_w + (1 - S_w) * (1 - np.asarray(sg, dtype=float))
    x, y = FIG_9_16[:, 0], np.log10(FIG_9_16[:, 1])
    out = np.interp(S_L, x, y)
    lo, hi = S_L < x[0], S_L > x[-1]
    out = np.where(lo, y[0] + (S_L - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), out)
    out = np.where(hi, y[-1] + (S_L - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), out)
    return 10.0**out


def kgko_table_9_2(sg):
    """The Mile Six Pool (Table 9.2, p. 357): zero up to the critical gas saturation 0.05 (p. 356), linear to the
    value at 0.10, log-linear between the tabulated points and beyond 0.55."""
    sg = np.asarray(sg, dtype=float)
    x, y = TABLE_9_2[1:, 0], np.log10(1 / TABLE_9_2[1:, 1])
    out = 10.0 ** np.interp(sg, x, y, right=np.nan)
    beyond = y[-1] + (sg - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    out = np.where(sg > x[-1], 10.0**beyond, out)
    ramp = np.clip((sg - 0.05) / 0.05, 0.0, 1.0) * 10.0 ** y[0]
    return np.where(sg < x[0], ramp, out)


def kgko_corey(sgc, n_g=2.0, n_o=2.0, s_org=0.3, k_rg0=1.0):
    """
    A Corey form in the hydrocarbon pore space (own assumption): k_rg = k_rg0 ((sg - sgc)/(1 - sgc - s_org))^n_g,
    k_ro = ((1 - s_org - sg)/(1 - s_org))^n_o, both relative to the oil's end point.
    """

    def f(sg):
        sg = np.asarray(sg, dtype=float)
        krg = k_rg0 * np.clip((sg - sgc) / (1 - sgc - s_org), 0.0, None) ** n_g
        kro = np.clip((1 - s_org - sg) / (1 - s_org), 1e-9, None) ** n_o
        return krg / kro

    return f


KGKO = {
    'C&H Table 10.2': kgko_table_10_2,
    'C&H Fig. 9.16': kgko_fig_9_16,
    'C&H Table 9.2': kgko_table_9_2,
}


# ----------------------------------------------------------------------------------------------------------------
# Muskat's material balance (Eq. 9.34), RK4 in pressure, vectorised over wells
# ----------------------------------------------------------------------------------------------------------------


def muskat(p, Rs, Bo, Bg, mu_r, dRs, dBo, dBg, kgko):
    """
    Integrate Eq. 9.34 in the hydrocarbon pore space from the bubble point, s = 1, down a pressure grid.

    Every argument but kgko is an array (n_wells, 2 n + 1) on the grid p (any pressure unit, consistent with the
    derivatives), falling from the bubble point; the odd points are RK4's midpoints. mu_r = mu_o/mu_g. Returns s, R
    (in R_s's unit) and N_p/N_b (N_b: the oil in place at the bubble point) at the even points.
    """
    X, Y, Z = Bg / Bo * dRs, dBo / Bo, dBg / Bg  # Eq. 9.35

    def f(i, s):
        km = kgko(1 - s) * mu_r[:, i]
        return (s * X[:, i] + s * km * Y[:, i] - (1 - s) * Z[:, i]) / (1 + km)

    n = (p.shape[1] - 1) // 2
    s = np.ones(p.shape[0])
    out = np.empty((p.shape[0], n + 1))
    out[:, 0] = s
    for k in range(n):
        i = 2 * k
        h = p[:, i + 2] - p[:, i]
        k1 = f(i, s)
        k2 = f(i + 1, s + h / 2 * k1)
        k3 = f(i + 1, s + h / 2 * k2)
        k4 = f(i + 2, s + h * k3)
        s = np.clip(s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
        out[:, k + 1] = s
    e = slice(0, None, 2)
    R = Rs[:, e] + kgko(1 - out) * mu_r[:, e] * Bo[:, e] / Bg[:, e]  # Eq. 9.33
    rec = 1 - out * Bo[:, :1] / Bo[:, e]  # Eq. 9.57 in the hydrocarbon pore space
    return out, R, rec


# ----------------------------------------------------------------------------------------------------------------
# Schilthuis' material balance (Eqs. 9.40, 9.41, 9.43 to 9.46), the book's program (Table 10.1)
# ----------------------------------------------------------------------------------------------------------------


def schilthuis(p, props, kgko_pore, S_w):
    """
    N_p/N and R at each pressure of the falling grid p, starting saturated at p[0]. props(p) gives
    (B_o, R_so, B_g, mu_o, mu_g) in consistent units; kgko_pore takes the gas saturation of the pore space.
    """
    Bo_i, Rs_i = props(p[0])[:2]
    NpN, GpN, R_prev = 0.0, 0.0, Rs_i
    rows = [(p[0], 0.0, Rs_i)]
    for pj in p[1:]:
        Bo, Rs, Bg, mu_o, mu_g = props(pj)
        Bt, Bt_i = Bo + (Rs_i - Rs) * Bg, Bo_i

        def gor(npn):
            S_g = max((1 - (1 - npn) * Bo / Bo_i) * (1 - S_w), 0.0)  # Eq. 9.40
            return Rs + kgko_pore(S_g) * mu_o * Bo / (mu_g * Bg)  # Eq. 9.33

        def mb(npn):
            dnp = npn - NpN
            Rp = (GpN + dnp * (R_prev + gor(npn)) / 2) / npn  # Eqs. 9.43 to 9.46
            return npn * (Bt + Bg * (Rp - Rs_i)) / (Bt - Bt_i) - 1  # Eq. 9.41

        npn = brentq(mb, NpN + 1e-12, 0.999, xtol=1e-12)
        R = gor(npn)
        GpN += (npn - NpN) * (R_prev + R) / 2
        NpN, R_prev = npn, R
        rows.append((pj, npn, R))
    return np.array(rows)


# ----------------------------------------------------------------------------------------------------------------
# Check 1: Example 9.1
# ----------------------------------------------------------------------------------------------------------------


def table_9_4_props():
    """Smooth interpolants of Table 9.4 (1/B_g interpolated, as the book advises, p. 364)."""
    t = TABLE_9_4[::-1]
    p = t[:, 0]
    f = {k: PchipInterpolator(p, t[:, j]) for k, j in (('Bo', 1), ('Rs', 2), ('mu_o', 4), ('mu_g', 5))}
    f['inv_Bg'] = PchipInterpolator(p, 1 / t[:, 3])
    return f


def example_9_1():
    print('Example 9.1 (Craft and Hawkins 1991, pp. 370-375): first step, 2500 to 2300 psia')
    N, S_w = EX_9_1['N'], EX_9_1['S_wi']
    book = EX_9_1_BOOK
    # Muskat with the book's own groups X, Y, Z (p. 371) and its k_g/k_o = 0.00001 (S_L above the curve)
    X, Y, Z = EX_9_1_BOOK_SLOPES['X'], EX_9_1_BOOK_SLOPES['Y'], EX_9_1_BOOK_SLOPES['Z']
    dp = EX_9_1['p_i'] - EX_9_1['p_1']
    So0 = 1 - S_w
    d1 = So0 * X[0]
    So_est = So0 - dp * d1  # Eq. 9.38
    km = 1e-5 * 0.539 / 0.0166  # k_g/k_o = 0.00001 off the curve, and mu_o/mu_g at 2300 psia (p. 371)
    d2 = (So_est * X[1] + So_est * km * Y[1] + (1 - S_w - So_est) * Z[1]) / (1 + km)  # Eq. 9.36, Z as on p. 371
    So1 = So0 - dp * (d1 + d2) / 2
    Bo_i, Bo_1 = TABLE_9_4[0, 1], TABLE_9_4[1, 1]
    Np = N * (1 - So1 / (1 - S_w) * Bo_i / Bo_1)  # Eq. 9.57
    print(
        f"  Muskat, the book's X, Y, Z: S_o {So1:.5f} (book {book['muskat_So']}), N_p {Np:,.0f} STB (book {book['muskat_Np']:,})"
    )
    So_r = So0 - dp * 0.000159  # the book's average of its rounded slopes 0.000146 and 0.000173 (p. 371)
    Np_r = N * (1 - So_r / (1 - S_w) * Bo_i / Bo_1)
    print(f"    with the book's rounded average slope 0.000159: S_o {So_r:.4f}, N_p {Np_r:,.0f} STB")

    f = table_9_4_props()

    def props(pp):
        return (
            float(f['Bo'](pp)),
            float(f['Rs'](pp)),
            1 / float(f['inv_Bg'](pp)),
            float(f['mu_o'](pp)),
            float(f['mu_g'](pp)),
        )

    # Schilthuis with the single-phase form (Eq. 9.41 as on p. 367) is the same balance as with B_t
    sch = schilthuis(np.array([2500.0, 2300.0]), props, lambda S_g: kgko_fig_9_16(S_g / (1 - S_w)), S_w)
    print(
        f'  Schilthuis: N_p/N {sch[1, 1]:.4f} (book {book["schilthuis_NpN"]}), N_p {N * sch[1, 1]:,.0f} STB '
        f'(book {book["schilthuis_Np"]:,}), R {sch[1, 2]:.1f} scf/STB (book 669.4)'
    )
    # Tarner (Eqs. 9.49, 9.50, 9.56) with R = R_so at 2300 psia, as the book's iteration converges
    Bo, Rs, Bg = 1.463, 669.0, 0.001155
    den = Bo - Bo_i + (721 - Rs) * Bg
    phi_n, phi_g = (Bo - Rs * Bg) / den, Bg / den
    R_ave = (721 + 669.4) / 2
    Np_t = N / (phi_n + phi_g * R_ave)
    print(
        f'  Tarner: Phi_n {phi_n:.3f} (book 27.546), Phi_g {phi_g:.5f} (book 0.04609), N_p {Np_t:,.0f} STB (book {book["tarner_Np"]:,})'
    )

    # The whole depletion by Muskat (RK4) with Table 9.4 and Fig. 9.16; the book stops at 2300 psia
    p = np.linspace(2500.0, 100.0, 2 * 960 + 1)[None, :]
    a = {k: f[k](p) for k in ('Bo', 'Rs', 'mu_o', 'mu_g')}
    a['Bg'] = 1 / f['inv_Bg'](p)
    d = {k: f[k].derivative()(p) for k in ('Bo', 'Rs')}
    dBg = -f['inv_Bg'].derivative()(p) * a['Bg'] ** 2
    s, R, rec = muskat(p, a['Rs'], a['Bo'], a['Bg'], a['mu_o'] / a['mu_g'], d['Rs'], d['Bo'], dBg, kgko_fig_9_16)
    pe = p[0, ::2]
    k1 = np.argmin(abs(pe - 2300))
    print(f'  Muskat RK4 on Table 9.4: N_p at 2300 psia {N * rec[0, k1]:,.0f} STB, S_o {s[0, k1] * (1 - S_w):.4f}')
    print('  Muskat RK4 to 100 psia (own extension):  p psia, N_p/N, R scf/STB, R/R_si')
    for pp in (2300, 2100, 1900, 1700, 1500, 1300, 1100, 900, 700, 500, 300, 100):
        k = np.argmin(abs(pe - pp))
        print(f'    {pe[k]:6.0f}  {rec[0, k]:.4f}  {R[0, k]:7.0f}  {R[0, k] / 721:5.2f}')
    kp = np.argmax(R[0])
    print(
        f'    peak R {R[0, kp]:.0f} scf/STB ({R[0, kp] / 721:.2f} R_si) at {pe[kp]:.0f} psia (p/p_b {pe[kp] / 2500:.2f}), N_p/N {rec[0, kp]:.3f}'
    )


# ----------------------------------------------------------------------------------------------------------------
# Check 2: the history match of Ch. 10
# ----------------------------------------------------------------------------------------------------------------


def table_10_1_props(p):
    """The fluid regressions of Table 10.1 (pp. 400-401), p in psia: B_o, R_so, B_g (bbl/scf), mu_o, mu_g (cp)."""
    p = np.asarray(p, dtype=float)
    hi = p >= 200.0
    mu_o = np.where(hi, 2.354 - 0.001134 * p, 2.670 - 0.004566 * p + 0.9461e-5 * p**2)
    mu_g = np.where(hi, 0.009684 + 0.3267e-5 * p + 0.1116e-9 * p**2, 0.007580 + 0.2497e-4 * p - 0.5622e-7 * p**2)
    Rs = np.where(hi, 52.45 + 0.2475 * p - 0.2097e-4 * p**2, 16.99 + 0.6565 * p - 0.001198 * p**2)
    Bo = np.where(hi, 1.062 + 0.9257e-4 * p, 1.033 + 0.3829e-3 * p - 0.7262e-6 * p**2)
    Bg = 1 / np.where(hi, 7.446 + 0.3625 * p + 0.4532e-4 * p**2, 8.874 + 0.3676 * p)
    return Bo, Rs, Bg, mu_o, mu_g


def history_match_ch10():
    print('\nCh. 10 history match (Craft and Hawkins 1991, pp. 394-405): Table 10.3 against this script')
    S_w = CH10['S_w']
    p_grid = np.arange(CH10['p_i'], 320.0 - 1e-9, -CH10['dp'])  # the program's 60 pressures

    def props(pp):
        return tuple(float(v) for v in table_10_1_props(pp))

    def kgko_pore(S_g):
        return float(kgko_table_10_2(S_g / (1 - S_w)))

    sch = schilthuis(p_grid, props, kgko_pore, S_w)
    # Muskat on the same regressions, by RK4 in steps of 0.25 psi, on to 60 psia
    p = np.linspace(620.0, 60.0, 2 * 2240 + 1)[None, :]
    h = 1e-3
    Bo, Rs, Bg, mu_o, mu_g = table_10_1_props(p)
    dv = [(a - b) / (2 * h) for a, b in zip(table_10_1_props(p + h), table_10_1_props(p - h))]
    s, R, rec = muskat(p, Rs, Bo, Bg, mu_o / mu_g, dv[1], dv[0], dv[2], kgko_table_10_2)
    pe = p[0, ::2]
    print('     p   N_p/N: book  Schilthuis  Muskat    R: book  Schilthuis  Muskat (scf/STB)')
    worst = [0.0, 0.0]
    for pp, npn_b, r_b in TABLE_10_3:
        ks, km = np.argmin(abs(sch[:, 0] - pp)), np.argmin(abs(pe - pp))
        print(
            f'  {pp:4.0f}   {npn_b:.6f}  {sch[ks, 1]:.6f}  {rec[0, km]:.6f}     {r_b:5.0f}  {sch[ks, 2]:8.1f}  {R[0, km]:8.1f}'
        )
        worst[0] = max(worst[0], abs(sch[ks, 1] - npn_b))
        worst[1] = max(worst[1], abs(sch[ks, 2] - r_b) / r_b)
    print(f'  Schilthuis against Table 10.3: largest difference {worst[0]:.6f} in N_p/N, {100 * worst[1]:.2f}% in R')
    Rsi = Rs[0, 0]
    print(f'  Muskat on to 60 psia (own extension; R_si {Rsi:.0f} scf/STB):  p/p_b, N_p/N_b, R/R_si')
    for x in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = np.argmin(abs(pe - x * 620))
        print(f'    {pe[k] / 620:.2f}  {rec[0, k]:.3f}  {R[0, k] / Rsi:5.2f}')
    kp = np.argmax(R[0])
    print(f'    peak {R[0, kp] / Rsi:.2f} R_si at p/p_b {pe[kp] / 620:.2f}, N_p/N {rec[0, kp]:.3f}')


def gloyd_mitchell():
    print('\nGloyd-Mitchell zone (Craft and Hawkins 1991, Table 5.5, pp. 166-167), field, for comparison only')
    p_b = GLOYD_MITCHELL_PB + 14.7
    R_si = GLOYD_MITCHELL[0, 1]
    print('  p/p_b (p_b ~2200 psig, Fig. 5.6), GOR/R_si (R_si 625 scf/STB, p. 164)')
    print('  ' + ', '.join(f'{(p + 14.7) / p_b:.2f}: {r / R_si:.1f}' for p, r in GLOYD_MITCHELL))


# ----------------------------------------------------------------------------------------------------------------
# The model's black oils
# ----------------------------------------------------------------------------------------------------------------


def kgko_blend(w):
    """The book's laboratory curves blended in log10 k_g/k_o, from Fig. 9.16 (w = 0, the least mobile gas) to
    Table 10.2 (w = 1, the most); w may be an array with one value per well."""
    w = np.asarray(w, dtype=float)

    def f(sg):
        sg = np.asarray(sg, dtype=float)
        ww = w[:, None] if (w.ndim == 1 and sg.ndim == 2) else w
        return 10.0 ** ((1 - ww) * np.log10(kgko_fig_9_16(sg)) + ww * np.log10(kgko_table_10_2(sg)))

    return f


S_W_COREY = 0.25  # the connate water at which the Corey sensitivity's critical gas saturations are stated
COREY = {
    f'Corey S_gc {sgc:.2f}' + ('' if n == 2 else f', n {n}'): kgko_corey(sgc / (1 - S_W_COREY), n_g=n, n_o=n)
    for sgc, n in ((0.02, 2), (0.05, 2), (0.10, 2), (0.20, 2), (0.05, 3))
}
CURVES = dict(KGKO, **{'C&H blend w 0.5': kgko_blend(0.5)}, **COREY)
X_GRID = np.linspace(1.0, 0.05, 2 * 380 + 1)  # p/p_b, with RK4's midpoints
X_REPORT = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
REC_REPORT = (0.02, 0.05, 0.10, 0.15, 0.20)


def black_oil_props(wells, x=X_GRID, rel=1e-4):
    """The model's black oil of each well on the grid p = x p_sat at T_r (bar, m3/Sm3, Sm3/Sm3), with central
    differences in pressure."""
    W, M = len(wells), len(x)
    keys = ('p', 'Rs', 'Bo', 'Bg', 'mu_r', 'dRs', 'dBo', 'dBg')
    out = {k: np.empty((W, M)) for k in keys}
    for j, w in enumerate(wells.itertuples()):
        oil = BlackOil(rho_o=w.rho_o, rho_g=w.rho_g, gas_law='dak', p_sep=w.p_sep, T_sep=w.T_sep)
        p = x * w.p_sat
        P = ca.DM(np.concatenate([p, p * (1 + rel), p * (1 - rel)]))

        def arr(v):
            return np.array(v).ravel()

        rs, bo, rho = arr(oil.rs(P, w.T_r)), arr(oil.bo(P, w.T_r)), arr(oil.gas_density(P, w.T_r))
        bg = oil.rho_g / rho  # B_g: the gas density at standard conditions over the reservoir's (PVT-GAS-11)
        mu_o = arr(oil.oil_viscosity(ca.DM(p), w.T_r))
        mu_g = arr(oil.gas_viscosity(ca.DM(p), w.T_r, ca.DM(rho[:M])))
        dp = 2 * rel * p
        for k, v in (('Rs', rs), ('Bo', bo), ('Bg', bg)):
            out[k][j] = v[:M]
            out['d' + k][j] = (v[M : 2 * M] - v[2 * M :]) / dp
        out['p'][j], out['mu_r'][j] = p, mu_o / mu_g
    return out


def run_curves(props, curves):
    """R/R_si and N_p/N_b on X_GRID's even points, for each k_g/k_o curve."""
    a = props
    res = {}
    for name, kgko in curves.items():
        _, R, rec = muskat(a['p'], a['Rs'], a['Bo'], a['Bg'], a['mu_r'], a['dRs'], a['dBo'], a['dBg'], kgko)
        res[name] = (R / a['Rs'][:, :1], rec)
    return res


def grid_wells():
    """Black oils over the draft's ranges (sampling-draft.md, Oil and gas at standard conditions; Saturation
    pressure; Geothermal gradient): API 25, 33, 41; gas gravity 0.60, 0.67, 0.76; T_r 320, 358, 420 K; p_b 100, 250,
    450 bar; separator 33 bar and 312 K; only those with R_si up to 356 Sm3/Sm3 (D-17)."""
    rows = []
    for api in (25.0, 33.0, 41.0):
        for sg in (0.60, 0.67, 0.76):
            for T_r in (320.0, 358.0, 420.0):
                for p_b in (100.0, 250.0, 450.0):
                    w = dict(rho_o=density_from_api(api), rho_g=gas_density_from_sg(sg), p_sep=33.0, T_sep=312.0)
                    oil = BlackOil(gas_law='dak', **w)
                    R_i = float(oil.rs(p_b, T_r))
                    if R_i <= 356.0:
                        rows.append(dict(w, api=api, sg_gas=sg, T_r=T_r, p_sat=p_b, R_i=R_i, tail=False))
    return pd.DataFrame(rows)


def pct(v, q=(5, 50, 95), fmt='.2f'):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return '/'.join(f'{x:{fmt}}' for x in np.percentile(v, q)) if len(v) else '-'


def power_family(F, c, x_lo=0.25):
    """
    Fit each well's curve by R/R_si = x^c + k (x_on - x)^a below x_on (x = p/p_b; x^c is Vazquez-Beggs' R_so/R_si,
    c = 1.0937 or 1.187 by API gravity, PVT-OIL-6), in log over x from 1 down to x_lo or the peak. The draft's form is
    x_on = 1, a = 1. Returns x_on, k, a and the largest relative error of the fit over that range.
    """
    xe = X_GRID[::2]
    out = np.full((len(F), 4), np.nan)
    for j, f in enumerate(F):
        sel = xe >= max(x_lo, xe[np.argmax(f)])
        x, y, xc = xe[sel], f[sel], xe[sel] ** c[j]

        def r(q):
            return np.log((xc + np.exp(q[1]) * np.clip(q[0] - x, 0.0, None) ** q[2]) / y)

        q = least_squares(r, [0.85, np.log(50.0), 1.7], bounds=([0.3, -5.0, 0.5], [1.0, 10.0, 4.0])).x
        out[j] = q[0], np.exp(q[1]), q[2], np.max(abs(np.exp(r(q)) - 1))
    return out


def report_curves(res, label, api):
    xe = X_GRID[::2]
    print(f'\n{label}: R/R_si against p/p_b, P5/P50/P95 over the wells')
    print('  ' + 'k_g/k_o'.ljust(22) + ''.join(f'{x:>19.2f}' for x in X_REPORT))
    for name, (F, rec) in res.items():
        cols = ''.join(f'{pct(F[:, np.argmin(abs(xe - x))], fmt=".3g"):>19}' for x in X_REPORT)
        print('  ' + name.ljust(22) + cols)
    print(f'\n{label}: R/R_si against the recovery N_p/N_b (N_b: oil in place at the bubble point), P5/P50/P95')
    print('  ' + 'k_g/k_o'.ljust(22) + ''.join(f'{r:>19.2f}' for r in REC_REPORT))
    for name, (F, rec) in res.items():
        cols = []
        for r in REC_REPORT:
            # the rising branch: the first point where the recovery reaches r
            v = np.array([f[np.argmax(q >= r)] if (q >= r).any() else np.nan for f, q in zip(F, rec)])
            cols.append(f'{pct(v, fmt=".3g"):>19}')
        print('  ' + name.ljust(22) + ''.join(cols))
    print(f'\n{label}: the peak, where it lies and its recovery (P5/P50/P95)')
    hdr = ('peak R/R_si', 'p/p_b at peak', 'N_p/N_b at peak')
    print('  ' + 'k_g/k_o'.ljust(22) + ''.join(f'{h:>21}' for h in hdr))
    for name, (F, rec) in res.items():
        kp = np.argmax(F, axis=1)
        rows = np.arange(len(F))
        print('  ' + name.ljust(22) + ''.join(f'{pct(v, fmt=".3g"):>21}' for v in (F[rows, kp], xe[kp], rec[rows, kp])))
    print(
        f'\n{label}: the family R/R_si = x^c + k (x_on - x)^a fitted to each curve (the draft: x_on 1, a 1, k 2 to 15)'
    )
    hdr = ('x_on', 'k', 'a', 'largest error')
    print('  ' + 'k_g/k_o'.ljust(22) + ''.join(f'{h:>21}' for h in hdr))
    c = np.where(api > 30.0, 1.187, 1.0937)
    for name, (F, rec) in res.items():
        fit = power_family(F, c)
        print('  ' + name.ljust(22) + ''.join(f'{pct(fit[:, i], fmt=".3g"):>21}' for i in range(4)))


def compare_draft(wells, res):
    """The draft's factor R_go/R_si at each drawn state below the saturation pressure against the material balance's
    at the same p_r/p_sat: P50/P95, the share above 2 and 5 times R_si, and the share past the peak."""
    xe = X_GRID[::2][::-1]
    x = (wells['p_r'] / wells['p_sat']).to_numpy()
    print('\nDrawn states below the saturation pressure: R_go/R_si by the draft and by the material balance')
    print("  (P50/P95; share above 2 R_si; share above 5 R_si; share of states past the curve's peak)")
    for (drive, stage), g in wells[x < 1.0].groupby(['drive', 'stage']):
        pos = wells.index.get_indexer(g.index)
        draft = (g['R_go'] / g['R_i']).to_numpy()
        print(f'  {drive}, {stage}: {len(g)} states, p_r/p_sat {pct(x[pos])} (P5/P50/P95)')
        print(f'      {"draft":22s} {pct(draft, (50, 95)):>12}  {np.mean(draft > 2):.2f}  {np.mean(draft > 5):.2f}')
        for name, (F, _) in res.items():
            mb = np.array([np.interp(x[i], xe, F[i][::-1]) for i in pos])
            past = np.mean(x[pos] < X_GRID[::2][np.argmax(F[pos], axis=1)])
            print(f'      {name:22s} {pct(mb, (50, 95)):>12}  {np.mean(mb > 2):.2f}  {np.mean(mb > 5):.2f}  {past:.2f}')


def book_oil_with_curves():
    """Example 9.1's oil (Table 9.4) with every k_g/k_o: the curve, not the PVT, sets most of the spread."""
    f = table_9_4_props()
    p = np.linspace(2500.0, 100.0, 2 * 960 + 1)[None, :]
    a = {k: f[k](p) for k in ('Bo', 'Rs', 'mu_o', 'mu_g')}
    a['Bg'] = 1 / f['inv_Bg'](p)
    dRs, dBo = f['Rs'].derivative()(p), f['Bo'].derivative()(p)
    dBg = -f['inv_Bg'].derivative()(p) * a['Bg'] ** 2
    xe = p[0, ::2] / 2500
    print("\nExample 9.1's oil (Table 9.4) with each k_g/k_o: R/R_si at p/p_b 0.7, 0.5, 0.3; the peak and where")
    for name, kgko in CURVES.items():
        _, R, rec = muskat(p, a['Rs'], a['Bo'], a['Bg'], a['mu_o'] / a['mu_g'], dRs, dBo, dBg, kgko)
        F = R[0] / 721
        k = np.argmax(F)
        vals = ' '.join(f'{F[np.argmin(abs(xe - x))]:6.2f}' for x in (0.7, 0.5, 0.3))
        print(f'  {name:22s} {vals}   peak {F[k]:6.2f} at {xe[k]:.2f}, N_p/N {rec[0, k]:.3f}')


def gloyd_mitchell_like(T_r=360.0):
    """
    The model's black oil made like the Gloyd-Mitchell zone's (42.8 API and 627 scf/STB at 2700 psig, p. 164; bubble
    point near 2200 psig, Fig. 5.6), with each k_g/k_o, against the field's GOR (Table 5.5). Assumed, as the book
    gives neither: T_r = 360 K and a separator at 114.7 psia and 300 K (where Vazquez-Beggs' gas-gravity correction
    vanishes); the gas gravity is then the one at which R_so(p_b, T_r) = 627 scf/STB.
    """
    p_b = (GLOYD_MITCHELL_PB + 14.7) / PSI_PER_BAR
    R_si = 627 * CF_RS
    base = dict(rho_o=density_from_api(42.8), p_sep=114.7 / PSI_PER_BAR, T_sep=300.0)

    def miss(sg):
        return float(BlackOil(rho_g=gas_density_from_sg(sg), gas_law='dak', **base).rs(p_b, T_r)) - R_si

    sg = brentq(miss, 0.56, 1.5)
    well = pd.DataFrame([dict(base, rho_g=gas_density_from_sg(sg), T_r=T_r, p_sat=p_b)])
    res = run_curves(black_oil_props(well), CURVES)
    xe = X_GRID[::2]
    field = [((p + 14.7) / (GLOYD_MITCHELL_PB + 14.7), r / GLOYD_MITCHELL[0, 1]) for p, r in GLOYD_MITCHELL]
    field = [(x, f) for x, f in field if 0.15 < x < 0.8]
    print(f"\nA black oil like the Gloyd-Mitchell zone's (gas gravity {sg:.3f}, T_r {T_r:.0f} K assumed): R/R_si")
    print('  ' + 'p/p_b'.ljust(22) + ''.join(f'{x:>7.2f}' for x, _ in field))
    print('  ' + 'field, Table 5.5'.ljust(22) + ''.join(f'{f:>7.1f}' for _, f in field))
    for name, (F, _) in res.items():
        print('  ' + name.ljust(22) + ''.join(f'{np.interp(x, xe[::-1], F[0][::-1]):>7.1f}' for x, _ in field))


def recommended(wells, props, seed=20261106):
    """The draw proposed for D-19/D-44: per well, w ~ U(0, 1) in the book's blend, the factor from the material
    balance at the state's p_r/p_sat; R_si wherever p_r >= p_sat."""
    w = np.random.default_rng(seed).uniform(0.0, 1.0, len(wells))
    F, rec = run_curves(props, {'w': kgko_blend(w)})['w']
    xe = X_GRID[::2][::-1]
    x = (wells['p_r'] / wells['p_sat']).to_numpy()
    mb = np.array([np.interp(min(xi, 1.0), xe, f[::-1]) if xi < 1 else 1.0 for xi, f in zip(x, F)])
    draft = (wells['R_go'] / wells['R_i']).to_numpy()
    print('\nProposed draw (w ~ U(0, 1) in the book blend) against the draft, all drawn core black oils, R_go/R_si')
    print('  (P50/P90/P95/P99; share above 2, 5 and 10 R_si)')
    for (drive, stage), g in wells.groupby(['drive', 'stage']):
        pos = wells.index.get_indexer(g.index)
        for lab, v in (('draft', draft[pos]), ('material balance', mb[pos])):
            shares = '  '.join(f'{np.mean(v > t):.2f}' for t in (2, 5, 10))
            print(f'  {drive:16s} {stage:8s} {lab:16s} {pct(v, (50, 90, 95, 99)):>24}  {shares}')
    for lab, v in (('draft', draft), ('material balance', mb)):
        shares = '  '.join(f'{np.mean(v > t):.2f}' for t in (2, 5, 10))
        print(f'  {"all":25s} {lab:16s} {pct(v, (50, 90, 95, 99)):>24}  {shares}')
    R_i = wells['R_i'].to_numpy()
    print(
        f"  R_go above 1000 Sm3/Sm3 (the end of Hasan et al.'s data, sampling-draft.md): draft {np.mean(draft * R_i > 1000):.3f}, "
        f'material balance {np.mean(mb * R_i > 1000):.3f} of the drawn core black oils'
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('--draws', help="the R6 pilot's synthetic draws (CSV); without it, a grid of black oils")
    ap.add_argument('--no-book', action='store_true', help='skip the checks against the book')
    args = ap.parse_args()
    if not args.no_book:
        example_9_1()
        history_match_ch10()
        gloyd_mitchell()
        book_oil_with_curves()
        gloyd_mitchell_like()
    wells = grid_wells()
    report_curves(
        run_curves(black_oil_props(wells), CURVES), f'Grid of {len(wells)} black oils', wells['api'].to_numpy()
    )
    if args.draws:
        d = pd.read_csv(args.draws)
        d = d[(d['fluid_class'] == 'black_oil') & ~d['tail'].astype(bool)].reset_index(drop=True)
        props = black_oil_props(d)
        err = np.max(abs(props['Rs'][:, 0] / d['R_i'].to_numpy() - 1))
        print(
            f"\nDrawn core black oils: {len(d)}; R_so(p_sat, T_r) against the draws' R_si: largest difference {err:.1e}"
        )
        res = run_curves(props, CURVES)
        report_curves(res, f'{len(d)} drawn core black oils', d['api'].to_numpy())
        compare_draft(d, res)
        recommended(d, props)


if __name__ == '__main__':
    main()
