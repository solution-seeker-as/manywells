"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The gas's Joule-Thomson factor J = T (d ln Z / dT)_p over the range develop's sampler draws (feature spec
specs/features/016-joule-thomson.md): Papay's z-factor (PVT-GAS-4) differentiated analytically, the
Dranchuk-Abou-Kassem equation of state (DAK; Dranchuk and Abou-Kassem 1975, Eq. 2) in the closed form of the proposed PVT-GAS-10
and by a finite difference of its own Z, both with Sutton's pseudo-critical properties (PVT-GAS-5), against
CoolProp's reference equations of state: Setzmann and Wagner (1991) for methane, and CoolProp's mixture model for
two natural gases. Also the sign of DAK's denominator Z + rho_r dZ/drho_r over reduced temperatures.

Needs CoolProp, which is not a dependency of manywells. From the repository root:

    uv run --no-project --with CoolProp --with numpy --with scipy python plans/evidence/jt_factor.py
"""

import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import brentq

CF_PSI = 6894.76    # Pa/psi
M_AIR = 28.97       # kg/kmol
A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)

GASES = {
    'methane': 'Methane',
    'natural gas, gravity 0.67': 'HEOS::Methane[0.85]&Ethane[0.09]&Propane[0.04]&n-Butane[0.02]',
    'natural gas, gravity 0.80': 'HEOS::Methane[0.70]&Ethane[0.15]&Propane[0.10]&n-Butane[0.05]',
}
PRESSURES = (5, 20, 50, 100, 150, 200, 270, 350, 460)    # bar
TEMPERATURES = (280, 300, 330, 360, 390, 425)            # K


def sutton(sg):
    """Pseudo-critical pressure (Pa) and temperature (K), PVT-GAS-5."""
    return (756.8 - 131.07 * sg - 3.6 * sg ** 2) * CF_PSI, (169.2 + 349.5 * sg - 74.0 * sg ** 2) / 1.8


def j_papay(p, T, sg):
    """T (d ln Z / dT)_p of Papay's z-factor, PVT-GAS-4, differentiated analytically. p in Pa."""
    ppc, tpc = sutton(sg)
    ppr, tpr = p / ppc, T / tpc
    a = 3.52 * ppr * 10 ** (-0.9813 * tpr)
    b = 0.274 * ppr ** 2 * 10 ** (-0.8157 * tpr)
    return tpr * np.log(10) * (0.9813 * a - 0.8157 * b) / (1 - a + b)


def dak(r, t):
    """DAK's Z, dZ/dr and t dZ/dt at reduced density r and reduced temperature t (proposed PVT-GAS-9)."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A
    c1 = a1 + a2 / t + a3 / t ** 3 + a4 / t ** 4 + a5 / t ** 5
    c2 = a6 + a7 / t + a8 / t ** 2
    c3 = a9 * (a7 / t + a8 / t ** 2)
    c4 = a10 / t ** 3
    e = np.exp(-a11 * r ** 2)
    Z = 1 + c1 * r + c2 * r ** 2 - c3 * r ** 5 + c4 * r ** 2 * (1 + a11 * r ** 2) * e
    Z_r = c1 + 2 * c2 * r - 5 * c3 * r ** 4 + 2 * c4 * r * e * (1 + a11 * r ** 2 - a11 ** 2 * r ** 4)
    tZ_t = ((-a2 / t - 3 * a3 / t ** 3 - 4 * a4 / t ** 4 - 5 * a5 / t ** 5) * r
            + (-a7 / t - 2 * a8 / t ** 2) * r ** 2
            - a9 * (-a7 / t - 2 * a8 / t ** 2) * r ** 5
            - 3 * a10 / t ** 3 * r ** 2 * (1 + a11 * r ** 2) * e)
    return Z, Z_r, tZ_t


def j_dak_closed_form(r, t):
    """The proposed PVT-GAS-10: J = (t Z_t - r Z_r) / (Z + r Z_r)."""
    Z, Z_r, tZ_t = dak(r, t)
    return (tZ_t - r * Z_r) / (Z + r * Z_r)


def dak_reduced_density(p, T, sg):
    """DAK's reduced density at (p, T), solving p_pr = r t Z(r, t) / 0.27 for r. p in Pa."""
    ppc, tpc = sutton(sg)
    ppr, t = p / ppc, T / tpc
    return brentq(lambda r: r * t * dak(r, t)[0] / 0.27 - ppr, 1e-12, 3.0, xtol=1e-15, rtol=1e-15)


def z_dak(p, T, sg):
    ppc, tpc = sutton(sg)
    return dak(dak_reduced_density(p, T, sg), T / tpc)[0]


def j_finite_difference(z, p, T, h):
    return T * (np.log(z(p, T + h)) - np.log(z(p, T - h))) / (2 * h)


def main():
    for name, fluid in GASES.items():
        sg = CP.PropsSI('M', 'T', 300, 'P', 1e5, fluid) * 1e3 / M_AIR
        ppc, tpc = sutton(sg)
        print(f'\n{name}: gravity {sg:.3f}, p_pc {ppc / 1e5:.1f} bar, T_pc {tpc:.1f} K')
        print('J per (p, T): reference / Papay / DAK closed form (DAK finite difference)')
        print('p bar  p_pr ' + ''.join(f'| T = {T} K{"":22}' for T in TEMPERATURES))
        for p_bar in PRESSURES:
            p = p_bar * 1e5
            line = f'{p_bar:<6} {p / ppc:5.2f} '
            for T in TEMPERATURES:
                try:
                    j_ref = j_finite_difference(lambda pp, TT: CP.PropsSI('Z', 'T', TT, 'P', pp, fluid), p, T, 0.05)
                except ValueError:
                    j_ref = np.nan
                j_cf = j_dak_closed_form(dak_reduced_density(p, T, sg), T / tpc)
                j_fd = j_finite_difference(lambda pp, TT: z_dak(pp, TT, sg), p, T, 1e-3)
                line += f'| {j_ref:6.3f} {j_papay(p, T, sg):6.3f} {j_cf:6.3f} ({j_fd:6.3f}) '
            print(line)
        p, T = 460e5, 360.0
        print(f'Z at {p / 1e5:.0f} bar, {T:.0f} K: reference {CP.PropsSI("Z", "T", T, "P", p, fluid):.3f}, '
              f'Papay {1 - 3.52 * p / ppc * 10 ** (-0.9813 * T / tpc) + 0.274 * (p / ppc) ** 2 * 10 ** (-0.8157 * T / tpc):.3f}, '
              f'DAK {z_dak(p, T, sg):.3f}')

    print('\nDAK: minimum of Z + r dZ/dr over the reduced densities of p_pr <= 15, by reduced temperature')
    r = np.linspace(1e-6, 3.0, 3001)
    for t in (1.0, 1.05, 1.1, 1.18, 1.3, 1.5, 2.0, 3.0):
        Z, Z_r, _ = dak(r, t)
        in_range = r * t * Z / 0.27 <= 15
        print(f'  T_pr = {t:4.2f}: {np.min((Z + r * Z_r)[in_range]):6.3f}')


if __name__ == '__main__':
    main()
