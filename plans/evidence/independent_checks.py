"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Independent checks of develop's correlations (plans/model-lock-plan.md, Step L3; specs/model/README.md, Test
vectors, Independent checks). The develop vectors in specs/model/ come from develop itself, and the Rust core
implements the same spec, so neither checks the values. This script compares the component functions of the CasADi
backend, over a grid of inputs, with:

1. fluids (Caleb Bell, MIT licence), an implementation independent of ManyWells:
   - FRIC-4, Chen (1979), with fluids.friction.Chen_1979;
   - FRIC-5, Haaland (1983), with fluids.friction.Haaland;
   - both with the exact Colebrook-White solution, fluids.friction.Clamond, and FRIC-4 with Chen's own form of
     Colebrook's equation (his eq 4, with 3.7065 and 2.5226), solved here;
   - CHK-4, the critical pressure ratio, with fluids.compressible.P_critical_flow;
   - PVT-MIX-9, the mixture viscosity, with fluids.two_phase_voidage.gas_liquid_viscosity (Method='Cicchitti').
2. The sources' equations, transcribed here, for the correlations with no worked example in hand:
   - PVT-GAS-9 and PVT-GAS-11, Dranchuk and Abou-Kassem (1975), Eqs. (2) and (3), with the reduced density found by
     bisection rather than develop's Newton iteration;
   - PVT-GAS-10, the Joule-Thomson factor T (d ln Z / dT)_p, by a central difference of that transcription along an
     isobar;
   - PVT-GAS-5, Sutton (1985), Eqs. (14) and (15) with the coefficients as printed;
   - PVT-OIL-3 and PVT-OIL-12, Abdul-Majeed and Abu Al-Soof (2000), Eqs. (1) to (5), and the dead-oil correlation
     against the data of their Table 1.

The worked examples are in tests/test_worked_examples.py. fluids is not a dependency of manywells; run by hand, from
the repository root:

    uv run --with fluids==1.3.1 python plans/evidence/independent_checks.py

The run is recorded in plans/evidence/README.md (Independent checks).
"""

import math

import casadi as ca
import numpy as np
from fluids.compressible import P_critical_flow
from fluids.friction import Chen_1979, Clamond, Haaland
from fluids.two_phase_voidage import gas_liquid_viscosity
from scipy.optimize import brentq

import fluids
from manywells.choke import SimpsonChokeModel
from manywells.friction import chen_friction_factor, haaland_friction_factor
from manywells.pvt import density_from_api, mixture_viscosity
from manywells.pvt.black_oil import live_oil_surface_tension
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.pvt.gas import dak_jt_factor, dak_reduced_density, dak_z_factor, sutton_pseudo_critical
from manywells.units import CF_DYNCM, CF_PSI, CF_RS

# The grid of the friction comparison: Re from 4000 to 1e8, and eps/D 0 and from 1e-7 to 0.05
RE = np.logspace(np.log10(4.0e3), 8.0, 81)
EPS_D = np.concatenate(([0.0], np.logspace(-7.0, np.log10(0.05), 41)))


def num(x):
    return float(ca.DM(x)) if isinstance(x, (ca.DM, ca.SX, ca.MX)) else float(x)


def rel(a, b):
    return abs(a - b) / abs(b)


def report(name, value, unit=''):
    print(f'  {name:<72s} {value:.3g}{unit}')


# ---------------------------------------------------------------------------------------------------------------
# 1. fluids

def chen_colebrook(Re, eD):
    """Chen's form of Colebrook's equation (Chen 1979, eq 4): 1/sqrt(f) = -2 log10(eD/3.7065 + 2.5226/(Re sqrt(f)))."""
    g = lambda x: x + 2.0 * math.log10(eD / 3.7065 + 2.5226 * x / Re)      # x = 1/sqrt(f)
    x = brentq(g, 1.0, 50.0, xtol=1e-14, rtol=1e-15)
    return 1.0 / x ** 2


def friction():
    print('FRIC-4 and FRIC-5 against fluids, over Re 4000 to 1e8 and eps/D 0 to 0.05 '
          f'({len(RE)} x {len(EPS_D)} points)')
    d_chen, d_haaland = 0.0, 0.0
    cw_chen, cw_haaland = [], []
    for Re in RE:
        for eD in EPS_D:
            chen, haaland = num(chen_friction_factor(Re, eD)), num(haaland_friction_factor(Re, eD))
            d_chen = max(d_chen, rel(chen, Chen_1979(Re, eD)))
            d_haaland = max(d_haaland, rel(haaland, Haaland(Re, eD)))
            exact = Clamond(Re, eD)
            cw_chen.append((chen - exact) / exact)
            cw_haaland.append((haaland - exact) / exact)
    report('FRIC-4 against fluids.friction.Chen_1979, largest relative difference', d_chen)
    report('FRIC-5 against fluids.friction.Haaland, largest relative difference', d_haaland)
    for name, d in (('FRIC-4', cw_chen), ('FRIC-5', cw_haaland)):
        report(f'{name} against Colebrook-White (fluids.friction.Clamond), largest |difference|',
               100 * max(abs(v) for v in d), ' %')
        report(f'{name} against Colebrook-White, range', 100 * min(d), f' % to {100 * max(d):.3g} %')
    # Chen's Table I: 36 points over Re 4000 to 4e8 and eps/D 5e-7 to 0.05, max +0.4651 % and min -0.2286 %, which
    # match (Colebrook - Chen) / Chen
    own = [(chen_colebrook(Re, eD) - num(chen_friction_factor(Re, eD))) / num(chen_friction_factor(Re, eD))
           for Re in np.logspace(np.log10(4.0e3), np.log10(4.0e8), 61)
           for eD in np.logspace(np.log10(5e-7), np.log10(0.05), 31)]
    report("Chen's form of Colebrook (3.7065, 2.5226) minus FRIC-4, over his range, range", 100 * min(own),
           f' % to {100 * max(own):.4g} %')


def choke():
    print('CHK-4 against fluids.compressible.P_critical_flow(1, k), for k from 1.05 to 1.67')
    ks = np.linspace(1.05, 1.67, 63)
    d = max(rel(SimpsonChokeModel.critical_pressure_ratio(k), P_critical_flow(1.0, k)) for k in ks)
    report('largest relative difference', d)
    report('at k = 1.307: develop', SimpsonChokeModel.critical_pressure_ratio(1.307), '')
    report('at k = 1.307: fluids', P_critical_flow(1.0, 1.307), '')


def mixture():
    print("PVT-MIX-9 against fluids.two_phase_voidage.gas_liquid_viscosity(x, mu_l, mu_g, Method='Cicchitti')")
    print('  fluids takes x, the mass fraction; develop computes it from the state as the in-situ mass fraction')
    print('  alpha rho_g / (alpha rho_g + (1 - alpha) rho_l), which is given to fluids: the comparison checks the')
    print("  mixing rule, not which mass fraction it takes (tests/test_worked_examples.py, PVT-MIX-9's finding)")
    d = 0.0
    for alpha in np.linspace(0.0, 1.0, 21):
        for rho_g in (1.0, 20.0, 100.0, 300.0):
            for rho_l, mu_l, mu_g in ((700.0, 5e-4, 1.2e-5), (850.0, 5e-3, 1.5e-5), (1000.0, 1e-3, 2.5e-5)):
                x = alpha * rho_g / (alpha * rho_g + (1 - alpha) * rho_l)
                ours = num(mixture_viscosity(mu_l, mu_g, alpha, rho_l, rho_g))
                d = max(d, rel(ours, gas_liquid_viscosity(x, mu_l, mu_g, Method='Cicchitti')))
    report('largest relative difference', d)


# ---------------------------------------------------------------------------------------------------------------
# 2. The source's equations, transcribed

# Dranchuk and Abou-Kassem (1975), the constants under Eq. (2), and Z_c of Eq. (3)
DAK_A = {1: 0.3265, 2: -1.0700, 3: -0.5339, 4: 0.01569, 5: -0.05165, 6: 0.5475, 7: -0.7361, 8: 0.1844, 9: 0.1056,
         10: 0.6134, 11: 0.7210}
DAK_ZC = 0.270


def dak_eq2(rho_r, T_r):
    """Dranchuk and Abou-Kassem (1975), Eq. (2), as printed."""
    A = DAK_A
    return (1 + (A[1] + A[2] / T_r + A[3] / T_r ** 3 + A[4] / T_r ** 4 + A[5] / T_r ** 5) * rho_r
            + (A[6] + A[7] / T_r + A[8] / T_r ** 2) * rho_r ** 2
            - A[9] * (A[7] / T_r + A[8] / T_r ** 2) * rho_r ** 5
            + A[10] * (1 + A[11] * rho_r ** 2) * (rho_r ** 2 / T_r ** 3) * math.exp(-A[11] * rho_r ** 2))


def dak_density(P_r, T_r):
    """The reduced density of Eq. (3), rho_r = Z_c P_r / (Z T_r), with Z from Eq. (2), by bisection."""
    return brentq(lambda r: r * T_r * dak_eq2(r, T_r) - DAK_ZC * P_r, 1e-12, 3.0, xtol=1e-15, rtol=1e-15)


def dak():
    P_r = np.concatenate((np.linspace(0.2, 1.0, 5), np.linspace(1.5, 15.0, 28)))
    T_r = np.linspace(1.05, 3.0, 40)
    print(f'PVT-GAS-9, PVT-GAS-11 against Dranchuk and Abou-Kassem (1975), Eqs. (2) and (3), solved by bisection, '
          f'over P_r 0.2 to 15 and T_r 1.05 to 3 ({len(P_r)} x {len(T_r)} points)')
    dz, dr, dj = 0.0, 0.0, 0.0
    for p in P_r:
        for t in T_r:
            r = dak_density(p, t)
            r_dev = num(dak_reduced_density(p, t))
            dr = max(dr, rel(r_dev, r))
            dz = max(dz, rel(num(dak_z_factor(r_dev, t)), dak_eq2(r, t)))
            # J = T (d ln Z / dT)_p = t (d ln Z / dt) along the isobar P_r = p, by a central difference
            h = 1e-5 * t
            z_hi, z_lo = dak_eq2(dak_density(p, t + h), t + h), dak_eq2(dak_density(p, t - h), t - h)
            j = t * (math.log(z_hi) - math.log(z_lo)) / (2 * h)
            dj = max(dj, abs(num(dak_jt_factor(r_dev, t)) - j))
    report('reduced density, largest relative difference', dr)
    report('Z, largest relative difference', dz)
    print('PVT-GAS-10 against T (d ln Z / dT)_p of that transcription, by a central difference along the isobar')
    report('J, largest absolute difference', dj)


def sutton():
    print('PVT-GAS-5 against Sutton (1985), Eqs. (14) and (15) as printed, over gamma_g 0.571 to 1.679 (his data)')
    dp, dt = [], []
    for g in np.linspace(0.571, 1.679, 112):
        ppc, tpc = sutton_pseudo_critical(g)
        p_eq14 = 756.8 - 131.0 * g - 3.6 * g ** 2            # psia
        t_eq15 = 169.2 + 349.5 * g - 74.0 * g ** 2           # deg R
        dp.append((ppc / CF_PSI - p_eq14) / p_eq14)
        dt.append(rel(tpc * 1.8, t_eq15))
    report('T_pc, largest relative difference', max(dt))
    report('p_pc, relative difference (develop has 131.07 where Eq. (14) has 131.0)', min(dp), f' to {max(dp):.3g}')


# Abdul-Majeed and Abu Al-Soof (2000), Table 1: dead-oil surface tension (dyn/cm) by API gravity, at 54.44, 37.78
# and 15.56 deg C
AM_TABLE_1 = {15: (32.8, 34.0, 36.0), 20: (31.5, 33.0, 35.0), 30: (29.0, 30.2, 32.2), 35: (28.0, 29.4, 31.4),
              40: (26.2, 27.8, 30.0), 50: (23.1, 24.9, 26.9)}
AM_T = (54.44, 37.78, 15.56)


def abdul_majeed_dead(api, T_C):
    """Eqs. (1) and (3), in dyn/cm."""
    return (1.11591 - 0.00305 * T_C) * (38.085 - 0.259 * api)


def abdul_majeed_ratio(R_s):
    """sigma_lo / sigma_do: Eq. (4) for R_s < 50 Sm3/Sm3, Eq. (5) above."""
    return 1 / (1 + 0.02549 * R_s ** 1.0157) if R_s < 50 else 32.0436 * R_s ** -1.1367


def abdul_majeed():
    print('PVT-OIL-3 against Abdul-Majeed and Abu Al-Soof (2000), Eqs. (1) and (3), over API 15 to 50 and 0 to 150 C')
    d = 0.0
    for api in np.linspace(15.0, 50.0, 36):
        for T_C in np.linspace(0.0, 150.0, 31):
            ours = num(dead_oil_surface_tension(density_from_api(api), T_C + 273.15)) / CF_DYNCM
            d = max(d, rel(ours, abdul_majeed_dead(api, T_C)))
    report('largest relative difference', d)
    res = [num(dead_oil_surface_tension(density_from_api(api), T_C + 273.15)) / CF_DYNCM - s
           for api, row in AM_TABLE_1.items() for T_C, s in zip(AM_T, row)]
    report('against the data of their Table 1 (18 points), largest |residual|', max(abs(v) for v in res), ' dyn/cm')
    report('against the data of their Table 1, mean |residual|', float(np.mean(np.abs(res))), ' dyn/cm')
    print('PVT-OIL-12 against their Eqs. (4) and (5), outside the blend of the two at 50 Sm3/Sm3 (R_s below 20 and '
          'above 80, where the blend weight is below 1e-6)')
    d = 0.0
    for R_s in np.concatenate((np.linspace(0.5, 20.0, 40), np.linspace(80.0, 400.0, 65))):
        ours = num(live_oil_surface_tension(1.0, R_s / CF_RS))
        d = max(d, rel(ours, abdul_majeed_ratio(R_s)))
    report('largest relative difference', d)
    for R_s in (30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0):
        ours = num(live_oil_surface_tension(1.0, R_s / CF_RS))
        print(f'  at {R_s:.0f} Sm3/Sm3: develop {ours:.4f}, Eq. (4) {1 / (1 + 0.02549 * R_s ** 1.0157):.4f}, '
              f'Eq. (5) {32.0436 * R_s ** -1.1367:.4f}')


def main():
    print(f'fluids {fluids.__version__}, casadi {ca.__version__}, numpy {np.__version__}\n')
    friction()
    choke()
    mixture()
    dak()
    sutton()
    abdul_majeed()


if __name__ == '__main__':
    main()
