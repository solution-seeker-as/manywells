"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Gas thermophysical properties (CasADi-compatible).

Equation of state (ideal gas, or a real gas with the Z-factor of Papay or of
the Dranchuk-Abou-Kassem equation of state), the Joule-Thomson factor,
formation volume factor, density, viscosity, and molecular weight / specific
gas constant conversions.
"""

import casadi as ca

from manywells.units import (
    R_UNIVERSAL, P_REF, T_REF,
    CF_PSI, CF_KGM3_TO_GCC, CF_UP,
    kelvin_to_rankine,
)


def specific_gas_constant(rho):  # spec: PVT-GAS-2
    """
    Compute the specific gas constant, denoted R_s with unit J / (kg K)
    :param rho: Gas density at standard conditions
    :return: specific gas constant
    """
    return P_REF / (rho * T_REF)


def gas_density(R_s: float, p: float = P_REF, T: float = T_REF, Z: float = 1.0):  # spec: PVT-GAS-1, PVT-GAS-2
    """
    Compute gas density using the real gas equation of state:
        density = p / (Z * R_s * T),
    where R_s is the specific gas constant and Z is the compressibility factor.

    With the default Z=1.0 this reduces to the ideal gas law.

    :param R_s: Specific gas constant (J/kg/K)
    :param p: Pressure (Pa)
    :param T: Temperature (K)
    :param Z: Gas compressibility factor (dimensionless), default 1.0
    :return: density (kg/m³)
    """
    return p / (Z * R_s * T)


gas_density_std = gas_density
"""Alias for :func:`gas_density` with default (standard-condition) arguments."""


def gas_fvf(p, T, Z=1.0, Z_ref=1.0):  # spec: PVT-GAS-8
    """
    Gas formation volume factor.

    Bg = V_reservoir / V_standard = (Z / Z_ref) * (p_ref / p) * (T / T_ref)

    With the default Z=Z_ref=1.0 this reduces to the ideal gas form.

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), may be CasADi symbolic
    :param Z: Compressibility factor at (p, T), default 1.0
    :param Z_ref: Compressibility factor at standard conditions, default 1.0
    :return: Bg (m3 at reservoir conditions / Sm3 at standard conditions).
             Multiply standard-condition volume by Bg to get reservoir volume.
    """
    return (Z / Z_ref) * (P_REF * T) / (T_REF * p)


def molecular_weight(R_s):  # spec: PVT-GAS-6
    """
    Compute gas molecular weight from specific gas constant.

    :param R_s: Specific gas constant (J/(kg·K))
    :return: Molecular weight (g/mol)
    """
    return R_UNIVERSAL / R_s


def gas_viscosity(T, rho_g, M_g):  # spec: PVT-GAS-7
    """
    Gas viscosity using the Lee-Gonzalez-Eakin (1966) correlation.
    CasADi-compatible.

    Reference: Lee, A.L., Gonzalez, M.H. and Eakin, B.E., "The Viscosity of
    Natural Gases", J Pet Technol 18 (1966): 997-1000.

    :param T: Temperature (K), may be a CasADi symbolic
    :param rho_g: Gas density (kg/m³), may be a CasADi symbolic
    :param M_g: Molecular weight of gas (g/mol), float constant
    :return: Gas viscosity (Pa·s)
    """
    T_R = kelvin_to_rankine(T)  # From Kelvin (K) to degrees Rankine (degR)
    rho_gcc = rho_g * CF_KGM3_TO_GCC  # From kg/m³ to g/cm³
    K = (9.4 + 0.02 * M_g) * ca.constpow(T_R, 1.5) / (209 + 19 * M_g + T_R)
    X = 3.5 + 986 / T_R + 0.01 * M_g
    Y = 2.4 - 0.2 * X
    return K * ca.exp(X * ca.constpow(rho_gcc, Y)) * CF_UP


def sutton_pseudo_critical(sg_gas):  # spec: PVT-GAS-5
    """
    Pseudo-critical pressure and temperature from gas specific gravity.

    Reference: Sutton, R.P., "Compressibility Factors for High-Molecular-Weight
    Reservoir Gases", SPE-14265, 1985.

    :param sg_gas: Gas specific gravity relative to air (dimensionless)
    :return: (ppc, tpc) — pseudo-critical pressure (Pa) and temperature (K)
    """
    ppc_psia = 756.8 - 131.07 * sg_gas - 3.6 * sg_gas ** 2
    tpc_R = 169.2 + 349.5 * sg_gas - 74.0 * sg_gas ** 2
    return ppc_psia * CF_PSI, tpc_R / 1.8


DAK_A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)
"""The coefficients A_1 to A_11 of the Dranchuk-Abou-Kassem equation of state (1975, Eq. 2)."""

DAK_ZC = 0.27
"""The critical compressibility factor of DAK's reduced density (1975, Eq. 3)."""

DAK_NEWTON_STEPS = 20
"""Newton steps of dak_reduced_density, unrolled so that it accepts CasADi symbols. From the ideal-gas density,
Newton converges to 1e-12 in at most 17 steps for 1.05 <= T_pr <= 3 and p_pr <= 30
(specs/features/016-joule-thomson.md)."""


def _dak_terms(r, t):
    """DAK's Z, dZ/dr and t dZ/dt at reduced density r and pseudo-reduced temperature t."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = DAK_A
    c1 = a1 + a2 / t + a3 / t ** 3 + a4 / t ** 4 + a5 / t ** 5
    c2 = a6 + a7 / t + a8 / t ** 2
    c3 = a9 * (a7 / t + a8 / t ** 2)
    c4 = a10 / t ** 3
    e = ca.exp(-a11 * r ** 2)
    Z = 1 + c1 * r + c2 * r ** 2 - c3 * r ** 5 + c4 * r ** 2 * (1 + a11 * r ** 2) * e
    Z_r = c1 + 2 * c2 * r - 5 * c3 * r ** 4 + 2 * c4 * r * e * (1 + a11 * r ** 2 - a11 ** 2 * r ** 4)
    tZ_t = ((-a2 / t - 3 * a3 / t ** 3 - 4 * a4 / t ** 4 - 5 * a5 / t ** 5) * r
            + (-a7 / t - 2 * a8 / t ** 2) * r ** 2
            - a9 * (-a7 / t - 2 * a8 / t ** 2) * r ** 5
            - 3 * a10 / t ** 3 * r ** 2 * (1 + a11 * r ** 2) * e)
    return Z, Z_r, tZ_t


def dak_z_factor(r, t):  # spec: PVT-GAS-9
    """
    Gas compressibility factor from the Dranchuk-Abou-Kassem (1975) equation of state, explicit in the reduced
    density. CasADi-compatible.

    Reference: Dranchuk, P.M. and Abou-Kassem, J.H., "Calculation of Z Factors for Natural Gases Using Equations of
    State", J Can Pet Technol 14(3) (1975): 34-36, Eq. (2). Recommended for 0.2 <= p_pr < 30 and 1.0 < T_pr <= 3.0.

    :param r: Reduced gas density (dimensionless), may be CasADi symbolic
    :param t: Pseudo-reduced temperature T / T_pc (dimensionless), may be CasADi symbolic
    :return: Z-factor (dimensionless)
    """
    return _dak_terms(r, t)[0]


def dak_jt_factor(r, t):  # spec: PVT-GAS-10
    """
    The gas's Joule-Thomson factor J = T (d ln Z / dT)_p of the DAK equation of state, at reduced density r and
    pseudo-reduced temperature t: J = (t Z_t - r Z_r) / (Z + r Z_r). CasADi-compatible. The Joule-Thomson
    coefficient is J / (rho_g c_pg). The denominator is positive for T_pr >= 1.05 and p_pr <= 15.

    :param r: Reduced gas density (dimensionless), may be CasADi symbolic
    :param t: Pseudo-reduced temperature (dimensionless), may be CasADi symbolic
    :return: J (dimensionless)
    """
    Z, Z_r, tZ_t = _dak_terms(r, t)
    return (tZ_t - r * Z_r) / (Z + r * Z_r)


def dak_reduced_density(ppr, t):  # spec: PVT-GAS-11
    """
    DAK's reduced density at pseudo-reduced pressure and temperature: the root of r t Z(r, t) / Z_c = p_pr, by
    DAK_NEWTON_STEPS Newton steps from the ideal-gas density Z_c p_pr / t. CasADi-compatible.

    :param ppr: Pseudo-reduced pressure p / p_pc (dimensionless), may be CasADi symbolic
    :param t: Pseudo-reduced temperature (dimensionless), may be CasADi symbolic
    :return: Reduced gas density (dimensionless)
    """
    r = DAK_ZC * ppr / t
    for _ in range(DAK_NEWTON_STEPS):
        Z, Z_r, _ = _dak_terms(r, t)
        r = r - (r * t * Z - DAK_ZC * ppr) / (t * (Z + r * Z_r))
    return r


def gas_z_factor(p, T, sg_gas):  # spec: PVT-GAS-4
    """
    Gas compressibility factor using the Papay (1968) correlation.
    CasADi-compatible (explicit, smooth, no iteration).

    Reference: Papay, J., "A Termelestechnologiai Parameterek Valtozasa
    a Gaztelepek Muvelese Soran", OGIL MUSZ Tud Kozl (1968): 267-273.

    Accurate for p_pr < 6, T_pr > 1.05 (covers most wellbore conditions).

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), may be CasADi symbolic
    :param sg_gas: Gas specific gravity relative to air (float constant)
    :return: Z-factor (dimensionless)
    """
    ppc, tpc = sutton_pseudo_critical(sg_gas)
    ppr = p / ppc
    tpr = T / tpc
    return (
        1
        - 3.52 * ppr * ca.power(10, -0.9813 * tpr)
        + 0.274 * ppr ** 2 * ca.power(10, -0.8157 * tpr)
    )
