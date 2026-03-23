"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Gas thermophysical properties (CasADi-compatible).

Equation of state (ideal gas with optional Z-factor), formation volume
factor, density, viscosity, and molecular weight / specific gas constant
conversions.
"""

import casadi as ca

from manywells.units import (
    R_UNIVERSAL, P_REF, T_REF,
    CF_PSI, CF_KGM3_TO_GCC, CF_UP,
    kelvin_to_rankine,
)


def specific_gas_constant(rho):
    """
    Compute the specific gas constant, denoted R_s with unit J / (kg K)
    :param rho: Gas density at standard conditions
    :return: specific gas constant
    """
    return P_REF / (rho * T_REF)


def gas_density(R_s: float, p: float = P_REF, T: float = T_REF, Z: float = 1.0):
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


def gas_fvf(p, T, Z=1.0, Z_ref=1.0):
    """
    Gas formation volume factor.

    Bg = V_reservoir / V_standard = (Z / Z_ref) * (p_ref / p) * (T / T_ref)

    With the default Z=Z_ref=1.0 this reduces to the ideal gas form.

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), may be CasADi symbolic
    :param Z: Compressibility factor at (p, T), default 1.0
    :param Z_ref: Compressibility factor at standard conditions, default 1.0
    :return: Bg (Sm3 at standard / m3 at reservoir conditions).
             Multiply standard-condition volume by Bg to get reservoir volume.
    """
    return (Z / Z_ref) * (P_REF * T) / (T_REF * p)


def molecular_weight(R_s):
    """
    Compute gas molecular weight from specific gas constant.

    :param R_s: Specific gas constant (J/(kg·K))
    :return: Molecular weight (g/mol)
    """
    return R_UNIVERSAL / R_s


def gas_viscosity(T, rho_g, M_g):
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


def sutton_pseudo_critical(sg_gas):
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


def gas_z_factor(p, T, sg_gas):
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
