"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Gas thermophysical properties (CasADi-compatible).

Ideal-gas equation of state, formation volume factor, density, viscosity,
and molecular weight / specific gas constant conversions.
"""

import casadi as ca

from manywells.units import (
    R_UNIVERSAL, P_REF, T_REF,
    CF_KGM3_TO_GCC, CF_UP,
    kelvin_to_rankine,
)


def specific_gas_constant(rho):
    """
    Compute the specific gas constant, denoted R_s with unit J / (kg K)
    :param rho: Gas density at standard conditions
    :return: specific gas constant
    """
    return P_REF / (rho * T_REF)


def gas_density(R_s: float, p: float = P_REF, T: float = T_REF):
    """
    Compute gas density using ideal gas law:
        density = p / (R_s T),
    where R_s is the specific gas constant.

    :param R_s: Specific gas constant (J/kg/K)
    :param p: Pressure (Pa)
    :param T: Temperature (K)
    :return: density (kg/m³)
    """
    return p / (R_s * T)


gas_density_std = gas_density
"""Alias for :func:`gas_density` with default (standard-condition) arguments."""


def gas_fvf(p, T):
    """
    Gas formation volume factor using the ideal gas law.

    Bg = V_reservoir / V_standard = (p_ref / p) * (T / T_ref)

    Consistent with the gas equation of state used in the simulator
    (ideal gas: p = rho_g * R_s * T).

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), may be CasADi symbolic
    :return: Bg (Sm3 at standard / m3 at reservoir conditions).
             Multiply standard-condition volume by Bg to get reservoir volume.
    """
    return (P_REF * T) / (T_REF * p)


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
