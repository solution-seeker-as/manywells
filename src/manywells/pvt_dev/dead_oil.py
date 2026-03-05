"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Correlations for dead oil properties (CasADi-compatible)
"""

import casadi as ca

from manywells.pvt import api_from_density


def dead_oil_viscosity(api, T):
    """
    Beggs-Robinson (1975) correlation for dead oil viscosity.

    Reference: Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of
    Crude Oil Systems", J Pet Technol 27 (1975): 1140-1141.

    Valid for API 10-58, temperature 100-295 degF (~38-146 degC).

    :param api: Oil API gravity (dimensionless, float constant)
    :param T: Temperature (K), may be a CasADi symbolic
    :return: Dead oil viscosity (Pa-s)
    """
    T_F = 1.8 * (T - 273.15) + 32  # From Kelvin (K) to degrees Fahrenheit (degF)
    y = ca.power(10, 3.0324 - 0.02023 * api)
    X = y * ca.constpow(T_F, -1.163)
    mu_cP = ca.power(10, X) - 1
    return mu_cP * 1e-3  # Unit conversion: 1 centipoise is 1e-3 Pa-s


def dead_oil_surface_tension(rho, T):
    """
    Correlation for dead oil surface tension from paper "Estimation of gas-oil surface tension" by Abdul-Majeed & Al-Soof (2000)

    The correlation gives surface tension in dyn/cm, which can be converted to SI units as 1 dyn/cm = 0.001 J/m².

    :param rho: Density of oil (kg/m³)
    :param T: Temperature (K)
    :return: Surface tension of oil (J/m²)
    """
    cf = 0.001  # Unit conversion factor (1 dyn/cm = 0.001 J/m²)
    T_degC = T - 273.15  # From kelvin to degC
    api = api_from_density(rho)
    return cf * (1.11591 - 0.00305 * T_degC) * (38.085 - 0.259 * api)
