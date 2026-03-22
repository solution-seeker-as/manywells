"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Correlations for dead oil properties (CasADi-compatible)
"""

import casadi as ca

from manywells.pvt import api_from_density
from manywells.units import kelvin_to_fahrenheit, kelvin_to_celsius, CF_CP, CF_DYNCM


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
    T_F = kelvin_to_fahrenheit(T)
    y = ca.power(10, 3.0324 - 0.02023 * api)
    X = y * ca.constpow(T_F, -1.163)
    mu_cP = ca.power(10, X) - 1
    return mu_cP * CF_CP


def dead_oil_surface_tension(rho, T):
    """
    Correlation for dead oil surface tension.

    Reference: Abdul-Majeed, G.H. and Al-Soof, N.B.A., "Estimation of gas-oil
    surface tension", J Pet Sci Eng 27 (2000): 197-200.

    The correlation gives surface tension in dyn/cm, which can be converted 
    to SI units as 1 dyn/cm = 0.001 J/m².

    :param rho: Density of oil (kg/m³)
    :param T: Temperature (K)
    :return: Surface tension of oil (J/m²)
    """
    T_degC = kelvin_to_celsius(T)
    api = api_from_density(rho)
    return CF_DYNCM * (1.11591 - 0.00305 * T_degC) * (38.085 - 0.259 * api)
