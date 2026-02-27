"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Correlations for dead oil properties (CasADi-compatible)
"""

import casadi as ca


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
