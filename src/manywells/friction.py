"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Friction factor models (CasADi-compatible)
"""

import casadi as ca

from manywells.ca_functions import ca_max_approx, ca_sigmoid


def haaland_friction_factor(Re, eps_D):
    """
    Haaland (1983) explicit approximation of the Colebrook-White equation
    for the Darcy friction factor in turbulent pipe flow.
    Accuracy within ~1.5% of Colebrook-White.

    Reference: Haaland, S.E., "Simple and Explicit Formulas for the Friction
    Factor in Turbulent Pipe Flow", J Fluids Eng 105 (1983): 89-90.

    :param Re: Reynolds number (CasADi symbolic)
    :param eps_D: Relative roughness = roughness / D (float or symbolic)
    :return: Darcy friction factor (dimensionless)
    """
    inv_sqrt_f = -1.8 * ca.log10(ca.constpow(eps_D / 3.7, 1.11) + 6.9 / Re)
    return 1.0 / (inv_sqrt_f ** 2)


def chen_friction_factor(Re, eps_D):
    """
    Chen (1979) explicit approximation of the Colebrook-White equation
    for the Darcy friction factor in turbulent pipe flow.
    Accuracy within ~0.5% of Colebrook-White.

    Reference: Chen, N.H., "An Explicit Equation for Friction Factor in Pipe",
    Ind. Eng. Chem. Fundamentals 18(3) (1979): 296.

    As used in: Hasan, Kabir & Sayarpour, "Simplified two-phase flow modeling
    in wellbores", J. Pet. Sci. Eng. 72 (2010): Eqs. (A-4), (A-5).

    :param Re: Reynolds number (CasADi symbolic)
    :param eps_D: Relative roughness = roughness / D (float or symbolic)
    :return: Darcy friction factor (dimensionless)
    """
    Lambda = ca.constpow(eps_D, 1.1098) / 2.8257 + ca.constpow(7.149 / Re, 0.8981)
    arg = eps_D / 3.7065 - (5.0452 / Re) * ca.log10(Lambda)
    inv_sqrt_f = -2.0 * ca.log10(arg)
    return 1.0 / (inv_sqrt_f ** 2)


def friction_factor(Re, eps_D, correlation='chen'):
    """
    Darcy friction factor with smooth laminar-turbulent transition.

    Laminar regime (Re < ~2000): f = 64/Re
    Turbulent regime (Re > ~4000): f from selected correlation
    Transition: sigmoid blend centered at Re = 3000.

    :param Re: Reynolds number (CasADi symbolic)
    :param eps_D: Relative roughness = roughness / D (float or symbolic)
    :param correlation: 'chen' (default) or 'haaland'
    :return: Darcy friction factor (dimensionless)
    """
    Re_safe = ca_max_approx(Re, 1.0)
    Re_turb = ca_max_approx(Re_safe, 1000.0)
    f_lam = 64.0 / Re_safe
    if correlation == 'chen':
        f_turb = chen_friction_factor(Re_turb, eps_D)
    else:
        f_turb = haaland_friction_factor(Re_turb, eps_D)
    sigma = ca_sigmoid(Re_safe, 3000, 0.005)
    return (1 - sigma) * f_lam + sigma * f_turb
