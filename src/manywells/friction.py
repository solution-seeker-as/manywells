"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Friction models (CasADi-compatible): the viscous pressure gradient of the momentum balance
"""

import abc
from dataclasses import dataclass

import casadi as ca

from manywells.ca_functions import ca_max_approx, ca_sigmoid

CORRELATIONS = ('chen', 'haaland')


def haaland_friction_factor(Re, eps_D):  # spec: FRIC-5
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


def chen_friction_factor(Re, eps_D):  # spec: FRIC-4
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


def friction_factor(Re, eps_D, correlation='chen'):  # spec: FRIC-6
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


class FrictionModel(abc.ABC):
    """
    Abstract friction model: the viscous pressure gradient F (Pa/m) of the momentum balance,
        F = f_D / (2 D) * rho_m * v_m^2,
    where the Darcy friction factor f_D depends on the model.
    """

    @abc.abstractmethod
    def friction_factor(self, s, fluid, D):
        """
        Darcy friction factor at a point.

        :param s: State at the point (a PointState: p, v_g, v_l, alpha, rho_g, rho_l, T, rho_m, v_m)
        :param fluid: Fluid model (FluidModel), for the viscosities
        :param D: Inner pipe diameter (m)
        :return: Darcy friction factor (dimensionless)
        """
        pass

    def pressure_gradient(self, s, fluid, D):  # spec: FRIC-1
        """
        Viscous pressure gradient at a point.

        :param s: State at the point (a PointState)
        :param fluid: Fluid model (FluidModel)
        :param D: Inner pipe diameter (m)
        :return: F (Pa/m)
        """
        return (self.friction_factor(s, fluid, D) / D / 2) * s.rho_m * s.v_m ** 2


@dataclass(frozen=True)
class FixedFrictionFactor(FrictionModel):
    """One Darcy friction factor for the whole well."""

    f_D: float    # Darcy friction factor (dimensionless)

    def __post_init__(self):
        if not self.f_D > 0:
            raise ValueError('Friction factor must be positive')

    def friction_factor(self, s, fluid, D):  # spec: FRIC-2
        return self.f_D


@dataclass(frozen=True)
class RoughnessFriction(FrictionModel):
    """
    Darcy friction factor from the Reynolds number of the mixture and the relative roughness of the pipe wall,
    with a smooth laminar-turbulent transition (friction_factor above).

    Roughness of new/smooth tubing: 1.5e-6 to 4.5e-5 m. Commercial/welded steel: 4.5e-5 m.
    """

    roughness: float = 4.5e-5     # Pipe wall roughness (m). Default: commercial steel.
    correlation: str = 'chen'     # Turbulent correlation, one of CORRELATIONS

    def __post_init__(self):
        if not self.roughness > 0:
            raise ValueError('Pipe roughness must be positive')
        if self.correlation not in CORRELATIONS:
            raise ValueError(f'Friction correlation {self.correlation!r} is not one of {CORRELATIONS}')

    def friction_factor(self, s, fluid, D):
        mu_m = fluid.mixture_viscosity(s.p, s.T, s.alpha, s.rho_g, s.rho_l)
        Re = s.rho_m * ca.fabs(s.v_m) * D / mu_m  # spec: FRIC-3
        return friction_factor(Re, self.roughness / D, self.correlation)
