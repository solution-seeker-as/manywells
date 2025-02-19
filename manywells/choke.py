"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of choke model
"""

import abc
import typing as ty
from dataclasses import dataclass, field
from math import sqrt, exp

import numpy as np
import casadi as ca

from manywells.ca_functions import ca_max_approx
from manywells.constants import CF_PRES


@dataclass
class ChokeModel(abc.ABC):
    """
    Abstract class for choke models based on the Bernoulli equation with support for two-phase correction multipliers:
        w = (K_c * sigma(u) / Phi) * sqrt(2 * rho * (p_in - max(p_out, cpm * p_in)))
    where
        w is mass rate of the mixture (kg/s)
        K_c is a choke coefficient (m²)
        u is choke position in [0, 1] (dimensionless)
        sigma(u) is relative choke opening mapping u to [0, 1] (dimensionless)
        Phi is a two-phase correction multiplier (dimensionless)
        rho is a fluid density (kg/m³)
        p_in and p_out are inlet and outlet pressure (Pa)
        cpm is the critical pressure ration (dimensionless)

    See concrete implementations for different choices of multipliers and densities.

    Choked flow is modeled to occur at the critical pressure p_out = cpm * p_in.
    The flow rate becomes independent of the downstream pressure when p_out is below the critical pressure.
    This is modeled by replacing p_out in the choke equation by the following expression
        p_c = max{p_out, cpm * p_in}
    """

    # Choke properties
    K_c: float = 0.1 * np.pi * (0.1554 / 2) ** 2  # Choke coefficient (m²). Defaults to 10% of area of 6.11 inch pipe.
    cpr: float = None  # Critical pressure ratio (dimensionless)
    chk_profile: str = 'linear'    # Choke profile. Can be any of the profiles listed in _chk_profiles.

    # Admissible choke curve profiles
    _chk_profiles: ty.List[str] = field(default_factory=lambda: ['linear', 'sigmoid', 'convex', 'concave'])

    def __post_init__(self):
        assert self.K_c > 0, 'Choke coefficient must be positive'
        assert self.chk_profile in self._chk_profiles, f'Choke profile {self.chk_profile} is not supported'
        self.cpr = self.critical_pressure_ratio()

    def choke_opening(self, u: float):
        """
        Compute choke opening given choke position and the choke type

        :param u: Choke position
        :return: Choke (relative) opening
        """
        if self.chk_profile == 'linear':
            return u
        elif self.chk_profile == 'sigmoid':
            b = 1.5
            return (u**b)/(u**b + (1-u)**b)
        elif self.chk_profile == 'convex':
            b = 0.25  # Number in [0, 1]
            return b * u + (1 - b) * u ** 2
        elif self.chk_profile == 'concave':
            # This is also known as a quick open valve characteristics
            b = 0.75  # Number in (0, 1], changed from 0.5 to 0.75
            return u ** b
        else:
            raise NotImplementedError('Choke profile not supported')

    @staticmethod
    def critical_pressure_ratio(gamma: float = 1.307):
        """
        Compute the critical pressure ratio:
            cpr = p_crit / p_in = (2 / (gamma + 1)) ** (gamma / (gamma - 1)),
        where p_crit is the critical pressure (downstream) and gamma is the heat capacity ratio of the gas.

        The flow is choked if the downstream pressure is below p_crit = cpr * p_in.

        :param gamma: Heat capacity ratio (c_p / c_v). Default value is for methane gas at 20 degC.
        :return: Critical pressure ratio
        """
        return (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    def choke_equation(self, u: float, p_in: float, p_out: float, rho: float, multiplier: float):
        """
        Choke equation for mass flow rate:
            mass flow rate = (K_c * sigma(u) / Phi) * sqrt(2 * rho * dp)
        where Phi is the two-phase correction multiplier.

        :param u: Choke position in [0, 1] (dimensionless)
        :param p_in: Upstream (inlet) pressure (bar)
        :param p_out: Downstream (outlet) pressure (bar)
        :param rho: Fluid density (kg/m³)
        :param multiplier: Two-phase correction multiplier (dimensionless)
        :return: Mass flow rate (kg/s)
        """
        chk = self.choke_opening(u)
        p_c = ca_max_approx(self.cpr * p_in, p_out)  # Approximation of max(cpr * p_in, p_out)
        dp = CF_PRES * (p_in - p_c)  # Pressure difference (Pa)
        return self.K_c * chk * ca.sqrt(2 * rho * dp / multiplier)

    @abc.abstractmethod
    def mass_flow_rate(self, *args, **kwargs):
        """
        Computes the mass flow rate through the choke
        """
        pass

    def is_choked(self, p_in, p_out):
        """
        Return True if flow is choked, otherwise False

        :param p_in: Upstream pressure (bar)
        :param p_out: Downstream pressure (bar)
        :return: True if flow is choked, otherwise False
        """
        return p_out <= self.cpr * p_in


class BernoulliChokeModel(ChokeModel):

    def mass_flow_rate(self, u, p_in, p_out, rho_m):
        """
        Compute mass flow rate through choke using Bernoulli model
            rho = rho_m
            Phi = 1 (no two-phase correction)

        :param u: Choke position in [0, 1] (dimensionless)
        :param p_in: Upstream pressure (bar)
        :param p_out: Downstream pressure (bar)
        :param rho_m: Mixture density (kg/m³)
        :return: Mass flow rate
        """
        return self.choke_equation(u, p_in, p_out, rho=rho_m, multiplier=1.0)


class SimpsonChokeModel(ChokeModel):

    def mass_flow_rate(self, u, p_in, p_out, x_g, rho_g, rho_l):
        """
        Compute mass flow rate through choke with two-phase correction
            rho = rho_l
            Phi = multiplier of Simpson et al.

        :param u: Choke position in [0, 1] (dimensionless)
        :param p_in: Upstream pressure (bar)
        :param p_out: Downstream pressure (bar)
        :param x_g: Mass fraction of gas (dimensionless)
        :param rho_g: Gas density (kg/m³)
        :param rho_l: Liquid density (kg/m³)
        :return: Mass flow rate
        """
        Phi = self.simpson_multiplier(x_g, rho_g, rho_l)  # Two-phase correction multiplier
        return self.choke_equation(u, p_in, p_out, rho=rho_l, multiplier=Phi)

    @staticmethod
    def simpson_multiplier(x_g, rho_g, rho_l):
        """
        Compute two-phase correction multiplier of Simpson et al. (1983)

        Using Simpson's multiplier with liquid density, is equivalent to using the momentum density
            1 / rho_e = (x_g / rho_g + k * (x_l / rho_l)) * (x_g + x_l / k),
        with Simpson's slip model k = (rho_l / rho_g)^(1/6).

        :param x_g: Mass fraction of gas (dimensionless)
        :param rho_g: Gas density (kg/m³)
        :param rho_l: Liquid density (kg/m³)
        :return: Multiplier
        """
        s = ca.constpow(rho_l / rho_g, 1 / 6)  # Simpson slip model S = u_g / u_l = (rho_l / rho_g)^(1/6)
        # return (x_g * rho_l / rho_g + s * (1 - x_g)) * (x_g + (1 - x_g) / s)
        return (1 + x_g * (s - 1)) * (1 + x_g * (ca.constpow(s, 5) - 1))
