"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of choke model
"""

import abc
from dataclasses import dataclass

import numpy as np
import casadi as ca

from manywells.ca_functions import ca_max_approx
from manywells.units import CF_BAR


CHOKE_PROFILES = ('linear', 'sigmoid', 'convex', 'concave')


@dataclass(frozen=True)
class ChokeModel(abc.ABC):
    """
    Abstract class for choke models based on the Bernoulli equation with support for two-phase correction multipliers:
        w = K_c * sigma(u) * sqrt(2 * rho * max(p_in - max(p_out, cpm * p_in), 0) / Phi)
    where
        w is mass rate of the mixture (kg/s)
        K_c is a choke coefficient (m²)
        u is choke position in [0, 1] (dimensionless)
        sigma(u) is relative choke opening mapping u to [0, 1] (dimensionless)
        Phi is a two-phase correction multiplier (dimensionless)
        rho is a fluid density (kg/m³)
        p_in and p_out are inlet and outlet pressure (bar)
        cpm is the critical pressure ration (dimensionless)

    A concrete model chooses the density and the multiplier from the state upstream of the choke
    (density_and_multiplier); mass_flow_rate is the same for every model.

    Choked flow is modeled to occur at the critical pressure p_out = cpm * p_in.
    The flow rate becomes independent of the downstream pressure when p_out is below the critical pressure.
    This is modeled by replacing p_out in the choke equation by the following expression
        p_c = max{p_out, cpm * p_in}
    """

    # Choke properties
    K_c: float = 0.1 * np.pi * (0.1554 / 2) ** 2  # Choke coefficient (m²). Defaults to 10% of area of 6.11 inch pipe.
    cpr: float = None  # Critical pressure ratio (dimensionless). Always set from critical_pressure_ratio().
    chk_profile: str = 'linear'    # Choke profile, one of CHOKE_PROFILES

    def __post_init__(self):
        if not self.K_c > 0:
            raise ValueError('Choke coefficient must be positive')
        if self.chk_profile not in CHOKE_PROFILES:
            raise ValueError(f'Choke profile {self.chk_profile} is not supported')
        object.__setattr__(self, 'cpr', self.critical_pressure_ratio())

    def choke_opening(self, u: float):
        """
        Compute choke opening given choke position and the choke type

        :param u: Choke position
        :return: Choke (relative) opening
        """
        if self.chk_profile == 'linear':  # spec: CHK-7
            return u
        elif self.chk_profile == 'sigmoid':  # spec: CHK-8
            b = 1.5
            return (u**b)/(u**b + (1-u)**b)
        elif self.chk_profile == 'convex':  # spec: CHK-9
            b = 0.25  # Number in [0, 1]
            return b * u + (1 - b) * u ** 2
        elif self.chk_profile == 'concave':  # spec: CHK-10
            # This is also known as a quick open valve characteristics
            b = 0.75  # Number in (0, 1], changed from 0.5 to 0.75
            return u ** b
        else:
            raise NotImplementedError('Choke profile not supported')

    @staticmethod
    def critical_pressure_ratio(gamma: float = 1.307):  # spec: CHK-4
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
            mass flow rate = K_c * sigma(u) * sqrt(2 * rho * dp / Phi)
        where Phi is the two-phase correction multiplier.

        :param u: Choke position in [0, 1] (dimensionless)
        :param p_in: Upstream (inlet) pressure (bar)
        :param p_out: Downstream (outlet) pressure (bar)
        :param rho: Fluid density (kg/m³)
        :param multiplier: Two-phase correction multiplier (dimensionless)
        :return: Mass flow rate (kg/s)
        """
        chk = self.choke_opening(u)
        # spec: CHK-3
        p_c = ca_max_approx(self.cpr * p_in, p_out)  # Approximation of max(cpr * p_in, p_out)
        dp = CF_BAR * (p_in - p_c)  # Pressure difference (Pa)
        w = self.K_c * chk * ca.sqrt(2 * rho * dp / multiplier)  # spec: CHK-2
        # spec: CHK-11. No flow from the well where p_in <= p_c. if_else, unlike sqrt(max(dp, 0)), keeps the derivative
        # finite (zero) there, so a solver that steps into that region is not stopped by a NaN Jacobian.
        return ca.if_else(dp > 0, w, 0)

    @abc.abstractmethod
    def density_and_multiplier(self, s, A):
        """
        The density and the two-phase multiplier of the choke equation, from the state upstream of the choke.

        :param s: State at the wellhead (a PointState: p, v_g, v_l, alpha, rho_g, rho_l, T, rho_m, v_m)
        :param A: Cross-sectional area of the pipe (m²)
        :return: Density (kg/m³) and multiplier (dimensionless)
        """
        pass

    def mass_flow_rate(self, u, p_s, s, A):
        """
        Mass flow rate through the choke, from the wellhead state s at pressure s.p to the pressure p_s.

        :param u: Choke position in [0, 1] (dimensionless)
        :param p_s: Pressure downstream of the choke (bar)
        :param s: State at the wellhead (a PointState)
        :param A: Cross-sectional area of the pipe (m²)
        :return: Mass flow rate (kg/s)
        """
        rho, multiplier = self.density_and_multiplier(s, A)
        return self.choke_equation(u, s.p, p_s, rho=rho, multiplier=multiplier)

    def is_choked(self, p_in, p_out):  # spec: CHK-12
        """
        Return True if flow is choked, otherwise False

        :param p_in: Upstream pressure (bar)
        :param p_out: Downstream pressure (bar)
        :return: True if flow is choked, otherwise False
        """
        return p_out <= self.cpr * p_in


class BernoulliChokeModel(ChokeModel):

    def density_and_multiplier(self, s, A):  # spec: CHK-6
        """
        Bernoulli model: the mixture density, and no two-phase correction (Phi = 1).
        """
        return s.rho_m, 1.0


class SimpsonChokeModel(ChokeModel):

    def density_and_multiplier(self, s, A):  # spec: CHK-5
        """
        Two-phase correction of Simpson et al.: the liquid density, and Simpson's multiplier at the gas mass
        fraction of the flow, x_g = w_g / w_m.
        """
        w_g = A * s.alpha * s.rho_g * s.v_g  # Gas mass flow rate
        w_l = A * (1 - s.alpha) * s.rho_l * s.v_l  # Liquid mass flow rate
        x_g = w_g / (w_g + w_l)  # Mass fraction of gas
        return s.rho_l, self.simpson_multiplier(x_g, s.rho_g, s.rho_l)

    @staticmethod
    def simpson_multiplier(x_g, rho_g, rho_l):  # spec: CHK-5
        """
        Compute two-phase correction multiplier of Simpson et al.

        Reference: Simpson, H.C., Rooney, D.H. and Grattan, E., "Two-phase flow
        through gate valves and orifice plates", Int. Conf. Physical Modelling of
        Multi-Phase Flow, Coventry, England (1983): 57-76.

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
