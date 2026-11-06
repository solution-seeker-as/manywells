"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of inflow performance relationships
"""

import abc
import math
from dataclasses import dataclass

import casadi as ca

from manywells.units import CF_DAY


class InflowModel(abc.ABC):
    """
    Abstract representation of an inflow model.

    An inflow model describes the reservoir's ability to deliver liquid at a
    given drawdown (p_r - p).  The gas-liquid split is a fluid property: the
    simulator derives the gas rate from FluidModel.f_g.  A gas-rate inflow
    (Forchheimer) gives the gas rate instead, and the liquid that comes with it
    in the fluid's ratios, so it needs the fluid.
    """

    @abc.abstractmethod
    def liquid_mass_flow_rate(self, p, p_r, fluid=None):
        """
        Compute total liquid mass flow rate (oil + water) from the reservoir.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :param fluid: The well's FluidModel. The liquid inflows ignore it; the gas-rate inflow needs it
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        pass


@dataclass(frozen=True)
class ProductivityIndex(InflowModel):
    """
    Productivity index (PI) model:
        w_l = k_l * (p_r - p)
    where
        w_l is liquid mass flow rate (kg/s),
        p is bottomhole flowing pressure (bar),
        p_r is reservoir pressure (bar),
        k_l is the liquid productivity index (kg/s/bar).

    The PI model is linear.
    """

    k_l: float  # Liquid productivity index (kg/s/bar)

    def __post_init__(self):
        if not self.k_l >= 0:
            raise ValueError('Liquid productivity index must be non-negative')

    def liquid_mass_flow_rate(self, p, p_r, fluid=None):
        """
        Compute liquid inflow rate.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :param fluid: Ignored
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        return self.k_l * (p_r - p)  # spec: INF-2


@dataclass(frozen=True)
class Vogel(InflowModel):
    """
    Vogel's inflow performance relationship (IPR):
        w_l = w_l_max * [1 - 0.2 * (p / p_r) - 0.8 * (p / p_r) ** 2]
    where
        w_l is liquid mass flow rate (kg/s),
        p is bottomhole flowing pressure (bar),
        p_r is reservoir pressure (bar),
        w_l_max is the maximum liquid flow rate (kg/s).

    Vogel's IPR is quadratic.
    """

    w_l_max: float  # Maximum liquid mass flow rate (kg/s)

    def __post_init__(self):
        if not self.w_l_max >= 0:
            raise ValueError('Maximum liquid mass flow rate must be non-negative')

    def liquid_mass_flow_rate(self, p, p_r, fluid=None):
        """
        Compute liquid inflow rate.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :param fluid: Ignored
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        r = p / p_r
        return self.w_l_max * (1 - 0.2 * r - 0.8 * r**2)  # spec: INF-1


@dataclass(frozen=True)
class FixedFlowRate(InflowModel):
    """
    Fixed flow rate model:
        w_l = w_l_const

    This model can be used to enforce specific flow rates, which can be useful
    when performing calibration.
    """

    w_l_const: float  # Constant liquid mass flow rate (kg/s)

    def __post_init__(self):
        if not self.w_l_const >= 0:
            raise ValueError('Liquid mass flow rate must be non-negative')

    def liquid_mass_flow_rate(self, p, p_r, fluid=None):
        """
        Return the fixed liquid mass flow rate.

        :param p: Bottomhole flowing pressure (bar) -- ignored
        :param p_r: Reservoir pressure (bar) -- ignored
        :param fluid: Ignored
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        return self.w_l_const  # spec: INF-8


@dataclass(frozen=True)
class Forchheimer(InflowModel):
    """
    Forchheimer's law for a gas well, in the pressure squared (INF-10):
        p_r^2 - p^2 = a * w_g + b * w_g^2
    where
        w_g is the reservoir gas's mass rate at standard conditions (kg/s),
        p is bottomhole flowing pressure (bar),
        p_r is reservoir pressure (bar),
        a is the Darcy coefficient (bar^2 s/kg),
        b is the non-Darcy coefficient (bar^2 s^2/kg^2).

    The rate is the root that is zero at p = p_r, in closed form, and the odd extension above p_r. The liquid comes
    with the gas in the fluid's ratios at standard conditions (INF-11), so liquid_mass_flow_rate needs the fluid.

    A test or a draw that gives the coefficients for a standard volume rate in Sm3/d converts them with
    from_gas_rate. specs/features/022-gas-rate-inflow.md gives them from the reservoir's properties.
    """

    a: float  # Darcy coefficient (bar^2 s/kg)
    b: float  # Non-Darcy coefficient (bar^2 s^2/kg^2)

    def __post_init__(self):
        if not (math.isfinite(self.a) and self.a > 0):
            raise ValueError(f'Forchheimer: a must be positive and finite, got {self.a}')
        if not (math.isfinite(self.b) and self.b >= 0):
            raise ValueError(f'Forchheimer: b must be non-negative and finite, got {self.b}')

    @classmethod
    def from_gas_rate(cls, a_q, b_q, rho_g):
        """
        The inflow from coefficients for the separator gas's standard volume rate q (Sm3/d), p_r^2 - p^2 =
        a_q q + b_q q^2, with w_g = rho_g q / 86 400.

        :param a_q: Darcy coefficient (bar^2 per Sm3/d)
        :param b_q: Non-Darcy coefficient (bar^2 per (Sm3/d)^2)
        :param rho_g: The gas's density at standard conditions (kg/m3), the fluid's rho_g
        """
        if not (math.isfinite(rho_g) and rho_g > 0):
            raise ValueError(f'Forchheimer.from_gas_rate: rho_g must be positive and finite, got {rho_g}')
        s = CF_DAY / rho_g  # s/m3: (Sm3/d) per (kg/s)
        return cls(a=s * a_q, b=s**2 * b_q)

    def gas_mass_flow_rate(self, p, p_r):
        """
        The reservoir gas's mass rate at standard conditions (INF-10).

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Gas mass flow rate, w_g_res (kg/s)
        """
        delta = (p_r - p) * (p_r + p)  # p_r^2 - p^2 (bar^2), factored so that it keeps its precision near p_r
        return 2 * delta / (self.a + ca.sqrt(self.a**2 + 4 * self.b * ca.fabs(delta)))  # spec: INF-10

    def liquid_mass_flow_rate(self, p, p_r, fluid=None):
        """
        The liquid that comes with the gas in the fluid's ratios at standard conditions (INF-11).

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :param fluid: The well's FluidModel, with gas (f_g > 0)
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        if fluid is None:
            raise ValueError("Forchheimer's gas inflow gives the liquid from the gas rate, so it needs the fluid")
        return fluid.reservoir_liquid_rate(self.gas_mass_flow_rate(p, p_r))  # spec: INF-11
