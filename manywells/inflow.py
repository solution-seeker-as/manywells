"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 27 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of inflow performance relationships

TODO: Test constant flow model
"""

import abc
from dataclasses import dataclass


def compute_gas_mass_fraction(q_l, q_g, rho_l, rho_g):
    """
    Compute gas mass fraction from volumetric flow rates and densities
        f_g = w_g / (w_l + w_g) = (rho_g * q_g) / (rho_l * q_l + rho_g * q_g)

    Note that q_l, q_g, rho_l, rho_g must be specified for the same conditions (pressure and temperature).
    For example, all can be specified for standard reference conditions (see pvt.py).

    Note that the gas mass fraction is not the same as the gas volumetric fraction.

    :param q_l: Volumetric flow rate of liquid (m³/s)
    :param q_g: Volumetric flow rate of gas (m³/s)
    :param rho_l: Density of liquid (kg/m³)
    :param rho_g: Density of gas (kg/m³)
    :return: Gas mass fraction, f_g (dimensionless)
    """
    return (rho_g * q_g) / (rho_l * q_l + rho_g * q_g)


class InflowModel(abc.ABC):
    """
    Abstract representation of an inflow model
    """

    @abc.abstractmethod
    def mass_flow_rates(self, p, p_r):
        """
        Compute mass flow rates given bottomhole pressure and reservoir pressure

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid and gas mass rates, (w_l, w_g)
        """
        pass


@dataclass
class ProductivityIndex(InflowModel):
    """
    Productivity index (PI) model:
        w_l = k_l * (p_r - p),
        w_g = (f_g / (1 - f_g)) * w_l
    where
        w_l is liquid mass flow rate (kg/s),
        w_g is gas mass flow rate (kg/s),
        p is bottomhole flowing pressure (bar),
        p_r is reservoir pressure (bar),
        k_l is the liquid productivity index: change in liquid flow rate per unit change in pressure (kg/s/bar),
        f_g is the fraction of gas to total mass flow rate (dimensionless). Must be in (0, 1).

    The PI model is linear.
    """

    k_l: float    # Liquid productivity index (kg/s/bar)
    f_g: float    # Fraction of gas to total mass flow rate (dimensionless)

    def __post_init__(self):
        assert self.k_l >= 0, 'Liquid productivity index must be non-negative'
        assert 0 < self.f_g < 1, 'Gas mass fraction must be in (0, 1)'

    def mass_flow_rates(self, p, p_r):
        """
        Compute inflow rates

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid and gas mass rates, (w_l, w_g)
        """
        w_l = self.k_l * (p_r - p)
        w_g = (self.f_g / (1 - self.f_g)) * w_l
        return w_l, w_g


@dataclass
class Vogel(InflowModel):
    """
    Vogel's inflow performance relationship (IPR):
        w_l = w_l_max * [1 - 0.2 * (p / p_r) - 0.8 * (p / p_r) ** 2],
        w_g = (f_g / (1 - f_g)) * w_l,
    where
        w_l is liquid mass flow rate (kg/s),
        w_g is gas mass flow rate (kg/s),
        p is bottomhole flowing pressure (bar),
        p_r is reservoir pressure (bar),
        k_l is the maximum liquid flow rate (kg/s),
        f_g is the fraction of gas to total mass flow rate (dimensionless),

    Vogel's IPR is quadratic.
    """

    w_l_max: float    # Maximum liquid mass flow rate (kg/s)
    f_g: float        # Fraction of gas to total mass flow rate (dimensionless)

    def __post_init__(self):
        assert self.w_l_max >= 0, 'Maximum liquid mass flow rate must be non-negative'
        assert 0 < self.f_g < 1, 'Gas mass fraction must be in (0, 1)'

    def mass_flow_rates(self, p, p_r):
        """
        Compute mass flow rates

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid and gas mass flow rates, (w_l, w_g)
        """
        r = p / p_r
        w_l = self.w_l_max * (1 - 0.2 * r - 0.8 * r ** 2)
        w_g = (self.f_g / (1 - self.f_g)) * w_l
        return w_l, w_g


@dataclass
class FixedFlowRate(InflowModel):
    """
    Fixed flow rate model:
        w_l = const,
        w_g = const,

    This model can be used to enforce specific flow rates, which can be useful when performing calibration.
    """

    w_l_const: float    # Constant liquid mass flow rate (kg/s)
    w_g_const: float    # Constant gas mass flow rate (kg/s)

    def __post_init__(self):
        assert self.w_l_const >= 0, 'Liquid mass flow rate must be non-negative'
        assert self.w_l_const >= 0, 'Gas mass flow rate must be non-negative'

    def mass_flow_rates(self, p, p_r):
        """
        Compute mass flow rates

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid and gas mass flow rates, (w_l, w_g)
        """

        return self.w_l_const, self.w_g_const


