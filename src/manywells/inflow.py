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


class InflowModel(abc.ABC):
    """
    Abstract representation of an inflow model.

    An inflow model describes the reservoir's ability to deliver liquid at a
    given drawdown (p_r - p).  The gas-liquid split is a fluid property and is
    handled by FluidModel.gas_mass_flow_rate.
    """

    @abc.abstractmethod
    def liquid_mass_flow_rate(self, p, p_r):
        """
        Compute total liquid mass flow rate (oil + water) from the reservoir.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        pass


@dataclass
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

    k_l: float    # Liquid productivity index (kg/s/bar)

    def __post_init__(self):
        assert self.k_l >= 0, 'Liquid productivity index must be non-negative'

    def liquid_mass_flow_rate(self, p, p_r):
        """
        Compute liquid inflow rate.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        return self.k_l * (p_r - p)


@dataclass
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

    w_l_max: float    # Maximum liquid mass flow rate (kg/s)

    def __post_init__(self):
        assert self.w_l_max >= 0, 'Maximum liquid mass flow rate must be non-negative'

    def liquid_mass_flow_rate(self, p, p_r):
        """
        Compute liquid inflow rate.

        :param p: Bottomhole flowing pressure (bar)
        :param p_r: Reservoir pressure (bar)
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        r = p / p_r
        return self.w_l_max * (1 - 0.2 * r - 0.8 * r ** 2)


@dataclass
class FixedFlowRate(InflowModel):
    """
    Fixed flow rate model:
        w_l = w_l_const

    This model can be used to enforce specific flow rates, which can be useful
    when performing calibration.  An optional w_g_const is kept for cases where
    the caller needs a specific gas rate that may differ from what FluidModel
    would compute.
    """

    w_l_const: float    # Constant liquid mass flow rate (kg/s)
    w_g_const: float    # Constant gas mass flow rate (kg/s)

    def __post_init__(self):
        assert self.w_l_const >= 0, 'Liquid mass flow rate must be non-negative'
        assert self.w_g_const >= 0, 'Gas mass flow rate must be non-negative'

    def liquid_mass_flow_rate(self, p, p_r):
        """
        Return the fixed liquid mass flow rate.

        :param p: Bottomhole flowing pressure (bar) -- ignored
        :param p_r: Reservoir pressure (bar) -- ignored
        :return: Liquid mass flow rate, w_l (kg/s)
        """
        return self.w_l_const


