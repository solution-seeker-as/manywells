"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Fluid mixing functions for the black oil model (CasADi-compatible).

These functions compute the distribution of mass between free gas and liquid
phases, and the density of the liquid phase, given the local thermodynamic
state (p, T) and the black oil PVT correlations.
"""

from manywells.ca_functions import ca_min_approx, ca_max_approx


def dissolved_gas_mass_ratio(Rs, rho_g_sc, rho_o_sc):
    """
    Mass of dissolved gas per unit mass of stock-tank oil.

        r = Rs * rho_g_sc / rho_o_sc

    :param Rs: Solution gas-oil ratio (Sm3 gas / Sm3 oil at std conditions)
    :param rho_g_sc: Gas density at standard conditions (kg/m3)
    :param rho_o_sc: Oil density at standard conditions (kg/m3)
    :return: Dissolved gas mass ratio (kg gas / kg stock-tank oil)
    """
    return Rs * rho_g_sc / rho_o_sc


def liquid_density_bo(rho_o_sc, rho_g_sc, rho_w, Rs, Bo, wlr):
    """
    Density of the liquid phase (live oil + water) at reservoir conditions.

    The live oil density accounts for dissolved gas and volumetric expansion:
        rho_live_oil = (rho_o_sc + Rs * rho_g_sc) / Bo

    The mixed liquid density is the volume-weighted average of live oil and
    water densities. The water-to-liquid volume ratio at reservoir conditions
    is approximated using dead-oil densities (wlr), which is acceptable since
    Bo is close to 1 for typical conditions.

    :param rho_o_sc: Stock-tank oil density at standard conditions (kg/m3)
    :param rho_g_sc: Gas density at standard conditions (kg/m3)
    :param rho_w: Water density (kg/m3)
    :param Rs: Solution gas-oil ratio (Sm3/Sm3), may be CasADi symbolic
    :param Bo: Oil formation volume factor (dimensionless), may be CasADi symbolic
    :param wlr: Water-to-liquid volume ratio at standard conditions (dimensionless)
    :return: Liquid density (kg/m3)
    """
    rho_live_oil = (rho_o_sc + Rs * rho_g_sc) / Bo
    return wlr * rho_w + (1 - wlr) * rho_live_oil


def free_gas_flux(w_g_total, w_lg, w_o, Rs, rho_g_sc, rho_o_sc):
    """
    Free (undissolved) gas mass flow rate.

    Total gas = reservoir gas + lift gas.
    Dissolved gas = Rs * rho_g_sc / rho_o_sc * w_o  (capped at w_g_total).
    Free gas = total gas - dissolved gas.

    The smooth min/max approximations ensure the result is differentiable
    and that free gas is non-negative.

    :param w_g_total: Total gas mass flow from reservoir (kg/s), CasADi symbolic
    :param w_lg: Lift gas mass flow rate (kg/s)
    :param w_o: Stock-tank oil mass flow rate (kg/s), CasADi symbolic
    :param Rs: Solution gas-oil ratio (Sm3/Sm3), CasADi symbolic
    :param rho_g_sc: Gas density at standard conditions (kg/m3)
    :param rho_o_sc: Oil density at standard conditions (kg/m3)
    :return: Free gas mass flow rate (kg/s)
    """
    w_dissolved = ca_min_approx(
        dissolved_gas_mass_ratio(Rs, rho_g_sc, rho_o_sc) * w_o,
        w_g_total,
    )
    return ca_max_approx(w_g_total + w_lg - w_dissolved, 0.0)


def liquid_flux(w_l_inflow, w_o, w_g_total, Rs, rho_g_sc, rho_o_sc):
    """
    Total liquid-phase mass flow rate (oil + water + dissolved gas).

    :param w_l_inflow: Liquid inflow from reservoir, oil + water (kg/s), CasADi symbolic
    :param w_o: Stock-tank oil mass flow rate (kg/s), CasADi symbolic
    :param w_g_total: Total gas mass flow from reservoir (kg/s), CasADi symbolic
    :param Rs: Solution gas-oil ratio (Sm3/Sm3), CasADi symbolic
    :param rho_g_sc: Gas density at standard conditions (kg/m3)
    :param rho_o_sc: Oil density at standard conditions (kg/m3)
    :return: Total liquid mass flow rate (kg/s)
    """
    w_dissolved = ca_min_approx(
        dissolved_gas_mass_ratio(Rs, rho_g_sc, rho_o_sc) * w_o,
        w_g_total,
    )
    return w_l_inflow + w_dissolved
