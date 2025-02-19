"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 09 February 2024
Bjarne Grimstad, bjarne.grimstad@gmail.no

Utilities for doing various PVT (Pressure/Volume/Temperature) calculations
"""

from dataclasses import dataclass


################################################
# STANDARD REFERENCE CONDITIONS
################################################

# Standard reference conditions per ISO 13443 (15 degC and 1 atm = 101.325 kPa)
P_REF = 101_325  # Reference pressure (Pa)
T_REF = 288.15   # Reference temperature (K)


################################################
# FLUIDS
################################################

@dataclass
class LiquidProperties:
    name: str       # Name of liquid
    rho: float      # Density (kg/m³)
    cp: float       # Specific heat capacity (J/kg/K) - note that this is a temperature dependent property


@dataclass
class GasProperties:
    name: str       # Name of gas
    R_s: float      # Specific gas constant (J/kg/K)
    cp: float       # Specific heat capacity (J/kg/K) - note that this is a temperature dependent property


# Common fluids
WATER = LiquidProperties(name='water', rho=999.1, cp=4184)  # Pure water
SEAWATER = LiquidProperties(name='seawater', rho=1025, cp=4000)  # Seawater (salinity = 3.5%)
NORTH_SEA_BRENT_CRUDE = LiquidProperties(name='north-sea-brent-crude', rho=826, cp=2000)  # API = 39.8
METHANE = GasProperties(name='methane', R_s=518.3, cp=2225)


################################################
# COMPUTATION AND CONVERSION METHODS
################################################

def specific_gas_constant(rho):
    """
    Compute the specific gas constant, denoted R_s with unit J / (kg K)
    :param rho: Gas density at standard conditions
    :return: specific gas constant
    """
    return P_REF / (rho * T_REF)


def gas_density(R_s: float, p: float = P_REF, T: float = T_REF):
    """
    Compute gas density using ideal gas law:
        density = p / (R_s T),
    where R_s is the specific gas constant.

    :param R_s: Specific gas constant (J/kg/K)
    :param p: Pressure (Pa)
    :param T: Temperature (K)
    :return: density (kg/m³)
    """
    return p / (R_s * T)


def liquid_mix(liquid_1: LiquidProperties, liquid_2: LiquidProperties, mass_fraction: float):
    """
    Mix two liquids based on mass fraction. The mixture properties are computed from the volume fraction.

    :param liquid_1: Liquid 1
    :param liquid_2: Liquid 2
    :param mass_fraction: Mass fraction of liquid 1, computed as mass_fraction = m1 / (m1 + m2),
                          where m1 and m2 is the mass of liquid 1 and 2. The mass fraction must be in [0, 1].
    :return: Mixed liquid
    """
    assert 0 <= mass_fraction <= 1, 'Mass fraction must be in [0, 1]'
    if mass_fraction == 1:
        return liquid_1
    elif mass_fraction == 0:
        return liquid_2
    else:
        vol_fraction = 1 / (1 + (liquid_1.rho / liquid_2.rho) * (1 / mass_fraction - 1))  # Water-liquid volume fraction
        mix_rho = vol_fraction * liquid_1.rho + (1 - vol_fraction) * liquid_2.rho  # Density of liquid mixture
        mix_cp = vol_fraction * liquid_1.cp + (1 - vol_fraction) * liquid_2.cp  # Specific heat capacity of mixture
        mix_name = f'{100*mass_fraction:.1f}% of {liquid_1.name} + {100*(1 - mass_fraction):.1f}% of {liquid_2.name}'
        return LiquidProperties(name=mix_name, rho=mix_rho, cp=mix_cp)


def water_liquid_ratio(rho_l, rho_o, rho_w):
    """
    Compute water to liquid volumetric ratio from densities, assuming no-slip:
        rho_l = (1 - wlr) * rho_o + wlr * rho_w
        => wlr = (rho_l - rho_o) / (rho_w - rho_o)
    :return: water-liquid-ratio (wlr)
    """
    return (rho_l - rho_o) / (rho_w - rho_o)


def api_from_density(rho):
    """
    Compute API gravity from density
    :param rho: Oil density at standard reference conditions
    :return: API gravity of oil
    """
    sg = rho / WATER.rho
    return 141.5 / sg - 131.5


def density_from_api(api):
    """
    Compute oil density at standard reference conditions from API gravity

    :param api: API gravity of oil
    :return: Oil density at standard reference conditions
    """
    return 141.5 * WATER.rho / (api + 131.5)


def dead_oil_surface_tension(rho, T):
    """
    Correlation for dead oil surface tension from paper "Estimation of gas–oil surface tension" by Abdul-Majeed & Al-Soof (2000)

    The correlation gives surface tension in dyn/cm, which can be converted to SI units as 1 dyn/cm = 0.001 J/m².

    :param rho: Density of oil (kg/m³)
    :param T: Temperature (K)
    :return: Surface tension of oil (J/m²)
    """
    cf = 0.001  # Unit conversion factor (1 dyn/cm = 0.001 J/m²)
    T_degC = T - 273.15  # From kelvin to degC
    api = api_from_density(rho)
    return cf * (1.11591 - 0.00305 * T_degC) * (38.085 - 0.259 * api)


"""
Petroleum liquids (oils) can be categorized into three types:

Type                API gravity         Density (kg/m^3)
---------------------------------------------------------
Heavy               14-22               973-922
Intermediate        22-31               922-871
Light               31-40               871-825
---------------------------------------------------------

From Wikipedia: 
    "The API gravity is a measure of how heavy or light a petroleum liquid is compared to water: 
     if its API gravity is greater than 10, it is lighter and floats on water; 
     if less than 10, it is heavier and sinks."

API gravity is computed as follows:
    API = 141.5 / SG - 131.5,
where SG is the specific gravity of the oil; SG = density of oil / density of water, computed at standard conditions.

Given the API gravity, we can compute the density of the oil (at standard conditions):
    density of oil = 141.5 * (density of water) / (API + 131.5) 

Example:
    API = 40
    density of oil ~= 141.5 * 1000 / (40 + 131.5) ~= 825 kg/m^3 at standard conditions

North Sea Brent Crude is a sweet and light crude oil with an API gravity of 39.8

In the United States, most of the produced oil has an API gravity above 30.
"""


if __name__ == '__main__':

    rho_methane = 0.6798  # kg / m^3 at standard conditions
    # rho_methane = 0.6785  # kg / m^3
    R_s_methane = specific_gas_constant(rho_methane)
    print('Specific gas constant of methane:', R_s_methane)  # Expecting 518.3

    rho_eg = 1.1
    R_s_eg = specific_gas_constant(rho_eg)
    print(R_s_eg)

    import numpy as np
    T_eg = 353  # K
    v_sound_eg = np.sqrt(R_s_eg * T_eg)
    print(v_sound_eg)

    # # Specific heat
    # def specific_heat_capacity(rho):
    #     return 25389 / (rho ** 0.39)
    #
    # print(specific_heat_capacity(1000))
    # print(specific_heat_capacity(826))
    # print(specific_heat_capacity(920))

