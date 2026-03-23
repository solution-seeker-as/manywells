"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 09 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Utilities for doing various PVT (Pressure/Volume/Temperature) calculations
"""

from dataclasses import dataclass
import casadi as ca
from manywells.units import R_UNIVERSAL, P_REF, T_REF


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

North Sea Brent Crude is a sweet and light crude oil with an API gravity of 39.8

In the United States, most of the produced oil has an API gravity above 30.

See api_from_density() and density_from_api() for more details on how to convert between API gravity and density.
"""


################################################
# LIQUID UTILITIES AND API CONVERSIONS
################################################

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

    API gravity is computed as follows:
        API = 141.5 / SG - 131.5,
    where SG is the specific gravity of the oil; SG = density of oil / density of water, computed at standard reference conditions.

    :param rho: Oil density at standard reference conditions
    :return: API gravity of oil
    """
    sg = rho / WATER.rho
    return 141.5 / sg - 131.5


def density_from_api(api):
    """
    Compute oil density at standard reference conditions from API gravity

    Given the API gravity, we can compute the density of the oil (at standard conditions):
        density of oil = 141.5 * (density of water) / (API + 131.5) 

    Example:
        API = 40
        density of oil ~= 141.5 * 1000 / (40 + 131.5) ~= 825 kg/m^3 at standard conditions

    :param api: API gravity of oil
    :return: Oil density at standard reference conditions
    """
    return 141.5 * WATER.rho / (api + 131.5)


################################################
# MIXTURE VISCOSITY (general mixing rules)
################################################

def liquid_mixture_viscosity(mu_o, mu_w, wlr):
    """
    Volume-weighted arithmetic mean viscosity for oil-water liquid mixture.

    :param mu_o: Oil viscosity (Pa·s)
    :param mu_w: Water viscosity (Pa·s)
    :param wlr: Water-to-liquid volume fraction in [0, 1]
    :return: Liquid mixture viscosity (Pa·s)
    """
    return wlr * mu_w + (1 - wlr) * mu_o


def mixture_viscosity(mu_l, mu_g, alpha, rho_l=None, rho_g=None, method='mass_weighted'):
    """
    Gas-liquid mixture viscosity. CasADi-compatible.

    Supported methods:
      - 'mass_weighted': Mass-weighted arithmetic mean (Hasan, Kabir &
        Sayarpour 2010, Eq. A-3).  Requires ``rho_l`` and ``rho_g``.
      - 'arithmetic': Volume-weighted arithmetic mean (Dukler et al. 1964).
      - 'geometric':  Volume-weighted geometric mean (Arrhenius 1887).

    :param mu_l: Liquid viscosity (Pa·s)
    :param mu_g: Gas viscosity (Pa·s)
    :param alpha: Gas void fraction in [0, 1]
    :param rho_l: Liquid density (kg/m³), required for 'mass_weighted'
    :param rho_g: Gas density (kg/m³), required for 'mass_weighted'
    :param method: Mixing rule ('mass_weighted', 'arithmetic', or 'geometric')
    :return: Mixture viscosity (Pa·s)
    """
    if method == 'mass_weighted':
        if rho_l is None or rho_g is None:
            raise ValueError("rho_l and rho_g are required for 'mass_weighted' method")
        x = alpha * rho_g / (alpha * rho_g + (1 - alpha) * rho_l)
        return mu_g * x + mu_l * (1 - x)
    elif method == 'arithmetic':
        return alpha * mu_g + (1 - alpha) * mu_l
    elif method == 'geometric':
        return ca.exp(alpha * ca.log(mu_g) + (1 - alpha) * ca.log(mu_l))
    else:
        raise ValueError(f"Unknown mixture viscosity method: {method}")


################################################
# RE-EXPORTS from submodules
################################################

from manywells.pvt.gas import (  # noqa: E402, F401
    specific_gas_constant, gas_density, gas_density_std,
    gas_fvf, molecular_weight, gas_viscosity,
)
from manywells.pvt.water import water_fvf, water_viscosity  # noqa: E402, F401
from manywells.pvt.dead_oil import dead_oil_viscosity, dead_oil_surface_tension  # noqa: E402, F401
from manywells.pvt.black_oil import live_oil_viscosity  # noqa: E402, F401
