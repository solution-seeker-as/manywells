"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Correlations for dead oil properties (CasADi-compatible), and the dead-oil PVT model, DeadOil
"""

from dataclasses import dataclass

import casadi as ca

from manywells.pvt import api_from_density, gas_density_from_sg
from manywells.pvt.model import CorrelationPVT
from manywells.units import kelvin_to_fahrenheit, kelvin_to_celsius, CF_CP, CF_DYNCM


def dead_oil_viscosity(api, T):  # spec: PVT-OIL-10
    """
    Beggs-Robinson (1975) correlation for dead oil viscosity.

    Reference: Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of
    Crude Oil Systems", J Pet Technol 27 (1975): 1140-1141.

    Fitted to oils of API 16 to 58 at 70 to 295 degF (21 to 146 degC) (Beggs and Robinson, Table 1). It
    extrapolates outside (specs/model/validity.md); the black oil refuses API gravities outside that range.

    :param api: Oil API gravity (dimensionless, float constant)
    :param T: Temperature (K), may be a CasADi symbolic
    :return: Dead oil viscosity (Pa-s)
    """
    T_F = kelvin_to_fahrenheit(T)
    y = ca.power(10, 3.0324 - 0.02023 * api)
    X = y * ca.constpow(T_F, -1.163)
    mu_cP = ca.power(10, X) - 1
    return mu_cP * CF_CP


def dead_oil_surface_tension(rho, T):  # spec: PVT-OIL-3
    """
    Correlation for dead oil surface tension.

    Reference: Abdul-Majeed, G.H. and Al-Soof, N.B.A., "Estimation of gas-oil
    surface tension", J Pet Sci Eng 27 (2000): 197-200.

    The correlation gives surface tension in dyn/cm, which can be converted
    to SI units as 1 dyn/cm = 0.001 J/m².

    :param rho: Density of oil (kg/m³)
    :param T: Temperature (K)
    :return: Surface tension of oil (J/m²)
    """
    T_degC = kelvin_to_celsius(T)
    api = api_from_density(rho)
    return CF_DYNCM * (1.11591 - 0.00305 * T_degC) * (38.085 - 0.259 * api)


@dataclass(frozen=True)
class DeadOil(CorrelationPVT):
    """
    Dead oil (PVT-OIL-1): incompressible, with no gas dissolved, so its phase rates are the same at every point
    (INF-5). The gas law is one of the correlations' (CorrelationPVT); v1.0.0's is the ideal gas.

    :param rho_o: Stock-tank oil density (kg/m3)
    :param rho_g: Gas density at standard conditions (kg/m3)
    :param gas_law: 'ideal', 'dak' or 'papay'
    """

    rho_o: float = 850.0
    rho_g: float = gas_density_from_sg(0.554)
    gas_law: str = 'dak'

    dissolves_gas = False

    def __post_init__(self):
        self._validate()
        self._derive()

    def rs(self, p, T):  # spec: PVT-OIL-4
        """No gas dissolves: 0."""
        return 0

    def bo(self, p, T):  # spec: PVT-OIL-4
        """The oil does not swell: 1."""
        return 1.0

    def oil_viscosity(self, p, T):  # spec: PVT-OIL-10
        """Beggs-Robinson's dead-oil viscosity at T (Pa s)."""
        return dead_oil_viscosity(self._api, T)

    def oil_surface_tension(self, p, T):  # spec: PVT-OIL-3
        """The dead-oil surface tension at the stock-tank oil's density and T (J/m2)."""
        return dead_oil_surface_tension(self.rho_o, T)
