"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

The interface of a fluid's hydrocarbon PVT, which FluidModel takes as its component pvt (specs/features/018-pvt-tables.md):
the oil and the gas at standard conditions, the gas the oil dissolves, the gas law, and the oil's and the gas's
viscosity and surface tension. One class per option: DeadOil (pvt/dead_oil.py) and BlackOil (pvt/black_oil.py),
which share the correlations' gas law (CorrelationPVT).
"""

import math
from abc import ABC, abstractmethod

from manywells.pvt import R_UNIVERSAL, api_from_density, sg_from_gas_density
from manywells.pvt.gas import (
    DAK_ZC,
    dak_jt_factor,
    dak_reduced_density,
    dak_z_factor,
    gas_z_factor,
    gas_viscosity as _gas_viscosity,
    sutton_pseudo_critical,
)
from manywells.units import M_AIR, CF_BAR

GAS_LAWS = ('ideal', 'dak', 'papay')


class PVTModel(ABC):
    """
    The hydrocarbon PVT of a fluid. A subclass is a frozen dataclass with the fields rho_o and rho_g, the stock-tank
    oil density and the gas density at standard conditions (kg/m3), and gas_law, the name of its gas law; its
    __post_init__ calls _validate and _derive. dissolves_gas says whether gas dissolves into its oil: where it does
    not, FluidModel's phase rates are INF-5's, exactly.

    Every method takes pressures in bar and temperatures in kelvin, and accepts CasADi symbols.
    """

    dissolves_gas = True

    def _validate(self):
        for name in ('rho_o', 'rho_g'):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f'{name} must be positive and finite, got {getattr(self, name)}')

    def _derive(self):
        """The constants derived from the densities at standard conditions, once."""
        object.__setattr__(self, '_api', api_from_density(self.rho_o))
        object.__setattr__(self, '_sg_gas', sg_from_gas_density(self.rho_g))
        object.__setattr__(self, '_pseudo_critical', sutton_pseudo_critical(self._sg_gas))  # (Pa, K)

    def at_gor(self, gor) -> 'PVTModel':
        """The PVT model a fluid of gas-oil ratio gor (Sm3/Sm3) uses: itself, but for a table's GOR axis (PVT-TAB-7)."""
        return self

    def covers(self, p, T) -> bool:
        """Whether (p, T), floats, lies where the model is defined: everywhere, but for a table's grid."""
        return True

    # ------------------------------------------------------------------
    # The oil and the gas at standard conditions
    # ------------------------------------------------------------------

    @property
    def api(self) -> float:
        """Oil API gravity (degrees)."""
        return self._api

    @property
    def sg_gas(self) -> float:
        """Gas specific gravity relative to air (dimensionless)."""
        return self._sg_gas

    @property
    def pseudo_critical(self) -> tuple:  # spec: PVT-GAS-5
        """Pseudo-critical pressure (Pa) and temperature (K) of the gas, from its specific gravity (Sutton, 1985)."""
        return self._pseudo_critical

    @property
    def R_s(self) -> float:  # spec: PVT-GAS-6
        """Specific gas constant of the gas phase (J/(kg K)). NOTE: Easily confused with the solution gas-oil ratio (Rs)"""
        return R_UNIVERSAL / (M_AIR * self._sg_gas)

    @property
    def M_g(self) -> float:  # spec: PVT-GAS-6
        """Gas molecular weight (g/mol = kg/kmol)."""
        return M_AIR * self._sg_gas

    # ------------------------------------------------------------------
    # The oil
    # ------------------------------------------------------------------

    @abstractmethod
    def rs(self, p, T):
        """Solution gas-oil ratio of the saturated oil at (p, T) (Sm3 gas / Sm3 oil at standard conditions)."""

    @abstractmethod
    def bo(self, p, T):
        """Oil formation volume factor at (p, T) (dimensionless)."""

    @abstractmethod
    def oil_viscosity(self, p, T):
        """Oil viscosity at (p, T) (Pa s)."""

    @abstractmethod
    def oil_surface_tension(self, p, T):
        """Gas-oil surface tension at (p, T) (J/m2)."""

    # ------------------------------------------------------------------
    # The gas
    # ------------------------------------------------------------------

    @abstractmethod
    def z_factor(self, p, T):
        """Gas compressibility factor at (p, T) (dimensionless)."""

    @abstractmethod
    def gas_density(self, p, T):
        """Gas density at (p, T) (kg/m3)."""

    @abstractmethod
    def gas_law_row(self, p, T, rho_g):
        """The gas law as a row of the discretized system (bar), zero where rho_g is the gas density at (p, T)."""

    def reduced_density(self, rho_g):  # spec: PVT-GAS-9
        """DAK's reduced gas density Z_c rho_g R_s T_pc / p_pc at gas density rho_g (kg/m3), may be symbolic."""
        ppc, tpc = self._pseudo_critical
        return DAK_ZC * rho_g * self.R_s * tpc / ppc

    def jt_factor(self, T, rho_g):  # spec: PVT-GAS-10
        """
        The gas's Joule-Thomson factor J = T (d ln Z / dT)_p at temperature T and gas density rho_g: 0 for an ideal
        gas, and otherwise the Dranchuk-Abou-Kassem equation of state's at the reduced density of rho_g, whatever
        the gas law (specs/features/016-joule-thomson.md).

        :param T: Temperature (K), may be CasADi symbolic
        :param rho_g: Gas density (kg/m3), may be CasADi symbolic
        :return: J (dimensionless)
        """
        if self.gas_law == 'ideal':
            return 0.0
        return dak_jt_factor(self.reduced_density(rho_g), T / self._pseudo_critical[1])

    def gas_viscosity(self, p, T, rho_g):  # spec: PVT-GAS-7
        """Gas viscosity at (p, T) and gas density rho_g (Pa s), by Lee-Gonzalez-Eakin, which needs no pressure."""
        return _gas_viscosity(T, rho_g, self.M_g)


class CorrelationPVT(PVTModel):
    """
    The correlations' gas law, from the gas's gravity at standard conditions: an ideal gas ('ideal', PVT-GAS-1), the
    Dranchuk-Abou-Kassem equation of state ('dak', PVT-GAS-11) or Papay's z-factor ('papay', PVT-GAS-3, PVT-GAS-4).
    """

    def _validate(self):
        super()._validate()
        if self.gas_law not in GAS_LAWS:
            raise ValueError(f'Unknown gas_law: {self.gas_law!r}, expected one of {GAS_LAWS}')

    def z_factor(self, p, T):  # spec: PVT-GAS-4, PVT-GAS-9, PVT-GAS-11
        """
        Gas compressibility factor at (p, T).

        Returns 1.0 for ideal gas; otherwise the Dranchuk-Abou-Kassem (1975) equation of state at the density it
        gives at (p, T) ('dak'), or Papay's (1968) correlation ('papay').

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Z-factor (dimensionless)
        """
        if self.gas_law == 'ideal':
            return 1.0
        if self.gas_law == 'papay':
            return gas_z_factor(p * CF_BAR, T, self._sg_gas)
        ppc, tpc = self._pseudo_critical
        return dak_z_factor(dak_reduced_density(p * CF_BAR / ppc, T / tpc), T / tpc)

    def gas_density(self, p, T):  # spec: PVT-GAS-1, PVT-GAS-3, PVT-GAS-11
        """
        Gas density at (p, T) from the real gas equation of state.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Gas density (kg/m3)
        """
        if self.gas_law == 'dak':
            ppc, tpc = self._pseudo_critical
            return dak_reduced_density(p * CF_BAR / ppc, T / tpc) * ppc / (DAK_ZC * self.R_s * tpc)
        Z = self.z_factor(p, T)
        return CF_BAR * p / (Z * self.R_s * T)

    def gas_law_row(self, p, T, rho_g):  # spec: PVT-GAS-1, PVT-GAS-3, PVT-GAS-11
        """
        The gas law as a row of the discretized system, in its canonical form p - rho_g Z R_s T / c_bar (bar),
        zero where rho_g is the gas density at (p, T). With DAK, Z is the equation of state's at rho_g and T.

        The form matters to the solver, not to the roots: in bar, like the momentum row, it lets Ipopt converge in
        fewer iterations and more tightly than the density form rho_g - gas_density(p, T).

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :param rho_g: Gas density (kg/m3), may be CasADi symbolic
        :return: Row value (bar)
        """
        if self.gas_law == 'dak':
            Z = dak_z_factor(self.reduced_density(rho_g), T / self._pseudo_critical[1])
        else:
            Z = self.z_factor(p, T)
        return p - rho_g * Z * self.R_s * T / CF_BAR
