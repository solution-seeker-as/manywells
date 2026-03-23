"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Unified fluid model for wellbore simulations.
"""

from dataclasses import dataclass

from manywells.pvt import (
    R_UNIVERSAL, P_REF, T_REF,
    api_from_density, gas_density_from_sg, sg_from_gas_density,
    water_viscosity, gas_viscosity as _gas_viscosity,
)
from manywells.pvt.gas import gas_z_factor
from manywells.pvt.black_oil import BlackOilPVT, live_oil_viscosity, live_oil_surface_tension
from manywells.pvt.dead_oil import dead_oil_viscosity, dead_oil_surface_tension
from manywells.units import M_AIR, CF_BAR, CF_RS
from manywells.ca_functions import ca_min_approx, ca_max_approx


@dataclass
class FluidModel:
    """
    Unified fluid model for two-phase wellbore flow.

    Parameterized by phase densities at standard conditions, gas-oil ratio,
    and water-liquid ratio.  The oil model ('black_oil' or 'dead_oil') controls
    whether pressure-dependent solution gas (Rs) and formation volume factor
    (Bo) are computed.  The gas model (ideal_gas flag) controls whether the
    z-factor correlation is used.

    All pressure arguments to methods are in bar; temperatures in Kelvin.

    Users who prefer API gravity or gas specific gravity can use the helpers
    ``density_from_api`` and ``gas_density_from_sg``::

        FluidModel(rho_o=density_from_api(35), rho_g=gas_density_from_sg(0.65))
    """

    # Phase densities at standard conditions (kg/m3)
    rho_o: float = 850.0
    rho_g: float = gas_density_from_sg(0.554)
    rho_w: float = 999.1

    # Volumetric ratios at standard conditions
    gor: float = 200.0          # Gas-oil ratio (Sm3/Sm3)
    wlr: float = 0.0           # Water-liquid ratio (also known as water cut), in [0, 1)

    # Model selection
    oil_model: str = 'black_oil'    # 'black_oil' or 'dead_oil'
    ideal_gas: bool = False         # True: z=1 (ideal gas law); False: Papay correlation

    # Separator / bubble point (used by black oil correlations)
    p_sep: float = P_REF       # Separator pressure (Pa)
    T_sep: float = T_REF       # Separator temperature (K)
    p_bubble: float = None     # Bubble point pressure (Pa), or None

    # Heat capacities (J/kg/K)
    cp_g: float = 2225.0
    cp_o: float = 2000.0
    cp_w: float = 4184.0

    def __post_init__(self):
        self._api = api_from_density(self.rho_o)
        self._sg_gas = sg_from_gas_density(self.rho_g)

        if self.oil_model == 'black_oil':
            self._black_oil = BlackOilPVT(
                api=self._api,
                sg_gas=self._sg_gas,
                p_sep=self.p_sep,
                T_sep=self.T_sep,
                p_bubble=self.p_bubble,
            )
        elif self.oil_model == 'dead_oil':
            self._black_oil = None
        else:
            raise ValueError(f"Unknown oil_model: {self.oil_model!r}")

    # ------------------------------------------------------------------
    # Derived properties
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
    def R_s(self) -> float:
        """Specific gas constant of the gas phase (J/(kg K))."""
        return R_UNIVERSAL / (M_AIR * self._sg_gas)

    @property
    def M_g(self) -> float:
        """Gas molecular weight (g/mol = kg/kmol)."""
        return M_AIR * self._sg_gas

    @property
    def rho_l(self) -> float:
        """Liquid density at standard conditions (kg/m3)."""
        return self.wlr * self.rho_w + (1 - self.wlr) * self.rho_o

    @property
    def cp_l(self) -> float:
        """Liquid specific heat capacity (J/kg/K), volume-weighted."""
        return self.wlr * self.cp_w + (1 - self.wlr) * self.cp_o

    @property
    def f_g(self) -> float:
        """Gas mass fraction at standard conditions."""
        if self.wlr >= 1.0:
            return 0.0
        denom = self.rho_g * self.gor + self.rho_o + self.rho_w * self.wlr / (1 - self.wlr)
        return (self.rho_g * self.gor) / denom

    @property
    def f_o_in_liquid(self) -> float:
        """Oil mass fraction in the liquid phase at standard conditions."""
        if self.rho_l == 0:
            return 0.0
        return (1 - self.wlr) * self.rho_o / self.rho_l

    @property
    def glr(self) -> float:
        """Gas-liquid ratio (Sm3/Sm3)."""
        return self.gor * (1 - self.wlr)

    # ------------------------------------------------------------------
    # Unified PVT methods
    # ------------------------------------------------------------------

    def rs(self, p, T):
        """
        Solution gas-oil ratio at (p, T).

        Returns 0 for dead oil; uses Vazquez-Beggs correlation for black oil.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Rs (Sm3 gas / Sm3 oil at standard conditions)
        """
        if self._black_oil is not None:
            return self._black_oil.rs(p * CF_BAR, T)
        return 0

    def bo(self, p, T):
        """
        Oil formation volume factor at (p, T).

        Returns 1.0 for dead oil; uses Vazquez-Beggs correlation for black oil.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Bo (dimensionless)
        """
        if self._black_oil is not None:
            return self._black_oil.bo(p * CF_BAR, T)
        return 1.0

    def z_factor(self, p, T):
        """
        Gas compressibility factor at (p, T).

        Returns 1.0 for ideal gas; uses Papay (1968) correlation otherwise.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Z-factor (dimensionless)
        """
        if self.ideal_gas:
            return 1.0
        return gas_z_factor(p * CF_BAR, T, self._sg_gas)

    def gas_mass_flow_rate(self, w_l):
        """Gas mass flow rate from liquid mass flow rate and gas fraction."""
        return (self.f_g / (1 - self.f_g)) * w_l

    def liquid_density(self, p, T):
        """
        Liquid density at (p, T).

        Uses the live-oil density correlation blended with water.
        For dead oil (Rs=0, Bo=1) this reduces to the constant ``rho_l``.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Liquid density (kg/m3)
        """
        Rs_i = self.rs(p, T)
        Bo_i = self.bo(p, T)
        rho_live_oil = (self.rho_o + Rs_i * self.rho_g) / Bo_i
        return self.wlr * self.rho_w + (1 - self.wlr) * rho_live_oil

    def liquid_viscosity(self, p, T):
        """
        Liquid mixture viscosity at (p, T) (CasADi-compatible).

        For dead oil, viscosity depends only on temperature.
        For black oil, the Beggs-Robinson live oil correction reduces
        viscosity to account for dissolved gas.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Liquid mixture viscosity (Pa-s)
        """
        mu_o = dead_oil_viscosity(self._api, T)
        if self._black_oil is not None:
            Rs_scf = self.rs(p, T) / CF_RS
            mu_o = live_oil_viscosity(mu_o, Rs_scf)
        mu_w = water_viscosity(T)
        return self.wlr * mu_w + (1 - self.wlr) * mu_o

    def surface_tension(self, p, T):
        """
        Oil-gas surface tension at (p, T) (CasADi-compatible).

        For dead oil, depends only on density and temperature.
        For black oil, the Abdul-Majeed correction reduces surface
        tension to account for dissolved gas.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Oil-gas surface tension (J/m2)
        """
        sigma = dead_oil_surface_tension(self.rho_o, T)
        if self._black_oil is not None:
            Rs_scf = self.rs(p, T) / CF_RS
            sigma = live_oil_surface_tension(sigma, Rs_scf)
        return sigma

    def gas_viscosity(self, T, rho_g):
        """Gas viscosity at (T, rho_g) (CasADi-compatible)."""
        return _gas_viscosity(T, rho_g, self.M_g)

    def free_gas_flux(self, Rs, w_g_total, w_lg, w_o):
        """
        Free (undissolved) gas mass flow rate (CasADi-compatible).

        :param Rs: Solution gas-oil ratio (Sm3/Sm3), may be CasADi symbolic
        :param w_g_total: Total gas mass flow from reservoir (kg/s)
        :param w_lg: Lift gas mass flow rate (kg/s)
        :param w_o: Stock-tank oil mass flow rate (kg/s)
        :return: Free gas mass flow rate (kg/s)
        """
        if self._black_oil is None:
            return w_g_total + w_lg
        w_dissolved = ca_min_approx(
            Rs * self.rho_g / self.rho_o * w_o,
            w_g_total,
        )
        return ca_max_approx(w_g_total + w_lg - w_dissolved, 0.0)

    def liquid_flux(self, Rs, w_l_inflow, w_o, w_g_total):
        """
        Total liquid-phase mass flow rate including dissolved gas (CasADi-compatible).

        :param Rs: Solution gas-oil ratio (Sm3/Sm3), may be CasADi symbolic
        :param w_l_inflow: Liquid inflow from reservoir (kg/s)
        :param w_o: Stock-tank oil mass flow rate (kg/s)
        :param w_g_total: Total gas mass flow from reservoir (kg/s)
        :return: Total liquid mass flow rate (kg/s)
        """
        if self._black_oil is None:
            return w_l_inflow
        w_dissolved = ca_min_approx(
            Rs * self.rho_g / self.rho_o * w_o,
            w_g_total,
        )
        return w_l_inflow + w_dissolved
