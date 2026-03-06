"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Unified fluid model for dead oil and black oil simulations.
"""

from dataclasses import dataclass, field

from manywells.pvt import (
    R_UNIVERSAL, P_REF, T_REF, WATER,
    api_from_density, density_from_api,
    water_viscosity, gas_viscosity as _gas_viscosity,
    liquid_mixture_viscosity, molecular_weight,
)
from manywells.pvt.black_oil import BlackOilPVT, M_AIR
from manywells.pvt.dead_oil import dead_oil_viscosity
import manywells.pvt.fluid_mix as fluid_mix


@dataclass
class FluidModel:
    """
    Unified fluid model parameterized by standard petroleum engineering
    quantities: API gravity, gas specific gravity, gas-oil ratio, and water cut.

    All internal quantities (R_s, rho_o, f_g, wlr, rho_l, etc.) are derived
    automatically from these four inputs.

    When ``black_oil`` is ``None`` the model represents dead oil (no mass
    transfer between phases).
    """

    api: float = 35.0               # Oil API gravity (degrees)
    sg_gas: float = 0.554           # Gas specific gravity relative to air
    GOR: float = 200.0              # Producing gas-oil ratio (Sm3/Sm3 at standard conditions)
    water_cut: float = 0.0          # q_w / (q_w + q_o) at standard conditions, in [0, 1)
    cp_g: float = 2225.0            # Gas specific heat capacity (J/kg/K)
    cp_o: float = 2000.0            # Oil specific heat capacity (J/kg/K)
    rho_w: float = 999.1            # Water density at standard conditions (kg/m3)
    cp_w: float = 4184.0            # Water specific heat capacity (J/kg/K)
    black_oil: BlackOilPVT = None   # Optional; None = dead oil (no mass transfer)

    # ------------------------------------------------------------------
    # Derived properties (read-only)
    # ------------------------------------------------------------------

    @property
    def rho_o(self) -> float:
        """Oil density at standard conditions (kg/m3)."""
        return density_from_api(self.api)

    @property
    def R_s(self) -> float:
        """Specific gas constant (J/(kg K))."""
        return R_UNIVERSAL / (M_AIR * self.sg_gas)

    @property
    def M_g(self) -> float:
        """Gas molecular weight (g/mol = kg/kmol)."""
        return M_AIR * self.sg_gas

    @property
    def wlr(self) -> float:
        """Water-liquid volume ratio at standard conditions (= water_cut)."""
        return self.water_cut

    @property
    def rho_l(self) -> float:
        """Dead-oil liquid density at standard conditions (kg/m3)."""
        return self.wlr * self.rho_w + (1 - self.wlr) * self.rho_o

    @property
    def cp_l(self) -> float:
        """Liquid specific heat capacity (J/kg/K), volume-weighted."""
        return self.wlr * self.cp_w + (1 - self.wlr) * self.cp_o

    @property
    def f_g(self) -> float:
        """Gas mass fraction at standard conditions."""
        rho_g_std = P_REF / (self.R_s * T_REF)
        wc = self.water_cut
        if wc >= 1.0:
            return 0.0
        denominator = rho_g_std * self.GOR + self.rho_o + self.rho_w * wc / (1 - wc)
        return (rho_g_std * self.GOR) / denominator

    @property
    def f_o_in_liquid(self) -> float:
        """Oil mass fraction in the liquid phase."""
        if self.rho_l == 0:
            return 0.0
        return (1 - self.wlr) * self.rho_o / self.rho_l

    @property
    def GLR(self) -> float:
        """Gas-liquid ratio (Sm3/Sm3)."""
        return self.GOR * (1 - self.water_cut)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def gas_mass_flow_rate(self, w_l):
        """Gas mass flow rate from liquid mass flow rate and gas fraction."""
        return (self.f_g / (1 - self.f_g)) * w_l

    def liquid_density(self, p, T):
        """
        Liquid density at (p, T).

        For dead oil, returns the constant ``rho_l``.
        For black oil, uses the live-oil density correlation blended with water.
        """
        if self.black_oil is not None:
            from manywells.constants import CF_PRES
            bo = self.black_oil
            p_Pa = p * CF_PRES
            Rs_i = bo.rs(p_Pa, T)
            Bo_i = bo.bo(p_Pa, T)
            return fluid_mix.liquid_density_bo(
                bo.rho_o_sc, bo.rho_g_sc, self.rho_w, Rs_i, Bo_i, self.wlr,
            )
        return self.rho_l

    def liquid_viscosity(self, T):
        """Liquid mixture viscosity at temperature T (CasADi-compatible)."""
        mu_o = dead_oil_viscosity(self.api, T)
        mu_w = water_viscosity(T)
        return liquid_mixture_viscosity(mu_o, mu_w, self.wlr)

    def gas_viscosity(self, T, rho_g):
        """Gas viscosity at (T, rho_g) (CasADi-compatible)."""
        return _gas_viscosity(T, rho_g, self.M_g)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def default() -> "FluidModel":
        """Backward-compatible defaults matching current WellProperties."""
        return FluidModel(
            api=api_from_density(850),
            sg_gas=0.554,
            GOR=200.0,
            water_cut=0.0,
        )
