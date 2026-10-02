"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Unified fluid model for wellbore simulations.
"""

from dataclasses import dataclass

from manywells.ca_functions import ca_min_approx, ca_max_approx
from manywells.pvt import (
    R_UNIVERSAL, P_REF, T_REF,
    api_from_density, gas_density_from_sg, sg_from_gas_density, mixture_viscosity,
)
from manywells.pvt.gas import (DAK_ZC, dak_jt_factor, dak_reduced_density, dak_z_factor, gas_z_factor,
                               gas_viscosity as _gas_viscosity, sutton_pseudo_critical)
from manywells.pvt.black_oil import BlackOilPVT, live_oil_viscosity, live_oil_surface_tension
from manywells.pvt.dead_oil import dead_oil_viscosity, dead_oil_surface_tension
from manywells.pvt.water import water_viscosity
from manywells.units import M_AIR, CF_BAR, CF_RS


OIL_MODELS = ('black_oil', 'dead_oil')
Z_FACTOR_MODELS = ('dak', 'papay')
SURFACE_TENSION_MODELS = ('oil', 'liquid')


@dataclass(frozen=True)
class FluidModel:
    """
    Unified fluid model for three-phase wellbore flow (gas, oil, water).

    Parameterized by phase densities at standard conditions, gas-oil ratio,
    and water-liquid ratio.  The oil model ('black_oil' or 'dead_oil') controls
    whether pressure-dependent solution gas (Rs) and formation volume factor
    (Bo) are computed, and with it whether gas dissolves into the oil.  The gas
    is ideal (ideal_gas=True) or real, with the z-factor of the Dranchuk-Abou-Kassem
    equation of state ('dak') or of Papay's correlation ('papay').
    The surface tension model chooses the density the dead-oil correlation is
    evaluated at: the oil's at standard conditions with a live-oil correction
    ('oil'), or the local liquid density ('liquid', as in v1.0.0).

    All pressures, in fields and method arguments, are in bar; temperatures in Kelvin.

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
    wlr: float = 0.0            # Water-liquid ratio (also known as water cut), in [0, 1)

    # Model selection
    oil_model: str = 'black_oil'          # One of OIL_MODELS
    ideal_gas: bool = False               # True: z=1 (ideal gas law); False: the z-factor model's
    z_factor_model: str = 'dak'           # One of Z_FACTOR_MODELS, for a real gas
    surface_tension_model: str = 'oil'    # One of SURFACE_TENSION_MODELS

    # Separator / bubble point (used by black oil correlations)
    p_sep: float = P_REF / CF_BAR   # Separator pressure (bar)
    T_sep: float = T_REF            # Separator temperature (K)
    p_bubble: float = None          # Bubble point pressure (bar), or None

    # Heat capacities (J/kg/K)
    cp_g: float = 2225.0
    cp_o: float = 2000.0
    cp_w: float = 4184.0

    def __post_init__(self):
        if self.oil_model not in OIL_MODELS:
            raise ValueError(f"Unknown oil_model: {self.oil_model!r}")
        if self.surface_tension_model not in SURFACE_TENSION_MODELS:
            raise ValueError(f"Unknown surface_tension_model: {self.surface_tension_model!r}")
        if self.z_factor_model not in Z_FACTOR_MODELS:
            raise ValueError(f"Unknown z_factor_model: {self.z_factor_model!r}")
        if not 0 <= self.wlr < 1:
            raise ValueError('Water-liquid ratio must be in [0, 1)')

        object.__setattr__(self, '_api', api_from_density(self.rho_o))
        object.__setattr__(self, '_sg_gas', sg_from_gas_density(self.rho_g))
        object.__setattr__(self, '_pseudo_critical', sutton_pseudo_critical(self._sg_gas))  # (Pa, K)

        black_oil = None
        if self.oil_model == 'black_oil':
            black_oil = BlackOilPVT(
                api=self._api,
                sg_gas=self._sg_gas,
                p_sep=self.p_sep * CF_BAR,
                T_sep=self.T_sep,
                p_bubble=None if self.p_bubble is None else self.p_bubble * CF_BAR,
            )
        object.__setattr__(self, '_black_oil', black_oil)

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

    @property
    def rho_l(self) -> float:  # spec: PVT-MIX-10
        """Liquid density at standard conditions (kg/m3)."""
        return self.wlr * self.rho_w + (1 - self.wlr) * self.rho_o

    @property
    def cp_l(self) -> float:  # spec: PVT-MIX-10
        """Liquid specific heat capacity (J/kg/K), volume-weighted."""
        return self.wlr * self.cp_w + (1 - self.wlr) * self.cp_o

    @property
    def f_g(self) -> float:  # spec: PVT-MIX-10
        """Gas mass fraction at standard conditions."""
        denom = self.rho_g * self.gor + self.rho_o + self.rho_w * self.wlr / (1 - self.wlr)
        return (self.rho_g * self.gor) / denom

    @property
    def f_o_in_liquid(self) -> float:  # spec: PVT-MIX-10
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

    def rs(self, p, T):  # spec: PVT-OIL-4
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

    def bo(self, p, T):  # spec: PVT-OIL-4
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

    def _dak(self):
        return self.z_factor_model == 'dak' and not self.ideal_gas

    def reduced_density(self, rho_g):  # spec: PVT-GAS-9
        """DAK's reduced gas density Z_c rho_g R_s T_pc / p_pc at gas density rho_g (kg/m3), may be symbolic."""
        ppc, tpc = self._pseudo_critical
        return DAK_ZC * rho_g * self.R_s * tpc / ppc

    def z_factor(self, p, T):  # spec: PVT-GAS-4, PVT-GAS-9, PVT-GAS-11
        """
        Gas compressibility factor at (p, T).

        Returns 1.0 for ideal gas; otherwise the Dranchuk-Abou-Kassem (1975) equation of state at the density it
        gives at (p, T) ('dak'), or Papay's (1968) correlation ('papay').

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Z-factor (dimensionless)
        """
        if self.ideal_gas:
            return 1.0
        if self.z_factor_model == 'papay':
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
        if self._dak():
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
        if self._dak():
            Z = dak_z_factor(self.reduced_density(rho_g), T / self._pseudo_critical[1])
        else:
            Z = self.z_factor(p, T)
        return p - rho_g * Z * self.R_s * T / CF_BAR

    def jt_factor(self, T, rho_g):  # spec: PVT-GAS-10
        """
        The gas's Joule-Thomson factor J = T (d ln Z / dT)_p at temperature T and gas density rho_g: 0 for an ideal
        gas, and otherwise the Dranchuk-Abou-Kassem equation of state's at the reduced density of rho_g, whatever
        the z-factor model (specs/features/016-joule-thomson.md).

        :param T: Temperature (K), may be CasADi symbolic
        :param rho_g: Gas density (kg/m3), may be CasADi symbolic
        :return: J (dimensionless)
        """
        if self.ideal_gas:
            return 0.0
        return dak_jt_factor(self.reduced_density(rho_g), T / self._pseudo_critical[1])

    def liquid_density(self, p, T):  # spec: PVT-MIX-1, PVT-MIX-6, PVT-OIL-9
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

    def liquid_viscosity(self, p, T):  # spec: PVT-MIX-8
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

    def gas_viscosity(self, T, rho_g):
        """Gas viscosity at (T, rho_g) (CasADi-compatible)."""
        return _gas_viscosity(T, rho_g, self.M_g)

    def mixture_viscosity(self, p, T, alpha, rho_g, rho_l):
        """
        Gas-liquid mixture viscosity at a point (CasADi-compatible), mass-weighted.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :param alpha: Void fraction, may be CasADi symbolic
        :param rho_g: Gas density (kg/m3), may be CasADi symbolic
        :param rho_l: Liquid density (kg/m3), may be CasADi symbolic
        :return: Mixture viscosity (Pa-s)
        """
        mu_l = self.liquid_viscosity(p, T)
        mu_g = self.gas_viscosity(T, rho_g)
        return mixture_viscosity(mu_l, mu_g, alpha, rho_l=rho_l, rho_g=rho_g)

    def surface_tension(self, p, T, rho_l):
        """
        Gas-liquid surface tension at a point (CasADi-compatible).

        With surface_tension_model='oil', the dead-oil correlation at the oil
        density at standard conditions; for black oil, the Abdul-Majeed
        correction reduces it to account for dissolved gas.  With 'liquid',
        the dead-oil correlation at the local liquid density, as in v1.0.0.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :param rho_l: Liquid density at the point (kg/m3), may be CasADi symbolic
        :return: Gas-liquid surface tension (J/m2)
        """
        if self.surface_tension_model == 'liquid':
            return dead_oil_surface_tension(rho_l, T)  # spec: PVT-MIX-5
        sigma = dead_oil_surface_tension(self.rho_o, T)  # spec: PVT-MIX-7
        if self._black_oil is not None:
            Rs_scf = self.rs(p, T) / CF_RS
            sigma = live_oil_surface_tension(sigma, Rs_scf)
        return sigma

    def dissolved_gas(self, p, T, w_o):  # spec: PVT-OIL-13
        """Dissolved gas mass at (p, T) for a given oil mass flow rate w_o (kg/s)."""
        Rs = self.rs(p, T)  # Solution gas-oil ratio (Sm3/Sm3) - returns 0 for dead oil
        return Rs * self.rho_g / self.rho_o * w_o

    # ------------------------------------------------------------------
    # Phase mass rates
    # ------------------------------------------------------------------

    def reservoir_gas_rate(self, w_res):  # spec: INF-4
        """
        Gas mass flow rate from the reservoir, by the gas mass fraction at standard conditions.

        :param w_res: Liquid mass flow rate from the reservoir (kg/s), may be CasADi symbolic
        :return: Gas mass flow rate from the reservoir (kg/s)
        """
        return (self.f_g / (1 - self.f_g)) * w_res

    def phase_rates(self, p, T, w_res, w_lg):
        """
        Gas and liquid mass flow rates at (p, T), given the reservoir liquid rate and the lift gas rate.

        Dead oil has no mass transfer: the rates are the same at every point, exactly.  With black oil,
        gas dissolves into the oil up to the solution gas-oil ratio at (p, T).

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :param w_res: Liquid mass flow rate from the reservoir (kg/s), may be CasADi symbolic
        :param w_lg: Lift gas mass flow rate (kg/s), may be CasADi symbolic
        :return: Gas and liquid mass flow rates, (w_g, w_l) (kg/s)
        """
        w_g_res = self.reservoir_gas_rate(w_res)
        if self._black_oil is None:
            return w_g_res + w_lg, w_res  # spec: INF-5
        w_o = w_res * self.f_o_in_liquid  # Oil mass flow rate
        w_d = ca_min_approx(self.dissolved_gas(p, T, w_o), w_g_res)  # Dissolved gas mass flow rate
        return ca_max_approx(w_g_res + w_lg - w_d, 0.0), w_res + w_d  # spec: PVT-OIL-13
