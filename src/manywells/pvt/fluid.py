"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Unified fluid model for wellbore simulations.
"""

import math
from dataclasses import dataclass

import numpy as np

from manywells.ca_functions import ca_min_approx, ca_max_approx
from manywells.pvt import mixture_viscosity
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.pvt.model import PVTModel
from manywells.pvt.water import water_viscosity


SURFACE_TENSION_MODELS = ('oil', 'liquid')


@dataclass(frozen=True)
class FluidModel:
    """
    Unified fluid model for three-phase wellbore flow (gas, oil, water).

    The hydrocarbon PVT is the component pvt, a PVTModel: DeadOil (no gas dissolves) or BlackOil (Vazquez-Beggs,
    with gas dissolving into the oil), each with its gas law ('ideal', 'dak' or 'papay'), or PVTTable (a table on a
    (p, T) grid; a PVTTableSet's tables at several GORs are read at gor). FluidModel holds the
    stream: the water, the gas-oil and water-liquid ratios at standard conditions and the heat capacities, and
    mixes oil and water into one liquid. The surface tension model chooses the density the dead-oil correlation is
    evaluated at: the oil's at standard conditions with a live-oil correction ('oil'), or the local liquid density
    ('liquid', as in v1.0.0).

    All pressures, in fields and method arguments, are in bar; temperatures in Kelvin.

    Users who prefer API gravity or gas specific gravity can use the helpers
    ``density_from_api`` and ``gas_density_from_sg``::

        FluidModel(pvt=BlackOil(rho_o=density_from_api(35), rho_g=gas_density_from_sg(0.65)))
    """

    pvt: PVTModel = BlackOil()  # The hydrocarbon PVT, with the oil and gas densities at standard conditions

    # Water density at standard conditions (kg/m3)
    rho_w: float = 999.1

    # Volumetric ratios at standard conditions
    gor: float = 200.0          # Gas-oil ratio (Sm3/Sm3)
    wlr: float = 0.0            # Water-liquid ratio (also known as water cut), in [0, 1)

    surface_tension_model: str = 'oil'    # One of SURFACE_TENSION_MODELS

    # Heat capacities (J/kg/K)
    cp_g: float = 2225.0
    cp_o: float = 2000.0
    cp_w: float = 4184.0

    def __post_init__(self):
        if not isinstance(self.pvt, PVTModel):
            raise ValueError(f'pvt must be a PVTModel, such as BlackOil() or DeadOil(), got {self.pvt!r}')
        if self.surface_tension_model not in SURFACE_TENSION_MODELS:
            raise ValueError(f"Unknown surface_tension_model: {self.surface_tension_model!r}")
        if not 0 <= self.wlr < 1:
            raise ValueError('Water-liquid ratio must be in [0, 1)')
        for name in ('rho_w', 'cp_g', 'cp_o', 'cp_w'):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f'{name} must be positive and finite, got {getattr(self, name)}')
        if not (math.isfinite(self.gor) and self.gor >= 0):
            raise ValueError(f'gor must be non-negative and finite, got {self.gor}')
        object.__setattr__(self, '_hc', self.pvt.at_gor(self.gor))  # The PVT model at this GOR (PVT-TAB-7)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def rho_o(self) -> float:
        """Stock-tank oil density (kg/m3), the PVT model's."""
        return self._hc.rho_o

    @property
    def rho_g(self) -> float:
        """Gas density at standard conditions (kg/m3), the PVT model's."""
        return self._hc.rho_g

    @property
    def api(self) -> float:
        """Oil API gravity (degrees)."""
        return self._hc.api

    @property
    def sg_gas(self) -> float:
        """Gas specific gravity relative to air (dimensionless)."""
        return self._hc.sg_gas

    @property
    def pseudo_critical(self) -> tuple:
        """Pseudo-critical pressure (Pa) and temperature (K) of the gas (PVT-GAS-5)."""
        return self._hc.pseudo_critical

    @property
    def R_s(self) -> float:
        """Specific gas constant of the gas phase (J/(kg K)) (PVT-GAS-6). NOTE: Easily confused with the solution
        gas-oil ratio (Rs)"""
        return self._hc.R_s

    @property
    def M_g(self) -> float:
        """Gas molecular weight (g/mol = kg/kmol) (PVT-GAS-6)."""
        return self._hc.M_g

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
    # The PVT model's properties at (p, T), may be CasADi symbolic: bar and K
    # ------------------------------------------------------------------

    def rs(self, p, T):
        """Solution gas-oil ratio at (p, T) (Sm3 gas / Sm3 oil at standard conditions): 0 for dead oil."""
        return self._hc.rs(p, T)

    def bo(self, p, T):
        """Oil formation volume factor at (p, T) (dimensionless): 1 for dead oil."""
        return self._hc.bo(p, T)

    def reduced_density(self, rho_g):
        """DAK's reduced gas density at gas density rho_g (kg/m3) (PVT-GAS-9)."""
        return self._hc.reduced_density(rho_g)

    def z_factor(self, p, T):
        """Gas compressibility factor at (p, T) (dimensionless)."""
        return self._hc.z_factor(p, T)

    def gas_density(self, p, T):
        """Gas density at (p, T) (kg/m3)."""
        return self._hc.gas_density(p, T)

    def gas_law_row(self, p, T, rho_g):
        """The gas law as a row of the discretized system (bar), zero where rho_g is the gas density at (p, T)."""
        return self._hc.gas_law_row(p, T, rho_g)

    def covers(self, p, T) -> bool:
        """Whether (p, T), floats, lies on the PVT model's table: always, without a table."""
        return self._hc.covers(p, T)

    @property
    def table_range(self) -> str:
        """The range of the PVT model's table, for messages; empty without a table."""
        return getattr(self._hc, 'grid_range', '')

    def off_table(self, X) -> bool:
        """
        Whether any point of a state lies off the PVT model's table (specs/features/018-pvt-tables.md, ruling 3).

        :param X: The state, one row per point in state-vector order (bar, ..., K)
        """
        X = np.reshape(X, (-1, 7))
        return any(not self.covers(float(p), float(T)) for p, T in zip(X[:, 0], X[:, 6]))

    def jt_factor(self, T, rho_g):
        """The gas's Joule-Thomson factor J = T (d ln Z / dT)_p at T (K) and gas density rho_g (kg/m3) (PVT-GAS-10)."""
        return self._hc.jt_factor(T, rho_g)

    def liquid_fvf(self, p, T):  # spec: PVT-MIX-6
        """
        Liquid formation volume factor at (p, T): the volume (m3) there of one Sm3 of stock-tank liquid, the oil
        swollen by Bo and the water incompressible (B_w = 1). Exactly 1 for dead oil.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: B_l (dimensionless)
        """
        return (1 - self.wlr) * self.bo(p, T) + self.wlr

    def liquid_density(self, p, T):  # spec: PVT-MIX-1, PVT-MIX-6, PVT-OIL-9
        """
        Liquid density at (p, T): the mass of one Sm3 of stock-tank liquid with the gas dissolved in its oil, over
        its volume at (p, T), so that oil and water mix by their in-situ volumes.
        For dead oil (Rs=0, Bo=1) this reduces to the constant ``rho_l``, exactly.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Liquid density (kg/m3)
        """
        Rs_i = self.rs(p, T)
        return ((1 - self.wlr) * (self.rho_o + Rs_i * self.rho_g) + self.wlr * self.rho_w) / self.liquid_fvf(p, T)

    def liquid_viscosity(self, p, T):  # spec: PVT-MIX-8
        """
        Liquid mixture viscosity at (p, T) (CasADi-compatible).

        The oil's viscosity is the PVT model's: for dead oil it depends only on
        temperature; for black oil the Beggs-Robinson live oil correction reduces
        it to account for dissolved gas.  Oil and water are weighted
        by their volume fractions at (p, T), as in the liquid density.

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Liquid mixture viscosity (Pa-s)
        """
        mu_o = self._hc.oil_viscosity(p, T)
        mu_w = water_viscosity(T)
        lambda_w = self.wlr / self.liquid_fvf(p, T)  # In-situ water fraction of the liquid (PVT-MIX-6)
        return lambda_w * mu_w + (1 - lambda_w) * mu_o

    def gas_viscosity(self, p, T, rho_g):
        """Gas viscosity at (p, T) and gas density rho_g (Pa s) (CasADi-compatible)."""
        return self._hc.gas_viscosity(p, T, rho_g)

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
        mu_g = self.gas_viscosity(p, T, rho_g)
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
        return self._hc.oil_surface_tension(p, T)  # spec: PVT-MIX-7

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

        Where the PVT model dissolves no gas (dead oil) there is no mass transfer: the rates are the same at every
        point, exactly.  Otherwise gas dissolves into the oil up to the solution gas-oil ratio at (p, T).

        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :param w_res: Liquid mass flow rate from the reservoir (kg/s), may be CasADi symbolic
        :param w_lg: Lift gas mass flow rate (kg/s), may be CasADi symbolic
        :return: Gas and liquid mass flow rates, (w_g, w_l) (kg/s)
        """
        w_g_res = self.reservoir_gas_rate(w_res)
        if not self._hc.dissolves_gas:
            return w_g_res + w_lg, w_res  # spec: INF-5
        w_o = w_res * self.f_o_in_liquid  # Oil mass flow rate
        w_d = ca_min_approx(self.dissolved_gas(p, T, w_o), w_g_res)  # Dissolved gas mass flow rate
        return ca_max_approx(w_g_res + w_lg - w_d, 0.0), w_res + w_d  # spec: PVT-OIL-13
