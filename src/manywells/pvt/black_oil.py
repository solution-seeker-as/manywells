"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Black oil PVT model using Vazquez-Beggs correlations (CasADi-compatible, SI units).

Reference: Vasquez, M. and Beggs, H.D., "Correlations for Fluid Physical
Property Prediction", J Pet Technol 32 (1980): 968-970.
"""

from dataclasses import dataclass
from math import log as math_log

import casadi as ca

from manywells.pvt import P_REF, T_REF, R_UNIVERSAL, api_from_density, density_from_api
from manywells.ca_functions import ca_min_approx

# Unit conversion constants
CF_PSI = 6894.76    # Pa per psi
CF_RS = 0.178108    # scf/STB to Sm3/Sm3
M_AIR = 28.97       # Molecular weight of air (g/mol = kg/kmol)


@dataclass
class BlackOilPVT:
    """
    Black oil PVT model with Vazquez-Beggs correlations for Rs and Bo.

    All constructor parameters are plain floats (resolved at well setup time).
    Methods rs(), bo(), live_oil_density() accept CasADi symbolic pressure
    and temperature for use inside the NLP solver.

    Inputs and outputs use SI units. Internally, the correlations are evaluated
    in field units (psia, degF, scf/STB) with conversions at the boundaries.
    """
    api: float              # Oil API gravity (degrees, valid range 10-40)
    sg_gas: float           # Gas specific gravity relative to air (dimensionless)
    p_sep: float            # Separator pressure (Pa)
    T_sep: float            # Separator temperature (K)
    p_bubble: float = None  # Bubble point pressure (Pa). None = no Rs cap.

    def __post_init__(self):
        if not (10 <= self.api <= 40):
            raise ValueError(f"Oil API {self.api} outside valid range 10-40")
        if self.sg_gas <= 0:
            raise ValueError(f"Gas specific gravity must be positive, got {self.sg_gas}")

        # Separator conditions in field units (float constants)
        p_sep_psia = self.p_sep / CF_PSI
        T_sep_F = 1.8 * (self.T_sep - 273.15) + 32

        # Corrected gas gravity at reference separator (100 psig = 114.7 psia)
        self.sg_gas_corr = self.sg_gas * (
            1 + 5.912e-5 * self.api * T_sep_F * math_log(p_sep_psia / 114.7)
        )

        # API-dependent coefficients resolved once (no CasADi branching)
        if self.api <= 30:
            self._c1, self._c2, self._c3 = 0.0362, 1.0937, 25.7240
            self._f1, self._f2, self._f3 = 4.677e-4, 1.751e-5, -1.811e-8
        else:
            self._c1, self._c2, self._c3 = 0.0178, 1.1870, 23.9310
            self._f1, self._f2, self._f3 = 4.670e-4, 1.100e-5, 1.337e-9

        # Densities at standard conditions
        R_s_gas = R_UNIVERSAL / (M_AIR * self.sg_gas)
        self.rho_o_sc = density_from_api(self.api)          # kg/m3
        self.rho_g_sc = P_REF / (R_s_gas * T_REF)          # kg/m3

    @property
    def R_s_gas(self):
        """Specific gas constant of the associated gas (J/(kg K))."""
        return R_UNIVERSAL / (M_AIR * self.sg_gas)

    # ------------------------------------------------------------------
    # Internal field-unit helpers (CasADi-compatible)
    # ------------------------------------------------------------------

    def _rs_field(self, p_psia, T_F):
        """Vazquez-Beggs Rs in scf/STB (CasADi-compatible)."""
        return (
            self._c1 * self.sg_gas_corr
            * ca.constpow(p_psia, self._c2)
            * ca.exp(self._c3 * self.api / (T_F + 460))
        )

    def _bo_field(self, rs_scf, T_F):
        """Vazquez-Beggs Bo in bbl/STB (CasADi-compatible)."""
        return (
            1
            + self._f1 * rs_scf
            + (self._f2 + self._f3 * rs_scf) * (T_F - 60) * (self.api / self.sg_gas_corr)
        )

    def _to_field(self, p, T):
        """Convert SI pressure/temperature to field units."""
        return p / CF_PSI, 1.8 * (T - 273.15) + 32

    def _rs_field_capped(self, p_psia, T_F):
        """Rs in scf/STB, capped at bubble point if set."""
        rs = self._rs_field(p_psia, T_F)
        if self.p_bubble is not None:
            p_b_psia = self.p_bubble / CF_PSI
            rs_max = self._rs_field(p_b_psia, T_F)
            rs = ca_min_approx(rs, rs_max)
        return rs

    # ------------------------------------------------------------------
    # Public SI-unit API (CasADi-compatible)
    # ------------------------------------------------------------------

    def rs(self, p, T):
        """
        Solution gas-oil ratio at given pressure and temperature.

        :param p: Pressure (Pa), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Rs (Sm3 gas / Sm3 oil at standard conditions)
        """
        p_psia, T_F = self._to_field(p, T)
        return self._rs_field_capped(p_psia, T_F) * CF_RS

    def bo(self, p, T):
        """
        Oil formation volume factor at given pressure and temperature.

        :param p: Pressure (Pa), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Bo (dimensionless, m3 at conditions / m3 at standard)
        """
        p_psia, T_F = self._to_field(p, T)
        rs_scf = self._rs_field_capped(p_psia, T_F)
        return self._bo_field(rs_scf, T_F)

    def live_oil_density(self, p, T):
        """
        Density of live oil (stock-tank oil + dissolved gas) at (p, T).

        :param p: Pressure (Pa), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        :return: Live oil density (kg/m3)
        """
        Rs = self.rs(p, T)      # Sm3/Sm3
        Bo = self.bo(p, T)      # dimensionless
        return (self.rho_o_sc + Rs * self.rho_g_sc) / Bo

    def bubble_point_pressure(self, Rs_total, T):
        """
        Bubble point pressure from Standing's (1947) correlation, or the
        stored value if one was provided at construction.

        Reference: Standing, M.B., "A Pressure-Volume-Temperature Correlation
        for Mixtures of California Oils and Gases", Drilling and Production
        Practice, API (1947).

        :param Rs_total: Total solution GOR at saturation (Sm3/Sm3)
        :param T: Temperature (K)
        :return: Bubble point pressure (Pa)
        """
        if self.p_bubble is not None:
            return self.p_bubble

        Rs_scf = Rs_total / CF_RS
        T_F = 1.8 * (T - 273.15) + 32
        yg = (Rs_scf / self.sg_gas) ** 0.83
        p_b_psia = 18.2 * (yg * 10 ** (0.00091 * T_F - 0.0125 * self.api) - 1.4)
        return p_b_psia * CF_PSI

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_well_params(R_s_gas, rho_o, p_sep, T_sep, p_bubble=None):
        """
        Construct a BlackOilPVT from the parameters already present on
        WellProperties / BoundaryConditions.

        :param R_s_gas: Specific gas constant (J/(kg K))
        :param rho_o: Oil density at standard conditions (kg/m3)
        :param p_sep: Separator / downstream pressure (Pa)
        :param T_sep: Separator / surface temperature (K)
        :param p_bubble: Bubble point pressure (Pa), or None
        """
        api = api_from_density(rho_o)
        sg_gas = R_UNIVERSAL / (M_AIR * R_s_gas)
        return BlackOilPVT(
            api=api, sg_gas=sg_gas, p_sep=p_sep, T_sep=T_sep, p_bubble=p_bubble,
        )
