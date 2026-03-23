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
from manywells.units import CF_PSI, CF_RS, CF_CP, M_AIR, kelvin_to_fahrenheit
from manywells.ca_functions import ca_min_approx, ca_max_approx, ca_sigmoid


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
        T_sep_F = kelvin_to_fahrenheit(self.T_sep)

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
        return p / CF_PSI, kelvin_to_fahrenheit(T)

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
        T_F = kelvin_to_fahrenheit(T)
        yg = (Rs_scf / self.sg_gas) ** 0.83
        p_b_psia = 18.2 * (yg * 10 ** (0.00091 * T_F - 0.0125 * self.api) - 1.4)
        return p_b_psia * CF_PSI



def live_oil_viscosity(mu_dead, Rs_scf):
    """
    Beggs-Robinson (1975) live oil viscosity correction.

    Reduces dead oil viscosity to account for dissolved gas.  The correction
    is significant: at Rs = 500 scf/STB a typical 5 cP dead oil drops to
    about 1 cP.

    Reference: Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of
    Crude Oil Systems", J Pet Technol 27 (1975): 1140-1141.

    :param mu_dead: Dead oil viscosity (Pa·s), may be CasADi symbolic
    :param Rs_scf: Solution gas-oil ratio (scf/STB), may be CasADi symbolic
    :return: Live oil viscosity (Pa·s)
    """
    mu_dead_cP = mu_dead / CF_CP
    a = 10.715 * ca.constpow(Rs_scf + 100, -0.515)
    b = 5.44 * ca.constpow(Rs_scf + 150, -0.338)
    mu_live_cP = a * ca.constpow(mu_dead_cP, b)
    return mu_live_cP * CF_CP


def live_oil_surface_tension(sigma_dead, Rs_scf):
    """
    Abdul-Majeed & Al-Soof (2000) live oil surface tension correction.

    Dissolved gas reduces surface tension. At Rs = 500 scf/STB, surface
    tension drops to roughly 40% of the dead-oil value.

    The two branches of the original correlation meet continuously at
    Rs_vol = 50 Sm3/Sm3 and are blended with a sigmoid for CasADi
    differentiability.

    Reference: Abdul-Majeed, G.H. and Al-Soof, N.B.A., "Estimation of
    gas-oil surface tension", J Pet Sci Eng 27 (2000): 197-200.

    :param sigma_dead: Dead oil surface tension (J/m2), may be CasADi symbolic
    :param Rs_scf: Solution gas-oil ratio (scf/STB), may be CasADi symbolic
    :return: Live oil surface tension (J/m2)
    """
    Rs_vol = Rs_scf * CF_RS  # Sm3/Sm3

    Rs_safe = ca_max_approx(Rs_vol, 1e-6)
    sigma_low = sigma_dead / (1 + 0.02549 * ca.constpow(Rs_safe, 1.0157))
    sigma_high = sigma_dead * 32.0436 * ca.constpow(Rs_safe, -1.1367)

    blend = ca_sigmoid(Rs_vol, 50.0, 0.5)
    return (1 - blend) * sigma_low + blend * sigma_high
