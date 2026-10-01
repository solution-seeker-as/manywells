"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Thermal model (CasADi-compatible): heat exchanged with the surroundings, the ambient temperature profile, the
temperature of the fluid entering the well, and the temperature gradient of the energy balance.
Derivations of the frictional-heating and gravity terms: docs/thermal_energy_modeling.md.
"""

from dataclasses import dataclass

from manywells.units import STD_GRAVITY


@dataclass(frozen=True)
class ThermalModel:
    """
    Thermal model. With every option off it is v1.0.0's: heat loss to a linear ambient profile, and the fluid enters
    at the reservoir temperature.

    The temperature gradient along the flow path is
        dT/dMD = -H + (frictional heating) - (gravity term),
    where H is the heat loss of Zhang et al. (2006).
    """

    h: float = 20.0                    # Overall heat transfer coefficient (W/m²/K)
    frictional_heating: bool = True    # Viscous dissipation heats the liquid
    gravity_term: bool = True          # Work against gravity cools the flow
    lift_gas_mixing: bool = True       # Lift gas at T_lg mixes with the reservoir fluid at the bottomhole

    def __post_init__(self):
        if not self.h >= 0:
            raise ValueError('Heat transfer coefficient must be non-negative')

    @staticmethod
    def ambient_temperature(tvd_frac, T_r, T_s):  # spec: THM-4
        """
        Ambient temperature, linear in true vertical depth from T_s at the surface to T_r at the bottomhole.

        :param tvd_frac: True vertical depth as a fraction of the bottomhole's (dimensionless)
        :param T_r: Reservoir temperature (K)
        :param T_s: Surface temperature (K)
        :return: Ambient temperature (K)
        """
        return T_s + (T_r - T_s) * tvd_frac

    def inflow_temperature(self, w_res, w_lg, T_r, T_lg, fluid):
        """
        Temperature of the fluid at the bottomhole.

        :param w_res: Liquid mass flow rate from the reservoir (kg/s)
        :param w_lg: Lift gas mass flow rate (kg/s)
        :param T_r: Reservoir temperature (K)
        :param T_lg: Lift gas temperature (K)
        :param fluid: Fluid model (FluidModel), for the heat capacities and the reservoir gas rate
        :return: Temperature (K)
        """
        if not self.lift_gas_mixing:
            return T_r  # spec: THM-3
        # Heat-capacity-weighted mix of the reservoir fluid at T_r and the lift gas at T_lg; exactly T_r if T_lg = T_r
        H_res = w_res * fluid.cp_l + fluid.reservoir_gas_rate(w_res) * fluid.cp_g
        H_lg = w_lg * fluid.cp_g
        return T_r + H_lg * (T_lg - T_r) / (H_res + H_lg)  # spec: THM-5

    def temperature_gradient(self, s, fluid, T_a, F, dp_dmd, cos_incl, D):
        """
        Temperature gradient dT/dMD along the flow path at a point.

        :param s: State at the point (a PointState: p, v_g, v_l, alpha, rho_g, rho_l, T, rho_m, v_m)
        :param fluid: Fluid model (FluidModel), for the heat capacities
        :param T_a: Ambient temperature at the point (K)
        :param F: Viscous pressure gradient at the point (Pa/m)
        :param dp_dmd: Pressure gradient of the cell, c_bar (p_i - p_{i-1}) / delta_md (Pa/m); for pressure-dependent
                       terms such as Joule-Thomson cooling, which no option uses yet
        :param cos_incl: Cosine of the inclination from vertical of the cell (dimensionless)
        :param D: Inner pipe diameter (m)
        :return: dT/dMD (K/m)
        """
        cp_flux = fluid.cp_g * s.alpha * s.rho_g * s.v_g + fluid.cp_l * (1 - s.alpha) * s.rho_l * s.v_l
        dT = -4 * self.h * (s.T - T_a) / (D * cp_flux)  # spec: THM-1
        if self.frictional_heating:
            dT += (1 - s.alpha) * s.v_l * F / cp_flux  # spec: THM-6
        if self.gravity_term:
            mass_flux = s.alpha * s.rho_g * s.v_g + (1 - s.alpha) * s.rho_l * s.v_l
            liq_flux = (1 - s.alpha) * s.v_l
            dT -= cos_incl * STD_GRAVITY * (mass_flux - liq_flux * s.rho_m) / cp_flux  # spec: THM-7
        return dT
