// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The thermal model (specs/model/thermal.md), as ThermalModel in src/manywells/thermal.py: heat loss to an ambient
//! profile linear in true vertical depth, and optionally frictional heating and the gravity term; the fluid enters at
//! the reservoir temperature. Derivations of the terms: docs/thermal_energy_modeling.md.

use crate::discretization::State;
use crate::pvt::fluid::Fluid;
use crate::units::STD_GRAVITY;

#[derive(Clone, Copy, Debug)]
pub struct Thermal {
    pub h: f64,                   // Overall heat transfer coefficient (W/(m² K))
    pub frictional_heating: bool, // Viscous dissipation heats the liquid (THM-6)
    pub gravity_term: bool,       // Work against gravity cools the flow (THM-7)
}

/// Ambient temperature (K) at a point whose true vertical depth is tvd_frac of the bottomhole's, linear from T_s at
/// the surface to T_r at the bottomhole
pub fn ambient_temperature(tvd_frac: f64, t_r: f64, t_s: f64) -> f64 {  // spec: THM-4
    t_s + (t_r - t_s) * tvd_frac
}

/// The heat flux capacity cp_g α ρ_g v_g + cp_l (1 - α) ρ_l v_l (W/(m² K)) at a point
pub fn heat_flux_capacity(fluid: &Fluid, s: &State) -> f64 {
    fluid.cp_g * s.alpha * s.rho_g * s.v_g + fluid.cp_l * (1.0 - s.alpha) * s.rho_l * s.v_l
}

impl Thermal {
    /// The temperature gradient dT/dMD (K/m) along the flow path at a point with state s, ambient temperature t_a
    /// (K), viscous pressure gradient f (Pa/m) and inclination cos_incl, in a pipe of inner diameter d (m):
    /// -H + (frictional heating) - (gravity term), with H the heat loss
    pub fn temperature_gradient(&self, s: &State, fluid: &Fluid, t_a: f64, f: f64, cos_incl: f64, d: f64) -> f64 {
        let cp_flux = heat_flux_capacity(fluid, s);
        let mut dt = -4.0 * self.h * (s.t - t_a) / (d * cp_flux); // spec: THM-1
        if self.frictional_heating {
            dt += (1.0 - s.alpha) * s.v_l * f / cp_flux; // spec: THM-6
        }
        if self.gravity_term {
            let mass_flux = s.alpha * s.rho_g * s.v_g + (1.0 - s.alpha) * s.rho_l * s.v_l;
            let liq_flux = (1.0 - s.alpha) * s.v_l;
            dt -= cos_incl * STD_GRAVITY * (mass_flux - liq_flux * s.rho_m()) / cp_flux; // spec: THM-7
        }
        dt
    }

    /// Whether the energy row is linear in the temperature and does not depend on the pressure: with heat loss
    /// alone, where the phase rates fix the heat flux capacity (no mass transfer)
    pub fn is_linear(&self, fluid: &Fluid) -> bool {
        !self.frictional_heating && !self.gravity_term && !fluid.has_mass_transfer()
    }

    /// An upper bound on the gravity term (K/m) at inclination cos_incl, over every state: its numerator
    /// α ρ_g v_g + α (1 - α) v_l (ρ_l - ρ_g) lies in [0, G_g + G_l], and cp_flux >= min(c_pg, c_pl) (G_g + G_l)
    pub fn gravity_term_bound(&self, fluid: &Fluid, cos_incl: f64) -> f64 {
        if self.gravity_term { cos_incl * STD_GRAVITY / fluid.cp_g.min(fluid.cp_l) } else { 0.0 }
    }
}

/// The temperature (K) at point i that zeroes the energy row of cell i (DISC-10 with THM-1 alone), given t_prev at
/// point i - 1, the ambient temperature t_a at point i and the cell's length delta_md (m). The row is linear in it,
/// because the heat flux capacity cp_g α ρ_g v_g + cp_l (1 - α) ρ_l v_l = cp_g G_g + cp_l G_l (W/(m² K)) is fixed by the
/// mass fluxes G (kg/(m² s)), which the mass rows hold constant without mass transfer:
/// T_i = (T_{i-1} + ΔMD k T_a,i) / (1 + ΔMD k) with k = 4h / (D capacity).
pub fn energy_step(t_prev: f64, t_a: f64, delta_md: f64, h: f64, d: f64, capacity: f64) -> f64 {  // spec: DISC-10, THM-1
    let k = 4.0 * h / (d * capacity);
    (t_prev + delta_md * k * t_a) / (1.0 + delta_md * k)
}

/// Temperature of the fluid entering the well (K)
pub fn inflow_temperature(t_r: f64) -> f64 {  // spec: THM-3
    t_r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::test_wells::v1_fluid;

    fn state(alpha: f64, v_g: f64, v_l: f64) -> State {
        State { p: 100.0, v_g, v_l, alpha, rho_g: 60.0, rho_l: 820.0, t: 350.0 }
    }

    #[test]
    fn frictional_heating_of_a_liquid_is_f_over_rho_c() {
        let fluid = v1_fluid(820.0, 420.0, 2225.0, 3000.0, 0.1);
        let th = Thermal { h: 0.0, frictional_heating: true, gravity_term: false };
        let dt = th.temperature_gradient(&state(0.0, 1.0, 2.0), &fluid, 350.0, 500.0, 1.0, 0.1);
        assert!((dt - 500.0 / (820.0 * 3000.0)).abs() < 1e-15);
    }

    #[test]
    fn the_gravity_term_is_g_over_c_for_a_gas_and_zero_for_a_liquid() {
        let fluid = v1_fluid(820.0, 420.0, 2225.0, 3000.0, 0.1);
        let th = Thermal { h: 0.0, frictional_heating: false, gravity_term: true };
        let gas = -th.temperature_gradient(&state(1.0, 10.0, 1.0), &fluid, 350.0, 0.0, 1.0, 0.1);
        assert!((gas - STD_GRAVITY / 2225.0).abs() < 1e-15);
        assert_eq!(th.temperature_gradient(&state(0.0, 1.0, 2.0), &fluid, 350.0, 0.0, 1.0, 0.1), 0.0);
        for (alpha, v_g, v_l) in [(0.3, 2.0, 1.0), (0.8, 15.0, 3.0), (0.05, 0.5, 0.4)] {
            let phi = -th.temperature_gradient(&state(alpha, v_g, v_l), &fluid, 350.0, 0.0, 0.7, 0.1);
            assert!(phi >= 0.0 && phi <= th.gravity_term_bound(&fluid, 0.7), "{phi}");
        }
    }
}
