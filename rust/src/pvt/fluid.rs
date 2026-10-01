// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The fluid: the one interface to the fluid properties that the rest of the core calls, as FluidModel in
//! src/manywells/pvt/fluid.py. So far the v1.0.0 configuration: an ideal gas, and a dead oil that is the whole liquid.

use crate::pvt::{gas, mixture};

#[derive(Clone, Copy, Debug)]
pub struct Fluid {
    pub rho_l: f64, // Liquid density (kg/m³)
    pub r_s: f64,   // Specific gas constant (J/(kg K))
    pub cp_g: f64,  // Gas heat capacity (J/(kg K))
    pub cp_l: f64,  // Liquid heat capacity (J/(kg K))
    pub f_g: f64,   // Gas mass fraction of the reservoir inflow, in (0, 1)
}

impl Fluid {
    pub fn gas_density(&self, p: f64, t: f64) -> f64 {
        gas::ideal_gas_density(p, t, self.r_s)
    }

    pub fn gas_law_row(&self, p: f64, t: f64, rho_g: f64) -> f64 {
        gas::ideal_gas_row(p, t, rho_g, self.r_s)
    }

    pub fn liquid_density(&self) -> f64 {
        self.rho_l
    }

    pub fn liquid_density_row(&self, rho_l_state: f64) -> f64 {
        mixture::constant_liquid_density_row(rho_l_state, self.rho_l)
    }

    pub fn surface_tension(&self, rho_l: f64, t: f64) -> f64 {
        mixture::liquid_surface_tension(rho_l, t)
    }

    /// Gas mass rate from the reservoir (kg/s) for a reservoir liquid rate w_res (kg/s)
    pub fn reservoir_gas_rate(&self, w_res: f64) -> f64 {  // spec: INF-4
        (self.f_g / (1.0 - self.f_g)) * w_res
    }

    /// Gas and liquid mass rates (kg/s), the same at every point without mass transfer
    pub fn phase_rates(&self, w_res: f64, w_lg: f64) -> (f64, f64) {  // spec: INF-5, BAL-3
        (self.reservoir_gas_rate(w_res) + w_lg, w_res)
    }
}
