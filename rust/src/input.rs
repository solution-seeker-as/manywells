// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The core's inputs: a well, built once from the Python dataclasses (src/manywells/solvers/rust.py), and an
//! operating point. Validation stays in the dataclasses; the core checks only what it needs to run.

use crate::choke::Choke;
use crate::geometry;
use crate::inflow::Inflow;
use crate::pvt::fluid::Fluid;

/// A well in the v1.0.0 configuration
#[derive(Clone, Copy, Debug)]
pub struct WellSpec {
    pub l: f64,        // Pipe length (m), vertical
    pub d: f64,        // Inner pipe diameter (m)
    pub n_cells: usize,
    pub fluid: Fluid,
    pub f_d: f64,      // Darcy friction factor
    pub h: f64,        // Heat transfer coefficient (W/(m² K))
    pub inflow: Inflow,
    pub choke: Choke,
}

impl WellSpec {
    pub fn check(&self) -> Result<(), String> {
        if !(self.l > 0.0 && self.d > 0.0 && self.n_cells > 0) {
            return Err("the well needs a positive length, diameter and number of cells".into());
        }
        if !(self.fluid.f_g > 0.0 && self.fluid.f_g < 1.0) {
            return Err("the gas mass fraction must be in (0, 1)".into());
        }
        Ok(())
    }

    /// Cross-section (m²)
    pub fn a(&self) -> f64 {
        geometry::cross_section(self.d)
    }

    /// Cell length (m)
    pub fn delta_z(&self) -> f64 {
        geometry::cell_length(self.l, self.n_cells)
    }

    /// Number of state values, 7 (N + 1)
    pub fn n_x(&self) -> usize {
        crate::discretization::DIM_X * (self.n_cells + 1)
    }
}

/// The operating point: the boundary conditions and the controls
#[derive(Clone, Copy, Debug)]
pub struct OperatingPoint {
    pub p_r: f64,  // Reservoir pressure (bar)
    pub p_s: f64,  // Separator pressure (bar)
    pub t_r: f64,  // Reservoir temperature (K)
    pub t_s: f64,  // Ambient temperature at the wellhead (K)
    pub u: f64,    // Choke position in [0, 1]
    pub w_lg: f64, // Lift gas rate (kg/s), injected at the bottomhole
}

impl OperatingPoint {
    pub fn check(&self) -> Result<(), String> {
        if !(self.p_s > 0.0 && self.p_r > self.p_s) {
            return Err(format!("the reservoir pressure ({}) must exceed the separator pressure ({}) > 0", self.p_r,
                               self.p_s));
        }
        Ok(())
    }
}

/// Wells for the tests: W1 of specs/model/vectors/v1_rows.json (Vogel inflow, Simpson choke, sigmoid profile, lift
/// gas) and W2 (productivity index, Bernoulli choke, linear profile, choked at its root), on n cells
#[cfg(test)]
pub mod test_wells {
    use super::*;
    use crate::choke::{ChokeModel, Profile};

    fn cpr() -> f64 {
        let gamma: f64 = 1.307;
        (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0))
    }

    pub fn w1(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let spec = WellSpec {
            l: 2500.0, d: 0.127, n_cells,
            fluid: Fluid { rho_l: 900.0, r_s: 420.0, cp_g: 2225.0, cp_l: 3000.0, f_g: 0.15 },
            f_d: 0.03, h: 25.0,
            inflow: Inflow::Vogel { w_l_max: 80.0 },
            choke: Choke { model: ChokeModel::Simpson, k_c: 0.0015201224372924933, cpr: cpr(), profile: Profile::Sigmoid },
        };
        (spec, OperatingPoint { p_r: 249.2, p_s: 30.0, t_r: 363.15, t_s: 277.15, u: 0.6, w_lg: 0.8 })
    }

    pub fn w2(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let spec = WellSpec {
            l: 1800.0, d: 0.1524, n_cells,
            fluid: Fluid { rho_l: 820.0, r_s: 500.0, cp_g: 2225.0, cp_l: 2200.0, f_g: 0.4 },
            f_d: 0.05, h: 15.0,
            inflow: Inflow::ProductivityIndex { k_l: 0.6 },
            choke: Choke { model: ChokeModel::Bernoulli, k_c: 0.001824146924750992, cpr: cpr(), profile: Profile::Linear },
        };
        (spec, OperatingPoint { p_r: 150.0, p_s: 20.0, t_r: 345.0, t_s: 277.15, u: 0.8, w_lg: 0.0 })
    }
}
