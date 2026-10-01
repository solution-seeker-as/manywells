// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The friction model (specs/model/friction.md), as src/manywells/friction.py: the viscous pressure gradient of the
//! momentum balance, with a fixed Darcy friction factor, or one from the mixture's Reynolds number and the pipe's
//! relative roughness, blending the laminar 64/Re into a turbulent correlation.

use crate::discretization::State;
use crate::pvt::fluid::Fluid;
use crate::smoothing::{max_approx, sigmoid};

/// The turbulent correlation of friction from roughness
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Correlation {
    Chen,
    Haaland,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Friction {
    /// One Darcy friction factor for the whole well
    FixedFactor { f_d: f64 },
    /// From the Reynolds number and the pipe wall's roughness (m)
    Roughness { roughness: f64, correlation: Correlation },
}

/// Darcy friction factor of turbulent flow at Reynolds number re and relative roughness eps_d, Chen (1979)
pub fn chen_friction_factor(re: f64, eps_d: f64) -> f64 {  // spec: FRIC-4
    let lambda = eps_d.powf(1.1098) / 2.8257 + (7.149 / re).powf(0.8981);
    let arg = eps_d / 3.7065 - (5.0452 / re) * lambda.log10();
    let inv_sqrt_f = -2.0 * arg.log10();
    1.0 / (inv_sqrt_f * inv_sqrt_f)
}

/// Darcy friction factor of turbulent flow, Haaland (1983)
pub fn haaland_friction_factor(re: f64, eps_d: f64) -> f64 {  // spec: FRIC-5
    let inv_sqrt_f = -1.8 * ((eps_d / 3.7).powf(1.11) + 6.9 / re).log10();
    1.0 / (inv_sqrt_f * inv_sqrt_f)
}

/// Darcy friction factor at Reynolds number re and relative roughness eps_d: 64/Re blended into the turbulent
/// correlation by a sigmoid centred at Re = 3000
pub fn friction_factor_of_re(re: f64, eps_d: f64, correlation: Correlation) -> f64 {  // spec: FRIC-6
    let re_safe = max_approx(re, 1.0, 1e-6);
    let re_turb = max_approx(re_safe, 1000.0, 1e-6);
    let f_lam = 64.0 / re_safe;
    let f_turb = match correlation {
        Correlation::Chen => chen_friction_factor(re_turb, eps_d),
        Correlation::Haaland => haaland_friction_factor(re_turb, eps_d),
    };
    let s = sigmoid(re_safe, 3000.0, 0.005);
    (1.0 - s) * f_lam + s * f_turb
}

impl Friction {
    /// Darcy friction factor at a point with state s, in a pipe of inner diameter d (m)
    pub fn friction_factor(&self, s: &State, fluid: &Fluid, d: f64) -> f64 {
        match *self {
            Friction::FixedFactor { f_d } => f_d, // spec: FRIC-2
            Friction::Roughness { roughness, correlation } => {
                let mu_m = fluid.mixture_viscosity(s.p, s.t, s.alpha, s.rho_g, s.rho_l);
                let re = s.rho_m() * s.v_m().abs() * d / mu_m; // spec: FRIC-3
                friction_factor_of_re(re, roughness / d, correlation)
            }
        }
    }

    /// Viscous pressure gradient (Pa/m) at a point with state s, in a pipe of inner diameter d (m)
    pub fn pressure_gradient(&self, s: &State, fluid: &Fluid, d: f64) -> f64 {  // spec: FRIC-1
        let v_m = s.v_m();
        self.friction_factor(s, fluid, d) / d / 2.0 * s.rho_m() * (v_m * v_m.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laminar_flow_has_64_over_re() {
        for c in [Correlation::Chen, Correlation::Haaland] {
            assert!((friction_factor_of_re(100.0, 1e-4, c) - 0.64).abs() < 1e-6);
        }
    }

    #[test]
    fn turbulent_flow_follows_the_correlation() {
        let (re, eps) = (1e6, 3e-4);
        assert!((friction_factor_of_re(re, eps, Correlation::Chen) - chen_friction_factor(re, eps)).abs() < 1e-12);
        assert!((chen_friction_factor(re, eps) / haaland_friction_factor(re, eps) - 1.0).abs() < 0.02);
    }
}
