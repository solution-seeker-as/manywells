// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The top boundary (specs/model/choke.md): the mass rate the choke passes from the wellhead to the separator.

use crate::smoothing::max_approx;
use crate::units::CF_BAR;

/// Smoothing parameter of the critical downstream pressure (bar²)
pub const CRITICAL_PRESSURE_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChokeModel {
    /// Liquid density with Simpson's two-phase multiplier
    Simpson,
    /// Mixture density, no multiplier
    Bernoulli,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Profile {
    Linear,
    Sigmoid,
    Convex,
    Concave,
}

impl Profile {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "linear" => Ok(Self::Linear),
            "sigmoid" => Ok(Self::Sigmoid),
            "convex" => Ok(Self::Convex),
            "concave" => Ok(Self::Concave),
            _ => Err(format!("choke profile {name:?} is not supported")),
        }
    }

    /// Relative choke opening at choke position u in [0, 1]
    pub fn opening(&self, u: f64) -> f64 {
        match self {
            Self::Linear => u, // spec: CHK-7
            Self::Sigmoid => {
                let b = 1.5;
                u.powf(b) / (u.powf(b) + (1.0 - u).powf(b)) // spec: CHK-8
            }
            Self::Convex => {
                let b = 0.25;
                b * u + (1.0 - b) * (u * u) // spec: CHK-9
            }
            Self::Concave => {
                let b = 0.75;
                u.powf(b) // spec: CHK-10
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Choke {
    pub model: ChokeModel,
    pub k_c: f64,         // Choke coefficient (m²)
    pub cpr: f64,         // Critical pressure ratio r_c, from the Python choke (CHK-4)
    pub profile: Profile,
}

/// Simpson's two-phase multiplier at gas mass fraction x_g
pub fn simpson_multiplier(x_g: f64, rho_g: f64, rho_l: f64) -> f64 {  // spec: CHK-5
    let s = (rho_l / rho_g).powf(1.0 / 6.0);
    (1.0 + x_g * (s - 1.0)) * (1.0 + x_g * (s.powf(5.0) - 1.0))
}

impl Choke {
    /// Critical downstream pressure (bar): the smooth max of r_c p_in and p_out
    pub fn critical_pressure(&self, p_in: f64, p_out: f64) -> f64 {  // spec: CHK-3
        max_approx(self.cpr * p_in, p_out, CRITICAL_PRESSURE_EPS)
    }

    /// The choke equation (kg/s) at choke position u, from p_in to p_out (bar), for density rho (kg/m³) and
    /// multiplier phi. Where p_in is at or below the critical pressure, the choke passes no flow.
    pub fn choke_equation(&self, u: f64, p_in: f64, p_out: f64, rho: f64, phi: f64) -> f64 {  // spec: CHK-2, CHK-11
        let chk = self.profile.opening(u);
        let p_c = self.critical_pressure(p_in, p_out);
        let dp = CF_BAR * (p_in - p_c);
        if dp > 0.0 { self.k_c * chk * (2.0 * rho * dp / phi).sqrt() } else { 0.0 }
    }

    /// The rate through the choke (kg/s) from the wellhead at pressure p_in (bar) with gas and liquid mass rates
    /// w_g and w_l (kg/s), void fraction alpha and densities rho_g and rho_l (kg/m³), to the separator at p_s
    pub fn mass_flow_rate(&self, u: f64, p_in: f64, p_s: f64, w_g: f64, w_l: f64, alpha: f64, rho_g: f64,
                          rho_l: f64) -> f64 {
        match self.model {
            ChokeModel::Simpson => {
                let x_g = w_g / (w_g + w_l);
                let phi = simpson_multiplier(x_g, rho_g, rho_l);
                self.choke_equation(u, p_in, p_s, rho_l, phi) // spec: CHK-5
            }
            ChokeModel::Bernoulli => {
                let rho_m = alpha * rho_g + (1.0 - alpha) * rho_l;
                self.choke_equation(u, p_in, p_s, rho_m, 1.0) // spec: CHK-6
            }
        }
    }

    /// CHOKED: the separator pressure is at or below the critical pressure, r_c p_in
    pub fn is_choked(&self, p_in: f64, p_out: f64) -> bool {  // spec: CHK-12
        p_out <= self.cpr * p_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choke(model: ChokeModel, profile: Profile) -> Choke {
        let gamma: f64 = 1.307;
        Choke { model, k_c: 0.002, cpr: (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0)), profile }
    }

    #[test]
    fn profiles_open_from_zero_to_one() {
        for p in [Profile::Linear, Profile::Sigmoid, Profile::Convex, Profile::Concave] {
            assert_eq!(p.opening(0.0), 0.0);
            assert!((p.opening(1.0) - 1.0).abs() < 1e-15);
        }
        assert!((Profile::Sigmoid.opening(0.5) - 0.5).abs() < 1e-12);
        assert!(Profile::from_name("invalid").is_err());
    }

    #[test]
    fn no_flow_at_or_below_the_critical_pressure() {
        let c = choke(ChokeModel::Bernoulli, Profile::Linear);
        assert_eq!(c.choke_equation(0.5, 20.0, 20.0, 800.0, 1.0), 0.0);
        assert_eq!(c.choke_equation(0.5, 15.0, 20.0, 800.0, 1.0), 0.0);
        assert!(c.choke_equation(0.5, 30.0, 20.0, 800.0, 1.0) > 0.0);
    }

    #[test]
    fn choked_below_the_critical_ratio() {
        let c = choke(ChokeModel::Simpson, Profile::Linear);
        assert!(c.is_choked(100.0, 0.9 * c.cpr * 100.0));
        assert!(!c.is_choked(100.0, 1.1 * c.cpr * 100.0));
    }
}
