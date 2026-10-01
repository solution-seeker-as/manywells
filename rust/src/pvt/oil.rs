// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The oil phase (specs/model/pvt/oil.md): dead-oil properties, and black oil by the Vazquez-Beggs correlations, as
//! src/manywells/pvt/black_oil.py and dead_oil.py. The correlations are in field units (psia, °F, scf/STB), with the
//! conversions at their boundaries.

use crate::smoothing::min_approx;
use crate::units::{kelvin_to_fahrenheit, CF_PSI, CF_RS};

/// Density of water at standard conditions (kg/m³), the reference of the specific gravity
pub const WATER_RHO: f64 = 999.1;

/// API gravity of a liquid of density rho (kg/m³)
pub fn api_from_density(rho: f64) -> f64 {  // spec: PVT-OIL-2
    let sg = rho / WATER_RHO;
    141.5 / sg - 131.5
}

/// Dead-oil surface tension (J/m²) at density rho (kg/m³) and temperature T (K), by Abdul-Majeed and Abu Al-Soof
/// (2000), whose correlation gives dyn/cm (1 dyn/cm = 0.001 J/m²)
pub fn dead_oil_surface_tension(rho: f64, t: f64) -> f64 {  // spec: PVT-OIL-3
    let cf = 0.001;
    let t_deg_c = t - 273.15;
    let api = api_from_density(rho);
    cf * (1.11591 - 0.00305 * t_deg_c) * (38.085 - 0.259 * api)
}

/// Gas specific gravity corrected to a reference separator at 114.7 psia, from the separator's pressure p_sep (Pa)
/// and temperature t_sep (K), for an oil of the given API gravity
pub fn separator_gravity(api: f64, sg_gas: f64, p_sep: f64, t_sep: f64) -> f64 {
    let p_sep_psia = p_sep / CF_PSI;
    let t_sep_f = kelvin_to_fahrenheit(t_sep);
    sg_gas * (1.0 + 5.912e-5 * api * t_sep_f * (p_sep_psia / 114.7).log10()) // spec: PVT-OIL-5
}

/// Black oil by the Vazquez-Beggs correlations, as BlackOilPVT, with the coefficients of its API range
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackOil {
    pub api: f64,
    pub sg_gas_corr: f64,      // Gas specific gravity at the reference separator (PVT-OIL-5)
    c: [f64; 3],               // Coefficients of R_so (PVT-OIL-6)
    f: [f64; 3],               // Coefficients of B_o (PVT-OIL-8)
    p_bubble: Option<f64>,     // Bubble point pressure (Pa), which caps R_so (PVT-OIL-7)
}

impl BlackOil {
    /// For an oil of API gravity api (10 to 40), gas of specific gravity sg_gas, a separator at p_sep (Pa) and
    /// t_sep (K), and an optional bubble point p_bubble (Pa)
    pub fn new(api: f64, sg_gas: f64, p_sep: f64, t_sep: f64, p_bubble: Option<f64>) -> Self {
        let (c, f) = if api <= 30.0 {
            ([0.0362, 1.0937, 25.7240], [4.677e-4, 1.751e-5, -1.811e-8])
        } else {
            ([0.0178, 1.1870, 23.9310], [4.670e-4, 1.100e-5, 1.337e-9])
        };
        Self { api, sg_gas_corr: separator_gravity(api, sg_gas, p_sep, t_sep), c, f, p_bubble }
    }

    /// Solution gas-oil ratio (scf/STB) at p_psia (psia) and t_f (°F)
    fn rs_field(&self, p_psia: f64, t_f: f64) -> f64 {  // spec: PVT-OIL-6
        self.c[0] * self.sg_gas_corr * p_psia.powf(self.c[1]) * (self.c[2] * self.api / (t_f + 460.0)).exp()
    }

    /// Solution gas-oil ratio (scf/STB), capped at the bubble point's if one is set
    fn rs_field_capped(&self, p_psia: f64, t_f: f64) -> f64 {  // spec: PVT-OIL-7
        let rs = self.rs_field(p_psia, t_f);
        match self.p_bubble {
            Some(p_b) => min_approx(rs, self.rs_field(p_b / CF_PSI, t_f), 1e-6),
            None => rs,
        }
    }

    /// Formation volume factor at a solution gas-oil ratio rs_scf (scf/STB) and t_f (°F)
    fn bo_field(&self, rs_scf: f64, t_f: f64) -> f64 {  // spec: PVT-OIL-8
        1.0 + self.f[0] * rs_scf + (self.f[1] + self.f[2] * rs_scf) * (t_f - 60.0) * (self.api / self.sg_gas_corr)
    }

    /// Solution gas-oil ratio (Sm³/Sm³) at p (Pa) and T (K)
    pub fn rs(&self, p: f64, t: f64) -> f64 {
        self.rs_field_capped(p / CF_PSI, kelvin_to_fahrenheit(t)) * CF_RS
    }

    /// Oil formation volume factor at p (Pa) and T (K)
    pub fn bo(&self, p: f64, t: f64) -> f64 {
        let t_f = kelvin_to_fahrenheit(t);
        self.bo_field(self.rs_field_capped(p / CF_PSI, t_f), t_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bubble_point_caps_the_solution_gas() {
        let free = BlackOil::new(35.0, 0.65, 101325.0, 288.15, None);
        let capped = BlackOil::new(35.0, 0.65, 101325.0, 288.15, Some(150e5));
        assert!(free.rs(250e5, 350.0) > capped.rs(250e5, 350.0));
        assert!((capped.rs(250e5, 350.0) - capped.rs(150e5, 350.0)).abs() < 1e-3 * capped.rs(150e5, 350.0));
        assert!((capped.rs(50e5, 350.0) - free.rs(50e5, 350.0)).abs() < 1e-3);
    }

    #[test]
    fn water_has_api_ten() {
        assert!((api_from_density(WATER_RHO) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn surface_tension_is_positive_at_typical_conditions() {
        assert!(dead_oil_surface_tension(850.0, 293.15) > 0.0);
    }
}
