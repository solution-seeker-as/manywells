// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The oil phase (specs/model/pvt/oil.md).

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_has_api_ten() {
        assert!((api_from_density(WATER_RHO) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn surface_tension_is_positive_at_typical_conditions() {
        assert!(dead_oil_surface_tension(850.0, 293.15) > 0.0);
    }
}
