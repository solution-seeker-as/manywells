// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The gas phase (specs/model/pvt/gas.md).

use crate::units::CF_BAR;

/// Ideal-gas density (kg/m³) at p (bar) and T (K), with specific gas constant R_s (J/(kg K))
pub fn ideal_gas_density(p: f64, t: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1
    CF_BAR * p / (r_s * t)
}

/// The ideal-gas-law row (bar), zero where rho_g is the density at p and T
pub fn ideal_gas_row(p: f64, t: f64, rho_g: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1
    p - rho_g * r_s * t / CF_BAR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_density_zeroes_the_row() {
        let (p, t, r_s) = (85.0, 330.0, 420.0);
        assert!(ideal_gas_row(p, t, ideal_gas_density(p, t, r_s), r_s).abs() < 1e-12);
    }
}
