// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The gas phase (specs/model/pvt/gas.md), as src/manywells/pvt/gas.py: the real-gas law with a compressibility
//! factor Z, which is 1 for an ideal gas and from Papay's correlation for a real one.

use crate::units::{CF_BAR, CF_PSI};

/// Gas density (kg/m³) at p (bar) and T (K), with compressibility factor z and specific gas constant r_s (J/(kg K))
pub fn gas_density(p: f64, t: f64, z: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1, PVT-GAS-3
    CF_BAR * p / (z * r_s * t)
}

/// The gas-law row (bar), p - rho_g Z R_s T / c_bar, zero where rho_g is the density at p and T
pub fn gas_law_row(p: f64, t: f64, rho_g: f64, z: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1, PVT-GAS-3
    p - rho_g * z * r_s * t / CF_BAR
}

/// Ideal-gas density (kg/m³) at p (bar) and T (K), with specific gas constant R_s (J/(kg K))
pub fn ideal_gas_density(p: f64, t: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1
    gas_density(p, t, 1.0, r_s)
}

/// Pseudo-critical pressure (Pa) and temperature (K) of a gas of specific gravity sg_gas, Sutton (1985)
pub fn sutton_pseudo_critical(sg_gas: f64) -> (f64, f64) {  // spec: PVT-GAS-5
    let ppc_psia = 756.8 - 131.07 * sg_gas - 3.6 * (sg_gas * sg_gas);
    let tpc_r = 169.2 + 349.5 * sg_gas - 74.0 * (sg_gas * sg_gas);
    (ppc_psia * CF_PSI, tpc_r / 1.8)
}

/// Compressibility factor at pressure p_pa (Pa) and temperature t (K), by Papay's (1968) correlation, for the
/// pseudo-critical pressure ppc (Pa) and temperature tpc (K)
pub fn papay_z_factor(p_pa: f64, t: f64, ppc: f64, tpc: f64) -> f64 {  // spec: PVT-GAS-4
    let ppr = p_pa / ppc;
    let tpr = t / tpc;
    1.0 - 3.52 * ppr * 10f64.powf(-0.9813 * tpr) + 0.274 * (ppr * ppr) * 10f64.powf(-0.8157 * tpr)
}

/// Gas viscosity (Pa s) at T (K) and density rho_g (kg/m³), for molecular weight m_g (kg/kmol), by the
/// Lee-Gonzalez-Eakin (1966) correlation
pub fn gas_viscosity(t: f64, rho_g: f64, m_g: f64) -> f64 {  // spec: PVT-GAS-7
    let t_r = 1.8 * t;
    let rho_gcc = rho_g * 1e-3;
    let k = (9.4 + 0.02 * m_g) * t_r.powf(1.5) / (209.0 + 19.0 * m_g + t_r);
    let x = 3.5 + 986.0 / t_r + 0.01 * m_g;
    let y = 2.4 - 0.2 * x;
    k * (x * rho_gcc.powf(y)).exp() * 1e-7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_density_zeroes_the_row() {
        let (p, t, r_s) = (85.0, 330.0, 420.0);
        assert!(gas_law_row(p, t, ideal_gas_density(p, t, r_s), 1.0, r_s).abs() < 1e-12);
        assert!(gas_law_row(p, t, gas_density(p, t, 0.87, r_s), 0.87, r_s).abs() < 1e-12);
    }

    #[test]
    fn a_real_gas_is_ideal_at_low_pressure() {
        let (ppc, tpc) = sutton_pseudo_critical(0.65);
        assert!((papay_z_factor(1e3, 300.0, ppc, tpc) - 1.0).abs() < 1e-3);
        assert!(papay_z_factor(150e5, 350.0, ppc, tpc) < 0.95);
    }
}
