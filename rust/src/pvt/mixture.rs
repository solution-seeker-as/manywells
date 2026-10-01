// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The liquid mixture (specs/model/pvt/mixture.md).

use crate::pvt::oil::dead_oil_surface_tension;

/// The liquid-density row (kg/m³), zero where the state's liquid density is rho_l
pub fn liquid_density_row(rho_l_state: f64, rho_l: f64) -> f64 {  // spec: PVT-MIX-1, PVT-MIX-6
    rho_l_state - rho_l
}

/// Gas-liquid mixture viscosity (Pa s), mass-weighted, Hasan, Kabir and Sayarpour (2010), Eq. A-3
pub fn mixture_viscosity(mu_l: f64, mu_g: f64, alpha: f64, rho_l: f64, rho_g: f64) -> f64 {  // spec: PVT-MIX-9
    let x = alpha * rho_g / (alpha * rho_g + (1.0 - alpha) * rho_l);
    mu_g * x + mu_l * (1.0 - x)
}

/// Gas-liquid surface tension (J/m²): the dead-oil correlation at the local liquid density
pub fn liquid_surface_tension(rho_l: f64, t: f64) -> f64 {  // spec: PVT-MIX-5
    dead_oil_surface_tension(rho_l, t)
}
