// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The liquid mixture (specs/model/pvt/mixture.md).

use crate::pvt::oil::dead_oil_surface_tension;

/// The constant-liquid-density row (kg/m³)
pub fn constant_liquid_density_row(rho_l_state: f64, rho_l: f64) -> f64 {  // spec: PVT-MIX-1
    rho_l_state - rho_l
}

/// Gas-liquid surface tension (J/m²): the dead-oil correlation at the local liquid density
pub fn liquid_surface_tension(rho_l: f64, t: f64) -> f64 {  // spec: PVT-MIX-5
    dead_oil_surface_tension(rho_l, t)
}
