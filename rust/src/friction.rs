// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The friction model (specs/model/friction.md). So far the fixed Darcy friction factor of the v1.0.0 configuration.

/// Viscous pressure gradient (Pa/m) at mixture density rho_m (kg/m³) and mixture velocity v_m (m/s), for a Darcy
/// friction factor f_d in a pipe of inner diameter d (m)
pub fn pressure_gradient(f_d: f64, d: f64, rho_m: f64, v_m: f64) -> f64 {  // spec: FRIC-1, FRIC-2
    f_d / d / 2.0 * rho_m * (v_m * v_m.abs())
}
