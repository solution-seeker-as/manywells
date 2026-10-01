// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 02 October 2026

//! The water phase (specs/model/pvt/water.md), as src/manywells/pvt/water.py. Water is incompressible in every
//! configuration, so its formation volume factor (PVT-WAT-2) enters no row and the core does not have it.

/// Water viscosity (Pa s) at T (K), a Vogel-Fulcher-Tammann type correlation
pub fn water_viscosity(t: f64) -> f64 {  // spec: PVT-WAT-3
    2.414e-5 * 10f64.powf(247.8 / (t - 140.0))
}
