// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The well's geometry and grid (specs/model/geometry.md, DISC-1). So far the vertical pipe of the v1.0.0
//! configuration, on a uniform grid.

use std::f64::consts::PI;

/// Cross-section (m²) of a circular pipe of inner diameter d (m)
pub fn cross_section(d: f64) -> f64 {  // spec: GEO-2
    PI * (d / 2.0) * (d / 2.0)
}

/// Length (m) of each of the n cells of a vertical pipe of length l (m)
pub fn cell_length(l: f64, n: usize) -> f64 {  // spec: GEO-1, DISC-1
    l / n as f64
}
