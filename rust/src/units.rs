// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Constants and unit conversions, as in src/manywells/units.py.

/// Standard acceleration of gravity (m/s²)
pub const STD_GRAVITY: f64 = 9.80665;

/// Universal gas constant (J/(kmol K))
pub const R_UNIVERSAL: f64 = 8314.46;

/// Molecular weight of air (kg/kmol)
pub const M_AIR: f64 = 28.97;

/// Reference pressure (Pa) and temperature (K) of the standard conditions
pub const P_REF: f64 = 101_325.0;
pub const T_REF: f64 = 288.15;

/// Pascal per bar
pub const CF_BAR: f64 = 1e5;
