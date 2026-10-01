// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The thermal model (specs/model/thermal.md). So far the v1.0.0 configuration: heat loss to a linear ambient
//! profile, and the fluid enters at the reservoir temperature.

/// Ambient temperature (K) at point i of n, falling linearly from T_r at the bottomhole to T_s at the wellhead
pub fn ambient_temperature(i: usize, n: usize, t_r: f64, t_s: f64) -> f64 {  // spec: THM-2
    t_r - i as f64 * (t_r - t_s) / n as f64
}

/// The heat loss H (K/m) of the energy balance dT/dz = -H, at temperature t and ambient temperature t_a (K), for
/// the heat transfer coefficient h (W/(m² K)), the pipe's inner diameter d (m), and the gas and liquid heat flux
/// capacities cp_g α ρ_g v_g and cp_l (1 - α) ρ_l v_l (W/(m² K))
pub fn heat_loss(h: f64, d: f64, t: f64, t_a: f64, gas_capacity: f64, liquid_capacity: f64) -> f64 {  // spec: THM-1
    4.0 * h * (t - t_a) / (d * (gas_capacity + liquid_capacity))
}

/// Temperature of the fluid entering the well (K)
pub fn inflow_temperature(t_r: f64) -> f64 {  // spec: THM-3
    t_r
}
