// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The thermal model (specs/model/thermal.md). So far heat loss to an ambient profile linear in true vertical depth,
//! and the fluid enters at the reservoir temperature.

/// Ambient temperature (K) at a point whose true vertical depth is tvd_frac of the bottomhole's, linear from T_s at
/// the surface to T_r at the bottomhole
pub fn ambient_temperature(tvd_frac: f64, t_r: f64, t_s: f64) -> f64 {  // spec: THM-4
    t_s + (t_r - t_s) * tvd_frac
}

/// The heat loss H (K/m) of the energy balance dT/dz = -H, at temperature t and ambient temperature t_a (K), for
/// the heat transfer coefficient h (W/(m² K)), the pipe's inner diameter d (m), and the gas and liquid heat flux
/// capacities cp_g α ρ_g v_g and cp_l (1 - α) ρ_l v_l (W/(m² K))
pub fn heat_loss(h: f64, d: f64, t: f64, t_a: f64, gas_capacity: f64, liquid_capacity: f64) -> f64 {  // spec: THM-1
    4.0 * h * (t - t_a) / (d * (gas_capacity + liquid_capacity))
}

/// The temperature (K) at point i that zeroes the energy row of cell i (DISC-10 with THM-1 alone), given t_prev at
/// point i - 1, the ambient temperature t_a at point i and the cell's length delta_md (m). The row is linear in it,
/// because the heat flux capacity cp_g α ρ_g v_g + cp_l (1 - α) ρ_l v_l = cp_g G_g + cp_l G_l (W/(m² K)) is fixed by the
/// mass fluxes G (kg/(m² s)), which the mass rows hold constant without mass transfer:
/// T_i = (T_{i-1} + ΔMD k T_a,i) / (1 + ΔMD k) with k = 4h / (D capacity).
pub fn energy_step(t_prev: f64, t_a: f64, delta_md: f64, h: f64, d: f64, capacity: f64) -> f64 {  // spec: DISC-10, THM-1
    let k = 4.0 * h / (d * capacity);
    (t_prev + delta_md * k * t_a) / (1.0 + delta_md * k)
}

/// Temperature of the fluid entering the well (K)
pub fn inflow_temperature(t_r: f64) -> f64 {  // spec: THM-3
    t_r
}
