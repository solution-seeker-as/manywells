// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The discretized system of a well (specs/model/discretization.md): the rows at each grid point, in the order of
//! DISC-6, as functions of the state. Each row is defined once, here; the march (march.rs) solves them point by
//! point, and the bindings evaluate them for the tests against v1.0.0's row vectors.
//!
//! The state holds seven values at each of the N + 1 points, from the bottomhole (point 0) to the wellhead (point N):
//! [p, v_g, v_l, alpha, rho_g, rho_l, T] (bar, m/s, m/s, -, kg/m³, kg/m³, K). Cell i lies between points i - 1 and i.

use crate::friction;
use crate::input::{OperatingPoint, WellSpec};
use crate::slip;
use crate::thermal;
use crate::units::{CF_BAR, STD_GRAVITY};

pub const DIM_X: usize = 7;

/// The state at one point
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub p: f64,
    pub v_g: f64,
    pub v_l: f64,
    pub alpha: f64,
    pub rho_g: f64,
    pub rho_l: f64,
    pub t: f64,
}

impl State {
    pub fn of(x: &[f64]) -> Self {
        Self { p: x[0], v_g: x[1], v_l: x[2], alpha: x[3], rho_g: x[4], rho_l: x[5], t: x[6] }
    }

    pub fn to_array(&self) -> [f64; DIM_X] {
        [self.p, self.v_g, self.v_l, self.alpha, self.rho_g, self.rho_l, self.t]
    }

    /// Mixture density (kg/m³)
    pub fn rho_m(&self) -> f64 {  // spec: BAL-7
        self.alpha * self.rho_g + (1.0 - self.alpha) * self.rho_l
    }

    /// Mixture velocity (m/s)
    pub fn v_m(&self) -> f64 {  // spec: BAL-8
        self.alpha * self.v_g + (1.0 - self.alpha) * self.v_l
    }

    /// Gas mass flux (kg/(m² s))
    pub fn gas_flux(&self) -> f64 {
        self.alpha * self.rho_g * self.v_g
    }

    /// Liquid mass flux (kg/(m² s))
    pub fn liquid_flux(&self) -> f64 {
        (1.0 - self.alpha) * self.rho_l * self.v_l
    }

    /// Momentum flux (Pa)
    pub fn momentum_flux(&self) -> f64 {
        self.alpha * self.rho_g * (self.v_g * self.v_g) + (1.0 - self.alpha) * self.rho_l * (self.v_l * self.v_l)
    }

    /// Gas and liquid mass rates (kg/s) through a cross-section a (m²)
    pub fn rates(&self, a: f64) -> (f64, f64) {
        (a * self.alpha * self.rho_g * self.v_g, a * (1.0 - self.alpha) * self.rho_l * self.v_l)
    }
}

/// The reservoir liquid rate (kg/s) at bottomhole pressure p_0 (bar)
pub fn reservoir_rate(spec: &WellSpec, op: &OperatingPoint, p_0: f64) -> f64 {
    spec.inflow.liquid_rate(p_0, op.p_r)
}

/// Rows at the bottomhole, point 0: the gas and liquid inflow (kg/s) and the inflow temperature (K)
pub fn bottom_rows(spec: &WellSpec, op: &OperatingPoint, s: &State) -> [f64; 3] {
    let a = spec.a();
    let (w_g, w_l) = spec.fluid.phase_rates(reservoir_rate(spec, op, s.p), op.w_lg);
    [
        a * s.alpha * s.rho_g * s.v_g - w_g,         // spec: INF-6
        a * (1.0 - s.alpha) * s.rho_l * s.v_l - w_l, // spec: INF-7
        s.t - thermal::inflow_temperature(op.t_r),   // spec: THM-3
    ]
}

/// The momentum row of cell i (bar), between points i - 1 (s_prev) and i (s): implicit Euler, with friction and
/// gravity at point i
pub fn momentum_row(spec: &WellSpec, s: &State, s_prev: &State) -> f64 {  // spec: DISC-4, BAL-4
    let f = friction::pressure_gradient(spec.f_d, spec.d, s.rho_m(), s.v_m());
    let g = STD_GRAVITY * s.rho_m(); // spec: BAL-6
    (s.momentum_flux() / CF_BAR + s.p) - (s_prev.momentum_flux() / CF_BAR + s_prev.p) + spec.delta_z() * (f + g) / CF_BAR
}

/// The heat flux capacities cp_g α ρ_g v_g and cp_l (1 - α) ρ_l v_l (W/(m² K)) at a point
pub fn heat_capacities(spec: &WellSpec, s: &State) -> (f64, f64) {
    (spec.fluid.cp_g * s.alpha * s.rho_g * s.v_g, spec.fluid.cp_l * (1.0 - s.alpha) * s.rho_l * s.v_l)
}

/// The energy row of cell i (K): implicit Euler, with the heat loss at point i
pub fn energy_row(spec: &WellSpec, op: &OperatingPoint, i: usize, s: &State, s_prev: &State) -> f64 {  // spec: DISC-5, BAL-5
    let t_a = thermal::ambient_temperature(i, spec.n_cells, op.t_r, op.t_s);
    let (c_g, c_l) = heat_capacities(spec, s);
    s.t - s_prev.t + spec.delta_z() * thermal::heat_loss(spec.h, spec.d, s.t, t_a, c_g, c_l)
}

/// Balance rows of cell i: gas and liquid mass (kg/(m² s)), momentum (bar) and energy (K)
pub fn cell_rows(spec: &WellSpec, op: &OperatingPoint, i: usize, s: &State, s_prev: &State) -> [f64; 4] {
    [
        s.gas_flux() - s_prev.gas_flux(),       // spec: DISC-2, BAL-1, BAL-3
        s.liquid_flux() - s_prev.liquid_flux(), // spec: DISC-3, BAL-2, BAL-3
        momentum_row(spec, s, s_prev),
        energy_row(spec, op, i, s, s_prev),
    ]
}

/// The wellhead row (kg/s): the rate leaving the tubing minus the rate the choke passes
pub fn choke_row(spec: &WellSpec, op: &OperatingPoint, s: &State) -> f64 {  // spec: CHK-1
    let (w_g, w_l) = s.rates(spec.a());
    let w_m = w_g + w_l;
    w_m - spec.choke.mass_flow_rate(op.u, s.p, op.p_s, w_g, w_l, s.alpha, s.rho_g, s.rho_l)
}

/// Closure relations at a point: the slip law (m/s), the gas law (bar) and the liquid density (kg/m³)
pub fn closure_rows(spec: &WellSpec, s: &State) -> [f64; 3] {
    let sigma = spec.fluid.surface_tension(s.rho_l, s.t);
    let (c_0, v_inf) = slip::identify_parameters(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, spec.d);
    [
        s.v_g - c_0 * s.v_m() - v_inf, // spec: SLIP-1
        spec.fluid.gas_law_row(s.p, s.t, s.rho_g),
        spec.fluid.liquid_density_row(s.rho_l),
    ]
}

/// Every row of the system at state x, with its ID, point by point in the order of DISC-6
pub fn rows(spec: &WellSpec, op: &OperatingPoint, x: &[f64]) -> Vec<(&'static str, f64)> {  // spec: DISC-6
    let n = spec.n_cells;
    let points: Vec<State> = x.chunks_exact(DIM_X).map(State::of).collect();
    let closure_ids = ["SLIP-1", "PVT-GAS-1", "PVT-MIX-1"];
    let mut out = Vec::with_capacity(DIM_X * (n + 1));
    out.extend(["INF-6", "INF-7", "THM-3"].into_iter().zip(bottom_rows(spec, op, &points[0])));
    out.extend(closure_ids.into_iter().zip(closure_rows(spec, &points[0])));
    for i in 1..=n {
        out.extend(["DISC-2", "DISC-3", "DISC-4", "DISC-5"].into_iter()
            .zip(cell_rows(spec, op, i, &points[i], &points[i - 1])));
        if i == n {
            out.push(("CHK-1", choke_row(spec, op, &points[n])));
        }
        out.extend(closure_ids.into_iter().zip(closure_rows(spec, &points[i])));
    }
    out
}

/// The regime label at each point (SLIP-8)
pub fn flow_regimes(spec: &WellSpec, x: &[f64]) -> Vec<&'static str> {
    x.chunks_exact(DIM_X).map(State::of).map(|s| {
        let sigma = spec.fluid.surface_tension(s.rho_l, s.t);
        slip::regime_label(slip::classify(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, spec.d))
    }).collect()
}
