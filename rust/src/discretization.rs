// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The discretized system of a well (specs/model/discretization.md): the rows at each grid point, in the order of
//! DISC-11, as functions of the state. Each row is defined once, here; the march (march.rs) solves them point by
//! point, and the bindings evaluate them for the tests against v1.0.0's row vectors and the CasADi backend's rows.
//!
//! The state holds seven values at each of the N + 1 points, from the bottomhole (point 0) to the wellhead (point N):
//! [p, v_g, v_l, alpha, rho_g, rho_l, T] (bar, m/s, m/s, -, kg/m³, kg/m³, K). Cell i lies between points i - 1 and i.
//! The reservoir liquid rate w_res, from the inflow at p_0, enters every point's rows through the phase rates.

use crate::geometry::Cell;
use crate::input::{OperatingPoint, WellSpec};
use crate::pvt::fluid::GasLaw;
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
pub fn bottom_rows(spec: &WellSpec, op: &OperatingPoint, s: &State, w_res: f64) -> [f64; 3] {
    let a = spec.a();
    let (w_g, w_l) = spec.fluid.phase_rates(s.p, s.t, w_res, op.w_lg);
    [
        a * s.alpha * s.rho_g * s.v_g - w_g,         // spec: INF-6
        a * (1.0 - s.alpha) * s.rho_l * s.v_l - w_l, // spec: INF-7
        s.t - spec.thermal.inflow_temperature(w_res, op.w_lg, op.t_r, op.t_lg, &spec.fluid), // THM-3 or THM-5
    ]
}

/// The viscous pressure gradient (Pa/m) at a point, which the momentum and energy rows share
pub fn friction_gradient(spec: &WellSpec, s: &State) -> f64 {
    spec.friction.pressure_gradient(s, &spec.fluid, spec.geometry.d)
}

/// The momentum row of cell i (bar), between points i - 1 (s_prev) and i (s): implicit Euler, with friction along the
/// flow path and gravity along the vertical, at point i
pub fn momentum_row(spec: &WellSpec, cell: Cell, s: &State, s_prev: &State) -> f64 {
    let f = friction_gradient(spec, s);
    let g = STD_GRAVITY * s.rho_m(); // spec: BAL-6
    (s.momentum_flux() / CF_BAR + s.p) - (s_prev.momentum_flux() / CF_BAR + s_prev.p)
        + cell.delta_md * (f + cell.cos_incl * g) / CF_BAR // spec: DISC-9
}

/// The energy row of cell i (K): implicit Euler, with the temperature gradient at point i
pub fn energy_row(spec: &WellSpec, op: &OperatingPoint, cell: Cell, s: &State, s_prev: &State) -> f64 {
    let t_a = thermal::ambient_temperature(cell.tvd_frac, op.t_r, op.t_s);
    let dt = spec.thermal.temperature_gradient(s, &spec.fluid, t_a, friction_gradient(spec, s), cell.cos_incl,
                                               spec.geometry.d);
    s.t - s_prev.t - cell.delta_md * dt // spec: DISC-10
}

/// Balance rows of cell i: gas and liquid mass (kg/(m² s)), momentum (bar) and energy (K). The change in each mass
/// flux equals the change in the phase rate, which is zero without mass transfer.
pub fn cell_rows(spec: &WellSpec, op: &OperatingPoint, cell: Cell, s: &State, s_prev: &State, w_res: f64) -> [f64; 4] {
    let a = spec.a();
    let (w_g, w_l) = spec.fluid.phase_rates(s.p, s.t, w_res, op.w_lg);
    let (w_g_prev, w_l_prev) = spec.fluid.phase_rates(s_prev.p, s_prev.t, w_res, op.w_lg);
    [
        s.gas_flux() - s_prev.gas_flux() - (w_g - w_g_prev) / a,          // spec: DISC-7
        s.liquid_flux() - s_prev.liquid_flux() - (w_l - w_l_prev) / a,    // spec: DISC-8
        momentum_row(spec, cell, s, s_prev),
        energy_row(spec, op, cell, s, s_prev),
    ]
}

/// The wellhead row (kg/s): the rate leaving the tubing minus the rate the choke passes
pub fn choke_row(spec: &WellSpec, op: &OperatingPoint, s: &State) -> f64 {  // spec: CHK-1
    let (w_g, w_l) = s.rates(spec.a());
    let w_m = w_g + w_l;
    w_m - spec.choke.mass_flow_rate(op.u, s.p, op.p_s, w_g, w_l, s.alpha, s.rho_g, s.rho_l)
}

/// Closure relations at a point in a cell of inclination cos_incl: the slip law (m/s), the gas law (bar) and the liquid
/// density (kg/m³)
pub fn closure_rows(spec: &WellSpec, s: &State, cos_incl: f64) -> [f64; 3] {
    let sigma = spec.fluid.surface_tension(s.p, s.t, s.rho_l);
    let (c_0, v_inf) = spec.slip.identify_parameters(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, spec.geometry.d,
                                                     cos_incl);
    [
        s.v_g - c_0 * s.v_m() - v_inf, // spec: SLIP-1
        spec.fluid.gas_law_row(s.p, s.t, s.rho_g),
        spec.fluid.liquid_density_row(s.p, s.t, s.rho_l),
    ]
}

/// The spec ID of every row of the system, in order: the rows of each point depend on the well's options
pub fn row_ids(spec: &WellSpec) -> Vec<&'static str> {  // spec: DISC-11
    let n = spec.n_cells();
    let bottom = ["INF-6", "INF-7", if spec.thermal.lift_gas_mixing { "THM-5" } else { "THM-3" }];
    let cell = ["DISC-7", "DISC-8", "DISC-9", "DISC-10"];
    let gas_law = if spec.fluid.gas_law == GasLaw::Ideal { "PVT-GAS-1" } else { "PVT-GAS-3" };
    let liquid = if spec.fluid.has_mass_transfer() { "PVT-MIX-6" } else { "PVT-MIX-1" };
    let closures = ["SLIP-1", gas_law, liquid];
    let mut ids = Vec::with_capacity(DIM_X * (n + 1));
    ids.extend(bottom);
    ids.extend(closures);
    for i in 1..=n {
        ids.extend(cell);
        if i == n {
            ids.push("CHK-1");
        }
        ids.extend(closures);
    }
    ids
}

/// Every row of the system at state x, with its ID, point by point in the order of DISC-11
pub fn rows(spec: &WellSpec, op: &OperatingPoint, x: &[f64]) -> Vec<(&'static str, f64)> {
    let n = spec.n_cells();
    let points: Vec<State> = x.chunks_exact(DIM_X).map(State::of).collect();
    let w_res = reservoir_rate(spec, op, points[0].p);
    let mut values = Vec::with_capacity(DIM_X * (n + 1));
    values.extend(bottom_rows(spec, op, &points[0], w_res));
    values.extend(closure_rows(spec, &points[0], spec.geometry.point_cos(0)));
    for i in 1..=n {
        values.extend(cell_rows(spec, op, spec.geometry.cell(i), &points[i], &points[i - 1], w_res));
        if i == n {
            values.push(choke_row(spec, op, &points[n]));
        }
        values.extend(closure_rows(spec, &points[i], spec.geometry.point_cos(i)));
    }
    row_ids(spec).into_iter().zip(values).collect()
}

/// The regime label at each point (SLIP-8)
pub fn flow_regimes(spec: &WellSpec, x: &[f64]) -> Vec<&'static str> {
    x.chunks_exact(DIM_X).map(State::of).enumerate().map(|(i, s)| {
        let sigma = spec.fluid.surface_tension(s.p, s.t, s.rho_l);
        let cos_incl = spec.geometry.point_cos(i);
        slip::regime_label(spec.slip.classify(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, spec.geometry.d, cos_incl))
    }).collect()
}
