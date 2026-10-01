// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The march: given the bottomhole pressure p_0, the inflow gives the reservoir liquid rate w_res, and the state at
//! every point follows from the rows of discretization.rs, point by point up the well. At each point the phase
//! rates, the densities and the void fraction follow from the closures at a trial pressure and temperature, and the
//! cell's momentum row is solved for the pressure. What is left at the wellhead is the choke row (CHK-1), whose value
//! is the shooting residual R(p_0) (shoot.rs).
//!
//! The pressure falls up the well, so once it falls below the separator pressure p_s it stays there: the wellhead
//! is then at or below the critical pressure, the choke passes no flow (CHK-11), and R = w_m without marching on.

use std::cell::Cell;

use crate::discretization::{self, State, DIM_X};
use crate::input::{OperatingPoint, WellSpec};
use crate::scalar::{brentq, minimize, RootError, RTOL};
use crate::slip;
use crate::thermal;

/// Absolute tolerance of the cell's Brent on the pressure (bar): none, so that it converges to a few ulp
const CELL_XTOL: f64 = 0.0;

/// Largest |momentum row| (bar) at which a cell counts as solved. At a root of the row, Brent leaves about 1e-13 bar;
/// where the closures jump between solutions, as the slip law can where it has several void fractions, the row jumps
/// across zero instead of crossing it, by far more than this, and the cell is not solved.
const CELL_ROW_TOL: f64 = 1e-8;

/// A march from p_0 to the wellhead
pub struct March {
    /// The state at every point the march reached
    pub x: Vec<f64>,
    /// The reservoir liquid rate (kg/s), from the inflow at p_0
    pub w_res: f64,
    /// A cell's momentum row was not solved (choked, or a jump), a state could not be computed, or the march fell
    /// below p_s, so x is not a solution of the rows
    pub failed: bool,
    /// The pressure fell below p_s, and the march stopped there
    pub below_separator: bool,
}

enum CellStep {
    /// The cell's momentum row is zero at this pressure
    Solved(f64),
    /// No subsonic root above p_s (choked), or the row jumps across zero: the march continues at the pressure p*
    /// where the row is smallest, or at the jump, so that R stays continuous in p_0, but it is not a solution
    Unsolved(f64),
    /// The row is positive down to p_s, so its subsonic root, if any, lies below p_s
    BelowSeparator,
}

/// How much work the searches of one marcher did, for the measurements of the feature specs
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    /// Marches from a p_0
    pub marches: usize,
    /// States computed from the closures at a trial pressure and temperature
    pub states: usize,
}

pub struct Marcher<'a> {
    pub spec: &'a WellSpec,
    pub op: &'a OperatingPoint,
    counts: Cell<Counts>,
}

impl<'a> Marcher<'a> {
    pub fn new(spec: &'a WellSpec, op: &'a OperatingPoint) -> Self {
        Self { spec, op, counts: Cell::new(Counts::default()) }
    }

    pub fn counts(&self) -> Counts {
        self.counts.get()
    }

    fn count(&self, f: impl FnOnce(&mut Counts)) {
        let mut c = self.counts.get();
        f(&mut c);
        self.counts.set(c);
    }

    /// The temperature at point i that zeroes the energy row of cell i, given the state at point i - 1. Without mass
    /// transfer the phase rates are the same at every point, so the row's heat flux capacity is fixed by them, and
    /// the row is linear in the temperature and does not depend on the pressure (thermal::energy_step).
    fn temperature(&self, i: usize, prev: &State, w_res: f64) -> f64 {
        let (spec, op) = (self.spec, self.op);
        let a = spec.a();
        let (w_g, w_l) = spec.fluid.phase_rates(prev.p, prev.t, w_res, op.w_lg);
        let capacity = spec.fluid.cp_g * (w_g / a) + spec.fluid.cp_l * (w_l / a);
        let t_a = thermal::ambient_temperature(i, spec.n_cells, op.t_r, op.t_s);
        thermal::energy_step(prev.t, t_a, spec.delta_z(), spec.h, spec.d, capacity)
    }

    /// The void fraction that zeroes the slip row (SLIP-1) at a point where the phase rates fix the superficial
    /// velocities j_g and j_l (m/s): Brent on h(α) = α (C_0 j_m + v_inf) - j_g, which is -α times the slip row with
    /// v_g = j_g / α. The classifier sees α only through its features c_2 and c_4, and C_0 >= 1 and v_inf >= 0 for
    /// every mix of the regimes, so h(0) = -j_g < 0 and h(1) >= j_l + v_inf > 0: the bracket [0, 1] always holds a
    /// root. None where the rise velocities are not real (rho_g >= rho_l).
    fn void_fraction(&self, j_g: f64, j_l: f64, rho_g: f64, rho_l: f64, sigma: f64) -> Option<f64> {
        if !(rho_g > 0.0 && rho_g < rho_l) {
            return None;
        }
        let j_m = j_g + j_l;
        let terms = slip::SlipTerms::new(j_g, j_l, rho_g, rho_l, sigma, self.spec.d);
        let mut h = |alpha: f64| -> Result<f64, RootError> {
            let (c_0, v_inf) = terms.parameters(alpha);
            Ok(alpha * (c_0 * j_m + v_inf) - j_g)
        };
        brentq(&mut h, 0.0, 1.0, 0.0, RTOL, 100).ok().map(|(alpha, _)| alpha)
    }

    /// The state at a point at pressure p and temperature t: the phase rates there, the densities, the surface
    /// tension, the void fraction from the slip law, and the velocities that carry the rates. So the inflow and mass
    /// rows (INF-6, INF-7, DISC-7, DISC-8) and the closure rows hold by construction.
    fn point_state(&self, p: f64, t: f64, w_res: f64) -> Option<State> {
        self.count(|c| c.states += 1);
        let (a, fluid) = (self.spec.a(), &self.spec.fluid);
        let (w_g, w_l) = fluid.phase_rates(p, t, w_res, self.op.w_lg);
        let rho_g = fluid.gas_density(p, t);
        let rho_l = fluid.liquid_density();
        let sigma = fluid.surface_tension(rho_l, t);
        let alpha = self.void_fraction(w_g / (a * rho_g), w_l / (a * rho_l), rho_g, rho_l, sigma)?;
        let v_g = w_g / (a * alpha * rho_g);
        let v_l = w_l / (a * (1.0 - alpha) * rho_l);
        Some(State { p, v_g, v_l, alpha, rho_g, rho_l, t })
    }

    /// The pressure at a point at temperature t from the momentum row of the cell below it, given the state at the
    /// cell's lower point, searched on [p_s, p_prev]
    fn solve_cell(&self, t: f64, s_prev: &State, w_res: f64) -> CellStep {
        let (p_prev, p_s) = (s_prev.p, self.op.p_s);
        let mut row = |p: f64| -> Result<f64, RootError> {
            let s = self.point_state(p, t, w_res).ok_or(RootError::NoSignChange)?;
            Ok(discretization::momentum_row(self.spec, &s, s_prev))
        };
        // The row is U-shaped in p, with its minimum at the cell's sonic pressure p*, and positive at p_prev. Below
        // zero at p_s: p_s lies right of p*, or left of it where the row still falls, so the only sign change on
        // [p_s, p_prev] is the subsonic root
        let solved = |(p, f): (f64, f64)| if f.abs() <= CELL_ROW_TOL { CellStep::Solved(p) } else { CellStep::Unsolved(p) };
        let f_s = match row(p_s) {
            Ok(f) => f,
            Err(_) => return CellStep::Unsolved(p_prev),
        };
        if f_s < 0.0 {
            return match brentq(&mut row, p_s, p_prev, CELL_XTOL, RTOL, 100) {
                Ok(root) => solved(root),
                Err(_) => CellStep::Unsolved(p_prev),
            };
        }
        // Otherwise the row's minimum on [p_s, p_prev] decides: below zero, the subsonic root lies between it and
        // p_prev; at p_s, the row falls all the way to p_s; elsewhere above zero, the cell is choked
        let (p_star, f_star) = minimize(&mut |p| match row(p) {
            Ok(v) if v.is_finite() => v,
            _ => f64::INFINITY,
        }, p_s, p_prev, 1e-2, 200, f64::NEG_INFINITY);
        if f_star < 0.0 {
            return match brentq(&mut row, p_star, p_prev, CELL_XTOL, RTOL, 100) {
                Ok(root) => solved(root),
                Err(_) => CellStep::Unsolved(p_star),
            };
        }
        if f_s <= f_star { CellStep::BelowSeparator } else { CellStep::Unsolved(p_star) }
    }

    /// March from p_0 to the wellhead
    pub fn march(&self, p_0: f64) -> March {
        self.count(|c| c.marches += 1);
        let n = self.spec.n_cells;
        let w_res = discretization::reservoir_rate(self.spec, self.op, p_0);
        let mut x = Vec::with_capacity(DIM_X * (n + 1));
        let mut failed = false;
        let Some(mut prev) = self.point_state(p_0, thermal::inflow_temperature(self.op.t_r), w_res) else {
            x.extend_from_slice(&[f64::NAN; DIM_X]);
            return March { x, w_res, failed: true, below_separator: false };
        };
        x.extend_from_slice(&prev.to_array());
        for i in 1..=n {
            let t = self.temperature(i, &prev, w_res);
            let p = match self.solve_cell(t, &prev, w_res) {
                CellStep::Solved(p) => p,
                CellStep::Unsolved(p) => {
                    failed = true;
                    p
                }
                CellStep::BelowSeparator => return March { x, w_res, failed: true, below_separator: true },
            };
            let Some(s) = self.point_state(p, t, w_res) else {
                x.extend_from_slice(&[f64::NAN; DIM_X]);
                return March { x, w_res, failed: true, below_separator: false };
            };
            x.extend_from_slice(&s.to_array());
            prev = s;
        }
        March { x, w_res, failed, below_separator: false }
    }

    /// The shooting residual R(p_0) (kg/s), the choke row at the wellhead, and whether the march failed; None if R
    /// is not finite. Below p_s the choke passes no flow, and R is the rate at the last point reached.
    pub fn residual(&self, p_0: f64) -> Option<(f64, bool)> {
        let m = self.march(p_0);
        let last = State::of(&m.x[m.x.len() - DIM_X..]);
        let r = if m.below_separator {
            let (w_g, w_l) = self.spec.fluid.phase_rates(last.p, last.t, m.w_res, self.op.w_lg);
            w_g + w_l // spec: CHK-11
        } else {
            discretization::choke_row(self.spec, self.op, &last)
        };
        r.is_finite().then_some((r, m.failed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::test_wells::{w1, w2};

    /// The largest |row| of each ID over a march's state
    fn largest_rows(spec: &WellSpec, op: &OperatingPoint, x: &[f64]) -> std::collections::HashMap<&'static str, f64> {
        let mut out = std::collections::HashMap::new();
        for (id, v) in discretization::rows(spec, op, x) {
            let e = out.entry(id).or_insert(0.0_f64);
            *e = e.max(v.abs());
        }
        out
    }

    #[test]
    fn a_march_zeroes_the_inflow_mass_energy_and_closure_rows() {
        for (spec, op) in [w1(20), w2(20)] {
            let m = Marcher::new(&spec, &op);
            let p_0 = op.p_s + 0.8 * (op.p_r - op.p_s);
            let march = m.march(p_0);
            assert!(!march.failed);
            let rows = largest_rows(&spec, &op, &march.x);
            for id in ["INF-6", "INF-7", "THM-3", "SLIP-1", "PVT-GAS-1", "PVT-MIX-1", "DISC-7", "DISC-8", "DISC-10"] {
                assert!(rows[id] < 1e-10, "{id}: {}", rows[id]);
            }
        }
    }
}
