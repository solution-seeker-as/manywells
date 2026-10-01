// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The march: given the bottomhole pressure p_0, the inflow gives the phase rates, and the state at every point
//! follows from the rows of discretization.rs, point by point up the well. At each point the closures give the
//! densities and the void fraction at a trial pressure, and the cell's momentum row is solved for the pressure.
//! What is left at the wellhead is the choke row (CHK-1), whose value is the shooting residual R(p_0) (shoot.rs).
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

/// The phase rates of a march, the same at every point (BAL-3)
#[derive(Clone, Copy, Debug)]
pub struct Rates {
    pub w_res: f64, // Reservoir liquid rate (kg/s)
    pub w_g: f64,   // Gas rate, with the lift gas (kg/s)
    pub w_l: f64,   // Liquid rate (kg/s)
}

/// A march from p_0 to the wellhead
pub struct March {
    /// The state at every point the march reached
    pub x: Vec<f64>,
    pub rates: Rates,
    /// A cell had no subsonic root (choked), a state could not be computed, or the march fell below p_s, so x is
    /// not a solution of the rows
    pub failed: bool,
    /// The pressure fell below p_s, and the march stopped there
    pub below_separator: bool,
}

enum CellStep {
    /// The cell's momentum row is zero at this pressure
    Solved(f64),
    /// No subsonic root above p_s: the march continues at the pressure p* where the row is smallest, so that R stays
    /// continuous in p_0, but it is not a solution
    Choked(f64),
    /// The row is positive down to p_s, so its subsonic root, if any, lies below p_s
    BelowSeparator,
}

pub struct Marcher<'a> {
    pub spec: &'a WellSpec,
    pub op: &'a OperatingPoint,
    marches: Cell<usize>,
}

impl<'a> Marcher<'a> {
    pub fn new(spec: &'a WellSpec, op: &'a OperatingPoint) -> Self {
        Self { spec, op, marches: Cell::new(0) }
    }

    /// The number of marches so far
    pub fn marches(&self) -> usize {
        self.marches.get()
    }

    pub fn rates(&self, p_0: f64) -> Rates {
        let w_res = discretization::reservoir_rate(self.spec, self.op, p_0);
        let (w_g, w_l) = self.spec.fluid.phase_rates(w_res, self.op.w_lg);
        Rates { w_res, w_g, w_l }
    }

    /// Temperature at every point: the inflow temperature at the bottomhole, and each cell's energy row solved for
    /// the temperature at its upper point. It does not depend on the pressure, so it is computed once per march.
    fn temperatures(&self, rates: &Rates) -> Vec<f64> {
        let (spec, op) = (self.spec, self.op);
        let a = spec.a();
        let capacity = spec.fluid.cp_g * (rates.w_g / a) + spec.fluid.cp_l * (rates.w_l / a);
        let mut t = vec![thermal::inflow_temperature(op.t_r)];
        for i in 1..=spec.n_cells {
            let t_a = thermal::ambient_temperature(i, spec.n_cells, op.t_r, op.t_s);
            t.push(thermal::energy_step(t[i - 1], t_a, spec.delta_z(), spec.h, spec.d, capacity));
        }
        t
    }

    /// The void fraction that zeroes the slip row (SLIP-1) at a point where the mass rows fix the superficial
    /// velocities j_g and j_l (m/s): Brent on h(α) = α (C_0 j_m + v_inf) - j_g, which is -α times the slip row with
    /// v_g = j_g / α. The classifier sees α only through its features c_2 and c_4, and C_0 >= 1 and v_inf >= 0 for
    /// every mix of the regimes, so h(0) = -j_g < 0 and h(1) >= j_l + v_inf > 0: the bracket [0, 1] always holds a
    /// root. None where the rise velocities are not real (rho_g >= rho_l).
    fn void_fraction(&self, rates: &Rates, rho_g: f64, rho_l: f64, t: f64) -> Option<f64> {
        if !(rho_g > 0.0 && rho_g < rho_l) {
            return None;
        }
        let a = self.spec.a();
        let (j_g, j_l) = (rates.w_g / (a * rho_g), rates.w_l / (a * rho_l));
        let j_m = j_g + j_l;
        let sigma = self.spec.fluid.surface_tension(rho_l, t);
        let terms = slip::SlipTerms::new(j_g, j_l, rho_g, rho_l, sigma, self.spec.d);
        let mut h = |alpha: f64| -> Result<f64, RootError> {
            let (c_0, v_inf) = terms.parameters(alpha);
            Ok(alpha * (c_0 * j_m + v_inf) - j_g)
        };
        brentq(&mut h, 0.0, 1.0, 0.0, RTOL, 100).ok()
    }

    /// The state at a point at pressure p and temperature t: the closures at the march's rates
    fn point_state(&self, p: f64, t: f64, rates: &Rates) -> Option<State> {
        let (a, fluid) = (self.spec.a(), &self.spec.fluid);
        let rho_g = fluid.gas_density(p, t);
        let rho_l = fluid.liquid_density();
        let alpha = self.void_fraction(rates, rho_g, rho_l, t)?;
        let v_g = rates.w_g / (a * alpha * rho_g);
        let v_l = rates.w_l / (a * (1.0 - alpha) * rho_l);
        Some(State { p, v_g, v_l, alpha, rho_g, rho_l, t })
    }

    /// The pressure at a point at temperature t from the momentum row of the cell below it, given the state at the
    /// cell's lower point, searched on [p_s, p_prev]
    fn solve_cell(&self, t: f64, s_prev: &State, rates: &Rates) -> CellStep {
        let (p_prev, p_s) = (s_prev.p, self.op.p_s);
        let mut row = |p: f64| -> Result<f64, RootError> {
            let s = self.point_state(p, t, rates).ok_or(RootError::NoSignChange)?;
            Ok(discretization::momentum_row(self.spec, &s, s_prev))
        };
        // The row is U-shaped in p, with its minimum at the cell's sonic pressure p*, and positive at p_prev: the
        // subsonic root lies between p* and p_prev, usually close to p_prev, so a narrow bracket is tried first
        let lo = p_prev - 0.1 * (p_prev - p_s);
        if let Ok(p) = brentq(&mut row, lo, p_prev, CELL_XTOL, RTOL, 100) {
            return CellStep::Solved(p);
        }
        // Below zero at p_s: p_s lies right of p*, or left of it where the row still falls, so the only sign change
        // on [p_s, p_prev] is the subsonic root
        let f_s = match row(p_s) {
            Ok(f) => f,
            Err(_) => return CellStep::Choked(p_prev),
        };
        if f_s < 0.0 {
            return match brentq(&mut row, p_s, p_prev, CELL_XTOL, RTOL, 100) {
                Ok(p) => CellStep::Solved(p),
                Err(_) => CellStep::Choked(p_prev),
            };
        }
        let (p_star, f_star) = minimize(&mut |p| match row(p) {
            Ok(v) if v.is_finite() => v,
            _ => f64::INFINITY,
        }, p_s, p_prev, 1e-2, 200, 0.0);
        if f_star < 0.0 {
            return match brentq(&mut row, p_star, p_prev, CELL_XTOL, RTOL, 100) {
                Ok(p) => CellStep::Solved(p),
                Err(_) => CellStep::Choked(p_star),
            };
        }
        if f_s <= f_star { CellStep::BelowSeparator } else { CellStep::Choked(p_star) }
    }

    /// March from p_0 to the wellhead
    pub fn march(&self, p_0: f64) -> March {
        self.marches.set(self.marches.get() + 1);
        let n = self.spec.n_cells;
        let rates = self.rates(p_0);
        let t = self.temperatures(&rates);
        let mut x = Vec::with_capacity(DIM_X * (n + 1));
        let mut failed = false;
        let Some(mut prev) = self.point_state(p_0, t[0], &rates) else {
            x.extend_from_slice(&[f64::NAN; DIM_X]);
            return March { x, rates, failed: true, below_separator: false };
        };
        x.extend_from_slice(&prev.to_array());
        for i in 1..=n {
            let p = match self.solve_cell(t[i], &prev, &rates) {
                CellStep::Solved(p) => p,
                CellStep::Choked(p) => {
                    failed = true;
                    p
                }
                CellStep::BelowSeparator => return March { x, rates, failed: true, below_separator: true },
            };
            let Some(s) = self.point_state(p, t[i], &rates) else {
                x.extend_from_slice(&[f64::NAN; DIM_X]);
                return March { x, rates, failed: true, below_separator: false };
            };
            x.extend_from_slice(&s.to_array());
            prev = s;
        }
        March { x, rates, failed, below_separator: false }
    }

    /// The shooting residual R(p_0) (kg/s), the choke row at the wellhead, and whether the march failed; None if R
    /// is not finite
    pub fn residual(&self, p_0: f64) -> Option<(f64, bool)> {
        let m = self.march(p_0);
        let r = if m.below_separator {
            m.rates.w_g + m.rates.w_l // spec: CHK-11
        } else {
            discretization::choke_row(self.spec, self.op, &State::of(&m.x[m.x.len() - DIM_X..]))
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
            for id in ["INF-6", "INF-7", "THM-3", "SLIP-1", "PVT-GAS-1", "PVT-MIX-1", "DISC-2", "DISC-3", "DISC-5"] {
                assert!(rows[id] < 1e-10, "{id}: {}", rows[id]);
            }
        }
    }
}
