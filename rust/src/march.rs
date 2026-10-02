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
use crate::geometry;
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

/// Largest error (K) of a solved temperature: |energy row| over the row's slope in T from the heat loss,
/// 1 + ΔMD 4h / (D cp_flux), which is large where the heat flux capacity is small. At a root, Brent leaves about
/// 1e-13 K; a jump in the closures, as in the slip law with several void fractions, leaves far more, as for
/// CELL_ROW_TOL.
const TEMPERATURE_TOL: f64 = 1e-8;

/// Most doublings of the step beyond either end of the temperature bracket (Marcher::solve_temperature)
const MAX_STEP_OUTS: usize = 30;

/// Most steps of the chord iteration for the temperature before the bracketed solve takes over
const CHORD_MAXITER: usize = 10;

/// The cell step's descent from p_prev where the state at p_s cannot be computed (descend): its first step is
/// (p_prev - p_s) / DESCENT_STEPS, and it doubles; and the relative width to which it bisects the edge of the
/// pressures with a state
const DESCENT_STEPS: f64 = 1024.0;
const EDGE_XTOL: f64 = 1e-9;

/// The temperature solve's test for a row U-shaped in T (Marcher::solve_temperature): the step (K) below T_lo at
/// which it samples the row, and the width (K) to which it narrows the row's minimum before it concludes that the
/// row stays positive
const U_SAMPLE_DT: f64 = 1e-2;
const U_MIN_XTOL: f64 = 1e-3;

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
    /// Temperature solves, where the energy row depends on the pressure (Marcher::solve_temperature)
    pub temperature_solves: usize,
    /// Temperature solves that the chord iteration did not finish, so the bracketed solve took over
    pub chord_fallbacks: usize,
    /// Temperature solves whose bracket needed steps beyond max(T_{i-1}, T_a), those that needed steps below its
    /// lower end (Joule-Thomson cooling, THM-8), and those that failed
    pub step_outs: usize,
    pub lower_step_outs: usize,
    /// Temperature solves whose row was U-shaped in T, so that its minimum bracketed the root
    pub temperature_minima: usize,
    pub temperature_failures: usize,
    /// Samples of the scan where R is not finite, which it leaves out, and the edges of the finite region it refines
    /// (shoot.rs)
    pub non_finite: usize,
    pub edge_refinements: usize,
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

    pub fn count(&self, f: impl FnOnce(&mut Counts)) {
        let mut c = self.counts.get();
        f(&mut c);
        self.counts.set(c);
    }

    /// The temperature at point i that zeroes the energy row of cell i, given the state at point i - 1, where the row
    /// is linear in the temperature and does not depend on the pressure (Thermal::is_linear): heat loss alone, with
    /// a heat flux capacity that the phase rates fix (thermal::energy_step). None where the row depends on the state.
    fn linear_temperature(&self, cell: geometry::Cell, prev: &State, w_res: f64) -> Option<f64> {
        let (spec, op) = (self.spec, self.op);
        if !spec.thermal.is_linear(&spec.fluid) {
            return None;
        }
        let a = spec.a();
        let (w_g, w_l) = spec.fluid.phase_rates(prev.p, prev.t, w_res, op.w_lg);
        let capacity = spec.fluid.cp_g * (w_g / a) + spec.fluid.cp_l * (w_l / a);
        let t_a = thermal::ambient_temperature(cell.tvd_frac, op.t_r, op.t_s);
        Some(thermal::energy_step(prev.t, t_a, cell.delta_md, spec.thermal.h, spec.geometry.d, capacity))
    }

    /// The slope of cell i's energy row in the temperature from the heat loss alone, 1 + ΔMD 4h / (D cp_flux), at s
    fn heat_loss_slope(&self, cell: geometry::Cell, s: &State) -> f64 {
        let spec = self.spec;
        1.0 + cell.delta_md * 4.0 * spec.thermal.h / (spec.geometry.d * thermal::heat_flux_capacity(&spec.fluid, s))
    }

    /// The state at point i at pressure p whose temperature zeroes the energy row of cell i, given the state prev at
    /// point i - 1 (specs/architecture.md, Rust core, design point 4).
    ///
    /// First a chord iteration: Newton's method with the heat loss's slope (heat_loss_slope) for the derivative, from
    /// guess, the temperature at the cell's previous trial pressure, until its step is a few ulp. Where it does not
    /// converge, Brent on the row r_T(T), with the closures at (p, T), on a bracket where it changes sign. With
    /// dT/dMD = -H + Φ_f - Φ_g - Φ_JT, the row is r_T = T - T_{i-1} + ΔMD (H - Φ_f + Φ_g + Φ_JT), where the heat loss
    /// H has the sign of T - T_a, frictional heating Φ_f >= 0, and the gravity term 0 <= Φ_g <= Φ_max
    /// (Thermal::gravity_term_bound) at every state. So without the Joule-Thomson term, r_T <= 0 at
    /// T_lo = min(T_{i-1}, T_a) - ΔMD Φ_max, and r_T >= 0 at max(T_{i-1}, T_a) without frictional heating. Where
    /// frictional heating, or Joule-Thomson heating (Φ_JT < 0), keeps r_T negative at the upper end, it steps out by
    /// doubling steps until it is not; where Joule-Thomson cooling (Φ_JT > 0, which has no simple bound) keeps r_T
    /// positive at T_lo, the lower end steps out in the same way. Near a gas well's choked wellhead, the cooling can
    /// make r_T U-shaped in T, with two roots; the solve takes the one on the rising side of its minimum, which
    /// continues the root without the term (specs/features/016-joule-thomson.md).
    ///
    /// Where the row jumps across zero instead of crossing it, as where the slip law switches between several void
    /// fractions, Brent converges onto the jump: the state there is returned, as not solved, so that the march can
    /// continue at it, as at a cell whose momentum row is not solved (CellStep::Unsolved). None if there is no state.
    fn solve_temperature(&self, p: f64, cell: geometry::Cell, prev: &State, w_res: f64, guess: f64)
                         -> Option<(State, bool)> {
        self.count(|c| c.temperature_solves += 1);
        let (spec, op) = (self.spec, self.op);
        let mut t = guess;
        let mut last_step = f64::INFINITY;
        for _ in 0..CHORD_MAXITER {
            let Some(s) = self.point_state(p, t, w_res, cell.cos_incl) else { break };
            let step = discretization::energy_row(spec, op, cell, &s, prev) / self.heat_loss_slope(cell, &s);
            if step.abs() <= RTOL * t.abs() {
                return Some((s, true));
            }
            if step.is_nan() || step.abs() >= last_step {
                break;
            }
            (last_step, t) = (step.abs(), t - step);
        }
        self.count(|c| c.chord_fallbacks += 1);
        let mut row = |t: f64| -> Result<f64, RootError> {
            let s = self.point_state(p, t, w_res, cell.cos_incl).ok_or(RootError::NoSignChange)?;
            Ok(discretization::energy_row(spec, op, cell, &s, prev))
        };
        let t_a = thermal::ambient_temperature(cell.tvd_frac, op.t_r, op.t_s);
        let t_lo = prev.t.min(t_a) - cell.delta_md * spec.thermal.gravity_term_bound(&spec.fluid, cell.cos_incl);
        let (mut a, mut b) = (t_lo, prev.t.max(t_a));
        let mut f_a = row(a).ok()?;
        let mut f_b = row(b).ok()?;
        if f_b < 0.0 {
            self.count(|c| c.step_outs += 1);
            let mut step = -f_b;
            for k in 0..=MAX_STEP_OUTS {
                (a, b) = (b, b + step);
                f_b = row(b).ok()?;
                if f_b >= 0.0 {
                    break;
                }
                if k == MAX_STEP_OUTS {
                    self.count(|c| c.temperature_failures += 1);
                    return None;
                }
                step *= 2.0;
            }
        } else if f_a > 0.0 {
            // Positive at both ends. Where the row rises below T_lo, T_lo is on the falling side of a row that is
            // U-shaped in T, as Joule-Thomson cooling can make it near a gas well's choked wellhead: the root on the
            // rising side of its minimum, which continues the root without the term, lies between the ends, and the
            // minimum brackets it. Otherwise the root lies below T_lo, and the lower end steps out.
            let falling = row(a - U_SAMPLE_DT).map_or(true, |f| !(f <= f_a));
            if falling {
                self.count(|c| c.temperature_minima += 1);
                let (t_m, f_m) = minimize(&mut |t| row(t).ok().filter(|f| f.is_finite()).unwrap_or(f64::INFINITY),
                                          a, b, U_MIN_XTOL, 200, 0.0);
                if !(f_m < 0.0) {
                    self.count(|c| c.temperature_failures += 1);
                    return None;
                }
                a = t_m;
            } else {
                self.count(|c| c.lower_step_outs += 1);
                let mut step = f_a;
                for k in 0..=MAX_STEP_OUTS {
                    b = a;
                    a -= step;
                    f_a = row(a).ok()?;
                    if f_a <= 0.0 {
                        break;
                    }
                    if k == MAX_STEP_OUTS {
                        self.count(|c| c.temperature_failures += 1);
                        return None;
                    }
                    step *= 2.0;
                }
            }
        }
        let (t, f) = brentq(&mut row, a, b, 0.0, RTOL, 100).ok()?;
        let s = self.point_state(p, t, w_res, cell.cos_incl)?;
        let solved = f.abs() <= TEMPERATURE_TOL * self.heat_loss_slope(cell, &s);
        if !solved {
            self.count(|c| c.temperature_failures += 1);
        }
        Some((s, solved))
    }

    /// The state at point i at pressure p, and whether its energy row is solved: at the temperature t_fixed where the
    /// row is linear, and otherwise at the temperature that zeroes it at p
    fn state_at(&self, p: f64, cell: geometry::Cell, t_fixed: Option<f64>, prev: &State, w_res: f64, guess: f64)
                -> Option<(State, bool)> {
        match t_fixed {
            Some(t) => self.point_state(p, t, w_res, cell.cos_incl).map(|s| (s, true)),
            None => self.solve_temperature(p, cell, prev, w_res, guess),
        }
    }

    /// The void fraction that zeroes the slip row (SLIP-1) at a point where the phase rates fix the superficial
    /// velocities j_g and j_l (m/s): Brent on h(α) = α (C_0 j_m + v_inf) - j_g, which is -α times the slip row with
    /// v_g = j_g / α. The classifier sees α only through its features c_2 and c_4, and C_0 >= 1 and v_inf >= 0 for
    /// every mix of the regimes, so h(0) = -j_g < 0 and h(1) >= j_l + v_inf > 0: the bracket [0, 1] always holds a
    /// root. None where the rise velocities are not real (rho_g >= rho_l).
    fn void_fraction(&self, j_g: f64, j_l: f64, rho_g: f64, rho_l: f64, sigma: f64, cos_incl: f64) -> Option<f64> {
        if !(rho_g > 0.0 && rho_g < rho_l) {
            return None;
        }
        let j_m = j_g + j_l;
        let spec = self.spec;
        let terms = slip::SlipTerms::new(j_g, j_l, rho_g, rho_l, sigma, spec.geometry.d, cos_incl, &spec.slip);
        let mut h = |alpha: f64| -> Result<f64, RootError> {
            let (c_0, v_inf) = terms.parameters(alpha);
            Ok(alpha * (c_0 * j_m + v_inf) - j_g)
        };
        brentq(&mut h, 0.0, 1.0, 0.0, RTOL, 100).ok().map(|(alpha, _)| alpha)
    }

    /// The state at a point at pressure p and temperature t, in a cell of inclination cos_incl: the phase rates there,
    /// the densities, the surface tension, the void fraction from the slip law, and the velocities that carry the
    /// rates. So the inflow and mass rows (INF-6, INF-7, DISC-7, DISC-8) and the closure rows hold by construction.
    fn point_state(&self, p: f64, t: f64, w_res: f64, cos_incl: f64) -> Option<State> {
        self.count(|c| c.states += 1);
        let (a, fluid) = (self.spec.a(), &self.spec.fluid);
        let (w_g, w_l) = fluid.phase_rates(p, t, w_res, self.op.w_lg);
        let rho_g = fluid.gas_density(p, t);
        let rho_l = fluid.liquid_density(p, t);
        let sigma = fluid.surface_tension(p, t, rho_l);
        let alpha = self.void_fraction(w_g / (a * rho_g), w_l / (a * rho_l), rho_g, rho_l, sigma, cos_incl)?;
        let v_g = w_g / (a * alpha * rho_g);
        let v_l = w_l / (a * (1.0 - alpha) * rho_l);
        Some(State { p, v_g, v_l, alpha, rho_g, rho_l, t })
    }

    /// The pressure at a point from the momentum row of the cell below it, given the state at the cell's lower point,
    /// searched on [p_s, p_prev]; the temperature is t_fixed, or solved at each trial pressure from the last trial's.
    /// Also the temperature at the last trial pressure, the first guess for the state at the answer.
    fn solve_cell(&self, cell: geometry::Cell, t_fixed: Option<f64>, s_prev: &State, w_res: f64) -> (CellStep, f64) {
        let last_t = Cell::new(s_prev.t);
        let mut row = |p: f64| -> Result<f64, RootError> {
            let (s, _) = self.state_at(p, cell, t_fixed, s_prev, w_res, last_t.get()).ok_or(RootError::NoSignChange)?;
            last_t.set(s.t);
            Ok(discretization::momentum_row(self.spec, cell, &s, s_prev))
        };
        let step = cell_step(&mut row, self.op.p_s, s_prev.p);
        (step, last_t.get())
    }

    /// March from p_0 to the wellhead
    pub fn march(&self, p_0: f64) -> March {
        self.count(|c| c.marches += 1);
        let n = self.spec.n_cells();
        let w_res = discretization::reservoir_rate(self.spec, self.op, p_0);
        let mut x = Vec::with_capacity(DIM_X * (n + 1));
        let mut failed = false;
        let cos_0 = self.spec.geometry.point_cos(0);
        let (spec, op) = (self.spec, self.op);
        let t_0 = spec.thermal.inflow_temperature(w_res, op.w_lg, op.t_r, op.t_lg, &spec.fluid);
        let Some(mut prev) = self.point_state(p_0, t_0, w_res, cos_0) else {
            x.extend_from_slice(&[f64::NAN; DIM_X]);
            return March { x, w_res, failed: true, below_separator: false };
        };
        x.extend_from_slice(&prev.to_array());
        for i in 1..=n {
            let cell = self.spec.geometry.cell(i);
            let t_fixed = self.linear_temperature(cell, &prev, w_res);
            let (step, guess) = self.solve_cell(cell, t_fixed, &prev, w_res);
            let p = match step {
                CellStep::Solved(p) => p,
                CellStep::Unsolved(p) => {
                    failed = true;
                    p
                }
                CellStep::BelowSeparator => return March { x, w_res, failed: true, below_separator: true },
            };
            let Some((s, solved)) = self.state_at(p, cell, t_fixed, &prev, w_res, guess) else {
                x.extend_from_slice(&[f64::NAN; DIM_X]);
                return March { x, w_res, failed: true, below_separator: false };
            };
            failed |= !solved;
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

/// The pressure that zeroes a cell's momentum row(p) (bar) on [p_s, p_prev]: its subsonic root, on the rising side
/// of the row's minimum at the sonic pressure p*
fn cell_step(row: &mut impl FnMut(f64) -> Result<f64, RootError>, p_s: f64, p_prev: f64) -> CellStep {  // spec: SOL-8
    // The row is U-shaped in p, with its minimum at the cell's sonic pressure p*, and positive at p_prev. Below
    // zero at p_s: p_s lies right of p*, or left of it where the row still falls, so the only sign change on
    // [p_s, p_prev] is the subsonic root
    let solved = |(p, f): (f64, f64)| if f.abs() <= CELL_ROW_TOL { CellStep::Solved(p) } else { CellStep::Unsolved(p) };
    let f_s = match row(p_s) {
        Ok(f) => f,
        Err(_) => return descend(row, p_s, p_prev),
    };
    if f_s < 0.0 {
        return match brentq(row, p_s, p_prev, CELL_XTOL, RTOL, 100) {
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
        return match brentq(row, p_star, p_prev, CELL_XTOL, RTOL, 100) {
            Ok(root) => solved(root),
            Err(_) => CellStep::Unsolved(p_star),
        };
    }
    if f_s <= f_star { CellStep::BelowSeparator } else { CellStep::Unsolved(p_star) }
}

/// The cell step where the state cannot be computed at p_s, as where Joule-Thomson cooling (THM-8) leaves a gas
/// well's energy row without a root at pressures far below the cell's, and the pressures with a state need not form
/// one interval. The subsonic root is the first sign change below p_prev: the search steps down from p_prev,
/// doubling its step from (p_prev - p_s) / DESCENT_STEPS, to the first pressure where the row is negative, and Brent
/// takes it from there. Where a step reaches a pressure without a state first, the edge of the stretch with states
/// above it is found by bisection, and the row's minimum on that stretch decides, as in cell_step.
fn descend(row: &mut impl FnMut(f64) -> Result<f64, RootError>, p_s: f64, p_prev: f64) -> CellStep {
    let finite = |r: Result<f64, RootError>| r.ok().filter(|f| f.is_finite());
    let mut hi = p_prev; // the lowest pressure reached with a state, where the row is positive
    let mut d = (p_prev - p_s) / DESCENT_STEPS;
    loop {
        let p = (p_prev - d).max(p_s);
        match finite(row(p)) {
            Some(f) if f < 0.0 => return bracket(row, p, hi, hi),
            Some(_) if p > p_s => hi = p,
            Some(_) => return CellStep::BelowSeparator, // positive down to p_s
            None => {
                let (mut lo, mut up) = (p, hi);
                while up - lo > EDGE_XTOL * p_prev {
                    let c = 0.5 * (lo + up);
                    match finite(row(c)) {
                        Some(f) if f < 0.0 => return bracket(row, c, up, up),
                        Some(_) => up = c,
                        None => lo = c,
                    }
                }
                let (p_star, f_star) = minimize(&mut |p| finite(row(p)).unwrap_or(f64::INFINITY), up, p_prev, 1e-2, 200,
                                                f64::NEG_INFINITY);
                return if f_star < 0.0 { bracket(row, p_star, p_prev, p_star) } else { CellStep::Unsolved(up) };
            }
        }
        d *= 2.0;
    }
}

/// Brent on a sign change of the cell's row on [a, b]: solved where the row is zero there, and otherwise unsolved at
/// fail
fn bracket(row: &mut impl FnMut(f64) -> Result<f64, RootError>, a: f64, b: f64, fail: f64) -> CellStep {
    match brentq(row, a, b, CELL_XTOL, RTOL, 100) {
        Ok((p, f)) if f.abs() <= CELL_ROW_TOL => CellStep::Solved(p),
        Ok((p, _)) => CellStep::Unsolved(p),
        Err(_) => CellStep::Unsolved(fail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::test_wells::all;
    use crate::shoot;

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
    fn a_march_zeroes_every_row_but_the_choke_row() {
        for (name, spec, op) in all(20) {
            let m = Marcher::new(&spec, &op);
            // A march that reaches the wellhead: from near p_r, where the rate is small
            let march = [0.8, 0.9, 0.95, 0.99].iter().map(|f| m.march(op.p_s + f * (op.p_r - op.p_s)))
                .find(|march| !march.failed).unwrap_or_else(|| panic!("{name}: no march reached the wellhead"));
            for (id, largest) in largest_rows(&spec, &op, &march.x) {
                let bound = if id == "DISC-9" { 1e-8 } else { 1e-10 };
                assert!(id == "CHK-1" || largest < bound, "{name} {id}: {largest}");
            }
        }
    }

    /// The number of sign changes of the differences of a sampled function that exceed its rounding: a U-shaped
    /// function has at most one, from falling to rising
    fn turns(values: &[f64]) -> (usize, bool) {
        let scale = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let rising: Vec<bool> = values.windows(2).filter(|w| (w[1] - w[0]).abs() > 1e-12 * scale)
            .map(|w| w[1] > w[0]).collect();
        let changes = rising.windows(2).filter(|w| w[0] != w[1]).count();
        (changes, rising.first().copied().unwrap_or(true))
    }

    /// What the cell solve assumes (specs/features/015-rust-develop-model.md): at every cell of every root of the
    /// test wells, which cover every option, the cell's momentum row, with the temperature solved at each pressure, is
    /// U-shaped in the pressure on [p_s, p_{i-1}] and positive at p_{i-1}; and where the energy row depends on the
    /// pressure, it has one root in the temperature at the root's pressure, within 40 K of the root's. Samples where
    /// the row is not finite are left out: the dead-oil viscosity has none below 0 °F (255 K), which a cold
    /// wellhead's window reaches with Joule-Thomson cooling.
    #[test]
    fn the_cell_rows_have_the_shapes_the_cell_solve_assumes() {
        for (name, spec, op) in all(20) {
            let m = Marcher::new(&spec, &op);
            for root in shoot::root_set(&spec, &op).unwrap().roots {
                let points: Vec<State> = root.x.chunks_exact(DIM_X).map(State::of).collect();
                for i in 1..points.len() {
                    let (cell, prev, s) = (spec.geometry.cell(i), points[i - 1], points[i]);
                    let t_fixed = m.linear_temperature(cell, &prev, root.w_res);
                    let row = |p: f64| m.state_at(p, cell, t_fixed, &prev, root.w_res, s.t)
                        .map(|(s, _)| discretization::momentum_row(&spec, cell, &s, &prev));
                    let rows: Vec<f64> = (0..=200).filter_map(|k| row(op.p_s + (prev.p - op.p_s) * k as f64 / 200.0))
                        .collect();
                    let (changes, rising_first) = turns(&rows);
                    assert!(changes == 0 || (changes == 1 && !rising_first),
                            "{name}, root {}, cell {i}: {changes} turns", root.x[0]);
                    assert!(row(prev.p).unwrap() > 0.0, "{name}, root {}, cell {i}: row at p_prev", root.x[0]);
                    if t_fixed.is_none() {
                        let r_t: Vec<f64> = (0..=200).map(|k| s.t - 40.0 + 80.0 * k as f64 / 200.0)
                            .filter_map(|t| m.point_state(s.p, t, root.w_res, cell.cos_incl))
                            .map(|st| discretization::energy_row(&spec, &op, cell, &st, &prev))
                            .filter(|r| r.is_finite()).collect();
                        let crossings = r_t.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
                        assert_eq!(crossings, 1, "{name}, root {}, cell {i}: energy row", root.x[0]);
                    }
                }
            }
        }
    }
}
