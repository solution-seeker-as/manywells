// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The march: given the bottomhole pressure p_0, the inflow gives the phase rates, and the state at every point
//! follows from the rows of discretization.rs, point by point up the well. At each point the closures give the
//! densities and the void fraction at a trial pressure, and the cell's momentum row is solved for the pressure.
//! What is left at the wellhead is the choke row, whose value is the shooting residual R(p_0) (shoot.rs).

use std::cell::Cell;

use crate::discretization::{self, State, DIM_X};
use crate::input::{OperatingPoint, WellSpec};
use crate::scalar::{brentq, minimize, RootError, RTOL};
use crate::slip;

/// Lowest trial pressure of a cell solve (bar)
const P_MIN: f64 = 1e-3;

/// The phase rates of a march, the same at every point (BAL-3)
#[derive(Clone, Copy, Debug)]
pub struct Rates {
    pub w_res: f64, // Reservoir liquid rate (kg/s)
    pub w_g: f64,   // Gas rate, with the lift gas (kg/s)
    pub w_l: f64,   // Liquid rate (kg/s)
}

/// A march from p_0 to the wellhead
pub struct March {
    pub x: Vec<f64>,
    pub rates: Rates,
    /// A cell had no subsonic root (choked), or a state could not be computed, so x is not a solution of the rows
    pub failed: bool,
}

enum CellStep {
    /// The cell's momentum row is zero at this pressure
    Solved(f64),
    /// No subsonic root: the march continues at the pressure p* where the row is smallest, so that R stays continuous
    /// in p_0, but it is not a solution
    Choked(f64),
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

    /// Temperature at point i: the exact solution of the energy balance's ODE with T(0) = T_r
    fn temperature(&self, i: usize, rates: &Rates) -> f64 {
        let (spec, op) = (self.spec, self.op);
        let k = std::f64::consts::PI * spec.h * spec.d / (spec.fluid.cp_g * rates.w_g + spec.fluid.cp_l * rates.w_l);
        let z = i as f64 * spec.delta_z();
        op.t_r - z / spec.l * (op.t_r - op.t_s) + 1.0 / (spec.l * k) * (op.t_r - op.t_s) * (1.0 - (-k * z).exp())
    }

    /// The void fraction that zeroes the slip row, by fixed-point iteration on alpha = w_g / (A rho_g (C_0 v_m + v_inf))
    fn void_fraction(&self, rates: &Rates, rho_g: f64, rho_l: f64, t: f64) -> Option<f64> {
        const ALPHA_LO: f64 = 1e-6;
        const ALPHA_HI: f64 = 1.0 - 1e-6;
        const ALPHA_TOL: f64 = 1e-3;
        const MAX_ITER: usize = 100;
        let (a, w_g, w_l) = (self.spec.a(), rates.w_g, rates.w_l);
        let rho_g = if rho_g <= 0.0 { 1e-3 } else { rho_g };
        if w_g <= 0.0 {
            return Some(ALPHA_LO);
        }
        let sigma = self.spec.fluid.surface_tension(rho_l, t);
        let vm = w_g / (a * rho_g) + w_l / (a * rho_l);
        let mut alpha = (w_g / (a * rho_g * (1.1 * vm + 0.5 + 1e-6))).clamp(ALPHA_LO, ALPHA_HI);
        let mut converged = false;
        for _ in 0..MAX_ITER {
            let v_g = w_g / (a * alpha * rho_g);
            let v_l = w_l / (a * (1.0 - alpha) * rho_l);
            let (c_0, v_inf) = slip::identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, self.spec.d);
            let next = (w_g / (a * rho_g * (c_0 * vm + v_inf + 1e-6))).clamp(ALPHA_LO, ALPHA_HI);
            converged = (next - alpha).abs() < ALPHA_TOL;
            alpha = next;
            if converged {
                break;
            }
        }
        (converged && alpha.is_finite()).then_some(alpha)
    }

    /// The state at point i at pressure p: the closures at the point's temperature and the march's rates
    fn point_state(&self, i: usize, p: f64, rates: &Rates) -> Option<State> {
        let (a, fluid) = (self.spec.a(), &self.spec.fluid);
        let t = self.temperature(i, rates);
        let rho_g = fluid.gas_density(p, t);
        let rho_l = fluid.liquid_density();
        let alpha = self.void_fraction(rates, rho_g, rho_l, t)?;
        let v_g = rates.w_g / (a * alpha * rho_g);
        let v_l = rates.w_l / (a * (1.0 - alpha) * rho_l);
        Some(State { p, v_g, v_l, alpha, rho_g, rho_l, t })
    }

    /// The pressure at point i from the momentum row of cell i, given the state at point i - 1
    fn solve_cell(&self, i: usize, s_prev: &State, rates: &Rates) -> CellStep {
        let p_prev = s_prev.p;
        let mut row = |p: f64| -> Result<f64, RootError> {
            let s = self.point_state(i, p, rates).ok_or(RootError::NoSignChange)?;
            Ok(discretization::momentum_row(self.spec, &s, s_prev))
        };
        // The row is U-shaped in p, with its minimum at the cell's sonic pressure p*: the subsonic root lies between
        // p* and p_prev, usually close to p_prev, so a narrow bracket is tried first
        let lo = (p_prev - 0.1 * (p_prev - self.op.p_s)).max(P_MIN);
        if lo < p_prev {
            if let Ok(p) = brentq(&mut row, lo, p_prev, 1e-6, RTOL, 100) {
                if p > 0.0 {
                    return CellStep::Solved(p);
                }
            }
        }
        let p_star = minimize(&mut |p| match row(p) {
            Ok(v) if v.is_finite() => v,
            _ => f64::INFINITY,
        }, P_MIN, p_prev, 1e-2, 200).0;
        match brentq(&mut row, p_star, p_prev, 1e-6, RTOL, 100) {
            Ok(p) if p > 0.0 => CellStep::Solved(p),
            _ => CellStep::Choked(p_star),
        }
    }

    /// March from p_0 to the wellhead
    pub fn march(&self, p_0: f64) -> March {
        self.marches.set(self.marches.get() + 1);
        let n = self.spec.n_cells;
        let rates = self.rates(p_0);
        let mut x = Vec::with_capacity(DIM_X * (n + 1));
        let mut failed = false;
        let mut prev = self.point_state(0, p_0, &rates);
        match prev {
            Some(s) => x.extend_from_slice(&s.to_array()),
            None => x.extend_from_slice(&[f64::NAN; DIM_X]),
        }
        for i in 1..=n {
            // After a state could not be computed, the last computed state is copied to every point above
            let state = prev.and_then(|s_prev| {
                let p = match self.solve_cell(i, &s_prev, &rates) {
                    CellStep::Solved(p) => p,
                    CellStep::Choked(p) => {
                        failed = true;
                        p
                    }
                };
                self.point_state(i, p, &rates)
            });
            match state.or(prev) {
                Some(s) => x.extend_from_slice(&s.to_array()),
                None => x.extend_from_slice(&[f64::NAN; DIM_X]),
            }
            if state.is_none() {
                failed = true;
            }
            prev = state.or(prev);
        }
        March { x, rates, failed: failed || prev.is_none() }
    }

    /// The shooting residual R(p_0) and whether the march failed, or None if R is not finite: the choke row, squared
    pub fn residual(&self, p_0: f64) -> Option<(f64, bool)> {
        let m = self.march(p_0);
        let s = State::of(&m.x[m.x.len() - DIM_X..]);
        let (spec, op) = (self.spec, self.op);
        let (w_g, w_l) = s.rates(spec.a());
        let w_m = w_g + w_l;
        let w_c = spec.choke.mass_flow_rate(op.u, s.p, op.p_s, w_g, w_l, s.alpha, s.rho_g, s.rho_l);
        let r = if w_c > 0.0 { w_m * w_m - w_c * w_c } else {
            // Below the critical pressure, the port's squared row: w_m² - (K_c σ(u))² 2 ρ Δp / Φ with Δp <= 0
            let p_c = spec.choke.critical_pressure(s.p, op.p_s);
            let dp = crate::units::CF_BAR * (s.p - p_c);
            let (rho, phi) = match spec.choke.model {
                crate::choke::ChokeModel::Simpson => {
                    let x_g = w_g / w_m;
                    (s.rho_l, crate::choke::simpson_multiplier(x_g, s.rho_g, s.rho_l))
                }
                crate::choke::ChokeModel::Bernoulli => (s.rho_m(), 1.0),
            };
            let kc = spec.choke.k_c * spec.choke.profile.opening(op.u);
            w_m * w_m - kc * kc * (2.0 * rho * dp) / phi
        };
        r.is_finite().then_some((r, m.failed))
    }
}
