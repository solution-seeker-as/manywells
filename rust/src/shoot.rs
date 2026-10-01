// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The root search by shooting (specs/model/solution.md): the roots of the well are the bottomhole pressures p_0
//! in (p_s, p_r) where the march's shooting residual R(p_0) is zero, found by a scan of R and Brent on each sign
//! change. Whether R rises or falls through a root gives the sign of dR/dp_0, the stability label (SOL-3).

use crate::discretization::{self, State, DIM_X};
use crate::input::{OperatingPoint, WellSpec};
use crate::march::Marcher;
use crate::scalar::{brentq, RootError, RTOL};

/// Number of scan intervals on (p_s, p_r)
const SCAN_INTERVALS: usize = 100;

/// A root and what it carries besides its state
pub struct Root {
    pub x: Vec<f64>,
    /// R rises with p_0 through the root: dR/dp_0 > 0, unstable (SOL-3)
    pub rising: bool,
    /// dR/dp_0 at the root (kg/s per bar), NaN if not computed
    pub slope: f64,
    pub choked: bool,
    pub flow_regime: Vec<&'static str>,
    pub w_res: f64,
    pub w_g_res: f64,
}

/// The roots of a search, sorted by p_0, and the number of marches it took
pub struct Search {
    pub roots: Vec<Root>,
    pub marches: usize,
}

/// p_0 of the roots, each with whether R rises through it. Scans R(p_0) from p_r down to p_s until it first turns
/// negative, assuming R has the sign pattern + - + or - +, and refines the brackets on each side of that sample.
fn shoot(m: &Marcher) -> Vec<(f64, bool)> {
    let op = m.op;
    let p_lo = op.p_s + 1e-3;
    let p_hi = op.p_r - 1e-6;
    let step = (p_hi - p_lo) / SCAN_INTERVALS as f64;
    let mut residual = |p: f64| m.residual(p).map(|(r, _)| r).ok_or(RootError::NoSignChange);
    let accept = |p: f64| matches!(m.residual(p), Some((_, false)));

    let mut prev: Option<f64> = None; // Last sample with R >= 0
    let mut roots = Vec::new();
    for k in 0..=SCAN_INTERVALS {
        let p = p_hi - step * k as f64;
        let r = match m.residual(p) {
            Some((r, _)) => r,
            None => {
                prev = None; // No bracket across a hole
                continue;
            }
        };
        if r < 0.0 {
            if let Some(p_prev) = prev {
                if let Ok(root) = brentq(&mut residual, p, p_prev, 1e-6, RTOL, 100) {
                    if accept(root) {
                        roots.push((root, true));
                    }
                }
            }
            if let Ok(root) = brentq(&mut residual, p_lo, p, 1e-6, RTOL, 100) {
                if accept(root) {
                    roots.push((root, false));
                }
            }
            return roots;
        }
        prev = Some(p);
    }
    roots
}

/// A root at p_0, with its outputs, or None if its march fails
fn root_at(m: &Marcher, p_0: f64, rising: bool) -> Option<Root> {
    let (spec, op) = (m.spec, m.op);
    let march = m.march(p_0);
    if march.failed {
        return None;
    }
    let top = State::of(&march.x[march.x.len() - DIM_X..]);
    Some(Root {
        choked: spec.choke.is_choked(top.p, op.p_s),
        flow_regime: discretization::flow_regimes(spec, &march.x),
        w_res: march.rates.w_res,
        w_g_res: spec.fluid.reservoir_gas_rate(march.rates.w_res),
        rising,
        slope: f64::NAN,
        x: march.x,
    })
}

/// Every root the search finds for the well at the operating point
pub fn root_set(spec: &WellSpec, op: &OperatingPoint) -> Result<Search, String> {
    spec.check()?;
    op.check()?;
    let m = Marcher::new(spec, op);
    let mut roots: Vec<Root> = shoot(&m).into_iter().filter_map(|(p_0, rising)| root_at(&m, p_0, rising)).collect();
    roots.sort_by(|a, b| a.x[0].total_cmp(&b.x[0]));
    Ok(Search { roots, marches: m.marches() })
}

/// The shooting residual R(p_0), for tests and diagnostics
pub fn residual(spec: &WellSpec, op: &OperatingPoint, p_0: f64) -> Option<f64> {
    Marcher::new(spec, op).residual(p_0).map(|(r, _)| r)
}
