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
use crate::march::{Counts, Marcher};
use crate::scalar::{brentq, minimize, RootError, RTOL};

/// Number of scan intervals on (p_s, p_r)
const SCAN_INTERVALS: usize = 100;

/// Width, relative to p_r - p_s, to which the scan's golden-section search narrows a local minimum of R before it
/// concludes that R stays positive there. Two roots closer than this are 1/100 of the verifier's tol_x apart in p_0.
const REFINE_XTOL: f64 = 1e-6;

/// A root is accepted only where |R| is at most this fraction of the rate, so that a jump in R is not taken for a
/// root. At the extreme trickle roots of the case set, one ulp of p_0 moves the choke rate by about 5e-5 of w_m.
const ACCEPT_REL: f64 = 1e-3;

/// Step of the central difference for dR/dp_0 at a root, relative to its distance from p_r and from p_s
const SLOPE_STEP: f64 = 1e-4;

/// A root and what it carries besides its state
pub struct Root {
    pub x: Vec<f64>,
    /// R rises with p_0 through the root: dR/dp_0 > 0, unstable (SOL-3)
    pub rising: bool,
    /// dR/dp_0 at the root (kg/s per bar), by a central difference; NaN if R is not finite on either side
    pub slope: f64,
    pub choked: bool,
    pub flow_regime: Vec<&'static str>,
    pub w_res: f64,
    pub w_g_res: f64,
}

/// The roots of a search, sorted by p_0, the work it took, and the number of sign changes of R that were not accepted
/// as roots
pub struct Search {
    pub roots: Vec<Root>,
    pub counts: Counts,
    pub rejected: usize,
}

/// p_0 of the roots, each with whether R rises through it. Samples R(p_0) on a uniform scan of (p_s, p_r) and runs
/// Brent on every sign change between neighbouring samples. Trickle roots, and both roots of a nearly closed choke,
/// can lie within one step of p_r, so the top interval gets a ladder of samples whose drawdown halves down to that
/// of the top sample. A negative region narrower than the spacing, near the fold where two roots merge, leaves R
/// positive at every sample but with a local minimum: a golden-section search between the minimum's neighbours looks
/// for a negative R there, and Brent then runs on both sides of it.
fn shoot(m: &Marcher) -> Result<Vec<(f64, bool)>, String> {
    let op = m.op;
    let p_lo = op.p_s + 1e-3;
    let d_hi = 1e-6; // Drawdown of the top sample (bar)
    let p_hi = op.p_r - d_hi;
    let step = (p_hi - p_lo) / SCAN_INTERVALS as f64;
    let residual = |p: f64| -> Result<f64, String> {
        match m.residual(p) {
            Some((r, _)) => Ok(r),
            None => Err(format!("the shooting residual is not finite at p_0 = {p} bar")),
        }
    };
    let brent_r = |p: f64| residual(p).map_err(|_| RootError::NoSignChange);

    // The scan, from p_s up to p_r, with the ladder in the top interval
    let mut p: Vec<f64> = (0..=SCAN_INTERVALS).rev().map(|k| p_hi - step * k as f64).collect();
    let mut d = step / 2.0;
    while d > 2.0 * d_hi {
        p.push(op.p_r - d);
        d /= 2.0;
    }
    p.sort_by(f64::total_cmp);
    let r: Vec<f64> = p.iter().map(|&p| residual(p)).collect::<Result<_, _>>()?;
    let n = p.len() - 1;

    let mut brackets = Vec::new(); // (a, b, rising): a sign change on [a, b], where R rises if it is negative at a
    for j in 0..n {
        if (r[j] < 0.0) != (r[j + 1] < 0.0) {
            brackets.push((p[j], p[j + 1], r[j] < 0.0));
        }
    }
    let xtol_refine = REFINE_XTOL * (op.p_r - op.p_s);
    for j in 1..n {
        if r[j] >= 0.0 && r[j - 1] >= r[j] && r[j + 1] >= r[j] {
            let mut f = |p: f64| residual(p).unwrap_or(f64::INFINITY);
            let (q, r_q) = minimize(&mut f, p[j - 1], p[j + 1], xtol_refine, 200, 0.0);
            if r_q < 0.0 {
                brackets.push((p[j - 1], q, false));
                brackets.push((q, p[j + 1], true));
            }
        }
    }

    // Brent in the drawdown d = p_r - p_0, so that its relative tolerance is relative to the drawdown, which a trickle
    // root has little of; p_0 itself is resolved to a few ulp of p_r
    let mut r_of_d = |d: f64| brent_r(op.p_r - d);
    let xtol = 4.0 * f64::EPSILON * op.p_r;
    let mut roots = Vec::new();
    for (a, b, rising) in brackets {
        if let Ok((d, _)) = brentq(&mut r_of_d, op.p_r - b, op.p_r - a, xtol, RTOL, 100) {
            roots.push((op.p_r - d, rising));
        }
    }
    roots.sort_by(|x, y| x.0.total_cmp(&y.0));
    roots.dedup_by(|x, y| x.0 == y.0);
    Ok(roots)
}

/// A root at p_0, with its outputs, or None if its march fails or R is not close enough to zero there
fn root_at(m: &Marcher, p_0: f64, rising: bool) -> Option<Root> {
    let (spec, op) = (m.spec, m.op);
    let march = m.march(p_0);
    if march.failed {
        return None;
    }
    let top = State::of(&march.x[march.x.len() - DIM_X..]);
    let (w_g, w_l) = spec.fluid.phase_rates(top.p, top.t, march.w_res, op.w_lg);
    if discretization::choke_row(spec, op, &top).abs() > ACCEPT_REL * (w_g + w_l) {
        return None;
    }
    let h = SLOPE_STEP * (op.p_r - p_0).min(p_0 - op.p_s);
    let slope = match (m.residual(p_0 + h), m.residual(p_0 - h)) { // spec: SOL-3
        (Some((above, _)), Some((below, _))) => (above - below) / (2.0 * h),
        _ => f64::NAN,
    };
    Some(Root {
        choked: spec.choke.is_choked(top.p, op.p_s),
        flow_regime: discretization::flow_regimes(spec, &march.x),
        w_res: march.w_res,
        w_g_res: spec.fluid.reservoir_gas_rate(march.w_res),
        rising,
        slope,
        x: march.x,
    })
}

/// Every root the search finds for the well at the operating point
pub fn root_set(spec: &WellSpec, op: &OperatingPoint) -> Result<Search, String> {
    spec.check()?;
    op.check()?;
    let m = Marcher::new(spec, op);
    let found = shoot(&m)?;
    let n = found.len();
    let roots: Vec<Root> = found.into_iter().filter_map(|(p_0, rising)| root_at(&m, p_0, rising)).collect();
    Ok(Search { rejected: n - roots.len(), roots, counts: m.counts() })
}

/// The shooting residual R(p_0), for tests and diagnostics
pub fn residual(spec: &WellSpec, op: &OperatingPoint, p_0: f64) -> Option<f64> {
    Marcher::new(spec, op).residual(p_0).map(|(r, _)| r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::test_wells::{w1, w2};

    #[test]
    fn every_root_zeroes_every_row() {
        for (spec, op) in [w1(20), w2(20)] {
            let search = root_set(&spec, &op).unwrap();
            assert!(!search.roots.is_empty());
            for root in &search.roots {
                // The choke row is as small as p_0's resolution allows, times dR/dp_0, which is steep at a trickle root:
                // at W2's, 4e-7 of the rate
                let (w_g, w_l) = State::of(&root.x[root.x.len() - DIM_X..]).rates(spec.a());
                for (id, v) in discretization::rows(&spec, &op, &root.x) {
                    let bound = if id == "CHK-1" { 1e-6 * (w_g + w_l) } else { 1e-8 };
                    assert!(v.abs() < bound, "{id}: {v}");
                }
            }
        }
    }

    #[test]
    fn the_slope_has_the_sign_of_the_bracket() {
        for (spec, op) in [w1(20), w2(20)] {
            for root in root_set(&spec, &op).unwrap().roots {
                assert_eq!(root.slope > 0.0, root.rising, "slope {} at p_0 = {}", root.slope, root.x[0]);
            }
        }
    }
}
