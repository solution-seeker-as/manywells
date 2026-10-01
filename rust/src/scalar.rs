// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Bracketed scalar root finding and minimization, with no model in them.
//!
//! `brentq` is a transcription of SciPy's `scipy/optimize/Zeros/brentq.c` (BSD-3-Clause, Copyright (c) 2001-2002
//! Enthought, Inc., 2003-2024 SciPy Developers): the same bisection and interpolation steps and the same
//! convergence test, `delta = (xtol + rtol |x|) / 2`. The function returns a Result, so that a failure inside a
//! nested solve stops the outer one.

/// SciPy's default rtol for brentq (4 times machine epsilon)
pub const RTOL: f64 = 4.0 * f64::EPSILON;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RootError {
    /// f(a) and f(b) have the same sign
    NoSignChange,
    /// No convergence within maxiter iterations
    MaxIter,
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::NoSignChange => write!(f, "f(a) and f(b) must have different signs"),
            RootError::MaxIter => write!(f, "failed to converge within maxiter iterations"),
        }
    }
}

/// A root of f in [xa, xb], where f(xa) and f(xb) differ in sign, by Brent's method
pub fn brentq<F>(f: &mut F, xa: f64, xb: f64, xtol: f64, rtol: f64, maxiter: usize) -> Result<f64, RootError>
where
    F: FnMut(f64) -> Result<f64, RootError>,
{
    let (mut xpre, mut xcur) = (xa, xb);
    let (mut xblk, mut fblk) = (0.0_f64, 0.0_f64);
    let (mut spre, mut scur) = (0.0_f64, 0.0_f64);

    let mut fpre = f(xpre)?;
    let mut fcur = f(xcur)?;
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre * fcur > 0.0 {
        return Err(RootError::NoSignChange);
    }

    for _ in 0..maxiter {
        if fpre != 0.0 && fcur != 0.0 && (fpre * fcur < 0.0) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = xcur - xpre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // Secant
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // Inverse quadratic interpolation
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur)?;
    }
    Err(RootError::MaxIter)
}

/// The minimum of f on [a, b] by golden-section search, as (x, f(x)), until the interval is at most xtol wide.
/// It assumes f has one minimum on the interval and does not check it.
pub fn minimize<F>(f: &mut F, mut a: f64, mut b: f64, xtol: f64, maxiter: usize) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    const INV_PHI: f64 = 0.618_033_988_749_895; // 1 / golden ratio
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..maxiter {
        if (b - a).abs() <= xtol {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
    }
    let x = 0.5 * (a + b);
    (x, f(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brentq_finds_simple_roots() {
        let mut f = |x: f64| Ok(x * x - 2.0);
        let r = brentq(&mut f, 0.0, 2.0, 2e-12, RTOL, 100).unwrap();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-10);

        let mut g = |x: f64| Ok(2.0 - x);
        let r = brentq(&mut g, 0.0, 5.0, 2e-12, RTOL, 100).unwrap();
        assert!((r - 2.0).abs() < 1e-10);
    }

    #[test]
    fn brentq_needs_a_sign_change() {
        let mut f = |x: f64| Ok(x * x + 1.0);
        assert_eq!(brentq(&mut f, -1.0, 1.0, 1e-8, RTOL, 100), Err(RootError::NoSignChange));
    }

    #[test]
    fn minimize_finds_the_minimum() {
        let mut f = |x: f64| (x - 2.0) * (x - 2.0);
        let (x, fx) = minimize(&mut f, 0.0, 5.0, 1e-8, 200);
        assert!((x - 2.0).abs() < 1e-4);
        assert!(fx < 1e-8);
    }
}
