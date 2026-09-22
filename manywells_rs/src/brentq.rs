//! Rust port of scipy's root finders, so the module has no scipy dependency.
//!
//! - `brentq` is a transcription of scipy's `scipy/optimize/Zeros/brentq.c`
//!   (BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc., 2003-2024 SciPy
//!   Developers). Same bisect/interpolate acceptance logic and the same
//!   convergence criterion `delta = (xtol + rtol*|x|) / 2`, so it finds the
//!   same roots with the same iteration behavior as `scipy.optimize.brentq`.
//! - `secant` mirrors the derivative-free branch of `scipy.optimize.newton`
//!   (same starting perturbation for x1 and the same update formulas).
//!
//! The function argument returns Result so that failures in *nested* solves
//! (e.g. the alpha closure losing its bracket inside the cell equation) can
//! propagate out, exactly like Python exceptions propagate through
//! scipy.optimize.brentq.

/// Same default as scipy.optimize.brentq's rtol (4 * machine epsilon).
pub const RTOL: f64 = 4.0 * f64::EPSILON;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RootError {
    /// f(a) and f(b) must have different signs (scipy raises ValueError)
    NoSignChange,
    /// Failed to converge within maxiter iterations (scipy raises RuntimeError)
    MaxIter,
    /// Secant stalled: f(x0) == f(x1) with x0 != x1 (scipy raises RuntimeError)
    SecantStalled,
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::NoSignChange => write!(f, "f(a) and f(b) must have different signs"),
            RootError::MaxIter => write!(f, "failed to converge within maxiter iterations"),
            RootError::SecantStalled => write!(f, "secant stalled: equal function values at distinct points"),
        }
    }
}

/// Find a root of f in [xa, xb] with Brent's method (scipy brentq semantics).
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
            return Ok(xcur); // converged
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate (secant)
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate (inverse quadratic)
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
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

/// Derivative-free Newton (= secant) matching scipy.optimize.newton's secant branch:
/// x1 = x0 * (1 + 1e-4), then +/- 1e-4 depending on sign; converges when the
/// step |p - p1| <= tol.
pub fn secant<F>(f: &mut F, x0: f64, tol: f64, maxiter: usize) -> Result<f64, RootError>
where
    F: FnMut(f64) -> Result<f64, RootError>,
{
    let eps = 1e-4;
    let mut p0 = x0;
    let mut p1 = x0 * (1.0 + eps);
    p1 += if p1 >= 0.0 { eps } else { -eps };

    let mut q0 = f(p0)?;
    let mut q1 = f(p1)?;
    if q1.abs() < q0.abs() {
        std::mem::swap(&mut p0, &mut p1);
        std::mem::swap(&mut q0, &mut q1);
    }

    for _ in 0..maxiter {
        let p = if q1 == q0 {
            if p1 != p0 {
                return Err(RootError::SecantStalled);
            }
            return Ok((p1 + p0) / 2.0);
        } else if q1.abs() > q0.abs() {
            (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1)
        } else {
            (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0)
        };

        if (p - p1).abs() <= tol {
            return Ok(p);
        }
        p0 = p1;
        q0 = q1;
        p1 = p;
        q1 = f(p1)?;
    }
    Err(RootError::MaxIter)
}

/// Golden-section search for the minimum of `f` on `[a, b]`.
///
/// Returns `(x_min, f_min)`. Derivative-free and always stays inside `[a, b]`, so
/// it is safe on the shooting residual (whose infeasible points the caller maps to
/// +inf). Assumes `f` is roughly unimodal on the interval, which the momentum-flux
/// "dip" between the two operating points satisfies.
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

        let mut g = |x: f64| Ok(2.0 - x); // decreasing
        let r = brentq(&mut g, 0.0, 5.0, 2e-12, RTOL, 100).unwrap();
        assert!((r - 2.0).abs() < 1e-10);
    }

    #[test]
    fn brentq_no_sign_change() {
        let mut f = |x: f64| Ok(x * x + 1.0);
        assert_eq!(brentq(&mut f, -1.0, 1.0, 1e-8, RTOL, 100), Err(RootError::NoSignChange));
    }

    #[test]
    fn secant_converges() {
        let mut f = |x: f64| Ok(x * x - 2.0);
        let r = secant(&mut f, 1.0, 1e-10, 50).unwrap();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn minimize_finds_dip() {
        let mut f = |x: f64| (x - 2.0) * (x - 2.0);
        let (x, fx) = minimize(&mut f, 0.0, 5.0, 1e-8, 200);
        assert!((x - 2.0).abs() < 1e-4);
        assert!(fx < 1e-8);
    }
}
