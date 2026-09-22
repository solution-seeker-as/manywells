//! Steady-state drift-flux for two-phase flow in vertical well. The momentum
//! differential equation is discretized with backwards Euler. Temperature has
//! been integrated analytically with boundary value T(z = 0) = T_r.
//! The BVP is solved by single shooting on the bottomhole pressure.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::brentq::{brentq, minimize, RootError, RTOL};
use crate::choke::{extract_choke, BernoulliChokeModel, ChokeSpec};
use crate::constants::{CF_PRES, STD_GRAVITY};
use crate::inflow::{extract_inflow, InflowSpec, ProductivityIndex};
use crate::math::max_approx;
use crate::slip;
use crate::SimError;

const DIM_X: usize = 7; // state: [p, v_g, v_l, alpha, rho_g, rho_l, T]

/// Well properties -- same fields and defaults as simulator.WellProperties.
/// The inflow/choke attributes hold the Python-facing model objects; their
/// parameters are re-read on every simulate() call, so mutating e.g.
/// `wp.inflow.f_g` from Python works exactly like it does with the dataclass.
#[pyclass(skip_from_py_object)]
pub struct WellProperties {
    #[pyo3(get, set)]
    pub L: f64,
    #[pyo3(get, set)]
    pub D: f64,
    #[pyo3(get, set)]
    pub rho_l: f64,
    #[pyo3(get, set)]
    pub R_s: f64,
    #[pyo3(get, set)]
    pub cp_g: f64,
    #[pyo3(get, set)]
    pub cp_l: f64,
    #[pyo3(get, set)]
    pub f_D: f64,
    #[pyo3(get, set)]
    pub h: f64,
    #[pyo3(get, set)]
    pub slip: Option<Py<PyAny>>, // accepted for API parity; the slip constants are built in
    #[pyo3(get, set)]
    pub inflow: Py<PyAny>,
    #[pyo3(get, set)]
    pub choke: Py<PyAny>,
}

#[pymethods]
impl WellProperties {
    #[new]
    #[pyo3(signature = (L=2000.0, D=0.1554, rho_l=850.0, R_s=518.3, cp_g=2225.0, cp_l=4180.0,
                        f_D=0.05, h=20.0, slip=None, inflow=None, choke=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        L: f64,
        D: f64,
        rho_l: f64,
        R_s: f64,
        cp_g: f64,
        cp_l: f64,
        f_D: f64,
        h: f64,
        slip: Option<Py<PyAny>>,
        inflow: Option<Py<PyAny>>,
        choke: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if L <= 0.0 {
            return Err(PyValueError::new_err("Pipe length must be positive"));
        }
        if D <= 0.0 {
            return Err(PyValueError::new_err("Pipe diameter must be positive"));
        }

        // Same defaults as the Python dataclass / __post_init__
        let inflow = match inflow {
            Some(o) => o,
            None => Py::new(py, ProductivityIndex { k_l: 0.5, f_g: 0.1379 })?.into_any(),
        };
        let choke = match choke {
            Some(o) => o,
            None => {
                let a = std::f64::consts::PI * (D / 2.0) * (D / 2.0);
                let default = BernoulliChokeModel::new_default(0.1 * a);
                Py::new(py, default)?.into_any()
            }
        };

        Ok(Self { L, D, rho_l, R_s, cp_g, cp_l, f_D, h, slip, inflow, choke })
    }

    /// Cross-sectional area of pipe (m²)
    #[getter]
    fn A(&self) -> f64 {
        std::f64::consts::PI * (self.D / 2.0) * (self.D / 2.0)
    }
}

/// Boundary conditions -- same fields and defaults as simulator.BoundaryConditions.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct BoundaryConditions {
    #[pyo3(get, set)]
    pub p_r: f64,
    #[pyo3(get, set)]
    pub p_s: f64,
    #[pyo3(get, set)]
    pub T_r: f64,
    #[pyo3(get, set)]
    pub T_s: f64,
    #[pyo3(get, set)]
    pub u: f64,
    #[pyo3(get, set)]
    pub w_lg: f64,
}

#[pymethods]
impl BoundaryConditions {
    #[new]
    #[pyo3(signature = (p_r=170.0, p_s=20.0, T_r=373.15, T_s=277.15, u=1.0, w_lg=0.0))]
    fn new(p_r: f64, p_s: f64, T_r: f64, T_s: f64, u: f64, w_lg: f64) -> PyResult<Self> {
        if p_r <= 0.0 {
            return Err(PyValueError::new_err("Reservoir pressure must be positive"));
        }
        if p_s <= 0.0 {
            return Err(PyValueError::new_err("Separator pressure must be positive"));
        }
        if !(0.0..=1.0).contains(&u) {
            return Err(PyValueError::new_err("Choke opening must be in [0, 1]"));
        }
        if w_lg < 0.0 {
            return Err(PyValueError::new_err("Gas lift rate must be non-negative"));
        }
        Ok(Self { p_r, p_s, T_r, T_s, u, w_lg })
    }
}

/// Outcome of solving a single cell's discretized momentum equation.
enum CellStep {
    /// A genuine root of the discretized momentum equation was found.
    /// Carries `(cell_state, p_next)`.
    Feasible([f64; DIM_X], f64),
    /// No subsonic root exists: the cell is choked. We continue the march at the
    /// sonic pressure `p*` so the outer residual stays continuous, but this is NOT
    /// a solution of the discretized equations, so the march is flagged as failed.
    /// Carries `(cell_state, p_star)`.
    Choked([f64; DIM_X], f64),
}

/// Plain-Rust snapshot of (wp, bc, n_cells), extracted once per simulate() call
/// so the hot loop never touches Python objects.
struct Core {
    a: f64,
    l: f64,
    d: f64,
    rho_l: f64,
    r_s: f64,
    cp_g: f64,
    cp_l: f64,
    f_d: f64,
    h: f64,
    inflow: InflowSpec,
    choke: ChokeSpec,
    p_r: f64,
    p_s: f64,
    t_r: f64,
    t_s: f64,
    u: f64,
    w_lg: f64,
    n_cells: usize,
    delta_z: f64,
}

impl Core {
    /// Analytic solution for T(z)
    fn temp(&self, z: f64, w_g: f64, w_l: f64) -> f64 {
        let k = std::f64::consts::PI * self.h * self.d / (self.cp_g * w_g + self.cp_l * w_l);
        self.t_r - z / self.l * (self.t_r - self.t_s)
            + 1.0 / (self.l * k) * (self.t_r - self.t_s) * (1.0 - (-k * z).exp())
    }

    fn v_g(&self, w_g: f64, alpha: f64, rho_g: f64) -> f64 {
        w_g / (self.a * alpha * rho_g)
    }

    fn v_l(&self, w_l: f64, alpha: f64, rho_l: f64) -> f64 {
        w_l / (self.a * (1.0 - alpha) * rho_l)
    }

    fn rho_gas(&self, p: f64, t: f64) -> f64 {
        CF_PRES * p / (self.r_s * t)
    }

    /// solve_alpha: fixed-point iteration for the void fraction satisfying the slip law.
    /// alpha = w_g / (A * rho_g * (C_0*v_m + v_inf)); iterated until it stops moving
    /// or hits MAX_ITER
    fn solve_alpha(&self, w_g: f64, w_l: f64, rho_g: f64, rho_l: f64, t: f64) -> Option<f64> {
        const ALPHA_LO: f64 = 1e-6;
        const ALPHA_HI: f64 = 1.0 - 1e-6;
        const ALPHA_TOL: f64 = 1e-3;
        const MAX_ITER: usize = 100;
        let rho_g = if rho_g <= 0.0 { 1e-3 } else { rho_g };
        // No gas: single-phase liquid, alpha ~ 0.
        if w_g <= 0.0 {
            return Some(ALPHA_LO);
        }

        let vm = w_g / (self.a * rho_g) + w_l / (self.a * rho_l);
        let mut alpha = (w_g / (self.a * rho_g * (1.1 * vm + 0.5 + 1e-6))).clamp(ALPHA_LO, ALPHA_HI);
        let mut converged = false;
        for _ in 0..MAX_ITER {
            let v_g = self.v_g(w_g, alpha, rho_g);
            let v_l = self.v_l(w_l, alpha, rho_l);
            let (c_0, v_inf) = slip::identify_parameters(v_g, v_l, alpha, rho_g, rho_l, t, self.d);
            let alpha_next = (w_g / (self.a * rho_g * (c_0 * vm + v_inf + 1e-6))).clamp(ALPHA_LO, ALPHA_HI);
            converged = (alpha_next - alpha).abs() < ALPHA_TOL;
            alpha = alpha_next;
            if converged {
                break;
            }
        }

        if !converged {
            return None;
        }
        
        // Guard against non-finite iterates so feasibility holes are still detected.
        alpha.is_finite().then_some(alpha)
    }

    /// Momentum flux.
    fn mom(&self, p: f64, alpha: f64, rho_g: f64, rho_l: f64, v_g: f64, v_l: f64) -> f64 {
        (alpha * rho_g * v_g * v_g + (1.0 - alpha) * rho_l * v_l * v_l) / CF_PRES + p
    }

    /// Friction term.
    fn f_fric(&self, alpha: f64, rho_g: f64, rho_l: f64, v_g: f64, v_l: f64) -> f64 {
        let rho_m = alpha * rho_g + (1.0 - alpha) * rho_l;
        let v_m = alpha * v_g + (1.0 - alpha) * v_l;
        self.f_d / (2.0 * self.d) * rho_m * v_m * v_m.abs()
    }

    /// Gravity term.
    fn g_grav(&self, alpha: f64, rho_g: f64, rho_l: f64) -> f64 {
        let rho_m = alpha * rho_g + (1.0 - alpha) * rho_l;
        rho_m * STD_GRAVITY
    }

    /// return cell state or None if solve_alpha fails.
    fn compute_cell_state(&self, z: f64, p: f64, w_l: f64, w_g: f64) -> Option<[f64; DIM_X]> {
        let t = self.temp(z, w_g, w_l);
        let rho_g = self.rho_gas(p, t);
        let rho_l = self.rho_l;
        let alpha = self.solve_alpha(w_g, w_l, rho_g, rho_l, t)?;
        let v_g = self.v_g(w_g, alpha, rho_g);
        let v_l = self.v_l(w_l, alpha, rho_l);
        Some([p, v_g, v_l, alpha, rho_g, rho_l, t]) // order is important
    }

    /// Solve one cell at pressure `p_in`.
    ///
    /// The discretized momentum equation `M(p_next) = l2` has a U-shaped `M` with a
    /// minimum at the sonic pressure `p*` (fold/choke point). There are two roots:
    /// the physical subsonic one at `p_next > p*` and a spurious supersonic one at
    /// `p_next < p*`. To always land on the physical branch we first locate `p*` by
    /// minimizing `M`, then bracket the root on `[p*, p_in]` (where `M` is monotone).
    ///
    /// - Root found  -> `CellStep::Feasible`.
    /// - No root (choked, `M(p*) > l2`) -> `CellStep::Choked` at `p*` for continuity.
    /// - Cell state itself not computable (bad pressure / slip law) -> `None`.
    fn solve_cell(&self, i: usize, p_in: f64, w_l: f64, w_g: f64) -> Option<CellStep> {
        let z = i as f64 * self.delta_z;
        let z_next = z + self.delta_z;

        // Current cell
        let x = self.compute_cell_state(z, p_in, w_l, w_g)?;
        let [p, v_g, v_l, alpha, rho_g, rho_l, _t] = x;

        // Next cell
        let rho_l_next = self.rho_l;
        let t_next = self.temp(z_next, w_g, w_l);

        let l2 = self.mom(p, alpha, rho_g, rho_l, v_g, v_l);

        // Discretized momentum equation, implicit Euler step (friction/gravity at the next cell)
        let mut diff_eq = |p_next: f64| -> Result<f64, RootError> {
            let rho_g_next = self.rho_gas(p_next, t_next);
            let alpha_next = self.solve_alpha(w_g, w_l, rho_g_next, rho_l_next, t_next).ok_or(RootError::NoSignChange)?;
            let v_g_next = self.v_g(w_g, alpha_next, rho_g_next);
            let v_l_next = self.v_l(w_l, alpha_next, rho_l_next);

            let l1 = self.mom(p_next, alpha_next, rho_g_next, rho_l_next, v_g_next, v_l_next);
            let l3 = self.delta_z
                * (self.f_fric(alpha_next, rho_g_next, rho_l_next, v_g_next, v_l_next)
                    + self.g_grav(alpha_next, rho_g_next, rho_l_next))
                / CF_PRES;

            Ok(l1 - l2 + l3)
        };

        const P_MIN: f64 = 1e-3;
        // Fast path: the physical (subsonic) root sits just below p_in, so try a
        // narrow bracket [p_in - frac*(p_in - p_s), p_in] first. The upper endpoint is
        // pinned at p_in (always above p*), so any sign change here brackets the
        // subsonic root and never the spurious supersonic one -- no extra check needed.
        // Only when this misses do we pay for the p* minimization below.
        let lo = (p_in - 0.1 * (p_in - self.p_s)).max(P_MIN);
        if lo < p_in {
            if let Ok(p_next) = brentq(&mut diff_eq, lo, p_in, 1e-6, RTOL, 100) {
                if p_next > 0.0 {
                    return Some(CellStep::Feasible(x, p_next));
                }
            }
        }
        // Slow path: locate the sonic pressure p* = argmin M on (P_MIN, p_in] and
        // bracket the subsonic root on [p*, p_in]. Infeasible evaluations map to +inf
        // so the minimizer avoids them. No sign change on [p*, p_in] => choked cell.
        //
        // Most solve_cell calls come from off-solution trial marches in shoot() that
        // choke, so this p* search dominates runtime. A coarse tolerance is fine: p*
        // is only used to continue the (discarded) choked march smoothly and as the
        // lower bracket for the subsonic root, so 1e-2 bar leaves the accepted
        // solutions unchanged while roughly halving the golden-section iterations.
        let p_star = {
            let mut m = |p_next: f64| match diff_eq(p_next) {
                Ok(v) if v.is_finite() => v,
                _ => f64::INFINITY,
            };
            let (p_star, _) = minimize(&mut m, P_MIN, p_in, 1e-2, 200);
            p_star
        };
        match brentq(&mut diff_eq, p_star, p_in, 1e-6, RTOL, 100) {
            Ok(p_next) if p_next > 0.0 => Some(CellStep::Feasible(x, p_next)),
            _ => Some(CellStep::Choked(x, p_star)),
        }
    }

    /// March from z=0 to z=L, returning the flat state list.
    fn simulate_inner(&self, p0: f64) -> (Vec<f64>, bool) {
        // Inflow from reservoir + lift gas
        let (w_l, mut w_g) = self.inflow.mass_flow_rates(p0, self.p_r);
        w_g += self.w_lg;

        let mut x: Vec<f64> = Vec::with_capacity((self.n_cells + 1) * DIM_X);
        let mut p = p0;
        let mut prev: Option<[f64; DIM_X]> = None; // last computed cell state
        let mut failed = false; // some cell did not solve the discretized equations
        let mut stopped = false; // a cell state could not be computed at all
        for i in 0..self.n_cells {
            if !stopped {
                match self.solve_cell(i, p, w_l, w_g) {
                    // Genuine root: keep marching.
                    Some(CellStep::Feasible(cell, p_next)) => {
                        prev = Some(cell);
                        p = p_next;
                        x.extend_from_slice(&cell);
                        continue;
                    }
                    // Choked: not a real solution, but continue at the sonic pressure
                    // so the residual is smooth. Flag the whole march as failed.
                    Some(CellStep::Choked(cell, p_next)) => {
                        prev = Some(cell);
                        p = p_next;
                        x.extend_from_slice(&cell);
                        failed = true;
                        continue;
                    }
                    // Cell state not computable: stop and propagate the last state.
                    None => {
                        if let Some(cell) = self.compute_cell_state(i as f64 * self.delta_z, p, w_l, w_g) {
                            prev = Some(cell);
                        }
                        failed = true;
                        stopped = true;
                    }
                }
            }
            // Stopped: propagate the previous good state to the top.
            match prev {
                Some(cell) => x.extend_from_slice(&cell),
                None => x.extend_from_slice(&[f64::NAN; DIM_X]), // full-length null solution
            }
        }

        // z = L
        let top = if !stopped {
            self.compute_cell_state(self.n_cells as f64 * self.delta_z, p, w_l, w_g)
        } else {
            None
        };

        match top.or(prev) {
            Some(cell) => x.extend_from_slice(&cell),
            None => x.extend_from_slice(&[f64::NAN; DIM_X]),
        }
        
        debug_assert_eq!(x.len(), (self.n_cells + 1) * DIM_X);
        (x, failed || prev.is_none())
    }

    /// Squared choke residual
    ///     R = w_m² - (K_c * sigma(u))² * 2 * rho * dp / Phi
    /// with the smooth-max critical pressure, rho = rho_m (Bernoulli) or rho_l (Simpson).
    fn right_boundary(&self, last_cell: &[f64]) -> f64 {
        let (p, v_g, v_l, alpha, rho_g, rho_l) =
            (last_cell[0], last_cell[1], last_cell[2], last_cell[3], last_cell[4], last_cell[5]);

        let w_g = self.a * alpha * rho_g * v_g;
        let w_l = self.a * (1.0 - alpha) * rho_l * v_l;
        let w_m = w_g + w_l;

        let chk = self.choke.choke_opening(self.u);
        let p_c = max_approx(self.choke.cpr * p, self.p_s); // approximation of max(cpr * p_in, p_out)
        let dp = CF_PRES * (p - p_c); // pressure difference (Pa)

        let (rho, multiplier) = if self.choke.simpson {
            let x_g = w_g / w_m; // mass fraction of gas
            let s = (rho_l / rho_g).powf(1.0 / 6.0);
            (rho_l, (1.0 + x_g * (s - 1.0)) * (1.0 + x_g * (s.powi(5) - 1.0)))
        } else {
            (alpha * rho_g + (1.0 - alpha) * rho_l, 1.0)
        };

        let kc_chk = self.choke.k_c * chk;
        w_m * w_m - kc_chk * kc_chk * (2.0 * rho * dp) / multiplier
    }

    /// Right boundary residual. How "wrong" is the guess p0 (bottomhole pressure)? 
    /// Note: the outer "shoot" method assumes that this residual function
    /// has a general 'U' shape (but does not assume that it is convex).
    fn residual(&self, p0: f64) -> Option<(f64, bool)> {
        let (x, failed) = self.simulate_inner(p0);
        let r = self.right_boundary(&x[x.len() - DIM_X..]);
        r.is_finite().then_some((r, failed))
    }

    /// Find every bottomhole pressure p0 that satisfies the wellhead choke condition.
    ///
    fn shoot(&self) -> Vec<f64> {
        const N: usize = 100;
        let p_lo = self.p_s + 1e-3;
        let p_hi = self.p_r - 1e-6;
        let step = (p_hi - p_lo) / N as f64;

        // Walk right-to-left until R first goes negative. R is +,-,+ across the
        // interval, so the two operating points flank that negative region:
        //   right root in [p_neg, p_prev]  (p_prev = last non-negative sample)
        //   left  root in [p_lo,  p_neg]   (p_lo assumed non-negative)
        // Bracket both, then stop.
        // Note: This method assumes that the residual has a general "U" shape (+ then - then +).
        let mut prev: Option<f64> = None; // previous non-negative feasible sample p
        let mut roots = Vec::new();
        for k in 0..=N {
            let p = p_hi - step * k as f64;
            let r = match self.residual(p) {
                Some((r, _)) => r,
                None => { prev = None; continue; } // hole: can't bracket across it
            };
            if r < 0.0 {
                // Right root (higher p0): between this negative sample and the last non-negative one.
                if let Some(p_prev) = prev {
                    if let Ok(root) = brentq(
                        &mut |p0| self.residual(p0).map(|(r, _)| r).ok_or(RootError::NoSignChange),
                        p, p_prev, 1e-6, RTOL, 100,
                    ) {
                        if let Some((_, false)) = self.residual(root) { roots.push(root); }
                    }
                }
                // Left root (lower p0): between p_lo and this negative sample.
                if let Ok(root) = brentq(
                    &mut |p0| self.residual(p0).map(|(r, _)| r).ok_or(RootError::NoSignChange),
                    p_lo, p, 1e-6, RTOL, 100,
                ) {
                    if let Some((_, false)) = self.residual(root) { roots.push(root); }
                }
                return roots;
            }
            prev = Some(p);
        }
        return roots;
    }
}

/// Steady-state drift-flux simulator
#[pyclass(skip_from_py_object)]
pub struct SSDFSimulator {
    wp: Py<WellProperties>,
    bc: Py<BoundaryConditions>,
    #[pyo3(get)]
    n_cells: usize,
    #[pyo3(get)]
    dim_x: usize,
}

impl SSDFSimulator {
    /// Snapshot the current wp/bc into a plain-Rust Core (re-reads the inflow/choke
    /// objects so Python-side mutations are picked up, like the Python dataclasses).
    fn core(&self, py: Python<'_>) -> PyResult<Core> {
        let wp = self.wp.borrow(py);
        let bc = self.bc.borrow(py);
        let inflow = extract_inflow(wp.inflow.bind(py))?;
        let choke = extract_choke(wp.choke.bind(py))?;
        Ok(Core {
            a: wp.A(),
            l: wp.L,
            d: wp.D,
            rho_l: wp.rho_l,
            r_s: wp.R_s,
            cp_g: wp.cp_g,
            cp_l: wp.cp_l,
            f_d: wp.f_D,
            h: wp.h,
            inflow,
            choke,
            p_r: bc.p_r,
            p_s: bc.p_s,
            t_r: bc.T_r,
            t_s: bc.T_s,
            u: bc.u,
            w_lg: bc.w_lg,
            n_cells: self.n_cells,
            delta_z: wp.L / self.n_cells as f64,
        })
    }
}

#[pymethods]
impl SSDFSimulator {
    #[new]
    #[pyo3(signature = (well_properties, boundary_conditions, n_cells=100))]
    fn new(well_properties: Py<WellProperties>, boundary_conditions: Py<BoundaryConditions>, n_cells: usize) -> Self {
        Self { wp: well_properties, bc: boundary_conditions, n_cells, dim_x: DIM_X }
    }

    #[getter]
    fn wp(&self, py: Python<'_>) -> Py<WellProperties> {
        self.wp.clone_ref(py)
    }

    #[setter]
    fn set_wp(&mut self, wp: Py<WellProperties>) {
        self.wp = wp;
    }

    #[getter]
    fn bc(&self, py: Python<'_>) -> Py<BoundaryConditions> {
        self.bc.clone_ref(py)
    }

    #[setter]
    fn set_bc(&mut self, bc: Py<BoundaryConditions>) {
        self.bc = bc;
    }

    #[getter]
    fn delta_z(&self, py: Python<'_>) -> f64 {
        self.wp.borrow(py).L / self.n_cells as f64
    }

    /// Simulate the well: find every valid bottomhole pressure p_0 on the physical
    /// interval, then march each one. Returns one flat state list per solution,
    /// ordered highest-p_0 (least drawdown) first. Errors only if no valid solution
    /// exists.
    fn simulate(&self, py: Python<'_>) -> PyResult<Vec<Vec<f64>>> {
        let core = self.core(py)?;

        let solutions = py.detach(|| {
            core.shoot()
                .into_iter()
                .filter_map(|pbh| {
                    let (x, failed) = core.simulate_inner(pbh);
                    (!failed).then_some(x) // shoot() already discards choked roots; guard anyway
                })
                .collect::<Vec<_>>()
        });

        if solutions.is_empty() {
            return Err(SimError::new_err("No valid bottomhole pressure found"));
        }

        Ok(solutions)
    }

    /// Outer shooting residual for a trial bottomhole pressure. Returns
    /// ``(R(p0), shady)`` where ``shady`` is true when R was computed from a
    /// propagated (non-physical) march, or ``None`` if p0 is entirely infeasible.
    fn _residual(&self, py: Python<'_>, p0: f64) -> PyResult<Option<(f64, bool)>> {
        let core = self.core(py)?;
        Ok(py.detach(|| core.residual(p0)))
    }

    /// Right-boundary (choke) residual for a wellhead state vector.
    fn _right_boundary_eqs(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<f64> {
        if x.len() < DIM_X {
            return Err(PyValueError::new_err(format!("state vector must have at least {DIM_X} entries")));
        }
        let core = self.core(py)?;
        Ok(core.right_boundary(&x[x.len() - DIM_X..]))
    }

    /// Represent a solution (flat list) as a pandas DataFrame
    fn solution_as_df(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Py<PyAny>> {
        let n = self.n_cells + 1;
        if x.len() != n * DIM_X {
            return Err(PyValueError::new_err(format!(
                "solution vector has {} entries, expected {} ({} cells x {})",
                x.len(),
                n * DIM_X,
                n,
                DIM_X
            )));
        }
        let delta_z = self.delta_z(py);

        let mut cols: [Vec<f64>; DIM_X] = Default::default();
        let mut z = Vec::with_capacity(n);
        let mut regime: Vec<&'static str> = Vec::with_capacity(n);
        for i in 0..n {
            let s = &x[i * DIM_X..(i + 1) * DIM_X];
            for (j, col) in cols.iter_mut().enumerate() {
                col.push(s[j]);
            }
            z.push(i as f64 * delta_z);
            regime.push(slip::flow_regime_name(s[1], s[2], s[3], s[4], s[5], s[6]));
        }

        let d = PyDict::new(py);
        d.set_item("z", z)?;
        for (name, col) in ["p", "v_g", "v_l", "alpha", "rho_g", "rho_l", "T"].iter().zip(cols) {
            d.set_item(name, col)?;
        }
        d.set_item("flow-regime", regime)?;

        let pd = py.import("pandas")?;
        Ok(pd.call_method1("DataFrame", (d,))?.unbind())
    }
}

#[cfg(test)]
mod tests {
    //! Ported from the subset of tests/test_simulator.py on the develop branch that
    //! does not require the Python interpreter. WellProperties / SSDFSimulator
    //! construction and simulate() are exercised through the Python API (see the
    //! Python-level tests), since they build Py<...> objects and need the GIL.
    //! The develop branch also uses a WellGeometry/FluidModel API that this port
    //! does not have, so those tests are not portable here.
    use super::*;

    fn bc(p_r: f64, p_s: f64, u: f64, w_lg: f64) -> PyResult<BoundaryConditions> {
        BoundaryConditions::new(p_r, p_s, 373.15, 277.15, u, w_lg)
    }

    #[test]
    fn boundary_conditions_defaults() {
        let bc = bc(170.0, 20.0, 1.0, 0.0).unwrap();
        assert_eq!(bc.p_r, 170.0);
        assert_eq!(bc.p_s, 20.0);
        assert_eq!(bc.u, 1.0);
        assert!((0.0..=1.0).contains(&bc.u));
    }

    #[test]
    fn boundary_conditions_reject_nonpositive_pressure() {
        assert!(bc(0.0, 20.0, 1.0, 0.0).is_err());
        assert!(bc(170.0, -1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn boundary_conditions_reject_invalid_choke_opening() {
        assert!(bc(170.0, 20.0, -0.1, 0.0).is_err());
        assert!(bc(170.0, 20.0, 1.5, 0.0).is_err());
    }

    #[test]
    fn boundary_conditions_reject_negative_lift_gas() {
        assert!(bc(170.0, 20.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn dim_x_is_seven() {
        assert_eq!(DIM_X, 7);
    }
}
