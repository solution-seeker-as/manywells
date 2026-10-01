// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port by Kajrakso, rust_implementation), restructured 01 October 2026

//! The Rust core of ManyWells (specs/architecture.md, Rust core): the steady-state drift-flux model of a well,
//! solved by shooting on the bottomhole pressure. It covers the v1.0.0 configuration so far, and is built as the
//! Python extension manywells._core; the Python layer (src/manywells/solvers/rust.py) converts the dataclasses to a
//! well once and builds the root set from what the core returns.
//!
//! The model's parts follow specs/model/, one module per spec file. On top of them, discretization.rs holds the rows
//! of each grid point, march.rs solves them from the bottomhole to the wellhead, and shoot.rs finds the roots.

pub mod choke;
pub mod discretization;
pub mod friction;
pub mod geometry;
pub mod inflow;
pub mod input;
pub mod march;
pub mod pvt;
pub mod scalar;
pub mod shoot;
pub mod slip;
pub mod smoothing;
pub mod thermal;
pub mod units;

/// The bindings, as the module manywells._core
#[cfg(feature = "python")]
#[pyo3::pymodule]
mod _core {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    use crate::choke::{Choke, ChokeModel, Profile};
    use crate::discretization;
    use crate::inflow::Inflow;
    use crate::input::{OperatingPoint, WellSpec};
    use crate::pvt::fluid::Fluid;
    use crate::shoot;

    /// The operating point as (p_r, p_s, T_r, T_s, u, w_lg), in bar, K and kg/s
    type Op = (f64, f64, f64, f64, f64, f64);

    fn operating_point(op: Op) -> PyResult<OperatingPoint> {
        let (p_r, p_s, t_r, t_s, u, w_lg) = op;
        let op = OperatingPoint { p_r, p_s, t_r, t_s, u, w_lg };
        op.check().map_err(PyValueError::new_err)?;
        Ok(op)
    }

    /// A well in the v1.0.0 configuration, built once
    #[pyclass(frozen)]
    struct Well {
        spec: WellSpec,
    }

    /// A root, with its state x (7 (N + 1) values, state-vector order) and what it carries besides
    #[pyclass(frozen, get_all)]
    struct Root {
        x: Vec<f64>,
        rising: bool,
        slope: f64,
        choked: bool,
        flow_regime: Vec<String>,
        w_res: f64,
        w_g_res: f64,
    }

    #[pymethods]
    impl Well {
        #[new]
        #[pyo3(signature = (*, L, D, n_cells, rho_l, R_s, cp_g, cp_l, f_g, f_D, h, inflow, inflow_coefficient, choke,
                            K_c, cpr, profile))]
        #[allow(non_snake_case, clippy::too_many_arguments)]
        fn new(L: f64, D: f64, n_cells: usize, rho_l: f64, R_s: f64, cp_g: f64, cp_l: f64, f_g: f64, f_D: f64, h: f64,
               inflow: &str, inflow_coefficient: f64, choke: &str, K_c: f64, cpr: f64, profile: &str) -> PyResult<Self> {
            let inflow = match inflow {
                "vogel" => Inflow::Vogel { w_l_max: inflow_coefficient },
                "pi" => Inflow::ProductivityIndex { k_l: inflow_coefficient },
                _ => return Err(PyValueError::new_err(format!("inflow {inflow:?} is not 'vogel' or 'pi'"))),
            };
            let model = match choke {
                "simpson" => ChokeModel::Simpson,
                "bernoulli" => ChokeModel::Bernoulli,
                _ => return Err(PyValueError::new_err(format!("choke {choke:?} is not 'simpson' or 'bernoulli'"))),
            };
            let profile = Profile::from_name(profile).map_err(PyValueError::new_err)?;
            let spec = WellSpec {
                l: L,
                d: D,
                n_cells,
                fluid: Fluid { rho_l, r_s: R_s, cp_g, cp_l, f_g },
                f_d: f_D,
                h,
                inflow,
                choke: Choke { model, k_c: K_c, cpr, profile },
            };
            spec.check().map_err(PyValueError::new_err)?;
            Ok(Self { spec })
        }

        /// Every root the search finds at the operating point, sorted by p_0, and the number of marches it took
        fn root_set(&self, py: Python<'_>, op: Op) -> PyResult<(Vec<Root>, usize)> {
            let (spec, op) = (self.spec, operating_point(op)?);
            let search = py.detach(|| shoot::root_set(&spec, &op)).map_err(PyRuntimeError::new_err)?;
            let roots = search.roots.into_iter().map(|r| Root {
                x: r.x,
                rising: r.rising,
                slope: r.slope,
                choked: r.choked,
                flow_regime: r.flow_regime.into_iter().map(String::from).collect(),
                w_res: r.w_res,
                w_g_res: r.w_g_res,
            }).collect();
            Ok((roots, search.marches))
        }

        /// Every row of the system at state x, as (IDs, values), point by point in the order of DISC-6
        fn rows(&self, op: Op, x: Vec<f64>) -> PyResult<(Vec<String>, Vec<f64>)> {
            self.check_length(&x)?;
            let rows = discretization::rows(&self.spec, &operating_point(op)?, &x);
            Ok(rows.into_iter().map(|(id, v)| (id.to_string(), v)).unzip())
        }

        /// The regime label at each point of state x
        fn flow_regimes(&self, x: Vec<f64>) -> PyResult<Vec<String>> {
            self.check_length(&x)?;
            Ok(discretization::flow_regimes(&self.spec, &x).into_iter().map(String::from).collect())
        }

        /// The shooting residual R(p_0), or None if it is not finite
        fn residual(&self, op: Op, p_0: f64) -> PyResult<Option<f64>> {
            Ok(shoot::residual(&self.spec, &operating_point(op)?, p_0))
        }
    }

    impl Well {
        fn check_length(&self, x: &[f64]) -> PyResult<()> {
            if x.len() != self.spec.n_x() {
                return Err(PyValueError::new_err(format!("the state has {} values, not 7 (N + 1) = {}", x.len(),
                                                         self.spec.n_x())));
            }
            Ok(())
        }
    }
}
