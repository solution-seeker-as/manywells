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
    use std::collections::HashMap;

    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    use crate::choke::{self, Choke, ChokeModel, Profile};
    use crate::discretization;
    use crate::inflow::Inflow;
    use crate::input::{OperatingPoint, WellSpec};
    use crate::march::Marcher;
    use crate::pvt::{fluid::Fluid, gas, oil};
    use crate::shoot;
    use crate::slip;
    use crate::smoothing;

    /// The operating point as (p_r, p_s, T_r, T_s, T_lg, u, w_lg), in bar, K and kg/s: the parameters of the Python
    /// system (discretization.PARAMS), in their order
    type Op = (f64, f64, f64, f64, f64, f64, f64);

    fn operating_point(op: Op) -> PyResult<OperatingPoint> {
        let (p_r, p_s, t_r, t_s, t_lg, u, w_lg) = op;
        let op = OperatingPoint { p_r, p_s, t_r, t_s, t_lg, u, w_lg };
        op.check().map_err(PyValueError::new_err)?;
        Ok(op)
    }

    /// A well, built once
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
                            K_c, profile))]
        #[allow(non_snake_case, clippy::too_many_arguments)]
        fn new(L: f64, D: f64, n_cells: usize, rho_l: f64, R_s: f64, cp_g: f64, cp_l: f64, f_g: f64, f_D: f64, h: f64,
               inflow: &str, inflow_coefficient: f64, choke: &str, K_c: f64, profile: &str) -> PyResult<Self> {
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
                choke: Choke::new(model, K_c, profile),
            };
            spec.check().map_err(PyValueError::new_err)?;
            Ok(Self { spec })
        }

        /// Every root the search finds at the operating point, sorted by p_0, and the work it took: the marches, the
        /// states computed, and the sign changes of R that were not accepted as roots (rejected)
        fn root_set(&self, py: Python<'_>, op: Op) -> PyResult<(Vec<Root>, HashMap<&'static str, usize>)> {
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
            let c = search.counts;
            let counts = HashMap::from([("marches", c.marches), ("states", c.states), ("rejected", search.rejected)]);
            Ok((roots, counts))
        }

        /// Every row of the system at state x, as (IDs, values), point by point in the order of DISC-11
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

        /// The march from p_0, as (x of the points reached, failed, below_separator)
        fn march(&self, op: Op, p_0: f64) -> PyResult<(Vec<f64>, bool, bool)> {
            let op = operating_point(op)?;
            let m = Marcher::new(&self.spec, &op).march(p_0);
            Ok((m.x, m.failed, m.below_separator))
        }

        /// For the tests against the component vectors (tests/test_spec_vectors.py): the core's function `name` at
        /// the arguments, with the well's own components where it needs them
        fn _component(&self, name: &str, args: Vec<f64>) -> PyResult<Vec<f64>> {
            component(&self.spec, name, &args).map_err(PyValueError::new_err)
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

    fn take<const N: usize>(name: &str, a: &[f64]) -> Result<[f64; N], String> {
        a.try_into().map_err(|_| format!("{name} takes {N} arguments, not {}", a.len()))
    }

    /// The core's component functions by name, for the vector tests
    fn component(spec: &WellSpec, name: &str, a: &[f64]) -> Result<Vec<f64>, String> {
        let regime_index = |label| ["annular", "slug-churn", "bubbly"].iter().position(|&r| r == label).unwrap() as f64;
        Ok(match name {
            "max_approx" => { let [x, y, eps] = take(name, a)?; vec![smoothing::max_approx(x, y, eps)] }
            "softmax" => { let y = take(name, a)?; smoothing::softmax3(y).to_vec() }
            "critical_pressure_ratio" => { let [gamma] = take(name, a)?; vec![choke::critical_pressure_ratio(gamma)] }
            "simpson_multiplier" => {
                let [x_g, rho_g, rho_l] = take(name, a)?;
                vec![choke::simpson_multiplier(x_g, rho_g, rho_l)]
            }
            "choke_opening" => { let [u] = take(name, a)?; vec![spec.choke.profile.opening(u)] }
            "choke_equation" => {
                let [u, p_in, p_out, rho, phi] = take(name, a)?;
                vec![spec.choke.choke_equation(u, p_in, p_out, rho, phi)]
            }
            "choke_rate" => {
                let [u, p_in, p_s, w_g, w_l, alpha, rho_g, rho_l] = take(name, a)?;
                vec![spec.choke.mass_flow_rate(u, p_in, p_s, w_g, w_l, alpha, rho_g, rho_l)]
            }
            "is_choked" => { let [p_in, p_out] = take(name, a)?; vec![spec.choke.is_choked(p_in, p_out) as u8 as f64] }
            "liquid_rate" => { let [p, p_r] = take(name, a)?; vec![spec.inflow.liquid_rate(p, p_r)] }
            "reservoir_gas_rate" => { let [w_res] = take(name, a)?; vec![spec.fluid.reservoir_gas_rate(w_res)] }
            "slip_parameters" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma, d] = take(name, a)?;
                let (c_0, v_inf) = slip::identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, d);
                vec![c_0, v_inf]
            }
            "regime_probabilities" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma] = take(name, a)?;
                slip::classify(v_g, v_l, alpha, rho_g, rho_l, sigma, spec.d).to_vec()
            }
            "regime" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma] = take(name, a)?;
                vec![regime_index(slip::regime_label(slip::classify(v_g, v_l, alpha, rho_g, rho_l, sigma, spec.d)))]
            }
            "harmathy_rise_velocity" => {
                let [rho_g, rho_l, sigma] = take(name, a)?;
                vec![slip::harmathy_rise_velocity(rho_g, rho_l, sigma)]
            }
            "taylor_rise_velocity" => {
                let [rho_g, rho_l, d] = take(name, a)?;
                vec![slip::taylor_rise_velocity(rho_g, rho_l, d)]
            }
            "ideal_gas_density" => { let [p, t, r_s] = take(name, a)?; vec![gas::ideal_gas_density(p, t, r_s)] }
            "api_from_density" => { let [rho] = take(name, a)?; vec![oil::api_from_density(rho)] }
            "dead_oil_surface_tension" => { let [rho, t] = take(name, a)?; vec![oil::dead_oil_surface_tension(rho, t)] }
            _ => return Err(format!("the core has no component {name:?}")),
        })
    }
}
