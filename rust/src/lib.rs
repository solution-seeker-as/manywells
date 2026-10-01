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
    use crate::friction::{self, Correlation, Friction};
    use crate::geometry::Geometry;
    use crate::inflow::Inflow;
    use crate::thermal::{self, Thermal};
    use crate::discretization::State;
    use crate::input::{OperatingPoint, WellSpec};
    use crate::march::Marcher;
    use crate::pvt::fluid::{Fluid, FluidInputs, SurfaceTensionModel};
    use crate::pvt::{gas, mixture, oil, water};
    use crate::shoot;
    use crate::slip::{self, Slip};
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
        /// A well from the fields of WellProperties' components (src/manywells/solvers/rust.py, core_well), each
        /// option by its name, as in the Python classes
        #[new]
        #[pyo3(signature = (
            *,
            md, tvd, D,
            rho_o, rho_g, rho_w, gor, wlr, cp_g, cp_o, cp_w, ideal_gas, oil_model, surface_tension_model, p_sep, T_sep,
            p_bubble=None,
            friction, f_D=None, roughness=None, correlation=None,
            h, frictional_heating, gravity_term, lift_gas_mixing,
            C_0_annular, C_0_slug, C_0_bubbly, v_inf_annular,
            inflow, inflow_coefficient,
            choke, K_c, profile
        ))]
        #[allow(non_snake_case, clippy::too_many_arguments)]
        fn new(md: Vec<f64>, tvd: Vec<f64>, D: f64,
               rho_o: f64, rho_g: f64, rho_w: f64, gor: f64, wlr: f64, cp_g: f64, cp_o: f64, cp_w: f64, ideal_gas: bool,
               oil_model: &str, surface_tension_model: &str, p_sep: f64, T_sep: f64, p_bubble: Option<f64>,
               friction: &str, f_D: Option<f64>, roughness: Option<f64>, correlation: Option<&str>,
               h: f64, frictional_heating: bool, gravity_term: bool, lift_gas_mixing: bool,
               C_0_annular: f64, C_0_slug: f64, C_0_bubbly: f64, v_inf_annular: f64,
               inflow: &str, inflow_coefficient: f64,
               choke: &str, K_c: f64, profile: &str) -> PyResult<Self> {
            let black_oil = match oil_model {
                "dead_oil" => false,
                "black_oil" => true,
                m => return Err(PyValueError::new_err(format!("oil model {m:?} is not 'dead_oil' or 'black_oil'"))),
            };
            let surface_tension = match surface_tension_model {
                "oil" => SurfaceTensionModel::Oil,
                "liquid" => SurfaceTensionModel::Liquid,
                m => return Err(PyValueError::new_err(format!("surface tension model {m:?} is not 'oil' or 'liquid'"))),
            };
            let friction = match (friction, f_D, roughness, correlation) {
                ("fixed", Some(f_d), _, _) => Friction::FixedFactor { f_d },
                ("roughness", _, Some(roughness), Some("chen")) => {
                    Friction::Roughness { roughness, correlation: Correlation::Chen }
                }
                ("roughness", _, Some(roughness), Some("haaland")) => {
                    Friction::Roughness { roughness, correlation: Correlation::Haaland }
                }
                _ => return Err(PyValueError::new_err(
                    "friction is 'fixed' with f_D, or 'roughness' with roughness and correlation 'chen' or 'haaland'")),
            };
            let inflow = match inflow {
                "vogel" => Inflow::Vogel { w_l_max: inflow_coefficient },
                "pi" => Inflow::ProductivityIndex { k_l: inflow_coefficient },
                "fixed" => Inflow::FixedRate { w_l: inflow_coefficient },
                m => return Err(PyValueError::new_err(format!("inflow {m:?} is not 'vogel', 'pi' or 'fixed'"))),
            };
            let model = match choke {
                "simpson" => ChokeModel::Simpson,
                "bernoulli" => ChokeModel::Bernoulli,
                m => return Err(PyValueError::new_err(format!("choke {m:?} is not 'simpson' or 'bernoulli'"))),
            };
            let profile = Profile::from_name(profile).map_err(PyValueError::new_err)?;
            let spec = WellSpec {
                geometry: Geometry::from_grid(&md, &tvd, D).map_err(PyValueError::new_err)?,
                fluid: Fluid::new(FluidInputs { rho_o, rho_g, rho_w, gor, wlr, cp_g, cp_o, cp_w, ideal_gas, black_oil, p_sep,
                                                t_sep: T_sep, p_bubble, surface_tension }),
                friction,
                thermal: Thermal { h, frictional_heating, gravity_term, lift_gas_mixing },
                slip: Slip { c_0_annular: C_0_annular, c_0_slug: C_0_slug, c_0_bubbly: C_0_bubbly, v_inf_annular },
                inflow,
                choke: Choke::new(model, K_c, profile),
            };
            spec.check().map_err(PyValueError::new_err)?;
            Ok(Self { spec })
        }

        /// Every root the search finds at the operating point, sorted by p_0, and the work it took: the marches, the
        /// states computed, and the sign changes of R that were not accepted as roots (rejected)
        fn root_set(&self, py: Python<'_>, op: Op) -> PyResult<(Vec<Root>, HashMap<&'static str, usize>)> {
            let (spec, op) = (&self.spec, operating_point(op)?);
            let search = py.detach(|| shoot::root_set(spec, &op)).map_err(PyRuntimeError::new_err)?;
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
            let counts = HashMap::from([("marches", c.marches), ("states", c.states), ("rejected", search.rejected),
                                        ("temperature_solves", c.temperature_solves), ("step_outs", c.step_outs),
                                        ("chord_fallbacks", c.chord_fallbacks),
                                        ("temperature_failures", c.temperature_failures),
                                        ("non_finite", c.non_finite)]);
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
            "min_approx" => { let [x, y, eps] = take(name, a)?; vec![smoothing::min_approx(x, y, eps)] }
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
            "gas_parameters" => { take::<0>(name, a)?; vec![spec.fluid.sg_gas, spec.fluid.m_g, spec.fluid.r_s] }
            "fluid_parameters" => {
                take::<0>(name, a)?;
                vec![spec.fluid.f_g, spec.fluid.rho_l, spec.fluid.cp_l, spec.fluid.x_o]
            }
            "slip_parameters" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma, d, cos_incl] = take(name, a)?;
                let (c_0, v_inf) = spec.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, d, cos_incl);
                vec![c_0, v_inf]
            }
            "regime_probabilities" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl] = take(name, a)?;
                spec.slip.classify(v_g, v_l, alpha, rho_g, rho_l, sigma, spec.geometry.d, cos_incl).to_vec()
            }
            "regime" => {
                let [v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl] = take(name, a)?;
                let probs = spec.slip.classify(v_g, v_l, alpha, rho_g, rho_l, sigma, spec.geometry.d, cos_incl);
                vec![regime_index(slip::regime_label(probs))]
            }
            "harmathy_rise_velocity" => {
                let [rho_g, rho_l, sigma] = take(name, a)?;
                vec![slip::harmathy_rise_velocity(rho_g, rho_l, sigma)]
            }
            "taylor_rise_velocity" => {
                let [rho_g, rho_l, d] = take(name, a)?;
                vec![slip::taylor_rise_velocity(rho_g, rho_l, d)]
            }
            "ambient_temperature" => {
                let [tvd_frac, t_r, t_s] = take(name, a)?;
                vec![thermal::ambient_temperature(tvd_frac, t_r, t_s)]
            }
            "inflow_temperature" => {
                let [w_res, w_lg, t_r, t_lg] = take(name, a)?;
                vec![spec.thermal.inflow_temperature(w_res, w_lg, t_r, t_lg, &spec.fluid)]
            }
            "temperature_gradient" => {
                let [p, v_g, v_l, alpha, rho_g, rho_l, t, t_a, f, cos_incl] = take(name, a)?;
                let s = State { p, v_g, v_l, alpha, rho_g, rho_l, t };
                vec![spec.thermal.temperature_gradient(&s, &spec.fluid, t_a, f, cos_incl, spec.geometry.d)]
            }
            "gas_density" => { let [p, t] = take(name, a)?; vec![spec.fluid.gas_density(p, t)] }
            "z_factor" => { let [p, t] = take(name, a)?; vec![spec.fluid.z_factor(p, t)] }
            "sutton_pseudo_critical" => {
                let [sg_gas] = take(name, a)?;
                let (ppc, tpc) = gas::sutton_pseudo_critical(sg_gas);
                vec![ppc, tpc]
            }
            "separator_gravity" => {
                let [api, sg_gas, p_sep, t_sep] = take(name, a)?;
                vec![oil::separator_gravity(api, sg_gas, p_sep * crate::units::CF_BAR, t_sep)]
            }
            "live_oil_surface_tension" => {
                let [sigma_od, rs_scf] = take(name, a)?;
                vec![oil::live_oil_surface_tension(sigma_od, rs_scf)]
            }
            "surface_tension" => { let [p, t, rho_l] = take(name, a)?; vec![spec.fluid.surface_tension(p, t, rho_l)] }
            "rs" => { let [p, t] = take(name, a)?; vec![spec.fluid.rs(p, t)] }
            "bo" => { let [p, t] = take(name, a)?; vec![spec.fluid.bo(p, t)] }
            "liquid_density" => { let [p, t] = take(name, a)?; vec![spec.fluid.liquid_density(p, t)] }
            "phase_rates" => {
                let [p, t, w_res, w_lg] = take(name, a)?;
                let (w_g, w_l) = spec.fluid.phase_rates(p, t, w_res, w_lg);
                vec![w_g, w_l]
            }
            "friction" => {
                let [p, v_g, v_l, alpha, rho_g, rho_l, t] = take(name, a)?;
                let s = State { p, v_g, v_l, alpha, rho_g, rho_l, t };
                let d = spec.geometry.d;
                vec![spec.friction.friction_factor(&s, &spec.fluid, d), spec.friction.pressure_gradient(&s, &spec.fluid, d)]
            }
            "chen_friction_factor" => { let [re, eps] = take(name, a)?; vec![friction::chen_friction_factor(re, eps)] }
            "haaland_friction_factor" => { let [re, eps] = take(name, a)?; vec![friction::haaland_friction_factor(re, eps)] }
            "friction_factor_of_re" => {
                let [re, eps] = take(name, a)?;
                let Friction::Roughness { correlation, .. } = spec.friction else {
                    return Err("the well's friction is not from roughness".into());
                };
                vec![friction::friction_factor_of_re(re, eps, correlation)]
            }
            "gas_viscosity" => { let [t, rho_g, m_g] = take(name, a)?; vec![gas::gas_viscosity(t, rho_g, m_g)] }
            "dead_oil_viscosity" => { let [api, t] = take(name, a)?; vec![oil::dead_oil_viscosity(api, t)] }
            "live_oil_viscosity" => { let [mu, rs_scf] = take(name, a)?; vec![oil::live_oil_viscosity(mu, rs_scf)] }
            "water_viscosity" => { let [t] = take(name, a)?; vec![water::water_viscosity(t)] }
            "liquid_viscosity" => { let [p, t] = take(name, a)?; vec![spec.fluid.liquid_viscosity(p, t)] }
            "mixture_viscosity" => {
                let [mu_l, mu_g, alpha, rho_l, rho_g] = take(name, a)?;
                vec![mixture::mixture_viscosity(mu_l, mu_g, alpha, rho_l, rho_g)]
            }
            "ideal_gas_density" => { let [p, t, r_s] = take(name, a)?; vec![gas::ideal_gas_density(p, t, r_s)] }
            "api_from_density" => { let [rho] = take(name, a)?; vec![oil::api_from_density(rho)] }
            "dead_oil_surface_tension" => { let [rho, t] = take(name, a)?; vec![oil::dead_oil_surface_tension(rho, t)] }
            _ => return Err(format!("the core has no component {name:?}")),
        })
    }
}
