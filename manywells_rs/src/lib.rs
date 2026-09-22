//! 
//! manywells_rs: Rust version of manywells/simulator.py (steady-state drift-flux
//! well simulator) with alternative solver (implicit Euler with Brent root finder
//! + shooting method), exposed to Python via PyO3.
//!
//! TODO: CHANGE THIS:
//! Usage from Python:
//!
//!     from manywells_rs import (Simulator, WellProperties, BoundaryConditions,
//!                               Vogel, ProductivityIndex,
//!                               SimpsonChokeModel, BernoulliChokeModel, SimError)
//!
//!     wp = WellProperties(L=2000.0, inflow=Vogel(w_l_max=30.0, f_g=0.2),
//!                         choke=SimpsonChokeModel(K_c=1e-3, chk_profile='linear'))
//!     bc = BoundaryConditions(p_r=170.0, p_s=20.0, u=0.5)
//!     sim = Simulator(wp, bc)
//!     solutions = sim.simulate()          # list of solutions (>= 1), highest p0 first
//!     df = sim.solution_as_df(solutions[0])


#![allow(non_snake_case)] // field/argument names mirror the Python API (K_c, R_s, T_r, ...)
use pyo3::prelude::*;

pub mod brentq;
pub mod choke;
pub mod constants;
pub mod inflow;
pub mod math;
pub mod pvt;
pub mod simulator;
pub mod slip;

pyo3::create_exception!(
    manywells_rs,
    SimError,
    pyo3::exceptions::PyException,
    "Exception caused by simulator"
);

#[pymodule]
mod manywells_rs {
    #[pymodule_export]
    use crate::simulator::{BoundaryConditions, SSDFSimulator, WellProperties};

    #[pymodule_export]
    use crate::inflow::{ProductivityIndex, Vogel};

    #[pymodule_export]
    use crate::choke::{BernoulliChokeModel, SimpsonChokeModel};

    #[pymodule_export]
    use crate::SimError;
}
