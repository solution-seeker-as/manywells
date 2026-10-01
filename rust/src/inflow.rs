// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The bottom boundary (specs/model/inflow.md): the reservoir liquid rate at the bottomhole pressure, or a fixed one.
//! The gas rate follows from the fluid (INF-4, pvt/fluid.rs).

#[derive(Clone, Copy, Debug)]
pub enum Inflow {
    /// Vogel's inflow performance relationship, with the liquid rate at zero bottomhole pressure (kg/s)
    Vogel { w_l_max: f64 },
    /// Linear inflow, with the liquid productivity index (kg/s/bar)
    ProductivityIndex { k_l: f64 },
    /// A fixed liquid rate (kg/s), whatever the bottomhole pressure
    FixedRate { w_l: f64 },
}

impl Inflow {
    /// Reservoir liquid mass rate (kg/s) at bottomhole pressure p (bar) and reservoir pressure p_r (bar)
    pub fn liquid_rate(&self, p: f64, p_r: f64) -> f64 {
        match *self {
            Inflow::Vogel { w_l_max } => {
                let r = p / p_r;
                w_l_max * (1.0 - 0.2 * r - 0.8 * (r * r)) // spec: INF-1
            }
            Inflow::ProductivityIndex { k_l } => k_l * (p_r - p), // spec: INF-2
            Inflow::FixedRate { w_l } => w_l,                     // spec: INF-8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_flow_at_reservoir_pressure() {
        assert_eq!(Inflow::Vogel { w_l_max: 30.0 }.liquid_rate(170.0, 170.0), 0.0);
        assert_eq!(Inflow::ProductivityIndex { k_l: 0.5 }.liquid_rate(170.0, 170.0), 0.0);
    }

    #[test]
    fn a_fixed_rate_does_not_depend_on_the_pressure() {
        assert_eq!(Inflow::FixedRate { w_l: 12.0 }.liquid_rate(170.0, 170.0), 12.0);
    }

    #[test]
    fn vogel_gives_its_maximum_at_zero_pressure() {
        assert_eq!(Inflow::Vogel { w_l_max: 30.0 }.liquid_rate(0.0, 170.0), 30.0);
    }
}
