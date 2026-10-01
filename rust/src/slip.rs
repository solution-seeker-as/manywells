// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Drift-flux closure and flow-regime classification (specs/model/slip.md): v_g = C_0 v_m + v_inf, with C_0 and
//! v_inf blended over three regimes (annular, slug/churn, bubbly) by a classifier. So far the vertical pipe of the
//! v1.0.0 configuration (SLIP-1 to SLIP-8).

use crate::smoothing::softmax3;
use crate::units::STD_GRAVITY;

// Profile parameter and drift velocity of each regime
pub const C_0_ANNULAR: f64 = 1.0;
pub const C_0_SLUG: f64 = 1.175;
pub const C_0_BUBBLY: f64 = 1.2;
pub const V_INF_ANNULAR: f64 = 0.0;

/// Bubble rise velocity (m/s), Harmathy, at gas-liquid surface tension sigma (J/m²)
pub fn harmathy_rise_velocity(rho_g: f64, rho_l: f64, sigma: f64) -> f64 {  // spec: SLIP-4
    1.53 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_l * rho_l)).powf(0.25)
}

/// Taylor-bubble rise velocity (m/s) in a pipe of inner diameter d (m)
pub fn taylor_rise_velocity(rho_g: f64, rho_l: f64, d: f64) -> f64 {  // spec: SLIP-5
    0.35 * (STD_GRAVITY * d * (1.0 - rho_g / rho_l)).sqrt()
}

/// The terms of the slip law at a point that do not depend on the void fraction: the classifier's two velocity
/// features, from the superficial velocities, and the rise velocities. With the phase mass fluxes fixed, as in the
/// march, the superficial velocities do not depend on alpha either, so these are computed once per pressure.
#[derive(Clone, Copy, Debug)]
pub struct SlipTerms {
    c_1: f64,
    c_3: f64,
    v_inf_slug: f64,
    v_inf_bubbly: f64,
}

impl SlipTerms {
    /// v_gs and v_ls are the superficial gas and liquid velocities (m/s), sigma the surface tension (J/m²) and d
    /// the pipe's inner diameter (m)
    pub fn new(v_gs: f64, v_ls: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64) -> Self {
        let annular_boundary = 3.1 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_g * rho_g)).powf(0.25);
        Self {
            c_1: (v_gs - annular_boundary).tanh(), // spec: SLIP-6
            c_3: (v_gs - 1.08 * v_ls).tanh(),      // spec: SLIP-6
            v_inf_slug: taylor_rise_velocity(rho_g, rho_l, d),
            v_inf_bubbly: harmathy_rise_velocity(rho_g, rho_l, sigma),
        }
    }

    /// Regime probabilities [p_annular, p_slug, p_bubbly] at void fraction alpha
    pub fn probabilities(&self, alpha: f64) -> [f64; 3] {  // spec: SLIP-6, SLIP-7
        let c_2 = ((alpha - 0.7) * 2.0).tanh();
        let c_4 = ((alpha - 0.25) * 2.0).tanh();
        let (c_1, c_3) = (self.c_1, self.c_3);
        let y_1 = 3.17715258 * c_1 + 6.81938489 * c_2 + 0.30182974 * c_3 + 3.58362465 * c_4 - 3.92904391;
        let y_2 = -1.47973427 * c_1 - 4.34033317 * c_2 + 2.58200006 * c_3 + 3.49656911 * c_4 - 1.46509477;
        let y_3 = -1.6974183 * c_1 - 2.47905172 * c_2 - 2.8838298 * c_3 - 7.08019376 * c_4 + 5.39413869;
        softmax3([y_1, y_2, y_3])
    }

    /// The slip parameters (C_0, v_inf) at void fraction alpha
    pub fn parameters(&self, alpha: f64) -> (f64, f64) {  // spec: SLIP-2, SLIP-3
        let [p_annular, p_slug, p_bubbly] = self.probabilities(alpha);
        let c_0 = p_annular * C_0_ANNULAR + p_slug * C_0_SLUG + p_bubbly * C_0_BUBBLY;
        let v_inf = p_annular * V_INF_ANNULAR + p_slug * self.v_inf_slug + p_bubbly * self.v_inf_bubbly;
        (c_0, v_inf)
    }
}

/// Regime probabilities [p_annular, p_slug, p_bubbly] at a point
pub fn classify(v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64) -> [f64; 3] {
    SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d).probabilities(alpha)
}

/// The slip parameters (C_0, v_inf) at a point
pub fn identify_parameters(v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64) -> (f64, f64) {
    SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d).parameters(alpha)
}

/// The most probable regime; bubbly takes ties
pub fn regime_label(probs: [f64; 3]) -> &'static str {  // spec: SLIP-8
    let [p_annular, p_slug, p_bubbly] = probs;
    if p_annular > p_slug && p_annular > p_bubbly {
        "annular"
    } else if p_slug > p_annular && p_slug > p_bubbly {
        "slug-churn"
    } else {
        "bubbly"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 0.02;

    #[test]
    fn probabilities_sum_to_one() {
        let p = classify(20.0, 5.0, 0.5, 1.0, 900.0, SIGMA, 0.15);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn parameters_are_blends_of_the_regime_values() {
        for alpha in [0.0, 0.1, 0.3, 0.7, 0.95, 1.0] {
            let (c_0, v_inf) = identify_parameters(5.0, 2.0, alpha, 50.0, 700.0, SIGMA, 0.15);
            assert!((C_0_ANNULAR - 1e-12..=C_0_BUBBLY + 1e-12).contains(&c_0));
            assert!(v_inf >= 0.0);
        }
    }

    #[test]
    fn label_is_a_regime() {
        let label = regime_label(classify(20.0, 2.0, 0.8, 1.0, 900.0, SIGMA, 0.15));
        assert!(matches!(label, "annular" | "slug-churn" | "bubbly"));
        assert_eq!(regime_label([1.0 / 3.0; 3]), "bubbly");
    }
}
