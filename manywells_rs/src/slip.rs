//! Slip model: v_g = C_0 * v_m + v_inf, with parameters blended over flow regimes.
//! Direct port of manywells/slip.py (SlipModel + classify_flow_regime), CasADi-free.

use crate::constants::STD_GRAVITY;
use crate::math::softmax3;
use crate::pvt::dead_oil_surface_tension;

// Profile parameters for the different flow regimes (SlipModel class attributes)
pub const C_0_ANNULAR: f64 = 1.0;
pub const C_0_SLUG: f64 = 1.175;
pub const C_0_BUBBLY: f64 = 1.2;
pub const V_INF_ANNULAR: f64 = 0.0;

/// Classify flow regime (annular, slug/churn, bubbly).
/// Returns [p_annular, p_slug, p_bubbly]. Coefficients reproduced digit-for-digit
/// from slip.py::classify_flow_regime (logistic regression fit, L2 C=0.01).
pub fn classify_flow_regime(v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, t: f64) -> [f64; 3] {
    let v_gs = alpha * v_g; // Superficial gas velocity (m/s)
    let v_ls = (1.0 - alpha) * v_l; // Superficial liquid velocity (m/s)

    // Surface tension and velocity condition for annular flow
    let sigma = dead_oil_surface_tension(rho_l, t);
    let annular_boundary = 3.1 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_g * rho_g)).powf(0.25);

    // Conditions in flow regime hierarchy
    let c_1 = (v_gs - annular_boundary).tanh(); // Eq. A-19 in Hasan, Kabir & Sayarpur (2010)
    let c_2 = ((alpha - 0.7) * 2.0).tanh(); // Multiplied by 2 to increase sensitivity
    let c_3 = (v_gs - 1.08 * v_ls).tanh(); // Eq. A-18 in Hasan, Kabir & Sayarpur (2010)
    let c_4 = ((alpha - 0.25) * 2.0).tanh(); // Multiplied by 2 to increase sensitivity

    // Output layer
    let y_1 = 3.17715258 * c_1 + 6.81938489 * c_2 + 0.30182974 * c_3 + 3.58362465 * c_4 - 3.92904391; // Annular
    let y_2 = -1.47973427 * c_1 - 4.34033317 * c_2 + 2.58200006 * c_3 + 3.49656911 * c_4 - 1.46509477; // Slug/churn
    let y_3 = -1.6974183 * c_1 - 2.47905172 * c_2 - 2.8838298 * c_3 - 7.08019376 * c_4 + 5.39413869; // Bubbly

    softmax3([y_1, y_2, y_3])
}

/// Harmathy correlation for small bubble rise velocity (m/s)
pub fn harmathy_rise_velocity(rho_g: f64, rho_l: f64, t: f64) -> f64 {
    let s = dead_oil_surface_tension(rho_l, t); // Liquid surface tension (J/m²)
    1.53 * (STD_GRAVITY * s * (rho_l - rho_g) / (rho_l * rho_l)).powf(0.25)
}

/// Correlation for Taylor-bubble rise velocity (m/s)
pub fn taylor_rise_velocity(rho_g: f64, rho_l: f64, d: f64) -> f64 {
    0.35 * (STD_GRAVITY * d * (1.0 - rho_g / rho_l)).sqrt()
}

/// Compute slip model parameters (C_0, v_inf) -- SlipModel.identify_parameters
pub fn identify_parameters(v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, t: f64, d: f64) -> (f64, f64) {
    let [p_annular, p_slug, p_bubbly] = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, t);

    let c_0 = p_annular * C_0_ANNULAR + p_slug * C_0_SLUG + p_bubbly * C_0_BUBBLY;

    let v_inf_slug = taylor_rise_velocity(rho_g, rho_l, d);
    let v_inf_bubbly = harmathy_rise_velocity(rho_g, rho_l, t);
    let v_inf = p_annular * V_INF_ANNULAR + p_slug * v_inf_slug + p_bubbly * v_inf_bubbly;

    (c_0, v_inf)
}

/// Textual description of the most probable flow regime -- SlipModel.flow_regime
/// (same tie-breaking order as the Python implementation).
pub fn flow_regime_name(v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, t: f64) -> &'static str {
    let [p_annular, p_slug, p_bubbly] = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, t);
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
    //! Ported from tests/test_slip.py on the develop branch. The develop API takes
    //! (sigma, cos_incl); this port takes (T, D) and computes sigma internally, so
    //! the calls are adapted accordingly.
    use super::*;

    #[test]
    fn classify_flow_regime_sums_to_one() {
        let probs = classify_flow_regime(20.0, 5.0, 0.5, 1.0, 900.0, 273.15 + 20.0);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn classify_flow_regime_non_negative() {
        let probs = classify_flow_regime(20.0, 5.0, 0.5, 1.0, 900.0, 273.15 + 20.0);
        assert!(probs.iter().all(|&p| p >= -1e-10));
    }

    #[test]
    fn harmathy_rise_velocity_positive() {
        assert!(harmathy_rise_velocity(10.0, 800.0, 293.15) > 0.0);
    }

    #[test]
    fn taylor_rise_velocity_positive() {
        assert!(taylor_rise_velocity(10.0, 800.0, 0.1) > 0.0);
    }

    #[test]
    fn identify_parameters_in_range() {
        // C_0 is a blend of the regime profile parameters (1.0, 1.175, 1.2) and
        // v_inf is a non-negative rise velocity.
        let (c_0, v_inf) = identify_parameters(5.0, 2.0, 0.3, 50.0, 700.0, 293.15, 0.15);
        assert!((1.0..=1.25).contains(&c_0));
        assert!(v_inf >= 0.0);
    }

    #[test]
    fn flow_regime_name_is_valid() {
        let name = flow_regime_name(20.0, 2.0, 0.8, 1.0, 900.0, 293.15);
        assert!(matches!(name, "annular" | "slug-churn" | "bubbly"));
    }
}
