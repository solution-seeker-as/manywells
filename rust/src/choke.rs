//! Choke models. Port of manywells/choke.py (the parts simulator uses:
//! choke opening profiles, critical pressure ratio, K_c/cpr parameters).
//! The choke *equation* itself lives in the simulator's right-boundary residual.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Default K_c: 10% of the area of a 6.11-inch pipe (choke.py ChokeModel.K_c default)
pub fn default_k_c() -> f64 {
    0.1 * std::f64::consts::PI * (0.1554 / 2.0) * (0.1554 / 2.0)
}

/// Critical pressure ratio: cpr = (2 / (gamma + 1)) ^ (gamma / (gamma - 1)),
/// gamma = 1.307 (methane at 20 degC) -- choke.py ChokeModel.critical_pressure_ratio
pub fn critical_pressure_ratio() -> f64 {
    let gamma: f64 = 1.307;
    (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChkProfile {
    Linear,
    Sigmoid,
    Convex,
    Concave,
}

impl ChkProfile {
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "linear" => Ok(Self::Linear),
            "sigmoid" => Ok(Self::Sigmoid),
            "convex" => Ok(Self::Convex),
            "concave" => Ok(Self::Concave),
            _ => Err(PyValueError::new_err(format!("Choke profile {s} is not supported"))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Sigmoid => "sigmoid",
            Self::Convex => "convex",
            Self::Concave => "concave",
        }
    }

    /// Relative choke opening sigma(u) -- choke.py ChokeModel.choke_opening
    pub fn choke_opening(&self, u: f64) -> f64 {
        match self {
            Self::Linear => u,
            Self::Sigmoid => {
                let b = 1.5;
                u.powf(b) / (u.powf(b) + (1.0 - u).powf(b))
            }
            Self::Convex => {
                let b = 0.25; // Number in [0, 1]
                b * u + (1.0 - b) * u * u
            }
            Self::Concave => {
                // Also known as quick open valve characteristics
                let b = 0.75; // Number in (0, 1]
                u.powf(b)
            }
        }
    }
}

/// Plain-Rust view of a choke model for the hot loop.
#[derive(Clone, Copy, Debug)]
pub struct ChokeSpec {
    pub k_c: f64,
    pub cpr: f64,
    pub profile: ChkProfile,
    pub simpson: bool, // false = Bernoulli (rho_m, Phi=1), true = Simpson (rho_l, Phi)
}

impl ChokeSpec {
    pub fn choke_opening(&self, u: f64) -> f64 {
        self.profile.choke_opening(u)
    }
}

fn validate_k_c(k_c: f64) -> PyResult<()> {
    if k_c <= 0.0 {
        return Err(PyValueError::new_err("Choke coefficient must be positive"));
    }
    Ok(())
}

/// Bernoulli choke model: rho = rho_m, Phi = 1
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct BernoulliChokeModel {
    #[pyo3(get, set)]
    pub K_c: f64,
    #[pyo3(get)]
    pub cpr: f64,
    pub profile: ChkProfile,
}

impl BernoulliChokeModel {
    /// Rust-side constructor for the WellProperties default choke (K_c = 0.1 * A)
    pub fn new_default(k_c: f64) -> Self {
        Self { K_c: k_c, cpr: critical_pressure_ratio(), profile: ChkProfile::Linear }
    }
}

#[pymethods]
impl BernoulliChokeModel {
    #[new]
    #[pyo3(signature = (K_c=None, cpr=None, chk_profile="linear"))]
    fn new(K_c: Option<f64>, cpr: Option<f64>, chk_profile: &str) -> PyResult<Self> {
        let _ = cpr; // Accepted for signature parity; recomputed like the Python __post_init__
        let k_c = K_c.unwrap_or_else(default_k_c);
        validate_k_c(k_c)?;
        Ok(Self { K_c: k_c, cpr: critical_pressure_ratio(), profile: ChkProfile::from_str(chk_profile)? })
    }

    #[getter]
    fn chk_profile(&self) -> &'static str {
        self.profile.as_str()
    }

    fn choke_opening(&self, u: f64) -> f64 {
        self.profile.choke_opening(u)
    }

    /// True if flow is choked (p_out below the critical pressure)
    fn is_choked(&self, p_in: f64, p_out: f64) -> bool {
        p_out <= self.cpr * p_in
    }
}

/// Simpson choke model: rho = rho_l, Phi = Simpson et al. (1983) multiplier
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SimpsonChokeModel {
    #[pyo3(get, set)]
    pub K_c: f64,
    #[pyo3(get)]
    pub cpr: f64,
    pub profile: ChkProfile,
}

#[pymethods]
impl SimpsonChokeModel {
    #[new]
    #[pyo3(signature = (K_c=None, cpr=None, chk_profile="linear"))]
    fn new(K_c: Option<f64>, cpr: Option<f64>, chk_profile: &str) -> PyResult<Self> {
        let _ = cpr;
        let k_c = K_c.unwrap_or_else(default_k_c);
        validate_k_c(k_c)?;
        Ok(Self { K_c: k_c, cpr: critical_pressure_ratio(), profile: ChkProfile::from_str(chk_profile)? })
    }

    #[getter]
    fn chk_profile(&self) -> &'static str {
        self.profile.as_str()
    }

    fn choke_opening(&self, u: f64) -> f64 {
        self.profile.choke_opening(u)
    }

    fn is_choked(&self, p_in: f64, p_out: f64) -> bool {
        p_out <= self.cpr * p_in
    }
}

/// Extract a ChokeSpec from either Python-facing choke class.
pub fn extract_choke(obj: &Bound<'_, PyAny>) -> PyResult<ChokeSpec> {
    if let Ok(c) = obj.cast::<SimpsonChokeModel>() {
        let c = c.borrow();
        return Ok(ChokeSpec { k_c: c.K_c, cpr: c.cpr, profile: c.profile, simpson: true });
    }
    if let Ok(c) = obj.cast::<BernoulliChokeModel>() {
        let c = c.borrow();
        return Ok(ChokeSpec { k_c: c.K_c, cpr: c.cpr, profile: c.profile, simpson: false });
    }
    Err(PyValueError::new_err(
        "choke must be a manywells_rs.SimpsonChokeModel or manywells_rs.BernoulliChokeModel instance",
    ))
}

#[cfg(test)]
mod tests {
    //! Ported from tests/test_choke.py on the develop branch. Tests for the choke
    //! *equation* / Simpson multiplier are omitted here: in manywells_rs that logic
    //! lives in the simulator's right-boundary residual, not in this module.
    use super::*;

    #[test]
    fn critical_pressure_ratio_methane() {
        // Critical pressure ratio for gamma = 1.307 is about 0.545.
        let cpr = critical_pressure_ratio();
        let gamma = 1.307_f64;
        let expected = (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0));
        assert!((cpr - expected).abs() < 1e-12);
        assert!(cpr > 0.5 && cpr < 0.6);
    }

    #[test]
    fn choke_opening_linear() {
        let p = ChkProfile::Linear;
        assert_eq!(p.choke_opening(0.0), 0.0);
        assert_eq!(p.choke_opening(1.0), 1.0);
        assert_eq!(p.choke_opening(0.5), 0.5);
    }

    #[test]
    fn choke_opening_sigmoid() {
        let p = ChkProfile::Sigmoid;
        assert_eq!(p.choke_opening(0.0), 0.0);
        assert_eq!(p.choke_opening(1.0), 1.0);
        assert!((p.choke_opening(0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn choke_opening_convex_endpoints() {
        let p = ChkProfile::Convex;
        assert_eq!(p.choke_opening(0.0), 0.0);
        assert_eq!(p.choke_opening(1.0), 1.0);
    }

    #[test]
    fn choke_opening_concave_endpoints() {
        let p = ChkProfile::Concave;
        assert_eq!(p.choke_opening(0.0), 0.0);
        assert_eq!(p.choke_opening(1.0), 1.0);
    }

    #[test]
    fn invalid_profile_rejected() {
        assert!(ChkProfile::from_str("invalid").is_err());
    }

    #[test]
    fn negative_k_c_rejected() {
        assert!(BernoulliChokeModel::new(Some(-0.01), None, "linear").is_err());
        assert!(SimpsonChokeModel::new(Some(-0.01), None, "linear").is_err());
    }

    #[test]
    fn default_k_c_is_ten_percent_of_default_pipe_area() {
        let a = std::f64::consts::PI * (0.1554 / 2.0) * (0.1554 / 2.0);
        assert!((default_k_c() - 0.1 * a).abs() < 1e-15);
    }

    #[test]
    fn is_choked_threshold() {
        // Flow is choked when p_out <= cpr * p_in.
        let model = BernoulliChokeModel::new(None, None, "linear").unwrap();
        let cpr = model.cpr;
        let p_in = 100.0;
        assert!(model.is_choked(p_in, p_in * cpr * 0.9));
        assert!(!model.is_choked(p_in, p_in * cpr * 1.1));
    }
}
