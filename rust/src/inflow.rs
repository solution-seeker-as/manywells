//! Inflow performance relationships. Port of manywells/inflow.py
//! Only ProductivityIndex and Vogel is ported, since FixedFlowRate 
//! is not in use in simulator.py

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Plain-Rust view of an inflow model, extracted from the Python-facing
/// classes below so the hot loop never touches Python objects.
#[derive(Clone, Copy, Debug)]
pub enum InflowSpec {
    ProductivityIndex { k_l: f64, f_g: f64 },
    Vogel { w_l_max: f64, f_g: f64 },
}

impl InflowSpec {
    /// Liquid and gas mass rates (w_l, w_g) given bottomhole and reservoir pressure (bar)
    pub fn mass_flow_rates(&self, p: f64, p_r: f64) -> (f64, f64) {
        match *self {
            InflowSpec::ProductivityIndex { k_l, f_g } => {
                let w_l = k_l * (p_r - p);
                let w_g = (f_g / (1.0 - f_g)) * w_l;
                (w_l, w_g)
            }
            InflowSpec::Vogel { w_l_max, f_g } => {
                let r = p / p_r;
                let w_l = w_l_max * (1.0 - 0.2 * r - 0.8 * r * r);
                let w_g = (f_g / (1.0 - f_g)) * w_l;
                (w_l, w_g)
            }
        }
    }
}

/// Productivity index (PI) model: w_l = k_l * (p_r - p)
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct ProductivityIndex {
    #[pyo3(get, set)]
    pub k_l: f64,
    #[pyo3(get, set)]
    pub f_g: f64,
}

#[pymethods]
impl ProductivityIndex {
    #[new]
    #[pyo3(signature = (k_l, f_g))]
    fn new(k_l: f64, f_g: f64) -> PyResult<Self> {
        if k_l < 0.0 {
            return Err(PyValueError::new_err("Liquid productivity index must be non-negative"));
        }
        if !(0.0 < f_g && f_g < 1.0) {
            return Err(PyValueError::new_err("Gas mass fraction must be in (0, 1)"));
        }
        Ok(Self { k_l, f_g })
    }

    fn mass_flow_rates(&self, p: f64, p_r: f64) -> (f64, f64) {
        InflowSpec::ProductivityIndex { k_l: self.k_l, f_g: self.f_g }.mass_flow_rates(p, p_r)
    }
}

/// Vogel's inflow performance relationship:
/// w_l = w_l_max * (1 - 0.2 * (p / p_r) - 0.8 * (p / p_r)²)
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Vogel {
    #[pyo3(get, set)]
    pub w_l_max: f64,
    #[pyo3(get, set)]
    pub f_g: f64,
}

#[pymethods]
impl Vogel {
    #[new]
    #[pyo3(signature = (w_l_max, f_g))]
    fn new(w_l_max: f64, f_g: f64) -> PyResult<Self> {
        if w_l_max < 0.0 {
            return Err(PyValueError::new_err("Maximum liquid mass flow rate must be non-negative"));
        }
        if !(0.0 < f_g && f_g < 1.0) {
            return Err(PyValueError::new_err("Gas mass fraction must be in (0, 1)"));
        }
        Ok(Self { w_l_max, f_g })
    }

    fn mass_flow_rates(&self, p: f64, p_r: f64) -> (f64, f64) {
        InflowSpec::Vogel { w_l_max: self.w_l_max, f_g: self.f_g }.mass_flow_rates(p, p_r)
    }
}

/// Extract an InflowSpec from either Python-facing inflow class.
pub fn extract_inflow(obj: &Bound<'_, PyAny>) -> PyResult<InflowSpec> {
    if let Ok(v) = obj.cast::<Vogel>() {
        let v = v.borrow();
        return Ok(InflowSpec::Vogel { w_l_max: v.w_l_max, f_g: v.f_g });
    }
    if let Ok(pi) = obj.cast::<ProductivityIndex>() {
        let pi = pi.borrow();
        return Ok(InflowSpec::ProductivityIndex { k_l: pi.k_l, f_g: pi.f_g });
    }
    Err(PyValueError::new_err(
        "inflow must be a manywells_rs.Vogel or manywells_rs.ProductivityIndex instance",
    ))
}

#[cfg(test)]
mod tests {
    //! Ported from tests/test_inflow.py on the develop branch. FixedFlowRate is not
    //! ported to manywells_rs, so its tests are omitted. The develop API exposes
    //! liquid_mass_flow_rate(p, p_r); this port returns (w_l, w_g) from
    //! mass_flow_rates(p, p_r), so the liquid rate is the first tuple element.
    use super::*;

    #[test]
    fn productivity_index_init_valid() {
        let pi = ProductivityIndex::new(0.5, 0.1379).unwrap();
        assert_eq!(pi.k_l, 0.5);
    }

    #[test]
    fn productivity_index_rejects_negative_k_l() {
        assert!(ProductivityIndex::new(-0.1, 0.1379).is_err());
    }

    #[test]
    fn productivity_index_rejects_invalid_f_g() {
        assert!(ProductivityIndex::new(0.5, 0.0).is_err());
        assert!(ProductivityIndex::new(0.5, 1.0).is_err());
    }

    #[test]
    fn productivity_index_liquid_mass_flow_rate() {
        // w_l = k_l * (p_r - p) = 1.0 * (100 - 80) = 20.
        let pi = ProductivityIndex::new(1.0, 0.1).unwrap();
        let (w_l, _w_g) = pi.mass_flow_rates(80.0, 100.0);
        assert!((w_l - 20.0).abs() < 1e-12);
    }

    #[test]
    fn productivity_index_gas_fraction() {
        // w_g = f_g / (1 - f_g) * w_l.
        let f_g = 0.2;
        let pi = ProductivityIndex::new(1.0, f_g).unwrap();
        let (w_l, w_g) = pi.mass_flow_rates(80.0, 100.0);
        assert!((w_g - (f_g / (1.0 - f_g)) * w_l).abs() < 1e-12);
    }

    #[test]
    fn vogel_init_valid() {
        let v = Vogel::new(10.0, 0.1).unwrap();
        assert_eq!(v.w_l_max, 10.0);
    }

    #[test]
    fn vogel_rejects_negative_w_l_max() {
        assert!(Vogel::new(-1.0, 0.1).is_err());
    }

    #[test]
    fn vogel_liquid_mass_flow_rate() {
        // w_l is w_l_max at p = 0 and 0 at p = p_r, positive in between.
        let v = Vogel::new(10.0, 0.1).unwrap();
        let (w_l_max, _) = v.mass_flow_rates(0.0, 100.0);
        assert!((w_l_max - 10.0).abs() < 1e-12);
        let (w_l_at_pr, _) = v.mass_flow_rates(100.0, 100.0);
        assert!(w_l_at_pr.abs() < 1e-12);
        let (w_l_mid, _) = v.mass_flow_rates(50.0, 100.0);
        assert!(w_l_mid > 0.0 && w_l_mid < 10.0);
    }
}
