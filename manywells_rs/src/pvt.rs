//! PVT correlations needed by the slip model.
//! Port of the required subset of manywells/pvt.py.

/// Density of pure water at standard reference conditions (kg/m³) -- pvt.WATER.rho
pub const WATER_RHO: f64 = 999.1;

/// Compute API gravity from density:
///     API = 141.5 / SG - 131.5,  SG = rho / rho_water
pub fn api_from_density(rho: f64) -> f64 {
    let sg = rho / WATER_RHO;
    141.5 / sg - 131.5
}

/// Correlation for dead oil surface tension (J/m²) from
/// "Estimation of gas-oil surface tension" by Abdul-Majeed & Al-Soof (2000).
/// The correlation gives dyn/cm; 1 dyn/cm = 0.001 J/m².
pub fn dead_oil_surface_tension(rho: f64, t: f64) -> f64 {
    let cf = 0.001; // Unit conversion factor (1 dyn/cm = 0.001 J/m²)
    let t_deg_c = t - 273.15; // From kelvin to degC
    let api = api_from_density(rho);
    cf * (1.11591 - 0.00305 * t_deg_c) * (38.085 - 0.259 * api)
}

#[cfg(test)]
mod tests {
    //! Ported from the subset of tests/test_pvt.py on the develop branch that
    //! covers the correlations present in this port (API gravity, dead-oil surface
    //! tension, water reference density). Black-oil / gas / viscosity / Z-factor
    //! correlations are not part of manywells_rs, so those tests are omitted.
    use super::*;

    #[test]
    fn water_reference_density() {
        assert!((WATER_RHO - 999.1).abs() < 1e-9);
    }

    #[test]
    fn api_from_density_formula() {
        // API gravity from density (SG = rho / rho_water).
        let rho = 825.0; // light oil
        let api = api_from_density(rho);
        let sg = rho / WATER_RHO;
        assert!((api - (141.5 / sg - 131.5)).abs() < 1e-9);
    }

    #[test]
    fn dead_oil_surface_tension_positive() {
        // Positive for typical conditions (850 kg/m³, 20 °C).
        assert!(dead_oil_surface_tension(850.0, 273.15 + 20.0) > 0.0);
    }
}

