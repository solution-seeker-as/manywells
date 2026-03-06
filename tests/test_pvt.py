"""Tests for manywells.pvt."""

import pytest

from manywells.pvt import (
    P_REF,
    T_REF,
    R_UNIVERSAL,
    LiquidProperties,
    GasProperties,
    WATER,
    SEAWATER,
    NORTH_SEA_BRENT_CRUDE,
    METHANE,
    specific_gas_constant,
    gas_density,
    liquid_mix,
    water_liquid_ratio,
    api_from_density,
    density_from_api,
    molecular_weight,
    water_viscosity,
    gas_viscosity,
    liquid_mixture_viscosity,
    mixture_viscosity,
)
from manywells.pvt.dead_oil import dead_oil_viscosity, dead_oil_surface_tension


def test_reference_conditions():
    """Reference pressure and temperature match ISO 13443."""
    assert P_REF == 101_325
    assert T_REF == pytest.approx(288.15)


def test_water_properties():
    """Water has expected density and cp."""
    assert WATER.name == "water"
    assert WATER.rho == pytest.approx(999.1)
    assert WATER.cp == 4184


def test_specific_gas_constant():
    """Specific gas constant R_s = P_REF / (rho * T_REF)."""
    rho = 0.7  # kg/m³ at standard conditions
    R_s = specific_gas_constant(rho)
    expected = P_REF / (rho * T_REF)
    assert R_s == pytest.approx(expected)


def test_gas_density_ideal_gas_law():
    """Gas density follows ideal gas law p / (R_s * T)."""
    R_s = 518.3  # methane
    p = P_REF
    T = T_REF
    rho = gas_density(R_s, p, T)
    assert rho == pytest.approx(p / (R_s * T))


def test_gas_density_defaults():
    """gas_density with only R_s uses reference conditions."""
    R_s = 518.3
    rho = gas_density(R_s)
    assert rho == pytest.approx(P_REF / (R_s * T_REF))


def test_liquid_mix_mass_fraction_bounds():
    """liquid_mix rejects mass_fraction outside [0, 1]."""
    with pytest.raises(AssertionError, match="Mass fraction must be in"):
        liquid_mix(WATER, SEAWATER, -0.1)
    with pytest.raises(AssertionError, match="Mass fraction must be in"):
        liquid_mix(WATER, SEAWATER, 1.5)


def test_liquid_mix_pure():
    """liquid_mix returns same liquid for mass_fraction 0 or 1."""
    assert liquid_mix(WATER, SEAWATER, 1.0) is WATER
    assert liquid_mix(WATER, SEAWATER, 0.0) is SEAWATER


def test_liquid_mix_half():
    """liquid_mix at 0.5 gives mixture with density between the two (by volume fraction)."""
    mix = liquid_mix(WATER, NORTH_SEA_BRENT_CRUDE, 0.5)
    # Mix density is between the two densities (water is denser than Brent crude)
    assert min(WATER.rho, NORTH_SEA_BRENT_CRUDE.rho) <= mix.rho <= max(WATER.rho, NORTH_SEA_BRENT_CRUDE.rho)
    assert "50.0%" in mix.name


def test_water_liquid_ratio():
    """wlr from no-slip mixture density."""
    rho_o, rho_w = 850.0, 1000.0
    rho_l = 0.3 * rho_o + 0.7 * rho_w  # wlr = 0.7
    wlr = water_liquid_ratio(rho_l, rho_o, rho_w)
    assert wlr == pytest.approx(0.7)


def test_api_from_density():
    """API gravity from density (SG = rho/rho_water)."""
    rho = 825  # light oil
    api = api_from_density(rho)
    sg = rho / WATER.rho
    assert api == pytest.approx(141.5 / sg - 131.5)


def test_density_from_api():
    """Density from API gravity."""
    api = 40.0
    rho = density_from_api(api)
    assert rho == pytest.approx(141.5 * WATER.rho / (api + 131.5))


def test_api_density_roundtrip():
    """API and density conversions are consistent."""
    api = 35.0
    rho = density_from_api(api)
    api_back = api_from_density(rho)
    assert api_back == pytest.approx(api)


def test_dead_oil_surface_tension_positive():
    """Dead oil surface tension is positive for typical conditions."""
    rho = 850  # kg/m³
    T = 273.15 + 20  # 20 °C
    sigma = dead_oil_surface_tension(rho, T)
    assert sigma > 0


# ---- Viscosity and molecular weight tests ----


def test_molecular_weight_methane():
    """Molecular weight of methane from its specific gas constant."""
    M = molecular_weight(METHANE.R_s)
    assert M == pytest.approx(16.04, rel=0.01)


def test_molecular_weight_inverse():
    """R_UNIVERSAL / M_g recovers R_s."""
    R_s = 518.3
    M = molecular_weight(R_s)
    assert R_UNIVERSAL / M == pytest.approx(R_s)


def test_water_viscosity_at_20C():
    """Water viscosity at 20 degC is approximately 1e-3 Pa-s."""
    T = 273.15 + 20
    mu = float(water_viscosity(T))
    assert mu == pytest.approx(1.0e-3, rel=0.15)


def test_water_viscosity_decreases_with_temperature():
    """Water viscosity decreases as temperature increases."""
    mu_20 = float(water_viscosity(273.15 + 20))
    mu_80 = float(water_viscosity(273.15 + 80))
    assert mu_80 < mu_20


def test_gas_viscosity_methane_standard():
    """Gas viscosity of methane at standard conditions is ~1.1e-5 Pa-s."""
    T = T_REF
    rho_g = gas_density(METHANE.R_s)
    M_g = molecular_weight(METHANE.R_s)
    mu = float(gas_viscosity(T, rho_g, M_g))
    assert mu == pytest.approx(1.1e-5, rel=0.15)


def test_gas_viscosity_positive():
    """Gas viscosity is positive for typical well conditions."""
    T = 273.15 + 100
    rho_g = 50.0
    M_g = molecular_weight(METHANE.R_s)
    mu = float(gas_viscosity(T, rho_g, M_g))
    assert mu > 0


def test_liquid_mixture_viscosity_pure_oil():
    """With wlr=0, liquid mixture viscosity equals oil viscosity."""
    mu_o, mu_w = 5e-3, 1e-3
    assert liquid_mixture_viscosity(mu_o, mu_w, 0.0) == pytest.approx(mu_o)


def test_liquid_mixture_viscosity_pure_water():
    """With wlr=1, liquid mixture viscosity equals water viscosity."""
    mu_o, mu_w = 5e-3, 1e-3
    assert liquid_mixture_viscosity(mu_o, mu_w, 1.0) == pytest.approx(mu_w)


def test_liquid_mixture_viscosity_between():
    """Liquid mixture viscosity is between oil and water viscosity."""
    mu_o, mu_w = 5e-3, 1e-3
    mu_mix = liquid_mixture_viscosity(mu_o, mu_w, 0.5)
    assert mu_w < mu_mix < mu_o


def test_mixture_viscosity_pure_liquid():
    """With alpha=0 (no gas), mixture viscosity equals liquid viscosity."""
    mu_l, mu_g = 1e-3, 1e-5
    mu = float(mixture_viscosity(mu_l, mu_g, 0.0))
    assert mu == pytest.approx(mu_l)


def test_mixture_viscosity_pure_gas():
    """With alpha=1 (all gas), mixture viscosity equals gas viscosity."""
    mu_l, mu_g = 1e-3, 1e-5
    mu = float(mixture_viscosity(mu_l, mu_g, 1.0))
    assert mu == pytest.approx(mu_g)


def test_mixture_viscosity_between():
    """Mixture viscosity (Beggs-Brill geometric) is between gas and liquid."""
    mu_l, mu_g = 1e-3, 1e-5
    mu = float(mixture_viscosity(mu_l, mu_g, 0.5))
    assert mu_g < mu < mu_l


# ---- Dead oil viscosity tests ----


def test_dead_oil_viscosity_typical():
    """Dead oil viscosity for API 35 at 150 degF (~65.6 degC) is in a reasonable range (1-10 cP)."""
    T = 273.15 + 65.6
    mu = float(dead_oil_viscosity(35, T))
    assert 1e-3 < mu < 10e-3  # 1 to 10 cP in Pa-s


def test_dead_oil_viscosity_decreases_with_temperature():
    """Dead oil viscosity decreases as temperature increases."""
    api = 30
    mu_low_T = float(dead_oil_viscosity(api, 273.15 + 40))
    mu_high_T = float(dead_oil_viscosity(api, 273.15 + 120))
    assert mu_high_T < mu_low_T


def test_dead_oil_viscosity_decreases_with_api():
    """Lighter oils (higher API) have lower viscosity."""
    T = 273.15 + 80
    mu_heavy = float(dead_oil_viscosity(20, T))
    mu_light = float(dead_oil_viscosity(40, T))
    assert mu_light < mu_heavy


def test_dead_oil_viscosity_positive():
    """Dead oil viscosity is positive."""
    mu = float(dead_oil_viscosity(30, 273.15 + 60))
    assert mu > 0
