"""Tests for manywells.pvt."""

import pytest

from manywells.pvt import (
    P_REF,
    T_REF,
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
    dead_oil_surface_tension,
)


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
