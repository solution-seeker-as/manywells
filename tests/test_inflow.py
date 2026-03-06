"""Tests for manywells.inflow."""

import pytest

from manywells.inflow import (
    compute_gas_mass_fraction,
    InflowModel,
    ProductivityIndex,
    Vogel,
    FixedFlowRate,
)


def test_compute_gas_mass_fraction():
    """Gas mass fraction from volumetric rates and densities."""
    q_l, q_g = 0.01, 0.005  # m³/s
    rho_l, rho_g = 800.0, 1.2  # kg/m³
    f_g = compute_gas_mass_fraction(q_l, q_g, rho_l, rho_g)
    w_l = rho_l * q_l
    w_g = rho_g * q_g
    expected = w_g / (w_l + w_g)
    assert f_g == pytest.approx(expected)
    assert 0 < f_g < 1


def test_compute_gas_mass_fraction_pure_liquid():
    """Zero gas flow gives zero gas mass fraction."""
    f_g = compute_gas_mass_fraction(0.01, 0.0, 800.0, 1.2)
    assert f_g == 0.0


def test_compute_gas_mass_fraction_pure_gas():
    """Zero liquid flow gives gas mass fraction 1."""
    f_g = compute_gas_mass_fraction(0.0, 0.01, 800.0, 1.2)
    assert f_g == 1.0


def test_productivity_index_init_valid():
    """ProductivityIndex accepts valid k_l."""
    pi = ProductivityIndex(k_l=0.5)
    assert pi.k_l == 0.5


def test_productivity_index_init_invalid_k_l():
    """ProductivityIndex rejects negative k_l."""
    with pytest.raises(AssertionError, match="non-negative"):
        ProductivityIndex(k_l=-0.1)


def test_productivity_index_liquid_mass_flow_rate():
    """PI: w_l = k_l*(p_r - p)."""
    pi = ProductivityIndex(k_l=1.0)
    p, p_r = 80.0, 100.0
    w_l = pi.liquid_mass_flow_rate(p, p_r)
    assert w_l == pytest.approx(20.0)  # 1.0 * (100 - 80)


def test_vogel_init_valid():
    """Vogel accepts valid w_l_max."""
    v = Vogel(w_l_max=10.0)
    assert v.w_l_max == 10.0


def test_vogel_liquid_mass_flow_rate():
    """Vogel: w_l at p=0 is w_l_max, at p=p_r is 0."""
    v = Vogel(w_l_max=10.0)
    w_l_max = v.liquid_mass_flow_rate(0.0, 100.0)
    assert w_l_max == pytest.approx(10.0)
    w_l_at_pr = v.liquid_mass_flow_rate(100.0, 100.0)
    assert w_l_at_pr == pytest.approx(0.0)
    w_l_mid = v.liquid_mass_flow_rate(50.0, 100.0)
    assert 0 < w_l_mid < 10.0


def test_fixed_flow_rate():
    """FixedFlowRate returns constant w_l regardless of pressure."""
    fix = FixedFlowRate(w_l_const=5.0, w_g_const=1.0)
    w_l = fix.liquid_mass_flow_rate(80.0, 100.0)
    assert w_l == 5.0
    w_l2 = fix.liquid_mass_flow_rate(20.0, 50.0)
    assert w_l2 == 5.0
    assert fix.w_g_const == 1.0


def test_fixed_flow_rate_negative_rejected():
    """FixedFlowRate rejects negative liquid rate."""
    with pytest.raises(AssertionError, match="non-negative"):
        FixedFlowRate(w_l_const=-1.0, w_g_const=0.0)
