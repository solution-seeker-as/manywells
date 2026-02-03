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
    """ProductivityIndex accepts valid k_l and f_g."""
    pi = ProductivityIndex(k_l=0.5, f_g=0.2)
    assert pi.k_l == 0.5
    assert pi.f_g == 0.2


def test_productivity_index_init_invalid_k_l():
    """ProductivityIndex rejects negative k_l."""
    with pytest.raises(AssertionError, match="non-negative"):
        ProductivityIndex(k_l=-0.1, f_g=0.2)


def test_productivity_index_init_invalid_f_g():
    """ProductivityIndex requires f_g in (0, 1)."""
    with pytest.raises(AssertionError, match="Gas mass fraction"):
        ProductivityIndex(k_l=0.5, f_g=0.0)
    with pytest.raises(AssertionError, match="Gas mass fraction"):
        ProductivityIndex(k_l=0.5, f_g=1.0)


def test_productivity_index_mass_flow_rates():
    """PI: w_l = k_l*(p_r - p), w_g = (f_g/(1-f_g))*w_l."""
    pi = ProductivityIndex(k_l=1.0, f_g=0.25)
    p, p_r = 80.0, 100.0
    w_l, w_g = pi.mass_flow_rates(p, p_r)
    assert w_l == pytest.approx(20.0)  # 1.0 * (100 - 80)
    assert w_g == pytest.approx(20.0 * 0.25 / 0.75)


def test_vogel_init_valid():
    """Vogel accepts valid w_l_max and f_g."""
    v = Vogel(w_l_max=10.0, f_g=0.2)
    assert v.w_l_max == 10.0
    assert v.f_g == 0.2


def test_vogel_mass_flow_rates():
    """Vogel: w_l at p=0 is w_l_max, at p=p_r is 0."""
    v = Vogel(w_l_max=10.0, f_g=0.2)
    _, w_g_zero = v.mass_flow_rates(0.0, 100.0)
    w_l_max, _ = v.mass_flow_rates(0.0, 100.0)
    assert w_l_max == pytest.approx(10.0)
    w_l_at_pr, _ = v.mass_flow_rates(100.0, 100.0)
    assert w_l_at_pr == pytest.approx(0.0)
    # Gas proportional to liquid
    w_l, w_g = v.mass_flow_rates(50.0, 100.0)
    assert w_g == pytest.approx((v.f_g / (1 - v.f_g)) * w_l)


def test_fixed_flow_rate():
    """FixedFlowRate returns constant w_l and w_g."""
    fix = FixedFlowRate(w_l_const=5.0, w_g_const=1.0)
    w_l, w_g = fix.mass_flow_rates(80.0, 100.0)
    assert w_l == 5.0
    assert w_g == 1.0
    w_l2, w_g2 = fix.mass_flow_rates(20.0, 50.0)
    assert w_l2 == 5.0 and w_g2 == 1.0


def test_fixed_flow_rate_negative_rejected():
    """FixedFlowRate rejects negative liquid rate."""
    with pytest.raises(AssertionError, match="non-negative"):
        FixedFlowRate(w_l_const=-1.0, w_g_const=0.0)
