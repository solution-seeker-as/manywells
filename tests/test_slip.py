"""Tests for manywells.slip."""

import pytest
import casadi as ca

from manywells.slip import (
    classify_flow_regime,
    SlipModel,
)
from manywells.pvt.dead_oil import dead_oil_surface_tension


def _sigma(rho_l, T):
    """Convenience: compute dead-oil surface tension for test inputs."""
    return float(dead_oil_surface_tension(rho_l, T))


def test_classify_flow_regime_sum_to_one():
    """Flow regime probabilities sum to 1."""
    v_g, v_l, alpha = 20.0, 5.0, 0.5
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 273.15 + 20)
    probs = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma)
    total = float(ca.sum1(probs).full())
    assert total == pytest.approx(1.0)


def test_classify_flow_regime_non_negative():
    """Flow regime probabilities are non-negative."""
    v_g, v_l, alpha = 20.0, 5.0, 0.5
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 273.15 + 20)
    probs = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma)
    p = probs.full().flatten()
    assert (p >= -1e-10).all()


def test_harmathy_rise_velocity_positive():
    """Harmathy bubble rise velocity is positive."""
    rho_g, rho_l = 10.0, 800.0
    sigma = _sigma(rho_l, 293.15)
    v = SlipModel.harmathy_rise_velocity(rho_g, rho_l, sigma)
    v_val = float(v.full()) if hasattr(v, "full") else float(v)
    assert v_val > 0


def test_taylor_rise_velocity_positive():
    """Taylor bubble rise velocity is positive."""
    rho_g, rho_l, D = 10.0, 800.0, 0.1
    v = SlipModel.taylor_rise_velocity(rho_g, rho_l, D)
    v_val = float(v.full()) if hasattr(v, "full") else float(v)
    assert v_val > 0


def test_identify_parameters_returns_two():
    """identify_parameters returns C_0 and v_inf."""
    model = SlipModel()
    v_g, v_l, alpha = 5.0, 2.0, 0.3
    rho_g, rho_l, D = 50.0, 700.0, 0.15
    sigma = _sigma(rho_l, 293.15)
    C_0, v_inf = model.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D)
    assert hasattr(C_0, "full") or isinstance(C_0, (int, float))
    assert hasattr(v_inf, "full") or isinstance(v_inf, (int, float))
    c0_val = float(C_0.full()) if hasattr(C_0, "full") else float(C_0)
    v_val = float(v_inf.full()) if hasattr(v_inf, "full") else float(v_inf)
    assert 1.0 <= c0_val <= 1.25
    assert v_val >= 0


def test_slip_equation_residual():
    """slip_equation v_g - (C_0*v_m + v_inf) can be evaluated."""
    model = SlipModel()
    v_g, v_l, alpha = 5.0, 2.0, 0.3
    rho_g, rho_l, D = 50.0, 700.0, 0.15
    sigma = _sigma(rho_l, 293.15)
    eq = model.slip_equation(v_g, v_l, alpha, rho_g, rho_l, sigma, D)
    val = float(eq.full())
    assert abs(val) < 100.0


def test_flow_regime_string():
    """flow_regime returns one of annular, slug-churn, bubbly."""
    model = SlipModel()
    v_g, v_l, alpha = 20.0, 2.0, 0.8
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 293.15)
    regime = model.flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma)
    assert regime in ("annular", "slug-churn", "bubbly")
