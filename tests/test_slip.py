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
    probs = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl=1.0)
    total = ca.sum1(probs).full().item()
    assert total == pytest.approx(1.0)


def test_classify_flow_regime_non_negative():
    """Flow regime probabilities are non-negative."""
    v_g, v_l, alpha = 20.0, 5.0, 0.5
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 273.15 + 20)
    probs = classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl=1.0)
    p = probs.full().flatten()
    assert (p >= -1e-10).all()


def test_harmathy_rise_velocity_positive():
    """Harmathy bubble rise velocity is positive."""
    rho_g, rho_l = 10.0, 800.0
    sigma = _sigma(rho_l, 293.15)
    v = SlipModel.harmathy_rise_velocity(rho_g, rho_l, sigma)
    v_val = v.full().item() if hasattr(v, "full") else float(v)
    assert v_val > 0


def test_taylor_rise_velocity_positive():
    """Taylor bubble rise velocity is positive."""
    rho_g, rho_l, D = 10.0, 800.0, 0.1
    v = SlipModel.taylor_rise_velocity(rho_g, rho_l, D)
    v_val = v.full().item() if hasattr(v, "full") else float(v)
    assert v_val > 0


def test_identify_parameters_returns_two():
    """identify_parameters returns C_0 and v_inf."""
    model = SlipModel()
    v_g, v_l, alpha = 5.0, 2.0, 0.3
    rho_g, rho_l, D = 50.0, 700.0, 0.15
    sigma = _sigma(rho_l, 293.15)
    cos_incl = 1.0
    C_0, v_inf = model.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl)
    assert hasattr(C_0, "full") or isinstance(C_0, (int, float))
    assert hasattr(v_inf, "full") or isinstance(v_inf, (int, float))
    c0_val = C_0.full().item() if hasattr(C_0, "full") else float(C_0)
    v_val = v_inf.full().item() if hasattr(v_inf, "full") else float(v_inf)
    assert 1.0 <= c0_val <= 1.25
    assert v_val >= 0


def test_slip_equation_residual():
    """slip_equation v_g - (C_0*v_m + v_inf) can be evaluated."""
    model = SlipModel()
    v_g, v_l, alpha = 5.0, 2.0, 0.3
    rho_g, rho_l, D = 50.0, 700.0, 0.15
    sigma = _sigma(rho_l, 293.15)
    cos_incl = 1.0
    eq = model.slip_equation(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl)
    val = eq.full().item()
    assert abs(val) < 100.0


def test_flow_regime_string():
    """flow_regime returns one of annular, slug-churn, bubbly."""
    model = SlipModel()
    v_g, v_l, alpha = 20.0, 2.0, 0.8
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 293.15)
    regime = model.flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl=1.0)
    assert regime in ("annular", "slug-churn", "bubbly")


# ---------------------------------------------------------------------------
# Inclination (SLIP-10, SLIP-11) and the slip parameters as fields
# ---------------------------------------------------------------------------

def _params(cos_incl, **state):
    s = dict(v_g=6.0, v_l=2.0, alpha=0.4, rho_g=80.0, rho_l=800.0) | state
    sigma = _sigma(s['rho_l'], 330.0)
    C_0, v_inf = SlipModel().identify_parameters(**s, sigma=sigma, D=0.15, cos_incl=cos_incl)
    return float(C_0), float(v_inf)


def test_deviation_factor_is_one_in_a_vertical_cell():
    """SLIP-10 is exactly 1 at cos_incl = 1, so the vertical slip law is v1.0.0's (no 1e-9 guard)."""
    rho_g, rho_l, D = 80.0, 800.0, 0.15
    s = dict(v_g=6.0, v_l=2.0, alpha=0.4, rho_g=rho_g, rho_l=rho_l)
    sigma = _sigma(rho_l, 330.0)
    probs = [float(p) for p in ca.DM(classify_flow_regime(**s, sigma=sigma, cos_incl=1.0)).full().ravel()]
    v_inf_T = float(SlipModel.taylor_rise_velocity(rho_g, rho_l, D))
    v_inf_b = float(SlipModel.harmathy_rise_velocity(rho_g, rho_l, sigma))
    assert _params(1.0)[1] == probs[1] * v_inf_T + probs[2] * v_inf_b


def test_taylor_rise_velocity_vanishes_in_a_horizontal_cell():
    """In a horizontal cell (cos_incl = 0) the Taylor bubble does not rise: only bubbly drift remains."""
    s = dict(v_g=6.0, v_l=2.0, alpha=0.4, rho_g=80.0, rho_l=800.0)
    sigma = _sigma(800.0, 330.0)
    probs = [float(p) for p in ca.DM(classify_flow_regime(**s, sigma=sigma, cos_incl=0.0)).full().ravel()]
    v_inf_b = float(SlipModel.harmathy_rise_velocity(80.0, 800.0, sigma))
    assert _params(0.0)[1] == pytest.approx(probs[2] * v_inf_b, rel=1e-12)


def test_inclined_cell_moves_bubbly_flow_to_slug_flow():
    """SLIP-11: the bubbly-slug threshold falls with inclination, 0.25 cos(theta)."""
    s = dict(v_g=3.0, v_l=1.5, alpha=0.2, rho_g=100.0, rho_l=750.0, sigma=0.025)
    vertical = [float(p) for p in ca.DM(classify_flow_regime(**s, cos_incl=1.0)).full().ravel()]
    inclined = [float(p) for p in ca.DM(classify_flow_regime(**s, cos_incl=0.5)).full().ravel()]
    assert inclined[1] > vertical[1] and inclined[2] < vertical[2]


def test_slip_parameters_are_fields():
    """plans/improvements.md 2.7: the regime constants are fields, with v1.0.0's values as defaults."""
    import dataclasses
    model = SlipModel(C_0_slug=1.2)
    assert model.C_0_slug == 1.2 and SlipModel().C_0_slug == 1.175
    assert [f.name for f in dataclasses.fields(SlipModel)] == ['C_0_annular', 'C_0_slug', 'C_0_bubbly', 'v_inf_annular']
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.C_0_bubbly = 1.0
