"""Tests for manywells.choke."""

import pytest
import numpy as np
import casadi as ca

from manywells.choke import (
    ChokeModel,
    BernoulliChokeModel,
    SimpsonChokeModel,
)
from manywells.discretization import PointState

A = 0.02  # Pipe cross-section (m²)


def _state(p, x_g=0.1, rho_g=50.0, rho_l=700.0, alpha=0.5, v_l=1.0, T=330.0):
    """A wellhead state at pressure p (bar) whose gas mass fraction of the flow is x_g."""
    v_g = x_g / (1 - x_g) * (1 - alpha) * rho_l * v_l / (alpha * rho_g)
    return PointState(p=p, v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l, T=T)


def test_critical_pressure_ratio():
    """Critical pressure ratio for gamma=1.307 is about 0.545."""
    cpr = ChokeModel.critical_pressure_ratio(1.307)
    expected = (2 / (1.307 + 1)) ** (1.307 / (1.307 - 1))
    assert cpr == pytest.approx(expected)
    assert 0.5 < cpr < 0.6


def test_choke_opening_linear():
    """Linear profile: opening equals position."""
    model = BernoulliChokeModel(chk_profile="linear")
    assert model.choke_opening(0.0) == 0.0
    assert model.choke_opening(1.0) == 1.0
    assert model.choke_opening(0.5) == 0.5


def test_choke_opening_sigmoid():
    """Sigmoid profile: 0->0, 1->1, 0.5->0.5."""
    model = BernoulliChokeModel(chk_profile="sigmoid")
    assert model.choke_opening(0.0) == 0.0
    assert model.choke_opening(1.0) == 1.0
    assert model.choke_opening(0.5) == pytest.approx(0.5)


def test_choke_opening_convex():
    """Convex profile at 0 and 1."""
    model = BernoulliChokeModel(chk_profile="convex")
    assert model.choke_opening(0.0) == 0.0
    assert model.choke_opening(1.0) == 1.0


def test_choke_opening_concave():
    """Concave profile at 0 and 1."""
    model = BernoulliChokeModel(chk_profile="concave")
    assert model.choke_opening(0.0) == 0.0
    assert model.choke_opening(1.0) == 1.0


def test_choke_invalid_profile():
    """Invalid choke profile raises."""
    with pytest.raises(ValueError, match="not supported"):
        BernoulliChokeModel(chk_profile="invalid")


def test_choke_negative_K_c():
    """Negative choke coefficient raises."""
    with pytest.raises(ValueError, match="Choke coefficient must be positive"):
        BernoulliChokeModel(K_c=-0.01)


def _eval_choke_mass_flow(choke_model, *args):
    """Evaluate mass_flow_rate (CasADi) with given numeric args."""
    w = choke_model.mass_flow_rate(*args)
    if hasattr(w, "full"):
        return w.full().item()
    return float(w)


def test_bernoulli_choke_mass_flow_rate():
    """Bernoulli choke: positive flow when p_in > p_out, with the mixture density and no multiplier."""
    model = BernoulliChokeModel(K_c=0.001, chk_profile="linear")
    s = _state(100.0)
    w = _eval_choke_mass_flow(model, 1.0, 20.0, s, A)
    assert w > 0
    assert w == pytest.approx(float(model.choke_equation(1.0, 100.0, 20.0, rho=s.rho_m, multiplier=1.0)), rel=1e-15)


def test_bernoulli_choke_zero_opening():
    """Zero choke opening gives zero mass flow."""
    model = BernoulliChokeModel(K_c=0.001, chk_profile="linear")
    w = _eval_choke_mass_flow(model, 0.0, 20.0, _state(100.0), A)
    assert w == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("model", [BernoulliChokeModel(K_c=0.001), SimpsonChokeModel(K_c=0.001)])
def test_no_flow_from_the_well_below_the_downstream_pressure(model):
    """CHK-11: where p_in <= p_c the choke passes nothing, and the rate's derivative is finite (zero) there."""
    p = ca.SX.sym("p")
    w = model.mass_flow_rate(1.0, 50.0, _state(p), A)
    f = ca.Function("f", [p], [w, ca.jacobian(w, p)])
    for p_in in (30.0, 50.0):
        rate, slope = (float(v) for v in f(p_in))
        assert rate == 0.0 and slope == 0.0
    rate, slope = (float(v) for v in f(120.0))
    assert rate > 0 and np.isfinite(slope)


def test_choke_models_are_frozen():
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        BernoulliChokeModel().K_c = 0.5


def test_is_choked():
    """Flow is choked when p_out <= cpr * p_in."""
    model = BernoulliChokeModel()
    cpr = model.cpr
    p_in = 100.0
    assert model.is_choked(p_in, p_in * cpr * 0.9) is True
    assert model.is_choked(p_in, p_in * cpr * 1.1) is False


def test_simpson_multiplier():
    """Simpson multiplier is positive for valid inputs."""
    x_g, rho_g, rho_l = 0.2, 10.0, 800.0
    Phi = SimpsonChokeModel.simpson_multiplier(x_g, rho_g, rho_l)
    val = Phi.full().item() if hasattr(Phi, "full") else float(Phi)
    assert val > 0


def test_simpson_choke_mass_flow_rate():
    """Simpson choke: the liquid density and Simpson's multiplier at the flow's gas mass fraction."""
    model = SimpsonChokeModel(K_c=0.001, chk_profile="linear")
    x_g, rho_g, rho_l = 0.1, 50.0, 700.0
    val = _eval_choke_mass_flow(model, 1.0, 20.0, _state(100.0, x_g, rho_g, rho_l), A)
    assert val > 0
    Phi = SimpsonChokeModel.simpson_multiplier(x_g, rho_g, rho_l)
    assert val == pytest.approx(float(model.choke_equation(1.0, 100.0, 20.0, rho=rho_l, multiplier=Phi)), rel=1e-13)
