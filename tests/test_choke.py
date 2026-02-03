"""Tests for manywells.choke."""

import pytest
import numpy as np
import casadi as ca

from manywells.choke import (
    ChokeModel,
    BernoulliChokeModel,
    SimpsonChokeModel,
)


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
    with pytest.raises(AssertionError, match="not supported"):
        BernoulliChokeModel(chk_profile="invalid")


def test_choke_negative_K_c():
    """Negative choke coefficient raises."""
    with pytest.raises(AssertionError, match="Choke coefficient must be positive"):
        BernoulliChokeModel(K_c=-0.01)


def _eval_choke_mass_flow(choke_model, *args):
    """Evaluate mass_flow_rate (CasADi) with given numeric args."""
    w = choke_model.mass_flow_rate(*args)
    if hasattr(w, "full"):
        return float(w.full())
    return float(w)


def test_bernoulli_choke_mass_flow_rate():
    """Bernoulli choke: positive flow when p_in > p_out."""
    model = BernoulliChokeModel(K_c=0.001, chk_profile="linear")
    u, p_in, p_out, rho_m = 1.0, 100.0, 20.0, 500.0
    w = _eval_choke_mass_flow(model, u, p_in, p_out, rho_m)
    assert w > 0


def test_bernoulli_choke_zero_opening():
    """Zero choke opening gives zero mass flow."""
    model = BernoulliChokeModel(K_c=0.001, chk_profile="linear")
    u, p_in, p_out, rho_m = 0.0, 100.0, 20.0, 500.0
    w = _eval_choke_mass_flow(model, u, p_in, p_out, rho_m)
    assert w == pytest.approx(0.0, abs=1e-10)


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
    val = float(Phi.full()) if hasattr(Phi, "full") else float(Phi)
    assert val > 0


def test_simpson_choke_mass_flow_rate():
    """Simpson choke returns positive flow for valid inputs."""
    model = SimpsonChokeModel(K_c=0.001, chk_profile="linear")
    u, p_in, p_out = 1.0, 100.0, 20.0
    x_g, rho_g, rho_l = 0.1, 50.0, 700.0
    w = model.mass_flow_rate(u, p_in, p_out, x_g, rho_g, rho_l)
    val = float(w.full()) if hasattr(w, "full") else float(w)
    assert val > 0
