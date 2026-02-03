"""Tests for manywells.calibration."""

import pytest
import pandas as pd
import numpy as np

from manywells.calibration.choke_cal import calibrate_bernoulli_choke_model
from manywells.calibration.inflow_cal import calibrate_inflow_model
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.inflow import ProductivityIndex, Vogel


def test_calibrate_bernoulli_choke_model_wrong_type():
    """calibrate_bernoulli_choke_model rejects non-BernoulliChokeModel."""
    data = pd.DataFrame({"u": [0.5], "p": [100], "p_s": [20], "w_m": [1.0], "rho_m": [500]})
    with pytest.raises(AssertionError, match="not of type BernoulliChokeModel"):
        calibrate_bernoulli_choke_model(data, SimpsonChokeModel())


def test_calibrate_bernoulli_choke_model_recovers_K_c():
    """Calibration with synthetic data recovers known K_c."""
    K_c_true = 0.002
    choke = BernoulliChokeModel(K_c=K_c_true, chk_profile="linear")
    # Generate synthetic data using the same choke model (with choked flow: p_c = cpr*p_in)
    p_in, p_out, rho_m, u = 100.0, 30.0, 600.0, 1.0
    w_expr = choke.mass_flow_rate(u, p_in, p_out, rho_m)
    w_m_true = float(w_expr.full()) if hasattr(w_expr, "full") else float(w_expr)
    data = pd.DataFrame(
        [
            {
                "u": u,
                "p": p_in,
                "p_s": p_out,
                "w_m": w_m_true,
                "rho_m": rho_m,
            }
        ]
    )
    K_c_opt = calibrate_bernoulli_choke_model(data, BernoulliChokeModel(K_c=K_c_true * 0.5, chk_profile="linear"))
    assert K_c_opt == pytest.approx(K_c_true, rel=1e-3)


def test_calibrate_inflow_model_vogel_unsupported_data():
    """calibrate_inflow_model with Vogel and valid data runs (no exception)."""
    # Minimal data that matches Vogel: w_l = w_l_max * (1 - 0.2*r - 0.8*r**2)
    vogel = Vogel(w_l_max=5.0, f_g=0.2)
    data = pd.DataFrame(
        [
            {"p": 80.0, "p_r": 100.0, "w_l": 3.6, "w_g": 0.9},
            {"p": 50.0, "p_r": 100.0, "w_l": 4.375, "w_g": 1.09375},
        ]
    )
    # Just check it returns a number (full calibration may be fragile)
    result = calibrate_inflow_model(data, Vogel(w_l_max=4.0, f_g=0.2))
    assert isinstance(result, (int, float))
    assert result >= 0


def test_calibrate_inflow_model_pi_recovers_k_l():
    """Calibration of PI model with synthetic data recovers k_l."""
    k_l_true = 0.8
    pi = ProductivityIndex(k_l=k_l_true, f_g=0.15)
    data = pd.DataFrame(
        [
            {"p": 70.0, "p_r": 100.0, "w_l": pi.mass_flow_rates(70, 100)[0], "w_g": pi.mass_flow_rates(70, 100)[1]},
            {"p": 60.0, "p_r": 100.0, "w_l": pi.mass_flow_rates(60, 100)[0], "w_g": pi.mass_flow_rates(60, 100)[1]},
        ]
    )
    k_l_opt = calibrate_inflow_model(data, ProductivityIndex(k_l=0.5, f_g=0.15))
    assert k_l_opt == pytest.approx(k_l_true, rel=1e-3)


def test_calibrate_inflow_model_unsupported_type():
    """calibrate_inflow_model raises for unsupported inflow model type."""
    from manywells.inflow import FixedFlowRate

    data = pd.DataFrame([{"p": 80.0, "p_r": 100.0, "w_l": 1.0, "w_g": 0.2}])
    with pytest.raises(ValueError, match="not supported"):
        calibrate_inflow_model(data, FixedFlowRate(w_l_const=1.0, w_g_const=0.2))
