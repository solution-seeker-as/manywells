"""Tests for manywells.slip."""

import dataclasses

import pytest
import casadi as ca
import numpy as np

from manywells.slip import (
    FITTED_REGIMES,
    HASAN_REGIMES,
    classify_flow_regime,
    hasan_probabilities,
    hasan_rule_quantities,
    SlipModel,
)
from manywells.pvt.dead_oil import dead_oil_surface_tension

FITTED = SlipModel(regime_model='fitted')


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


@pytest.mark.parametrize('model', [FITTED, SlipModel()], ids=['fitted', 'hasan'])
def test_flow_regime_string(model):
    """flow_regime returns one of the model's regimes: annular, slug-churn, bubbly, or the five."""
    v_g, v_l, alpha = 20.0, 2.0, 0.8
    rho_g, rho_l = 1.0, 900.0
    sigma = _sigma(rho_l, 293.15)
    regime = model.flow_regime(v_g, v_l, alpha, rho_g, rho_l, sigma, D=0.1, cos_incl=1.0)
    assert regime in model.regimes
    assert model.regimes == (FITTED_REGIMES if model.regime_model == 'fitted' else HASAN_REGIMES)


# ---------------------------------------------------------------------------
# Inclination (SLIP-10, SLIP-11) and the slip parameters as fields
# ---------------------------------------------------------------------------

def _params(cos_incl, **state):
    s = dict(v_g=6.0, v_l=2.0, alpha=0.4, rho_g=80.0, rho_l=800.0) | state
    sigma = _sigma(s['rho_l'], 330.0)
    C_0, v_inf = FITTED.identify_parameters(**s, sigma=sigma, D=0.15, cos_incl=cos_incl)
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
    """plans/improvements.md 2.7: the regime constants are fields, with v1.0.0's values as defaults; the five
    regimes' constants too (SLIP-12, SLIP-13), and the scale of the profile parameter (SLIP-18)."""
    model = SlipModel(C_0_slug=1.2)
    assert model.C_0_slug == 1.2 and SlipModel().C_0_slug == 1.175
    assert [f.name for f in dataclasses.fields(SlipModel)] == [
        'regime_model', 'C_0_annular', 'C_0_slug', 'C_0_bubbly', 'v_inf_annular', 'f_ms', 'transition_width',
        'lambda_floor', 'profile_scale']
    assert (SlipModel().regime_model, SlipModel().f_ms, SlipModel().transition_width, SlipModel().lambda_floor,
            SlipModel().profile_scale) == ('hasan', 0.02, 0.02, 0.1, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.C_0_bubbly = 1.0


@pytest.mark.parametrize('kw', [dict(regime_model='four'), dict(f_ms=0.0), dict(transition_width=0.0),
                                dict(lambda_floor=-0.1), dict(profile_scale=-0.1), dict(profile_scale=float('nan')),
                                dict(profile_scale=float('inf')), dict(C_0_annular=0.9),
                                dict(C_0_slug=float('nan')), dict(C_0_bubbly=float('inf')),
                                dict(v_inf_annular=float('nan'))])
def test_slip_model_validates_its_fields(kw):
    with pytest.raises(ValueError):
        SlipModel(**kw)


# ---------------------------------------------------------------------------
# The five regimes (SLIP-12 to SLIP-17)
# ---------------------------------------------------------------------------

def _states(n=4000, seed=20261028):
    """Random points across the sampler's range: superficial velocities, densities, surface tension, D, inclination."""
    rng = np.random.default_rng(seed)
    v_gs, v_ls = 10 ** rng.uniform(-2.5, 1.5, n), 10 ** rng.uniform(-2, 0.8, n)
    rho_l = rng.uniform(600, 1050, n)
    rho_g = rho_l * 10 ** rng.uniform(-3.5, -0.4, n)
    sigma, D = rng.uniform(0.003, 0.075, n), rng.uniform(0.0762, 0.1651, n)
    cos_incl = np.concatenate([[0.0, 1.0], rng.uniform(0, 1, n - 2)])
    return v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl


def _hasan(states, model=SlipModel()):
    """(P, C_0, v_inf, Q) of the five regimes at the states: P with one row per regime, Q the rule quantities of
    SLIP-12 (lambda_g, alpha_b, alpha_t, v_gc, v_ms), one row each."""
    x = ca.SX.sym('x', 7)
    v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl = (x[k] for k in range(7))
    P = hasan_probabilities(v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl, model.f_ms, model.transition_width,
                            model.lambda_floor)
    Q = ca.vertcat(*hasan_rule_quantities(v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl, model.f_ms,
                                          model.lambda_floor))
    alpha = 0.5  # any void fraction: C_0 and v_inf depend on the state through v_gs and v_ls only
    C_0, v_inf = model.identify_parameters(v_gs / alpha, v_ls / (1 - alpha), alpha, rho_g, rho_l, sigma, D, cos_incl)
    P, C_0, v_inf, Q = (np.asarray(o) for o in ca.Function('f', [x], [P, C_0, v_inf, Q]).map(len(states[0]))(
        np.vstack(states)))
    return P, C_0.ravel(), v_inf.ravel(), Q


def _crisp(states, Q, model=SlipModel()):
    """SLIP-17's regime at each state, and the smallest margin of its rules in units of w."""
    v_gs, v_ls = states[0], states[1]
    lam, a_b, a_t, v_gc, v_ms = Q
    z = np.vstack([v_gs / v_gc - 1, lam - 0.7, (v_gs + v_ls) / v_ms - 1, lam - 1.08 / 2.08, a_b - a_t])
    A1, A2, M, G, S = z > 0
    regime = np.where(A1 & A2, 'annular', np.where(M & G, 'churn', np.where(M, 'dispersed-bubbly',
                                                                             np.where(S, 'slug', 'bubbly'))))
    return regime, np.min(np.abs(z), axis=0) / model.transition_width


def test_five_regime_probabilities_lie_in_the_unit_interval_and_sum_to_one():  # SLIP-14
    P, _, _, _ = _hasan(_states())
    assert np.all(P >= 0) and np.all(P <= 1)
    assert np.allclose(P.sum(axis=0), 1.0, rtol=0, atol=1e-12)


def test_five_regime_labels_follow_the_hierarchy_away_from_its_boundaries():  # SLIP-13, SLIP-14, SLIP-16, SLIP-17
    states = _states()
    P, _, _, Q = _hasan(states)
    labels = np.array([SlipModel().label(P[:, k]) for k in range(P.shape[1])])
    crisp, margin = _crisp(states, Q)
    far = margin > 5
    assert far.sum() > len(far) / 2
    assert np.all(labels[far] == crisp[far])
    assert set(crisp[far]) == set(HASAN_REGIMES)


def test_five_regime_parameters_are_table_1_blends():  # SLIP-15
    _, C_0, v_inf, _ = _hasan(_states())
    assert np.all(C_0 >= 1.0 - 1e-12) and np.all(C_0 <= 1.2 + 1e-12)
    assert np.all(v_inf >= 0)


def test_five_regime_parameters_do_not_depend_on_the_void_fraction():  # SLIP-15: one void fraction
    """At fixed superficial velocities C_0 and v_inf do not depend on alpha, so the slip law has one root in alpha."""
    model = SlipModel()
    for v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl in [(0.6, 0.5, 80.0, 780.0, 0.018, 0.127, 1.0),
                                                        (2.45, 1.0, 80.0, 780.0, 0.018, 0.127, 0.5),
                                                        (4.0, 3.0, 15.0, 820.0, 0.025, 0.1, 0.0)]:
        values = [tuple(float(v) for v in model.identify_parameters(v_gs / a, v_ls / (1 - a), a, rho_g, rho_l, sigma,
                                                                    D, cos_incl)) for a in (0.05, 0.3, 0.6, 0.95)]
        assert np.allclose(values, values[0], rtol=1e-12, atol=0)


def test_no_bubbly_flow_in_a_horizontal_cell():  # SLIP-14
    """At cos(theta) = 0 the bubbly-slug threshold alpha_t is 0 (A-14): wherever alpha_b exceeds 5 w, the flow is not
    bubbly, and the Taylor bubble does not drift (SLIP-10)."""
    v_gs, v_ls, rho_g, rho_l, sigma, D, _ = _states()
    P, _, _, Q = _hasan((v_gs, v_ls, rho_g, rho_l, sigma, D, np.zeros_like(v_gs)))
    past = Q[1] > 5 * SlipModel().transition_width  # alpha_b
    assert past.sum() > len(past) / 2
    assert np.all(P[4][past] <= 1 / (1 + np.exp(5)))  # the soft truth of S is at least sigma(5) there
    assert not np.any(np.argmax(P[:, past], axis=0) == HASAN_REGIMES.index('bubbly'))


def test_dispersed_bubbly_flow_holds_less_gas_than_its_limit():  # SLIP-14: Hasan and Kabir (2018), at most 0.52
    states = _states()
    P, C_0, v_inf, _ = _hasan(states)
    v_gs, v_ls = states[0], states[1]
    alpha = v_gs / (C_0 * (v_gs + v_ls) + v_inf)
    dispersed = np.argmax(P, axis=0) == HASAN_REGIMES.index('dispersed-bubbly')
    assert dispersed.any() and np.all(alpha[dispersed] < 0.52)


# ---------------------------------------------------------------------------
# The scaled profile parameter (SLIP-18)
# ---------------------------------------------------------------------------

def _parameters(model, states, alpha):
    """C_0 and v_inf of a slip model at the states of _states, at void fraction alpha, as arrays."""
    x = ca.SX.sym('x', 7)
    v_gs, v_ls, rho_g, rho_l, sigma, D, cos_incl = (x[k] for k in range(7))
    C_0, v_inf = model.identify_parameters(v_gs / alpha, v_ls / (1 - alpha), alpha, rho_g, rho_l, sigma, D, cos_incl)
    out = ca.Function('f', [x], [C_0, v_inf]).map(len(states[0]))(np.vstack(states))
    return tuple(np.asarray(o).ravel() for o in out)


@pytest.mark.parametrize('regime_model', ['hasan', 'fitted'])
def test_profile_scale_moves_c_0_towards_no_slip(regime_model):  # SLIP-18
    """C_0* = 1 + k (C_0 - 1): the classifier's own C_0 at k = 1 (to the bit), no slip at k = 0, monotone in k and
    between 1 and C_0 for k in [0, 1]; v_inf does not depend on k."""
    states = _states(n=1000)
    for alpha in (0.1, 0.5, 0.9):
        C_0, v_inf = _parameters(SlipModel(regime_model=regime_model), states, alpha)
        assert np.array_equal(_parameters(SlipModel(regime_model=regime_model, profile_scale=1.0), states, alpha)[0],
                              C_0)
        last = np.ones_like(C_0)
        for k in (0.0, 0.25, 0.5, 1.0, 1.5):
            C_0_k, v_inf_k = _parameters(SlipModel(regime_model=regime_model, profile_scale=k), states, alpha)
            assert np.allclose(C_0_k, 1 + k * (C_0 - 1), rtol=0, atol=1e-14)
            assert np.all(C_0_k >= last - 1e-14) and np.all(C_0_k >= 1 - 1e-12)
            if k <= 1:
                assert np.all(C_0_k <= C_0 + 1e-14)
            assert np.array_equal(v_inf_k, v_inf)
            last = C_0_k
        assert np.all(_parameters(SlipModel(regime_model=regime_model, profile_scale=0.0), states, alpha)[0] == 1.0)


def test_profile_scale_leaves_annular_flow_alone():  # SLIP-18: C_0 = 1 in annular flow, in both classifiers
    for model in (SlipModel(profile_scale=0.3), SlipModel(regime_model='fitted', profile_scale=0.3)):
        C_0, _ = model.identify_parameters(33.3, 0.43, 0.97, 40.0, 820.0, 0.02, 0.15, 1.0)
        assert abs(float(C_0) - 1.0) < 1e-3
