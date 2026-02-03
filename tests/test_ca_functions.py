"""Tests for manywells.ca_functions."""

import numpy as np
import casadi as ca
import pytest

from manywells.ca_functions import (
    ca_max_approx,
    ca_min_approx,
    ca_softmax,
    ca_sigmoid,
    ca_double_sigmoid,
)


def _eval(f_expr, *args):
    """Evaluate a CasADi expression with scalar inputs."""
    symbols = [ca.SX.sym(f"x{i}") for i in range(len(args))]
    f = ca.Function("f", symbols, [f_expr])
    return float(f(*args))


def test_ca_max_approx():
    """ca_max_approx approximates max(x, y)."""
    x, y = 3.0, 5.0
    m = ca_max_approx(x, y)
    assert _eval(m, x, y) == pytest.approx(5.0, rel=1e-4)
    m2 = ca_max_approx(y, x)
    assert _eval(m2, y, x) == pytest.approx(5.0, rel=1e-4)
    # equal (approx adds sqrt(eps)/2, so use abs tolerance)
    z = 4.0
    m3 = ca_max_approx(z, z)
    assert _eval(m3, z, z) == pytest.approx(4.0, abs=1e-3)


def test_ca_min_approx():
    """ca_min_approx approximates min(x, y)."""
    x, y = 3.0, 5.0
    m = ca_min_approx(x, y)
    assert _eval(m, x, y) == pytest.approx(3.0, rel=1e-4)
    m2 = ca_min_approx(y, x)
    assert _eval(m2, y, x) == pytest.approx(3.0, rel=1e-4)


def test_ca_softmax():
    """ca_softmax sums to 1 and has positive entries."""
    x = ca.SX.sym("x", 3)
    p = ca_softmax(x)
    f = ca.Function("f", [x], [p, ca.sum1(p)])
    out, total = f([1.0, 2.0, 3.0])
    out = out.full().flatten()
    assert total.full().item() == pytest.approx(1.0)
    assert np.all(out > 0)
    assert np.all(out < 1)


def test_ca_sigmoid():
    """ca_sigmoid is 1/(1+exp(-k*(x-a)))."""
    x = ca.SX.sym("x")
    a, k = 0.5, 2.0
    s = ca_sigmoid(x, a, k)
    f = ca.Function("f", [x], [s])
    # at x = a, sigmoid = 0.5
    assert f(0.5) == pytest.approx(0.5)
    assert f(10.0) == pytest.approx(1.0, rel=1e-5)
    assert f(-10.0) == pytest.approx(0.0, abs=1e-8)


def test_ca_double_sigmoid():
    """ca_double_sigmoid has levels l1, l2, l3 and transitions at a, b."""
    x = ca.SX.sym("x")
    l1, l2, l3 = 0.0, 1.0, 2.0
    a, b = 1.0, 3.0
    s = ca_double_sigmoid(x, l1, l2, l3, a, b, k=20)
    f = ca.Function("f", [x], [s])
    assert f(0.0) == pytest.approx(0.0, abs=0.01)
    assert f(5.0) == pytest.approx(2.0, abs=0.01)
    # a < b required
    with pytest.raises(AssertionError, match="a must be less than b"):
        ca_double_sigmoid(x, l1, l2, l3, a=3.0, b=1.0)
