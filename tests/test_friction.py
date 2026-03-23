"""Tests for manywells.friction."""

import pytest
import casadi as ca

from manywells.friction import haaland_friction_factor, chen_friction_factor, friction_factor


def _eval(expr):
    """Evaluate a CasADi expression to a Python float."""
    return float(expr)


class TestHaalandFrictionFactor:

    def test_known_moody_value(self):
        """Haaland at Re=1e5, eps/D=0.001 should be close to Moody chart value ~0.0222."""
        f = _eval(haaland_friction_factor(1e5, 0.001))
        assert f == pytest.approx(0.0222, rel=0.05)

    def test_smooth_pipe(self):
        """Smooth pipe (very small roughness) gives friction factor close to Blasius."""
        Re = 1e5
        eps_D = 1e-7
        f = _eval(haaland_friction_factor(Re, eps_D))
        f_blasius = 0.3164 / Re**0.25
        assert f == pytest.approx(f_blasius, rel=0.10)

    def test_friction_decreases_with_Re(self):
        """For fixed roughness, friction factor decreases as Re increases (turbulent)."""
        eps_D = 0.001
        f_low = _eval(haaland_friction_factor(1e4, eps_D))
        f_high = _eval(haaland_friction_factor(1e6, eps_D))
        assert f_high < f_low

    def test_friction_increases_with_roughness(self):
        """For fixed Re, friction factor increases with roughness."""
        Re = 1e5
        f_smooth = _eval(haaland_friction_factor(Re, 1e-5))
        f_rough = _eval(haaland_friction_factor(Re, 0.01))
        assert f_rough > f_smooth

    def test_positive(self):
        """Friction factor is always positive."""
        f = _eval(haaland_friction_factor(5e4, 0.0005))
        assert f > 0


class TestFrictionFactor:

    def test_laminar_regime(self):
        """In laminar regime (Re=1000), friction factor is close to 64/Re."""
        Re = 1000
        eps_D = 0.001
        f = _eval(friction_factor(Re, eps_D))
        assert f == pytest.approx(64.0 / Re, rel=0.05)

    def test_turbulent_regime(self):
        """In turbulent regime (Re=1e5), friction factor matches Chen (default)."""
        Re = 1e5
        eps_D = 0.001
        f = _eval(friction_factor(Re, eps_D))
        f_chen = _eval(chen_friction_factor(Re, eps_D))
        assert f == pytest.approx(f_chen, rel=0.01)

    def test_turbulent_regime_haaland(self):
        """In turbulent regime (Re=1e5), friction factor matches Haaland when requested."""
        Re = 1e5
        eps_D = 0.001
        f = _eval(friction_factor(Re, eps_D, correlation='haaland'))
        f_haaland = _eval(haaland_friction_factor(Re, eps_D))
        assert f == pytest.approx(f_haaland, rel=0.01)

    def test_transition_smooth(self):
        """Friction factor in transition region is between laminar and turbulent values."""
        Re = 3000
        eps_D = 0.001
        f = _eval(friction_factor(Re, eps_D))
        f_lam = 64.0 / Re
        f_turb = _eval(chen_friction_factor(Re, eps_D))
        assert min(f_lam, f_turb) <= f <= max(f_lam, f_turb) * 1.1

    def test_positive_near_zero_Re(self):
        """Friction factor stays positive even for very small Re."""
        f = _eval(friction_factor(0.1, 0.001))
        assert f > 0

    def test_casadi_symbolic(self):
        """friction_factor works with CasADi symbolic variables."""
        Re_sym = ca.SX.sym('Re')
        eps_D = 0.001
        f_sym = friction_factor(Re_sym, eps_D)
        fun = ca.Function('f', [Re_sym], [f_sym])
        result = float(fun(1e5))
        assert result > 0
