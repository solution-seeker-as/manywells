"""Tests for manywells.inflow."""

import math

import casadi as ca
import numpy as np
import pytest

from manywells.inflow import (
    Forchheimer,
    ProductivityIndex,
    Vogel,
    FixedFlowRate,
)
from manywells.pvt import density_from_api
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.simulator import WellProperties


def test_productivity_index_init_valid():
    """ProductivityIndex accepts valid k_l."""
    pi = ProductivityIndex(k_l=0.5)
    assert pi.k_l == 0.5


def test_productivity_index_init_invalid_k_l():
    """ProductivityIndex rejects negative k_l."""
    with pytest.raises(ValueError, match='non-negative'):
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
    fix = FixedFlowRate(w_l_const=5.0)
    w_l = fix.liquid_mass_flow_rate(80.0, 100.0)
    assert w_l == 5.0
    w_l2 = fix.liquid_mass_flow_rate(20.0, 50.0)
    assert w_l2 == 5.0


def test_fixed_flow_rate_negative_rejected():
    """FixedFlowRate rejects negative liquid rate."""
    with pytest.raises(ValueError, match='non-negative'):
        FixedFlowRate(w_l_const=-1.0)


# ---------------------------------------------------------------------------------------------
# Forchheimer's gas inflow (INF-10, INF-11; specs/features/022-gas-rate-inflow.md)

GAS = Forchheimer(a=300.0, b=10.0)  # bar2 s/kg, bar2 s2/kg2: a gas rate of about 20 kg/s at 50 bar of drawdown
P_R = 150.0
# A wet gas (CGR 1/3000, WGR 0.3/0.7 of it) and a condensate-like black oil with water
WET_GAS = FluidModel(pvt=BlackOil(rho_o=density_from_api(55.0), rho_g=0.75), gor=10_000.0, wlr=0.5)
CONDENSATE = FluidModel(pvt=BlackOil(rho_o=density_from_api(50.0), rho_g=0.85), gor=3000.0, wlr=0.3)


def test_forchheimer_refuses_its_coefficients_out_of_range():
    for a, b in ((0.0, 1.0), (-1.0, 1.0), (math.nan, 1.0), (math.inf, 1.0), (1.0, -1e-9), (1.0, math.nan)):
        with pytest.raises(ValueError, match='Forchheimer'):
            Forchheimer(a=a, b=b)
    assert Forchheimer(a=1.0, b=0.0).b == 0.0


def test_forchheimer_refuses_a_fluid_without_gas():
    """INF-11 needs f_g > 0: WellProperties refuses the gas inflow on a fluid without gas, and the inflow needs a
    fluid at all."""
    with pytest.raises(ValueError, match='f_g > 0'):
        WellProperties(inflow=GAS, fluid=FluidModel(gor=0.0))
    with pytest.raises(ValueError, match='needs the fluid'):
        GAS.liquid_mass_flow_rate(100.0, P_R)
    assert WellProperties(inflow=GAS, fluid=CONDENSATE).inflow is GAS


@pytest.mark.parametrize('b', [0.0, 1e-3, 10.0, 1e4])
@pytest.mark.parametrize('p', [1.01325, 60.0, 120.0, 149.0, 149.999])
def test_forchheimer_solves_its_law(p, b):
    """The closed form is the root of a w + b w^2 = p_r^2 - p^2 that is zero at p_r, the quadratic's positive root,
    and Delta/a in the linear limit b = 0."""
    inflow = Forchheimer(a=GAS.a, b=b)
    w = inflow.gas_mass_flow_rate(p, P_R)
    delta = (P_R - p) * (P_R + p)  # p_r^2 - p^2 without the cancellation near p_r, which is 1e-12 at 149.999 bar
    assert w > 0
    assert inflow.a * w + b * w**2 == pytest.approx(delta, rel=1e-12)
    if b == 0:
        assert w == pytest.approx(delta / inflow.a, rel=1e-15)
    else:
        assert w == pytest.approx((-inflow.a + math.sqrt(inflow.a**2 + 4 * b * delta)) / (2 * b), rel=1e-6)


def test_forchheimer_is_zero_at_the_reservoir_pressure_and_odd_above_it():
    assert GAS.gas_mass_flow_rate(P_R, P_R) == 0.0
    for p in (60.0, 140.0, 149.9):
        above = math.sqrt(2 * P_R**2 - p**2)  # the same |Delta| above p_r
        assert GAS.gas_mass_flow_rate(above, P_R) == pytest.approx(-GAS.gas_mass_flow_rate(p, P_R), rel=1e-12)


def test_forchheimer_falls_with_the_bottomhole_pressure_with_a_finite_slope():
    """Strictly decreasing in p on (0, infinity), through p_r, with the slope -2 p / sqrt(a^2 + 4 b |Delta|), which
    is -2 p_r / a at p_r: a CasADi gradient, against the expression and a central difference."""
    p = ca.SX.sym('p')
    w = GAS.gas_mass_flow_rate(p, P_R)
    slope = ca.Function('slope', [p], [ca.gradient(w, p)])
    ps = np.r_[np.linspace(1.0, 200.0, 400), P_R]
    rates = np.array([GAS.gas_mass_flow_rate(x, P_R) for x in np.sort(ps)])
    assert np.all(np.diff(rates) < 0)
    assert float(slope(P_R)) == pytest.approx(-2 * P_R / GAS.a, rel=1e-14)
    h = 1e-4
    for x in (5.0, 80.0, 149.0, 151.0, 190.0):
        expected = -2 * x / math.sqrt(GAS.a**2 + 4 * GAS.b * abs(P_R**2 - x**2))
        central = (GAS.gas_mass_flow_rate(x + h, P_R) - GAS.gas_mass_flow_rate(x - h, P_R)) / (2 * h)
        assert float(slope(x)) == pytest.approx(expected, rel=1e-12)
        assert float(slope(x)) == pytest.approx(central, rel=1e-7)


def test_forchheimer_is_the_same_on_symbols_and_floats():
    p, p_r = ca.SX.sym('p'), ca.SX.sym('p_r')
    f = ca.Function('w', [p, p_r], [GAS.liquid_mass_flow_rate(p, p_r, CONDENSATE)])
    for x in (1.01325, 50.0, 149.5, 150.0, 160.0):
        assert float(f(x, P_R)) == GAS.liquid_mass_flow_rate(x, P_R, CONDENSATE)


@pytest.mark.parametrize('fluid', [WET_GAS, CONDENSATE], ids=['wet gas', 'condensate'])
def test_forchheimer_gives_the_liquid_in_the_fluids_ratios(fluid):
    """INF-11: the liquid is (1 - f_g)/f_g of the gas, which is (rho_o CGR + rho_w WGR) / rho_g in a gas well's
    ratios; INF-4 of it gives the gas rate back."""
    w_g = GAS.gas_mass_flow_rate(100.0, P_R)
    w_l = GAS.liquid_mass_flow_rate(100.0, P_R, fluid)
    cgr = 1 / fluid.gor
    wgr = fluid.wlr / (1 - fluid.wlr) / fluid.gor
    assert w_l == pytest.approx((fluid.rho_o * cgr + fluid.rho_w * wgr) * w_g / fluid.rho_g, rel=1e-12)
    assert fluid.reservoir_gas_rate(w_l) == pytest.approx(w_g, rel=1e-14)


def test_forchheimer_from_a_standard_gas_rate():
    """from_gas_rate converts bar2 per Sm3/d to bar2 s/kg with the gas's density: a w = a_q q, w = rho_g q / 86 400."""
    a_q, b_q, rho_g = 2.0e-4, 3.0e-10, 0.8  # bar2/(Sm3/d), bar2/(Sm3/d)2, kg/m3
    inflow = Forchheimer.from_gas_rate(a_q, b_q, rho_g)
    assert inflow.a == pytest.approx(a_q * 86400 / 0.8, rel=1e-15)
    assert inflow.b == pytest.approx(b_q * (86400 / 0.8) ** 2, rel=1e-15)
    q = 5.0e5  # Sm3/d
    w = inflow.gas_mass_flow_rate(math.sqrt(P_R**2 - (a_q * q + b_q * q**2)), P_R)
    assert w == pytest.approx(rho_g * q / 86400, rel=1e-10)
    with pytest.raises(ValueError, match='rho_g'):
        Forchheimer.from_gas_rate(a_q, b_q, 0.0)


def test_the_liquid_inflows_ignore_the_fluid():
    """The fluid argument changes nothing for the liquid inflows, and two arguments still work (022's interface)."""
    for inflow in (ProductivityIndex(k_l=0.5), Vogel(w_l_max=40.0), FixedFlowRate(w_l_const=12.0)):
        assert inflow.liquid_mass_flow_rate(100.0, P_R, CONDENSATE) == inflow.liquid_mass_flow_rate(100.0, P_R)
