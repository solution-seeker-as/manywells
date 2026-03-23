"""Tests for manywells.pvt.fluid.FluidModel."""

import pytest

from manywells.pvt import (
    R_UNIVERSAL, P_REF, T_REF, WATER,
    density_from_api, api_from_density,
)
from manywells.units import M_AIR
from manywells.pvt.fluid import FluidModel
from manywells.pvt.black_oil import BlackOilPVT
from manywells.units import CF_BAR, CF_PSI


class TestFluidModelDerivedProperties:

    def test_rho_o_from_api(self):
        """rho_o matches density_from_api(api)."""
        fl = FluidModel(api=35.0)
        assert fl.rho_o == pytest.approx(density_from_api(35.0))

    def test_R_s_from_sg_gas(self):
        """R_s == R_UNIVERSAL / (M_AIR * sg_gas)."""
        fl = FluidModel(sg_gas=0.65)
        assert fl.R_s == pytest.approx(R_UNIVERSAL / (M_AIR * 0.65))

    def test_M_g_from_sg_gas(self):
        """M_g == M_AIR * sg_gas."""
        fl = FluidModel(sg_gas=0.65)
        assert fl.M_g == pytest.approx(M_AIR * 0.65)

    def test_pure_oil_wlr_zero(self):
        """water_cut=0 => wlr=0."""
        fl = FluidModel(water_cut=0.0)
        assert fl.wlr == 0.0

    def test_pure_oil_rho_l_equals_rho_o(self):
        """water_cut=0 => rho_l == rho_o."""
        fl = FluidModel(water_cut=0.0, api=35.0)
        assert fl.rho_l == pytest.approx(fl.rho_o)

    def test_pure_oil_f_o_in_liquid(self):
        """water_cut=0 => f_o_in_liquid == 1."""
        fl = FluidModel(water_cut=0.0)
        assert fl.f_o_in_liquid == pytest.approx(1.0)

    def test_half_water_cut_rho_l(self):
        """50% water cut gives volume-weighted density."""
        fl = FluidModel(water_cut=0.5, api=35.0)
        expected = 0.5 * fl.rho_w + 0.5 * fl.rho_o
        assert fl.rho_l == pytest.approx(expected)

    def test_half_water_cut_cp_l(self):
        """50% water cut gives volume-weighted cp."""
        fl = FluidModel(water_cut=0.5)
        expected = 0.5 * fl.cp_w + 0.5 * fl.cp_o
        assert fl.cp_l == pytest.approx(expected)

    def test_GLR_no_water(self):
        """GLR == GOR when water_cut=0."""
        fl = FluidModel(GOR=200.0, water_cut=0.0)
        assert fl.GLR == pytest.approx(200.0)

    def test_GLR_with_water(self):
        """GLR == GOR * (1 - water_cut)."""
        fl = FluidModel(GOR=200.0, water_cut=0.3)
        assert fl.GLR == pytest.approx(200.0 * 0.7)


class TestFluidModelFg:

    def test_f_g_manual_formula_no_water(self):
        """f_g matches manual calculation for water_cut=0."""
        fl = FluidModel(api=35.0, sg_gas=0.65, GOR=200.0, water_cut=0.0)
        rho_g_std = P_REF / (fl.R_s * T_REF)
        expected = (rho_g_std * 200.0) / (rho_g_std * 200.0 + fl.rho_o)
        assert fl.f_g == pytest.approx(expected)

    def test_f_g_manual_formula_with_water(self):
        """f_g matches manual calculation for non-zero water_cut."""
        fl = FluidModel(api=35.0, sg_gas=0.65, GOR=200.0, water_cut=0.3)
        rho_g_std = P_REF / (fl.R_s * T_REF)
        wc = 0.3
        expected = (rho_g_std * 200.0) / (rho_g_std * 200.0 + fl.rho_o + fl.rho_w * wc / (1 - wc))
        assert fl.f_g == pytest.approx(expected)

    def test_f_g_in_range(self):
        """f_g is in (0, 1) for typical parameters."""
        fl = FluidModel()
        assert 0 < fl.f_g < 1

    def test_f_g_zero_gor(self):
        """GOR=0 => f_g=0 (no gas)."""
        fl = FluidModel(GOR=0.0)
        assert fl.f_g == pytest.approx(0.0)


class TestFluidModelMethods:

    def test_gas_mass_flow_rate(self):
        """gas_mass_flow_rate(w_l) == (f_g / (1 - f_g)) * w_l."""
        fl = FluidModel(GOR=200.0)
        w_l = 10.0
        w_g = fl.gas_mass_flow_rate(w_l)
        assert w_g == pytest.approx((fl.f_g / (1 - fl.f_g)) * w_l)

    def test_liquid_density_dead_oil(self):
        """Dead oil (no black_oil) returns constant rho_l."""
        fl = FluidModel(api=35.0, water_cut=0.0)
        rho = fl.liquid_density(100.0, 350.0)
        assert rho == pytest.approx(fl.rho_l)

    def test_liquid_viscosity_positive(self):
        """Liquid viscosity is positive (dead oil path)."""
        fl = FluidModel()
        p = 100.0  # bar
        T = 273.15 + 60
        mu = float(fl.liquid_viscosity(p, T))
        assert mu > 0

    def test_gas_viscosity_positive(self):
        """Gas viscosity is positive."""
        fl = FluidModel()
        T = 273.15 + 60
        rho_g = 50.0
        mu = float(fl.gas_viscosity(T, rho_g))
        assert mu > 0


class TestFluidModelFactory:

    def test_default_rho_o(self):
        """Default factory gives rho_o == 850 (backward compat)."""
        fl = FluidModel.default()
        assert fl.rho_o == pytest.approx(850.0, rel=1e-4)

    def test_default_api_roundtrip(self):
        """Default api is consistent with rho_o=850."""
        fl = FluidModel.default()
        assert fl.api == pytest.approx(api_from_density(850.0))

    def test_default_water_cut_zero(self):
        """Default has no water."""
        fl = FluidModel.default()
        assert fl.water_cut == 0.0
        assert fl.wlr == 0.0

    def test_default_f_g_near_old(self):
        """Default f_g is close to old default of 0.1379."""
        fl = FluidModel.default()
        assert fl.f_g == pytest.approx(0.1379, rel=0.01)


class TestLiveOilViscosity:

    @staticmethod
    def _make_black_oil_fluid():
        bo = BlackOilPVT(api=30, sg_gas=0.65, p_sep=200 * CF_PSI, T_sep=333.15)
        return FluidModel(api=30, sg_gas=0.65, GOR=150.0, water_cut=0.0, black_oil=bo)

    def test_black_oil_viscosity_less_than_dead(self):
        """With black oil, liquid viscosity is less than dead oil viscosity."""
        fl_dead = FluidModel(api=30, sg_gas=0.65, GOR=150.0, water_cut=0.0)
        fl_live = self._make_black_oil_fluid()
        p = 150.0  # bar (above atmospheric, so Rs > 0)
        T = 273.15 + 80
        mu_dead = float(fl_dead.liquid_viscosity(p, T))
        mu_live = float(fl_live.liquid_viscosity(p, T))
        assert mu_live < mu_dead

    def test_black_oil_viscosity_positive(self):
        """Live oil viscosity is positive."""
        fl = self._make_black_oil_fluid()
        mu = float(fl.liquid_viscosity(200.0, 273.15 + 80))
        assert mu > 0

    def test_dead_oil_path_ignores_pressure(self):
        """Without black_oil, viscosity does not depend on pressure."""
        fl = FluidModel(api=30, sg_gas=0.65, GOR=150.0, water_cut=0.0)
        T = 273.15 + 80
        mu_low_p = float(fl.liquid_viscosity(50.0, T))
        mu_high_p = float(fl.liquid_viscosity(300.0, T))
        assert mu_low_p == pytest.approx(mu_high_p)
