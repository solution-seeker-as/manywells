"""Tests for manywells.pvt.fluid.FluidModel."""

import pytest

from manywells.pvt import (
    R_UNIVERSAL, P_REF, T_REF,
    density_from_api, api_from_density,
    gas_density_from_sg, sg_from_gas_density,
)
from manywells.units import M_AIR, CF_PSI
from manywells.pvt.fluid import FluidModel


class TestHelperRoundtrips:

    def test_api_density_roundtrip(self):
        rho = density_from_api(35.0)
        assert api_from_density(rho) == pytest.approx(35.0)

    def test_sg_gas_density_roundtrip(self):
        rho = gas_density_from_sg(0.65)
        assert sg_from_gas_density(rho) == pytest.approx(0.65)

    def test_gas_density_from_sg_methane(self):
        """sg ~0.554 for methane-like gas gives ~0.68 kg/m3."""
        rho = gas_density_from_sg(0.554)
        expected = P_REF * M_AIR * 0.554 / (R_UNIVERSAL * T_REF)
        assert rho == pytest.approx(expected)


class TestFluidModelConstruction:

    def test_default_construction(self):
        fl = FluidModel()
        assert fl.rho_o == 850.0
        assert fl.gor == 200.0
        assert fl.wlr == 0.0
        assert fl.oil_model == 'black_oil'
        assert fl.ideal_gas is False

    def test_dead_oil_construction(self):
        fl = FluidModel(oil_model='dead_oil')
        assert fl._black_oil is None

    def test_black_oil_construction(self):
        fl = FluidModel(oil_model='black_oil')
        assert fl._black_oil is not None

    def test_invalid_oil_model(self):
        with pytest.raises(ValueError, match="Unknown oil_model"):
            FluidModel(oil_model='invalid')


class TestFluidModelDerivedProperties:

    def test_api_from_rho_o(self):
        fl = FluidModel(rho_o=850.0)
        assert fl.api == pytest.approx(api_from_density(850.0))

    def test_sg_gas_from_rho_g(self):
        rho_g = gas_density_from_sg(0.65)
        fl = FluidModel(rho_g=rho_g)
        assert fl.sg_gas == pytest.approx(0.65)

    def test_R_s_from_rho_g(self):
        fl = FluidModel(rho_g=gas_density_from_sg(0.65))
        assert fl.R_s == pytest.approx(R_UNIVERSAL / (M_AIR * 0.65))

    def test_M_g_from_rho_g(self):
        fl = FluidModel(rho_g=gas_density_from_sg(0.65))
        assert fl.M_g == pytest.approx(M_AIR * 0.65)

    def test_pure_oil_rho_l_equals_rho_o(self):
        fl = FluidModel(wlr=0.0, rho_o=850.0)
        assert fl.rho_l == pytest.approx(850.0)

    def test_half_wlr_rho_l(self):
        fl = FluidModel(wlr=0.5, rho_o=850.0)
        expected = 0.5 * fl.rho_w + 0.5 * 850.0
        assert fl.rho_l == pytest.approx(expected)

    def test_half_wlr_cp_l(self):
        fl = FluidModel(wlr=0.5)
        expected = 0.5 * fl.cp_w + 0.5 * fl.cp_o
        assert fl.cp_l == pytest.approx(expected)

    def test_glr_no_water(self):
        fl = FluidModel(gor=200.0, wlr=0.0)
        assert fl.glr == pytest.approx(200.0)

    def test_glr_with_water(self):
        fl = FluidModel(gor=200.0, wlr=0.3)
        assert fl.glr == pytest.approx(200.0 * 0.7)


class TestFluidModelFg:

    def test_f_g_manual_formula_no_water(self):
        fl = FluidModel(rho_o=850.0, rho_g=gas_density_from_sg(0.65), gor=200.0, wlr=0.0)
        expected = (fl.rho_g * 200.0) / (fl.rho_g * 200.0 + fl.rho_o)
        assert fl.f_g == pytest.approx(expected)

    def test_f_g_manual_formula_with_water(self):
        fl = FluidModel(rho_o=850.0, rho_g=gas_density_from_sg(0.65), gor=200.0, wlr=0.3)
        expected = (fl.rho_g * 200.0) / (fl.rho_g * 200.0 + fl.rho_o + fl.rho_w * 0.3 / 0.7)
        assert fl.f_g == pytest.approx(expected)

    def test_f_g_in_range(self):
        fl = FluidModel()
        assert 0 < fl.f_g < 1

    def test_f_g_zero_gor(self):
        fl = FluidModel(gor=0.0)
        assert fl.f_g == pytest.approx(0.0)


class TestFluidModelFoInLiquid:

    def test_pure_oil(self):
        fl = FluidModel(wlr=0.0)
        assert fl.f_o_in_liquid == pytest.approx(1.0)

    def test_with_water(self):
        fl = FluidModel(wlr=0.3)
        assert 0 < fl.f_o_in_liquid < 1

    def test_near_pure_water(self):
        fl = FluidModel(wlr=0.99)
        assert fl.f_o_in_liquid < 0.02


class TestFluidModelMethods:

    def test_gas_mass_flow_rate(self):
        fl = FluidModel(gor=200.0)
        w_l = 10.0
        w_g = fl.gas_mass_flow_rate(w_l)
        assert w_g == pytest.approx((fl.f_g / (1 - fl.f_g)) * w_l)


class TestDeadOilUnifiedInterface:

    def test_rs_returns_zero(self):
        fl = FluidModel(oil_model='dead_oil')
        assert fl.rs(100.0, 350.0) == 0

    def test_bo_returns_one(self):
        fl = FluidModel(oil_model='dead_oil')
        assert fl.bo(100.0, 350.0) == 1.0

    def test_liquid_density_constant(self):
        fl = FluidModel(oil_model='dead_oil', rho_o=850.0, wlr=0.0)
        rho = fl.liquid_density(100.0, 350.0)
        assert rho == pytest.approx(850.0)

    def test_liquid_viscosity_positive(self):
        fl = FluidModel(oil_model='dead_oil')
        mu = float(fl.liquid_viscosity(100.0, 273.15 + 60))
        assert mu > 0

    def test_liquid_viscosity_ignores_pressure(self):
        fl = FluidModel(oil_model='dead_oil')
        T = 273.15 + 80
        mu_low_p = float(fl.liquid_viscosity(50.0, T))
        mu_high_p = float(fl.liquid_viscosity(300.0, T))
        assert mu_low_p == pytest.approx(mu_high_p)

    def test_surface_tension_positive(self):
        fl = FluidModel(oil_model='dead_oil')
        sigma = float(fl.surface_tension(100.0, 273.15 + 60))
        assert sigma > 0

    def test_surface_tension_ignores_pressure(self):
        fl = FluidModel(oil_model='dead_oil')
        T = 273.15 + 60
        s1 = float(fl.surface_tension(50.0, T))
        s2 = float(fl.surface_tension(300.0, T))
        assert s1 == pytest.approx(s2)

    def test_free_gas_flux_returns_total(self):
        fl = FluidModel(oil_model='dead_oil')
        w = fl.free_gas_flux(Rs=0, w_g_total=2.0, w_lg=0.5, w_o=1.0)
        assert w == pytest.approx(2.5)

    def test_liquid_flux_returns_inflow(self):
        fl = FluidModel(oil_model='dead_oil')
        w = fl.liquid_flux(Rs=0, w_l_inflow=10.0, w_o=5.0, w_g_total=2.0)
        assert w == pytest.approx(10.0)


class TestBlackOilUnifiedInterface:

    @staticmethod
    def _make():
        return FluidModel(
            rho_o=density_from_api(30),
            rho_g=gas_density_from_sg(0.65),
            gor=150.0, wlr=0.0,
            p_sep=200 * CF_PSI, T_sep=333.15,
        )

    def test_rs_positive_at_pressure(self):
        fl = self._make()
        rs = float(fl.rs(150.0, 373.15))
        assert rs > 0

    def test_bo_greater_than_one(self):
        fl = self._make()
        bo = float(fl.bo(150.0, 373.15))
        assert bo > 1.0

    def test_viscosity_less_than_dead(self):
        fl_dead = FluidModel(
            rho_o=density_from_api(30),
            rho_g=gas_density_from_sg(0.65),
            gor=150.0, wlr=0.0,
            oil_model='dead_oil',
        )
        fl_live = self._make()
        p, T = 150.0, 273.15 + 80
        mu_dead = float(fl_dead.liquid_viscosity(p, T))
        mu_live = float(fl_live.liquid_viscosity(p, T))
        assert mu_live < mu_dead

    def test_viscosity_positive(self):
        fl = self._make()
        mu = float(fl.liquid_viscosity(200.0, 273.15 + 80))
        assert mu > 0

    def test_surface_tension_less_than_dead(self):
        fl_dead = FluidModel(
            rho_o=density_from_api(30),
            rho_g=gas_density_from_sg(0.65),
            gor=150.0, wlr=0.0,
            oil_model='dead_oil',
        )
        fl_live = self._make()
        p, T = 150.0, 273.15 + 80
        sigma_dead = float(fl_dead.surface_tension(p, T))
        sigma_live = float(fl_live.surface_tension(p, T))
        assert sigma_live < sigma_dead

    def test_surface_tension_positive(self):
        fl = self._make()
        sigma = float(fl.surface_tension(200.0, 273.15 + 80))
        assert sigma > 0

    def test_surface_tension_decreases_with_pressure(self):
        fl = self._make()
        T = 273.15 + 80
        sigma_low = float(fl.surface_tension(50.0, T))
        sigma_high = float(fl.surface_tension(250.0, T))
        assert sigma_high < sigma_low


class TestIdealGasZFactor:

    def test_ideal_gas_returns_one(self):
        fl = FluidModel(ideal_gas=True)
        assert fl.z_factor(200.0, 350.0) == 1.0

    def test_real_gas_near_one_at_low_pressure(self):
        fl = FluidModel(rho_g=gas_density_from_sg(0.65), ideal_gas=False)
        Z = float(fl.z_factor(1.01325, 288.15))
        assert Z == pytest.approx(1.0, abs=0.05)

    def test_real_gas_below_one_at_high_pressure(self):
        fl = FluidModel(rho_g=gas_density_from_sg(0.65), ideal_gas=False)
        Z = float(fl.z_factor(200.0, 350.0))
        assert Z < 0.95

    def test_z_factor_positive(self):
        fl = FluidModel(rho_g=gas_density_from_sg(0.65))
        for p_bar in [10, 100, 250]:
            Z = float(fl.z_factor(p_bar, 350.0))
            assert Z > 0


class TestGasViscosity:

    def test_positive(self):
        fl = FluidModel()
        mu = float(fl.gas_viscosity(273.15 + 60, 50.0))
        assert mu > 0
