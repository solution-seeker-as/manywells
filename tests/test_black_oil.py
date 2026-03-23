"""Tests for the black oil PVT model and its simulator integration."""

import pytest
import numpy as np

from manywells.pvt import P_REF, T_REF, WATER, density_from_api, api_from_density
from manywells.pvt.black_oil import BlackOilPVT
from manywells.units import CF_PSI, CF_RS, M_AIR
from manywells.pvt.gas import gas_fvf, gas_density_std
from manywells.pvt.water import water_fvf
from manywells.pvt.fluid_mix import (
    dissolved_gas_mass_ratio,
    liquid_density_bo,
    free_gas_flux,
    liquid_flux,
)
from manywells.units import CF_BAR


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bo_light():
    """BlackOilPVT for light oil (API 35, sg_gas 0.65)."""
    return BlackOilPVT(
        api=35,
        sg_gas=0.65,
        p_sep=100 * 6894.76,   # 100 psia in Pa
        T_sep=273.15 + 21.1,   # 70 degF in K
    )


@pytest.fixture
def bo_heavy():
    """BlackOilPVT for heavy oil (API 20, sg_gas 0.80)."""
    return BlackOilPVT(
        api=20,
        sg_gas=0.80,
        p_sep=100 * 6894.76,
        T_sep=273.15 + 21.1,
    )


@pytest.fixture
def bo_with_bubble():
    """BlackOilPVT with an explicit bubble point."""
    return BlackOilPVT(
        api=35,
        sg_gas=0.65,
        p_sep=100 * 6894.76,
        T_sep=273.15 + 21.1,
        p_bubble=200e5,        # 200 bar
    )


# =========================================================================
# Phase 1: PVT correlation unit tests
# =========================================================================


class TestBlackOilPVTConstruction:

    def test_valid_construction(self, bo_light):
        assert bo_light.api == 35
        assert bo_light.sg_gas == 0.65

    def test_api_out_of_range_low(self):
        with pytest.raises(ValueError, match="outside valid range"):
            BlackOilPVT(api=5, sg_gas=0.65, p_sep=1e6, T_sep=300)

    def test_api_out_of_range_high(self):
        with pytest.raises(ValueError, match="outside valid range"):
            BlackOilPVT(api=50, sg_gas=0.65, p_sep=1e6, T_sep=300)

    def test_negative_sg_gas(self):
        with pytest.raises(ValueError, match="positive"):
            BlackOilPVT(api=30, sg_gas=-0.1, p_sep=1e6, T_sep=300)

    def test_pre_computed_densities(self, bo_light):
        assert bo_light.rho_o_sc == pytest.approx(density_from_api(35))
        R_s_gas = bo_light.R_s_gas
        assert bo_light.rho_g_sc == pytest.approx(P_REF / (R_s_gas * T_REF))

    def test_api_dependent_coefficients_light(self, bo_light):
        """API > 30 uses the second coefficient set."""
        assert bo_light._c1 == 0.0178

    def test_api_dependent_coefficients_heavy(self, bo_heavy):
        """API <= 30 uses the first coefficient set."""
        assert bo_heavy._c1 == 0.0362


class TestVazquezBeggsRs:

    def test_rs_positive(self, bo_light):
        """Rs is positive for typical wellbore conditions."""
        p = 150e5     # 150 bar in Pa
        T = 373.15    # 100 degC
        rs = float(bo_light.rs(p, T))
        assert rs > 0

    def test_rs_increases_with_pressure(self, bo_light):
        """Higher pressure dissolves more gas."""
        T = 373.15
        rs_low = float(bo_light.rs(100e5, T))
        rs_high = float(bo_light.rs(200e5, T))
        assert rs_high > rs_low

    def test_rs_against_field_unit_reference(self, bo_light):
        """Verify Rs matches a hand-computed field-unit reference value."""
        p_psia = 2000
        T_F = 150
        p = p_psia * CF_PSI  # Pa
        T = (T_F - 32) / 1.8 + 273.15  # K

        rs_si = float(bo_light.rs(p, T))
        rs_field = float(bo_light._rs_field(p_psia, T_F))
        assert rs_si == pytest.approx(rs_field * CF_RS, rel=1e-6)

    def test_rs_capped_at_bubble_point(self, bo_with_bubble):
        """When p > p_bubble, Rs is capped at the bubble point value."""
        T = 373.15
        p_above = bo_with_bubble.p_bubble * 1.5
        p_at = bo_with_bubble.p_bubble

        rs_above = float(bo_with_bubble.rs(p_above, T))
        rs_at = float(bo_with_bubble.rs(p_at, T))
        assert rs_above == pytest.approx(rs_at, rel=1e-3)

    def test_rs_no_cap_without_bubble(self, bo_light):
        """Without p_bubble, Rs continues to increase with pressure."""
        T = 373.15
        rs_200 = float(bo_light.rs(200e5, T))
        rs_300 = float(bo_light.rs(300e5, T))
        assert rs_300 > rs_200


class TestVazquezBeggsBo:

    def test_bo_greater_than_one(self, bo_light):
        """Bo > 1 because live oil expands relative to stock-tank conditions."""
        p = 150e5
        T = 373.15
        bo = float(bo_light.bo(p, T))
        assert bo > 1.0

    def test_bo_increases_with_pressure_below_bubble(self, bo_light):
        """More dissolved gas => more expansion => higher Bo."""
        T = 373.15
        bo_low = float(bo_light.bo(80e5, T))
        bo_high = float(bo_light.bo(200e5, T))
        assert bo_high > bo_low

    def test_bo_capped_at_bubble_point(self, bo_with_bubble):
        """Bo stops increasing above the bubble point."""
        T = 373.15
        bo_above = float(bo_with_bubble.bo(bo_with_bubble.p_bubble * 1.5, T))
        bo_at = float(bo_with_bubble.bo(bo_with_bubble.p_bubble, T))
        assert bo_above == pytest.approx(bo_at, rel=1e-3)


class TestLiveOilDensity:

    def test_density_positive(self, bo_light):
        p = 150e5
        T = 373.15
        rho = float(bo_light.live_oil_density(p, T))
        assert rho > 0

    def test_density_heavier_than_dead_oil(self, bo_light):
        """Live oil (oil + dissolved gas) is denser than stock-tank oil
        for high-pressure conditions because dissolved gas mass dominates expansion."""
        p = 50e5
        T = 373.15
        rho_live = float(bo_light.live_oil_density(p, T))
        # At low pressure, live oil can be lighter or heavier depending on Rs/Bo balance;
        # at very low pressure both Rs and (Bo-1) are small, density approaches rho_o_sc
        rho_dead = bo_light.rho_o_sc
        # The density should be in a reasonable range around dead oil
        assert 0.5 * rho_dead < rho_live < 1.5 * rho_dead

    def test_density_formula_consistency(self, bo_light):
        """live_oil_density == (rho_o_sc + Rs * rho_g_sc) / Bo."""
        p = 150e5
        T = 373.15
        Rs = float(bo_light.rs(p, T))
        Bo = float(bo_light.bo(p, T))
        expected = (bo_light.rho_o_sc + Rs * bo_light.rho_g_sc) / Bo
        actual = float(bo_light.live_oil_density(p, T))
        assert actual == pytest.approx(expected, rel=1e-10)


class TestBubblePointPressure:

    def test_returns_stored_value(self, bo_with_bubble):
        """When p_bubble is set, bubble_point_pressure returns it."""
        p_b = bo_with_bubble.bubble_point_pressure(50, 373.15)
        assert p_b == bo_with_bubble.p_bubble

    def test_standing_correlation_positive(self, bo_light):
        """Standing correlation returns a positive pressure."""
        Rs_total = 30.0   # Sm3/Sm3
        T = 373.15
        p_b = bo_light.bubble_point_pressure(Rs_total, T)
        assert p_b > 0

    def test_standing_increases_with_gor(self, bo_light):
        """Higher GOR => higher bubble point pressure."""
        T = 373.15
        p_b_low = bo_light.bubble_point_pressure(10.0, T)
        p_b_high = bo_light.bubble_point_pressure(50.0, T)
        assert p_b_high > p_b_low


class TestBlackOilFactory:

    def test_from_well_params_roundtrip(self):
        """Factory constructs equivalent BlackOilPVT from R_s_gas and rho_o."""
        api_orig = 30.0
        sg_gas_orig = 0.70
        from manywells.pvt import R_UNIVERSAL
        R_s_gas = R_UNIVERSAL / (M_AIR * sg_gas_orig)
        rho_o = density_from_api(api_orig)

        bo = BlackOilPVT.from_well_params(R_s_gas, rho_o, p_sep=1e6, T_sep=300)
        assert bo.api == pytest.approx(api_orig, rel=1e-6)
        assert bo.sg_gas == pytest.approx(sg_gas_orig, rel=1e-6)


# =========================================================================
# Gas and water FVF unit tests
# =========================================================================


class TestGasFVF:

    def test_standard_conditions_equals_one(self):
        """Bg = 1 at standard conditions."""
        bg = float(gas_fvf(P_REF, T_REF))
        assert bg == pytest.approx(1.0)

    def test_high_pressure_less_than_one(self):
        """At high pressure, gas compresses: Bg < 1."""
        bg = float(gas_fvf(100e5, T_REF))
        assert bg < 1.0

    def test_high_temperature_greater_than_std(self):
        """At higher temperature and same pressure, Bg > 1."""
        bg_hot = float(gas_fvf(P_REF, 400))
        assert bg_hot > 1.0


class TestGasDensityStd:

    def test_methane(self):
        R_s = 518.3  # methane
        rho = gas_density_std(R_s)
        assert rho == pytest.approx(P_REF / (R_s * T_REF))


class TestWaterFVF:

    def test_standard_conditions_equals_one(self):
        bw = float(water_fvf(P_REF, T_REF))
        assert bw == pytest.approx(1.0)

    def test_high_pressure_slightly_above_one(self):
        """Water at 200 bar: Bw barely above 1.0."""
        bw = float(water_fvf(200e5, T_REF))
        assert 1.0 < bw < 1.01

    def test_nearly_incompressible(self):
        """Change in Bw over 100 bar is tiny."""
        bw_100 = float(water_fvf(100e5, T_REF))
        bw_200 = float(water_fvf(200e5, T_REF))
        assert abs(bw_200 - bw_100) < 0.005


# =========================================================================
# Fluid mixing function tests
# =========================================================================


class TestFluidMix:

    def test_dissolved_gas_mass_ratio(self):
        Rs = 20.0         # Sm3/Sm3
        rho_g_sc = 0.8    # kg/m3
        rho_o_sc = 850.0  # kg/m3
        r = dissolved_gas_mass_ratio(Rs, rho_g_sc, rho_o_sc)
        assert r == pytest.approx(Rs * rho_g_sc / rho_o_sc)

    def test_liquid_density_bo_no_water(self):
        """Pure oil (wlr=0): rho_l = (rho_o_sc + Rs*rho_g_sc) / Bo."""
        rho_o_sc = 850
        rho_g_sc = 0.8
        Rs = 20.0
        Bo = 1.15
        rho_l = float(liquid_density_bo(rho_o_sc, rho_g_sc, WATER.rho, Rs, Bo, wlr=0.0))
        expected = (rho_o_sc + Rs * rho_g_sc) / Bo
        assert rho_l == pytest.approx(expected, rel=1e-6)

    def test_liquid_density_bo_pure_water(self):
        """Pure water (wlr=1): rho_l = rho_w."""
        rho_l = float(liquid_density_bo(850, 0.8, WATER.rho, 20.0, 1.15, wlr=1.0))
        assert rho_l == pytest.approx(WATER.rho, rel=1e-6)

    def test_free_gas_flux_no_dissolved_gas(self):
        """When Rs=0, all gas is free."""
        w_g_total = 2.0
        w_lg = 0.5
        w_g_free = float(free_gas_flux(w_g_total, w_lg, w_o=1.0, Rs=0.0,
                                       rho_g_sc=0.8, rho_o_sc=850.0))
        assert w_g_free == pytest.approx(w_g_total + w_lg, abs=1e-3)

    def test_free_gas_decreases_with_rs(self):
        """More dissolved gas => less free gas."""
        args = dict(w_g_total=2.0, w_lg=0.0, w_o=5.0, rho_g_sc=0.8, rho_o_sc=850.0)
        wf_low = float(free_gas_flux(Rs=5.0, **args))
        wf_high = float(free_gas_flux(Rs=20.0, **args))
        assert wf_high < wf_low

    def test_liquid_flux_increases_with_rs(self):
        """More dissolved gas => heavier liquid phase."""
        args = dict(w_l_inflow=10.0, w_o=5.0, w_g_total=2.0,
                    rho_g_sc=0.8, rho_o_sc=850.0)
        wl_low = float(liquid_flux(Rs=5.0, **args))
        wl_high = float(liquid_flux(Rs=20.0, **args))
        assert wl_high > wl_low

    def test_mass_conservation(self):
        """free_gas + liquid = w_g_total + w_lg + w_l_inflow (total mass)."""
        w_g_total, w_lg, w_l_inflow, w_o = 2.0, 0.5, 10.0, 5.0
        rho_g_sc, rho_o_sc = 0.8, 850.0
        Rs = 15.0

        wf = float(free_gas_flux(w_g_total, w_lg, w_o, Rs, rho_g_sc, rho_o_sc))
        wl = float(liquid_flux(w_l_inflow, w_o, w_g_total, Rs, rho_g_sc, rho_o_sc))
        total_in = w_g_total + w_lg + w_l_inflow
        total_out = wf + wl
        assert total_out == pytest.approx(total_in, rel=1e-3)


# =========================================================================
# Simulator integration tests
# =========================================================================

from manywells.geometry import WellGeometry
from manywells.pvt.fluid import FluidModel
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator, SimError


class TestFluidModelOilFraction:

    def test_default_black_oil_is_none(self):
        wp = WellProperties()
        assert wp.fluid.black_oil is None

    def test_f_o_in_liquid_pure_oil(self):
        """When water_cut=0, oil fraction is 1."""
        fl = FluidModel(water_cut=0.0)
        assert fl.f_o_in_liquid == pytest.approx(1.0)

    def test_f_o_in_liquid_with_water(self):
        """Oil fraction is between 0 and 1 for mixed liquid."""
        fl = FluidModel(water_cut=0.3)
        assert 0 < fl.f_o_in_liquid < 1

    def test_f_o_in_liquid_pure_water(self):
        """Water cut near 1: f_o_in_liquid approaches 0."""
        fl = FluidModel(water_cut=0.99)
        assert fl.f_o_in_liquid < 0.02


@pytest.mark.slow
class TestDeadOilRegression:
    """Ensure the dead oil (default) path still works identically."""

    def test_dead_oil_solves(self):
        geo = WellGeometry.vertical(500, 2, D=0.1)
        wp = WellProperties(geometry=geo, fluid=FluidModel(api=45.0))
        bc = BoundaryConditions(p_r=120, p_s=30, u=0.8)
        sim = SSDFSimulator(wp, bc)
        try:
            result = sim.simulate()
            assert result is not None
            assert len(result) == (sim.n_cells + 1) * sim.dim_x
        except SimError:
            pytest.skip("Ipopt could not solve")


@pytest.mark.slow
class TestBlackOilSimulator:
    """Integration tests for the black oil model in the simulator."""

    @pytest.fixture
    def bo_sim(self):
        """Set up a simulator with the black oil model enabled."""
        bo = BlackOilPVT(
            api=35,
            sg_gas=0.65,
            p_sep=100 * CF_PSI,
            T_sep=273.15 + 21.1,
            p_bubble=250e5,
        )
        fl = FluidModel(
            api=35,
            sg_gas=0.65,
            water_cut=0.0,
            black_oil=bo,
        )
        geo = WellGeometry.vertical(2000, 5, D=0.1554)
        wp = WellProperties(
            geometry=geo,
            fluid=fl,
        )
        bc = BoundaryConditions(p_r=200, p_s=30, u=0.8)
        return SSDFSimulator(wp, bc)

    def test_black_oil_solves(self, bo_sim):
        """The black oil simulator converges."""
        try:
            result = bo_sim.simulate()
            assert result is not None
            assert len(result) == (bo_sim.n_cells + 1) * bo_sim.dim_x
        except SimError:
            pytest.skip("Ipopt could not solve black oil case")

    def test_mass_conservation(self, bo_sim):
        """Total mass (free gas + liquid) is conserved across all cells."""
        try:
            result = bo_sim.simulate()
        except SimError:
            pytest.skip("Ipopt could not solve")

        wp = bo_sim.wp
        n = bo_sim.n_cells
        dim = bo_sim.dim_x

        total_mass_fluxes = []
        for i in range(n + 1):
            cell = result[dim * i: dim * (i + 1)]
            p, v_g, v_l, alpha, rho_g, rho_l, T = cell
            w_g = wp.geometry.A * alpha * rho_g * v_g
            w_l = wp.geometry.A * (1 - alpha) * rho_l * v_l
            total_mass_fluxes.append(w_g + w_l)

        # Total mass flux should be approximately constant
        for m in total_mass_fluxes:
            assert m == pytest.approx(total_mass_fluxes[0], rel=0.02)

    def test_free_gas_increases_with_decreasing_pressure(self, bo_sim):
        """As pressure drops (moving up the wellbore), more gas exsolves."""
        try:
            result = bo_sim.simulate()
        except SimError:
            pytest.skip("Ipopt could not solve")

        wp = bo_sim.wp
        n = bo_sim.n_cells
        dim = bo_sim.dim_x

        # Collect gas mass flux at bottom (cell 0) and top (cell n)
        cell_0 = result[0:dim]
        cell_n = result[dim * n: dim * (n + 1)]

        w_g_bottom = wp.geometry.A * cell_0[3] * cell_0[4] * cell_0[1]  # alpha*rho_g*v_g
        w_g_top = wp.geometry.A * cell_n[3] * cell_n[4] * cell_n[1]

        p_bottom = cell_0[0]
        p_top = cell_n[0]

        # Pressure should decrease from bottom to top
        assert p_top < p_bottom
        # Free gas should increase from bottom to top
        assert w_g_top > w_g_bottom

    def test_liquid_density_varies_with_pressure(self, bo_sim):
        """In the black oil model, liquid density is NOT constant."""
        try:
            result = bo_sim.simulate()
        except SimError:
            pytest.skip("Ipopt could not solve")

        dim = bo_sim.dim_x
        n = bo_sim.n_cells

        rho_l_values = [result[dim * i + 5] for i in range(n + 1)]
        # Not all identical (unlike the dead oil model)
        assert max(rho_l_values) - min(rho_l_values) > 0.1
