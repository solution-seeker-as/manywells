"""
Black-oil PVT consistency checks following Singh et al. (SPE-109596).

Reference: K. Singh, Oi. Fevang, and C. H. Whitson, "Consistent Black-Oil
PVT Table Modification", SPE Annual Technical Conference and Exhibition,
paper SPE-109596-MS, 2007.

See also: https://wiki.whitson.com/bopvt/extrapolation/#consistency-check-for-black-oil-modeling

This codebase implements a dead-gas black-oil model (r_s = 0), so the checks
involving vaporized oil ratio are either trivially satisfied or reduce to
simpler forms.
"""

import numpy as np
import pytest

from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.fluid import FluidModel
from manywells.pvt.gas import gas_fvf, gas_density, sutton_pseudo_critical
from manywells.units import CF_BAR


# ---------------------------------------------------------------------------
# Fluid configurations spanning the valid API 10-40 range
# ---------------------------------------------------------------------------

FLUID_PARAMS = [
    pytest.param(
        dict(api=15, sg_gas=0.70, gor=50, p_bubble=150e5),
        id="heavy-oil-low-gor",
    ),
    pytest.param(
        dict(api=25, sg_gas=0.65, gor=150, p_bubble=200e5),
        id="medium-oil",
    ),
    pytest.param(
        dict(api=35, sg_gas=0.60, gor=300, p_bubble=300e5),
        id="light-oil-high-gor",
    ),
    pytest.param(
        dict(api=40, sg_gas=0.80, gor=200, p_bubble=250e5),
        id="light-oil-heavy-gas",
    ),
]


def _make_fluid(params):
    return FluidModel(
        rho_o=density_from_api(params["api"]),
        rho_g=gas_density_from_sg(params["sg_gas"]),
        gor=params["gor"],
        wlr=0.0,
        oil_model="black_oil",
        ideal_gas=False,
        p_bubble=params["p_bubble"],
    )


P_GRID = np.linspace(10, 400, 40)    # bar
T_GRID = np.linspace(300, 420, 7)     # K

# Papay Z-factor is documented for p_pr < 6; use conservative bound.
_PPR_MAX = 5.0


def _max_valid_pressure(sg_gas):
    """Max pressure (bar) where the Papay Z-factor correlation is reliable."""
    ppc, _ = sutton_pseudo_critical(sg_gas)      # Pa
    return _PPR_MAX * ppc / CF_BAR


# ===================================================================
# Check 1 — Basic physical consistency
# ===================================================================

@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_gas_density_less_than_oil_density(params):
    """rho_g < rho_o at all (p, T) in the wellbore operating window."""
    fl = _make_fluid(params)

    for T in T_GRID:
        for p_bar in P_GRID:
            p_pa = p_bar * CF_BAR
            Z = float(fl.z_factor(p_bar, T))
            rho_g = float(gas_density(fl.R_s, p_pa, T, Z))
            rho_o = float(fl.liquid_density(p_bar, T))
            assert rho_g < rho_o, (
                f"rho_g ({rho_g:.2f}) >= rho_o ({rho_o:.2f}) "
                f"at p={p_bar:.0f} bar, T={T:.0f} K"
            )


@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_gas_viscosity_less_than_oil_viscosity(params):
    """mu_g < mu_o at all (p, T) in the wellbore operating window."""
    fl = _make_fluid(params)

    for T in T_GRID:
        for p_bar in P_GRID:
            p_pa = p_bar * CF_BAR
            Z = float(fl.z_factor(p_bar, T))
            rho_g = float(gas_density(fl.R_s, p_pa, T, Z))
            mu_g = float(fl.gas_viscosity(T, rho_g))
            mu_o = float(fl.liquid_viscosity(p_bar, T))
            assert mu_g < mu_o, (
                f"mu_g ({mu_g:.3e}) >= mu_o ({mu_o:.3e}) "
                f"at p={p_bar:.0f} bar, T={T:.0f} K"
            )


# ===================================================================
# Check 2 — Compressibility at saturated conditions (r_s = 0)
# ===================================================================

@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_gas_fvf_decreases_with_pressure(params):
    """dB_gd/dp < 0: gas FVF must decrease with increasing pressure.

    Restricted to p_pr < 5 where the Papay Z-factor correlation is reliable.
    """
    fl = _make_fluid(params)
    dp = 0.1  # bar
    p_max = _max_valid_pressure(params["sg_gas"])
    p_valid = P_GRID[P_GRID <= p_max]

    for T in T_GRID:
        for p_bar in p_valid:
            p_lo, p_hi = p_bar - dp / 2, p_bar + dp / 2
            Z_lo = float(fl.z_factor(p_lo, T))
            Z_hi = float(fl.z_factor(p_hi, T))
            bgd_lo = float(gas_fvf(p_lo * CF_BAR, T, Z_lo))
            bgd_hi = float(gas_fvf(p_hi * CF_BAR, T, Z_hi))
            dBgd_dp = (bgd_hi - bgd_lo) / dp
            assert dBgd_dp < 0, (
                f"dBgd/dp ({dBgd_dp:.3e}) >= 0 at p={p_bar:.0f} bar, T={T:.0f} K"
            )


@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_oil_compressibility_bounded_by_gas_dissolving(params):
    """B_gd * dR_s/dp > dB_o/dp below the bubble point (r_s = 0 simplification)."""
    fl = _make_fluid(params)
    p_bub_bar = params["p_bubble"] / CF_BAR
    dp = 0.1  # bar

    p_saturated = P_GRID[P_GRID < p_bub_bar - dp]

    for T in T_GRID:
        for p_bar in p_saturated:
            p_lo, p_hi = p_bar - dp / 2, p_bar + dp / 2

            rs_lo = float(fl.rs(p_lo, T))
            rs_hi = float(fl.rs(p_hi, T))
            dRs_dp = (rs_hi - rs_lo) / dp

            bo_lo = float(fl.bo(p_lo, T))
            bo_hi = float(fl.bo(p_hi, T))
            dBo_dp = (bo_hi - bo_lo) / dp

            Z = float(fl.z_factor(p_bar, T))
            bgd = float(gas_fvf(p_bar * CF_BAR, T, Z))

            assert bgd * dRs_dp > dBo_dp, (
                f"Bgd*dRs/dp ({bgd * dRs_dp:.3e}) <= dBo/dp ({dBo_dp:.3e}) "
                f"at p={p_bar:.0f} bar, T={T:.0f} K"
            )


# ===================================================================
# Additional physical sanity checks on Rs and Bo
# ===================================================================

@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_rs_non_negative(params):
    """Rs >= 0 everywhere."""
    fl = _make_fluid(params)

    for T in T_GRID:
        for p_bar in P_GRID:
            rs = float(fl.rs(p_bar, T))
            assert rs >= 0, f"Rs ({rs}) < 0 at p={p_bar:.0f} bar, T={T:.0f} K"


@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_bo_at_least_one(params):
    """Bo >= 1 everywhere (live oil swells relative to stock-tank oil)."""
    fl = _make_fluid(params)

    for T in T_GRID:
        for p_bar in P_GRID:
            bo = float(fl.bo(p_bar, T))
            assert bo >= 1.0, f"Bo ({bo}) < 1 at p={p_bar:.0f} bar, T={T:.0f} K"


@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_rs_non_decreasing_below_bubble_point(params):
    """Rs must be non-decreasing with pressure below the bubble point."""
    fl = _make_fluid(params)
    p_bub_bar = params["p_bubble"] / CF_BAR
    p_below = np.sort(P_GRID[P_GRID < p_bub_bar])

    for T in T_GRID:
        rs_prev = float(fl.rs(p_below[0], T))
        for p_bar in p_below[1:]:
            rs_curr = float(fl.rs(p_bar, T))
            assert rs_curr >= rs_prev - 1e-12, (
                f"Rs decreased from {rs_prev:.4f} to {rs_curr:.4f} "
                f"at p={p_bar:.0f} bar, T={T:.0f} K"
            )
            rs_prev = rs_curr


@pytest.mark.parametrize("params", FLUID_PARAMS)
def test_rs_constant_above_bubble_point(params):
    """Rs must be capped (constant) above the bubble point."""
    fl = _make_fluid(params)
    p_bub_bar = params["p_bubble"] / CF_BAR
    p_above = P_GRID[P_GRID > p_bub_bar]
    if len(p_above) < 2:
        pytest.skip("Not enough pressure points above bubble point")

    for T in T_GRID:
        rs_at_pb = float(fl.rs(p_bub_bar, T))
        for p_bar in p_above:
            rs = float(fl.rs(p_bar, T))
            assert rs == pytest.approx(rs_at_pb, rel=1e-3), (
                f"Rs ({rs:.4f}) != Rs at Pb ({rs_at_pb:.4f}) "
                f"at p={p_bar:.0f} bar (above bubble point), T={T:.0f} K"
            )
