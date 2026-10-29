"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Rust core (rust/, built as manywells._core) as the root finder of SSDFSimulator(wp, backend='rust')
(specs/architecture.md, Rust core).

The well is converted to the core's inputs once, when the finder is built. A search returns every root the core's
shooting search finds, each a full state with its slope dR/dp_0; the slope's normalization and label (SOL-3), the
admissibility check (SOL-1) and the operating point (SOL-4 to SOL-6) are the same functions as for the CasADi
backend. The core needs no initial guess, so x_guess is ignored.

The core has a closed set of options per model part (specs/architecture.md, decision 1): a component that is not one
of its classes, such as a user's subclass of a component ABC, runs on the CasADi backend only.
"""

import time

import numpy as np

from manywells import _core
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.discretization import System
from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.dead_oil import DeadOil
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable, PVTTableSet
from manywells.slip import SlipModel
from manywells.solution import Root, RootSet
from manywells.solvers.roots import Attempt, admissible, label_of, normalized_slope
from manywells.thermal import ThermalModel

# The classes of each model part that the core implements, exactly: a subclass may override any method
CORE_CLASSES = {
    'geometry': (WellGeometry,),
    'fluid': (FluidModel,),
    'friction': (FixedFrictionFactor, RoughnessFriction),
    'thermal': (ThermalModel,),
    'slip': (SlipModel,),
    'inflow': (Vogel, ProductivityIndex, FixedFlowRate),
    'choke': (SimpsonChokeModel, BernoulliChokeModel),
}
PVT_CLASSES = (DeadOil, BlackOil, PVTTable, PVTTableSet)  # The fluid's PVT models that the core implements, exactly


def not_in_core(wp) -> list:
    """Every part of wp that the core cannot solve, as readable strings (empty if it can solve the well)."""
    out = [
        f"{name}: {type(getattr(wp, name)).__name__} is not one of the core's classes"
        for name, classes in CORE_CLASSES.items()
        if type(getattr(wp, name)) not in classes
    ]
    # The core's void-fraction solve brackets the slip law on [0, 1], which needs C_0 >= 1 and v_inf >= 0 in every
    # regime (manywells._core, march.rs); the five regimes' values are fixed (SLIP-15), the fitted classifier's fields
    slip = wp.slip
    if (
        type(slip) is SlipModel
        and slip.regime_model == 'fitted'
        and not (all(c >= 1 for c in (slip.C_0_annular, slip.C_0_slug, slip.C_0_bubbly)) and slip.v_inf_annular >= 0)
    ):
        out.append(f'slip: the core needs every C_0 >= 1 and v_inf_annular >= 0 ({slip})')
    if type(wp.fluid) is FluidModel and type(wp.fluid.pvt) not in PVT_CLASSES:
        out.append(f"fluid: pvt is {type(wp.fluid.pvt).__name__}, not one of the core's classes")
    # The core takes the gas mass fraction of the reservoir inflow in (0, 1), as INF-4 defines it
    if type(wp.fluid) is FluidModel and not wp.fluid.f_g > 0:
        out.append(f'fluid: the core needs gas in the reservoir inflow, f_g > 0 (gor = {wp.fluid.gor})')
    return out


def core_well(wp) -> '_core.Well':
    """The core's well for WellProperties; raises ValueError for a well the core cannot solve."""
    missing = not_in_core(wp)
    if missing:
        raise ValueError('the Rust core cannot solve this well: ' + '; '.join(missing) + '. Use backend="casadi".')
    geo, fluid, thermal, slip, choke = wp.geometry, wp.fluid, wp.thermal, wp.slip, wp.choke
    return _core.Well(
        md=list(geo.md),
        tvd=list(geo.tvd),
        D=geo.D,
        rho_o=fluid.rho_o,
        rho_g=fluid.rho_g,
        rho_w=fluid.rho_w,
        gor=fluid.gor,
        wlr=fluid.wlr,
        cp_g=fluid.cp_g,
        cp_o=fluid.cp_o,
        cp_w=fluid.cp_w,
        surface_tension_model=fluid.surface_tension_model,
        **_pvt(fluid),
        **_friction(wp.friction),
        h=thermal.h,
        frictional_heating=thermal.frictional_heating,
        gravity_term=thermal.gravity_term,
        lift_gas_mixing=thermal.lift_gas_mixing,
        joule_thomson=thermal.joule_thomson,
        regime_model=slip.regime_model,
        C_0_annular=slip.C_0_annular,
        C_0_slug=slip.C_0_slug,
        C_0_bubbly=slip.C_0_bubbly,
        v_inf_annular=slip.v_inf_annular,
        f_ms=slip.f_ms,
        transition_width=slip.transition_width,
        lambda_floor=slip.lambda_floor,
        profile_scale=slip.profile_scale,
        **_inflow(wp.inflow),
        choke='simpson' if type(choke) is SimpsonChokeModel else 'bernoulli',
        K_c=choke.K_c,
        profile=choke.chk_profile,
    )


def _pvt(fluid) -> dict:
    """The core's fluid options for the fluid's PVT model: its oil model and gas law, and the black oil's separator and
    bubble point (the dead oil's are not used); or its table at the fluid's GOR (PVT-TAB-7), which replaces them."""
    pvt = fluid._hc  # The PVT model at the fluid's GOR, which FluidModel reads once
    if type(pvt) is PVTTable:
        table = {
            'p': list(pvt.p),
            'T': list(pvt.T),
            **{c: np.array(getattr(pvt, c)).ravel(order='F').tolist() for c in pvt.columns},
        }
        return dict(
            oil_model='table',
            ideal_gas=False,
            z_factor_model='dak',
            p_sep=BlackOil.p_sep,
            T_sep=BlackOil.T_sep,
            p_bubble=None,
            table=table,
        )
    black_oil = type(pvt) is BlackOil
    return dict(
        oil_model='black_oil' if black_oil else 'dead_oil',
        ideal_gas=pvt.gas_law == 'ideal',
        z_factor_model='papay' if pvt.gas_law == 'papay' else 'dak',
        p_sep=pvt.p_sep if black_oil else BlackOil.p_sep,
        T_sep=pvt.T_sep if black_oil else BlackOil.T_sep,
        p_bubble=pvt.p_bubble if black_oil else None,
    )


def _inflow(inflow) -> dict:
    if type(inflow) is Vogel:
        return {'inflow': 'vogel', 'inflow_coefficient': inflow.w_l_max}
    if type(inflow) is ProductivityIndex:
        return {'inflow': 'pi', 'inflow_coefficient': inflow.k_l}
    return {'inflow': 'fixed', 'inflow_coefficient': inflow.w_l_const}


def _friction(friction) -> dict:
    if type(friction) is FixedFrictionFactor:
        return {'friction': 'fixed', 'f_D': friction.f_D}
    return {'friction': 'roughness', 'roughness': friction.roughness, 'correlation': friction.correlation}


def operating_point(bc) -> tuple:
    """The core's operating point: the parameters of the CasADi system, (p_r, p_s, T_r, T_s, T_lg, u, w_lg)."""
    return tuple(float(v) for v in System.params(bc))


class RustRootFinder:
    """The root search of one well by the Rust core, with the well converted once."""

    def __init__(self, wp):
        self.wp = wp
        self.core = core_well(wp)
        self.counts = {}  # The work of the last search: marches, states, ... (for the measurements of the feature specs)

    def find(self, bc, x_guess=None) -> RootSet:  # spec: SOL-2
        """Every root the core finds, labelled, with the operating point."""
        t0 = time.perf_counter()
        found, counts = self.core.root_set(operating_point(bc))
        self.counts = counts
        roots, rejected = [], 0
        for r in found:
            x = np.asarray(r.x, dtype=float)
            if not admissible(x, bc):
                rejected += 1
                continue
            slope = normalized_slope(r.slope, x, bc.p_r, bc.p_s, self.wp.geometry.A)
            roots.append(
                Root(
                    x=x,
                    label=label_of(slope),
                    slope=slope,
                    choked=r.choked,
                    flow_regime=tuple(r.flow_regime),
                    w_res=r.w_res,
                    w_g_res=r.w_g_res,
                    off_table=self.wp.fluid.off_table(x),
                )
            )
        unresolved = counts['rejected']
        outcome = (
            f'{len(roots)} roots'
            + (f', {rejected} not admissible' if rejected else '')
            + (f', {unresolved} sign changes of R not accepted' if unresolved else '')
        )
        search = [Attempt('shooting', np.nan, outcome, time.perf_counter() - t0, counts['marches'], found=len(roots))]
        return RootSet.of(roots, search=search)

    def flow_regimes(self, x) -> tuple:
        """The flow-regime label at each point of a state (SLIP-8)."""
        return tuple(self.core.flow_regimes(np.ravel(x).tolist()))

    def rows(self, bc, x) -> tuple:
        """Every row of the core's system at state x, as (IDs, values), in the order of DISC-11."""
        ids, values = self.core.rows(operating_point(bc), np.ravel(x).tolist())
        return tuple(ids), np.asarray(values)

    def march(self, bc, p_0) -> tuple:
        """The core's march from p_0: (x of the points reached, failed, below_separator)."""
        x, failed, below = self.core.march(operating_point(bc), float(p_0))
        return np.asarray(x), failed, below
