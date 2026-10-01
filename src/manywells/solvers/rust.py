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
from manywells.pvt.fluid import FluidModel
from manywells.slip import SlipModel
from manywells.solution import Root, RootSet
from manywells.solvers.roots import Attempt, admissible, label_of, normalized_slope
from manywells.thermal import ThermalModel

# The classes of each model part that the core implements, exactly: a subclass may override any method
CORE_CLASSES = {'geometry': (WellGeometry,), 'fluid': (FluidModel,), 'friction': (FixedFrictionFactor, RoughnessFriction),
                'thermal': (ThermalModel,), 'slip': (SlipModel,), 'inflow': (Vogel, ProductivityIndex, FixedFlowRate),
                'choke': (SimpsonChokeModel, BernoulliChokeModel)}


def _not_yet_ported(wp) -> list:
    """The options of wp that the core does not implement yet. Step 9 ports them one feature spec at a time."""
    geo, fluid, thermal, out = wp.geometry, wp.fluid, wp.thermal, []
    if not np.allclose(geo.delta_md, geo.L / geo.n_cells, rtol=1e-9, atol=0):
        out.append('a non-uniform grid (001)')
    if not np.allclose(geo.cos_incl, 1.0, rtol=0, atol=1e-12):
        out.append('an inclined well (001, 002)')
    if wp.slip != SlipModel():
        out.append('slip parameters other than the defaults (002)')
    if thermal.frictional_heating or thermal.gravity_term:
        out.append('frictional heating or the gravity term (009)')
    if not fluid.ideal_gas:
        out.append('a real gas (006)')
    if fluid.oil_model != 'dead_oil':
        out.append('black oil (007, 008)')
    if fluid.surface_tension_model != 'liquid':
        out.append("surface_tension_model='oil' (003)")
    if type(wp.friction) is not FixedFrictionFactor:
        out.append('friction from roughness (004)')
    if thermal.lift_gas_mixing:
        out.append('lift-gas mixing (010)')
    if type(wp.inflow) is FixedFlowRate:
        out.append('a fixed liquid rate (011)')
    return out


def not_in_core(wp) -> list:
    """Every part of wp that the core cannot solve, as readable strings (empty if it can solve the well)."""
    out = [f'{name}: {type(getattr(wp, name)).__name__} is not one of the core\'s classes'
           for name, classes in CORE_CLASSES.items() if type(getattr(wp, name)) not in classes]
    if out:
        return out
    # The core's void-fraction solve brackets the slip law on [0, 1], which needs C_0 >= 1 and v_inf >= 0 in every
    # regime (manywells._core, march.rs)
    slip = wp.slip
    if min(slip.C_0_annular, slip.C_0_slug, slip.C_0_bubbly) < 1 or slip.v_inf_annular < 0:
        out.append(f'slip: the core needs every C_0 >= 1 and v_inf_annular >= 0 ({slip})')
    return out + _not_yet_ported(wp)


def core_well(wp) -> '_core.Well':
    """The core's well for WellProperties; raises ValueError for a well the core cannot solve."""
    missing = not_in_core(wp)
    if missing:
        raise ValueError('the Rust core cannot solve this well: ' + '; '.join(missing) + '. Use backend="casadi".')
    geo, fluid, inflow, choke = wp.geometry, wp.fluid, wp.inflow, wp.choke
    vogel = isinstance(inflow, Vogel)
    return _core.Well(L=geo.L, D=geo.D, n_cells=geo.n_cells,
                      rho_o=fluid.rho_o, rho_g=fluid.rho_g, rho_w=fluid.rho_w, gor=fluid.gor, wlr=fluid.wlr,
                      cp_g=fluid.cp_g, cp_o=fluid.cp_o, cp_w=fluid.cp_w, f_D=wp.friction.f_D, h=wp.thermal.h,
                      inflow='vogel' if vogel else 'pi', inflow_coefficient=inflow.w_l_max if vogel else inflow.k_l,
                      choke='simpson' if isinstance(choke, SimpsonChokeModel) else 'bernoulli', K_c=choke.K_c,
                      profile=choke.chk_profile)


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
            label = label_of(slope) if np.isfinite(slope) else 'indeterminate'
            roots.append(Root(x=x, label=label, slope=slope, choked=r.choked, flow_regime=tuple(r.flow_regime),
                              w_res=r.w_res, w_g_res=r.w_g_res))
        unresolved = counts['rejected']
        outcome = (f'{len(roots)} roots' + (f', {rejected} not admissible' if rejected else '')
                   + (f', {unresolved} sign changes of R not accepted' if unresolved else ''))
        search = [Attempt('shooting', np.nan, outcome, time.perf_counter() - t0, counts['marches'])]
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
