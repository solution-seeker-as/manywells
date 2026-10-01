"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Rust core (rust/, built as manywells._core) as the root finder of SSDFSimulator(wp, backend='rust')
(specs/architecture.md, Rust core). It covers wells in the v1.0.0 configuration (manywells.configurations).

The well is converted to the core's inputs once, when the finder is built. A search returns every root the core's
shooting search finds, each a full state with its slope dR/dp_0; the label (SOL-3), the admissibility check (SOL-1)
and the operating point (SOL-4 to SOL-6) are the same functions as for the CasADi backend. The core needs no
initial guess, so x_guess is ignored.
"""

import time

import numpy as np

from manywells import _core
from manywells.choke import SimpsonChokeModel
from manywells.configurations import V1, differences
from manywells.inflow import Vogel
from manywells.solution import Root, RootSet
from manywells.solvers.roots import Attempt, admissible


def core_well(wp) -> '_core.Well':
    """The core's well for WellProperties in the v1.0.0 configuration; raises ValueError for any other well."""
    diff = differences(wp, V1)
    if diff:
        raise ValueError('the Rust core covers wells in the v1.0.0 configuration only: ' + '; '.join(diff))
    geo, fluid, inflow, choke = wp.geometry, wp.fluid, wp.inflow, wp.choke
    vogel = isinstance(inflow, Vogel)
    return _core.Well(L=geo.L, D=geo.D, n_cells=geo.n_cells, rho_l=fluid.rho_l, R_s=fluid.R_s, cp_g=fluid.cp_g,
                      cp_l=fluid.cp_l, f_g=fluid.f_g, f_D=wp.friction.f_D, h=wp.thermal.h,
                      inflow='vogel' if vogel else 'pi', inflow_coefficient=inflow.w_l_max if vogel else inflow.k_l,
                      choke='simpson' if isinstance(choke, SimpsonChokeModel) else 'bernoulli', K_c=choke.K_c,
                      cpr=choke.cpr, profile=choke.chk_profile)


def operating_point(bc) -> tuple:
    """The core's operating point: (p_r, p_s, T_r, T_s, u, w_lg)."""
    return bc.p_r, bc.p_s, bc.T_r, bc.T_s, bc.u, bc.w_lg


class RustRootFinder:
    """The root search of one well by the Rust core, with the well converted once."""

    def __init__(self, wp):
        self.wp = wp
        self.core = core_well(wp)

    def find(self, bc, x_guess=None) -> RootSet:  # spec: SOL-2
        """Every root the core finds, labelled, with the operating point."""
        t0 = time.perf_counter()
        found, marches = self.core.root_set(operating_point(bc))
        roots, rejected = [], 0
        for r in found:
            x = np.asarray(r.x, dtype=float)
            if not admissible(x, bc):
                rejected += 1
                continue
            roots.append(Root(x=x, label='unstable' if r.rising else 'stable', slope=r.slope, choked=r.choked,
                              flow_regime=tuple(r.flow_regime), w_res=r.w_res, w_g_res=r.w_g_res))
        outcome = f'{len(roots)} roots' + (f', {rejected} not admissible' if rejected else '')
        search = [Attempt('shooting', np.nan, outcome, time.perf_counter() - t0, marches)]
        return RootSet.of(roots, search=search)

    def flow_regimes(self, x) -> tuple:
        """The flow-regime label at each point of a state (SLIP-8)."""
        return tuple(self.core.flow_regimes(np.ravel(x).tolist()))

    def rows(self, bc, x) -> tuple:
        """Every row of the core's system at state x, as (IDs, values), in the order of DISC-6."""
        ids, values = self.core.rows(operating_point(bc), np.ravel(x).tolist())
        return tuple(ids), np.asarray(values)
