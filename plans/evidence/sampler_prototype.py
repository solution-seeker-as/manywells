"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 2 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The prototype of the sampler redesign (Step R6 of plans/sampler-redesign-plan.md). It draws wells and their samples
by the draft procedure (plans/sampler-redesign/sampling-draft.md: R3's well draws from sampler_draft_draws.py, R4's
sample draws from sampler_draft_samples.py), maps each well to v2.0.0 and checks it, solves every attempt with the
Rust core, applies the producibility filter, the nsol-2 life scan and the redraw rules with their limits (D-50, D-53,
D-55, D-56, D-80 to D-84, D-101, D-102), and writes the tables of plans/sampler-redesign/datasets-draft.md as parquet,
with the run's own log of every attempt (attempts.parquet). Since the checkpoint's rulings of 2026-11-06 the draws are
the rework's, and the core's solver failures are redrawn without a CasADi retry (H1: D-54 dropped); R6's revisions
('draft' and 'r6', with the retry) are this script at commit 8e6c175.

Interim mapping (sampling-draft.md, Mapping): volatile oils and gas condensates are built as the black oil with their
API gravity, gas gravity, separator and GOR, and gas and condensate wells flow in on a productivity index whose k_l
gives the liquid the gas carries at the design drawdown. Every result on those classes depends on it.

From the repository root, with the sandbox off (the pool needs Unix sockets), each process capped:

    (ulimit -v 2600000; OMP_NUM_THREADS=1 timeout 4500 .venv/bin/python plans/evidence/sampler_prototype.py \\
        run sol 400 OUT/sol-2-r6 --seed 2026 --processes 11)
    ... run nsol 100 OUT/nsol-2-r6 --seed 2027 --processes 11
    ... assemble sol 400 OUT/sol-2-r6 --seed 2026    the tables again from OUT/raw/, without solving

The measurements are sampler_prototype_report.py's (plans/sampler-redesign/r6-measurements.md).

    run sol|nsol N OUT   draw wells k = 0, 1, 2, ... until N are accepted; one task per well; a well's raw result is
                         kept in OUT/raw/, so that a stopped run resumes; then the tables are assembled in OUT
    --wells K0 K1        solve only the draw indices K0 to K1 - 1 (a speed and memory check), without assembling
"""

import argparse
import json
import math
import multiprocessing
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from sampler_draft_draws import OILS, draw_well, forchheimer, gas_rate, j_radial
from sampler_draft_samples import (
    N_OP,
    N_SAMPLES,
    N_SWEEP,
    N_TESTS,
    P_CHANGE,
    P_LIFT_OFF,
    P_OPEN,
    P_OPEN_SET,
    P_SMALL,
    PR_DRIFT,
    RATIO_DRIFT,
    STEP,
    U_MIN,
    WLR_MAX,
    Life,
    logu,
    producing_gor,
    reflect,
    separator_pressure,
)

from manywells.choke import SimpsonChokeModel
from manywells.configurations import V2, V2_SLIP, check
from manywells.datasets.rows import root_features
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.sampling.wells import rng_for
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.thermal import ThermalModel

pa.set_cpu_count(2)  # Arrow's thread pools reserve address space that the processes' memory cap does not allow
pa.set_io_thread_count(2)

N_CELLS = 100
DAY = 86400.0
FLOOR = {'platform': 1.6, 'onshore': 1.6, 'subsea': 20.0}  # Sm3/d of oil equivalent, D-34 as ruled
REVISION = 'rework'  # the checkpoint's rulings of 2026-11-06; R6's revisions are this script at commit 8e6c175
FLOOR_MARGIN = 0.5  # A well is dropped below half its floor, D-34
WLR_NOT_PRODUCED = 0.98  # An oil well above this water cut is not produced, D-50
BEST_RATE_LIFT = (0.0, 0.5, 1.0)  # Lift-gas rates of the best rate, fractions of w_lg_max, D-50
N_SCAN = 21  # Points of the nsol-2 life scan, every 5% of T_life, D-50
SLOT_LIMIT, REDRAW_LIMIT, SOLVER_FAILURE_LIMIT = 50, 400, 10  # D-81
WELL_TIME_LIMIT = 3600.0  # s: a guard of the prototype against a run-away well, not a rule of the draft
TUBING_NAMES = {0.0620: '2-7/8', 0.0760: '3-1/2', 0.1005: '4-1/2', 0.1243: '5-1/2', 0.1571: '7', 0.2205: '9-5/8'}
FEATURES = (
    'PBH PWH TBH TWH WGAS WLIQ WOIL WWAT WTOT QGL QGAS QLIQ QOIL QWAT QTOT FGAS FOIL FWAT CHOKED FRBH FRWH'.split()
)


# --- Building and solving one attempt ----------------------------------------------------------------------------------
class _TimedCore:
    """The core's well with its root_set timed, so that the time outside the core can be measured."""

    def __init__(self, core):
        self.core, self.seconds = core, 0.0

    def root_set(self, op):
        t0 = time.perf_counter()
        try:
            return self.core.root_set(op)
        finally:
            self.seconds += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self.core, name)


class Builder:
    """A drawn well's fixed parts (the survey's grid, the PVT model, the choke) and the WellProperties of an attempt,
    in the interim mapping. design is the lift design (a well's draw dict with md_inj and T_lg) or None."""

    def __init__(self, w, stations, design=None):
        self.w, self.design = w, design
        self.geometry = WellGeometry.from_survey(stations[0], stations[1], N_CELLS, D=w['D'])
        self.pvt = BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])
        self.choke = SimpsonChokeModel(K_c=w['K_c'], chk_profile=w['choke_profile'])

    def fluid(self, gor, wlr):
        w = self.w
        return FluidModel(
            pvt=self.pvt, rho_w=w['rho_w'], gor=gor, wlr=wlr, cp_g=w['cp_g'], cp_o=w['cp_o'], cp_w=w['cp_w']
        )

    def case(self, inp):
        """The WellProperties and BoundaryConditions of an attempt's inputs."""
        w, lift = self.w, bool(inp['lift_on'])
        if not math.isnan(inp.get('w_l_max', math.nan)):
            inflow = Vogel(w_l_max=inp['w_l_max'])
        else:
            inflow = ProductivityIndex(k_l=inp['k_l'])
        wp = WellProperties(
            geometry=self.geometry,
            fluid=self.fluid(inp['gor'], inp['wlr']),
            friction=RoughnessFriction(roughness=inp['roughness']),
            thermal=ThermalModel(h=inp['h']),
            slip=V2_SLIP,
            inflow=inflow,
            choke=self.choke,
            injection_md=self.design['md_inj'] if lift else None,
        )
        bc = BoundaryConditions(
            p_r=inp['p_r'],
            p_s=inp['p_s'],
            T_r=w['T_r'],
            T_s=w['T_s'],
            T_lg=self.design['T_lg'] if lift else None,
            u=inp['u'],
            w_lg=inp['w_lg'] if lift else 0.0,
        )
        return wp, bc


def features(root, bc, fluid):
    """The sample's operating point as the samples table's columns (datasets-draft.md; SMP-30's rates)."""
    x_w = 1 - fluid.f_o_in_liquid
    row = root_features(root, bc, fluid.rho_o, fluid.rho_w, fluid.rho_g, x_w)
    f_g = fluid.f_g
    row |= {'FGAS': f_g, 'FOIL': (1 - f_g) * (1 - x_w), 'FWAT': (1 - f_g) * x_w}
    return {k: row[k] for k in FEATURES}


def valid_row(row) -> bool:
    """SMP-28's row check: non-negative rates and fractions in [0, 1], and every number finite."""
    nums = [v for k, v in row.items() if k not in ('CHOKED', 'FRBH', 'FRWH')]
    if not all(math.isfinite(v) for v in nums):
        return False
    rates = row['WLIQ'] >= 0 and row['WGAS'] >= 0  # the lift gas is at least 0 by its draw
    return rates and all(0 <= row[k] <= 1 for k in ('FGAS', 'FOIL', 'FWAT'))


def oil_equivalent(row) -> float:
    """Sm3/d of oil plus the formation gas at 1000 Sm3 per Sm3 (sampling-draft.md, Conventions of the samples)."""
    return 24 * (row['QOIL'] + row['QGAS'] / 1000)


def solve(wp, bc, backend='rust', x_guess=None) -> dict:
    """One solve and its outcome and reason (D-53), with its search counts and times."""
    out = dict(
        outcome=None,
        reason=None,
        n_roots=0,
        n_stable=0,
        several_stable=False,
        t_solve=0.0,
        t_core=math.nan,
        marches=math.nan,
        non_finite=math.nan,
        rejected=math.nan,
        error=None,
        root=None,
        row=None,
    )
    t0 = time.perf_counter()
    if not bc.p_r > bc.p_s:  # No admissible state (SOL-1, SOL-2): no operating point, without a solve
        out.update(outcome='no_operating_point', reason='p_r_le_p_s')
        return out
    try:
        sim = SSDFSimulator(wp, backend=backend)
        if backend == 'rust':
            sim._roots.core = _TimedCore(sim._roots.core)
        rs = sim.root_set(bc, x_guess=x_guess)
    except Exception as e:  # noqa: BLE001 - any exception of a solve is a solver failure (D-53)
        out.update(outcome='solver_failure', reason='exception', error=f'{type(e).__name__}: {e}'[:300])
        out['t_solve'] = time.perf_counter() - t0
        return out
    if backend == 'rust':
        c = sim._roots.counts
        out.update(
            t_core=sim._roots.core.seconds, marches=c['marches'], non_finite=c['non_finite'], rejected=c['rejected']
        )
    out.update(n_roots=len(rs.roots), n_stable=sum(r.label == 'stable' for r in rs.roots))
    out['several_stable'] = bool(rs.several_stable)
    op = rs.operating_point
    if op is not None:
        row = features(op, bc, wp.fluid)
        if valid_row(row):
            out.update(outcome='operating_point', root=op, row=row)
        else:
            out.update(outcome='solver_failure', reason='invalid_row')
    elif not rs.roots:
        # D-53 counts a scan with non-finite points as incomplete; D-102 only one with a rejected sign change
        partial = out['rejected'] > 0  # D-102
        incomplete = backend == 'rust' and partial
        out.update(outcome='solver_failure' if incomplete else 'no_operating_point')
        out['reason'] = 'incomplete_scan' if incomplete else 'no_root'
    else:
        out.update(outcome='solver_failure', reason='unstable_only')
    out['t_solve'] = time.perf_counter() - t0
    return out


def attempt(builder, inp) -> dict:
    """An attempt's Rust solve: the log record, with the operating point's root under '_root' (not logged)."""
    t0 = time.perf_counter()
    rec = dict(inp, retried=False, start=None, t_casadi=0.0, casadi_outcome=None, casadi_reason=None)  # no retry, H1
    try:
        wp, bc = builder.case(inp)
    except ValueError as e:  # an input the model refuses: a solver failure of the generator, logged
        rec.update(outcome='solver_failure', reason='exception', error=f'build: {e}'[:300], t_total=0.0, _root=None)
        rec.update(rust_outcome='solver_failure', rust_reason='exception')
        return rec
    t_build = time.perf_counter() - t0
    res = solve(wp, bc)
    rec.update({k: res[k] for k in ('outcome', 'reason', 'n_roots', 'n_stable', 'several_stable', 'error')})
    rec.update({k: res[k] for k in ('t_core', 'marches', 'non_finite', 'rejected')})
    rec.update(rust_outcome=res['outcome'], rust_reason=res['reason'], t_build=t_build, t_rust=res['t_solve'])
    if res['row'] is not None:
        rec.update(res['row'])
        rec['q_oe'] = oil_equivalent(res['row'])
    rec['_root'] = res['root']
    rec['t_total'] = time.perf_counter() - t0
    return rec


def has_op(rec) -> bool:
    return rec['outcome'] in ('operating_point', 'recovered')


def best_rate(builder, state, lifted, w_lg_max, floor, log, base) -> dict:
    """D-50: the most oil equivalent at a fully open choke, the state's p_r, GOR, water and p_s,nom, over the lift-gas
    rates BEST_RATE_LIFT of w_lg_max for a lifted well. The status is 'pass', 'below_floor', 'water_cut',
    'no_operating_point' or 'solver_failure' (no operating point found, and a solver failure among the solves)."""
    recs = []
    for i, f in enumerate(BEST_RATE_LIFT if lifted else (0.0,)):
        inp = dict(state, u=1.0, w_lg=f * w_lg_max if lifted else 0.0, lift_on=lifted, p_s=state['p_s_nom'])
        rec = attempt(builder, inp)
        rec.update(base, attempt=i, lift_fraction=f)
        recs.append(rec)
        log.append(rec)
    ops = [r for r in recs if has_op(r)]
    out = dict(best=math.nan, best_w_lg=math.nan, outcomes=[r['outcome'] for r in recs])
    if builder.w['fluid_type'] in OILS and state['wlr'] > WLR_NOT_PRODUCED:
        out['status'] = 'water_cut'
    elif ops:
        top = max(ops, key=lambda r: r['q_oe'])
        out.update(best=top['q_oe'], best_w_lg=top['w_lg'])
        out['status'] = 'pass' if top['q_oe'] >= FLOOR_MARGIN * floor else 'below_floor'
    elif any(r['outcome'] == 'solver_failure' for r in recs):
        out['status'] = 'solver_failure'
    else:
        out['status'] = 'no_operating_point'
    return out


def solve_slots(k, draw, builder, log, t_start, slot_limit_outcome='slot_limit'):
    """D-80 to D-82: pass a solves attempt a of every slot still without an operating point with the Rust core, and
    counts its solver failures, without the CasADi retry (H1, D-54). draw(j, a) gives an attempt's inputs. Returns
    the accepted record of each slot and the well's outcome."""
    accepted, open_ = {}, list(range(N_SAMPLES))
    n_redraws = n_failures = 0
    for a in range(SLOT_LIMIT):
        for j in open_:
            if a > 0:
                if n_redraws >= REDRAW_LIMIT:
                    return accepted, 'redraw_limit'
                n_redraws += 1
            inp = draw(j, a)
            rec = attempt(builder, inp)
            rec.update(draw_index=k, kind='slot', slot=j, attempt=a)
            log.append(rec)
            if has_op(rec):
                accepted[j] = rec
            elif rec['outcome'] == 'solver_failure':
                n_failures += 1
                if n_failures > SOLVER_FAILURE_LIMIT:
                    return accepted, 'solver_failure_limit'
        open_ = [j for j in open_ if j not in accepted]
        if not open_:
            return accepted, 'accepted'
        if time.perf_counter() - t_start > WELL_TIME_LIMIT:
            return accepted, 'time_limit'
    return accepted, slot_limit_outcome


# --- sol-2 ---------------------------------------------------------------------------------------------------------------
def sol2_inputs(r, w):
    """A sol-2 attempt's draws from generator r, in the order of sampler_draft_samples.sol2_sample (D-37 to D-40)."""
    open_, u_c = r.random(), r.uniform(U_MIN, 1.0)
    u = 1.0 if open_ < P_OPEN else u_c
    off, v = r.random(), r.uniform(0.0, 1.0)
    lifted = w['lift'] == 'gas_lift'
    w_lg = (0.0 if off < P_LIFT_OFF else v * w['w_lg_max']) if lifted else 0.0
    p_s = separator_pressure(r, w['p_s'], w['dp_fl'])
    p_r = w['p_r'] * r.uniform(1 - PR_DRIFT, 1 + PR_DRIFT)
    R_go = producing_gor(w, p_r) * r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT)
    x = r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT)
    if w['fluid_type'] in OILS:
        wlr, wgr = min(WLR_MAX, w['wlr'] * x), math.nan
    else:
        wgr = w['wgr'] * x
        wlr = wgr / (wgr + 1e6 / R_go)
    return dict(u=u, w_lg=w_lg, p_s=p_s, p_r=p_r, gor=R_go, wlr=wlr, wgr=wgr)


def well_fixed(w):
    """The inputs that sol-2 keeps fixed over a well's samples."""
    if w['inflow'] == 'vogel':
        coef = dict(w_l_max=w['w_l_max'], k_l=math.nan)
    else:
        coef = dict(k_l=w['k_l'], w_l_max=math.nan)
    return dict(h=w['h'], roughness=w['roughness'], **coef)


def sol2_well(seed, k) -> dict:
    t_start = time.perf_counter()
    w, stations = draw_well(seed, k)
    lifted = w['lift'] == 'gas_lift'
    w_max = w['w_lg_max'] if lifted else 0.0
    builder = Builder(w, stations, design=w if lifted else None)
    fixed = well_fixed(w)
    log, info = [], dict(draw_index=k, outcome=None, best_rate=math.nan, best_rate_w_lg=math.nan)
    nominal = dict(fixed, p_r=w['p_r'], gor=w['R_go'], wlr=w['wlr'], wgr=w.get('wgr', math.nan), p_s_nom=w['p_s'])
    try:
        check(builder.case(dict(nominal, u=1.0, w_lg=0.0, lift_on=lifted, p_s=w['p_s']))[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        return dict(well=w, stations=stations, info=info, log=log, seconds=time.perf_counter() - t_start)
    floor = FLOOR[w['well_type']]
    br = best_rate(builder, nominal, lifted, w_max, floor, log, dict(draw_index=k, kind='best_rate', slot=-1))
    if not lifted and w['fluid_type'] in OILS and br['status'] in ('no_operating_point', 'below_floor'):
        # D-101: an oil well that cannot flow naturally at its best rate gets R3's lift design, as an operator would
        # lift it (nsol-2's rule 2 of D-49); the lift block's draws are the well's own (force_lift)
        w, _ = draw_well(seed, k, force_lift=True)
        w['lift_reason'] = 'no_natural_flow'
        lifted, w_max = True, w['w_lg_max']
        builder.design = w
        base = dict(draw_index=k, kind='best_rate_lifted', slot=-1)
        br = best_rate(builder, nominal, lifted, w_max, floor, log, base)
    info.update(best_rate=br['best'], best_rate_w_lg=br['best_w_lg'], best_rate_outcomes=br['outcomes'])
    if br['status'] != 'pass':
        info['outcome'] = 'best_rate_solver_failure' if br['status'] == 'solver_failure' else 'not_producible'
        info['not_producible_reason'] = br['status']
        return dict(well=w, stations=stations, info=info, log=log, seconds=time.perf_counter() - t_start)

    def draw(j, a):
        r = rng_for(seed, k, j, 'v2', 'sample') if a == 0 else rng_for(seed, k, j, a, 'v2', 'redraw')
        return dict(fixed, lift_on=lifted, **sol2_inputs(r, w))

    accepted, outcome = solve_slots(k, draw, builder, log, t_start)
    info['outcome'] = outcome
    return dict(well=w, stations=stations, info=info, log=log, seconds=time.perf_counter() - t_start)


# --- nsol-2 --------------------------------------------------------------------------------------------------------------
def inflow_at(w0, pvt, p_r, R_go, wlr, wgr):
    """R3's inflow coefficient at a reservoir state (D-45): the oils' PI at the fluid's mu_o and B_o (Vogel's w_l_max
    = k_l p_r / 1.8), or the gas wells' interim PI from Forchheimer's a and b at the gas's Z and mu_g."""
    T_r, ln_term, skin = w0['T_r'], w0['ln_re_rw'] - 0.75, w0['skin']
    rho_lsc = (1 - wlr) * w0['rho_o'] + wlr * w0['rho_w']
    if w0['fluid_type'] in OILS:
        b = pvt.at_gor(R_go)
        J = j_radial(w0['perm'], w0['h_net'], b.oil_viscosity(p_r, T_r), b.bo(p_r, T_r), ln_term, skin) * w0['f_geo']
        k_l = float(J) * rho_lsc / DAY
        if w0['inflow'] == 'vogel':
            return dict(w_l_max=k_l * p_r / 1.8, k_l=math.nan)
        return dict(k_l=k_l, w_l_max=math.nan)
    Z = pvt.z_factor(p_r, T_r)
    mu_g = pvt.gas_viscosity(p_r, T_r, pvt.gas_density(p_r, T_r))
    a, b = forchheimer(w0['perm'], w0['h_net'], mu_g, Z, T_r, ln_term, skin, w0['D_nondarcy'])
    a, b = a * w0['a_ratio'], b * w0['b_ratio']  # the geometry's factors at the drawn state (D-21)
    d = w0['delta_d']
    q_l = gas_rate(a, b, p_r**2 - ((1 - d) * p_r) ** 2) * (1e6 / R_go + wgr) / 1e6
    return dict(k_l=float(q_l * rho_lsc / DAY / (d * p_r)), w_l_max=math.nan)


class NsolWell:
    """An nsol-2 well: R4's Life (the trends), its lift design and start rules (D-49), and its state at a date."""

    def __init__(self, seed, k):
        self.seed, self.k = seed, k
        self.lf = lf = Life(seed, k)
        self.w0 = w0 = lf.w0
        self.oil = lf.oil
        _, self.stations = draw_well(seed, k, 'initial')
        late_lifted = lf.wL['lift'] == 'gas_lift'
        if w0['lift'] == 'gas_lift':
            self.design, self.lift_start, self.rule = w0, 0.0, 'from_start'
            self.rule2 = False
        else:
            self.design = lf.wL if late_lifted else (draw_well(seed, k, 'late', force_lift=True)[0] if lf.oil else None)
            cands = []
            if lf.oil:
                t1 = lf.static_check_time()
                if t1 is not None:
                    cands.append((t1, 'static_check'))
            if late_lifted:
                cands.append((lf.t_lift_drawn, 'drawn_time'))
            self.lift_start, self.rule = min(cands) if cands else (None, None)
            self.rule2 = lf.oil or late_lifted  # D-49, rule 2 (D-84)
        self.builder = Builder(w0, self.stations, design=self.design)

    def lift_on(self, t):
        return self.lift_start is not None and t >= self.lift_start - 1e-12

    def state(self, t, eps=(1.0, 1.0)):
        """The trends' state at date t (years) with the date's drift eps on the GOR and the water (D-40, D-42 to D-46)."""
        lf, w0 = self.lf, self.w0
        t = min(t, lf.T)  # a scan point at T_life can land an ulp past it, where (1 - F)**gamma is complex
        s = lf.state(t)
        R_go, water = s['R_go'] * eps[0], s['water'] * eps[1]
        if self.oil:
            wlr, wgr = min(WLR_MAX, water), math.nan
            wlr_trend, wgr_trend = min(WLR_MAX, s['water']), 0.0
        else:
            wlr, wgr = water / (water + 1e6 / R_go), water
            wlr_trend, wgr_trend = s['water'] / (s['water'] + 1e6 / s['R_go']), s['water']
        coef = inflow_at(w0, self.builder.pvt, s['p_r'], s['R_go'], wlr_trend, wgr_trend)
        return dict(
            t=t, p_r=s['p_r'], gor=R_go, wlr=wlr, wgr=wgr, h=lf.h(t), roughness=lf.roughness(t), p_s_nom=lf.p_s_nom(t),
            stage=lf.stage(t), **coef,
        )  # fmt: skip

    def plan(self, T_end):
        """The 200 slots over [0, T_end], as sampler_draft_samples.nsol2_samples draws them (D-47, D-48), with the dates
        scaled to T_end and the lift's start from the scan."""
        lf, k = self.lf, self.k
        r = rng_for(self.seed, k, 'v2', 'controls')
        p_c = logu(r, *P_CHANGE)
        t_op = np.linspace(0.0, T_end, N_OP)
        t_test = (np.arange(N_TESTS) + 0.5) * T_end / N_TESTS
        slots = [(float(t), 'operating', None) for t in t_op]
        for m, t in enumerate(t_test):
            slots += [(float(t), 'test', (m, i)) for i in range(N_SWEEP)]
        slots.sort(key=lambda s: (s[0], s[1] == 'test'))
        open_, u_0, w_0 = r.random(), r.uniform(U_MIN, 1.0), r.random()
        u_sp = 1.0 if open_ < P_OPEN_SET else u_0
        w_max = self.design['w_lg_max'] if self.design is not None else math.nan
        w_sp, plan, sweep, eps = None, [], None, None
        for t, kind, test in slots:
            lift_on = self.lift_on(t)
            if lift_on and w_sp is None:
                w_sp = w_0 * w_max
            if kind == 'operating' and t > 0:
                for which in ('u', 'w'):
                    change, small, step, redraw_open, redraw = (
                        r.random(),
                        r.random(),
                        r.uniform(-STEP, STEP),
                        r.random(),
                        r.random(),
                    )
                    if change >= p_c or (which == 'w' and not lift_on):
                        continue
                    if which == 'u':
                        if small < P_SMALL:
                            u_sp = reflect(u_sp + step, U_MIN, 1.0)
                        else:
                            u_sp = 1.0 if redraw_open < P_OPEN_SET else U_MIN + (1 - U_MIN) * redraw
                    elif small < P_SMALL:
                        w_sp = reflect(w_sp + step * w_max, 0.0, w_max)
                    else:
                        w_sp = 0.0 if redraw_open < P_LIFT_OFF else w_max * redraw
            u, w_lg, sweep_kind, m_test = u_sp, (w_sp if lift_on else 0.0), None, None
            if kind == 'test':
                m, i = test
                m_test = m + 1
                if i == 0:
                    sweep_kind = 'lift' if (lift_on and m % 2 == 1) else 'choke'
                    strata = [r.uniform(q / 4, (q + 1) / 4) for q in (3, 2, 1, 0)]
                    sweep = (sweep_kind, [1.0] + strata)
                    eps = (r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT), r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT))
                sweep_kind, x = sweep[0], sweep[1][i]
                if sweep_kind == 'choke':
                    u = 1.0 if i == 0 else U_MIN + (1 - U_MIN) * x
                else:
                    w_lg = w_max * x
            else:
                eps = (r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT), r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT))
            p_s = separator_pressure(r, lf.p_s_nom(t), self.w0['dp_fl'])
            plan.append(dict(t=t, kind=kind, test=m_test, sweep=sweep_kind, u=u, w_lg=w_lg, p_s=p_s, eps=eps))
        return plan, p_c


def nsol2_redraw(r, planned, lift_on, w_max, p_s_nom, dp_fl):
    """D-83: a redraw keeps the date and its state, and changes both controls by D-47's set-point change from the
    planned value (the lift gas only where lift is installed), and the separator pressure as D-39. Every draw is made
    whatever its branch."""
    small_u, step_u, open_u, new_u = r.random(), r.uniform(-STEP, STEP), r.random(), r.random()
    small_w, step_w, off_w, new_w = r.random(), r.uniform(-STEP, STEP), r.random(), r.random()
    if small_u < P_SMALL:
        u = reflect(planned['u'] + step_u, U_MIN, 1.0)
    else:
        u = 1.0 if open_u < P_OPEN_SET else U_MIN + (1 - U_MIN) * new_u
    w_lg = planned['w_lg']
    if lift_on:
        if small_w < P_SMALL:
            w_lg = reflect(planned['w_lg'] + step_w * w_max, 0.0, w_max)
        else:
            w_lg = 0.0 if off_w < P_LIFT_OFF else w_max * new_w
    return u, w_lg, separator_pressure(r, p_s_nom, dp_fl)


def nsol2_well(seed, k) -> dict:
    t_start = time.perf_counter()
    nw = NsolWell(seed, k)
    w0, lf, builder = nw.w0, nw.lf, nw.builder
    log = []
    info = dict(draw_index=k, outcome=None, best_rate=math.nan, best_rate_w_lg=math.nan, T_life=lf.T)
    out = dict(well=w0, stations=nw.stations, info=info, log=log, nsol=nw_summary(nw))
    floor = FLOOR[w0['well_type']]
    try:
        check(builder.case(dict(nw.state(0.0), u=1.0, w_lg=0.0, lift_on=nw.lift_on(0.0), p_s=w0['p_s']))[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        out['seconds'] = time.perf_counter() - t_start
        return out
    w_max = nw.design['w_lg_max'] if nw.design is not None else 0.0

    def scan_point(i, t):
        base = dict(draw_index=k, kind='scan', slot=-1 - i, t=t)
        return best_rate(builder, nw.state(t), nw.lift_on(t), w_max, floor, log, base)

    # The life scan (D-50, D-84): T_end is the last passing point before the first that fails
    T = lf.T
    scan, last_pass, first_fail = [], None, None
    for i in range(N_SCAN):
        t = float(np.linspace(0.0, T, N_SCAN)[i])
        br = scan_point(i, t)
        if br['status'] in ('below_floor', 'no_operating_point') and not nw.lift_on(t) and nw.rule2 and i > 0:
            # Rule 2 of D-49: lift starts at the last scan point before natural flow fails
            t_prev = float(np.linspace(0.0, T, N_SCAN)[i - 1])
            if nw.lift_start is None or t_prev < nw.lift_start:
                nw.lift_start, nw.rule = t_prev, 'end_of_natural_flow'
            br = scan_point(i, t)
        scan.append(dict(t=t, status=br['status'], best=br['best'], lift_on=nw.lift_on(t)))
        if i == 0:
            info.update(best_rate=br['best'], best_rate_w_lg=br['best_w_lg'], best_rate_outcomes=br['outcomes'])
            if br['status'] != 'pass':
                info['outcome'] = 'best_rate_solver_failure' if br['status'] == 'solver_failure' else 'not_producible'
                info['not_producible_reason'] = br['status']
                break
        if br['status'] == 'pass':
            last_pass = t
        elif br['status'] != 'solver_failure':  # a scan point with a solver failure is skipped
            first_fail = i
            break
    info.update(scan=scan, lift_start=nw.lift_start, lift_start_rule=nw.rule)
    out['nsol'] = nw_summary(nw)
    if info['outcome'] is not None:
        out['seconds'] = time.perf_counter() - t_start
        return out
    T_end = T if first_fail is None else last_pass
    info['T_end'] = T_end
    if T_end is None or T_end <= 0.0:
        info['outcome'] = 'short_life'
        out['seconds'] = time.perf_counter() - t_start
        return out
    plan, p_c = nw.plan(T_end)
    info['p_change'] = p_c

    def draw(j, a):
        sl = plan[j]
        t, lift_on = sl['t'], nw.lift_on(sl['t'])
        st = nw.state(t, sl['eps'])
        if a == 0:
            u, w_lg, p_s = sl['u'], sl['w_lg'], sl['p_s']
        else:
            r = rng_for(seed, k, j, a, 'v2', 'redraw')
            u, w_lg, p_s = nsol2_redraw(r, sl, lift_on, w_max, st['p_s_nom'], w0['dp_fl'])
        return dict(
            st, u=u, w_lg=w_lg if lift_on else 0.0, p_s=p_s, lift_on=lift_on, test=sl['test'], sweep=sl['sweep']
        )

    accepted, outcome = solve_slots(k, draw, builder, log, t_start, 'dead_date')
    info['outcome'] = outcome
    out['seconds'] = time.perf_counter() - t_start
    return out


def nw_summary(nw):
    lf = nw.lf
    return dict(
        T_life=lf.T,
        t_plateau=lf.t_pl,
        decline_D=lf.D,
        decline_b=lf.b,
        gamma_p=lf.gamma,
        r_p_start=lf.r0,
        r_p_end=lf.r1,
        F_bt=lf.F_bt,
        wlr_bt=lf.wlr_bt,
        water_start=lf.water0,
        water_end=lf.water1,
        t_sep_1=lf.t_sep[0],
        t_sep_2=lf.t_sep[1],
        r_s_mid=lf.wm['r_s'],
        r_s_late=lf.wL['r_s'],
        roughness_start=lf.w0['roughness'],
        roughness_end=max(lf.wL['roughness'], lf.w0['roughness']),
        lift_start=nw.lift_start,
        lift_start_rule=nw.rule,
        design=None if nw.design is None else {k: nw.design.get(k) for k in LIFT_KEYS},
    )


LIFT_KEYS = ('md_inj', 'z_inj', 'T_lg', 'w_lg_max', 'glr_cap', 'p_inj', 'dp_valve', 'packer_offset', 'phi_lg')


# --- The pool --------------------------------------------------------------------------------------------------------------
def run_task(task):
    kind, seed, k, raw = task
    path = Path(raw) / f'{k:06d}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            res = pickle.load(f)
        return k, res['info']['outcome'], res['seconds']
    t0, cpu0 = time.perf_counter(), time.process_time()
    try:
        res = (sol2_well if kind == 'sol' else nsol2_well)(seed, k)
    except Exception:  # noqa: BLE001 - a crash of the prototype is logged as the well's outcome
        res = dict(well=None, stations=None, log=[], seconds=0.0)
        res['info'] = dict(draw_index=k, outcome='crash', error=traceback.format_exc()[-2000:])
    res['seconds'], res['cpu_seconds'] = time.perf_counter() - t0, time.process_time() - cpu0
    for rec in res['log']:
        rec.pop('_root', None)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(res, f)
    tmp.rename(path)
    return k, res['info']['outcome'], res['seconds']


def run(kind, n_wells, out, seed, processes, wells=None):
    out = Path(out)
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context('forkserver')
    done = {}
    with ctx.Pool(processes) as pool:
        if wells is not None:
            ks = range(*wells)
            for k, outcome, sec in pool.imap_unordered(run_task, [(kind, seed, k, str(raw)) for k in ks]):
                done[k] = outcome
                print(f'well {k}: {outcome} in {sec:.1f} s', flush=True)
        else:
            nxt = 0
            while True:
                accepted = sorted(k for k, o in done.items() if o == 'accepted')
                if len(accepted) >= n_wells and all(k in done for k in range(accepted[n_wells - 1] + 1)):
                    break
                need = n_wells - len(accepted)
                batch = range(nxt, nxt + max(processes, int(math.ceil(need * 1.15))))
                nxt = batch[-1] + 1
                for k, outcome, sec in pool.imap_unordered(run_task, [(kind, seed, k, str(raw)) for k in batch]):
                    done[k] = outcome
                    n_acc = sum(o == 'accepted' for o in done.values())
                    print(f'well {k}: {outcome} in {sec:.1f} s; {n_acc} accepted of {len(done)}', flush=True)
    wall = time.perf_counter() - t0
    meta = dict(
        kind=kind,
        seed=seed,
        processes=processes,
        wall_seconds=wall,
        n_wells=n_wells,
        wells=wells,
        revision=REVISION,
        python=sys.version,
        platform=platform.platform(),
    )
    with open(out / f'run_{int(time.time())}.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{len(done)} wells in {wall:.0f} s on {processes} processes')
    if wells is None:
        assemble(kind, out, seed, n_wells)


# --- The tables (datasets-draft.md) ---------------------------------------------------------------------------------------
INPUT_COLS = {'u': 'CHK', 'p_s': 'PDC', 'w_lg': 'WGL', 'p_r': 'PRES', 'gor': 'GOR', 'wlr': 'WLR'}
NSOL_COLS = {'lift_on': 'LIFT', 'h': 'HTC', 'roughness': 'ROUGH', 'k_l': 'KL', 'w_l_max': 'WLMAX'}


def well_columns(w, info, kind, stations, nsol=None, theta_grid=math.nan) -> dict:
    """A drawn well's row of `wells` and `drawn_wells` (datasets-draft.md, Section 2), from R3's draws."""
    g = lambda key: w.get(key, math.nan)  # noqa: E731
    oil, lifted = w['fluid_type'] in OILS, w['lift'] == 'gas_lift'
    stage = w['stage']
    life_state = (
        stage if stage != 'late' else ('depleted' if w['drive'] == 'depletion' else 'watered_out')
    )  # fmt: skip
    p_lith = w['p_lith']
    row = dict(
        draw_index=w['well'],
        configuration=V2,
        n_cells=N_CELLS,
        coverage_tail='tail' if w['tail'] else 'core',
        well_type=w['well_type'],
        fluid_type=w['fluid_type'],
        fluid_class=w['fluid_class'],
        class_rule='mccain_1990',
        drive=w['drive'],
        pressure_class=w['pressure_class'],
        saturation=w['saturation'],
        geometry_class=w['geometry_class'],
        geometry_family=w['geometry_family'],
        pvt_model='black_oil',
        interim_pvt=w['fluid_class'] in ('volatile_oil', 'gas_condensate'),
        inflow=w['inflow'],
        interim_inflow=w['inflow'] == 'gas_rate',
        tubing=TUBING_NAMES.get(round(w['D'], 4), str(w['D'])),
        choke_model='simpson',
        chk_profile=w['choke_profile'],
        z=w['z'],
        water_depth=w['d_w'],
        L=w['L'],
        md_bottom=stations[0][-1],
        max_inclination=theta_grid,
        kickoff_tvd=g('geo_kickoff'),
        build_rate=g('geo_build_rate'),
        build_rate_2=g('geo_build_rate_2'),
        hold_inclination=g('geo_theta_hold'),
        drop_rate=g('geo_drop_rate'),
        end_inclination=g('geo_theta_end'),
        end_section_tvd=g('geo_end_section_tvd'),
        heel_inclination=g('geo_theta_heel'),
        tangent_md=g('geo_tangent_md'),
        drain_length=g('drain_length'),
        salinity=w['salinity'],
        rho_w=w['rho_w'],
        p_h=w['p_h'],
        overpressure_ratio=(w['p_i'] - w['p_h']) / (p_lith - w['p_h']),
        p_i=w['p_i'],
        T_s=w['T_s'],
        geothermal_gradient=w['G'],
        T_r=w['T_r'],
        p_sep=w['p_sep'],
        T_sep=w['T_sep'],
        dp_flowline=w['dp_fl'],
        api=w['api'],
        sg_gas=w['sg_gas'],
        rho_o=w['rho_o'],
        rho_g=w['rho_g'],
        p_sat=w['p_sat'] if w['p_sat'] is not None else math.nan,
        R_si=w['R_i'],
        pvt_table_id=None,
        kgko_weight=g('w_kgko') if oil else math.nan,
        gor_factor_cond=g('c_k') if w['fluid_class'] == 'gas_condensate' else math.nan,
        permeability=w['perm'],
        net_pay=w['h_net'],
        skin=w['skin'],
        ln_re_rw=w['ln_re_rw'],
        drawdown_design=w['delta_d'],
        anisotropy=g('I_ani'),
        productivity_factor=w['f_geo'],
        J=g('J') if oil else math.nan,
        forchheimer_a=math.nan if oil else w['a_gas'],
        forchheimer_b=math.nan if oil else w['b_gas'],
        non_darcy_D=math.nan if oil else w['D_nondarcy'] / 28.3168,  # d/Mscf to d/Sm3
        design_rate=w['q_l_ref'] if oil else w['q_g_ref'],
        floor=FLOOR[w['well_type']],
        best_rate=info.get('best_rate', math.nan),
        best_rate_w_lg=info.get('best_rate_w_lg', math.nan),
        D=w['D'],
        bean_ratio=w['beta'],
        C_D=w['c_d'],
        K_c=w['K_c'],
        U_t=w['U_t'],
        k_e=w['k_e'],
        well_age=w['age_years'] * 365,
        cp_o=w['cp_o'],
        cp_w=w['cp_w'],
        cp_g=w['cp_g'],
        flags=w['flags'],
    )
    design = (nsol or {}).get('design') if kind == 'nsol' else (w if lifted else None)
    lift_cols = dict(
        injection_md=g('md_inj'),
        injection_tvd=g('z_inj'),
        T_lg=g('T_lg'),
        w_lg_max=g('w_lg_max'),
        glr_cap=g('glr_cap'),
        p_inj=g('p_inj'),
        dp_valve=g('dp_valve'),
        packer_offset=g('packer_offset'),
        phi_lg=g('phi_lg'),
    )
    if design is not None:
        lift_cols = dict(
            injection_md=design['md_inj'],
            injection_tvd=design['z_inj'],
            **{c: design[c] for c in ('T_lg', 'w_lg_max', 'glr_cap', 'p_inj', 'dp_valve', 'packer_offset', 'phi_lg')},
        )
    else:
        lift_cols = {c: math.nan for c in lift_cols}
    row.update(lift_cols)
    if kind == 'sol':
        row.update(
            life_stage=stage,
            life_state=life_state,
            lift='gas_lift' if lifted else 'natural',
            lift_reason=w.get('lift_reason', 'needs_lift' if w['needs_lift'] else 'drawn') if lifted else None,
            p_s_nom=w['p_s'],
            r_s=w['r_s'],
            p_r_nom=w['p_r'],
            gor_nom=w['R_go'],
            wlr_nom=w['wlr'],
            wgr_nom=math.nan if oil else w['wgr'],
            k_l=w['k_l'] if w['inflow'] != 'vogel' else math.nan,
            w_l_max=w['w_l_max'] if w['inflow'] == 'vogel' else math.nan,
            roughness=w['roughness'],
            h=w['h'],
        )
    else:
        ns = {k: v for k, v in (nsol or {}).items() if k != 'design'}
        row.update(ns)
        row['lift'] = 'gas_lift' if ns.get('lift_start') is not None else 'natural'
        row['T_end'] = info.get('T_end', math.nan)
        row['p_change'] = info.get('p_change', math.nan)
    return row


def assemble(kind, out, seed, n_wells):
    """The tables of datasets-draft.md from the raw results: the first n_wells accepted wells by draw index."""
    out = Path(out)
    res = {}
    for p in sorted((out / 'raw').glob('*.pkl')):
        with open(p, 'rb') as f:
            r = pickle.load(f)
        res[r['info']['draw_index']] = r
    ks = sorted(res)
    acc = [k for k in ks if res[k]['info']['outcome'] == 'accepted'][:n_wells]
    last = acc[-1] if acc else -1
    ks = [k for k in ks if k <= last]
    ids = {k: i for i, k in enumerate(acc)}
    drawn, logs, geometry = [], [], []
    for k in ks:
        r = res[k]
        info, w = r['info'], r['well']
        if w is None:
            drawn.append(dict(draw_index=k, outcome=info['outcome'], error=info.get('error')))
            continue
        geo = WellGeometry.from_survey(r['stations'][0], r['stations'][1], N_CELLS, D=w['D'])
        theta = math.degrees(math.acos(min(1.0, float(np.min(np.asarray(geo.cos_incl))))))
        row = well_columns(w, info, kind, r['stations'], r.get('nsol'), theta)
        lg = pd.DataFrame(r['log'])
        slot = lg[lg.kind == 'slot'] if len(lg) else lg
        n_red = int((slot.attempt > 0).sum()) if len(slot) else 0
        redrawn = slot[~slot.outcome.isin(['operating_point', 'recovered'])] if len(slot) else slot
        row.update(
            ID=ids.get(k),
            outcome=info['outcome'],
            not_producible_reason=info.get('not_producible_reason'),
            n_attempts=len(slot),
            n_redraws=n_red,
            n_redraws_no_op=int((redrawn.outcome == 'no_operating_point').sum()) if len(slot) else 0,
            n_redraws_solver=int((redrawn.outcome == 'solver_failure').sum()) if len(slot) else 0,
            n_recovered=int((lg.outcome == 'recovered').sum()) if len(lg) else 0,
            n_several_stable=int(slot.several_stable.fillna(False).astype(bool).sum()) if len(slot) else 0,
            best_rate_outcomes=info.get('best_rate_outcomes'),
            seconds=r['seconds'],
            cpu_seconds=r.get('cpu_seconds', math.nan),
            error=info.get('error'),
        )
        if kind == 'nsol':
            row['scan_best_rate'] = [s['best'] for s in info.get('scan', [])]
            row['scan_outcome'] = [s['status'] for s in info.get('scan', [])]
        drawn.append(row)
        if len(lg):
            lg['ID'] = ids.get(k)
            lg['well_outcome'] = info['outcome']
            logs.append(lg)
        if k in ids:
            geometry.append(
                pd.DataFrame(
                    dict(ID=ids[k], station=np.arange(len(r['stations'][0])), md=r['stations'][0], tvd=r['stations'][1])
                )
            )
    drawn = pd.DataFrame(drawn)
    attempts = pd.concat(logs, ignore_index=True)
    rename = dict(INPUT_COLS, **(NSOL_COLS if kind == 'nsol' else {}))
    if kind == 'nsol':
        attempts['WEEKS'] = attempts['t'] * 52.1775
        rename |= {'stage': 'STAGE', 'test': 'TEST', 'sweep': 'SWEEP'}
    attempts = attempts.rename(columns=rename)
    attempts.to_parquet(out / 'attempts.parquet', index=False)
    drawn.to_parquet(out / 'drawn_wells.parquet', index=False)
    wells = drawn[drawn.ID.notna()].copy()
    wells['ID'] = wells['ID'].astype('int32')
    wells['split'] = split(wells, seed)
    wells = wells.sort_values('ID').drop(columns=['outcome', 'best_rate_outcomes', 'error', 'not_producible_reason'])
    wells.to_parquet(out / 'wells.parquet', index=False)
    slots = attempts[(attempts.kind == 'slot') & attempts.ID.notna()]
    keys = ['ID', 'slot', 'attempt']
    extra = ['WEEKS', 'STAGE', 'TEST', 'SWEEP', 'LIFT', 'HTC', 'ROUGH', 'KL', 'WLMAX'] if kind == 'nsol' else []
    inputs = list(INPUT_COLS.values()) + extra
    samples = slots[slots.outcome.isin(['operating_point', 'recovered'])][keys + inputs + list(FEATURES)]
    samples = samples.astype({'ID': 'int32', 'slot': 'int16', 'attempt': 'int16'}).sort_values(['ID', 'slot'])
    samples.to_parquet(out / 'samples.parquet', index=False)
    red = attempts[
        (attempts.kind == 'slot')
        & (~attempts.outcome.isin(['operating_point']) | (attempts.well_outcome != 'accepted'))
    ]
    red = red[['draw_index', 'ID', 'slot', 'attempt'] + inputs + ['outcome', 'reason', 'retried', 'start']].copy()
    red['n_roots'], red['n_stable'] = attempts.loc[red.index, 'n_roots'], attempts.loc[red.index, 'n_stable']
    red.to_parquet(out / 'redraws.parquet', index=False)
    pd.concat(geometry, ignore_index=True).to_parquet(out / 'geometry.parquet', index=False)
    pvt_cols = ['pvt_table_id', 'gor', 'p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma', 'rho_o', 'rho_g']
    pd.DataFrame(columns=pvt_cols).to_parquet(out / 'pvt.parquet', index=False)  # empty: the interim mapping
    meta = dict(
        dataset=f'manywells-{kind}-2 (R6 prototype)',
        revision=REVISION,
        seed=seed,
        configuration=V2,
        n_cells=N_CELLS,
        backend='rust',
        retry_backend=None,
        interim_mapping='volatile oils and condensates as BlackOil; gas and condensate wells on a ProductivityIndex',
        limits=dict(slot=SLOT_LIMIT, redraws=REDRAW_LIMIT, solver_failures=SOLVER_FAILURE_LIMIT),
        producibility=dict(floor=FLOOR, margin=FLOOR_MARGIN, lift_fractions=BEST_RATE_LIFT, scan_points=N_SCAN),
        wells_drawn=len(drawn),
        wells_accepted=len(wells),
        outcomes=drawn.outcome.value_counts().to_dict(),
        attempts=attempts[attempts.kind == 'slot'].outcome.value_counts().to_dict(),
        cpu_seconds=float(drawn.cpu_seconds.sum()),
        files={p.name: len(pd.read_parquet(p)) for p in sorted(out.glob('*.parquet'))},
    )
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps(meta, indent=1, default=str))


def split(wells, seed):
    """D-90: 80/20 by well, stratified by fluid type, well type, geometry class and the tail; within each stratum the
    wells are ordered by a draw from rng_for(seed, 'v2', 'split') and every fifth is test."""
    r = rng_for(seed, 'v2', 'split')
    key = r.random(len(wells))
    df = wells[['ID', 'fluid_type', 'well_type', 'geometry_class', 'coverage_tail']].assign(key=key)
    df['rank'] = df.groupby(['fluid_type', 'well_type', 'geometry_class', 'coverage_tail'])['key'].rank(method='first')
    return np.where(df['rank'].astype(int) % 5 == 0, 'test', 'train')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('kind', choices=('sol', 'nsol'))
    r.add_argument('n', type=int)
    r.add_argument('out')
    r.add_argument('--seed', type=int, default=2026)
    r.add_argument('--processes', type=int, default=12)
    r.add_argument('--wells', type=int, nargs=2)
    a = sub.add_parser('assemble')
    a.add_argument('kind', choices=('sol', 'nsol'))
    a.add_argument('n', type=int)
    a.add_argument('out')
    a.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    if args.cmd == 'run':
        run(args.kind, args.n, args.out, args.seed, args.processes, args.wells)
    else:
        assemble(args.kind, args.out, args.seed, args.n)


if __name__ == '__main__':
    main()
