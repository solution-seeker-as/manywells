"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The generation of one v2 well (specs/sampling.md, SMP-101 to SMP-106, the choke's sizing, SMP-77, and the lift that a
solve starts, SMP-90 and SMP-103): a well whose design rate lies below its start hurdle is dropped unsolved
(SMP-102); else the well's draws and life are mapped to the v2.0.0 configuration and checked, its choke is sized at a
solve, at its date (sol-2) or its life's start (nsol-2), its best rate is solved against the economic floor, at its
date (sol-2) or along its life (nsol-2: the life scan, which ends the life and may start gas lift), and its slots are
solved in passes, each attempt without an operating point redrawn, until every slot has one or the well reaches a
limit. In nsol-2 a dead date, a slot without an operating point at the slot limit,
starts the well's lift or ends its life there, and the slots are planned and solved again (SMP-106). Every attempt is
one solve by the Rust core, which takes no start (SMP-107); no other backend retries it. Nothing here runs many wells
or writes a dataset.

A well's result (generate_sol2_well, generate_nsol2_well) is a dict of:

- `well`: the well's draws and labels. sol-2: the life at the well's date (life.sol2_well), with the lift design and
  `lift_start_rule` `no_natural_flow` where the producibility filter lifted it. nsol-2: the initial-stage draws
  (draws.draw_well), which hold what is fixed for the life. A `table_failed` well's are drawn without its table.
  Its choke (`K_c`, `d_bean` and `beta`) is the sizing's, None for a well dropped before it or at it. The lift
  block's flags in `flags` are those of the lift design that the well uses, and none for a well that is not lifted.
- `stations`: the survey, (md, tvd).
- `table`: a table well's manifest (tables._read_table) without its PVTTableSet, with `status` `ok`, or `failed` with
  the `step` and the `error`; None for a well of another fluid class.
- `nsol`: None for a sol-2 well. nsol-2: the life's parameters (life.Life.parameters) with `lift_start` (years; None
  for a well that stays natural) and `lift_start_rule` after the life scan and the dead dates (`no_natural_flow`
  with a start of 0 for a natural oil that the producibility filter lifted at its start), and `design`, the lift
  design's draws (None for a well that no rule lifts).
- `info`: `draw_index`; `outcome`, the well's (SMP-106): `accepted`, `invalid_input`, `table_failed`,
  `not_producible`, `best_rate_solver_failure`, `slot_limit` (sol-2), `dead_date` (nsol-2), `redraw_limit`,
  `solver_failure_limit` or `short_life` (nsol-2); `error`, the message of an `invalid_input` or `table_failed` well
  (else None); `best_rate` (Sm3/d of oil equivalent) and `best_rate_w_lg` (kg/s), the best rate and its lift-gas
  rate, NaN without an operating point, and `best_rate_outcomes`, the outcomes of its solves (nsol-2: at the start);
  `not_producible_reason`, the best rate's status of a well dropped for it, `below_start_hurdle`, or
  `no_fluid_of_class` for a table well none of whose fluid draws is of its class (else None); a
  well dropped at its choke's sizing has the sizing's status and outcomes. nsol-2 adds `scan`, the life
  scan's points up to the one that ends the life, each a dict of `t` (years), `status`, `best` and `lift_on`;
  `T_end` (years), the end of the life (NaN for a well dropped before the scan ends it), and `T_scan`, the end that
  the scan gave it; `dead_dates`, one dict per dead date that changed the life, of `t` (years), `slots` (the slots
  without an operating point in that pass) and `action`, `lift_start` or `life_end`; and `p_change`, the well's
  probability of a set-point change (NaN before the plan is drawn).
- `samples`: the records of the attempts that gave a slot its operating point, in the order of the slots: all
  N_SLOTS of an accepted well, those found before the drop of a well dropped at a limit. Each is also in `log`.
- `log`: the record of every attempt, in the order solved: the best rate's, the life scan's and the slots'. An
  nsol-2 well whose life a dead date changed holds the slots' attempts of its last pass only.
- `seconds`: the wall time.

An attempt's record is a flat dict of:

- its inputs (samples._sol2_attempt or samples._nsol2_attempt; a best-rate solve's are the state's, with `p_s_nom`);
- `draw_index`; `kind`, `choke_sizing`, `erosional_cap` (with `cap_step` and `erosional_ratio`, _cap_choke),
  `best_rate`, `best_rate_lifted` (a sol-2 well's best rate with the filter's lift), `scan`, `scan_lifted` (an
  nsol-2 well's life scan with the filter's lift from its start) or `slot`; `slot`, j for a
  slot, -1 for a sizing or best-rate solve and -1 - i for the life scan's point i; `attempt`, a for a slot, else the
  index of the lift-gas rate, whose share of the capacity is `lift_fraction`;
- `outcome` and `reason` (SMP-104), and `error`, an exception's message (else None);
- `n_roots` and `n_stable`, the sizes of the root set, and `several_stable`; `off_table`, whether a point of the
  operating point lies off its fluid's PVT table (None without an operating point);
- the search's counts `marches`, `non_finite` (scan points whose march failed), `rejected` (sign changes of R
  not accepted as roots) and `rejected_at_edge` (those next to a failed march), NaN without a search, and the times
  `t_build` and `t_solve` (s);
- for an operating point, its FEATURES and `q_oe`, the oil equivalent (Sm3/d).
"""

import json
import math
import time

import numpy as np

from manywells.configurations import V2, check
from manywells.datasets.v2 import FEATURES, _sample_features  # noqa: F401 - FEATURES: the columns a record holds
from manywells.sampling.v2.inflow import _COEF_KEYS
from manywells.sampling.v2.life import _LIFT_KEYS, _WLR_MAX, Life, _lift_on, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.reservoir import _OILS
from manywells.sampling.v2.samples import _LIFT_GAS_ALWAYS_ON, _N_SLOTS, _nsol2_attempt, _nsol2_plan, _sol2_attempt
from manywells.sampling.v2.table_wells import _gor_axis, _table_well
from manywells.sampling.v2.tables import TableMismatch, _read_table, _wells_file
from manywells.solvers.rust import RustRootFinder

_FLOOR = {'platform': 1.6, 'onshore': 1.6, 'subsea': 20.0}  # Sm3/d of oil equivalent, by well type  # spec: SMP-102
_FLOOR_MARGIN = 0.5  # A well is not produced below this share of its floor  # spec: SMP-102
# Sm3/d of oil equivalent: a well is drilled only where its design rate reaches its type's start hurdle
_START_HURDLE = {'platform': 20.0, 'onshore': 0.0, 'subsea': 500.0}  # spec: SMP-102
# The chokes that the sizing solves a well with, shares of its largest: the second is a bean of half the diameter
_SIZING_CHOKES = (1.0, 1 / 4)  # spec: SMP-77
# The erosional cap: a capped choke lets the well flow at this share of its tubing's erosional velocity up to it, found
# among at most so many chokes, the sized one included
_EROSIONAL_LOW, _EROSIONAL_STEPS = 0.97, 12  # spec: SMP-77
_CAP_FLAGS = ('choke_at_erosional_rate', 'erosional_cap_unmet')  # spec: SMP-77
# The lift block's flags of a lift design (draws._lift): a well's are those of the design that it uses
_DESIGN_FLAGS = ('injection_at_packer', 'injection_raised_to_0.3L', 'lift_capacity_at_bound')  # spec: SMP-90
_SLOT_LIMIT, _REDRAW_LIMIT, _SOLVER_FAILURE_LIMIT = 100, 1000, 10  # spec: SMP-106 (D-137)
_DEAD_DATE_LIMIT = 5  # The dead dates that may start an nsol-2 well's lift or end its life  # spec: SMP-106 (D-142)
_BEST_RATE_LIFT = (0.0, 0.5, 1.0)  # The best rate's lift-gas rates, shares of the capacity  # spec: SMP-103
_N_SCAN = 21  # Dates of the life scan  # spec: SMP-101
_NO_SEARCH = dict(n_roots=0, n_stable=0, several_stable=False, marches=math.nan, non_finite=math.nan)
_NO_SEARCH |= dict(rejected=math.nan, rejected_at_edge=math.nan, off_table=None, t_solve=0.0)


# --- One attempt -----------------------------------------------------------------------------------------------------
def _valid_row(row) -> bool:  # spec: SMP-104
    """Whether every number of a row is finite, its liquid and gas rates are not negative and its mass fractions lie
    in [0, 1]. The lift gas is at least 0 by its draw."""
    if not all(math.isfinite(v) for key, v in row.items() if key not in ('CHOKED', 'FRBH', 'FRWH')):
        return False
    return row['WLIQ'] >= 0 and row['WGAS'] >= 0 and all(0 <= row[key] <= 1 for key in ('FGAS', 'FOIL', 'FWAT'))


def _oil_equivalent(row) -> float:  # spec: SMP-103
    """Sm3/d of oil, and of the formation gas at 1000 Sm3 per Sm3 (v2: scope and conventions)."""
    return 24 * (row['QOIL'] + row['QGAS'] / 1000)


def _is_panic(e: BaseException) -> bool:
    """Whether e is a panic of the Rust core: PyO3's PanicException, a BaseException of the module pyo3_runtime,
    which Python cannot import by name. A panic follows from its inputs, as an exception does."""
    return type(e).__name__ == 'PanicException' and type(e).__module__ == 'pyo3_runtime'


def _solve(wp, bc) -> dict:  # spec: SMP-104
    """
    One solve of a well at its boundary conditions by the Rust core, and its outcome. A MemoryError is the process's
    memory cap, not the inputs': it is not an outcome, and goes on to the run, which solves the well again (run.py).

    :return: A dict of `outcome`, `reason` and `error`; `n_roots`, `n_stable` and `several_stable`; the search's
        counts `marches`, `non_finite`, `rejected` and `rejected_at_edge` (NaN without a search); `off_table`,
        whether the operating point leaves its fluid's PVT table (None without one); `t_solve` (s); `row`, the
        operating point's features (None without one); and `wellhead`, its state at the wellhead, in the state
        vector's order (None without one)
    """
    out = dict(outcome=None, reason=None, error=None, **_NO_SEARCH, row=None, wellhead=None)
    if not bc.p_r > bc.p_s:  # No admissible state (SOL-1, SOL-2): the well cannot flow, without a solve
        out.update(outcome='no_operating_point', reason='p_r_le_p_s')
        return out
    t0 = time.perf_counter()
    try:
        fluid = wp.fluid
        if not fluid.covers(bc.p_r, bc.T_r):  # The model's refusal, as SSDFSimulator.root_set makes it
            raise ValueError(
                f"the fluid's PVT table, {fluid.table_range}, does not cover the reservoir: "
                f'p_r = {bc.p_r} bar, T_r = {bc.T_r} K'
            )
        finder = RustRootFinder(wp)
        rs = finder.find(bc)
    except MemoryError:
        raise
    except BaseException as e:  # noqa: BLE001 - an exception or a panic of a solve is a solver failure
        if not (isinstance(e, Exception) or _is_panic(e)):
            raise  # An interrupt or an exit, not the solve's
        out.update(outcome='solver_failure', reason='exception', error=f'{type(e).__name__}: {e}'[:300])
        out['t_solve'] = time.perf_counter() - t0
        return out
    counts = finder.counts
    out.update(marches=counts['marches'], non_finite=counts['non_finite'], rejected=counts['rejected'])
    out['rejected_at_edge'] = counts['rejected_at_edge']
    out.update(n_roots=len(rs.roots), n_stable=sum(r.label == 'stable' for r in rs.roots))
    out['several_stable'] = bool(rs.several_stable)
    op = rs.operating_point
    if op is not None:
        out['off_table'] = bool(op.off_table)
        row = _sample_features(op, bc, wp.fluid)
        if _valid_row(row):
            out.update(outcome='operating_point', row=row, wellhead=[float(v) for v in op.x[-7:]])
        else:
            out.update(outcome='solver_failure', reason='invalid_row')
    elif rs.roots:  # A continuous R(p_0) has at least as many stable roots as unstable ones: the search missed one
        out.update(outcome='solver_failure', reason='unstable_only')
    elif counts['rejected'] > counts['rejected_at_edge']:  # A sign change of R that the search could not resolve
        out.update(outcome='solver_failure', reason='incomplete_scan')
    else:  # No root, and no rejected sign change but at the edge of the states the march admits (D-138), whatever
        # the non-finite scan points
        out.update(outcome='no_operating_point', reason='no_root')
    out['t_solve'] = time.perf_counter() - t0
    return out


def _attempt(builder, inp, wellhead=None) -> dict:  # spec: SMP-104
    """The record of an attempt (the module's docstring), without its labels: the inputs inp (mapping.WellBuilder.case)
    of the well of builder, solved once. A list given as wellhead takes the operating point's state at the wellhead
    (_solve), or None without an operating point."""
    t0 = time.perf_counter()
    rec = dict(inp)
    try:
        wp, bc = builder.case(inp)
    except ValueError as e:  # The model refuses the inputs: a solver failure
        res = dict(outcome='solver_failure', reason='exception', error=f'build: {e}'[:300], **_NO_SEARCH, row=None)
        res['wellhead'] = None
        t_build = time.perf_counter() - t0
    else:
        t_build = time.perf_counter() - t0
        res = _solve(wp, bc)
    row, head = res.pop('row'), res.pop('wellhead')
    if wellhead is not None:
        wellhead.append(head)
    rec.update(res, t_build=t_build)
    if row is not None:
        rec.update(row, q_oe=_oil_equivalent(row))
    return rec


# --- The best rate and the slots -------------------------------------------------------------------------------------
def _best_rate(builder, state, lifted, w_lg_max, floor, log, base) -> dict:  # spec: SMP-103
    """
    A well's best rate at a state: the most oil equivalent among its operating points at a fully open choke and the
    state's nominal downstream pressure, over the lift-gas rates _BEST_RATE_LIFT of the capacity for a lifted well.

    :param builder: The well's WellBuilder
    :param state: The state (life.Life.state's keys): `p_r`, `gor`, `wlr`, `p_s_nom`, `h`, `roughness` and the
        inflow's coefficients
    :param lifted: Whether the well is lifted at the state
    :param w_lg_max: The lift-gas capacity (kg/s)
    :param floor: The well's economic floor (Sm3/d of oil equivalent)
    :param log: The well's log, to which the solves' records are added
    :param base: The labels of the solves' records
    :return: A dict of `status`: `water_cut`, `pass`, `below_floor`, `solver_failure` (no operating point, and a
        solver failure among the solves) or `no_operating_point`; `best` (Sm3/d) and `best_w_lg` (kg/s), NaN without
        an operating point; `outcomes`, the solves' outcomes; and `flows_without_lift_gas`, whether the solve without
        lift gas has an operating point of at least the floor's margin (SMP-94)
    """
    recs = []
    for i, f in enumerate(_BEST_RATE_LIFT if lifted else (0.0,)):
        inp = dict(state, u=1.0, w_lg=f * w_lg_max if lifted else 0.0, lift_on=lifted, p_s=state['p_s_nom'])
        rec = _attempt(builder, inp)
        rec.update(base, attempt=i, lift_fraction=f)
        recs.append(rec)
        log.append(rec)
    ops = [rec for rec in recs if rec['outcome'] == 'operating_point']
    out = dict(best=math.nan, best_w_lg=math.nan, outcomes=[rec['outcome'] for rec in recs])
    off = recs[0]  # The solve without lift gas, the first of _BEST_RATE_LIFT
    out['flows_without_lift_gas'] = off['outcome'] == 'operating_point' and off['q_oe'] >= _FLOOR_MARGIN * floor
    if builder.w['fluid_type'] in _OILS and state['wlr'] > _WLR_MAX:  # spec: SMP-102
        out['status'] = 'water_cut'
    elif ops:
        top = max(ops, key=lambda rec: rec['q_oe'])
        out.update(best=top['q_oe'], best_w_lg=top['w_lg'])
        out['status'] = 'pass' if top['q_oe'] >= _FLOOR_MARGIN * floor else 'below_floor'  # spec: SMP-102
    elif any(rec['outcome'] == 'solver_failure' for rec in recs):
        out['status'] = 'solver_failure'
    else:
        out['status'] = 'no_operating_point'
    return out


def _size_choke(builder, state, lifted, w_lg_max, log, base) -> tuple:  # spec: SMP-77
    """
    Sizes a well's choke at a state. The well is solved with its largest choke (mapping.WellBuilder.widest_choke),
    fully open, at the state's nominal downstream pressure, over the best rate's lift-gas rates for a lifted well;
    where it has no operating point, with the second of _SIZING_CHOKES, a bean of half the diameter. The solve with the
    most oil equivalent gives the mass rate and the wellhead's state at which the fully open choke takes its drawn
    share of the pressure upstream of it (mapping.WellBuilder.size_choke), and the choke is at most the one that the
    well was solved with.

    :param builder: The well's WellBuilder, which takes the choke
    :param state: The state (life.Life.state's keys), as _best_rate takes it
    :param lifted: Whether the well is lifted at the state
    :param w_lg_max: The lift-gas capacity (kg/s)
    :param log: The well's log, to which the solves' records are added, each with `sizing_choke`, its choke's share
        of the largest
    :param base: The labels of the solves' records
    :return: (status, outcomes): `sized`, or, with the largest choke left on the well, `solver_failure` (no
        operating point, and a solver failure among the solves) or `no_operating_point`; and the outcomes of the
        solves with the last choke tried
    """
    failed, outcomes = False, []
    for share in _SIZING_CHOKES:
        K_c, ops, outcomes = builder.widest_choke(share), [], []
        for i, f in enumerate(_BEST_RATE_LIFT if lifted else (0.0,)):
            inp = dict(state, u=1.0, w_lg=f * w_lg_max if lifted else 0.0, lift_on=lifted, p_s=state['p_s_nom'])
            head = []
            rec = _attempt(builder, inp, head)
            rec.update(base, attempt=i, lift_fraction=f, sizing_choke=share)
            log.append(rec)
            outcomes.append(rec['outcome'])
            if rec['outcome'] == 'operating_point':
                ops.append((rec, head[0]))
        if ops:
            top, head = max(ops, key=lambda op: op[0]['q_oe'])
            builder.size_choke(top['WTOT'], head, state['p_s_nom'], K_c)
            _cap_choke(builder, state, lifted, w_lg_max, log, base)
            return 'sized', outcomes
        failed |= 'solver_failure' in outcomes
    builder.widest_choke()
    return ('solver_failure' if failed else 'no_operating_point'), outcomes


def _erosional_ratio(wellhead, c_e) -> float:  # spec: SMP-77
    """The no-slip mixture velocity at the wellhead over the erosional velocity 1.22 C / sqrt(rho_ns) m/s (SMP-75),
    with the no-slip density rho_ns (kg/m3) of the stream there and the well's C."""
    _p, v_g, v_l, alpha, rho_g, rho_l, _T = (float(v) for v in wellhead)
    v_sg, v_sl = alpha * v_g, (1 - alpha) * v_l
    v_m = v_sg + v_sl
    return v_m * math.sqrt((v_sg * rho_g + v_sl * rho_l) / v_m) / (1.22 * c_e)


def _cap_choke(builder, state, lifted, w_lg_max, log, base) -> None:  # spec: SMP-77
    """
    Holds a sized choke below the erosional velocity of the well's tubing (SMP-77, the erosional cap). The well is
    solved at the sizing state with its sized choke fully open, at the best rate's lift-gas rates for a lifted well;
    where the mixture's velocity at the wellhead passes the erosional velocity at any of them (_erosional_ratio), the
    choke is made smaller, by a search on ln K_c, until the largest ratio lies within [_EROSIONAL_LOW, 1], for at most
    _EROSIONAL_STEPS chokes. A capped well's draws carry the flag `choke_at_erosional_rate`; where the search ends
    above the erosional velocity, it keeps the smallest choke tried with an operating point and the flag
    `erosional_cap_unmet`.

    :param builder: The well's WellBuilder, which holds the sized choke and takes the capped one
    :param state: The state (life.Life.state's keys), as _best_rate takes it
    :param lifted: Whether the well is lifted at the state
    :param w_lg_max: The lift-gas capacity (kg/s)
    :param log: The well's log, to which the solves' records are added with the kind `erosional_cap`, `cap_step`, the
        search's step, and `erosional_ratio`, the ratio of an operating point (NaN without one)
    :param base: The labels of the solves' records
    """
    w = builder.w
    flags = [f for f in w['flags'].split(';') if f and f not in _CAP_FLAGS]  # A well sized again is capped again

    def ratio(step) -> float:
        """The largest ratio over the lift-gas rates at the builder's choke, 0 without an operating point."""
        worst = 0.0
        for i, f in enumerate(_BEST_RATE_LIFT if lifted else (0.0,)):
            inp = dict(state, u=1.0, w_lg=f * w_lg_max if lifted else 0.0, lift_on=lifted, p_s=state['p_s_nom'])
            head = []
            rec = _attempt(builder, inp, head)
            r = _erosional_ratio(head[0], w['c_e']) if rec['outcome'] == 'operating_point' else math.nan
            rec.update(base, kind='erosional_cap', attempt=i, lift_fraction=f, cap_step=step, erosional_ratio=r)
            log.append(rec)
            worst = max(worst, 0.0 if math.isnan(r) else r)
        return worst

    r_hi = ratio(0)
    if r_hi <= 1.0:
        w['flags'] = ';'.join(flags)
        return
    K_hi = builder.choke.K_c
    lo = None  # (K_c, ratio) of the largest choke tried at or below the erosional velocity with an operating point
    K_dead = 0.0  # The largest choke tried without an operating point
    aim, slope = math.log((1 + _EROSIONAL_LOW) / 2), 0.5  # The middle of the band; d ln r / d ln K_c, a first guess
    K = K_hi * math.exp((aim - math.log(r_hi)) / slope)
    for step in range(1, _EROSIONAL_STEPS):
        K = builder.take_choke(K)
        r = ratio(step)
        if r > 1.0:  # The slope through the last two chokes above the band, for the next step
            slope = min(max(math.log(r_hi / r) / math.log(K_hi / K), 0.05), 2.0)
            K_hi, r_hi = K, r
        elif r > 0.0:
            lo = (K, r)
            if r >= _EROSIONAL_LOW:
                break
        else:
            K_dead = max(K_dead, K)
        if lo is None and K_dead == 0.0:  # Below the band is not bracketed yet
            K = K_hi * math.exp((aim - math.log(r_hi)) / slope)
        elif lo is None:
            K = math.sqrt(K_dead * K_hi)
        else:  # ln r linear in ln K_c through the bracket
            K_lo, r_lo = lo
            t = (aim - math.log(r_lo)) / (math.log(r_hi) - math.log(r_lo))
            K = K_lo * (K_hi / K_lo) ** min(max(t, 0.1), 0.9)
    if lo is not None:
        builder.take_choke(lo[0])
        flags.append('choke_at_erosional_rate')
    else:
        builder.take_choke(K_hi)
        flags.append('erosional_cap_unmet')
    w['flags'] = ';'.join(flags)


def _sized_best_rate(builder, state, lifted, w_lg_max, floor, log, base) -> dict:  # spec: SMP-77, SMP-103
    """A well's choke sized at a state (_size_choke), and its best rate there with it (_best_rate, whose arguments
    and result these are). A well that the sizing leaves without a choke has the sizing's status and outcomes, and no
    best rate."""
    status, outcomes = _size_choke(builder, state, lifted, w_lg_max, log, dict(base, kind='choke_sizing'))
    if status == 'sized':
        return _best_rate(builder, state, lifted, w_lg_max, floor, log, base)
    if builder.w['fluid_type'] in _OILS and state['wlr'] > _WLR_MAX:  # spec: SMP-102
        status = 'water_cut'
    return dict(status=status, best=math.nan, best_w_lg=math.nan, outcomes=outcomes)


def _lift_flags(flags: str, design, lifted: bool) -> str:  # spec: SMP-90
    """A well's flags with the lift block's flags (_DESIGN_FLAGS) of the lift design that the well uses: a lifted
    well's are its design's, which is the late-stage draw's where the lift starts after the life's start (SMP-90),
    and a well that is not lifted has none. The other flags stay, in their order."""
    own = [f for f in flags.split(';') if f and f not in _DESIGN_FLAGS]
    used = [f for f in design['flags'].split(';') if f in _DESIGN_FLAGS] if lifted else []
    if [f for f in flags.split(';') if f in _DESIGN_FLAGS] == used:
        return flags  # The design is the draw's own, or neither has a flag of it
    return ';'.join(own + used)


def _below_start_hurdle(w0) -> bool:  # spec: SMP-102
    """Whether a well's design rate in oil equivalent lies below its type's start hurdle: it is not drilled."""
    return w0['q_design_oe'] < _START_HURDLE[w0['well_type']]


def _solve_slots(k, draw, builder, log, slot_limit_outcome) -> tuple:  # spec: SMP-105, SMP-106
    """
    The passes over a well's slots: pass a solves attempt a of every slot still without an operating point, in the
    order of the slots, until every slot has one or the well reaches a limit.

    In nsol-2 a well at the redraw limit may hold a dead date that the limit hides, since ten slots without an
    operating point use the thousand redraws before one of them has its hundred attempts. The earliest slot still
    open then takes the rest of its attempts: without an operating point it is a dead date, and with one the well is
    at the redraw limit.

    :param k: The well's draw index
    :param draw: draw(j, a), the inputs of attempt a of slot j
    :param builder: The well's WellBuilder
    :param log: The well's log, to which the attempts' records are added
    :param slot_limit_outcome: The outcome of a well with a slot at the slot limit: `slot_limit` or `dead_date`
    :return: (the accepted record of each slot, by slot; the well's outcome)
    """
    accepted, open_slots = {}, list(range(_N_SLOTS))
    n_redraws = n_failures = 0

    def solve(j, a) -> str:
        """Attempt a of slot j, logged, and accepted where it has an operating point. Returns its outcome."""
        rec = _attempt(builder, draw(j, a))
        rec.update(draw_index=k, kind='slot', slot=j, attempt=a)
        log.append(rec)
        if rec['outcome'] == 'operating_point':
            accepted[j] = rec
        return rec['outcome']

    for a in range(_SLOT_LIMIT):
        for i, j in enumerate(open_slots):
            if a > 0:
                if n_redraws >= _REDRAW_LIMIT:
                    if slot_limit_outcome != 'dead_date':
                        return accepted, 'redraw_limit'
                    first = next(x for x in open_slots if x not in accepted)  # The earliest date still open
                    for b in range(a + 1 if open_slots.index(first) < i else a, _SLOT_LIMIT):
                        outcome = solve(first, b)
                        if outcome == 'operating_point':
                            return accepted, 'redraw_limit'
                        n_failures += outcome == 'solver_failure'
                        if n_failures > _SOLVER_FAILURE_LIMIT:
                            return accepted, 'solver_failure_limit'
                    return accepted, 'dead_date'
                n_redraws += 1
            if solve(j, a) == 'solver_failure':
                n_failures += 1
                if n_failures > _SOLVER_FAILURE_LIMIT:
                    return accepted, 'solver_failure_limit'
        open_slots = [j for j in open_slots if j not in accepted]
        if not open_slots:
            return accepted, 'accepted'
    return accepted, slot_limit_outcome


# --- A well ----------------------------------------------------------------------------------------------------------
def _table(seed, k, kind, tables):  # spec: SMP-112, SMP-116
    """
    A table well's manifest with its PVTTableSet (tables._read_table), or with the status `failed`; None for a well
    of another fluid class. Without a folder of tables the well's table is missing, and so it is for a draw index
    beyond the folder's.

    :raises TableMismatch: where the folder covers the draw index and its record of the well (table_wells._table_well,
        or none for a well of another class) or the well's GOR axis is not the one the run's draws give
    """
    own = _table_well(seed, k)
    if tables is None:
        return None if own is None else dict(status='failed', step='missing', error='no folder of tables was given')
    f_kind, f_seed, (k0, k1), records = _wells_file(tables)
    if (f_kind, f_seed) != (kind, seed) or not k0 <= k < k1:
        return None if own is None else _read_table(tables, k, kind, seed)  # Another's folder, or beyond its range
    if json.loads(json.dumps(own)) != records.get(k):
        raise TableMismatch(
            f'the tables of {tables} were not built from the draws of draw index {k}: its record in wells.json is '
            f'{records.get(k)}, and the draws give {own}. Build the tables again.'
        )
    if own is None:
        return None
    table = _read_table(tables, k, kind, seed)
    if table['status'] == 'ok':
        axis = json.loads(json.dumps(_gor_axis(kind, seed, k, table['hook'])))['gors']
        if axis != table['gors']:
            raise TableMismatch(
                f"the table of draw index {k} in {tables} was built on another GOR axis than the well's life gives "
                f'({len(table["gors"])} GORs from {table["gors"][0]:.6g}, against {len(axis)} from {axis[0]:.6g}). '
                'Build the tables again.'
            )
    return table


def _result(t_start, well, stations, table, info, log=(), accepted=None, nsol=None) -> dict:
    """A well's result (the module's docstring)."""
    if table is not None:
        table = {key: v for key, v in table.items() if key != 'pvt'}  # The set is read again from its file
    samples = [accepted[j] for j in sorted(accepted or {})]
    return dict(
        well=well,
        stations=stations,
        table=table,
        nsol=nsol,
        info=info,
        samples=samples,
        log=list(log),
        seconds=time.perf_counter() - t_start,
    )


def _nsol_info() -> dict:
    """An nsol-2 well's records of its life (the module's docstring), before its scan."""
    return dict(scan=[], T_end=math.nan, T_scan=math.nan, dead_dates=[], p_change=math.nan)


def _info(k, **more) -> dict:
    info = dict(draw_index=k, outcome=None, error=None, best_rate=math.nan, best_rate_w_lg=math.nan)
    return dict(info, best_rate_outcomes=[], not_producible_reason=None, **more)


def _life_summary(life, lift_start, rule) -> dict:
    """An nsol-2 well's life: its parameters, the lift's start and rule after the life scan, and the lift design."""
    design = None if life.design is None else {key: life.design[key] for key in _LIFT_KEYS}
    return dict(life.parameters(), lift_start=lift_start, lift_start_rule=rule, design=design)


def _table_failed(t_start, seed, k, kind, table) -> dict:  # spec: SMP-106
    """The result of a table well without a usable table: it is not solved, and its draws are made without the
    table's values."""
    life = Life(seed, k)
    info = _info(k, outcome='table_failed', error=table.get('error'))
    if table.get('step') == 'class':  # spec: SMP-115. Its reservoir holds no fluid of its class: no well
        info.update(outcome='not_producible', not_producible_reason='no_fluid_of_class')
    if kind == 'sol':
        return _result(t_start, sol2_well(life), life.stations, table, info)
    info.update(_nsol_info())
    nsol = _life_summary(life, life.lift_start, life.lift_start_rule)
    return _result(t_start, life.w0, life.stations, table, info, nsol=nsol)


def _dropped_at_best_rate(info, br) -> bool:  # spec: SMP-103
    """Whether a well's best rate drops it: the outcome and the reason are then set in info."""
    if br['status'] == 'pass':
        return False
    info['outcome'] = 'best_rate_solver_failure' if br['status'] == 'solver_failure' else 'not_producible'
    info['not_producible_reason'] = br['status']
    return True


def _sol2_nominal(w) -> dict:  # spec: SMP-92
    """A sol-2 well's nominal state, as its best rate takes it: life.Life.state's keys but the date and its stage."""
    state = dict(h=w['h'], roughness=w['roughness'], **{key: w[key] for key in _COEF_KEYS})
    state.update(p_r=w['p_r'], gor=w['R_go'], wlr=w['wlr'], wgr=w.get('wgr', math.nan), p_s_nom=w['p_s'])
    return state


def generate_sol2_well(seed: int, k: int, tables=None) -> dict:
    """
    A sol-2 well: its life at its date (SMP-92), its best rate there with the producibility filter (SMP-103), and
    its slots (SMP-105), each a draw around the well's nominal state.

    :param seed: The dataset's seed
    :param k: The well's draw index
    :param tables: The folder of the table fluids' tables built for the dataset and seed (table_build.py); None for
        no tables, with which a table well is `table_failed`
    :return: The well's result (the module's docstring)
    """
    t_start = time.perf_counter()
    table = _table(seed, k, 'sol', tables)
    if table is not None and table['status'] != 'ok':
        return _table_failed(t_start, seed, k, 'sol', table)
    hook, pvt = (table['hook'], table['pvt']) if table is not None else (None, None)
    life = Life(seed, k, fluid=hook, pvt=pvt)
    log, info = [], _info(k)
    try:  # spec: SMP-106. The table refuses a GOR off its axis at the inflow's coefficients of the well's date
        w = sol2_well(life)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        w = sol2_well(Life(seed, k, fluid=hook)) | dict.fromkeys(_COEF_KEYS, math.nan)  # The draws without the table
        return _result(t_start, w, life.stations, table, info)
    if _below_start_hurdle(life.w0):  # spec: SMP-102. Not drilled: nothing is solved
        info.update(outcome='not_producible', not_producible_reason='below_start_hurdle')
        return _result(t_start, w, life.stations, table, info)
    builder = WellBuilder(life.w0, life.stations, design=life.design, pvt=pvt)
    lifted = w['lift'] == 'gas_lift'
    nominal = _sol2_nominal(w)
    try:  # spec: SMP-106
        check(builder.case(dict(nominal, u=1.0, w_lg=0.0, lift_on=lifted, p_s=w['p_s']))[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        return _result(t_start, w, life.stations, table, info)
    floor = _FLOOR[w['well_type']]
    base = dict(draw_index=k, kind='best_rate', slot=-1)
    # spec: SMP-77. The choke is sized at the well's date, and the best rate is the well's with it
    br = _sized_best_rate(builder, nominal, lifted, w['w_lg_max'] if lifted else 0.0, floor, log, base)
    if not lifted and life.oil and br['status'] in ('no_operating_point', 'below_floor'):
        # spec: SMP-103. An oil well that cannot flow naturally takes its life's lift design, as an operator would,
        # and its choke is sized again, with the lift
        w, lifted = sol2_well(life, no_natural_flow=True), True
        base = dict(base, kind='best_rate_lifted')
        br = _sized_best_rate(builder, nominal, lifted, w['w_lg_max'], floor, log, base)
    w.update({key: life.w0[key] for key in ('K_c', 'd_bean', 'beta', 'flags')})  # The sized choke (SMP-77)
    w['flags'] = _lift_flags(w['flags'], life.design, lifted)  # spec: SMP-90
    info.update(best_rate=br['best'], best_rate_w_lg=br['best_w_lg'], best_rate_outcomes=br['outcomes'])
    if _dropped_at_best_rate(info, br):
        return _result(t_start, w, life.stations, table, info, log)
    if lifted and not br['flows_without_lift_gas']:  # spec: SMP-94. No slot of it is drawn without lift gas
        w['flags'] = ';'.join([f for f in w['flags'].split(';') if f] + [_LIFT_GAS_ALWAYS_ON])
    accepted, info['outcome'] = _solve_slots(k, lambda j, a: _sol2_attempt(seed, w, j, a), builder, log, 'slot_limit')
    return _result(t_start, w, life.stations, table, info, log, accepted)


def _life_scan(life, best_rate_at, lift=None) -> tuple:  # spec: SMP-101
    """
    The life scan of an nsol-2 well: its best rate at _N_SCAN dates from 0 to T_life, with the lift that it has at
    each. A point that fails ends the scan, and one whose solves gave a solver failure and no operating point is
    skipped. Where the natural well fails at a point after the first, a well with a lift design is lifted from the
    point before, and the point is solved again.

    :param life: The well's life (life.Life)
    :param best_rate_at: best_rate_at(i, t, lifted), the best rate (_best_rate) at point i, the date t (years), of
        the well natural or lifted
    :param lift: (lift_start, rule), the lift's start (years, or None) and its rule as the scan begins; None for the
        life's own, by its draws
    :return: (scan, first, T_end, lift_start, rule): the points solved, each a dict of `t`, `status`, `best` and
        `lift_on`; the best rate at the first point; the end of the life (years), the last point that passes before
        the first that fails, or T_life if none fails, and None if the first point does not pass; and the lift's
        start (years, or None) and its rule after the scan
    """
    lift_start, rule = (life.lift_start, life.lift_start_rule) if lift is None else lift
    dates = np.linspace(0.0, life.T_life, _N_SCAN)
    scan, first, last_pass, failed = [], None, None, False
    for i in range(_N_SCAN):
        t = float(dates[i])
        lifted = _lift_on(t, lift_start)
        br = best_rate_at(i, t, lifted)
        if i > 0 and not lifted and life.design is not None and br['status'] in ('below_floor', 'no_operating_point'):
            # spec: SMP-90. Rule 3: lift starts at the last scan point before natural flow fails
            lift_start, rule, lifted = float(dates[i - 1]), 'end_of_natural_flow', True
            br = best_rate_at(i, t, lifted)
        scan.append(dict(t=t, status=br['status'], best=br['best'], lift_on=lifted))
        if i == 0:
            first = br
            if br['status'] != 'pass':
                return scan, first, None, lift_start, rule
        if br['status'] == 'pass':
            last_pass = t
        elif br['status'] != 'solver_failure':
            failed = True
            break
    return scan, first, last_pass if failed else life.T_life, lift_start, rule


def _lifted_life(T_end, lift_start) -> bool:  # spec: SMP-101
    """Whether a life that ends at T_end (years) has lifted dates: its lift starts before its end. A life that ends
    where its lift would start is natural, as one that ends before."""
    return lift_start is not None and T_end > lift_start


def _after_dead_date(life, plan, accepted, T_end, lift_start, rule) -> tuple:  # spec: SMP-106, SMP-90
    """
    What a dead date does to an nsol-2 well's life. The dead date is the earliest date of a slot without an operating
    point at the slot limit. Where the well is natural there, has a lift design and the life goes on after it, lift
    starts at that date (`no_natural_flow`). Else the life ends at the last earlier date, at which every slot has an
    operating point, and at 0 where there is none.

    :param life: The well's life (life.Life)
    :param plan: The plan of the pass (samples._nsol2_plan)
    :param accepted: The pass's accepted records, by slot
    :param T_end: The end of the life (years) in the pass
    :param lift_start: The lift's start (years, or None) in the pass, and rule, its rule
    :return: (T_end, lift_start, rule, record): the life for the next pass, and the dead date's record, a dict of
        `t` (years), `slots` (the slots without an operating point) and `action`, `lift_start` or `life_end`
    """
    j_dead = min(j for j in range(_N_SLOTS) if j not in accepted)
    t_dead = plan[j_dead]['t']
    record = dict(t=t_dead, slots=_N_SLOTS - len(accepted))
    if life.design is not None and not plan[j_dead]['lift_on'] and t_dead < T_end:
        return T_end, t_dead, 'no_natural_flow', dict(record, action='lift_start')
    T_end = max((slot['t'] for slot in plan[:j_dead] if slot['t'] < t_dead), default=0.0)
    if not _lifted_life(T_end, lift_start):
        lift_start, rule = None, None
    return T_end, lift_start, rule, dict(record, action='life_end')


def generate_nsol2_well(seed: int, k: int, tables=None) -> dict:
    """
    An nsol-2 well: its life, the life scan that ends it and may start its gas lift (SMP-101, SMP-90), the plan of
    its slots over the life (SMP-97) and the slots (SMP-105), each at its date's state. A dead date starts the lift
    or ends the life there, and the slots are planned and solved again, for at most DEAD_DATE_LIMIT dead dates
    (SMP-106).

    :param seed: The dataset's seed
    :param k: The well's draw index
    :param tables: The folder of the table fluids' tables built for the dataset and seed (table_build.py); None for
        no tables, with which a table well is `table_failed`
    :return: The well's result (the module's docstring)
    """
    t_start = time.perf_counter()
    table = _table(seed, k, 'nsol', tables)
    if table is not None and table['status'] != 'ok':
        return _table_failed(t_start, seed, k, 'nsol', table)
    hook, pvt = (table['hook'], table['pvt']) if table is not None else (None, None)
    life = Life(seed, k, fluid=hook, pvt=pvt)
    w0 = life.w0
    builder = WellBuilder(w0, life.stations, design=life.design, pvt=pvt)
    log, info = [], _info(k, **_nsol_info())
    lift_start, rule = life.lift_start, life.lift_start_rule

    def result(log=(), accepted=None) -> dict:
        """The well's result, with the lift's start and rule as they then are, and the flags of the lift design that
        the well uses (SMP-90)."""
        if life.design is not None:
            w0['flags'] = _lift_flags(w0['flags'], life.design, lift_start is not None)
        return _result(t_start, w0, life.stations, table, info, log, accepted, _life_summary(life, lift_start, rule))

    if _below_start_hurdle(w0):  # spec: SMP-102. Not drilled: nothing is solved
        info.update(outcome='not_producible', not_producible_reason='below_start_hurdle')
        return result()
    try:  # spec: SMP-106
        start = dict(life.state(0.0), u=1.0, w_lg=0.0, lift_on=_lift_on(0.0, lift_start), p_s=w0['p_s'])
        check(builder.case(start)[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        return result()
    floor = _FLOOR[w0['well_type']]
    w_lg_max = life.design['w_lg_max'] if life.design is not None else 0.0

    def from_start(lift_start, rule, scan_kind) -> tuple:  # spec: SMP-77, SMP-101
        """The choke sized at the life's start, where it is the well's for its life, and the life scan, for a lift
        that starts as given: _life_scan's values, with the sizing's status and outcomes as the first point's where
        the sizing leaves the well without a choke."""
        base = dict(draw_index=k, kind='choke_sizing', slot=-1, t=0.0)
        status, outcomes = _size_choke(builder, life.state(0.0), _lift_on(0.0, lift_start), w_lg_max, log, base)
        if status != 'sized':
            return [], dict(status=status, best=math.nan, best_w_lg=math.nan, outcomes=outcomes), None, lift_start, rule

        def best_rate_at(i, t, lifted):
            base = dict(draw_index=k, kind=scan_kind, slot=-1 - i, t=t)
            return _best_rate(builder, life.state(t), lifted, w_lg_max, floor, log, base)

        return _life_scan(life, best_rate_at, (lift_start, rule))

    scan, first, T_end, lift_start, rule = from_start(lift_start, rule, 'scan')
    natural_oil = life.oil and life.design is not None and not _lift_on(0.0, life.lift_start)
    if natural_oil and first['status'] in ('no_operating_point', 'below_floor'):
        # spec: SMP-103. A natural oil that cannot flow at its life's start takes its lift design from the start, as
        # a sol-2 well does at its date, and its choke is sized again, with the lift
        scan, first, T_end, lift_start, rule = from_start(0.0, 'no_natural_flow', 'scan_lifted')
    if T_end is not None and not _lifted_life(T_end, lift_start):  # spec: SMP-52, SMP-90
        lift_start, rule = None, None  # The life ends where its lift would start, or before: no date of it is lifted
    info.update(scan=scan, best_rate=first['best'], best_rate_w_lg=first['best_w_lg'])
    info['best_rate_outcomes'] = first['outcomes']
    if _dropped_at_best_rate(info, first):
        return result(log)
    info['T_scan'] = T_end
    accepted, slots = {}, []
    while True:  # spec: SMP-106. Each pass plans and solves the slots; a dead date changes the life for the next
        info['T_end'] = T_end
        if T_end <= 0.0:  # spec: SMP-101
            info['outcome'] = 'short_life'
            break
        plan, info['p_change'] = _nsol2_plan(life, T_end, lift_start)
        slots = []
        accepted, info['outcome'] = _solve_slots(
            k, lambda j, a: _nsol2_attempt(life, plan, j, a), builder, slots, 'dead_date'
        )
        if info['outcome'] != 'dead_date' or len(info['dead_dates']) == _DEAD_DATE_LIMIT:
            break
        T_end, lift_start, rule, dead = _after_dead_date(life, plan, accepted, T_end, lift_start, rule)
        info['dead_dates'].append(dead)
    log += slots
    return result(log, accepted)
