"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The generation of one v2 well (specs/sampling.md, SMP-101 to SMP-106, and the lift that a solve starts, SMP-90 and
SMP-103): the well's draws and life are mapped to the v2.0.0 configuration and checked, its best rate is solved
against the economic floor, at its date (sol-2) or along its life (nsol-2: the life scan, which ends the life and may
start gas lift), and its slots are solved in passes, each attempt without an operating point redrawn, until every
slot has one or the well reaches a limit. Every attempt is one solve by the Rust core, which takes no start
(SMP-107); no other backend retries it. Nothing here runs many wells or writes a dataset.

A well's result (generate_sol2_well, generate_nsol2_well) is a dict of:

- `well`: the well's draws and labels. sol-2: the life at the well's date (life.sol2_well), with the lift design and
  `lift_start_rule` `no_natural_flow` where the producibility filter lifted it. nsol-2: the initial-stage draws
  (draws.draw_well), which hold what is fixed for the life. A `table_failed` well's are drawn without its table.
- `stations`: the survey, (md, tvd).
- `table`: a table well's manifest (tables.read_table) without its PVTTableSet, with `status` `ok`, or `failed` with
  the `step` and the `error`; None for a well of another fluid class.
- `nsol`: None for a sol-2 well. nsol-2: the life's parameters (life.Life.parameters) with `lift_start` (years; None
  for a well that stays natural) and `lift_start_rule` after the life scan, and `design`, the lift design's draws
  (None for a well that no rule lifts).
- `info`: `draw_index`; `outcome`, the well's (SMP-106): `accepted`, `invalid_input`, `table_failed`,
  `not_producible`, `best_rate_solver_failure`, `slot_limit` (sol-2), `dead_date` (nsol-2), `redraw_limit`,
  `solver_failure_limit` or `short_life` (nsol-2); `error`, the message of an `invalid_input` or `table_failed` well
  (else None); `best_rate` (Sm3/d of oil equivalent) and `best_rate_w_lg` (kg/s), the best rate and its lift-gas
  rate, NaN without an operating point, and `best_rate_outcomes`, the outcomes of its solves (nsol-2: at the start);
  `not_producible_reason`, the best rate's status of a well dropped for it (else None). nsol-2 adds `scan`, the life
  scan's points up to the one that ends the life, each a dict of `t` (years), `status`, `best` and `lift_on`;
  `T_end` (years), the end of the life (NaN for a well dropped before the scan ends it); and `p_change`, the
  well's probability of a set-point change (NaN before the plan is drawn).
- `samples`: the accepted attempts' records, by slot: all N_SLOTS of an accepted well.
- `log`: the record of every attempt, in the order solved: the best rate's, the life scan's and the slots'.
- `seconds`: the wall time.

An attempt's record is a flat dict of:

- its inputs (samples.sol2_attempt or samples.nsol2_attempt; a best-rate solve's are the state's, with `p_s_nom`);
- `draw_index`; `kind`, `best_rate`, `best_rate_lifted` (a sol-2 well's best rate with the filter's lift), `scan` or
  `slot`; `slot`, j for a slot, -1 for a best-rate solve and -1 - i for the life scan's point i; `attempt`, a for a
  slot, else the index of the lift-gas rate, whose share of the capacity is `lift_fraction`;
- `outcome` and `reason` (SMP-104), and `error`, an exception's message (else None);
- `n_roots` and `n_stable`, the sizes of the root set, and `several_stable`;
- the search's counts `marches`, `non_finite` (scan points whose march failed) and `rejected` (sign changes of R
  not accepted as roots), NaN without a search, and the times `t_build` and `t_solve` (s);
- for an operating point, its FEATURES and `q_oe`, the oil equivalent (Sm3/d).
"""

import math
import time

import numpy as np

from manywells.configurations import V2, check
from manywells.datasets import schema
from manywells.datasets.rows import root_features
from manywells.sampling.v2.draws import draw_well
from manywells.sampling.v2.inflow import COEF_KEYS
from manywells.sampling.v2.life import _LIFT_KEYS, WLR_MAX, Life, lift_on, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.reservoir import OILS
from manywells.sampling.v2.samples import N_SLOTS, nsol2_attempt, nsol2_plan, sol2_attempt
from manywells.sampling.v2.tables import TABLE_CLASSES, read_table
from manywells.solvers.rust import RustRootFinder

FLOOR = {'platform': 1.6, 'onshore': 1.6, 'subsea': 20.0}  # Sm3/d of oil equivalent, by well type  # spec: SMP-102
FLOOR_MARGIN = 0.5  # A well is not produced below this share of its floor  # spec: SMP-102
SLOT_LIMIT, REDRAW_LIMIT, SOLVER_FAILURE_LIMIT = 50, 400, 10  # spec: SMP-106
# The operating point's columns of a sample: SMP-30's features but the inputs
FEATURES = tuple(key for key in schema.FEATURES if key not in ('CHK', 'PDC', 'WGL'))
_BEST_RATE_LIFT = (0.0, 0.5, 1.0)  # The best rate's lift-gas rates, shares of the capacity  # spec: SMP-103
_N_SCAN = 21  # Dates of the life scan  # spec: SMP-101
_NO_SEARCH = dict(n_roots=0, n_stable=0, several_stable=False, marches=math.nan, non_finite=math.nan)
_NO_SEARCH |= dict(rejected=math.nan, t_solve=0.0)


# --- One attempt -----------------------------------------------------------------------------------------------------
def _features(root, bc, fluid) -> dict:  # spec: SMP-30
    """The operating point's features of a sample (FEATURES): the root's, and the stream's mass fractions."""
    x_w = 1 - fluid.f_o_in_liquid
    row = root_features(root, bc, fluid.rho_o, fluid.rho_w, fluid.rho_g, x_w)
    f_g = fluid.f_g
    row |= {'FGAS': f_g, 'FOIL': (1 - f_g) * (1 - x_w), 'FWAT': (1 - f_g) * x_w}
    return {key: row[key] for key in FEATURES}


def _valid_row(row) -> bool:  # spec: SMP-104
    """Whether every number of a row is finite, its liquid and gas rates are not negative and its mass fractions lie
    in [0, 1]. The lift gas is at least 0 by its draw."""
    if not all(math.isfinite(v) for key, v in row.items() if key not in ('CHOKED', 'FRBH', 'FRWH')):
        return False
    return row['WLIQ'] >= 0 and row['WGAS'] >= 0 and all(0 <= row[key] <= 1 for key in ('FGAS', 'FOIL', 'FWAT'))


def _oil_equivalent(row) -> float:  # spec: SMP-103
    """Sm3/d of oil, and of the formation gas at 1000 Sm3 per Sm3 (v2: scope and conventions)."""
    return 24 * (row['QOIL'] + row['QGAS'] / 1000)


def _solve(wp, bc) -> dict:  # spec: SMP-104
    """
    One solve of a well at its boundary conditions by the Rust core, and its outcome.

    :return: A dict of `outcome`, `reason` and `error`; `n_roots`, `n_stable` and `several_stable`; the search's
        counts `marches`, `non_finite` and `rejected` (NaN without a search); `t_solve` (s); and `row`, the
        operating point's features (None without one)
    """
    out = dict(outcome=None, reason=None, error=None, **_NO_SEARCH, row=None)
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
    except Exception as e:  # noqa: BLE001 - any exception of a solve is a solver failure
        out.update(outcome='solver_failure', reason='exception', error=f'{type(e).__name__}: {e}'[:300])
        out['t_solve'] = time.perf_counter() - t0
        return out
    counts = finder.counts
    out.update(marches=counts['marches'], non_finite=counts['non_finite'], rejected=counts['rejected'])
    out.update(n_roots=len(rs.roots), n_stable=sum(r.label == 'stable' for r in rs.roots))
    out['several_stable'] = bool(rs.several_stable)
    op = rs.operating_point
    if op is not None:
        row = _features(op, bc, wp.fluid)
        if _valid_row(row):
            out.update(outcome='operating_point', row=row)
        else:
            out.update(outcome='solver_failure', reason='invalid_row')
    elif rs.roots:  # A continuous R(p_0) has a stable root for every unstable one: the search missed it
        out.update(outcome='solver_failure', reason='unstable_only')
    elif counts['rejected'] > 0:  # A sign change of R that the search could not resolve to a root
        out.update(outcome='solver_failure', reason='incomplete_scan')
    else:  # No root and no rejected sign change, whatever the non-finite scan points
        out.update(outcome='no_operating_point', reason='no_root')
    out['t_solve'] = time.perf_counter() - t0
    return out


def _attempt(builder, inp) -> dict:  # spec: SMP-104
    """The record of an attempt (the module's docstring), without its labels: the inputs inp (mapping.WellBuilder.case)
    of the well of builder, solved once."""
    t0 = time.perf_counter()
    rec = dict(inp)
    try:
        wp, bc = builder.case(inp)
    except ValueError as e:  # The model refuses the inputs: a solver failure
        res = dict(outcome='solver_failure', reason='exception', error=f'build: {e}'[:300], **_NO_SEARCH, row=None)
        t_build = time.perf_counter() - t0
    else:
        t_build = time.perf_counter() - t0
        res = _solve(wp, bc)
    row = res.pop('row')
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
        an operating point; and `outcomes`, the solves' outcomes
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
    if builder.w['fluid_type'] in OILS and state['wlr'] > WLR_MAX:  # spec: SMP-102
        out['status'] = 'water_cut'
    elif ops:
        top = max(ops, key=lambda rec: rec['q_oe'])
        out.update(best=top['q_oe'], best_w_lg=top['w_lg'])
        out['status'] = 'pass' if top['q_oe'] >= FLOOR_MARGIN * floor else 'below_floor'  # spec: SMP-102
    elif any(rec['outcome'] == 'solver_failure' for rec in recs):
        out['status'] = 'solver_failure'
    else:
        out['status'] = 'no_operating_point'
    return out


def _solve_slots(k, draw, builder, log, slot_limit_outcome) -> tuple:  # spec: SMP-105, SMP-106
    """
    The passes over a well's slots: pass a solves attempt a of every slot still without an operating point, in the
    order of the slots, until every slot has one or the well reaches a limit.

    :param k: The well's draw index
    :param draw: draw(j, a), the inputs of attempt a of slot j
    :param builder: The well's WellBuilder
    :param log: The well's log, to which the attempts' records are added
    :param slot_limit_outcome: The outcome of a well with a slot at the slot limit: `slot_limit` or `dead_date`
    :return: (the accepted record of each slot, by slot; the well's outcome)
    """
    accepted, open_slots = {}, list(range(N_SLOTS))
    n_redraws = n_failures = 0
    for a in range(SLOT_LIMIT):
        for j in open_slots:
            if a > 0:
                if n_redraws >= REDRAW_LIMIT:
                    return accepted, 'redraw_limit'
                n_redraws += 1
            rec = _attempt(builder, draw(j, a))
            rec.update(draw_index=k, kind='slot', slot=j, attempt=a)
            log.append(rec)
            if rec['outcome'] == 'operating_point':
                accepted[j] = rec
            elif rec['outcome'] == 'solver_failure':
                n_failures += 1
                if n_failures > SOLVER_FAILURE_LIMIT:
                    return accepted, 'solver_failure_limit'
        open_slots = [j for j in open_slots if j not in accepted]
        if not open_slots:
            return accepted, 'accepted'
    return accepted, slot_limit_outcome


# --- A well ----------------------------------------------------------------------------------------------------------
def _table(seed, k, kind, tables):  # spec: SMP-112, SMP-116
    """A table well's manifest with its PVTTableSet (tables.read_table), or with the status `failed`; None for a well
    of another fluid class. Without a folder of tables the well's table is missing."""
    if draw_well(seed, k)[0]['fluid_class'] not in TABLE_CLASSES:
        return None
    if tables is None:
        return dict(status='failed', step='missing', error='no folder of tables was given')
    return read_table(tables, k, kind, seed)


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
    if kind == 'sol':
        return _result(t_start, sol2_well(life), life.stations, table, info)
    info.update(scan=[], T_end=math.nan, p_change=math.nan)
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
    state = dict(h=w['h'], roughness=w['roughness'], **{key: w[key] for key in COEF_KEYS})
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
    w = sol2_well(life)
    builder = WellBuilder(life.w0, life.stations, design=life.design, pvt=pvt)
    lifted = w['lift'] == 'gas_lift'
    log, info = [], _info(k)
    nominal = _sol2_nominal(w)
    try:  # spec: SMP-106
        check(builder.case(dict(nominal, u=1.0, w_lg=0.0, lift_on=lifted, p_s=w['p_s']))[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        return _result(t_start, w, life.stations, table, info)
    floor = FLOOR[w['well_type']]
    base = dict(draw_index=k, kind='best_rate', slot=-1)
    br = _best_rate(builder, nominal, lifted, w['w_lg_max'] if lifted else 0.0, floor, log, base)
    if not lifted and life.oil and br['status'] in ('no_operating_point', 'below_floor'):
        # spec: SMP-103. An oil well that cannot flow naturally takes its life's lift design, as an operator would
        w, lifted = sol2_well(life, no_natural_flow=True), True
        br = _best_rate(builder, nominal, lifted, w['w_lg_max'], floor, log, dict(base, kind='best_rate_lifted'))
    info.update(best_rate=br['best'], best_rate_w_lg=br['best_w_lg'], best_rate_outcomes=br['outcomes'])
    if _dropped_at_best_rate(info, br):
        return _result(t_start, w, life.stations, table, info, log)
    accepted, info['outcome'] = _solve_slots(k, lambda j, a: sol2_attempt(seed, w, j, a), builder, log, 'slot_limit')
    return _result(t_start, w, life.stations, table, info, log, accepted)


def _life_scan(life, best_rate_at) -> tuple:  # spec: SMP-101
    """
    The life scan of an nsol-2 well: its best rate at _N_SCAN dates from 0 to T_life, with the lift that it has at
    each. A point that fails ends the scan, and one whose solves gave a solver failure and no operating point is
    skipped. Where the natural well fails at a point after the first, a well with a lift design is lifted from the
    point before, and the point is solved again.

    :param life: The well's life (life.Life)
    :param best_rate_at: best_rate_at(i, t, lifted), the best rate (_best_rate) at point i, the date t (years), of
        the well natural or lifted
    :return: (scan, first, T_end, lift_start, rule): the points solved, each a dict of `t`, `status`, `best` and
        `lift_on`; the best rate at the first point; the end of the life (years), the last point that passes before
        the first that fails, or T_life if none fails, and None if the first point does not pass; and the lift's
        start (years, or None) and its rule after the scan
    """
    lift_start, rule = life.lift_start, life.lift_start_rule
    dates = np.linspace(0.0, life.T_life, _N_SCAN)
    scan, first, last_pass, failed = [], None, None, False
    for i in range(_N_SCAN):
        t = float(dates[i])
        lifted = lift_on(t, lift_start)
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


def generate_nsol2_well(seed: int, k: int, tables=None) -> dict:
    """
    An nsol-2 well: its life, the life scan that ends it and may start its gas lift (SMP-101, SMP-90), the plan of
    its slots over the life (SMP-97) and the slots (SMP-105), each at its date's state.

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
    log, info = [], _info(k, scan=[], T_end=math.nan, p_change=math.nan)
    lift_start, rule = life.lift_start, life.lift_start_rule
    try:  # spec: SMP-106
        start = dict(life.state(0.0), u=1.0, w_lg=0.0, lift_on=lift_on(0.0, lift_start), p_s=w0['p_s'])
        check(builder.case(start)[0], V2)
    except ValueError as e:
        info.update(outcome='invalid_input', error=str(e)[:300])
        return _result(t_start, w0, life.stations, table, info, nsol=_life_summary(life, lift_start, rule))
    floor = FLOOR[w0['well_type']]
    w_lg_max = life.design['w_lg_max'] if life.design is not None else 0.0

    def best_rate_at(i, t, lifted):
        base = dict(draw_index=k, kind='scan', slot=-1 - i, t=t)
        return _best_rate(builder, life.state(t), lifted, w_lg_max, floor, log, base)

    scan, first, T_end, lift_start, rule = _life_scan(life, best_rate_at)
    nsol = _life_summary(life, lift_start, rule)
    info.update(scan=scan, best_rate=first['best'], best_rate_w_lg=first['best_w_lg'])
    info['best_rate_outcomes'] = first['outcomes']
    if _dropped_at_best_rate(info, first):
        return _result(t_start, w0, life.stations, table, info, log, nsol=nsol)
    info['T_end'] = T_end
    if T_end <= 0.0:  # spec: SMP-101
        info['outcome'] = 'short_life'
        return _result(t_start, w0, life.stations, table, info, log, nsol=nsol)
    plan, info['p_change'] = nsol2_plan(life, T_end, lift_start)
    accepted, info['outcome'] = _solve_slots(k, lambda j, a: nsol2_attempt(life, plan, j, a), builder, log, 'dead_date')
    return _result(t_start, w0, life.stations, table, info, log, accepted, nsol)
