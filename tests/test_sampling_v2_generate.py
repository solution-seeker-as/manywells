"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the generation of one v2 well (manywells.sampling.v2.generate; specs/sampling.md, SMP-101 to SMP-106 and
the lift that a solve starts, SMP-90 and SMP-103): an attempt's outcome and reason, the best rate against the floor,
the producibility filter, the life scan, the slots' passes and the limits, and the wells that are dropped before a
solve. The fast tests put stand-ins in place of the solves (a root set, or an attempt's outcome), so that each rule
is tested on the case it covers; the slow tests generate a real well of each dataset end to end.
"""

import dataclasses
import json
import math
import types

import numpy as np
import pytest

from manywells.configurations import V2
from manywells.datasets import schema
from manywells.sampling.v2 import generate
from manywells.sampling.v2 import run as runner
from manywells.sampling.v2 import table_wells as tw
from manywells.sampling.v2.generate import (
    FEATURES,
    _DEAD_DATE_LIMIT,
    _FLOOR,
    _FLOOR_MARGIN,
    _REDRAW_LIMIT,
    _SIZING_CHOKES,
    _SLOT_LIMIT,
    _SOLVER_FAILURE_LIMIT,
    _START_HURDLE,
    generate_nsol2_well,
    generate_sol2_well,
)
from manywells.sampling.v2.inflow import _COEF_KEYS
from manywells.sampling.v2.life import Life, _lift_on, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.samples import _LIFT_GAS_ALWAYS_ON, _N_SLOTS, _nsol2_plan, _sol2_attempt
from manywells.sampling.v2.tables import _TABLE_CLASSES, TableMismatch
from manywells.solution import Root, RootSet

from .sampling_v2_wells import SEED, drawn_lives, drawn_sol2_wells

RECORD = (  # The keys of an attempt's record besides its inputs and labels (the module's docstring)
    'outcome reason error n_roots n_stable several_stable marches non_finite rejected rejected_at_edge off_table t_solve '
    't_build'.split()
)
INFO = 'draw_index outcome error best_rate best_rate_w_lg best_rate_outcomes not_producible_reason'.split()
NSOL_INFO = ['scan', 'T_end', 'T_scan', 'dead_dates', 'p_change']  # An nsol-2 well's, after INFO
RESULT = ['well', 'stations', 'table', 'nsol', 'info', 'samples', 'log', 'seconds']


def nan_or(a, b) -> bool:
    return a == b or (a != a and b != b)


# --- Stand-ins ---------------------------------------------------------------------------------------------------------
def drilled_lives():
    """The lives of the tests' seed whose wells pass the start hurdle (SMP-102)."""
    return [lf for lf in drawn_lives() if not generate._below_start_hurdle(lf.w0)]


def a_sol2_well(*, oil=None, lifted=None, table=False):
    """The first life of the tests' seed that passes the start hurdle, and its sol-2 well, with the properties asked
    for."""
    for lf, w in zip(drawn_lives(), drawn_sol2_wells()):
        if generate._below_start_hurdle(lf.w0):
            continue
        if oil is not None and lf.oil != oil:
            continue
        if lifted is not None and (w['lift'] == 'gas_lift') != lifted:
            continue
        if (w['fluid_class'] in _TABLE_CLASSES) != table:
            continue
        return lf, w
    raise LookupError('no such well among the drawn lives')


def solve_inputs(lf, w) -> tuple:
    """The WellProperties and BoundaryConditions of a sol-2 well's nominal state, with the choke open."""
    inp = dict(generate._sol2_nominal(w), u=1.0, w_lg=0.0, lift_on=False, p_s=w['p_s'])
    return WellBuilder(lf.w0, lf.stations, design=lf.design).case(inp)


def a_root(label='stable', w_res=5.0, w_g_res=0.5, p_0=200.0, n_points=101) -> Root:
    """A root of n_points points, the pressure falling from p_0 to a tenth of it."""
    X = np.tile([0.0, 3.0, 1.0, 0.3, 50.0, 800.0, 350.0], (n_points, 1))
    X[:, 0] = np.linspace(p_0, 0.1 * p_0, n_points)
    slope = -1.0 if label == 'stable' else 1.0
    regimes = ('bubbly',) * n_points
    return Root(x=X.ravel(), label=label, slope=slope, choked=False, flow_regime=regimes, w_res=w_res, w_g_res=w_g_res)


class StandInFinder:
    """In place of RustRootFinder: each search returns the roots and the counts given, or raises error."""

    def __init__(self, roots=(), rejected=0, non_finite=0, error=None, at_edge=0):
        self.roots, self.error, self.wells = roots, error, []
        self.counts = dict(marches=7, states=700, rejected=rejected, non_finite=non_finite, edge_refinements=0)
        self.counts['rejected_at_edge'] = at_edge

    def __call__(self, wp):
        self.wells.append(wp)
        if self.error is not None:
            raise self.error
        return self

    def find(self, bc, x_guess=None):
        return RootSet.of(self.roots)


WELLHEAD = [21.0, 4.0, 2.0, 0.6, 25.0, 780.0, 330.0]  # A stand-in state at the wellhead, in the state vector's order


def stand_in_attempt(outcome_of):
    """In place of generate._attempt: outcome_of(inp) gives an attempt's outcome and its oil equivalent (Sm3/d). An
    operating point has a total mass rate of 5 kg/s, and the stand-in state at the wellhead."""
    calls = []

    def attempt(builder, inp, wellhead=None):
        calls.append(inp)
        outcome, q_oe = outcome_of(inp)
        rec = dict(inp, outcome=outcome, reason=None, error=None)
        if outcome == 'operating_point':
            rec.update(q_oe=q_oe, WTOT=5.0)
        if wellhead is not None:
            wellhead.append(WELLHEAD if outcome == 'operating_point' else None)
        return rec

    return attempt, calls


# --- SMP-104: an attempt's outcome ------------------------------------------------------------------------------------
def test_an_operating_point(monkeypatch):
    """The stable root of a valid row is the operating point: its features and the search's counts are recorded."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    finder = StandInFinder([a_root('unstable', p_0=150.0), a_root(p_0=220.0)], non_finite=3)
    monkeypatch.setattr(generate, 'RustRootFinder', finder)
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == ('operating_point', None, None)
    assert (out['n_roots'], out['n_stable'], out['several_stable']) == (2, 1, False)
    assert (out['marches'], out['non_finite'], out['rejected']) == (7, 3, 0) and out['off_table'] is False
    row = out['row']
    assert list(row) == list(FEATURES) and row['PBH'] == 220.0 and row['WLIQ'] == 5.0 and row['WGAS'] == 0.5
    assert math.isclose(row['FGAS'] + row['FOIL'] + row['FWAT'], 1.0, rel_tol=1e-12)
    assert finder.wells == [wp]


@pytest.mark.parametrize(
    'roots, rejected, at_edge, outcome, reason',
    [
        ([], 0, 0, 'no_operating_point', 'no_root'),
        ([], 2, 0, 'solver_failure', 'incomplete_scan'),
        ([], 2, 1, 'solver_failure', 'incomplete_scan'),
        ([], 2, 2, 'no_operating_point', 'no_root'),
        (['unstable'], 0, 0, 'solver_failure', 'unstable_only'),
        (['unstable'], 1, 1, 'solver_failure', 'unstable_only'),
        (['stable', 'stable'], 0, 0, 'operating_point', None),
    ],
)
def test_the_outcome_of_a_root_set(monkeypatch, roots, rejected, at_edge, outcome, reason):
    """SMP-104: no root is no operating point unless the search rejected a sign change away from the edge of the
    states the march admits (D-138), whatever its non-finite scan points; unstable roots alone are a missed root; of
    several stable roots, the one with the lowest p_0 (SOL-6)."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    found = [a_root(label, p_0=200.0 + 10 * i) for i, label in enumerate(roots)]
    monkeypatch.setattr(
        generate, 'RustRootFinder', StandInFinder(found, rejected=rejected, non_finite=4, at_edge=at_edge)
    )
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason']) == (outcome, reason)
    assert (out['n_roots'], out['n_stable']) == (len(roots), roots.count('stable'))
    assert out['several_stable'] == (roots.count('stable') > 1)
    assert (out['row'] is not None) == (outcome == 'operating_point')
    if outcome == 'operating_point':
        assert out['row']['PBH'] == 200.0


@pytest.mark.parametrize(
    'root',
    [
        a_root(w_res=-1e-3),  # a negative liquid rate
        a_root(w_g_res=-1e-3),  # a negative gas rate
        a_root(p_0=math.nan),  # a non-finite value
    ],
)
def test_an_invalid_row_is_a_solver_failure(monkeypatch, root):
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder([root]))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['row']) == ('solver_failure', 'invalid_row', None)
    assert (out['n_roots'], out['n_stable']) == (1, 1)


def test_a_row_with_a_fraction_outside_zero_to_one_is_invalid():
    row = {key: 0.5 for key in FEATURES} | dict(CHOKED=False, FRBH='bubbly', FRWH='annular')
    assert generate._valid_row(row)
    for key in ('FGAS', 'FOIL', 'FWAT'):
        assert not generate._valid_row(row | {key: 1.0 + 1e-12}) and not generate._valid_row(row | {key: -1e-12})
    assert generate._valid_row(row | dict(WLIQ=0.0, WGAS=0.0))
    assert not generate._valid_row(row | dict(QOIL=math.inf))


def test_an_exception_is_a_solver_failure(monkeypatch):
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=RuntimeError('the core stopped')))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == (
        'solver_failure',
        'exception',
        'RuntimeError: the core stopped',
    )
    assert out['n_roots'] == 0 and all(math.isnan(out[key]) for key in ('marches', 'non_finite', 'rejected'))


PanicException = type('PanicException', (BaseException,), {'__module__': 'pyo3_runtime'})  # PyO3's, as it raises it


def test_a_panic_of_the_core_is_a_solver_failure(monkeypatch):
    """A panic follows from the inputs, as an exception does: it is the attempt's solver failure, and reproducible."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=PanicException('index out of bounds')))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == (
        'solver_failure',
        'exception',
        'PanicException: index out of bounds',
    )


@pytest.mark.parametrize('error', [MemoryError(), KeyboardInterrupt(), type('Other', (BaseException,), {})()])
def test_a_memory_error_or_an_interrupt_is_not_an_outcome(monkeypatch, error):
    """A MemoryError is the process's cap, not the inputs': it goes on to the run (run.py), as an interrupt does."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=error))
    with pytest.raises(type(error)):
        generate._solve(wp, bc)


def test_no_solve_where_p_r_is_not_above_p_s(monkeypatch):
    """SMP-96 and SMP-104: an attempt with p_r <= p_s has no operating point, without a solve."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    finder = StandInFinder([a_root()])
    monkeypatch.setattr(generate, 'RustRootFinder', finder)
    for p_s in (bc.p_r, bc.p_r + 1.0):
        out = generate._solve(wp, dataclasses.replace(bc, p_s=p_s))
        assert (out['outcome'], out['reason'], out['n_roots'], out['t_solve']) == (
            'no_operating_point',
            'p_r_le_p_s',
            0,
            0.0,
        )
    assert finder.wells == []


def test_an_attempts_record(monkeypatch):
    """The record holds the inputs and the solve's outcome, counts and times; an operating point's features and oil
    equivalent; and an input that the model refuses is a solver failure."""
    lf, w = a_sol2_well(oil=True)
    builder = WellBuilder(lf.w0, lf.stations, design=lf.design)
    inp = _sol2_attempt(SEED, w, 3)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder([a_root()]))
    rec = generate._attempt(builder, inp)
    assert list(rec) == [*inp, *RECORD, *FEATURES, 'q_oe']
    assert all(nan_or(rec[key], v) for key, v in inp.items())
    assert rec['q_oe'] == 24 * (rec['QOIL'] + rec['QGAS'] / 1000)
    assert rec['t_build'] >= 0 and rec['t_solve'] >= 0

    refused = generate._attempt(builder, dict(inp, h=-1.0))
    assert list(refused) == [*inp, *RECORD]
    assert (refused['outcome'], refused['reason']) == ('solver_failure', 'exception')
    assert refused['error'].startswith('build: ') and refused['n_roots'] == 0 and math.isnan(refused['marches'])


def test_features_are_the_samples_operating_point():
    """The operating point's columns are SMP-30's features but the inputs CHK, PDC and WGL."""
    assert FEATURES == tuple(key for key in schema.FEATURES if key not in ('CHK', 'PDC', 'WGL'))


# --- SMP-102, SMP-103: the best rate ----------------------------------------------------------------------------------
def test_the_floor():
    assert _FLOOR == {'platform': 1.6, 'onshore': 1.6, 'subsea': 20.0} and _FLOOR_MARGIN == 0.5


def best_rate(monkeypatch, outcomes, lifted=True, fluid_type='black_oil', wlr=0.3, floor=1.6, w_lg_max=2.0):
    """The best rate with each solve's (outcome, q_oe) given in turn, and its log and the solves' inputs."""
    it = iter(outcomes)
    attempt, calls = stand_in_attempt(lambda inp: next(it))
    monkeypatch.setattr(generate, '_attempt', attempt)
    builder = types.SimpleNamespace(w=dict(fluid_type=fluid_type))
    state = dict(p_r=200.0, gor=100.0, wlr=wlr, wgr=math.nan, p_s_nom=20.0, h=10.0, roughness=1e-5)
    log = []
    out = generate._best_rate(builder, state, lifted, w_lg_max, floor, log, dict(draw_index=4, kind='best_rate'))
    return out, log, calls


def test_the_best_rate_at_three_lift_gas_rates(monkeypatch):
    """SMP-103: at a fully open choke and the nominal downstream pressure, the most oil equivalent over the lift gas
    at 0, 1/2 and 1 of the capacity; each solve is logged with its labels."""
    out, log, calls = best_rate(
        monkeypatch, [('no_operating_point', None), ('operating_point', 5.0), ('operating_point', 3.0)]
    )
    outcomes = ['no_operating_point', 'operating_point', 'operating_point']
    assert out == dict(best=5.0, best_w_lg=1.0, outcomes=outcomes, flows_without_lift_gas=False, status='pass')
    assert [(c['u'], c['p_s'], c['w_lg'], c['lift_on']) for c in calls] == [
        (1.0, 20.0, f * 2.0, True) for f in (0, 0.5, 1)
    ]
    assert [(r['draw_index'], r['kind'], r['attempt'], r['lift_fraction']) for r in log] == [
        (4, 'best_rate', i, f) for i, f in enumerate((0.0, 0.5, 1.0))
    ]


def test_a_natural_well_is_solved_without_lift_gas(monkeypatch):
    out, log, calls = best_rate(monkeypatch, [('operating_point', 2.0)], lifted=False)
    assert out['status'] == 'pass' and len(calls) == 1 and (calls[0]['w_lg'], calls[0]['lift_on']) == (0.0, False)


@pytest.mark.parametrize(
    'q_oe, floor, status',
    [(0.8, 1.6, 'pass'), (0.8 - 1e-12, 1.6, 'below_floor'), (10.0, 20.0, 'pass'), (9.99, 20.0, 'below_floor')],
)
def test_the_best_rate_against_half_the_floor(monkeypatch, q_oe, floor, status):
    """SMP-102: a well is produced down to half its floor."""
    out, _, _ = best_rate(monkeypatch, [('operating_point', q_oe)], lifted=False, floor=floor)
    assert (out['status'], out['best']) == (status, q_oe)


@pytest.mark.parametrize(
    'outcomes, status',
    [
        (['no_operating_point', 'solver_failure', 'no_operating_point'], 'solver_failure'),
        (['no_operating_point'] * 3, 'no_operating_point'),
        (['solver_failure', 'operating_point', 'solver_failure'], 'pass'),
    ],
)
def test_the_best_rates_status_without_an_operating_point(monkeypatch, outcomes, status):
    out, _, _ = best_rate(monkeypatch, [(o, 4.0) for o in outcomes])
    assert out['status'] == status and (status == 'pass') == (out['best'] == out['best'])


def test_an_oil_above_the_water_cut_is_not_produced(monkeypatch):
    """SMP-102: an oil whose WLR is above 0.98 is not produced, whatever its rate, and has no best rate; the cap is an
    oil's only."""
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.981)
    assert out['status'] == 'water_cut' and math.isnan(out['best']) and out['outcomes'] == ['operating_point']
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.98)
    assert out['status'] == 'pass'
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.99, fluid_type='gas')
    assert out['status'] == 'pass'


# --- SMP-105, SMP-106: the slots' passes and the limits ---------------------------------------------------------------
def solve_slots(monkeypatch, fails, outcome='slot_limit'):
    """The passes over the slots with fails(j, a) the outcome of a failed attempt, or None for an operating point."""
    attempt, _ = stand_in_attempt(lambda inp: (fails(inp['j'], inp['a']) or 'operating_point', 1.0))
    monkeypatch.setattr(generate, '_attempt', attempt)
    log = []
    accepted, result = generate._solve_slots(3, lambda j, a: dict(j=j, a=a), None, log, outcome)
    assert all((r['draw_index'], r['kind'], r['slot'], r['attempt']) == (3, 'slot', r['j'], r['a']) for r in log)
    assert all(
        accepted[j] is next(r for r in log if r['slot'] == j and r['outcome'] == 'operating_point') for j in accepted
    )
    return accepted, result, [(r['slot'], r['attempt']) for r in log]


def test_every_slot_at_its_first_attempt(monkeypatch):
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: None)
    assert (result, sorted(accepted), order) == ('accepted', list(range(_N_SLOTS)), [(j, 0) for j in range(_N_SLOTS)])


def test_the_passes(monkeypatch):
    """SMP-105: pass a solves attempt a of each slot still without an operating point, in the order of the slots."""
    needs = {5: 1, 17: 3, 2: 2}  # The attempts that slot j needs without an operating point
    accepted, result, order = solve_slots(
        monkeypatch, lambda j, a: 'no_operating_point' if a < needs.get(j, 0) else None
    )
    assert result == 'accepted' and sorted(accepted) == list(range(_N_SLOTS))
    assert order == [(j, 0) for j in range(_N_SLOTS)] + [(2, 1), (5, 1), (17, 1), (2, 2), (17, 2), (17, 3)]
    assert {j: accepted[j]['attempt'] for j in needs} == {5: 1, 17: 3, 2: 2}


def test_a_sign_change_at_a_choked_cell_is_no_root():
    """D-138: seed 2026's well 323 at slot 16, a tail oil 2.3 bar above its separator, whose scan brackets a jump of
    R where the march stops at a choked cell (SOL-8): the core counts the rejected sign change as at the edge, and
    the attempt has no operating point."""
    life = Life(2026, 323)
    w = sol2_well(life)
    w = dict(w, p_s=life.state(w['date'])['p_s_nom'])  # The separator's factor at its date, a tail well's before D-159
    wp, bc = WellBuilder(life.w0, life.stations, design=life.design).case(_sol2_attempt(2026, w, 16, 0))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason']) == ('no_operating_point', 'no_root')
    assert out['rejected'] == out['rejected_at_edge'] == 1 and out['n_roots'] == 0


def test_the_slot_limit(monkeypatch):
    """SMP-106: a slot without an operating point after 100 attempts (D-137) drops the well, as `slot_limit` in sol-2
    and as `dead_date` in nsol-2; every other slot keeps its operating point."""
    for name in ('slot_limit', 'dead_date'):
        accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if j == 7 else None, name)
        assert result == name and _SLOT_LIMIT == 100 and 7 not in accepted and len(accepted) == _N_SLOTS - 1
        assert order[_N_SLOTS:] == [(7, a) for a in range(1, _SLOT_LIMIT)]


def test_the_redraw_limit(monkeypatch):
    """SMP-106: a well may redraw 1000 times over its slots (D-137); a 1001st redraw drops it."""
    assert _REDRAW_LIMIT == 1000
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if a < 5 else None)
    assert result == 'accepted' and len(order) == _N_SLOTS + _REDRAW_LIMIT  # 5 redraws for each of 200 slots
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if a < 5 or j == 0 else None)
    assert result == 'redraw_limit' and len(order) == _N_SLOTS + _REDRAW_LIMIT and order[-1] == (_N_SLOTS - 1, 5)
    assert 0 not in accepted and len(accepted) == _N_SLOTS - 1


def test_the_solver_failure_limit(monkeypatch):
    """SMP-106: 10 solver failures are allowed, the 11th drops the well; a missing operating point does not count."""
    assert _SOLVER_FAILURE_LIMIT == 10
    failing = set(range(0, 200, 20))  # 10 slots

    def fails(j, a):
        if j in failing and a == 0:
            return 'solver_failure'
        return 'no_operating_point' if j % 2 and a == 0 else None

    accepted, result, order = solve_slots(monkeypatch, fails)
    assert result == 'accepted'
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'solver_failure' if j in failing | {199} else None)
    assert result == 'solver_failure_limit' and order[-1] == (199, 0) and len(order) == _N_SLOTS
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'solver_failure' if j == 11 and a < 11 else None)
    assert result == 'solver_failure_limit' and order[-1] == (11, 10)


# --- SMP-101, SMP-90: the life scan -----------------------------------------------------------------------------------
def a_life(T_life=10.0, lift_start=None, rule=None, design=True):
    return types.SimpleNamespace(
        T_life=T_life, lift_start=lift_start, lift_start_rule=rule, design={} if design else None
    )


def scan(life, status_of):
    """The life scan with status_of(i, lifted) the status of the best rate at point i."""
    calls = []

    def best_rate_at(i, t, lifted):
        calls.append((i, t, lifted))
        status = status_of(i, lifted)
        return dict(status=status, best=float(i) if status in ('pass', 'below_floor') else math.nan, outcomes=[i])

    return (*generate._life_scan(life, best_rate_at), calls)


def test_a_life_that_passes_to_its_end():
    """SMP-101: 21 dates i T_life / 20, each at the lift the well has there; the life ends at T_life."""
    life = a_life(lift_start=4.0, rule='drawn_time')
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass')
    dates = [i * 10.0 / 20 for i in range(21)]
    assert [p['t'] for p in points] == [c[1] for c in calls] == pytest.approx(dates, rel=1e-15, abs=1e-15)
    assert [p['lift_on'] for p in points] == [c[2] for c in calls] == [t >= 4.0 for t in dates]
    assert (T_end, lift_start, rule, first['outcomes']) == (10.0, 4.0, 'drawn_time', [0])
    assert all(p['status'] == 'pass' and p['best'] == i for i, p in enumerate(points))


@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point', 'water_cut'])
def test_the_life_ends_at_the_last_point_before_the_first_that_fails(status):
    life = a_life(lift_start=0.0, rule='from_start')
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: status if i == 9 else 'pass')
    assert len(points) == 10 and T_end == 8 * 10.0 / 20 and (lift_start, rule) == (0.0, 'from_start')


def test_a_point_with_a_solver_failure_is_skipped():
    """A skipped point neither passes nor ends the life."""
    life = a_life(lift_start=0.0, rule='from_start')
    statuses = {3: 'solver_failure', 7: 'solver_failure', 8: 'no_operating_point'}
    points, _, T_end, *_ = scan(life, lambda i, lifted: statuses.get(i, 'pass'))
    assert [p['status'] for p in points] == ['pass'] * 3 + ['solver_failure'] + ['pass'] * 3 + list(statuses.values())[
        1:
    ]
    assert T_end == 6 * 10.0 / 20
    points, _, T_end, *_ = scan(life, lambda i, lifted: 'solver_failure' if i == 20 else 'pass')
    assert len(points) == 21 and T_end == 10.0


@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point', 'water_cut', 'solver_failure'])
def test_a_life_whose_first_point_does_not_pass(status):
    """SMP-101: the well is dropped at its start (generate._dropped_at_best_rate), and natural flow failing at the
    first point starts no lift."""
    life = a_life()
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: status)
    assert len(points) == 1 and T_end is None and (lift_start, rule) == (None, None) and first['status'] == status
    info = generate._info(4)
    assert generate._dropped_at_best_rate(info, first)
    reason = 'best_rate_solver_failure' if status == 'solver_failure' else 'not_producible'
    assert (info['outcome'], info['not_producible_reason']) == (reason, status)


@pytest.mark.parametrize('drawn', [(None, None), (7.5, 'drawn_time'), (9.0, 'static_check')])
@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point'])
def test_rule_3_lifts_a_well_before_its_natural_flow_fails(drawn, status):
    """SMP-90, rule 3: where the natural well fails at a point after the first, lift starts at the point before,
    earlier than the draws' start, and the point is solved again with lift."""
    life = a_life(lift_start=drawn[0], rule=drawn[1])
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if lifted or i < 6 else status)
    assert (lift_start, rule, T_end) == (5 * 10.0 / 20, 'end_of_natural_flow', 10.0)
    assert calls[6:8] == [(6, 3.0, False), (6, 3.0, True)] and all(c[2] for c in calls[7:])
    assert len(points) == 21 and points[6] == dict(t=3.0, status='pass', best=6.0, lift_on=True)


def test_rule_3_where_the_lifted_well_fails_too():
    life = a_life()
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if i < 4 else 'no_operating_point')
    assert (lift_start, rule, T_end) == (1.5, 'end_of_natural_flow', 1.5) and len(points) == 5
    assert calls[-2:] == [(4, 2.0, False), (4, 2.0, True)] and points[-1]['lift_on']
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if i < 1 else 'below_floor')
    assert (lift_start, rule, T_end) == (0.0, 'end_of_natural_flow', 0.0)  # a short life


def test_no_rule_3_without_a_design_or_for_the_water_cut():
    """A well that no rule lifts (a gas well without a lift design) ends where its natural flow fails; a water cut
    ends the life of a natural well as of a lifted one."""
    points, _, T_end, lift_start, rule, calls = scan(
        a_life(design=False), lambda i, lifted: 'pass' if i < 6 else 'below_floor'
    )
    assert (lift_start, rule, T_end, len(calls)) == (None, None, 2.5, 7)
    points, _, T_end, lift_start, rule, calls = scan(a_life(), lambda i, lifted: 'pass' if i < 6 else 'water_cut')
    assert (lift_start, rule, T_end, len(calls)) == (None, None, 2.5, 7)


# --- SMP-77: the choke's sizing ---------------------------------------------------------------------------------------
class StandInBuilder:
    """In place of a WellBuilder at the sizing: it records the choke that it is given."""

    def __init__(self, fluid_type='black_oil'):
        self.w, self.choke = dict(fluid_type=fluid_type, c_e=120.0, flags=''), None

    def widest_choke(self, share=1.0):
        self.choke = ('widest', share)
        return 8.0 * share

    def size_choke(self, w_m, wellhead, p_dn, K_c_most):
        self.choke = (w_m, wellhead, p_dn, K_c_most)


def size_choke(monkeypatch, outcomes, lifted=True):
    """The sizing with each solve's (outcome, q_oe) given in turn: its result, the builder, the log and the solves'
    inputs."""
    it = iter(outcomes)
    attempt, calls = stand_in_attempt(lambda inp: next(it))
    monkeypatch.setattr(generate, '_attempt', attempt)
    capped = []  # The cap's own solves are tested on their own
    monkeypatch.setattr(generate, '_cap_choke', lambda builder, *args: capped.append(builder.choke))
    builder, log = StandInBuilder(), []
    state = dict(p_r=200.0, gor=100.0, wlr=0.3, wgr=math.nan, p_s_nom=20.0, h=10.0, roughness=1e-5)
    out = generate._size_choke(builder, state, lifted, 2.0, log, dict(draw_index=4, kind='choke_sizing'))
    assert capped == ([builder.choke] if out[0] == 'sized' else [])  # A sized choke is capped, after its sizing
    return out, builder, log, calls


def test_the_choke_is_sized_at_the_solve_with_the_most_oil_equivalent(monkeypatch):
    """SMP-77: the well is solved with its largest choke, fully open, at the nominal downstream pressure and the best
    rate's lift-gas rates; the choke is sized at the mass rate and the wellhead's state of the solve with the most
    oil equivalent. Each solve is logged with its labels."""
    solves = [('operating_point', 3.0), ('operating_point', 5.0), ('no_operating_point', None)]
    (status, outcomes), builder, log, calls = size_choke(monkeypatch, solves)
    assert status == 'sized' and outcomes == ['operating_point', 'operating_point', 'no_operating_point']
    assert builder.choke == (5.0, WELLHEAD, 20.0, 8.0)  # At most the largest choke, which it was solved with
    assert [(c['u'], c['p_s'], c['w_lg'], c['lift_on']) for c in calls] == [
        (1.0, 20.0, f * 2.0, True) for f in (0, 0.5, 1)
    ]
    assert [(r['draw_index'], r['kind'], r['attempt'], r['lift_fraction'], r['sizing_choke']) for r in log] == [
        (4, 'choke_sizing', i, f, 1.0) for i, f in enumerate((0.0, 0.5, 1.0))
    ]
    (status, _), builder, _, calls = size_choke(monkeypatch, [('operating_point', 2.0)], lifted=False)
    assert status == 'sized' and len(calls) == 1 and (calls[0]['w_lg'], calls[0]['lift_on']) == (0.0, False)


def test_the_sizing_takes_a_smaller_choke_where_the_largest_gives_no_operating_point(monkeypatch):
    """SMP-77: a well without an operating point with its largest choke is solved with a bean of half the diameter,
    a quarter of the coefficient; the choke is at most the one that it was solved with."""
    assert _SIZING_CHOKES == (1.0, 1 / 4)
    solves = [('solver_failure', None), ('operating_point', 3.0)]
    (status, outcomes), builder, log, calls = size_choke(monkeypatch, solves, lifted=False)
    assert status == 'sized' and outcomes == ['operating_point'] and len(calls) == 2
    assert builder.choke == (5.0, WELLHEAD, 20.0, 8.0 / 4)
    assert [r['sizing_choke'] for r in log] == [1.0, 1 / 4]


@pytest.mark.parametrize(
    'outcomes, status',
    [
        (['no_operating_point'] * 6, 'no_operating_point'),
        (['no_operating_point', 'solver_failure'] + ['no_operating_point'] * 4, 'solver_failure'),
    ],
)
def test_a_well_without_an_operating_point_at_either_sizing_choke_is_not_sized(monkeypatch, outcomes, status):
    """SMP-77: the sizing's status is the best rate's, over the solves of both its chokes; its outcomes are the last
    choke's, and the well is left with its largest choke."""
    (got, seen), builder, log, _ = size_choke(monkeypatch, [(o, None) for o in outcomes])
    assert (got, seen) == (status, outcomes[-3:]) and builder.choke == ('widest', 1.0)
    assert [r['sizing_choke'] for r in log] == [share for share in _SIZING_CHOKES for _ in range(3)]


def test_the_best_rate_of_a_sized_well(monkeypatch):
    """SMP-77, SMP-103: the best rate is the sized well's; a well that the sizing leaves without a choke has the
    sizing's status and outcomes and no best rate, an oil above the water cut the water cut's."""
    it = iter(
        [('operating_point', 3.0)] * 6 + [('operating_point', 2.0), ('operating_point', 4.0), ('operating_point', 1.0)]
    )
    attempt, calls = stand_in_attempt(lambda inp: next(it))
    monkeypatch.setattr(generate, '_attempt', attempt)
    state = dict(p_r=200.0, gor=100.0, wlr=0.3, wgr=math.nan, p_s_nom=20.0, h=10.0, roughness=1e-5)
    builder, log = StandInBuilder(), []
    out = generate._sized_best_rate(builder, state, True, 2.0, 1.6, log, dict(draw_index=4, kind='best_rate'))
    assert out == dict(
        best=4.0, best_w_lg=1.0, outcomes=['operating_point'] * 3, flows_without_lift_gas=True, status='pass'
    )
    kinds = ['choke_sizing'] * 3 + ['erosional_cap'] * 3 + ['best_rate'] * 3  # The stand-in's wellhead is below v_e
    assert [r['kind'] for r in log] == kinds and builder.choke[0] == 5.0

    for wlr, status in ((0.3, 'no_operating_point'), (0.981, 'water_cut')):
        attempt, calls = stand_in_attempt(lambda inp: ('no_operating_point', None))
        monkeypatch.setattr(generate, '_attempt', attempt)
        out = generate._sized_best_rate(
            StandInBuilder(), dict(state, wlr=wlr), False, 0.0, 1.6, [], dict(kind='best_rate')
        )
        assert out['status'] == status and math.isnan(out['best']) and out['outcomes'] == ['no_operating_point']
        assert len(calls) == len(_SIZING_CHOKES)  # No best rate is solved


def test_the_erosional_ratio():
    """SMP-77, SMP-75: the no-slip mixture velocity at the wellhead over 1.22 C / sqrt(rho_ns), rho_ns the no-slip
    density of the stream there."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = WELLHEAD
    v_sg, v_sl = alpha * v_g, (1 - alpha) * v_l
    rho_ns = (v_sg * rho_g + v_sl * rho_l) / (v_sg + v_sl)
    assert generate._erosional_ratio(WELLHEAD, 120.0) == pytest.approx((v_sg + v_sl) / (1.22 * 120.0 / rho_ns**0.5))
    faster = [p, 2 * v_g, 2 * v_l, alpha, rho_g, rho_l, T]  # Twice the velocities: twice the ratio
    assert generate._erosional_ratio(faster, 120.0) == pytest.approx(2 * generate._erosional_ratio(WELLHEAD, 120.0))


class CapBuilder:
    """In place of a WellBuilder at the erosional cap: it holds the choke it is given, at most its largest."""

    def __init__(self, K_c, flags=''):
        self.w = dict(c_e=120.0, flags=flags, K_c_max=1.0)
        self.choke = types.SimpleNamespace(K_c=K_c)

    def take_choke(self, K_c):
        self.choke = types.SimpleNamespace(K_c=min(K_c, self.w['K_c_max']))
        return self.choke.K_c


def cap(monkeypatch, ratio_of, K_c=0.01, lifted=False, flags=''):
    """The cap of a well whose erosional ratio at the choke K_c is ratio_of(K_c, w_lg), or that has no operating point
    where that is None: its builder, log and solves' inputs."""
    r0 = generate._erosional_ratio(WELLHEAD, 120.0)

    def attempt(builder, inp, wellhead=None):
        calls.append(dict(inp, K_c=builder.choke.K_c))
        r = ratio_of(builder.choke.K_c, inp['w_lg'])
        if r is None:
            wellhead.append(None)
            return dict(inp, outcome='no_operating_point', reason=None, error=None)
        p, v_g, v_l, alpha, rho_g, rho_l, T = WELLHEAD
        wellhead.append([p, v_g * r / r0, v_l * r / r0, alpha, rho_g, rho_l, T])
        return dict(inp, outcome='operating_point', reason=None, error=None, q_oe=50.0, WTOT=5.0)

    calls, log = [], []
    monkeypatch.setattr(generate, '_attempt', attempt)
    builder = CapBuilder(K_c, flags)
    state = dict(p_r=200.0, gor=100.0, wlr=0.3, wgr=math.nan, p_s_nom=20.0, h=10.0, roughness=1e-5)
    generate._cap_choke(builder, state, lifted, 2.0, log, dict(draw_index=4, kind='choke_sizing'))
    return builder, log, calls


def test_a_well_below_its_erosional_velocity_keeps_its_choke(monkeypatch):
    """SMP-77: the check at the sized choke, fully open at the best rate's lift-gas rates, finds every ratio at most 1:
    the choke stays, and a cap flag of an earlier sizing goes."""
    builder, log, calls = cap(
        monkeypatch, lambda K, w_lg: 0.5 + w_lg / 4, lifted=True, flags='a;choke_at_erosional_rate'
    )
    assert builder.choke.K_c == 0.01 and builder.w['flags'] == 'a'
    assert [(c['u'], c['w_lg'], c['p_s']) for c in calls] == [(1.0, f * 2.0, 20.0) for f in (0.0, 0.5, 1.0)]
    assert [(r['kind'], r['cap_step'], r['attempt'], r['lift_fraction']) for r in log] == [
        ('erosional_cap', 0, i, f) for i, f in enumerate((0.0, 0.5, 1.0))
    ]
    assert [r['erosional_ratio'] for r in log] == pytest.approx([0.5, 0.75, 1.0])


@pytest.mark.parametrize('slope', [0.3, 0.5, 1.0, 1.8])
def test_the_cap_holds_the_well_at_its_erosional_velocity(monkeypatch, slope):
    """SMP-77: a well above its erosional velocity at its sized choke takes the largest choke found whose ratio, the
    largest over the lift-gas rates, lies within [0.97, 1], in at most 12 chokes; it is flagged."""
    K_e = 0.002  # The choke at which the full lift gas's ratio is 1
    builder, log, calls = cap(monkeypatch, lambda K, w_lg: (K / K_e) ** slope * (0.8 + 0.1 * w_lg), lifted=True)
    r = (builder.choke.K_c / K_e) ** slope
    assert generate._EROSIONAL_LOW <= r <= 1.0 and builder.w['flags'] == 'choke_at_erosional_rate'
    steps = sorted({rec['cap_step'] for rec in log})
    assert steps == list(range(len(steps))) and len(steps) <= generate._EROSIONAL_STEPS and len(log) == 3 * len(steps)
    assert len(steps) <= 6  # A smooth ratio is found in a few chokes


def test_the_cap_passes_chokes_without_an_operating_point(monkeypatch):
    """SMP-77: a choke so small that the well cannot flow is below the band: the search goes between it and the
    smallest choke above."""
    builder, _, _ = cap(monkeypatch, lambda K, w_lg: None if K < 0.0018 else (K / 0.002) ** 0.2)
    r = (builder.choke.K_c / 0.002) ** 0.2
    assert generate._EROSIONAL_LOW <= r <= 1.0 and builder.w['flags'] == 'choke_at_erosional_rate'


def test_a_cap_that_is_not_met(monkeypatch):
    """SMP-77: where the 12 chokes find none within the band, the well keeps the last choke above it and is
    flagged."""
    builder, log, calls = cap(monkeypatch, lambda K, w_lg: 1.5, flags='a')
    assert len(calls) == generate._EROSIONAL_STEPS and builder.w['flags'] == 'a;erosional_cap_unmet'
    assert builder.choke.K_c == calls[-1]['K_c']


# --- SMP-102: the start hurdle ----------------------------------------------------------------------------------------
def test_the_start_hurdle():
    """A platform or subsea well, core or tail, is drilled only where its design rate in oil equivalent reaches its
    type's hurdle; an onshore well has none."""
    assert _START_HURDLE == {'platform': 20.0, 'onshore': 0.0, 'subsea': 500.0}
    below = generate._below_start_hurdle
    assert below(dict(coverage_tail='core', well_type='platform', q_design_oe=19.99))
    assert not below(dict(coverage_tail='core', well_type='platform', q_design_oe=20.0))
    assert below(dict(coverage_tail='core', well_type='subsea', q_design_oe=499.0))
    assert below(dict(coverage_tail='tail', well_type='subsea', q_design_oe=499.0))
    assert not below(dict(coverage_tail='tail', well_type='subsea', q_design_oe=500.0))
    assert not below(dict(coverage_tail='core', well_type='onshore', q_design_oe=1e-3))


@pytest.mark.parametrize('run', [generate_sol2_well, generate_nsol2_well])
def test_a_well_below_the_start_hurdle_is_not_solved(monkeypatch, run):
    """SMP-102: a well whose design rate lies below its hurdle is dropped as not producible, without a solve, with
    its draws."""
    no_solve(monkeypatch)
    lf = next(lf for lf in drawn_lives() if generate._below_start_hurdle(lf.w0) and not is_table(lf))
    res = run(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['not_producible_reason']) == ('not_producible', 'below_start_hurdle')
    assert res['log'] == res['samples'] == [] and res['well']['well'] == lf.k and res['well']['K_c'] is None


# --- A sol-2 well -----------------------------------------------------------------------------------------------------
def by_lift(natural, lifted):
    """An attempt's outcome by whether lift is on: (outcome, q_oe) each."""
    return lambda inp: lifted if inp['lift_on'] else natural


SIZED = ('K_c', 'd_bean', 'beta')  # What the sizing gives a well's draws (SMP-77)


def same_but_the_choke(well, drawn) -> bool:
    """Whether a generated well's draws are the drawn ones, with a choke that the sizing gave it."""
    same = all(nan_or(well[key], v) for key, v in drawn.items() if key not in SIZED) and set(well) == set(drawn)
    return same and 0 < well['K_c'] <= well['K_c_max'] and well['beta'] == well['d_bean'] / well['D']


def test_a_sol2_wells_choke_best_rate_and_slots(monkeypatch):
    """The choke's sizing and the best rate at the nominal state, then the slots' attempts, as samples._sol2_attempt
    draws them."""
    lf, w = a_sol2_well(oil=True, lifted=True)
    attempt, calls = stand_in_attempt(by_lift(None, ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    info = res['info']
    assert list(res) == RESULT and list(info) == INFO and res['nsol'] is None and res['table'] is None
    assert (info['outcome'], info['best_rate'], info['best_rate_w_lg']) == ('accepted', 50.0, 0.0)
    assert info['best_rate_outcomes'] == ['operating_point'] * 3
    assert same_but_the_choke(res['well'], w)
    assert res['well']['K_c'] == WellBuilder(dict(lf.w0), lf.stations).size_choke(5.0, WELLHEAD, w['p_s'])
    assert res['stations'] == lf.stations and all(r['draw_index'] == lf.k for r in res['log'])
    nominal = dict(p_r=w['p_r'], gor=w['R_go'], wlr=w['wlr'], p_s=w['p_s'], h=w['h'], roughness=w['roughness'])
    for c in calls[:9]:  # The sizing's solves, the erosional cap's check, then the best rate's
        assert {key: c[key] for key in nominal} == nominal and (c['u'], c['lift_on']) == (1.0, True)
        assert all(nan_or(c[key], w[key]) for key in _COEF_KEYS)
    assert [c['w_lg'] for c in calls[:9]] == [f * w['w_lg_max'] for f in (0.0, 0.5, 1.0)] * 3
    for j, c in enumerate(calls[9:]):
        assert all(nan_or(c[key], v) for key, v in _sol2_attempt(SEED, w, j).items())
    assert [r['kind'] for r in res['log'][:9]] == ['choke_sizing'] * 3 + ['erosional_cap'] * 3 + ['best_rate'] * 3
    assert [r['slot'] for r in res['samples']] == list(range(_N_SLOTS)) and len(res['log']) == 9 + _N_SLOTS


def test_the_producibility_filter_lifts_a_natural_oil_well(monkeypatch):
    """SMP-103: a natural oil well without an operating point at its best rate takes its life's lift design, with
    `lift_start_rule` `no_natural_flow`, and its slots are drawn lifted."""
    lf, w = a_sol2_well(oil=True, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(('no_operating_point', None), ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    lifted = sol2_well(lf, no_natural_flow=True)
    assert res['info']['outcome'] == 'accepted' and res['info']['best_rate_w_lg'] == 0.0
    assert (res['well']['lift'], res['well']['lift_start_rule']) == ('gas_lift', 'no_natural_flow')
    assert same_but_the_choke(res['well'], lifted)
    kinds = [r['kind'] for r in res['log']]  # The natural well's sizing finds no operating point at either choke
    n = len(_SIZING_CHOKES)
    lifted_kinds = ['choke_sizing'] * 3 + ['erosional_cap'] * 3 + ['best_rate_lifted'] * 3
    assert kinds == ['choke_sizing'] * n + lifted_kinds + ['slot'] * _N_SLOTS
    assert res['info']['best_rate_outcomes'] == ['operating_point'] * 3  # the lifted well's
    assert all(c['lift_on'] for c in calls[n:]) and calls[n + 9]['w_lg'] == _sol2_attempt(SEED, lifted, 0)['w_lg']


@pytest.mark.parametrize('off_outcome', [('no_operating_point', None), ('operating_point', 1e-3), None])
def test_a_lifted_sol2_well_that_cannot_flow_without_lift_gas(monkeypatch, off_outcome):
    """SMP-94: a lifted sol-2 well whose best rate's solve without lift gas has no operating point, or one below the
    floor's margin, is flagged lift_gas_always_on, and none of its slots is drawn without lift gas; a well that flows
    without it keeps the draws of SMP-94 (off_outcome None: every solve at 50 Sm3/d)."""
    lf, w = a_sol2_well(oil=True, lifted=True)

    def outcome_of(inp):
        return off_outcome if (inp['w_lg'] == 0 and off_outcome is not None) else ('operating_point', 50.0)

    attempt, calls = stand_in_attempt(outcome_of)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    assert res['info']['outcome'] == 'accepted'
    flagged = _LIFT_GAS_ALWAYS_ON in res['well']['flags'].split(';')
    assert flagged == (off_outcome is not None)
    assert res['well']['flags'].split(';')[:-1] == [f for f in w['flags'].split(';') if f] or not flagged
    slots = calls[9:]  # After the sizing, the erosional cap's check and the best rate, three solves each
    assert len(slots) == _N_SLOTS
    assert all(c['w_lg'] > 0 for c in slots) if flagged else any(c['w_lg'] == 0 for c in slots)
    for j, c in enumerate(slots):
        assert c['w_lg'] == _sol2_attempt(SEED, res['well'], j)['w_lg']


@pytest.mark.parametrize(
    'natural, lifted, outcome, reason, solves',
    [
        # The sizing finds no natural flow at its two chokes; lifted, the well is sized and is below the floor
        (('no_operating_point', None), ('operating_point', 1e-3), 'not_producible', 'below_floor', 2 + 3 + 3 + 3),
        # Sized, the natural well is below the floor; the lifted well's sizing finds no operating point
        (
            ('operating_point', 1e-3),
            ('no_operating_point', None),
            'not_producible',
            'no_operating_point',
            1 + 1 + 1 + 6,
        ),
        (('no_operating_point', None), ('solver_failure', None), 'best_rate_solver_failure', 'solver_failure', 2 + 6),
        (('solver_failure', None), ('operating_point', 50.0), 'best_rate_solver_failure', 'solver_failure', 2),
    ],
)
def test_a_natural_oil_well_that_does_not_pass(monkeypatch, natural, lifted, outcome, reason, solves):
    """SMP-103, SMP-77: a natural well's solver failure is not lifted; a lifted well that does not pass, at its
    choke's sizing or at its best rate, is dropped."""
    lf, w = a_sol2_well(oil=True, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(natural, lifted))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['not_producible_reason']) == (outcome, reason)
    assert res['samples'] == [] and all(r['kind'] != 'slot' for r in res['log'])
    assert len(res['log']) == solves


def test_a_gas_well_is_not_lifted_by_the_filter(monkeypatch):
    lf, w = a_sol2_well(oil=False, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(('no_operating_point', None), ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['well']['lift'], len(res['log'])) == ('not_producible', 'natural', 2)


# --- An nsol-2 well ---------------------------------------------------------------------------------------------------
def an_nsol2_life():
    """The first black oil's life of the tests' seed that is natural at its start and whose draws start no lift
    before 0.6 of its life."""
    return next(
        lf
        for lf in drilled_lives()
        if lf.w0['fluid_class'] == 'black_oil'
        and lf.lift_start != 0.0
        and (lf.lift_start is None or lf.lift_start > 0.6 * lf.T_life)
    )


def test_an_nsol2_wells_scan_lift_and_slots(monkeypatch):
    """The life scan at the life's states, rule 3's lift from the scan, and the plan's slots with that lift."""
    lf = an_nsol2_life()
    T = lf.T_life
    flows = lambda inp: inp['lift_on'] or inp['t'] < 0.32 * T  # noqa: E731
    attempt, calls = stand_in_attempt(
        lambda inp: ('operating_point', 50.0) if flows(inp) else ('no_operating_point', None)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    assert list(res) == RESULT and list(info) == INFO + NSOL_INFO
    assert info['T_scan'] == T and info['dead_dates'] == []
    dates = np.linspace(0.0, T, 21)
    i_fail = 7  # The first date at or past 0.32 T_life
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (float(dates[i_fail - 1]), 'end_of_natural_flow')
    assert {
        key: v for key, v in nsol.items() if key not in ('lift_start', 'lift_start_rule', 'design')
    } == lf.parameters()
    assert nsol['design'] == {key: lf.design[key] for key in nsol['design']} and 'w_lg_max' in nsol['design']
    assert (info['outcome'], info['T_end'], info['best_rate'], info['best_rate_w_lg']) == ('accepted', T, 50.0, 0.0)
    assert [p['t'] for p in info['scan']] == [float(t) for t in dates]
    assert [p['lift_on'] for p in info['scan']] == [i >= i_fail for i in range(21)]
    # The choke is sized at the life's start, and its erosional cap checked
    assert [r['kind'] for r in res['log'][:3]] == ['choke_sizing', 'erosional_cap', 'scan']
    assert res['well']['K_c'] == WellBuilder(dict(lf.w0), lf.stations).size_choke(5.0, WELLHEAD, lf.p_s_nom(0.0))
    scan_calls = [c for c in calls if 'test' not in c][2:]  # The slots' inputs hold their test period
    assert all(nan_or(calls[0][key], v) for key, v in scan_calls[0].items()) and set(calls[0]) == set(scan_calls[0])
    assert len(scan_calls) == i_fail + 1 + 3 * (21 - i_fail)  # A lifted point at the three lift-gas rates
    assert [c['w_lg'] for c in scan_calls[: i_fail + 1]] == [0.0] * (i_fail + 1)
    assert [c['w_lg'] / lf.design['w_lg_max'] for c in scan_calls[i_fail + 1 : i_fail + 4]] == [0.0, 0.5, 1.0]
    for c in scan_calls:
        want = dict(lf.state(c['t']), u=1.0, lift_on=c['lift_on'], p_s=c['p_s_nom'])
        assert set(c) == {*want, 'w_lg'} and all(nan_or(c[key], v) for key, v in want.items())
    plan, p_c = _nsol2_plan(lf, T, nsol['lift_start'])
    assert info['p_change'] == p_c and len(res['samples']) == _N_SLOTS
    for slot, rec in zip(plan, res['samples']):
        assert (rec['t'], rec['lift_on'], rec['u'], rec['p_s']) == (
            slot['t'],
            _lift_on(slot['t'], nsol['lift_start']),
            slot['u'],
            slot['p_s'],
        )


def test_an_nsol2_well_whose_life_ends_before_its_lift_starts_is_natural(monkeypatch):
    """SMP-52, SMP-90 (D-131): where the draws start lift between two scan points and the lifted well fails at the
    later one, the life ends before lift starts; no date of it is lifted, so the well is natural, with no rule."""
    T = lambda lf: lf.T_life  # noqa: E731
    lf = next(
        lf
        for lf in drilled_lives()
        if lf.lift_start_rule in ('drawn_time', 'static_check')
        and (lf.lift_start * 20 / T(lf)) % 1 > 0.1
        and lf.w0['fluid_class'] not in _TABLE_CLASSES
    )
    attempt, _ = stand_in_attempt(
        lambda inp: ('no_operating_point', None) if inp['lift_on'] else ('operating_point', 50.0)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    dates = np.linspace(0.0, T(lf), 21)
    assert info['T_end'] == dates[dates < lf.lift_start][-1] and info['outcome'] == 'accepted'
    assert info['scan'][-1]['lift_on'] and info['scan'][-1]['status'] == 'no_operating_point'
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (None, None)
    assert not any(rec['lift_on'] for rec in res['samples']) and len(res['samples']) == _N_SLOTS
    plan, p_c = _nsol2_plan(lf, info['T_end'], None)  # The plan of a natural well: the same draws
    assert info['p_change'] == p_c and [rec['u'] for rec in res['samples']] == [slot['u'] for slot in plan]


def test_an_nsol2_wells_short_life(monkeypatch):
    """SMP-101: a life that ends at its first point is `short_life`, without slots. Rule 3's lift, which would start
    where the life ends, does not make the well a lifted one."""
    lf = an_nsol2_life()
    attempt, _ = stand_in_attempt(
        lambda inp: ('operating_point', 50.0) if inp['t'] == 0.0 else ('no_operating_point', None)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['T_end'], res['nsol']['lift_start']) == ('short_life', 0.0, None)
    assert res['samples'] == [] and math.isnan(res['info']['p_change']) and len(res['info']['scan']) == 2
    assert res['info']['T_scan'] == 0.0 and res['info']['dead_dates'] == []


@pytest.mark.parametrize(
    'outcome, well_outcome, reason',
    [
        ('no_operating_point', 'not_producible', 'no_operating_point'),
        ('solver_failure', 'best_rate_solver_failure', 'solver_failure'),
    ],
)
def test_an_nsol2_well_whose_choke_cannot_be_sized_is_dropped(monkeypatch, outcome, well_outcome, reason):
    """SMP-77: a well without an operating point at its life's start with its largest choke gets no choke: it is
    dropped before its life scan, with the sizing's outcomes. SMP-103: a natural oil without an operating point is
    first sized again with its lift design, at the three lift-gas rates; a solver failure lifts no well."""
    lf = an_nsol2_life()
    attempt, _ = stand_in_attempt(lambda inp: (outcome, None))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, lifted = res['info'], outcome == 'no_operating_point'
    assert (info['outcome'], info['not_producible_reason']) == (well_outcome, reason)
    assert info['best_rate_outcomes'] == [outcome] * (3 if lifted else 1)
    assert info['scan'] == [] and res['samples'] == [] and res['well']['K_c'] is None
    assert [r['kind'] for r in res['log']] == ['choke_sizing'] * len(_SIZING_CHOKES) * (1 + 3 if lifted else 1)
    assert [r['lift_on'] for r in res['log']] == [False] * len(_SIZING_CHOKES) + [True] * (len(res['log']) - 2)
    start = (0.0, 'no_natural_flow') if lifted else (lf.lift_start, lf.lift_start_rule)
    assert (res['nsol']['lift_start'], res['nsol']['lift_start_rule']) == start


@pytest.mark.parametrize('natural', ['no_operating_point', 'below_floor'])
def test_an_nsol2_natural_oil_that_cannot_flow_at_its_start_is_lifted_from_it(monkeypatch, natural):
    """SMP-103 at an nsol-2 well's start: a natural oil without an operating point there, at the sizing or at the life
    scan's first point, or with one below half its floor, takes its lift design from the start. Its choke is sized
    again with the lift, and the life is scanned again."""
    lf = an_nsol2_life()
    floor = _FLOOR[lf.w0['well_type']]
    weak = ('no_operating_point', None) if natural == 'no_operating_point' else ('operating_point', 0.1 * floor)
    attempt, _ = stand_in_attempt(lambda inp: ('operating_point', 50.0) if inp['lift_on'] else weak)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    assert (info['outcome'], info['T_end'], info['best_rate']) == ('accepted', lf.T_life, 50.0)
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (0.0, 'no_natural_flow')
    assert len(info['scan']) == 21 and all(p['lift_on'] and p['status'] == 'pass' for p in info['scan'])
    assert len(res['samples']) == _N_SLOTS and all(rec['lift_on'] for rec in res['samples'])
    kinds = [r['kind'] for r in res['log'] if r['kind'] != 'slot']
    if natural == 'no_operating_point':  # The natural well has no choke: its sizing's two chokes, one solve each
        first = ['choke_sizing'] * len(_SIZING_CHOKES)
    else:  # The natural well is sized and its cap checked, and its first scan point is below the floor's margin
        first = ['choke_sizing', 'erosional_cap', 'scan']
    assert kinds == first + ['choke_sizing'] * 3 + ['erosional_cap'] * 3 + ['scan_lifted'] * 3 * 21
    assert res['well']['K_c'] == WellBuilder(dict(lf.w0), lf.stations).size_choke(5.0, WELLHEAD, lf.p_s_nom(0.0))


def test_an_nsol2_gas_well_that_cannot_flow_at_its_start_is_dropped(monkeypatch):
    """SMP-103: only an oil is lifted for want of natural flow. A natural gas well without an operating point at its
    start is dropped, as before."""
    lf = next(
        lf
        for lf in drilled_lives()
        if not lf.oil and lf.lift_start != 0.0 and lf.w0['fluid_class'] not in _TABLE_CLASSES
    )
    attempt, _ = stand_in_attempt(lambda inp: ('no_operating_point', None))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['not_producible_reason']) == ('not_producible', 'no_operating_point')
    assert [r['kind'] for r in res['log']] == ['choke_sizing'] * len(_SIZING_CHOKES)
    assert not any(r['lift_on'] for r in res['log']) and res['nsol']['lift_start_rule'] != 'no_natural_flow'


# --- SMP-90: the flags of the lift design that a well uses ----------------------------------------------------------
def test_the_lift_blocks_flags_are_the_used_designs():
    """SMP-90: the lift block's flags of a well are those of the lift design that it uses. A lifted well whose design
    is another stage's draw takes that draw's, a well that is not lifted has none, and the other flags stay."""
    flags = generate._lift_flags
    own = 'wet_start;injection_at_packer;choke_at_erosional_rate'
    late = dict(flags='T_r_cap_binds;injection_raised_to_0.3L;lift_capacity_at_bound')
    assert flags(own, dict(flags=own), True) == own  # The design is the draw's own: nothing moves
    assert flags(own, late, True) == 'wet_start;choke_at_erosional_rate;injection_raised_to_0.3L;lift_capacity_at_bound'
    assert flags(own, late, False) == 'wet_start;choke_at_erosional_rate'
    assert flags('', late, True) == 'injection_raised_to_0.3L;lift_capacity_at_bound' and flags('', late, False) == ''
    assert flags('wet_start', dict(flags=''), True) == 'wet_start' and flags('', None, False) == ''
    assert set(generate._DESIGN_FLAGS) == {'injection_at_packer', 'injection_raised_to_0.3L', 'lift_capacity_at_bound'}


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_well_lifted_with_a_later_stages_design_has_that_designs_flags(monkeypatch, kind):
    """SMP-90: a well that is natural by its initial-stage draw and lifted in the dataset, here by the producibility
    filter (SMP-103), holds its lift design's flags, which its initial-stage draw does not have."""
    design_flags = lambda lf: [f for f in lf.design['flags'].split(';') if f in generate._DESIGN_FLAGS]  # noqa: E731
    lf, w = next(
        (lf, w)
        for lf, w in zip(drawn_lives(), drawn_sol2_wells())
        if lf.oil
        and not generate._below_start_hurdle(lf.w0)
        and w['lift'] == 'natural'
        and lf.lift_start != 0.0
        and lf.w0['fluid_class'] not in _TABLE_CLASSES
        and design_flags(lf)
    )
    assert not any(f in lf.w0['flags'].split(';') for f in generate._DESIGN_FLAGS)
    attempt, _ = stand_in_attempt(
        lambda inp: ('operating_point', 50.0) if inp['lift_on'] else ('no_operating_point', 0)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = (generate_sol2_well if kind == 'sol' else generate_nsol2_well)(SEED, lf.k)
    assert res['info']['outcome'] == 'accepted'
    assert [f for f in res['well']['flags'].split(';') if f in generate._DESIGN_FLAGS] == design_flags(lf)


def test_an_nsol2_life_that_ends_where_its_lift_would_start_is_natural(monkeypatch):
    """SMP-101 (D-146): rule 3 starts lift at the last point before natural flow fails; where the lifted well fails at
    the next point too, the life ends at the lift's start. The well is natural: no sample of it is lifted."""
    lf = an_nsol2_life()
    T = lf.T_life
    attempt, _ = stand_in_attempt(
        lambda inp: (
            ('operating_point', 50.0) if not inp['lift_on'] and inp['t'] < 0.32 * T else ('no_operating_point', 0)
        )
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    t_6 = float(np.linspace(0.0, T, 21)[6])  # The last scan point before 0.32 T_life
    assert (info['outcome'], info['T_end'], info['T_scan']) == ('accepted', t_6, t_6)
    assert info['scan'][-1] == dict(
        t=float(np.linspace(0.0, T, 21)[7]), status='no_operating_point', best=math.nan, lift_on=True
    ) or (info['scan'][-1]['lift_on'] and info['scan'][-1]['status'] == 'no_operating_point')
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (None, None)
    assert not any(rec['lift_on'] for rec in res['samples']) and len(res['samples']) == _N_SLOTS


def slots_fail(dead):
    """A stand-in attempt: the best rate and the life scan pass, and a slot has an operating point unless dead(inp)."""
    return stand_in_attempt(
        lambda inp: ('no_operating_point', None) if 'test' in inp and dead(inp) else ('operating_point', 50.0)
    )


def a_natural_life(design=True):
    """The first life of the tests' seed that no draw lifts, with a lift design (an oil) or without (a gas well)."""
    return next(
        lf
        for lf in drilled_lives()
        if lf.lift_start is None and (lf.design is not None) == design and lf.w0['fluid_class'] not in _TABLE_CLASSES
    )


def test_a_dead_date_starts_the_lift_of_a_natural_well(monkeypatch):
    """SMP-106, SMP-90 (D-142): a natural well with a lift design that flows at none of a date's 100 draws is lifted
    from that date, with `no_natural_flow`; its slots are planned and solved again, and the log holds that pass."""
    lf = a_natural_life()
    T = lf.T_life
    attempt, calls = slots_fail(lambda inp: not inp['lift_on'] and 0.33 * T <= inp['t'] < 0.36 * T)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    natural, _ = _nsol2_plan(lf, T, None)
    t_dead = min(slot['t'] for slot in natural if slot['t'] >= 0.33 * T)
    n_dead = sum(0.33 * T <= slot['t'] < 0.36 * T for slot in natural)
    assert n_dead == 5 and info['dead_dates'] == [dict(t=t_dead, slots=n_dead, action='lift_start')]
    assert (info['outcome'], info['T_end'], info['T_scan']) == ('accepted', T, T)
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (t_dead, 'no_natural_flow')
    plan, p_c = _nsol2_plan(lf, T, t_dead)
    assert info['p_change'] == p_c and len(res['samples']) == _N_SLOTS
    assert [(rec['t'], rec['lift_on'], rec['u']) for rec in res['samples']] == [
        (slot['t'], slot['t'] >= t_dead, slot['u']) for slot in plan
    ]
    slots = [rec for rec in res['log'] if rec['kind'] == 'slot']
    assert len(slots) == _N_SLOTS and all(rec['attempt'] == 0 for rec in slots)  # The second pass's attempts only
    assert sum('test' in c for c in calls) == _N_SLOTS + (_SLOT_LIMIT - 1) * n_dead + _N_SLOTS


def test_a_dead_date_behind_the_redraw_limit(monkeypatch):
    """SMP-106 (D-142): more than ten dead slots use the thousand redraws before any has its hundred attempts. The
    earliest slot still open then takes the rest of its attempts: without an operating point it is a dead date, and
    with one the well is dropped at the redraw limit."""
    lf = a_natural_life()
    T = lf.T_life
    window = lambda inp: not inp['lift_on'] and 0.3 * T <= inp['t'] < 0.5 * T  # noqa: E731
    attempt, calls = slots_fail(window)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    natural, _ = _nsol2_plan(lf, T, None)
    t_dead = min(slot['t'] for slot in natural if slot['t'] >= 0.3 * T)
    n_dead = sum(0.3 * T <= slot['t'] < 0.5 * T for slot in natural)
    assert n_dead == 42 and res['info']['dead_dates'] == [dict(t=t_dead, slots=n_dead, action='lift_start')]
    assert res['info']['outcome'] == 'accepted' and res['nsol']['lift_start'] == t_dead
    passes, rest = divmod(_REDRAW_LIMIT, n_dead)
    first = _SLOT_LIMIT - (1 + passes + (rest > 0))  # The attempts that the earliest dead slot had left
    assert sum('test' in c for c in calls) == _N_SLOTS + _REDRAW_LIMIT + first + _N_SLOTS

    seen = []  # The earliest slot of the window flows at its 96th attempt: the well is at the redraw limit

    def dead(inp):
        seen.append(inp['t'])
        return window(inp) and not (inp['t'] == t_dead and seen.count(t_dead) > 95)

    attempt, calls = slots_fail(dead)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    assert res['info']['outcome'] == 'redraw_limit' and res['info']['dead_dates'] == []
    assert len(res['samples']) == _N_SLOTS - n_dead + 1 and res['samples'][-1]['attempt'] < _SLOT_LIMIT
    assert max(rec['attempt'] for rec in res['log'] if rec['kind'] == 'slot') == 95


@pytest.mark.parametrize('design', [True, False])
def test_a_dead_date_ends_the_life_of_a_well_that_lift_cannot_help(monkeypatch, design):
    """SMP-106 (D-142): where a date is dead for a lifted well, or for one without a lift design, the life ends at the
    last earlier date. A natural well with a design is lifted first, and its life ends when the date stays dead; the
    lift that would start where the life ends leaves it natural."""
    lf = a_natural_life(design)
    T = lf.T_life
    attempt, _ = slots_fail(lambda inp: inp['t'] >= 0.8 * T)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    natural, _ = _nsol2_plan(lf, T, None)
    t_dead = min(slot['t'] for slot in natural if slot['t'] >= 0.8 * T)
    t_before = max(slot['t'] for slot in natural if slot['t'] < t_dead)
    n_dead = sum(slot['t'] >= 0.8 * T for slot in natural)
    assert [d['action'] for d in info['dead_dates']] == (['lift_start', 'life_end'] if design else ['life_end'])
    assert all((d['t'], d['slots']) == (t_dead, n_dead) for d in info['dead_dates'])
    assert (info['outcome'], info['T_end'], info['T_scan']) == ('accepted', t_before, T)
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (None, None)
    plan, _ = _nsol2_plan(lf, t_before, None)
    assert [rec['t'] for rec in res['samples']] == [slot['t'] for slot in plan] and plan[-1]['t'] == t_before
    assert not any(rec['lift_on'] for rec in res['samples'])


def test_a_dead_date_at_the_lifes_end_cuts_the_life(monkeypatch):
    """A natural well with a design whose last date is dead is not lifted for that date alone: its life ends at the
    date before."""
    lf = a_natural_life()
    T = lf.T_life
    attempt, _ = slots_fail(lambda inp: inp['t'] == T)
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    natural, _ = _nsol2_plan(lf, T, None)
    t_before = max(slot['t'] for slot in natural if slot['t'] < T)
    assert res['info']['dead_dates'] == [dict(t=T, slots=1, action='life_end')]
    assert (res['info']['outcome'], res['info']['T_end'], res['nsol']['lift_start']) == ('accepted', t_before, None)


def test_the_dead_date_limit_and_a_life_cut_to_nothing(monkeypatch):
    """SMP-106: a well is dropped as `dead_date` when a date is dead after DEAD_DATE_LIMIT dead dates have changed its
    life, with the attempts of its last pass; a life that a dead date cuts to its start is `short_life`."""
    lf = a_natural_life()
    attempt, _ = slots_fail(lambda inp: inp.get('test') == 8)  # The last test period is dead in every plan
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info = res['info']
    assert info['outcome'] == 'dead_date' and len(res['samples']) == _N_SLOTS - 5
    assert [d['action'] for d in info['dead_dates']] == ['lift_start', 'life_end'] * 2 + ['lift_start']
    assert len(info['dead_dates']) == _DEAD_DATE_LIMIT and info['T_end'] < info['T_scan']
    slots = [rec for rec in res['log'] if rec['kind'] == 'slot']
    assert len(slots) == _N_SLOTS - 5 + 5 * _SLOT_LIMIT

    attempt, _ = slots_fail(lambda inp: inp['t'] > 0.0 and not inp['lift_on'] or inp['lift_on'])
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, a_natural_life(design=False).k)
    assert (res['info']['outcome'], res['info']['T_end']) == ('short_life', 0.0)
    assert [d['action'] for d in res['info']['dead_dates']] == ['life_end']


# --- Wells dropped before a solve -------------------------------------------------------------------------------------
def no_solve(monkeypatch):
    def attempt(builder, inp, wellhead=None):
        raise AssertionError('a well dropped before its solves was solved')

    monkeypatch.setattr(generate, '_attempt', attempt)


def is_table(lf) -> bool:
    return lf.w0['fluid_class'] in _TABLE_CLASSES


def a_table_well() -> int:
    return next(lf.k for lf in drawn_lives() if is_table(lf))


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_table_well_without_a_usable_table(monkeypatch, tmp_path, kind):
    """SMP-106, SMP-116: a volatile oil or condensate without a table, or whose table failed, is `table_failed`
    and not solved; its draws are made without a table."""
    no_solve(monkeypatch)
    k = a_table_well()
    run = generate_sol2_well if kind == 'sol' else generate_nsol2_well
    tw.write_table_wells(tmp_path, kind, SEED, k + 1)  # The build's first step, and no table yet
    for tables, step in ((None, 'missing'), (tmp_path, 'missing')):
        res = run(SEED, k, tables)
        assert (res['info']['outcome'], res['table']['status'], res['table']['step']) == (
            'table_failed',
            'failed',
            step,
        )
        assert res['info']['error'] == res['table']['error'] and res['log'] == res['samples'] == []
    (tmp_path / f'k{k:06d}.json').write_text(
        json.dumps(dict(kind=kind, seed=SEED, status='failed', step='tables', error='p/Z falls'))
    )
    res = run(SEED, k, tmp_path)
    assert (res['info']['outcome'], res['info']['error'], res['table']['step']) == (
        'table_failed',
        'p/Z falls',
        'tables',
    )
    assert 'pvt' not in res['table'] and res['well']['fluid_class'] in _TABLE_CLASSES
    # SMP-115: no fluid of its class among its draws is no failure of the build: the well is not producible
    failed = dict(kind=kind, seed=SEED, status='failed', step='class', error='no fluid of the class')
    (tmp_path / f'k{k:06d}.json').write_text(json.dumps(failed))
    info = run(SEED, k, tmp_path)['info']
    assert (info['outcome'], info['not_producible_reason']) == ('not_producible', 'no_fluid_of_class')
    assert info['error'] == 'no fluid of the class'
    if kind == 'sol':
        drawn = sol2_well(drawn_lives()[k])
        assert set(res['well']) == set(drawn) and all(nan_or(res['well'][key], v) for key, v in drawn.items())
    else:
        assert math.isnan(res['info']['T_end']) and res['info']['scan'] == [] and res['nsol']['T_life'] > 0


def own_hook(w) -> dict:
    """A table fluid's values as the draws give them: the hook that changes nothing."""
    return dict(p_sat=w['p_sat'], R_i=w['R_i'], rho_o=w['rho_o'], rho_g=w['rho_g'])


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_folder_of_tables_built_from_other_draws_stops_the_run(monkeypatch, tmp_path, kind):
    """SMP-116: a folder whose record of a table well is not what the run's draws give, as after a rule changed
    since the build, or that records a well of another class, stops the run (TableMismatch): it is no outcome of the
    well. A draw index beyond the folder's range has no table."""
    no_solve(monkeypatch)
    k = a_table_well()
    other = next(lf.k for lf in drawn_lives() if lf.k > k and lf.w0['fluid_class'] not in _TABLE_CLASSES)
    run = generate_sol2_well if kind == 'sol' else generate_nsol2_well
    spec = tw.write_table_wells(tmp_path, kind, SEED, other + 1)
    path = tmp_path / 'wells.json'
    rec = next(w for w in spec['wells'] if w['well'] == k)
    for key, value in (('p_i', rec['p_i'] + 1.0), ('T_axis', rec['T_axis'][:-1]), ('composition_seed', 7)):
        tw._write_json(path, dict(spec, wells=[dict(w, **{key: value}) if w is rec else w for w in spec['wells']]))
        with pytest.raises(TableMismatch, match=f'were not built from the draws of draw index {k}'):
            run(SEED, k, tmp_path)
        with pytest.raises(TableMismatch):
            runner._run_well(kind, SEED, k, tmp_path)  # Not a crash
    stray = dict(rec, well=other)  # A well that the draws give another class
    tw._write_json(path, dict(spec, wells=spec['wells'] + [stray]))
    with pytest.raises(TableMismatch, match=f'draw index {other}'):
        run(SEED, other, tmp_path)
    tw._write_json(path, dict(spec, wells_range=[0, k]))  # The folder covers draw indices below k only
    assert run(SEED, k, tmp_path)['table']['step'] == 'missing'


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_gor_off_the_tables_axis_at_the_nominal_state_is_invalid_input(monkeypatch, kind):
    """SMP-106: where the table refuses the GOR of the well's nominal state (nsol-2: its start), the well is
    `invalid_input`, not a crash; its draws are made without the table's coefficients."""
    no_solve(monkeypatch)
    lf = next(lf for lf in drilled_lives() if lf.w0['fluid_class'] == 'volatile_oil')

    def refuse(gor):
        raise ValueError(f"gor {gor} is outside the PVT table's GOR axis, 1.0 to 2.0 Sm3/Sm3")

    pvt = types.SimpleNamespace(rho_o=lf.w0['rho_o'], rho_g=lf.w0['rho_g'], at_gor=refuse)
    table = dict(status='ok', hook=own_hook(lf.w0), pvt=pvt)
    monkeypatch.setattr(generate, '_table', lambda seed, k, kind, tables: table)
    res = (generate_sol2_well if kind == 'sol' else generate_nsol2_well)(SEED, lf.k)
    assert res['info']['outcome'] == 'invalid_input' and "outside the PVT table's GOR axis" in res['info']['error']
    assert res['log'] == [] and res['well']['well'] == lf.k
    if kind == 'sol':
        assert all(math.isnan(res['well'][key]) for key in _COEF_KEYS) and 'date' in res['well']


def test_a_well_of_another_class_takes_no_table(monkeypatch):
    lf, _ = a_sol2_well(oil=True, table=False)
    attempt, _ = stand_in_attempt(lambda inp: ('no_operating_point', None))
    monkeypatch.setattr(generate, '_attempt', attempt)
    assert generate_sol2_well(SEED, lf.k, tables=None)['table'] is None


@pytest.mark.parametrize('run', [generate_sol2_well, generate_nsol2_well])
def test_a_well_outside_the_configuration_is_invalid_input(monkeypatch, run):
    """SMP-106: a well that fails configurations.check at its nominal state (nsol-2: its start) is not solved."""
    no_solve(monkeypatch)

    def refuse(wp, configuration):
        assert configuration == V2
        raise ValueError('well is not in the v2.0.0 configuration: a stand-in difference')

    monkeypatch.setattr(generate, 'check', refuse)
    lf, _ = a_sol2_well(oil=True)
    res = run(SEED, lf.k)
    assert (res['info']['outcome'], res['log']) == ('invalid_input', [])
    assert res['info']['error'].endswith('a stand-in difference')


# --- Real wells, end to end -------------------------------------------------------------------------------------------
N_REAL_SLOTS = 12  # The slots of the slow tests' wells


def check_records(res):
    """What every record of a real well's log holds, by its outcome and reason (SMP-104)."""
    for rec in res['log']:
        outcome, reason = rec['outcome'], rec['reason']
        searched = reason not in ('p_r_le_p_s', 'exception')
        assert all(
            math.isfinite(rec[key]) == searched for key in ('marches', 'non_finite', 'rejected', 'rejected_at_edge')
        )
        if outcome == 'operating_point':
            assert reason is None and rec['n_stable'] >= 1 and generate._valid_row({key: rec[key] for key in FEATURES})
        elif reason == 'no_root':
            assert (
                outcome == 'no_operating_point' and rec['n_roots'] == 0 and rec['rejected'] == rec['rejected_at_edge']
            )
        elif reason == 'p_r_le_p_s':
            assert outcome == 'no_operating_point' and rec['p_r'] <= rec['p_s']
        elif reason == 'incomplete_scan':
            assert outcome == 'solver_failure' and rec['n_roots'] == 0 and rec['rejected'] > rec['rejected_at_edge']
        elif reason == 'unstable_only':
            assert outcome == 'solver_failure' and rec['n_roots'] > 0 == rec['n_stable']
        else:
            assert (outcome, reason) in (('solver_failure', 'exception'), ('solver_failure', 'invalid_row'))
    kinds = {r['kind'] for r in res['log']}
    kind = 'scan' if 'scan' in kinds else ('best_rate_lifted' if 'best_rate_lifted' in kinds else 'best_rate')
    first = [
        r for r in res['log'] if r['kind'] == kind and r['slot'] == -1
    ]  # The best rate's solves (nsol-2: at t = 0)
    ops = [r['q_oe'] for r in first if r['outcome'] == 'operating_point']
    assert nan_or(res['info']['best_rate'], max(ops) if ops else math.nan)
    assert res['info']['best_rate_outcomes'] == [r['outcome'] for r in first]


def first_accepted(run, ks, monkeypatch):
    monkeypatch.setattr(generate, '_N_SLOTS', N_REAL_SLOTS)
    for k in ks:
        res = run(SEED, k)
        if res['info']['outcome'] == 'accepted':
            return k, res
    raise LookupError('no well was accepted')


@pytest.mark.slow
def test_a_real_sol2_well(monkeypatch):
    """A natural black oil and a lifted one, each solved end to end and again to the bit."""
    for lifted in (False, True):
        ks = [
            lf.k
            for lf, w in zip(drawn_lives(), drawn_sol2_wells())
            if w['fluid_class'] == 'black_oil' and (w['lift'] == 'gas_lift') == lifted
        ]
        k, res = first_accepted(generate_sol2_well, ks[:6], monkeypatch)
        check_records(res)
        assert [r['slot'] for r in res['samples']] == list(range(N_REAL_SLOTS))
        assert all(r['outcome'] == 'operating_point' and r['kind'] == 'slot' for r in res['samples'])
        again = generate_sol2_well(SEED, k)  # A rerun gives the same attempts and solves, to the bit (SMP-109)
        assert len(again['log']) == len(res['log'])
        for r1, r2 in zip(again['log'], res['log']):
            assert set(r1) == set(r2) and all(nan_or(r1[a], r2[a]) for a in r1 if a not in ('t_build', 't_solve'))


@pytest.mark.slow
def test_a_real_nsol2_well(monkeypatch):
    """An oil well's life scan and slots, solved end to end."""
    ks = [lf.k for lf in drawn_lives() if lf.w0['fluid_class'] == 'black_oil']
    k, res = first_accepted(generate_nsol2_well, ks[:6], monkeypatch)
    check_records(res)
    info, nsol = res['info'], res['nsol']
    scan_recs = [r for r in res['log'] if r['kind'] == 'scan']
    assert {r['slot'] for r in scan_recs} == {-1 - i for i in range(len(info['scan']))}
    assert 0 < info['T_end'] <= nsol['T_life'] and len(res['samples']) == N_REAL_SLOTS
    assert all(
        rec['t'] <= info['T_end'] and rec['lift_on'] == _lift_on(rec['t'], nsol['lift_start']) for rec in res['samples']
    )
