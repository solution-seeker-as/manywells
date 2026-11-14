"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the v2 sample draws (manywells.sampling.v2.samples; specs/sampling.md, SMP-93 to SMP-100 and the redraws of
SMP-105): a sol-2 slot's controls, separator pressure and drift, an nsol-2 well's plan of 200 slots (the dates, the
control history, the test periods and the drift), and what a redraw draws again and what it keeps. No test solves a
well.
"""

import copy
import math

import numpy as np
import pytest

from manywells.configurations import V2, check
from manywells.sampling.v2 import samples
from manywells.sampling.v2.inflow import _COEF_KEYS
from manywells.sampling.v2.life import _lift_on
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.reservoir import _producing_gor
from manywells.sampling.v2.samples import _LIFT_GAS_ALWAYS_ON, _N_SLOTS, _nsol2_attempt, _nsol2_plan, _sol2_attempt
from manywells.sampling.wells import rng_for

from .sampling_v2_wells import N_MANY_LIVES, SEED, drawn_lives, drawn_sol2_wells, is_oil

STATE_KEYS = ['t', 'p_r', 'gor', 'wlr', 'wgr', 'h', 'roughness', 'p_s_nom', 'life_stage', *_COEF_KEYS]


@pytest.fixture(scope='session')
def lives():
    return drawn_lives()


@pytest.fixture(scope='session')
def sol2_wells():
    return drawn_sol2_wells()


@pytest.fixture(scope='session')
def sol2_attempts(sol2_wells):
    """Every slot's first attempt of every sol-2 well."""
    return [[_sol2_attempt(SEED, w, j) for j in range(_N_SLOTS)] for w in sol2_wells]


@pytest.fixture(scope='session')
def plans(lives):
    """Each life's plan to the end of its life span, with the lift's start as the draws give it."""
    return [_nsol2_plan(lf, lf.T_life, lf.lift_start) for lf in lives]


def between(x, lo, hi, tol=1e-9):
    return lo - tol * max(1.0, abs(lo)) <= x <= hi + tol * max(1.0, abs(hi))


def off_share(hits, n, p) -> bool:
    """Whether a share of hits in n lies more than 4 standard errors from p."""
    return abs(hits - n * p) > 4 * math.sqrt(n * p * (1 - p)) + 0.5


def ks(x) -> float:
    """Kolmogorov's statistic, sqrt(n) D, of a sample against U(0, 1): above 1.95 at the 0.1% level."""
    x = np.sort(np.asarray(x, dtype=float))
    n, i = len(x), np.arange(1, len(x) + 1)
    return math.sqrt(n) * max(np.max(i / n - x), np.max(x - (i - 1) / n))


def separator_place(p_s, p_nom, dp_fl) -> float:
    """SMP-95: the separator pressure's place in its interval, from 0 to 1."""
    lo, hi = max(0.9 * p_nom, 1.5, dp_fl), 1.1 * p_nom
    assert lo < hi and between(p_s, lo, hi)  # No drawn well's interval is empty
    return (p_s - lo) / (hi - lo)


def reflect(x, lo, hi):
    """A step out of [lo, hi] comes back in by as much."""
    x = 2 * hi - x if x > hi else x
    return 2 * lo - x if x < lo else x


# --- sol-2 -----------------------------------------------------------------------------------------------------------
def check_sol2_attempt(w, inp) -> dict:
    """SMP-93 to SMP-96: the bounds that no attempt of a sol-2 slot breaks. Returns each draw's place in its range."""
    lifted, oil = w['lift'] == 'gas_lift', is_oil(w)
    assert list(inp) == ['h', 'roughness', *_COEF_KEYS, 'lift_on', 'u', 'w_lg', 'p_s', 'p_r', 'gor', 'wlr', 'wgr']
    # The well's own: nothing of it drifts (SMP-96)
    assert (inp['h'], inp['roughness'], inp['lift_on']) == (w['h'], w['roughness'], lifted)
    assert all(inp[key] == w[key] or (math.isnan(inp[key]) and math.isnan(w[key])) for key in _COEF_KEYS)
    place = dict(u=inp['u'], p_s=separator_place(inp['p_s'], w['p_s'], w['dp_fl']))
    assert between(inp['u'], 0.05, 1)  # SMP-93
    if lifted:  # SMP-94
        place['w_lg'] = inp['w_lg'] / w['w_lg_max']
        assert between(place['w_lg'], 0, 1)
    else:
        assert inp['w_lg'] == 0
    # SMP-96: the pressure, the GOR at the drifted pressure, and the water
    place['p_r'] = (inp['p_r'] / w['p_r'] - 0.98) / 0.04
    place['gor'] = (inp['gor'] / _producing_gor(w, inp['p_r']) - 0.95) / 0.1
    assert between(place['p_r'], 0, 1) and between(place['gor'], 0, 1)
    if oil:
        assert math.isnan(inp['wgr']) and inp['wlr'] <= 0.98
        assert inp['wlr'] == 0.98 or w['wlr'] == inp['wlr'] == 0 or between(inp['wlr'] / w['wlr'], 0.95, 1.05)
    else:
        place['wgr'] = (inp['wgr'] / w['wgr'] - 0.95) / 0.1
        assert between(place['wgr'], 0, 1) and inp['wlr'] == inp['wgr'] / (inp['wgr'] + 1e6 / inp['gor'])
    return place


def sol2_failures(wells, attempts) -> list:
    """The bounds of every attempt, and the shares and marginals of SMP-93 to SMP-96 that lie further from the
    rule's than their sampling error allows."""
    places = {key: [] for key in ('u', 'w_lg', 'p_s', 'p_r', 'gor', 'wgr')}
    for w, slots in zip(wells, attempts):
        assert len(slots) == _N_SLOTS == 200
        for inp in slots:
            for key, x in check_sol2_attempt(w, inp).items():
                places[key].append(x)
    u, w_lg = np.array(places['u']), np.array(places['w_lg'])
    failures = []
    if off_share(np.sum(u == 1), len(u), 0.05):  # SMP-93
        failures.append(f'fully open: {np.mean(u == 1):.4f} of {len(u)}')
    if off_share(np.sum(w_lg == 0), len(w_lg), 0.20):  # SMP-94
        failures.append(f'lift gas off: {np.mean(w_lg == 0):.4f} of {len(w_lg)}')
    marginals = dict(places, u=(u[u < 1] - 0.05) / 0.95, w_lg=w_lg[w_lg > 0])
    for key, x in marginals.items():
        if len(x) < 500 or ks(x) > 1.95:
            failures.append(f'{key}: n = {len(x)}, sqrt(n) D = {ks(x):.2f}')
    return failures


def test_sol2_attempts(sol2_wells, sol2_attempts):
    """SMP-93 to SMP-96: every slot's bounds; 5% of the slots fully open and the rest uniform from 0.05; a lifted
    well's lift gas off in 20% and else uniform to its capacity, a natural well's off; the separator pressure,
    the reservoir pressure, the GOR at the drifted pressure and a gas well's WGR uniform over their ranges."""
    assert sol2_failures(sol2_wells, sol2_attempts) == []
    lifted = [w for w in sol2_wells if w['lift'] == 'gas_lift']
    assert 10 <= len(lifted) <= len(sol2_wells) - 10


def test_a_sol2_slot_depends_on_no_other(sol2_wells, sol2_attempts):
    """SMP-105, SMP-109: a slot's draw is its own, whatever was drawn before it; another seed gives another."""
    w = sol2_wells[3]
    for j in (150, 0, 77):
        assert _sol2_attempt(SEED, w, j) == sol2_attempts[3][j]
    assert _sol2_attempt(SEED + 1, w, 77)['u'] != sol2_attempts[3][77]['u']
    us = [inp['u'] for inp in sol2_attempts[3] if inp['u'] < 1]
    assert len(set(us)) == len(us)


def test_a_sol2_redraw_draws_the_slot_again(sol2_wells, sol2_attempts):
    """SMP-105: a redraw is the slot's whole draw again from the redraw's own generator, so it keeps the bounds and
    the distribution, and repeats neither the first attempt nor another redraw."""
    wells = sol2_wells[:25]
    redraws = [[_sol2_attempt(SEED, w, j, 1 + j % 3) for j in range(_N_SLOTS)] for w in wells]
    assert sol2_failures(wells, redraws) == []
    w, first = wells[0], sol2_attempts[0]
    for j in (0, 61, 199):
        tries = [first[j]] + [_sol2_attempt(SEED, w, j, a) for a in (1, 2, 49)]
        assert all(len({inp[key] for inp in tries}) == 4 for key in ('p_s', 'p_r', 'gor'))
        us = [inp['u'] for inp in tries if inp['u'] < 1]
        assert len(set(us)) == len(us) >= 2 and _sol2_attempt(SEED, w, j, 2) == tries[2]


def test_a_sol2_well_that_cannot_flow_without_lift_gas(sol2_wells, sol2_attempts):
    """SMP-94: a lifted well flagged lift_gas_always_on draws no slot without lift gas, at the first attempt or a
    redraw; the rest of each draw is the unflagged well's, the lift gas's off number being drawn and left unused."""
    i, w = next((i, w) for i, w in enumerate(sol2_wells) if w['lift'] == 'gas_lift')
    flagged = dict(w, flags=';'.join([f for f in w['flags'].split(';') if f] + [_LIFT_GAS_ALWAYS_ON]))
    for a in (0, 1, 7):
        for j in range(_N_SLOTS):
            plain = sol2_attempts[i][j] if a == 0 else _sol2_attempt(SEED, w, j, a)
            inp = _sol2_attempt(SEED, flagged, j, a)
            assert 0 < inp['w_lg'] <= w['w_lg_max']
            assert {key: v for key, v in inp.items() if key != 'w_lg'} == {
                key: v for key, v in plain.items() if key != 'w_lg'
            }
            assert plain['w_lg'] in (0.0, inp['w_lg'])
    off = [inp['w_lg'] == 0 for inp in sol2_attempts[i]]
    assert 0 < sum(off) < _N_SLOTS  # The unflagged well has slots without lift gas


def test_sol2_attempts_map_to_the_model(lives, sol2_wells, sol2_attempts):
    """An attempt's inputs are a well and its boundary conditions in the v2.0.0 configuration (SMP-111)."""
    for lf, w, slots in list(zip(lives, sol2_wells, sol2_attempts))[:12]:
        builder = WellBuilder(lf.w0, lf.stations, design=lf.design)
        for inp in slots[:3]:
            wp, bc = builder.case(inp)
            check(wp, V2)
            assert (bc.u, bc.p_r, bc.p_s, bc.w_lg) == (inp['u'], inp['p_r'], inp['p_s'], inp['w_lg'])
            assert (wp.fluid.gor, wp.fluid.wlr, wp.thermal.h) == (inp['gor'], inp['wlr'], inp['h'])
            assert (wp.injection_md is not None) == (w['lift'] == 'gas_lift')


# --- nsol-2: the plan ------------------------------------------------------------------------------------------------
def check_plan(lf, plan, p_c, T_end, lift_start) -> dict:
    """SMP-97 to SMP-100: what no plan breaks. Returns the plan's changes of the set points and its draws' places
    in their ranges, for the shares."""
    w0 = lf.w0
    w_max = lf.design['w_lg_max'] if lf.design is not None else None
    assert len(plan) == _N_SLOTS == 200 and between(p_c, 0.2, 0.95)

    # Dates and slots (SMP-97): 160 operating samples and 8 tests of 5, in the order of their dates
    op = [s for s in plan if s['kind'] == 'operating']
    assert len(op) == 160 and [s['kind'] for s in plan].count('test') == 40
    assert [s['t'] for s in op] == pytest.approx([j * T_end / 159 for j in range(160)], rel=1e-12)
    assert [s['t'] for s in plan] == sorted(s['t'] for s in plan)
    assert all(s['test'] is None and s['sweep'] is None for s in op)

    # Every slot: the lift by its start, the controls' ranges, the separator's jitter and the drift's bounds
    places = dict(p_s=[], eps=[])
    for s in plan:
        assert list(s) == ['t', 'kind', 'test', 'sweep', 'lift_on', 'u', 'w_lg', 'p_s', 'eps']
        assert s['lift_on'] == _lift_on(s['t'], lift_start) and between(s['u'], 0.05, 1)
        if not s['lift_on']:
            assert s['w_lg'] == 0  # No lift gas before lift starts (SMP-90)
        elif s['sweep'] == 'lift':
            assert between(s['w_lg'], 0, w_max)
        else:  # The set point: shut in, or at least a tenth of the capacity (SMP-98)
            assert s['w_lg'] == 0 or between(s['w_lg'], 0.1 * w_max, w_max)
        places['p_s'].append(separator_place(s['p_s'], lf.p_s_nom(s['t']), w0['dp_fl']))  # SMP-95
        assert all(between(e, 0.95, 1.05) for e in s['eps']) and len(s['eps']) == 2  # SMP-100
    assert len({s['eps'] for s in op}) == 160  # A drift of its own for every operating sample
    places['eps'] = [(e - 0.95) / 0.1 for s in op for e in s['eps']]

    # The control history (SMP-98): set points constant between changes, which come with the well's probability
    u = [s['u'] for s in op]
    du = [b - a for a, b in zip(u, u[1:]) if a != b]
    assert abs(len(du) - 159 * p_c) <= 4 * math.sqrt(159 * p_c * (1 - p_c)) + 1
    on = [s['w_lg'] / w_max for s in op if s['lift_on']]
    dw = [b - a for a, b in zip(on, on[1:]) if a != b and a > 0]  # A shut-in set point comes back by reflection

    # Test periods (SMP-99)
    for m in range(1, 9):
        j0 = next(j for j, s in enumerate(plan) if s['test'] == m)
        five, before = plan[j0 : j0 + 5], plan[j0 - 1]
        assert all(s['test'] == m and s['kind'] == 'test' for s in five) and before['kind'] == 'operating'
        assert all(s['t'] == pytest.approx((m - 0.5) * T_end / 8, rel=1e-12) for s in five)
        assert len({s['t'] for s in five}) == len({s['eps'] for s in five}) == 1  # One date, one drift
        lifted = five[0]['lift_on']
        sweep = 'lift' if lifted and m % 2 == 0 else 'choke'
        assert all(s['sweep'] == sweep for s in five)
        if sweep == 'choke':  # Fully open, then the quarters of [0.05, 1) from the top down
            x = [(s['u'] - 0.05) / 0.95 for s in five]
            assert five[0]['u'] == 1 and len({s['w_lg'] for s in five}) == 1  # At the operating lift-gas rate
            assert before['lift_on'] != lifted or five[0]['w_lg'] == before['w_lg']
        else:  # The capacity, then the quarters of [0, 1) of it from the top down
            x = [s['w_lg'] / w_max for s in five]
            assert five[0]['w_lg'] == w_max and all(s['u'] == before['u'] for s in five)  # At the operating choke
        assert all(between(x[i], (4 - i) / 4, (5 - i) / 4) for i in range(1, 5))
    return dict(places, du=du, dw=dw, on=on, u=u)


def plan_failures(stats) -> list:
    """The shares of SMP-95, SMP-98 and SMP-100, pooled over plans, that lie outside their sampling error."""
    failures = []
    du, dw = (np.abs(np.concatenate([s[key] for s in stats])) for key in ('du', 'dw'))
    on = np.concatenate([s['on'] for s in stats])
    # A change is a step of at most 0.1 with probability 0.9; a new draw lands within 0.1 in about a sixth of its
    # draws, and the lift gas's is 0 in a fifth of them
    for name, steps, lo, hi in (('choke', du, 0.9, 0.935), ('lift gas', dw, 0.88, 0.95)):
        small = np.mean(steps <= 0.1 + 1e-12)
        if len(steps) < 1000 or not lo < small < hi:
            failures.append(f'{name}: {small:.4f} of {len(steps)} changes are steps of at most 0.1')
    if not 0.005 < np.mean(on == 0) < 0.06:  # Shut in by a new draw only, until the next change
        failures.append(f'lift gas shut in: {np.mean(on == 0):.4f} of {len(on)} operating samples with lift')
    # A step is reflected into the range, not stopped at its end: only a new draw opens the choke fully
    u = np.concatenate([s['u'] for s in stats])
    if not 0.001 < np.mean(u == 1) < 0.02 or np.any(u == 0.05) or np.any(on == 1) or np.any(on == 0.1):
        failures.append(f'set points at the ends of their ranges: {np.mean(u == 1):.4f} fully open')
    for key in ('p_s', 'eps'):
        x = np.concatenate([s[key] for s in stats])
        if ks(x) > 1.95:
            failures.append(f'{key}: n = {len(x)}, sqrt(n) D = {ks(x):.2f}')
    return failures


def test_plans(lives, plans):
    """SMP-97 to SMP-100: 200 slots in the order of their dates, 160 operating samples and 8 tests of 5; the set
    points constant between changes, with the changes' share and steps; the tests' sweeps; the separator's jitter
    and the drift's bounds."""
    stats = [check_plan(lf, plan, p_c, lf.T_life, lf.lift_start) for lf, (plan, p_c) in zip(lives, plans)]
    assert plan_failures(stats) == []
    sweeps = [{s['sweep'] for s in plan} for plan, _ in plans]
    assert {None, 'choke', 'lift'} in sweeps and {None, 'choke'} in sweeps  # Wells with lift, and natural ones


def other_controls(lf, n) -> list:
    """The plans of n other wells' control draws on one life: the life under another well's index, which only the
    controls' generator takes."""
    plans = []
    for i in range(n):
        other = copy.copy(lf)
        other.k = 100_000 + i
        plans.append(_nsol2_plan(other, lf.T_life, lf.lift_start))
    return plans


def test_set_points_at_the_start(lives):
    """SMP-98: the choke's set point starts fully open with probability 0.05, else uniform from 0.05; the lift
    gas's starts uniform over [0.1, 1] of the capacity; and the probability of a change is log-uniform."""
    lf = next(lf for lf in lives if lf.lift_start_rule == 'from_start')
    drawn = other_controls(lf, 400)
    u_0 = np.array([plan[0]['u'] for plan, _ in drawn])
    w_0 = np.array([plan[0]['w_lg'] for plan, _ in drawn]) / lf.design['w_lg_max']
    p_c = np.array([p_c for _, p_c in drawn])
    assert not off_share(np.sum(u_0 == 1), len(u_0), 0.05) and ks((u_0[u_0 < 1] - 0.05) / 0.95) < 1.95
    assert ks((w_0 - 0.1) / 0.9) < 1.95
    assert ks(np.log(p_c / 0.2) / math.log(0.95 / 0.2)) < 1.95
    stats = [check_plan(lf, plan, p_c, lf.T_life, lf.lift_start) for plan, p_c in drawn[:40]]
    assert plan_failures(stats) == []


def test_a_plans_first_draws(lives, plans):
    """SMP-98, SMP-109: the plan's first numbers from the well's `controls` generator: the probability of a change,
    then the set points at the start, which the first operating sample holds."""
    for lf, (plan, p_c) in list(zip(lives, plans))[:30]:
        r = rng_for(SEED, lf.k, 'v2', 'controls')
        assert p_c == math.exp(r.uniform(math.log(0.2), math.log(0.95)))
        fully_open, u_0, w_0 = r.random(), r.uniform(0.05, 1.0), r.random()
        assert plan[0]['t'] == 0 and plan[0]['u'] == (1.0 if fully_open < 0.05 else u_0)
        if lf.lift_start == 0:
            assert plan[0]['w_lg'] == (0.1 + 0.9 * w_0) * lf.design['w_lg_max']


def test_a_plan_spans_the_life_to_its_end(lives, plans):
    """SMP-97, SMP-98: the dates scale with the end of the life, and the choke's history, the sweeps and the drift
    are the same draws whatever the end, since no set point follows a solve."""
    natural = next(i for i, lf in enumerate(lives) if lf.lift_start is None)
    lf, (plan, p_c) = lives[natural], plans[natural]
    T_end = float(np.linspace(0.0, lf.T_life, 21)[7])  # A scan point (SMP-101)
    short, p_c_short = _nsol2_plan(lf, T_end, None)
    check_plan(lf, short, p_c_short, T_end, None)
    assert p_c_short == p_c and short[-1]['t'] == T_end and plan[-1]['t'] == lf.T_life
    for a, b in zip(short, plan):
        assert a['t'] == pytest.approx(b['t'] * T_end / lf.T_life, rel=1e-12)
        assert all(a[key] == b[key] for key in ('kind', 'test', 'sweep', 'u', 'w_lg', 'eps', 'lift_on'))


def test_a_plan_takes_the_lift_start_of_the_scan(lives, plans):
    """SMP-90's rule 3, SMP-101: with the lift started earlier by the life scan, the lift gas follows its set point
    from that date, in sweeps too, and the choke's history stays; a well without a design stays natural."""
    i = next(i for i, lf in enumerate(lives) if lf.design is not None and lf.lift_start is None)
    lf, (plan, p_c) = lives[i], plans[i]
    start = float(np.linspace(0.0, lf.T_life, 21)[6])
    lifted, p_c_lifted = _nsol2_plan(lf, lf.T_life, start)
    check_plan(lf, lifted, p_c_lifted, lf.T_life, start)
    assert p_c_lifted == p_c
    for a, b in zip(lifted, plan):
        assert a['lift_on'] == (a['t'] >= start) and not b['lift_on'] and b['w_lg'] == 0
        if a['kind'] == 'operating':
            assert a['u'] == b['u'] and a['eps'] == b['eps']
    assert {s['sweep'] for s in lifted} == {None, 'choke', 'lift'} and {s['sweep'] for s in plan} == {None, 'choke'}
    gas = next(lf for lf in lives if lf.design is None)
    with pytest.raises(ValueError, match='lift design'):
        _nsol2_plan(gas, gas.T_life, 0.5 * gas.T_life)


# --- nsol-2: attempts and redraws ------------------------------------------------------------------------------------
def some_lives(lives, plans) -> list:
    """A few lives with their plans: oils and gas classes, natural and lifted. A gas well's state takes a fit per
    date, so the tests take few of its slots."""
    picked, seen = [], set()
    for lf, (plan, _) in zip(lives, plans):
        kind = (lf.oil, lf.lift_start is None)
        if kind not in seen or (lf.oil and len(picked) < 8):
            seen.add(kind)
            picked.append((lf, plan))
    assert len(seen) == 4
    return picked


def test_a_planned_attempt(lives, plans):
    """SMP-97, SMP-100: a slot's first attempt holds the life's state at its date with the date's drift, the
    reservoir pressure without noise, and the planned controls and separator pressure."""
    for lf, plan in some_lives(lives, plans):
        builder = WellBuilder(lf.w0, lf.stations, design=lf.design)
        for j in range(0, _N_SLOTS, 11 if lf.oil else 40):
            slot, inp = plan[j], _nsol2_attempt(lf, plan, j)
            assert list(inp) == STATE_KEYS + ['u', 'w_lg', 'p_s', 'lift_on', 'test', 'sweep']
            st = lf.state(slot['t'], slot['eps'])
            assert all(inp[key] == st[key] or (math.isnan(inp[key]) and math.isnan(st[key])) for key in STATE_KEYS)
            assert inp['p_r'] == lf.p_r(slot['t']) and inp['gor'] == _producing_gor(lf.w0, inp['p_r']) * slot['eps'][0]
            assert all(inp[key] == slot[key] for key in ('u', 'w_lg', 'p_s', 'lift_on', 'test', 'sweep'))
            wp, bc = builder.case(inp)  # A well and its boundary conditions in the v2.0.0 configuration (SMP-111)
            check(wp, V2)
            assert (bc.u, bc.p_r, bc.p_s, bc.w_lg) == (inp['u'], inp['p_r'], inp['p_s'], inp['w_lg'])
            assert (wp.injection_md is not None) == slot['lift_on']


def test_a_redraw_keeps_the_date_and_changes_the_controls(lives, plans):
    """SMP-105: an nsol-2 redraw keeps the slot's date and everything that follows from it, draws the separator
    pressure again, and changes both controls from the planned ones, the lift gas only where lift is installed:
    each by a step of at most 0.2 of its range with probability 0.6, reflected into the range, else a new draw.
    The redrawn controls hold for that slot only."""
    du, places, n_lifted = [], [], 0
    for lf, plan in some_lives(lives, plans):
        w_max = lf.design['w_lg_max'] if lf.design is not None else None
        before = copy.deepcopy(plan)
        for j in range(0, _N_SLOTS, 3 if lf.oil else 40):
            slot, first = plan[j], _nsol2_attempt(lf, plan, j)
            for a in (1, 2, 17):
                inp = _nsol2_attempt(lf, plan, j, a)
                kept = [key for key in inp if key not in ('u', 'w_lg', 'p_s')]
                assert all(inp[key] == first[key] or inp[key] != inp[key] for key in kept) and len(kept) == 16
                assert between(inp['u'], 0.05, 1)
                places.append(separator_place(inp['p_s'], inp['p_s_nom'], lf.w0['dp_fl']))
                # The redraw's numbers from its own generator, in their order
                r = rng_for(SEED, lf.k, j, a, 'v2', 'redraw')
                small_u, step_u, open_u, new_u = r.random(), r.uniform(-0.2, 0.2), r.random(), r.random()
                small_w, step_w, _off, new_w = r.random(), r.uniform(-0.2, 0.2), r.random(), r.random()
                if small_u < 0.6:
                    assert inp['u'] == reflect(slot['u'] + step_u, 0.05, 1.0)
                else:
                    assert inp['u'] == (1.0 if open_u < 0.05 else 0.05 + 0.95 * new_u)
                if not slot['lift_on']:
                    assert inp['w_lg'] == 0
                elif small_w < 0.6:  # Into the set point's range, from a sweep's point below it too
                    assert inp['w_lg'] == reflect(slot['w_lg'] + step_w * w_max, 0.1 * w_max, w_max)
                    assert between(inp['w_lg'] / w_max, 0.1, 1)
                else:
                    assert inp['w_lg'] == w_max * (0.1 + 0.9 * new_w)  # Never off in a redraw (D-136)
                assert inp['p_s'] == samples._separator_pressure(r, inp['p_s_nom'], lf.w0['dp_fl'])
                du.append(abs(inp['u'] - slot['u']))
                n_lifted += slot['lift_on']
                assert _nsol2_attempt(lf, plan, j, a) == inp
        assert plan == before  # A redraw changes its own slot's attempt only
    du = np.array(du)
    assert len(du) > 1000 and 0.6 < np.mean(du <= 0.2 + 1e-12) < 0.8 and du.max() > 0.5
    assert n_lifted > 100 and ks(places) < 1.95


def test_the_samples_generators(lives, plans, sol2_wells, monkeypatch):
    """SMP-109: a sol-2 slot's first attempt draws from (seed, k, j, 'v2', 'sample'), an nsol-2 well's plan from
    (seed, k, 'v2', 'controls'), and a redraw of either from (seed, k, j, a, 'v2', 'redraw'); a planned attempt
    draws nothing."""
    asked = []

    def recorded(seed, *parts):
        asked.append((seed, *parts))
        return rng_for(seed, *parts)

    monkeypatch.setattr(samples, 'rng_for', recorded)
    lf, (plan, _) = lives[5], plans[5]
    _sol2_attempt(SEED, sol2_wells[5], 12)
    _sol2_attempt(SEED, sol2_wells[5], 12, 3)
    _nsol2_plan(lf, lf.T_life, lf.lift_start)
    _nsol2_attempt(lf, plan, 30)
    _nsol2_attempt(lf, plan, 30, 2)
    assert asked == [
        (SEED, 5, 12, 'v2', 'sample'),
        (SEED, 5, 12, 3, 'v2', 'redraw'),
        (SEED, 5, 'v2', 'controls'),
        (SEED, 5, 30, 2, 'v2', 'redraw'),
    ]


def test_a_rerun_gives_the_same_plan(lives, plans):
    """SMP-109: a plan depends on the seed and the well only."""
    for i in (31, 2):
        lf = lives[i]
        again = _nsol2_plan(lf, lf.T_life, lf.lift_start)
        assert again == plans[i] and again[0] is not plans[i][0]


@pytest.mark.slow
def test_samples_of_many_wells():
    """The same checks on 300 wells, whose sampling errors are half as wide: every sol-2 slot's first attempt, and
    every well's plan."""
    many, wells = drawn_lives(N_MANY_LIVES), drawn_sol2_wells(N_MANY_LIVES)
    attempts = [[_sol2_attempt(SEED, w, j) for j in range(_N_SLOTS)] for w in wells]
    assert sol2_failures(wells, attempts) == []
    stats = []
    for lf in many:
        plan, p_c = _nsol2_plan(lf, lf.T_life, lf.lift_start)
        stats.append(check_plan(lf, plan, p_c, lf.T_life, lf.lift_start))
    assert plan_failures(stats) == []
