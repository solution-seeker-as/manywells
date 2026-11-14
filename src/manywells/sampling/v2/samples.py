"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The v2 sample draws (specs/sampling.md, SMP-93 to SMP-100 and the redraws of SMP-105): the inputs of an attempt of a
slot. A sol-2 slot draws its controls, its separator pressure and the reservoir's drift around the well's nominal
state. An nsol-2 well draws a plan of its 200 slots over its life: the dates, the control history, the test periods
and the drift; a redraw changes the controls of its own slot. Nothing here solves a well.

An attempt's inputs are the dict that mapping.WellBuilder.case takes: `p_r` and `p_s` (bar), `u`, `w_lg` (kg/s),
`lift_on`, `gor` (Sm3/Sm3), `wlr`, `h` (W/(m2 K)), `roughness` (m) and the inflow's coefficients (inflow._COEF_KEYS),
with `wgr` (Sm3 per 1e6 Sm3; NaN for an oil) beside them.
"""

import math

import numpy as np

from manywells.sampling.v2._distributions import logu
from manywells.sampling.v2.draws import _P_S_MIN
from manywells.sampling.v2.inflow import _COEF_KEYS
from manywells.sampling.v2.life import _WLR_MAX, _lift_on
from manywells.sampling.v2.reservoir import _OILS, _producing_gor
from manywells.sampling.wells import rng_for

_N_SLOTS = 200  # A well's slots  # spec: SMP-105
_N_OPERATING, _N_TESTS, _N_SWEEP = 160, 8, 5  # Operating samples, test periods, samples of a test  # spec: SMP-97
_U_MIN = 0.05  # The smallest choke position  # spec: SMP-93
_P_OPEN = 0.05  # Share of fully open chokes: of the sol-2 slots, and of the set point's draws  # spec: SMP-93, SMP-98
_P_LIFT_OFF = 0.20  # Share of a lifted well's draws without lift gas  # spec: SMP-94, SMP-98
_LIFT_GAS_ALWAYS_ON = 'lift_gas_always_on'  # The flag of a sol-2 well that cannot flow without lift gas  # spec: SMP-94
_P_S_JITTER = 0.10  # The separator pressure about its nominal  # spec: SMP-95
_P_R_DRIFT, _RATIO_DRIFT = 0.02, 0.05  # The drift of p_r, and of the GOR and the water  # spec: SMP-96, SMP-100
_P_CHANGE = (0.2, 0.95)  # A set point's probability of a change at an operating sample  # spec: SMP-98
_P_SMALL, _STEP = 0.9, 0.1  # A change is a step of at most 0.1 with this probability, else a new draw  # spec: SMP-98
_LIFT_FLOOR = 0.1  # The lift-gas set point's floor, a share of the capacity  # spec: SMP-98
_REDRAW_SMALL, _REDRAW_STEP = 0.6, 0.2  # A redraw's change of a control  # spec: SMP-105


def _generator(seed, k, j, a):  # spec: SMP-109
    """The generator of attempt a of slot j of well k: the sample's own, or a redraw's."""
    return rng_for(seed, k, j, 'v2', 'sample') if a == 0 else rng_for(seed, k, j, a, 'v2', 'redraw')


def _separator_pressure(r, p_nom, dp_fl) -> float:  # spec: SMP-95
    """Within 10% of the nominal pressure, at least 1.5 bar and a subsea well's flowline loss."""
    lo, hi, x = max((1 - _P_S_JITTER) * p_nom, _P_S_MIN, dp_fl), (1 + _P_S_JITTER) * p_nom, r.random()
    return lo + (hi - lo) * x if lo < hi else hi


def _reflect(x, lo, hi) -> float:
    if x > hi:
        x = 2 * hi - x
    if x < lo:
        x = 2 * lo - x
    return min(max(x, lo), hi)


# --- sol-2 -----------------------------------------------------------------------------------------------------------
def _sol2_attempt(seed: int, w: dict, j: int, a: int = 0) -> dict:  # spec: SMP-105
    """
    The inputs of attempt a of slot j of a sol-2 well: its controls, its separator pressure and the reservoir's
    drift, around the well's nominal state. Attempt 0 draws from the slot's generator and a redraw draws the whole
    draw again from its own, so no slot depends on another.

    :param seed: The dataset's seed
    :param w: The well at its date (life.sol2_well), with the flag _LIFT_GAS_ALWAYS_ON where its best rate found that
        it cannot flow without lift gas (generate.generate_sol2_well)
    :param j: The slot, 0 to N_SLOTS - 1
    :param a: The attempt, 0 for the slot's first
    :return: The attempt's inputs (the module's docstring); `h`, `roughness` and the coefficients are the well's
    """
    r = _generator(seed, w['well'], j, a)
    lifted = w['lift'] == 'gas_lift'
    # Every number is drawn whatever its branch, so that a change of one rule moves no other draw
    open_, u_c = r.random(), r.uniform(_U_MIN, 1.0)
    u = 1.0 if open_ < _P_OPEN else u_c  # spec: SMP-93
    off, v = r.random(), r.uniform(0.0, 1.0)
    off = off < _P_LIFT_OFF and _LIFT_GAS_ALWAYS_ON not in w['flags'].split(';')
    w_lg = (0.0 if off else v * w['w_lg_max']) if lifted else 0.0  # spec: SMP-94
    p_s = _separator_pressure(r, w['p_s'], w['dp_fl'])
    # spec: SMP-96
    p_r = w['p_r'] * r.uniform(1 - _P_R_DRIFT, 1 + _P_R_DRIFT)
    R_go = _producing_gor(w, p_r) * r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT)
    x = r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT)
    if w['fluid_type'] in _OILS:
        wlr, wgr = min(_WLR_MAX, w['wlr'] * x), math.nan
    else:
        wgr = w['wgr'] * x
        wlr = wgr / (wgr + 1e6 / R_go)
    inp = dict(h=w['h'], roughness=w['roughness'], **{key: w[key] for key in _COEF_KEYS})
    inp.update(lift_on=lifted, u=u, w_lg=w_lg, p_s=p_s, p_r=p_r, gor=R_go, wlr=wlr, wgr=wgr)
    return inp


# --- nsol-2 ----------------------------------------------------------------------------------------------------------
def _nsol2_plan(life, T_end: float, lift_start: float | None) -> tuple:  # spec: SMP-97
    """
    The plan of an nsol-2 well's 200 slots over [0, T_end], drawn in slot order from the well's `controls`
    generator before any slot is solved: the dates, the control history, the test periods' sweeps, the separator
    pressures and the dates' drift.

    :param life: The well's life (life.Life)
    :param T_end: The end of the life (years) that the life scan gives (SMP-101)
    :param lift_start: The date (years) at which gas lift starts, or None for a well that stays natural: the
        life's `lift_start`, or the earlier start that the life scan gives (SMP-90, rule 3)
    :return: (plan, p_c): a dict per slot, in the order of the dates, and the well's probability of a set-point
        change. A slot holds `t` (years); `kind`, `operating` or `test`; `test`, the test period from 1 to 8 (None
        for an operating sample), and `sweep`, `choke` or `lift` (None); `lift_on`; the planned controls `u` and
        `w_lg` (kg/s); `p_s` (bar); and `eps`, the date's drift of the GOR and of the water
    """
    if lift_start is not None and life.design is None:
        raise ValueError('a well without a lift design stays natural')
    w0 = life.w0
    r = rng_for(life.seed, life.k, 'v2', 'controls')  # spec: SMP-109
    p_c = logu(r, *_P_CHANGE)  # spec: SMP-98
    # spec: SMP-97. The slots, in the order of their dates
    t_op = np.linspace(0.0, T_end, _N_OPERATING)
    t_test = (np.arange(_N_TESTS) + 0.5) * T_end / _N_TESTS
    slots = [(float(t), 'operating', None) for t in t_op]
    for m, t in enumerate(t_test):
        slots += [(float(t), 'test', (m, i)) for i in range(_N_SWEEP)]
    slots.sort(key=lambda s: (s[0], s[1] == 'test'))
    # spec: SMP-98. The set points at the start; the lift gas's holds from the start of lift
    open_, u_0, w_0 = r.random(), r.uniform(_U_MIN, 1.0), r.random()
    u_sp = 1.0 if open_ < _P_OPEN else u_0
    w_max = life.design['w_lg_max'] if life.design is not None else math.nan
    w_sp, plan, sweep, eps = None, [], None, None
    for t, kind, test in slots:
        lifted = _lift_on(t, lift_start)
        if lifted and w_sp is None:
            w_sp = (_LIFT_FLOOR + (1 - _LIFT_FLOOR) * w_0) * w_max
        if kind == 'operating' and t > 0:  # spec: SMP-98. Each set point may change
            for control in ('u', 'w'):
                # Every number is drawn whatever its branch, so that a change of one rule moves no other draw
                change, small = r.random(), r.random()
                step, new_open, new = r.uniform(-_STEP, _STEP), r.random(), r.random()
                if change >= p_c or (control == 'w' and not lifted):
                    continue
                if control == 'u':
                    if small < _P_SMALL:
                        u_sp = _reflect(u_sp + step, _U_MIN, 1.0)
                    else:
                        u_sp = 1.0 if new_open < _P_OPEN else _U_MIN + (1 - _U_MIN) * new
                elif small < _P_SMALL:
                    w_sp = _reflect(w_sp + step * w_max, _LIFT_FLOOR * w_max, w_max)
                else:
                    w_sp = 0.0 if new_open < _P_LIFT_OFF else w_max * (_LIFT_FLOOR + (1 - _LIFT_FLOOR) * new)
        u, w_lg, sweep_kind, m_test = u_sp, (w_sp if lifted else 0.0), None, None
        if kind == 'test':  # spec: SMP-99
            m, i = test
            m_test = m + 1
            if i == 0:  # The test's sweep, from the top down, and its date's drift
                sweep_kind = 'lift' if (lifted and m % 2 == 1) else 'choke'
                strata = [r.uniform(q / 4, (q + 1) / 4) for q in (3, 2, 1, 0)]
                sweep = (sweep_kind, [1.0] + strata)
                eps = (r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT), r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT))
            sweep_kind, x = sweep[0], sweep[1][i]
            if sweep_kind == 'choke':
                u = 1.0 if i == 0 else _U_MIN + (1 - _U_MIN) * x
            else:
                w_lg = w_max * x
        else:  # spec: SMP-100
            eps = (r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT), r.uniform(1 - _RATIO_DRIFT, 1 + _RATIO_DRIFT))
        p_s = _separator_pressure(r, life.p_s_nom(t), w0['dp_fl'])
        slot = dict(t=t, kind=kind, test=m_test, sweep=sweep_kind, lift_on=lifted, u=u, w_lg=w_lg, p_s=p_s, eps=eps)
        plan.append(slot)
    return plan, p_c


def _nsol2_redraw(r, slot, w_max, p_s_nom, dp_fl) -> tuple:  # spec: SMP-105
    """A redraw's controls and separator pressure: each control changed from the slot's planned value, the lift gas
    only where lift is installed. Every number is drawn whatever its branch."""
    small_u, step_u, open_u, new_u = r.random(), r.uniform(-_REDRAW_STEP, _REDRAW_STEP), r.random(), r.random()
    small_w, step_w, _off, new_w = r.random(), r.uniform(-_REDRAW_STEP, _REDRAW_STEP), r.random(), r.random()
    if small_u < _REDRAW_SMALL:
        u = _reflect(slot['u'] + step_u, _U_MIN, 1.0)
    else:
        u = 1.0 if open_u < _P_OPEN else _U_MIN + (1 - _U_MIN) * new_u
    w_lg = slot['w_lg']
    if slot['lift_on']:  # A redraw keeps the lift gas on (D-136); _off is drawn, unused, for the count (SMP-51)
        if small_w < _REDRAW_SMALL:
            w_lg = _reflect(slot['w_lg'] + step_w * w_max, _LIFT_FLOOR * w_max, w_max)
        else:
            w_lg = w_max * (_LIFT_FLOOR + (1 - _LIFT_FLOOR) * new_w)
    return u, w_lg, _separator_pressure(r, p_s_nom, dp_fl)


def _nsol2_attempt(life, plan: list, j: int, a: int = 0) -> dict:  # spec: SMP-105
    """
    The inputs of attempt a of slot j of an nsol-2 well: the life's state at the slot's date with the date's drift,
    and the slot's planned controls and separator pressure, or for a redraw the controls changed from the planned
    ones and the separator pressure drawn again, from the redraw's own generator. A redraw holds for its slot only.

    :param life: The well's life (life.Life)
    :param plan: The well's plan (_nsol2_plan)
    :param j: The slot, 0 to N_SLOTS - 1
    :param a: The attempt, 0 for the planned one
    :return: The attempt's inputs (the module's docstring), with the state's `t`, `p_s_nom` and `life_stage`
        (life.Life.state) and the slot's `test` and `sweep`
    """
    slot = plan[j]
    st = life.state(slot['t'], slot['eps'])
    if a == 0:
        u, w_lg, p_s = slot['u'], slot['w_lg'], slot['p_s']
    else:
        r = _generator(life.seed, life.k, j, a)
        w_max = life.design['w_lg_max'] if slot['lift_on'] else 0.0
        u, w_lg, p_s = _nsol2_redraw(r, slot, w_max, st['p_s_nom'], life.w0['dp_fl'])
    lifted = slot['lift_on']
    return dict(st, u=u, w_lg=w_lg if lifted else 0.0, p_s=p_s, lift_on=lifted, test=slot['test'], sweep=slot['sweep'])
