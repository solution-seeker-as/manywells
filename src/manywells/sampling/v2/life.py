"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A v2 well's life (specs/sampling.md, SMP-83 to SMP-92): the trends of its reservoir, its separator and its ageing
properties from the well's start to its reservoir exhausted, the start of its gas lift, and a sol-2 well as its life
at one date. Both datasets use the one life: an nsol-2 well lives it, and a sol-2 well is the same life at a date in
its drawn stage. Nothing here solves a well: where a rule follows a solve (the lift that the life scan or the
producibility filter starts, SMP-101 and SMP-103), the generation passes its result in.
"""

import math

import numpy as np

from manywells.sampling.v2 import draws
from manywells.sampling.v2._distributions import logu
from manywells.sampling.v2.draws import P_S_MAX, P_S_MIN, black_oil, draw_well
from manywells.sampling.v2.inflow import COEF_KEYS, inflow_at
from manywells.sampling.v2.reservoir import OILS, producing_gor
from manywells.sampling.wells import rng_for

WLR_MAX = 0.98  # An oil's largest water-liquid ratio  # spec: SMP-83, SMP-96, SMP-100
_T_LIFE = (3.0, 30.0)  # years  # spec: SMP-84
_Q_END = (0.03, 0.3)  # The decline's rate at the end of the life over the plateau's  # spec: SMP-84
_R_END = 0.15  # p_r / p_i of an exhausted reservoir under depletion  # spec: SMP-83
_R_END_TOP = {True: 0.3, False: 0.2}  # The top of a core well's end under depletion: oils, gas classes  # spec: SMP-83
_WGR_END = 10000.0  # Sm3 per 1e6 Sm3: a gas well with pressure support waters out  # spec: SMP-83
_N_STATIC = 41  # Dates of the static check along the life  # spec: SMP-90
_LIFT_KEYS = ('z_inj', 'injection_md', 'z_inj_frac', 'p_inj', 'dp_valve', 'packer_offset', 'T_lg', 'phi_lg')
_LIFT_KEYS += ('T_lg_below_ambient', 'glr_cap', 'w_lg_max')  # A lift design's keys (draws.draw_well)


def _cumulative(t, t_pl, D, b) -> float:  # spec: SMP-84
    """The cumulative production at t of a unit rate to the plateau's end t_pl and then Arps's decline with the
    initial decline rate D and the exponent b (Ahmed 2006, Eqs. 16-3 and 16-4)."""
    if t <= t_pl:
        return t
    tau = t - t_pl
    if b < 1e-6:
        return t_pl + (1 - math.exp(-D * tau)) / D
    return t_pl + (1 - (1 + b * D * tau) ** (1 - 1 / b)) / ((1 - b) * D)


def _logit(x):
    return math.log(x / (1 - x))


def _expit(x):
    return 1 / (1 + math.exp(-x))


def lift_on(t, lift_start) -> bool:  # spec: SMP-90
    """Whether gas lift is installed at the date t (years) of a life whose lift starts at lift_start (years; None for
    a well that stays natural)."""
    return lift_start is not None and t >= lift_start - 1e-12


class Life:
    """
    The life of well k of a dataset with seed `seed` (SMP-83): its draws at the three life stages, the trends'
    parameters from the well's `evolution` generator, and the start of its gas lift as the draws give it. Dates are
    years from the well's first production, from 0 to T_life.

    :param seed: The dataset's seed
    :param k: The well's draw index
    :param fluid: A table fluid's values, as draws.draw_well takes them, or None
    :param pvt: A table fluid's PVTTableSet, with which the inflow's coefficients are made (SMP-72, SMP-117), or
        None for the well's black oil

    Attributes:

    - `seed`, `k`; `w0`, the well's initial-stage draws (draws.draw_well), which hold everything that is fixed for
      the life (SMP-88), and `stations`, its survey (md, tvd); `oil`, whether the fluid is an oil; `pvt`, the PVT
      model of the inflow's coefficients.
    - `T_life`: the life span (years).
    - `design`: the draws that hold the lift design (draws.draw_well's lift keys): the initial-stage draw's of a
      well lifted from the start, else the late-stage draw's, made as if lifted where that draw leaves an oil well
      natural; None for a gas well that the late-stage draw leaves natural, which no rule lifts.
    - `lift_start` and `lift_start_rule`: the date at which lift starts by the draws, and its rule: 0 and
      `from_start`, or the earliest of the static check's first failure (`static_check`) and the drawn date
      (`drawn_time`); None for a well that these leave natural. The life scan may start lift earlier in nsol-2
      (SMP-101), and the producibility filter may lift a sol-2 well (SMP-103): both are the generation's, which
      passes the start to lift_on and samples.nsol2_plan, or asks sol2_well for the lifted well.
    """

    def __init__(self, seed: int, k: int, fluid: dict | None = None, pvt=None):
        self.seed, self.k = seed, k
        # spec: SMP-83. The same well at its three stages (SMP-51)
        self.w0, self.stations = draw_well(seed, k, 'initial', fluid=fluid)
        self._wm = draw_well(seed, k, 'mid_life', fluid=fluid)[0]
        self._wL = draw_well(seed, k, 'late', fluid=fluid)[0]
        w0, wL = self.w0, self._wL
        self.oil = w0['fluid_type'] in OILS
        self._tail = w0['coverage_tail'] == 'tail'
        self._bo = black_oil(w0)  # The black oil of the draws, a table fluid's too (v2: scope and conventions)
        self.pvt = self._bo if pvt is None else pvt
        self._coef = {}
        support = w0['drive'] == 'pressure_support'
        r = rng_for(seed, k, 'v2', 'evolution')  # spec: SMP-109
        # spec: SMP-84. The life span and the depletion clock
        self.T_life = logu(r, *_T_LIFE)
        self._t_pl = r.uniform(0.0, 0.25) * self.T_life
        q_end = logu(r, *_Q_END)
        self._b = r.uniform(0.3, 0.5) if support else r.uniform(0.0, 0.4)
        tau = self.T_life - self._t_pl  # Arps's D from the rate at the end of the life: q = (1 + b D tau)^(-1/b)
        self._D = (q_end**-self._b - 1) / (self._b * tau) if self._b > 1e-6 else -math.log(q_end) / tau
        self._N_T = _cumulative(self.T_life, self._t_pl, self._D, self._b)
        self._gamma = r.uniform(1.0, 2.0)  # spec: SMP-85
        # spec: SMP-86. Breakthrough: its recovery, and the water-liquid ratio that an oil jumps to
        if support:
            self._F_bt, self._wlr_bt = r.uniform(0.05, 0.4), r.uniform(0.05, 0.3)
        else:
            self._F_bt, self._wlr_bt = r.uniform(0.2, 0.7), r.uniform(0.02, 0.1)
        self._t_sep = tuple(float(t) for t in np.sort(r.uniform(0.25, 0.9, 2)) * self.T_life)  # spec: SMP-89
        t_lift = r.uniform(0.2, 0.9) * self.T_life  # spec: SMP-90
        # spec: SMP-83. The start: the initial-stage draw's state, and for a tail well a core initial state. Its
        # numbers are drawn for every well
        r0_core, dry = r.uniform(0.97 if support else 0.85, 1.0), r.random()
        wlr0_core, wgr0_core = r.uniform(0.0, 0.1), logu(r, 2, 20)
        water = 'wlr' if self.oil else 'wgr'
        if self._tail:
            self._r0 = r0_core
            self._water0 = (0.0 if dry < 0.5 else wlr0_core) if self.oil else wgr0_core
        else:
            self._r0, self._water0 = w0['r_p'], w0[water]
        # spec: SMP-83. The end: a core well's reservoir exhausted, a tail well's tail state
        u_end = r.random()
        self._r1, self._water1 = min(wL['r_p'], self._r0), wL[water]
        if not self._tail and not support:
            self._r1 = min(_R_END + (_R_END_TOP[self.oil] - _R_END) * u_end, self._r0)
        elif not self._tail:
            self._water1 = WLR_MAX if self.oil else _WGR_END
        # spec: SMP-85. A gas well under depletion: p_r against F from p/Z linear in F, on a grid of pressures
        self._p_of_F = None
        if not self.oil and not support and self._r1 < self._r0:
            T_r, p_i = w0['T_r'], w0['p_i']
            p_grid = np.linspace(0.5 * self._r1 * p_i, 1.02 * self._r0 * p_i, 200)
            pz = p_grid / np.array([float(self._bo.z_factor(p, T_r)) for p in p_grid])
            pz0, pz1 = np.interp(self._r0 * p_i, p_grid, pz), np.interp(self._r1 * p_i, p_grid, pz)
            F = (pz0 - pz) / (pz0 - pz1)
            order = np.argsort(F)
            self._p_of_F = (F[order], p_grid[order])
        # spec: SMP-90. The lift's design, and its start by the draws
        if w0['lift'] == 'gas_lift':
            self.design, self.lift_start, self.lift_start_rule = w0, 0.0, 'from_start'
        else:
            late_lifted = wL['lift'] == 'gas_lift'
            if late_lifted:
                self.design = wL
            elif self.oil:
                self.design = draw_well(seed, k, 'late', force_lift=True, fluid=fluid)[0]
            else:
                self.design = None
            starts = []
            if self.oil:
                t_static = self._static_check_time()
                if t_static is not None:
                    starts.append((t_static, 'static_check'))
            if late_lifted:
                starts.append((t_lift, 'drawn_time'))
            self.lift_start, self.lift_start_rule = min(starts) if starts else (None, None)

    def parameters(self) -> dict:
        """
        The life's parameters, as the wells' table of nsol-2 holds them: `T_life`, `t_plateau` (years), `decline_D`
        (1/year) and `decline_b` (SMP-84); `gamma_p`, `r_p_start` and `r_p_end`, p_r/p_i at the start and the end
        (SMP-83, SMP-85); `F_bt`, `wlr_bt`, `water_start` and `water_end`, the water-liquid ratio of an oil and the
        water-gas ratio of a gas class (SMP-86); `t_sep_1`, `t_sep_2` (years), `r_s_mid` and `r_s_late`, the
        separator's steps (SMP-89); `roughness_start` and `roughness_end` (m; SMP-88).
        """
        e0 = self.w0['roughness']
        return dict(
            T_life=self.T_life,
            t_plateau=self._t_pl,
            decline_D=self._D,
            decline_b=self._b,
            gamma_p=self._gamma,
            r_p_start=self._r0,
            r_p_end=self._r1,
            F_bt=self._F_bt,
            wlr_bt=self._wlr_bt,
            water_start=self._water0,
            water_end=self._water1,
            t_sep_1=self._t_sep[0],
            t_sep_2=self._t_sep[1],
            r_s_mid=self._wm['r_s'],
            r_s_late=self._wL['r_s'],
            roughness_start=e0,
            roughness_end=max(self._wL['roughness'], e0),
        )

    def F(self, t) -> float:  # spec: SMP-84
        """The depletion clock: the cumulative production at t over the life span's, from 0 to 1."""
        return _cumulative(t, self._t_pl, self._D, self._b) / self._N_T

    def p_r(self, t) -> float:  # spec: SMP-85
        """The reservoir pressure (bar) at t."""
        F = self.F(t)
        if self._p_of_F is not None:
            return float(np.interp(F, *self._p_of_F))
        return self.w0['p_i'] * (self._r1 + (self._r0 - self._r1) * (1 - F) ** self._gamma)

    def water(self, t) -> float:  # spec: SMP-86
        """The water at t: an oil's water-liquid ratio, its logit linear in F after breakthrough, or a gas class's
        water-gas ratio (Sm3 per 1e6 Sm3), log-linear in F after it."""
        F, a, z = self.F(t), self._water0, self._water1
        if self.oil:
            if z <= a or F < self._F_bt:
                return a
            bt = min(self._wlr_bt, z)
            return max(a, _expit(_logit(bt) + (_logit(z) - _logit(bt)) * (F - self._F_bt) / (1 - self._F_bt)))
        if z <= a or F <= self._F_bt:
            return a
        return a * (z / a) ** ((F - self._F_bt) / (1 - self._F_bt))

    def r_s(self, t) -> float:  # spec: SMP-89
        """The separator's factor at t: 1, then the mid-life draw's, then the smaller of it and the late draw's."""
        if t < self._t_sep[0]:
            return 1.0
        r_mid, r_late = self._wm['r_s'], self._wL['r_s']
        return r_mid if t < self._t_sep[1] else min(r_mid, r_late)

    def p_s_nom(self, t) -> float:  # spec: SMP-89
        """The nominal pressure downstream of the choke (bar) at t: SMP-66's with the separator's factor at t."""
        w0 = self.w0
        if self._tail:
            return max((w0['p_s'] - w0['dp_fl']) * self.r_s(t) + w0['dp_fl'], P_S_MIN)
        return min(max(self.r_s(t) * w0['p_sep'] + w0['dp_fl'], P_S_MIN), P_S_MAX[w0['well_type']])

    def roughness(self, t) -> float:  # spec: SMP-88
        """The pipe's roughness (m) at t: log-linear in t from the initial draw's towards the late draw's."""
        e0, e1 = self.w0['roughness'], self._wL['roughness']
        return e0 * (max(e1, e0) / e0) ** (t / self.T_life)

    def h(self, t) -> float:  # spec: SMP-88
        """The heat-transfer coefficient (W/(m2 K)) at t: SMP-81's at the well's age; a tail well keeps its own."""
        w0 = self.w0
        if self._tail:
            return w0['h']
        return draws._heat_transfer(w0['D'], w0['U_t'], w0['k_e'], (w0['age_years'] + t) * 365 * 86400)

    def stage(self, t) -> str:  # spec: SMP-91
        """The life stage of the date t: `initial` in the first year, `late` from half the life span."""
        return 'initial' if t < 1.0 else ('late' if t >= 0.5 * self.T_life else 'mid_life')

    def _trends(self, t) -> tuple:  # spec: SMP-85, SMP-86, SMP-87
        """The trends' reservoir state at t: p_r (bar), the producing GOR at it (Sm3/Sm3) and the water."""
        p_r = self.p_r(t)
        return p_r, producing_gor(self.w0, p_r), self.water(t)

    def _static_check_time(self):  # spec: SMP-90
        """The first of the 41 dates i T_life / 40 at which an oil well's static check (SMP-78) fails at the trends'
        state and the nominal downstream pressure, or None."""
        for t in np.linspace(0, self.T_life, _N_STATIC):
            p_r, R_go, wlr = self._trends(t)
            if draws._needs_lift(self.w0, self._bo, p_r, self.p_s_nom(t), R_go, wlr):
                return float(t)
        return None

    def state(self, t, eps=(1.0, 1.0)) -> dict:  # spec: SMP-88, SMP-100
        """
        The well's state at the date t: what an attempt's inputs take from the life (mapping.WellBuilder.case), the
        controls and the separator pressure apart.

        :param t: Date (years)
        :param eps: The date's drift (SMP-100): the factors on the trend's GOR and on its water
        :return: A dict of `t`; `p_r` (bar), without drift; `gor` (Sm3/Sm3) and `wlr` with the drift, an oil's WLR
            capped at WLR_MAX, and `wgr` (Sm3 per 1e6 Sm3; NaN for an oil); `h` (W/(m2 K)), `roughness` (m),
            `p_s_nom` (bar) and `life_stage`; and the inflow's coefficients (inflow.COEF_KEYS) at the trends' state
            without the drift, made once per date
        """
        t = min(t, self.T_life)  # A date that rounding has put past the life's end
        p_r, R_trend, water_trend = self._trends(t)
        R_go, water = R_trend * eps[0], water_trend * eps[1]
        if self.oil:
            wlr, wgr = min(WLR_MAX, water), math.nan
            wlr_trend = min(WLR_MAX, water_trend)
        else:
            wlr, wgr = water / (water + 1e6 / R_go), water
            wlr_trend = water_trend / (water_trend + 1e6 / R_trend)
        if t not in self._coef:
            self._coef[t] = inflow_at(self.w0, self.pvt, p_r, R_trend, wlr_trend)
        return dict(
            t=t,
            p_r=p_r,
            gor=R_go,
            wlr=wlr,
            wgr=wgr,
            h=self.h(t),
            roughness=self.roughness(t),
            p_s_nom=self.p_s_nom(t),
            life_stage=self.stage(t),
            **self._coef[t],
        )


def _sol2_date(life) -> tuple:  # spec: SMP-92
    """A sol-2 well's drawn life stage, and its date (years): uniform within the stage, or the end of a tail well's
    life."""
    labels = {}
    draws._labels(rng_for(life.seed, life.k, 'v2', 'labels'), labels, None)  # The stage as the labels block draws it
    stage = labels['life_stage']
    u = rng_for(life.seed, life.k, 'v2', 'date').random()  # spec: SMP-109
    T = life.T_life
    if labels['coverage_tail'] == 'tail':
        return stage, T
    lo, hi = {'initial': (0.0, 1.0), 'mid_life': (1.0, 0.5 * T), 'late': (0.5 * T, 1.0 * T)}[stage]
    return stage, lo + (hi - lo) * u


def sol2_well(life, no_natural_flow: bool = False) -> dict:  # spec: SMP-92
    """
    A sol-2 well: its life at one date in its drawn stage, as the well's draws at its nominal state.

    :param life: The well's life
    :param no_natural_flow: True for a natural oil well that the producibility filter lifts (SMP-103): the well then
        takes its life's lift design, with `lift_start_rule` `no_natural_flow`
    :return: The initial-stage draws (draws.draw_well), with these keys at the date: `life_stage`, the drawn stage,
        and `date` (years); the nominal state `p_r`, `r_p`, `R_go`, `wlr`, [`wgr` and `ogr`], `saturation` and
        `below_hydrostatic`; `p_s`, the nominal pressure downstream of the choke, and `r_s`; `h` and `roughness`;
        the inflow's coefficients (inflow.COEF_KEYS), NaN where the inflow model does not use one; `lift`, with the
        design's lift keys where the well is lifted at the date, and `lift_start_rule` (None for a natural well).
        The other keys are the initial-stage draw's, the values derived at its state too (`J`, `q_l_ref`,
        `age_years`, `needs_lift` and the like).
    """
    w0 = life.w0
    stage, t = _sol2_date(life)
    st = life.state(t)
    p_r = st['p_r']
    w = dict(w0, life_stage=stage, date=t, p_r=p_r, r_p=p_r / w0['p_i'], R_go=st['gor'], p_s=st['p_s_nom'])
    w.update(r_s=life.r_s(t), h=st['h'], roughness=st['roughness'], below_hydrostatic=p_r < w0['p_h'])
    w['wlr'] = st['wlr']
    if not life.oil:
        w.update(wgr=st['wgr'], ogr=1e6 / st['gor'])
    if w0['fluid_class'] not in ('wet_gas', 'dry_gas'):
        w['saturation'] = 'undersaturated' if p_r > w0['p_sat'] else 'saturated'
    w.update({key: st[key] for key in COEF_KEYS})
    lifted = lift_on(t, life.lift_start)  # spec: SMP-90
    rule = life.lift_start_rule if lifted else None
    if no_natural_flow:  # The producibility filter's lift (SMP-103)
        if lifted or not life.oil:
            raise ValueError('only a natural oil well is lifted for want of natural flow')
        lifted, rule = True, 'no_natural_flow'
    w.update(lift='gas_lift' if lifted else 'natural', lift_start_rule=rule)
    if lifted:
        w.update({key: life.design[key] for key in _LIFT_KEYS})
    return w
