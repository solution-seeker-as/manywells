"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Draws the samples within a well from the draft's sample part (plans/sampler-redesign/sampling-draft.md, Step R4)
without solving them: sol-2's per-sample controls and drift, and nsol-2's trends and control history. It prints the
choke, lift-gas and separator marginals, the drift, the nsol-2 timeline (tests, set-point changes, lift starts), the
trends of a few nsol-2 wells at points of their life, and the nsol-2 states by life stage next to R3's sol-2 wells of
the same stage. The wells come from sampler_draft_draws.py (R3's draws); an nsol-2 well is the same well drawn at the
initial, mid-life and late stages, from the same generators.

Three rules of the draft need solves and are left to R6: the producibility filter, the life scan that ends an nsol-2
well's life at the economic floor, and the start of gas lift when natural flow ends. Here every nsol-2 well runs to
the end of its life span, and lift starts at the static check or at its drawn time.

From the repository root:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 600 .venv/bin/python plans/evidence/sampler_draft_samples.py 300)

    --seed N     the dataset seed (default 2026)
    --show K     print the trends of the first K nsol-2 wells (default 4)
"""

import argparse
import math

import numpy as np
import pandas as pd
from sampler_draft_draws import OILS, P_S_MAX, STD_GRAVITY, draw_well, gor_factor, logu, stream

from manywells.pvt.black_oil import BlackOil
from manywells.sampling.wells import rng_for

# --- sol-2 (sampling-draft.md, The samples within a well) ------------------------------------------------------------
N_SAMPLES = 200
U_MIN = 0.05  # Smallest choke position, as SMP-18
P_OPEN = 0.05  # Share of sol-2 samples at a fully open choke, D-37
P_LIFT_OFF = 0.20  # Share of a lifted well's samples without lift gas, D-38 as ruled
PS_JITTER = 0.10  # Separator pressure within +-10% of nominal, D-39
P_S_MIN = 1.5  # bar, R3's floor on the downstream pressure
PR_DRIFT, RATIO_DRIFT = 0.02, 0.05  # The Goals' +-2% and +-5%, D-40
WLR_MAX = 0.98  # Hasan et al.'s largest water cut, R3

# --- nsol-2 ---------------------------------------------------------------------------------------------------------
T_LIFE = (3.0, 30.0)  # years, log-uniform, D-41 as ruled: lives may be shorter than 8 years
Q_END = (0.03, 0.3)  # the decline's rate at the end of the life over the plateau's, log-uniform, D-41 as ruled
N_OP, N_TESTS, N_SWEEP = 160, 8, 5  # operating samples, test periods, samples per test: 160 + 8 x 5 = 200, D-48
P_OPEN_SET = 0.05  # Share of choke set points at fully open, D-47
P_CHANGE = (0.5, 0.95)  # Per-operating-sample probability of a set-point change, log-uniform per well, D-47 as ruled
P_SMALL, STEP = 0.9, 0.1  # A change is a small step of at most +-0.1 with probability 0.9, else a redraw, D-47
REDRAW_SMALL, REDRAW_STEP = 0.6, 0.2  # A redrawn slot's change of the controls, R4's set-point change as D-83 has it


# --- Shared helpers -------------------------------------------------------------------------------------------------
def producing_gor(w, p_r):
    """The producing GOR at reservoir pressure p_r: the oils' from the material balance (D-19, D-44), the
    condensates' from R3's factor (sampling-draft.md, Reservoir state at the life stage)."""
    if w['fluid_type'] in OILS:
        return w['R_i'] * gor_factor(w, p_r)
    if w['fluid_class'] == 'gas_condensate':
        return w['R_i'] * (1 + w['c_k'] * max(0.0, 1 - p_r / w['p_sat']))
    return w['R_i']


def separator_pressure(r, p_nom, dp_fl):
    """U(0.9, 1.1) times the nominal pressure, truncated below at 1.5 bar and at a subsea well's flowline loss."""
    lo, hi, x = max((1 - PS_JITTER) * p_nom, P_S_MIN, dp_fl), (1 + PS_JITTER) * p_nom, r.random()
    return lo + (hi - lo) * x if lo < hi else hi


def black_oil(w):
    return BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])


# --- sol-2 ----------------------------------------------------------------------------------------------------------
def sol2_sample(seed, w, j):
    """Sample j of a sol-2 well: its controls, separator pressure and drift. Every draw is made whatever its branch,
    so that a change of one rule moves no other draw."""
    r = rng_for(seed, w['well'], j, 'v2', 'sample')
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
        wlr = min(WLR_MAX, w['wlr'] * x)
    else:
        wlr = w['wgr'] * x / (w['wgr'] * x + 1e6 / R_go)
    return dict(
        well=w['well'],
        j=j,
        u=u,
        w_lg=w_lg,
        w_lg_frac=w_lg / w['w_lg_max'] if lifted else np.nan,
        p_s=p_s,
        p_r=p_r,
        R_go=R_go,
        wlr=wlr,
        ps_ratio=p_s / w['p_s'],
        pr_ratio=p_r / w['p_r'],
        gor_ratio=R_go / w['R_go'],
        wlr_ratio=wlr / w['wlr'] if w['wlr'] > 0 else np.nan,
        p_r_le_p_s=p_r <= p_s,
    )


# --- nsol-2: the trends ---------------------------------------------------------------------------------------------
def cumulative(t, t_pl, D, b):
    """Cumulative production of a unit plateau rate to t_pl, then Arps's decline (Ahmed 2006, Eqs. 16-3, 16-4)."""
    if t <= t_pl:
        return t
    tau = t - t_pl
    if b < 1e-6:
        return t_pl + (1 - math.exp(-D * tau)) / D
    return t_pl + (1 - (1 + b * D * tau) ** (1 - 1 / b)) / ((1 - b) * D)


def logit(x):
    return math.log(x / (1 - x))


def expit(x):
    return 1 / (1 + math.exp(-x))


class Life:
    """An nsol-2 well: R3's well at its initial, mid-life and late stages (the same generators), and the trend
    parameters drawn from its own generator."""

    def __init__(self, seed, k):
        self.w0, _ = draw_well(seed, k, 'initial')
        self.wm, _ = draw_well(seed, k, 'mid_life')
        self.wL, _ = draw_well(seed, k, 'late')
        w0 = self.w0
        self.oil = w0['fluid_type'] in OILS
        self.gas_depletion = not self.oil and w0['drive'] == 'depletion'
        r = rng_for(seed, k, 'v2', 'evolution')
        self.T = logu(r, *T_LIFE)
        self.t_pl = r.uniform(0.0, 0.25) * self.T
        self.q_end = logu(r, *Q_END)
        self.b = r.uniform(0.3, 0.5) if w0['drive'] == 'pressure_support' else r.uniform(0.0, 0.4)
        tau = self.T - self.t_pl  # Arps's D from the rate at the end of the life: q / q_pl = (1 + b D tau)^(-1/b)
        self.D = (self.q_end**-self.b - 1) / (self.b * tau) if self.b > 1e-6 else -math.log(self.q_end) / tau
        self.gamma = r.uniform(1.0, 2.0)
        support = w0['drive'] == 'pressure_support'
        F_bt, wlr_bt = (
            (r.uniform(0.05, 0.4), r.uniform(0.05, 0.3)) if support else (r.uniform(0.2, 0.7), r.uniform(0.02, 0.1))
        )
        self.F_bt, self.wlr_bt = F_bt, wlr_bt  # breakthrough: the recovery and the oils' water cut it jumps to, D-43
        steps = np.sort(r.uniform(0.25, 0.9, 2)) * self.T  # the two separator step-downs, D-46
        self.t_sep = tuple(float(s) for s in steps)
        self.t_lift_drawn = r.uniform(0.2, 0.9) * self.T
        # The start and end states: R3's initial and late states of the same well. A tail well's stage draws are
        # its tail draws, the same at every stage, so it starts from the core's initial state and ends at its tail
        # state (D-42). The start draws are made for every well.
        r0_lo = 0.85 if w0['drive'] == 'depletion' else 0.97  # R3's initial state, D-18
        r0_core, dry, wlr0_core, wgr0_core = r.uniform(r0_lo, 1.0), r.random(), r.uniform(0.0, 0.1), logu(r, 2, 20)
        key = 'wlr' if self.oil else 'wgr'
        if w0['tail']:
            self.r0, self.water0 = r0_core, ((0.0 if dry < 0.5 else wlr0_core) if self.oil else wgr0_core)
        else:
            self.r0, self.water0 = w0['r_p'], w0[key]
        self.r1, self.water1 = min(self.wL['r_p'], self.r0), self.wL[key]
        self.N_T = cumulative(self.T, self.t_pl, self.D, self.b)
        self.bo = black_oil(w0)
        if self.gas_depletion and self.r1 < self.r0:  # p as a function of F, p/Z linear in F (Ahmed 2006, Eq. 13-8)
            T_r, p_i = w0['T_r'], w0['p_i']
            p_grid = np.linspace(0.5 * self.r1 * p_i, 1.02 * self.r0 * p_i, 200)
            pz = p_grid / np.array([float(self.bo.z_factor(p, T_r)) for p in p_grid])
            pz0, pz1 = np.interp(self.r0 * p_i, p_grid, pz), np.interp(self.r1 * p_i, p_grid, pz)
            F = (pz0 - pz) / (pz0 - pz1)
            order = np.argsort(F)
            self.p_of_F = (F[order], p_grid[order])
        else:
            self.gas_depletion = False
        self.lift_start = 0.0 if w0['lift'] == 'gas_lift' else None
        self.lift_reason = 'from_start' if self.lift_start == 0.0 else None
        if self.lift_start is None and self.wL['lift'] == 'gas_lift':
            t_static = self.static_check_time()
            if t_static is not None and t_static < self.t_lift_drawn:
                self.lift_start, self.lift_reason = t_static, 'needs_lift'
            else:
                self.lift_start, self.lift_reason = self.t_lift_drawn, 'drawn'
        self.lift_w = None if self.lift_start is None else (w0 if self.lift_start == 0.0 else self.wL)

    def F(self, t):
        return cumulative(t, self.t_pl, self.D, self.b) / self.N_T

    def p_r(self, t):
        p_i, F = self.w0['p_i'], self.F(t)
        if self.gas_depletion:
            return float(np.interp(F, *self.p_of_F))
        return p_i * (self.r1 + (self.r0 - self.r1) * (1 - F) ** self.gamma)

    def water(self, t):
        """Oils: the water cut, logit-linear in F after breakthrough. Gases: the WGR, log-linear in F."""
        F = self.F(t)
        if self.oil:
            a, z = self.water0, self.water1
            if z <= a or F < self.F_bt:
                return a
            bt = min(self.wlr_bt, z)
            return max(a, expit(logit(bt) + (logit(z) - logit(bt)) * (F - self.F_bt) / (1 - self.F_bt)))
        a, z = self.water0, self.water1
        if z <= a or F <= self.F_bt:
            return a
        return a * (z / a) ** ((F - self.F_bt) / (1 - self.F_bt))

    def r_s(self, t):
        """The separator's stage factor: 1, then mid-life's, then the late stage's (R3's separator draws)."""
        if t < self.t_sep[0]:
            return 1.0
        r_mid, r_end = self.wm['r_s'], self.wL['r_s']
        return r_mid if t < self.t_sep[1] else min(r_mid, r_end)

    def p_s_nom(self, t):
        w0 = self.w0
        if w0['tail']:
            return max((w0['p_s'] - w0['dp_fl']) * self.r_s(t) + w0['dp_fl'], P_S_MIN)
        return min(max(self.r_s(t) * w0['p_sep'] + w0['dp_fl'], P_S_MIN), P_S_MAX[w0['well_type']])

    def roughness(self, t):
        e0, e1 = self.w0['roughness'], self.wL['roughness']
        return e0 * (max(e1, e0) / e0) ** (t / self.T)

    def h(self, t):
        w0 = self.w0
        if w0['tail']:
            return w0['h']
        age = (w0['age_years'] + t) * 365 * 86400
        r_hole = 0.108 if w0['D'] <= 0.101 else 0.156
        t_d = 1.03e-6 * age / r_hole**2
        T_D = math.log(math.exp(-0.2 * t_d) + (1.5 - 0.3719 * math.exp(-t_d)) * math.sqrt(t_d))
        return w0['U_t'] * w0['k_e'] / (w0['k_e'] + w0['D'] / 2 * w0['U_t'] * T_D)

    def state(self, t):
        p_r = self.p_r(t)
        return dict(p_r=p_r, R_go=producing_gor(self.w0, p_r), water=self.water(t))

    def static_check_time(self):
        """The first time on a grid of 41 at which R3's static check says the oil well needs lift."""
        if not self.oil:
            return None
        w0 = self.w0
        T_wh = w0['T_s'] + 0.6 * (w0['T_r'] - w0['T_s'])
        for t in np.linspace(0, self.T, 41):
            s = self.state(t)
            p_s = self.p_s_nom(t)
            p_m, T_m = 0.5 * (s['p_r'] + p_s), 0.5 * (w0['T_r'] + T_wh)
            vol, mass = stream(self.bo, s['R_go'], s['water'], w0['rho_w'], p_m, T_m)
            if s['p_r'] < p_s + mass / vol * STD_GRAVITY * w0['L'] / 1e5:
                return float(t)
        return None

    def stage(self, t):
        return 'initial' if t < 1.0 else ('late' if t >= 0.5 * self.T else 'mid_life')


def reflect(x, lo, hi):
    if x > hi:
        x = 2 * hi - x
    if x < lo:
        x = 2 * lo - x
    return min(max(x, lo), hi)


def nsol2_samples(seed, life):
    """The 200 slots of an nsol-2 well: 160 operating samples evenly over its life and 8 test periods of 5 samples,
    with the controls' set points, the sweeps, the separator jitter and the drift of the ratios."""
    w0, k, T = life.w0, life.w0['well'], life.T
    r = rng_for(seed, k, 'v2', 'controls')
    p_c = logu(r, *P_CHANGE)
    t_op = np.linspace(0.0, T, N_OP)
    t_test = (np.arange(N_TESTS) + 0.5) * T / N_TESTS
    slots = [(t, 'operating', None) for t in t_op]
    for m, t in enumerate(t_test):
        slots += [(t, 'test', (m, i)) for i in range(N_SWEEP)]
    slots.sort(key=lambda s: (s[0], s[1] == 'test'))
    open_, u_0, w_0 = r.random(), r.uniform(U_MIN, 1.0), r.random()
    u_sp = 1.0 if open_ < P_OPEN_SET else u_0
    w_sp, w_max = None, (life.lift_w or {}).get('w_lg_max')
    changes, rows, sweep, eps_date = 0, [], None, None
    for t, kind, test in slots:
        lift_on = life.lift_start is not None and t >= life.lift_start
        if lift_on and w_sp is None:
            w_sp = w_0 * w_max  # a newly installed lift starts with gas on
        if kind == 'operating' and t > 0:
            for which in ('u', 'w'):
                # Every draw is made whatever its branch, so that a rule's change moves no other draw
                change, small, step, redraw_open, redraw = (
                    r.random(),
                    r.random(),
                    r.uniform(-STEP, STEP),
                    r.random(),
                    r.random(),
                )
                if change >= p_c or (which == 'w' and not lift_on):
                    continue
                changes += which == 'u'
                if which == 'u':
                    if small < P_SMALL:
                        u_sp = reflect(u_sp + step, U_MIN, 1.0)
                    else:
                        u_sp = 1.0 if redraw_open < P_OPEN_SET else U_MIN + (1 - U_MIN) * redraw
                elif small < P_SMALL:
                    w_sp = reflect(w_sp + step * w_max, 0.0, w_max)
                else:
                    w_sp = 0.0 if redraw_open < P_LIFT_OFF else w_max * redraw
        u, w_lg, sweep_kind = u_sp, (w_sp if lift_on else 0.0), None
        if kind == 'test':
            m, i = test
            if i == 0:
                sweep_kind = 'lift' if (lift_on and m % 2 == 1) else 'choke'
                strata = [r.uniform(q / 4, (q + 1) / 4) for q in (3, 2, 1, 0)]
                sweep = (sweep_kind, [1.0] + strata)
                eps_date = (r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT), r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT))
            sweep_kind, x = sweep[0], sweep[1][i]
            if sweep_kind == 'choke':
                u = 1.0 if i == 0 else U_MIN + (1 - U_MIN) * x
            else:
                w_lg = w_max * x
        else:
            eps_date = (r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT), r.uniform(1 - RATIO_DRIFT, 1 + RATIO_DRIFT))
        s = life.state(t)
        p_s = separator_pressure(r, life.p_s_nom(t), w0['dp_fl'])
        water = s['water'] * eps_date[1]
        R_go = s['R_go'] * eps_date[0]
        wlr = min(WLR_MAX, water) if life.oil else water / (water + 1e6 / R_go)
        rows.append(
            dict(
                well=k,
                t=t,
                t_frac=t / T,
                F=life.F(t),
                kind=kind,
                sweep=sweep_kind,
                u=u,
                w_lg=w_lg,
                w_lg_frac=w_lg / w_max if lift_on else np.nan,
                lift_on=lift_on,
                p_s=p_s,
                p_s_nom=life.p_s_nom(t),
                p_r=s['p_r'],
                r_p=s['p_r'] / w0['p_i'],
                R_go=R_go,
                gor_ratio=R_go / w0['R_i'],
                wlr=wlr,
                wgr=None if life.oil else water,
                h=life.h(t),
                roughness=life.roughness(t),
                stage=life.stage(t),
                tail=w0['tail'],
                fluid_type=w0['fluid_type'],
                drive=w0['drive'],
                oil=life.oil,
            )
        )
    return rows, changes, p_c


# --- Report ---------------------------------------------------------------------------------------------------------
def q(s, ps=(5, 50, 95)):
    s = pd.Series(s).dropna()
    return ' / '.join(f'{v:.4g}' for v in np.percentile(s, ps)) + f'  (n={len(s)})' if len(s) else '-'


def report_sol2(wells, s):
    lifted = s['w_lg_frac'].notna()
    print(f'sol-2: {len(wells)} wells x {N_SAMPLES} samples (no solves)\n')
    print('Choke position u')
    print(f'  fully open: {(s.u == 1).mean():.1%}; below 0.1: {(s.u < 0.1).mean():.1%}; P5/P50/P95 {q(s.u)}')
    print(f'  mean {s.u.mean():.3f}, standard deviation {s.u.std():.3f} (U(0.05, 1): 0.525, 0.274)')
    print('Lift gas, lifted wells (w_lg / w_lg,max)')
    print(
        f'  lifted wells {wells.lift.eq("gas_lift").mean():.1%}; samples with lift off {(s.w_lg_frac[lifted] == 0).mean():.1%}'
    )
    print(f'  P5/P50/P95 {q(s.w_lg_frac[lifted])}; w_lg (kg/s) {q(s.w_lg[lifted])}')
    print('Separator pressure p_s / p_s,nom: ' + q(s.ps_ratio))
    print(f'  p_s below 10 bar: {(s.p_s < 10).mean():.1%} of samples')
    print('Drift')
    print(f'  p_r / p_r,nom: {q(s.pr_ratio)}')
    print(f'  R_go / R_go,nom: {q(s.gor_ratio)}')
    sat = s.merge(wells[['well', 'fluid_type', 'saturation']], on='well')
    m = sat.fluid_type.isin(OILS) & (sat.saturation == 'saturated')
    print(f'  R_go / R_go,nom, saturated oils (linked to p_r): {q(sat.gor_ratio[m])}')
    print(f'  water / water,nom (where the nominal is above 0): {q(s.wlr_ratio)}')
    print(f'  samples with p_r <= p_s (no operating point without a solve): {s.p_r_le_p_s.mean():.2%}')


def report_nsol2(lives, n, changes, p_cs, show):
    print(
        f'\nnsol-2: {len(lives)} wells x {N_SAMPLES} slots (no solves; every well runs to the end of its life span)\n'
    )
    T = pd.Series([lf.T for lf in lives])
    print(f'Life span (years): {q(T)}')
    print(f'Slots in test periods: {(n.kind == "test").mean():.1%}')
    dt = pd.Series([lf.T / (N_OP - 1) * 365 for lf in lives])
    print(f'Time between operating samples (days): {q(dt)}')
    print(f'Choke set-point changes per well: {q(changes)}; probability of a change per operating sample {q(p_cs)}')
    print(f'Separator pressure below 10 bar: {(n.p_s < 10).mean():.1%} of slots; p_s,nom at the end / start: ', end='')
    print(q([lf.p_s_nom(lf.T) / lf.p_s_nom(0.0) for lf in lives]))
    print(f'Slots with p_r <= p_s (no operating point without a solve): {(n.p_r <= n.p_s).mean():.2%}')
    op = n[n.kind == 'operating']
    print(f'Choke u, all slots: fully open {(n.u == 1).mean():.1%}, P5/P50/P95 {q(n.u)}')
    print(f'Choke u, operating samples: fully open {(op.u == 1).mean():.1%}, P5/P50/P95 {q(op.u)}')
    on = n[n.lift_on]
    print(f'Lift gas while lift is on, w_lg / w_lg,max: zero {(on.w_lg_frac == 0).mean():.1%}, {q(on.w_lg_frac)}')
    reasons = pd.Series([lf.lift_reason or 'natural' for lf in lives]).value_counts(normalize=True)
    print('Lift: ' + ', '.join(f'{k} {v:.1%}' for k, v in reasons.items()))
    later = pd.Series([lf.lift_start / lf.T for lf in lives if lf.lift_start not in (None, 0.0)])
    print(f'  start of a later lift, as a fraction of the life span: {q(later)}')
    oils = n[n.oil & ~n['tail']]
    print('\nTrends, core wells, by fraction of the life span: P5 / P50 / P95')
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = (oils.kind == 'operating') & (abs(oils.t_frac - f) < 0.004)
        g = n[(n.kind == 'operating') & ~n.oil & ~n['tail'] & (abs(n.t_frac - f) < 0.004)]
        print(
            f'  t/T = {f:.2f}: oils p_r/p_i {q(oils.r_p[m])}; water cut {q(oils.wlr[m])}; R_go/R_si {q(oils.gor_ratio[m])}'
        )
        print(
            f'              gases p_r/p_i {q(g.r_p)}; WGR {q(g.wgr)}; h {q(n.h[(n.kind == "operating") & (abs(n.t_frac - f) < 0.004)])}'
        )
    print('\nThe first nsol-2 wells')
    for lf in lives[:show]:
        w0 = lf.w0
        print(
            f'  well {w0["well"]}: {w0["fluid_class"]}, {w0["drive"]}, {w0["well_type"]}, life {lf.T:.1f} y, '
            f'lift {lf.lift_reason or "natural"}' + (f' at {lf.lift_start:.1f} y' if lf.lift_start else '')
        )
        for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            t = f * lf.T
            s = lf.state(t)
            water = f'water cut {s["water"]:.3f}' if lf.oil else f'WGR {s["water"]:.1f}'
            print(
                f'    t {t:5.1f} y  F {lf.F(t):.2f}  p_r {s["p_r"]:6.1f} bar ({s["p_r"] / w0["p_i"]:.2f} p_i)  '
                f'R_go/R_si {s["R_go"] / w0["R_i"]:.2f}  {water}  p_s,nom {lf.p_s_nom(t):5.1f}  '
                f'h {lf.h(t):5.2f}  eps {lf.roughness(t) * 1e6:5.1f} um'
            )


def report_water_by_years(n):
    """The oils' water cut by years on production, to compare with Norwegian oil fields' medians by years on production
    (an analysis of Sodir's FactMaps, kept outside the repo; fields, not wells)."""
    op = n[(n.kind == 'operating') & n.oil & ~n['tail']]
    print(
        "\nOils' water cut by years on production, core operating samples: P25 / P50 / P75 ([own analysis of the Sodir FactMaps])"
    )
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30)):
        for drive in ('depletion', 'pressure_support'):
            m = (op.t >= lo) & (op.t < hi) & (op.drive == drive)
            print(f'  {lo}-{hi} years, {drive}: {q(op.wlr[m], (25, 50, 75))}')


def report_consistency(wells, n):
    """nsol-2's states by per-sample stage against R3's sol-2 wells drawn at that stage (core wells)."""
    print('\nConsistency: nsol-2 operating samples by stage against R3 sol-2 wells of that stage (core), P5/P50/P95')
    core = wells[~wells['tail']]
    op = n[(n.kind == 'operating') & ~n['tail']]
    for stage in ('initial', 'mid_life', 'late'):
        for drive in ('depletion', 'pressure_support'):
            a = op[(op.stage == stage) & (op.drive == drive)]
            b = core[(core.stage == stage) & (core.drive == drive)]
            print(f'  {stage}, {drive}: p_r/p_i nsol-2 {q(a.r_p)} | sol-2 {q(b.r_p)}')
            ao, bo = a[a.oil], b[b.fluid_type.isin(OILS)]
            print(f'      oils water cut nsol-2 {q(ao.wlr)} | sol-2 {q(bo.wlr)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--show', type=int, default=4)
    args = ap.parse_args()
    wells = [draw_well(args.seed, k)[0] for k in range(args.n)]
    s = pd.DataFrame([sol2_sample(args.seed, w, j) for w in wells for j in range(N_SAMPLES)])
    wells = pd.DataFrame(wells)
    report_sol2(wells, s)
    lives, rows, changes, p_cs = [], [], [], []
    for k in range(args.n):
        lf = Life(args.seed, k)
        rs, c, p_c = nsol2_samples(args.seed, lf)
        lives.append(lf)
        rows += rs
        changes.append(c)
        p_cs.append(p_c)
    n = pd.DataFrame(rows)
    assert (n.groupby('well').size() == N_SAMPLES).all()
    report_nsol2(lives, n, changes, p_cs, args.show)
    report_water_by_years(n)
    report_consistency(wells, n)


if __name__ == '__main__':
    main()
