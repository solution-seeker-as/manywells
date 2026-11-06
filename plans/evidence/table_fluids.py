"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A proof of the table fluids of the sampler redesign (Step R5; plans/sampler-redesign/fluids-draft.md): synthetic
compositions drawn per fluid class and matched to a well's drawn descriptors, published example fluids, and their PVT
tables from NeqSim's Peng-Robinson equation of state with the Peneloux volume correction, as PVT-TAB-6 turns a phase
split into a table (specs/model/pvt/table.md). Nothing here is a real fluid. Three modes:

    pick OUT.json [--n 2000] [--seed 2026] [--per-class 2]
        ManyWells' environment. The first core volatile oils and gas condensates among R3's draws
        (plans/evidence/sampler_draft_draws.py), with their descriptors and the seed of their composition draw,
        rng_for(seed, well, 'v2', 'composition').
    make IN.json OUTDIR
        NeqSim's environment. For each picked well: draw a composition for its class, match its stock-tank API gravity
        and separator-gas gravity to the drawn ones and its GOR exactly (fluids-draft.md, The composition), find its
        saturation pressure at T_r, and tabulate it on the well's own grid. Then the published example fluids: their
        separation at standard conditions, and the table of Pedersen et al.'s (2015) Table 3.8 condensate, whose
        dew point at 155 C was measured (their Table 3.9). Writes one CSV per table and OUTDIR/summary.json.
    check OUTDIR
        ManyWells' environment. Each table against PVT-TAB-1's checks as they are and with the change that
        fluids-draft.md proposes; the fluid's phase split and densities at (p_r, T_r) from the table and from R3's
        interim black oil (Vazquez-Beggs at the drawn API, gas gravity and GOR).

The separation is one stage at standard conditions (1.01325 bar, 288.15 K), as pvt_table_eos.py has it, so that a
GOR axis recombines along one tie line (PVT-TAB-1). The drawn descriptors are the field's, as reported through the
well's own separator and a stock tank: the single-stage targets of the match are shifted by each fluid's offsets
between the two, in two passes, and the offsets are recorded (D-63, as Bjarne ruled at J2 on 2026-11-06). Where the stream is all vapour, the table keeps a liquid mass
fraction DELTA of the liquid at the nearest saturation point, so that B_o > 0 (fluids-draft.md, The table).

NeqSim needs Java. From the repository root, with the JVM sized to fit the memory cap:

    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/table_fluids.py pick picked.json)
    (ulimit -v 8000000; export MALLOC_ARENA_MAX=2 JAVA_TOOL_OPTIONS="-Xmx2g -XX:+UseSerialGC \\
        -XX:ActiveProcessorCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m"; timeout 1800 \\
        uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python plans/evidence/table_fluids.py make picked.json TABLES)
    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/table_fluids.py check TABLES)
"""

import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import pandas as pd

# ManyWells's constants (manywells.units), copied because make runs without ManyWells
R_UNIVERSAL = 8314.46  # J/(kmol K)
P_REF, T_REF = 101_325.0, 288.15
CF_BAR = 1e5
P_SC = P_REF / CF_BAR
M_AIR = 28.97
V_M_SC = R_UNIVERSAL * T_REF / P_REF  # m3/kmol of an ideal gas at standard conditions
DELTA = 1e-4  # liquid mass fraction kept where the stream is all vapour

LIGHT = (
    'nitrogen',
    'CO2',
    'methane',
    'ethane',
    'propane',
    'i-butane',
    'n-butane',
    'i-pentane',
    'n-pentane',
    'n-hexane',
)
INTERMEDIATE = slice(3, 10)  # C2 to C6

# Published example fluids: light ends in mol% in LIGHT's order, and the heptanes-plus either lumped, c7 = (mol%,
# molar mass g/mol, specific gravity), or as fractions, tbp = [(mol%, M, specific gravity), ...]
EXAMPLES = {
    'wb2000_gas_condensate': dict(
        cls='gas_condensate',
        ref='Whitson and Brule (2000), Table 2.1, gas condensate',
        light=(0.31, 2.37, 73.19, 7.80, 3.55, 0.71, 1.45, 0.64, 0.68, 1.09),
        c7=(8.21, 184.0, 0.816),
        reported=dict(gor=5450 * 0.178108, api=49, sg_gas=0.70, p_sat=6560 * 0.0689476),
    ),
    'pedersen2015_t1_3': dict(
        cls='gas_condensate',
        ref='Pedersen et al. (2015), Table 1.3, gas condensate',
        light=(0.53, 3.30, 72.98, 7.68, 4.10, 0.70, 1.42, 0.54, 0.67, 0.85),
        tbp=(
            (1.33, 91.3, 0.746),
            (1.33, 104.1, 0.768),
            (0.78, 118.8, 0.790),
            (0.61, 136, 0.787),
            (0.42, 150, 0.793),
            (0.33, 164, 0.804),
            (0.42, 179, 0.817),
            (0.24, 188, 0.830),
            (0.30, 204, 0.835),
            (0.17, 216, 0.843),
            (0.21, 236, 0.837),
            (0.15, 253, 0.840),
            (0.15, 270, 0.850),
            (0.80, 391, 0.877),
        ),
    ),
    'pedersen2015_t3_8': dict(
        cls='gas_condensate',
        ref='Pedersen et al. (2015), Table 3.8, gas condensate; constant-mass expansion at 155 C, Table 3.9',
        light=(0.60, 3.34, 74.16, 7.90, 4.15, 0.71, 1.44, 0.53, 0.66, 0.81),
        tbp=(
            (1.20, 91, 0.746),
            (1.15, 104, 0.770),
            (0.63, 119, 0.788),
            (0.50, 133, 0.795),
            (0.29, 144, 0.790),
            (0.27, 155, 0.802),
            (0.28, 168, 0.814),
            (0.22, 181, 0.824),
            (0.17, 195, 0.833),
            (0.15, 204, 0.836),
            (0.14, 224, 0.837),
            (0.09, 234, 0.839),
            (0.13, 248, 0.844),
            (0.47, 362, 0.877),
        ),
        measured=dict(T=428.15, p_dew=388.0),
    ),
    'wb2000_near_critical_oil': dict(
        cls='volatile_oil',
        ref='Whitson and Brule (2000), Table 2.1, near-critical oil',
        light=(0.56, 1.30, 69.44, 7.88, 4.26, 0.89, 2.14, 0.90, 1.13, 1.46),
        c7=(10.04, 219.0, 0.839),
        reported=dict(gor=3650 * 0.178108, api=45, sg_gas=0.71, p_sat=7015 * 0.0689476),
    ),
    'wb2000_volatile_oil': dict(
        cls='volatile_oil',
        ref='Whitson and Brule (2000), Table 2.1, volatile oil',
        light=(0.21, 0.93, 58.77, 7.57, 4.09, 0.91, 2.09, 0.77, 1.15, 1.75),
        c7=(21.76, 228.0, 0.858),
        reported=dict(gor=1490 * 0.178108, api=38, sg_gas=0.70, p_sat=5420 * 0.0689476),
    ),
    'pedersen2015_t1_4': dict(
        cls='volatile_oil',
        ref='Pedersen et al. (2015), Table 1.4, near-critical mixture',
        light=(0.46, 3.36, 62.36, 8.90, 5.31, 0.92, 2.08, 0.73, 0.85, 1.05),
        tbp=(
            (1.85, 95, 0.733),
            (1.75, 106, 0.756),
            (1.40, 121, 0.772),
            (1.07, 135, 0.791),
            (0.84, 150, 0.795),
            (0.76, 164, 0.809),
            (0.75, 177, 0.825),
            (0.64, 190, 0.835),
            (0.58, 201, 0.841),
            (0.50, 214, 0.847),
            (0.42, 232, 0.843),
            (0.42, 248, 0.846),
            (0.37, 256, 0.858),
            (2.63, 406, 0.897),
        ),
    ),
}

# The synthetic draw per class (fluids-draft.md, The composition): the examples whose light ends are mixed, the
# heptanes-plus mole fraction before recombination, and its molar mass
CLASSES = {
    'gas_condensate': dict(
        templates=('wb2000_gas_condensate', 'pedersen2015_t1_3', 'pedersen2015_t3_8'), z7=(0.04, 0.10), M7=(140, 200)
    ),
    'volatile_oil': dict(
        templates=('wb2000_near_critical_oil', 'wb2000_volatile_oil', 'pedersen2015_t1_4'),
        z7=(0.10, 0.22),
        M7=(190, 240),
    ),
}
GAMMA_BOUNDS = (135.0, 205.0, 275.0, 415.0)  # g/mol: about C9/C10, C14/C15, C19/C20 and C29/C30


def rng_for(seed, *parts):
    """manywells.sampling.wells.rng_for (SMP-31), copied because make runs without ManyWells."""
    key = [zlib.crc32(str(p).encode()) & 0xFFFFFFFF if isinstance(p, str) else int(p) for p in parts]
    return np.random.default_rng([int(seed)] + key)


def api_of(rho):
    return 141.5 / (rho / 999.0) - 131.5


# --- The heptanes-plus ------------------------------------------------------------------------------------------------
def gamma_split(M7, alpha):
    """Mole fractions and molar masses of five fractions of a heptanes-plus with mean molar mass M7 (g/mol), by
    Whitson's gamma distribution (Whitson and Brule 2000, Eqs. 5.13 to 5.15), with eta from alpha by Eq. 5.15."""
    eta = 110.0 * (1 - 1 / (1 + 4 / alpha**0.7))
    beta = (M7 - eta) / alpha
    M = np.linspace(eta, eta + 60 * beta * max(alpha, 1.0), 40001)[1:]
    pdf = np.exp((alpha - 1) * np.log(M - eta) - (M - eta) / beta - alpha * math.log(beta) - math.lgamma(alpha))
    edges = (eta, *GAMMA_BOUNDS, M[-1] + 1)
    z, Ms = [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (M > lo) & (M <= hi)
        z.append(np.trapezoid(pdf[m], M[m]))
        Ms.append(np.trapezoid(pdf[m] * M[m], M[m]) / z[-1])
    z = np.array(z) / sum(z)
    return z, np.array(Ms) * M7 / float(np.dot(z, Ms))


def soreide(Ms, Cf):
    """Specific gravities of fractions of molar masses Ms by Soreide (Whitson and Brule 2000, Eq. 5.44)."""
    return 0.2855 + Cf * (np.asarray(Ms) - 66.0) ** 0.13


# --- NeqSim -----------------------------------------------------------------------------------------------------------
class Flash:
    """A NeqSim system of one component set, flashed at (p, T) for any composition of that set."""

    def __init__(self, fractions):
        """fractions: [(name, M g/mol, specific gravity)] of the heptanes-plus, after LIGHT."""
        from neqsim import jneqsim

        self.j = jneqsim
        s = jneqsim.thermo.system.SystemPrEos(T_REF, P_SC)
        for name in LIGHT:
            s.addComponent(name, 1.0)
        for name, M, g in fractions:
            s.addTBPfraction(name, 1.0, float(M) / 1000, float(g))
        s.setMixingRule('classic')
        s.useVolumeCorrection(True)
        s.setMultiPhaseCheck(
            True
        )  # without it, the TP flash misses a condensate's liquid of 1 to 2 mol% near the dew point
        self.s, self.ops = s, jneqsim.thermodynamicoperations.ThermodynamicOperations(s)
        s.init(0)
        self.n = s.getPhase(0).getNumberOfComponents()

    def set(self, z, p, T):
        from jpype import JArray, JDouble

        self.s.setMolarComposition(JArray(JDouble)([float(v) for v in z]))
        self.s.setTemperature(float(T))
        self.s.setPressure(float(p))

    def flash(self, z, p, T, properties=True):
        """[(mass fraction, density kg/m3, molar mass kg/mol, viscosity Pa s, mole fraction, composition)] sorted by
        density, lightest first, and the interfacial tension (N/m) of two phases."""
        self.set(z, p, T)
        self.ops.TPflash()
        if properties:
            self.s.initPhysicalProperties()
        else:
            self.s.initProperties()
        k = int(self.s.getNumberOfPhases())
        out = []
        for i in range(k):
            ph = self.s.getPhase(i)
            beta = float(ph.getBeta()) if k > 1 else 1.0
            x = [float(ph.getComponent(j).getx()) for j in range(self.n)]
            mu = float(ph.getViscosity('kg/msec')) if properties else math.nan
            out.append([beta * float(ph.getMolarMass()), float(ph.getDensity('kg/m3')), float(ph.getMolarMass()), mu,
                        beta, x])  # fmt: skip
        total = sum(q[0] for q in out)
        for q in out:
            q[0] /= total
        out.sort(key=lambda q: q[1])
        ift = float(self.s.getInterfacialTension(0, 1)) if (k > 1 and properties) else math.nan
        if k > 2:  # two liquids: one liquid, by mass and volume, with the viscosity of the larger by mass
            liq = out[1:]
            mass, beta = sum(q[0] for q in liq), sum(q[4] for q in liq)
            big = max(liq, key=lambda q: q[0])
            x = list(sum(q[4] * np.array(q[5]) for q in liq) / beta)
            out = [out[0], [mass, mass / sum(q[0] / q[1] for q in liq), sum(q[4] * q[2] for q in liq) / beta, big[3],
                            beta, x]]  # fmt: skip
        return out, ift

    def separation(self, z):
        """One stage at standard conditions: rho_o (kg/m3), rho_g (kg/Sm3, ideal), GOR (Sm3/Sm3), and the gas's and the
        oil's compositions."""
        phases, _ = self.flash(z, P_SC, T_REF, properties=False)
        if len(phases) < 2:
            raise ValueError('no stock-tank oil: the stream is all gas at standard conditions')
        (X, _, M_V, _, _, y), (_, rho_L, M_L, _, _, x) = phases
        rho_g = P_REF * M_V * 1000 / (R_UNIVERSAL * T_REF)
        return dict(rho_o=rho_L, rho_g=rho_g, gor=(X / rho_g) / ((1 - X) / rho_L), y=y, x=x, M_o=M_L * 1000)

    def field(self, z, p_sep, T_sep):
        """The field's report of stream z (D-63, as ruled at J2): the well's separator at (p_sep, T_sep), then its
        liquid to a stock tank at standard conditions. The stock-tank oil's API gravity, the separator gas's gravity
        (the stock tank's where the separator has no gas), and the GOR of both gases per Sm3 of stock-tank oil."""
        phases, _ = self.flash(z, p_sep, T_sep, properties=False)
        if len(phases) == 2:
            (X1, _, M_V1, _, _, _), (_, _, _, _, _, x1) = phases
        elif phases[0][1] < 200.0:  # all vapour at the separator: no liquid to report
            return None
        else:
            X1, M_V1, x1 = 0.0, math.nan, phases[0][5]
        st, _ = self.flash(x1, P_SC, T_REF, properties=False)
        if len(st) < 2:
            if st[0][1] < 200.0:
                return None
            X2, M_V2, rho_L = 0.0, math.nan, st[0][1]
        else:
            (X2, _, M_V2, _, _, _), (_, rho_L, _, _, _, _) = st
        gas = (X1 / (P_REF * M_V1 * 1000 / (R_UNIVERSAL * T_REF)) if X1 > 0 else 0.0) + (
            (1 - X1) * X2 / (P_REF * M_V2 * 1000 / (R_UNIVERSAL * T_REF)) if X2 > 0 else 0.0
        )
        M_g = M_V1 if X1 > 0 else M_V2
        return dict(api=api_of(rho_L), sg_gas=M_g * 1000 / M_AIR, gor=gas / ((1 - X1) * (1 - X2) / rho_L))

    def recombine(self, sep, gor):
        """The stream of the separator gas and stock-tank oil of sep at the GOR gor: per Sm3 of oil, rho_o/M_o kmol of
        oil and gor/V_M_SC kmol of gas."""
        n_o, n_g = sep['rho_o'] / sep['M_o'], gor / V_M_SC
        z = n_g * np.array(sep['y']) + n_o * np.array(sep['x'])
        return z / z.sum()

    def saturation(self, z, T, p_hi, step=10.0):
        """The upper saturation pressure (bar) at T below p_hi, its kind ('bubble' or 'dew'), and the last two-phase
        flash below it; None if the stream has two phases nowhere on (1 bar, p_hi)."""
        grid = np.arange(1.0, p_hi + step, step)
        two = [len(self.flash(z, p, T, properties=False)[0]) == 2 for p in grid]
        if not any(two):
            return None
        i = max(k for k, t in enumerate(two) if t)
        if i == len(grid) - 1:
            return dict(p=float(grid[i]), kind='above_grid', lo=None, p_flash=float(grid[i]))
        lo, hi = grid[i], grid[i + 1]
        while hi - lo > 1e-3:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if len(self.flash(z, mid, T, properties=False)[0]) == 2 else (lo, mid)
        phases, _ = self.flash(z, lo, T)
        kind = 'bubble' if phases[0][4] < phases[1][4] else 'dew'  # the incipient phase has the smaller mole fraction
        p_flash = float(lo)
        return dict(p=float(lo), kind=kind, lo=(phases, p_flash), p_flash=p_flash)


# --- Composition --------------------------------------------------------------------------------------------------------
def example_stream(name, alpha=1.0):
    """An example fluid's TBP fractions [(name, M, g)] and its composition (mole fractions)."""
    ex = EXAMPLES[name]
    if 'tbp' in ex:
        fr = [(f'F{i + 1}', M, g) for i, (_, M, g) in enumerate(ex['tbp'])]
        z7 = [x for x, _, _ in ex['tbp']]
    else:
        x7, M7, g7 = ex['c7']
        zf, Ms = gamma_split(M7, alpha)
        lo, hi = 0.2, 0.45  # Soreide's Cf so that the fractions' volume-additive gravity is the measured one
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g = soreide(Ms, mid)
            lo, hi = (mid, hi) if np.dot(zf, Ms) / np.dot(zf, Ms / g) < g7 else (lo, mid)
        g = soreide(Ms, 0.5 * (lo + hi))
        fr = [(f'F{i + 1}', M, gi) for i, (M, gi) in enumerate(zip(Ms, g))]
        z7 = list(zf * x7)
    z = np.array(list(ex['light']) + z7)
    return fr, z / z.sum()


def synthetic(cls, rng, target, max_iter=8):
    """
    A synthetic composition of class cls matched to target's API gravity and separator-gas gravity (roughly, by
    Soreide's Cf and a factor on C2 to C6), recombined at its R_si (exactly). Returns the Flash, the stream, the
    separation and a record of the draw and the match.
    """
    c = CLASSES[cls]
    w = rng.dirichlet(np.ones(len(c['templates'])))
    light = sum(wi * np.array(EXAMPLES[t]['light']) / sum(EXAMPLES[t]['light']) for wi, t in zip(w, c['templates']))
    z7 = rng.uniform(*c['z7'])
    M7 = rng.uniform(*c['M7'])
    alpha = rng.uniform(0.5, 2.5)
    zf, Ms = gamma_split(M7, alpha)
    rec = dict(template_weights=dict(zip(c['templates'], w.round(4).tolist())), z7=z7, M7=M7, alpha=alpha)

    def build(u):
        Cf, ln_f = u
        lt = light.copy()
        lt[INTERMEDIATE] *= math.exp(ln_f)
        z = np.concatenate([lt / lt.sum() * (1 - z7), zf * z7])
        fl = Flash([(f'F{i + 1}', M, g) for i, (M, g) in enumerate(zip(Ms, soreide(Ms, Cf)))])
        sep = fl.separation(z)
        return fl, z, sep, np.array([api_of(sep['rho_o']) - target['api'], sep['rho_g'] / RHO_AIR - target['sg_gas']])

    u = np.array([0.29, 0.0])
    lo_b, hi_b = np.array([0.22, math.log(0.25)]), np.array([0.40, math.log(4.0)])
    for it in range(max_iter):
        fl, z, sep, r = build(u)
        if abs(r[0]) < 0.5 and abs(r[1]) < 0.005:
            break
        J = np.empty((2, 2))
        for k, h in enumerate((0.005, 0.05)):
            du = np.zeros(2)
            du[k] = h
            J[:, k] = (build(u + du)[3] - r) / h
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            break
        u = np.clip(u + step, lo_b, hi_b)
    fl, z, sep, r = build(u)
    rec |= dict(Cf=float(u[0]), intermediates_factor=math.exp(u[1]), iterations=it + 1)
    rec |= dict(api_error=float(r[0]), sg_gas_error=float(r[1]))
    return fl, fl.recombine(sep, target['R_si']), sep, rec


RHO_AIR = P_REF * M_AIR / (R_UNIVERSAL * T_REF)


def field_matched(w, passes=2):
    """
    D-63 as ruled at J2: the composition matched as single-stage descriptors at standard conditions (synthetic), then
    flashed through the well's separator and stock tank (Flash.field), and the single-stage targets shifted by the
    difference, so that the field-reported API, gas gravity and GOR are the drawn ones. Each pass redraws the same
    composition from the well's seed. Returns synthetic's four values, with the offsets and the field's descriptors
    in the record.
    """
    drawn = dict(api=w['api'], sg_gas=w['sg_gas'], R_si=w['R_si'])
    target = dict(drawn)
    for k in range(passes + 1):
        fl, z, sep, rec = synthetic(w['fluid_class'], np.random.default_rng(w['composition_seed']), target)
        f = fl.field(z, w['p_sep'], w['T_sep'])
        if f is None:
            rec |= dict(field=None, offsets=None, passes=k)
            break
        single = dict(api=api_of(sep['rho_o']), sg_gas=sep['rho_g'] / RHO_AIR, R_si=target['R_si'])
        offsets = dict(
            api=f['api'] - single['api'], sg_gas=f['sg_gas'] - single['sg_gas'], gor=f['gor'] / single['R_si']
        )
        rec |= dict(field=f, offsets=offsets, single_stage=single, passes=k)
        if k == passes:
            break
        target = dict(
            api=drawn['api'] - offsets['api'],
            sg_gas=drawn['sg_gas'] - offsets['sg_gas'],
            R_si=drawn['R_si'] / offsets['gor'],
        )
    rec['field_error'] = None if rec['field'] is None else dict(
        api=rec['field']['api'] - drawn['api'], sg_gas=rec['field']['sg_gas'] - drawn['sg_gas'],
        gor=rec['field']['gor'] / drawn['R_si'] - 1,
    )  # fmt: skip
    return fl, z, sep, rec


def saturated_at(fl, sep, gor, T, p_i):
    """The GOR on sep's recombination line whose saturation pressure at T is p_i, by bisection on ln GOR, searching
    out from gor; None if there is none within a factor 4."""

    def excess(g):
        s = fl.saturation(fl.recombine(sep, g), T, 1.3 * p_i)
        return (s['p'] if s else 0.0) - p_i

    for f in (0.8, 0.6, 0.4, 0.25, 1.25, 1.6, 2.5, 4.0):
        if excess(gor * f) < 0:
            lo, hi = sorted((math.log(gor), math.log(gor * f)))
            e_lo = excess(math.exp(lo))
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if (excess(math.exp(mid)) < 0) == (e_lo < 0):
                    lo = mid
                else:
                    hi = mid
            return math.exp(0.5 * (lo + hi)) if excess(math.exp(lo)) < 0 or excess(math.exp(hi)) < 0 else None
    return None


# --- The table ----------------------------------------------------------------------------------------------------------
def tabulate(fl, z, sep_gor, T_axis, p_top):
    """
    The table of stream z (PVT-TAB-6) on T_axis and a pressure axis from 1 bar to p_top with every temperature's
    saturation pressures as nodes. Two-phase nodes from the flash, with the vapour's mass fraction X at most
    1 - DELTA; all-liquid nodes with the incipient vapour flashed at (p, T), as pvt_table_eos.py has them; all-vapour
    nodes with X = 1 - DELTA and the liquid of the nearest saturation point.
    """
    rho_o, rho_g, gor = sep_gor
    m = rho_o + gor * rho_g
    R_s = P_REF / (rho_g * T_REF)
    base = np.unique(np.r_[1.0, 2.5, 5.0, 7.5, np.arange(10.0, p_top + 1e-9, 10.0)])
    sat = {T: fl.saturation(z, T, p_top) for T in T_axis}
    p_axis = np.unique(np.r_[base, [round(s['p'], 3) for s in sat.values() if s and s['kind'] != 'above_grid']])
    rows = []
    for T in T_axis:
        s = sat[T]
        flashes = [(p, *fl.flash(z, p, T)) for p in p_axis]
        two = [k for k, (_, ph, _) in enumerate(flashes) if len(ph) == 2]
        for k, (p, ph, ift) in enumerate(flashes):
            if len(ph) == 2:
                (X, rho_V, _, mu_g, _, _), (_, rho_L, _, mu_o, _, _) = ph
                sigma = ift
            else:
                near = min(two, key=lambda j: abs(j - k)) if two else None
                liquid = near is not None and p > flashes[near][0] and s and s['kind'] == 'bubble'
                if liquid:  # above a bubble point: no free gas; the gas is the incipient vapour
                    X, rho_L, mu_o = 0.0, ph[0][1], ph[0][3]
                    y_b = s['lo'][0][0][5]
                    gas, _ = fl.flash(y_b, p, T)
                    rho_V = sum(q[0] for q in gas) / sum(q[0] / q[1] for q in gas)
                    mu_g = gas[0][3]
                    sigma = flashes[near][2]
                else:  # all vapour: the liquid held at the nearest two-phase node's, or none at this temperature
                    X, rho_V, mu_g = 1.0, ph[0][1], ph[0][3]
                    if near is not None:
                        _, (_, rho_L, _, mu_o, _, _) = flashes[near][1]
                        sigma = flashes[near][2]
                    else:
                        rho_L, mu_o, sigma = math.nan, math.nan, math.nan
            X = min(X, 1 - DELTA)
            rows.append(dict(p=p, T=T, X=X, rho_L=rho_L, rho_V=rho_V, mu_o=mu_o, mu_g=mu_g, sigma=sigma))
    df = pd.DataFrame(rows)
    for col in ('rho_L', 'mu_o', 'sigma'):  # a temperature with no two-phase node takes its neighbour's
        df[col] = df.groupby('p')[col].transform(lambda c: c.ffill().bfill())
    df['R_so'] = gor - df.X * m / rho_g
    df['B_o'] = (1 - df.X) * m / df.rho_L
    df['Z'] = CF_BAR * df.p / (df.rho_V * R_s * df['T'])
    return df, sat


def write_table(path, df, name, source, rho_o, rho_g, gor, extra=()):
    header = [
        '# ManyWells PVT table (specs/model/pvt/table.md)',
        f'# name: {name}',
        f'# source: plans/evidence/table_fluids.py; NeqSim 3.4.0, Peng-Robinson with Peneloux; {source}',
        '# separation: single stage at 1.01325 bar and 288.15 K',
        f'# rho_o: {rho_o:.4f}',
        f'# rho_g: {rho_g:.6f}',
        f'# gor: {gor:.4f}',
        *extra,
    ]
    with open(path, 'w') as fh:
        fh.write('\n'.join(header) + '\n')
        df[['p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma']].to_csv(fh, index=False, float_format='%.10g')


def well_axes(w):
    """The well's grid: 10 K steps from below T_s - 15 K to above T_r, and pressures to 1.05 p_i plus a step."""
    T_lo = 10 * math.floor((w['T_s'] - 15) / 10)
    T_hi = 10 * math.ceil((w['T_r'] + 1) / 10)
    return np.arange(T_lo, T_hi + 1e-9, 10.0), 10 * math.ceil(1.05 * w['p_i'] / 10) + 10


def make(picked, outdir):
    os.makedirs(outdir, exist_ok=True)
    summary = []
    for w in picked:
        t0 = time.time()
        fl, z, sep, rec = field_matched(w)
        t_match = time.time() - t0
        sat_r = fl.saturation(z, w['T_r'], 1.3 * w['p_i'])
        gor = rec['single_stage']['R_si'] if rec.get('single_stage') else w['R_si']
        moved, rule = None, 'matched'
        if sat_r and sat_r['p'] > w['p_i']:  # above its saturation pressure at p_i: saturated at discovery
            g = saturated_at(fl, sep, gor, w['T_r'], w['p_i'])
            if g is not None:
                moved, gor, rule = g / gor, g, 'gor_moved'
                z = fl.recombine(sep, gor)
            else:  # no GOR on the line will do: the phase of the class in equilibrium at (p_i, T_r)
                phases, _ = fl.flash(z, w['p_i'], w['T_r'], properties=False)
                if len(phases) == 2:
                    z = np.array(phases[0 if w['fluid_class'] == 'gas_condensate' else 1][5])
                    sep = fl.separation(z)
                    moved, gor, rule = sep['gor'] / gor, sep['gor'], 'equilibrium_phase'
            sat_r = fl.saturation(z, w['T_r'], 1.3 * w['p_i'])
        rec['saturation_rule'] = rule
        t_sat = time.time() - t0 - t_match
        T_axis, p_top = well_axes(w)
        df, sat = tabulate(fl, z, (sep['rho_o'], sep['rho_g'], gor), T_axis, p_top)
        t_table = time.time() - t0 - t_match - t_sat
        name = f'well{w["well"]}_{w["fluid_class"]}'
        write_table(
            os.path.join(outdir, f'{name}.csv'), df, name,
            f'a synthetic {w["fluid_class"].replace("_", " ")} matched to well {w["well"]} of R3\'s draws (seed '
            f'{w["seed"]})', sep['rho_o'], sep['rho_g'], gor,
        )  # fmt: skip
        summary.append(
            dict(
                name=name, well=w['well'], fluid_class=w['fluid_class'], drawn=w, composition=rec,
                api=api_of(sep['rho_o']), sg_gas=sep['rho_g'] / RHO_AIR, gor=gor, gor_moved=moved,
                c7_mol=float(np.sum(z[len(LIGHT):])), c1_mol=float(z[2]),
                p_sat_T_r=sat_r['p'] if sat_r else None, kind_T_r=sat_r['kind'] if sat_r else None,
                p_sat_flash_T_r=sat_r['p_flash'] if sat_r else None,
                p_sat={str(T): (s['p'], s['kind']) if s else None for T, s in sat.items()},
                nodes=len(df), n_p=df.p.nunique(), n_T=len(T_axis), seconds=dict(match=t_match, saturation=t_sat,
                table=t_table),
            )
        )  # fmt: skip
        print(f'{name}: {json.dumps({k: v for k, v in summary[-1].items() if k not in ("drawn", "p_sat")})}')
    for name, ex in EXAMPLES.items():
        t0 = time.time()
        fr, z = example_stream(name)
        fl = Flash(fr)
        sep = fl.separation(z)
        rec = dict(name=name, ref=ex['ref'], fluid_class=ex['cls'], api=api_of(sep['rho_o']), c7_mol=float(np.sum(z[len(LIGHT):])),
                   sg_gas=sep['rho_g'] / RHO_AIR, gor=sep['gor'], reported=ex.get('reported'))  # fmt: skip
        if 'measured' in ex:
            T = ex['measured']['T']
            s = fl.saturation(z, T, 700.0)
            rec |= dict(measured=ex['measured'], p_sat_computed=s['p'], p_sat_flash=s['p_flash'], kind=s['kind'])
            df, sat = tabulate(fl, z, (sep['rho_o'], sep['rho_g'], sep['gor']), np.arange(280.0, 440.0 + 1e-9, 20.0),
                               460.0)  # fmt: skip
            write_table(os.path.join(outdir, f'{name}.csv'), df, name, ex['ref'], sep['rho_o'], sep['rho_g'],
                        sep['gor'])  # fmt: skip
            rec |= dict(nodes=len(df), p_sat={str(T): (s['p'], s['kind']) if s else None for T, s in sat.items()})
        rec['seconds'] = time.time() - t0
        summary.append(rec)
        print(f'{name}: {json.dumps({k: v for k, v in rec.items() if k != "p_sat"})}')
    with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)


# --- ManyWells' side ----------------------------------------------------------------------------------------------------
def pick(out, n, seed, per_class):
    from sampler_draft_draws import draw_well

    got = {'volatile_oil': [], 'gas_condensate': []}
    for k in range(n):
        w, _ = draw_well(seed, k)
        if w['tail'] or w['fluid_class'] not in got or len(got[w['fluid_class']]) >= per_class:
            continue
        keep = ('well', 'fluid_class', 'api', 'sg_gas', 'R_go', 'p_sat', 'p_i', 'p_r', 'T_r', 'T_s', 'p_s')
        keep += ('wlr', 'rho_w', 'rho_o', 'rho_g', 'p_sep', 'T_sep', 'stage', 'z')
        d = {key: w[key] for key in keep} | {'R_si': w['R_i']}
        d |= dict(seed=seed, composition_seed=int(rng_for(seed, k, 'v2', 'composition').integers(2**63)))
        got[w['fluid_class']].append(d)
        if all(len(v) >= per_class for v in got.values()):
            break
    with open(out, 'w') as fh:
        json.dump([d for v in got.values() for d in v], fh, indent=1, default=float)
    print(json.dumps(got, indent=1, default=float))


def check(outdir):
    from manywells.pvt.black_oil import BlackOil
    from manywells.pvt.fluid import FluidModel
    from manywells.pvt.table import PVTTable

    class Relaxed(PVTTable):
        """PVT-TAB-1 as fluids-draft.md proposes: R_so may fall as the pressure rises (retrograde condensation)."""

        def _validate(self):
            R_so = self.R_so
            object.__setattr__(self, 'R_so', tuple(map(tuple, np.sort(np.array(R_so), axis=0))))  # passes the check
            try:
                super()._validate()
            finally:
                object.__setattr__(self, 'R_so', R_so)

    with open(os.path.join(outdir, 'summary.json')) as fh:
        summary = json.load(fh)
    for rec in summary:
        path = os.path.join(outdir, f'{rec["name"]}.csv')
        if not os.path.exists(path):
            continue
        try:
            PVTTable.read_csv(path)
            today = 'accepted'
        except ValueError as e:
            today = f'refused: {e}'
        df = pd.read_csv(path, comment='#')
        header = {}
        with open(path) as fh:
            for line in fh:
                if line.startswith('#') and ':' in line:
                    k, v = line[1:].split(':', 1)
                    header[k.strip()] = v.strip()
        P, TT = np.sort(df.p.unique()), np.sort(df['T'].unique())
        cols = {c: df.pivot(index='p', columns='T', values=c).loc[P, TT].to_numpy() for c in df.columns[2:]}
        table = Relaxed(p=tuple(P), T=tuple(TT), rho_o=float(header['rho_o']), rho_g=float(header['rho_g']),
                        gor=float(header['gor']), **{c: tuple(map(tuple, v)) for c, v in cols.items()})  # fmt: skip
        falls = int((np.diff(cols['R_so'], axis=0) < -1e-9).sum())
        print(f'\n{rec["name"]} ({rec.get("fluid_class")}): {len(df)} nodes; PVT-TAB-1 today: {today}')
        print(
            f'  R_so falls with p at {falls} node pairs; B_o from {cols["B_o"].min():.3g} to {cols["B_o"].max():.3g};'
            f' R_so from {cols["R_so"].min():.4g} to {cols["R_so"].max():.4g}; sigma from {cols["sigma"].min():.3g}'
        )
        w = rec.get('drawn')
        if not w:
            continue
        fluid = FluidModel(pvt=table, gor=rec['gor'], wlr=w['wlr'], rho_w=w['rho_w'])
        bo = FluidModel(pvt=BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'],
                                     T_sep=w['T_sep']), gor=w['R_si'], wlr=w['wlr'], rho_w=w['rho_w'])  # fmt: skip
        print(
            f'  drawn: API {w["api"]:.1f}, gas gravity {w["sg_gas"]:.3f}, R_si {w["R_si"]:.0f}, p_sat '
            f'{w["p_sat"]:.0f} bar, p_i {w["p_i"]:.0f}, p_r {w["p_r"]:.0f}, T_r {w["T_r"]:.0f} K; table: API '
            f'{rec["api"]:.1f}, gas gravity {rec["sg_gas"]:.3f}, GOR {rec["gor"]:.0f} (moved x{rec["gor_moved"]}),'
            f' p_sat at T_r {rec["p_sat_T_r"]} ({rec["kind_T_r"]})'
        )
        p_b = next((p for p in np.arange(1.0, 1500.0, 1.0) if float(bo.rs(p, w['T_r'])) >= 0.999 * w['R_si']), None)
        print(f'  interim black oil: bubble point at T_r {p_b} bar (none below 1500 bar if None)')
        for label, (p, T) in (('(p_r, T_r)', (w['p_r'], w['T_r'])), ('(p_s, T_s + 20 K)', (w['p_s'], w['T_s'] + 20))):
            for nm, fm in (('table', fluid), ('black oil', bo)):
                w_g, w_l = (float(v) for v in fm.phase_rates(p, T, 1.0, 0.0))
                print(
                    f'  {label} {nm:9}: free-gas mass fraction {w_g / (w_g + w_l):.4f}, liquid density '
                    f'{float(fm.liquid_density(p, T)):.1f}, gas density {float(fm.gas_density(p, T)):.1f} kg/m3'
                )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('pick')
    a.add_argument('out')
    a.add_argument('--n', type=int, default=2000)
    a.add_argument('--seed', type=int, default=2026)
    a.add_argument('--per-class', type=int, default=2)
    a = sub.add_parser('make')
    a.add_argument('picked')
    a.add_argument('outdir')
    a = sub.add_parser('check')
    a.add_argument('outdir')
    args = ap.parse_args()
    if args.mode == 'pick':
        pick(args.out, args.n, args.seed, args.per_class)
    elif args.mode == 'make':
        with open(args.picked) as fh:
            make(json.load(fh), args.outdir)
    else:
        check(args.outdir)


if __name__ == '__main__':
    main()
