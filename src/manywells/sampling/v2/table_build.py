"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The NeqSim side of the v2 datasets' table fluids (specs/sampling.md, SMP-112 to SMP-116): a volatile oil's or a gas
condensate's composition, synthetic and matched to the well's drawn descriptors or a published example, its saturation
pressure at the reservoir temperature, and its PVT tables at the GORs of the well's axis, from NeqSim's Peng-Robinson
equation of state with the Peneloux volume correction, as PVT-TAB-6 turns a phase split into a table
(specs/model/pvt/table.md). Nothing here is a real fluid.

The build has four steps (SMP-116), two of them here. ManyWells writes DIR/wells.json, the table wells with their
draws; build makes each well's fluid (DIR/fluids/kK.json); ManyWells writes DIR/axes.json, each well's GOR axis from
its life with the fluid's values in it; build, run again, makes each well's table set at the GORs of its axis
(DIR/kK.csv, which PVTTable.read_csv reads as a PVTTableSet) and its manifest (DIR/kK.json). A well whose step fails
is written with the status 'failed' and its error; a well whose output exists is skipped, so that a stopped build
resumes.

NeqSim needs Java and runs in an environment of its own, outside ManyWells' dependencies. So this module imports only
the standard library, numpy and pandas, NeqSim where a flash is made, and nothing of ManyWells: there it is run by its
path, and in ManyWells' environment it is imported for its functions that need no flash. From the repository root,
with the JVM sized to fit the memory cap of each process:

    (ulimit -v 8000000; export MALLOC_ARENA_MAX=2 JAVA_TOOL_OPTIONS="-Xmx1g -XX:+UseSerialGC \\
        -XX:ActiveProcessorCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m"; \\
        uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python src/manywells/sampling/v2/table_build.py DIR --processes 4)

The files. DIR/wells.json holds kind ('sol' or 'nsol'), seed and wells, one record per table well: well (its draw
index), fluid_class, source ('synthetic' or an example's key), composition_seed, the drawn api, sg_gas, R_si and p_sat,
p_i, T_r, the separator's p_sep and T_sep, and the grid's T_axis and p_top (well_grid). DIR/axes.json holds, by draw
index, gors, or error where the axis could not be made. A fluid's record holds its labels, hook (the values that
replace the fluid block's draws: p_sat, R_i, rho_o, rho_g, and api and sg_gas where they are not the drawn ones) and
stream (the fractions, the composition and the separation, which the tables are made from); a manifest holds the
fluid's record without the stream, and csv, gors, nodes, n_T, p_top and dropped_pressures.
"""

import argparse
import json
import math
import multiprocessing
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

NEQSIM_VERSION = '3.4.0'  # spec: SMP-112
SOURCE = 'manywells.sampling.v2.table_build'  # The tables' maker, in their header

# ManyWells' constants (manywells.units), copied because the build runs without ManyWells
R_UNIVERSAL = 8314.46  # J/(kmol K)
P_REF, T_REF = 101_325.0, 288.15
CF_BAR = 1e5
P_SC = P_REF / CF_BAR
M_AIR = 28.97
V_M_SC = R_UNIVERSAL * T_REF / P_REF  # m3/kmol of an ideal gas at standard conditions
RHO_AIR = P_REF * M_AIR / (R_UNIVERSAL * T_REF)

DELTA = 1e-4  # The liquid mass fraction kept where the stream is all vapour (PVT-TAB-6)
MAX_DROPPED = 0.05  # The largest share of a table's pressures that the repair may drop; beyond it the table fails
P_EXAMPLE = 0.2  # A table well's probability of a published example fluid of its class

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
EXAMPLES = {  # spec: SMP-114
    'wb2000_gas_condensate': dict(
        cls='gas_condensate',
        ref='Whitson and Brule (2000), Table 2.1, gas condensate',
        light=(0.31, 2.37, 73.19, 7.80, 3.55, 0.71, 1.45, 0.64, 0.68, 1.09),
        c7=(8.21, 184.0, 0.816),
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
    ),
    'wb2000_near_critical_oil': dict(
        cls='volatile_oil',
        ref='Whitson and Brule (2000), Table 2.1, near-critical oil',
        light=(0.56, 1.30, 69.44, 7.88, 4.26, 0.89, 2.14, 0.90, 1.13, 1.46),
        c7=(10.04, 219.0, 0.839),
    ),
    'wb2000_volatile_oil': dict(
        cls='volatile_oil',
        ref='Whitson and Brule (2000), Table 2.1, volatile oil',
        light=(0.21, 0.93, 58.77, 7.57, 4.09, 0.91, 2.09, 0.77, 1.15, 1.75),
        c7=(21.76, 228.0, 0.858),
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

# The synthetic draw per class: the examples whose light ends are mixed, the heptanes-plus mole fraction before the
# GOR is set, and its molar mass (g/mol)
CLASSES = {  # spec: SMP-113
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


def fluid_source(fluid_class, u_source, u_key) -> str:  # spec: SMP-112
    """A table well's source from two uniform draws: with probability P_EXAMPLE one of its class's published examples,
    with equal shares, else 'synthetic'."""
    keys = [name for name, ex in EXAMPLES.items() if ex['cls'] == fluid_class]
    return keys[int(u_key * len(keys))] if u_source < P_EXAMPLE else 'synthetic'


def _api_of(rho):
    return 141.5 / (rho / 999.0) - 131.5


# --- The heptanes-plus ------------------------------------------------------------------------------------------------
def _gamma_split(M7, alpha):  # spec: SMP-113
    """Mole fractions and molar masses (g/mol) of the five fractions of a heptanes-plus with mean molar mass M7, split
    at GAMMA_BOUNDS by Whitson's gamma distribution of shape alpha (Whitson and Brule 2000, Eqs. 5.13 to 5.15), with
    eta from alpha by Eq. 5.15."""
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


def _soreide(Ms, Cf):  # spec: SMP-113
    """Specific gravities of fractions of molar masses Ms (g/mol) by Soreide (Whitson and Brule 2000, Eq. 5.44)."""
    return 0.2855 + Cf * (np.asarray(Ms) - 66.0) ** 0.13


# --- NeqSim -----------------------------------------------------------------------------------------------------------
class _Flash:  # spec: SMP-112
    """A NeqSim system of one component set, Peng-Robinson with the Peneloux volume correction, NeqSim's
    characterisation of the fractions and its default interaction parameters, flashed at (p, T) for any composition
    of that set."""

    def __init__(self, fractions):
        """fractions: [(name, M g/mol, specific gravity)] of the heptanes-plus, after LIGHT."""
        from neqsim import jneqsim

        self.j, self.fractions = jneqsim, [(name, float(M), float(g)) for name, M, g in fractions]
        s = jneqsim.thermo.system.SystemPrEos(T_REF, P_SC)
        for name in LIGHT:
            s.addComponent(name, 1.0)
        for name, M, g in fractions:
            s.addTBPfraction(name, 1.0, float(M) / 1000, float(g))
        s.setMixingRule('classic')
        s.useVolumeCorrection(True)
        s.setMultiPhaseCheck(True)  # Without it, the TP flash misses a condensate's liquid near the dew point: SMP-115
        self.s, self.ops = s, jneqsim.thermodynamicoperations.ThermodynamicOperations(s)
        s.init(0)
        self.n = s.getPhase(0).getNumberOfComponents()

    def set(self, z, p, T):
        from jpype import JArray, JDouble

        self.s.setMolarComposition(JArray(JDouble)([float(v) for v in z]))
        self.s.setTemperature(float(T))
        self.s.setPressure(float(p))

    def flash(self, z, p, T, properties=True):
        """[[mass fraction, density kg/m3, molar mass kg/mol, viscosity Pa s, mole fraction, composition]] sorted by
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
            M = float(ph.getMolarMass())
            out.append([beta * M, float(ph.getDensity('kg/m3')), M, mu, beta, x])
        total = sum(q[0] for q in out)
        for q in out:
            q[0] /= total
        out.sort(key=lambda q: q[1])
        ift = float(self.s.getInterfacialTension(0, 1)) if (k > 1 and properties) else math.nan
        if k > 2:  # Two liquids: one liquid, by mass and volume, with the viscosity of the larger by mass
            liq = out[1:]
            mass, beta = sum(q[0] for q in liq), sum(q[4] for q in liq)
            big = max(liq, key=lambda q: q[0])
            x = list(sum(q[4] * np.array(q[5]) for q in liq) / beta)
            rho, M = mass / sum(q[0] / q[1] for q in liq), sum(q[4] * q[2] for q in liq) / beta
            out = [out[0], [mass, rho, M, big[3], beta, x]]
        return out, ift

    def separation(self, z):  # spec: SMP-113
        """One stage at standard conditions: rho_o (kg/m3), rho_g (kg/Sm3, ideal), GOR (Sm3/Sm3), the gas's and the
        oil's compositions, and the oil's molar mass (g/mol)."""
        phases, _ = self.flash(z, P_SC, T_REF, properties=False)
        if len(phases) < 2:
            raise ValueError('no stock-tank oil: the stream is all gas at standard conditions')
        (X, _, M_V, _, _, y), (_, rho_L, M_L, _, _, x) = phases
        rho_g = P_REF * M_V * 1000 / (R_UNIVERSAL * T_REF)
        return dict(rho_o=rho_L, rho_g=rho_g, gor=(X / rho_g) / ((1 - X) / rho_L), y=y, x=x, M_o=M_L * 1000)

    def field(self, z, p_sep, T_sep):  # spec: SMP-113
        """The field's report of stream z: the well's separator at (p_sep, T_sep), then its liquid to a stock tank at
        standard conditions. The stock-tank oil's API gravity, the separator gas's gravity (the stock tank's where the
        separator has no gas), and the GOR of both gases per Sm3 of stock-tank oil; None where no liquid is left."""
        phases, _ = self.flash(z, p_sep, T_sep, properties=False)
        if len(phases) == 2:
            (X1, _, M_V1, _, _, _), (_, _, _, _, _, x1) = phases
        elif phases[0][1] < 200.0:  # All vapour at the separator: no liquid to report
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
        return dict(api=_api_of(rho_L), sg_gas=M_g * 1000 / M_AIR, gor=gas / ((1 - X1) * (1 - X2) / rho_L))

    def recombine(self, sep, gor):  # spec: SMP-113
        """The stream of the separator gas and stock-tank oil of sep at the GOR gor: per Sm3 of oil, rho_o/M_o kmol of
        oil and gor/V_M_SC kmol of gas."""
        n_o, n_g = sep['rho_o'] / sep['M_o'], gor / V_M_SC
        z = n_g * np.array(sep['y']) + n_o * np.array(sep['x'])
        return z / z.sum()

    def saturation(self, z, T, p_hi, step=10.0):  # spec: SMP-115
        """The upper saturation pressure (bar) at T below p_hi, by TP flashes every step from 1 bar and bisection to
        1e-3 bar on the edge of the two-phase region; its kind ('bubble' or 'dew', or 'above_grid' where the stream
        has two phases at the last pressure); and the last two-phase flash below it. None if the stream has two phases
        nowhere on (1 bar, p_hi)."""
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
        kind = 'bubble' if phases[0][4] < phases[1][4] else 'dew'  # The incipient phase has the smaller mole fraction
        p_flash = float(lo)
        return dict(p=float(lo), kind=kind, lo=(phases, p_flash), p_flash=p_flash)


# --- The composition ----------------------------------------------------------------------------------------------------
def _example_stream(name, alpha=1.0):  # spec: SMP-114
    """An example fluid's heptanes-plus fractions [(name, M, specific gravity)] and its composition (mole fractions).
    A lumped heptanes-plus is split as a synthetic one's, with Soreide's Cf such that the fractions' volume-additive
    gravity is the measured one."""
    ex = EXAMPLES[name]
    if 'tbp' in ex:
        fr = [(f'F{i + 1}', M, g) for i, (_, M, g) in enumerate(ex['tbp'])]
        z7 = [x for x, _, _ in ex['tbp']]
    else:
        x7, M7, g7 = ex['c7']
        zf, Ms = _gamma_split(M7, alpha)
        lo, hi = 0.2, 0.45
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g = _soreide(Ms, mid)
            lo, hi = (mid, hi) if np.dot(zf, Ms) / np.dot(zf, Ms / g) < g7 else (lo, mid)
        g = _soreide(Ms, 0.5 * (lo + hi))
        fr = [(f'F{i + 1}', M, gi) for i, (M, gi) in enumerate(zip(Ms, g))]
        z7 = list(zf * x7)
    z = np.array(list(ex['light']) + z7)
    return fr, z / z.sum()


def _draw_composition(cls, rng):  # spec: SMP-113
    """The draws of a synthetic composition of class cls from the generator rng: the weights of the class's examples
    and their mixed light ends (normalised over N2 to C6), the heptanes-plus mole fraction z7 and molar mass M7, the
    gamma distribution's shape alpha, and the split's mole fractions and molar masses. The weights are
    Dirichlet(1, 1, 1) as exponential variates over their sum, each the inverse CDF of one uniform number, so that
    the draw takes the same count of the generator's numbers for every well (SMP-51); numpy's dirichlet does not."""
    c = CLASSES[cls]
    e = -np.log1p(-rng.random(len(c['templates'])))
    w = e / e.sum()
    light = sum(wi * np.array(EXAMPLES[t]['light']) / sum(EXAMPLES[t]['light']) for wi, t in zip(w, c['templates']))
    z7 = rng.uniform(*c['z7'])
    M7 = rng.uniform(*c['M7'])
    alpha = rng.uniform(0.5, 2.5)
    zf, Ms = _gamma_split(M7, alpha)
    return dict(w=w, light=light, z7=z7, M7=M7, alpha=alpha, zf=zf, Ms=Ms)


def _compose(draw, Cf, ln_f):  # spec: SMP-113
    """A drawn composition with its levers set: the heptanes-plus fractions [(name, M, specific gravity)] with
    Soreide's Cf, and the mole fractions with C2 to C6 scaled by exp(ln_f) and the light ends renormalised."""
    lt = draw['light'].copy()
    lt[INTERMEDIATE] *= math.exp(ln_f)
    z = np.concatenate([lt / lt.sum() * (1 - draw['z7']), draw['zf'] * draw['z7']])
    return [(f'F{i + 1}', M, g) for i, (M, g) in enumerate(zip(draw['Ms'], _soreide(draw['Ms'], Cf)))], z


def _synthetic(cls, rng, target, max_iter=8):  # spec: SMP-113
    """
    A synthetic composition of class cls matched to target's API gravity and separator-gas gravity, to 0.5 API and
    0.005, by Newton's method on Soreide's Cf and the logarithm of the factor on C2 to C6, and recombined at its R_si
    (exactly). Returns the flash, the stream, the separation and a record of the draw and the match.
    """
    c = CLASSES[cls]
    draw = _draw_composition(cls, rng)
    rec = dict(template_weights=dict(zip(c['templates'], draw['w'].round(4).tolist())))
    rec |= dict(z7=draw['z7'], M7=draw['M7'], alpha=draw['alpha'])

    def build(u):
        fractions, z = _compose(draw, *u)
        fl = _Flash(fractions)
        sep = fl.separation(z)
        return fl, z, sep, np.array([_api_of(sep['rho_o']) - target['api'], sep['rho_g'] / RHO_AIR - target['sg_gas']])

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


def _field_matched(w, passes=2):  # spec: SMP-113
    """
    The well's synthetic composition, matched so that the field reports the drawn values: matched as single-stage
    descriptors at standard conditions (_synthetic), then flashed through the well's separator and stock tank
    (_Flash.field), and the single-stage targets shifted by the difference, in passes. Each pass draws the same
    composition from the well's composition_seed. Returns _synthetic's four values, with the offsets and the field's
    descriptors in the record.
    """
    drawn = dict(api=w['api'], sg_gas=w['sg_gas'], R_si=w['R_si'])
    target = dict(drawn)
    for k in range(passes + 1):
        fl, z, sep, rec = _synthetic(w['fluid_class'], np.random.default_rng(w['composition_seed']), target)
        f = fl.field(z, w['p_sep'], w['T_sep'])
        if f is None:
            rec |= dict(field=None, offsets=None, passes=k)
            break
        single = dict(api=_api_of(sep['rho_o']), sg_gas=sep['rho_g'] / RHO_AIR, R_si=target['R_si'])
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
    if rec['field'] is None:
        rec['field_error'] = None
    else:
        rec['field_error'] = dict(
            api=rec['field']['api'] - drawn['api'],
            sg_gas=rec['field']['sg_gas'] - drawn['sg_gas'],
            gor=rec['field']['gor'] / drawn['R_si'] - 1,
        )
    return fl, z, sep, rec


def _example_matched(w, passes=2):  # spec: SMP-114
    """
    The published example w['source'] recombined along its own single-stage tie line, its GOR matched to the drawn one
    in the field as _field_matched does a synthetic fluid's: the stream flashed through the well's separator and a
    stock tank, and the single-stage GOR shifted by the difference, in passes. Its stock-tank oil and separator gas
    are the example's, so its API and gas gravity are its own. Returns _field_matched's four values.
    """
    fr, z0 = _example_stream(w['source'])
    fl = _Flash(fr)
    sep = fl.separation(z0)
    rec = dict(example=w['source'], reference=EXAMPLES[w['source']]['ref'])
    target = w['R_si']
    for k in range(passes + 1):
        z = fl.recombine(sep, target)
        f = fl.field(z, w['p_sep'], w['T_sep'])
        if f is None:
            rec |= dict(field=None, offsets=None, passes=k)
            break
        single = dict(api=_api_of(sep['rho_o']), sg_gas=sep['rho_g'] / RHO_AIR, R_si=target)
        offsets = dict(api=f['api'] - single['api'], sg_gas=f['sg_gas'] - single['sg_gas'], gor=f['gor'] / target)
        rec |= dict(field=f, offsets=offsets, single_stage=single, passes=k)
        if k == passes:
            break
        target = w['R_si'] / offsets['gor']
    rec['field_error'] = None if rec['field'] is None else dict(gor=rec['field']['gor'] / w['R_si'] - 1)
    return fl, z, sep, rec


# --- The saturation pressure --------------------------------------------------------------------------------------------
def _saturated_at(fl, sep, gor, T, p_i):  # spec: SMP-115
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


def build_fluid(w) -> dict:  # spec: SMP-115
    """
    A table well's fluid, from its record w of wells.json: the example or the synthetic composition, matched; its
    saturation pressure at T_r; and, where that lies above p_i, the GOR moved along the tie line until
    p_sat(T_r) = p_i, else the class's phase in equilibrium at (p_i, T_r). Returns the fluid's record: the labels,
    hook, the values that replace the fluid block's draws (the table's p_sat and initial GOR, its single-stage rho_o
    and rho_g, and the fluid's field-reported API and gas gravity where they are not the drawn ones), and stream, which
    build_tables reads (its fractions, its composition and its separation).
    """
    t0 = time.time()
    fl, z, sep, rec = _field_matched(w) if w['source'] == 'synthetic' else _example_matched(w)
    t_match = time.time() - t0
    gor = rec['single_stage']['R_si'] if rec.get('single_stage') else w['R_si']
    sat = fl.saturation(z, w['T_r'], 1.3 * w['p_i'])
    moved, rule = None, 'matched'
    if sat and sat['p'] > w['p_i']:  # It could not be one phase at p_i: saturated at discovery
        g = _saturated_at(fl, sep, gor, w['T_r'], w['p_i'])
        if g is not None:
            moved, gor, rule = g / gor, g, 'gor_moved'
            z = fl.recombine(sep, gor)
        else:  # No GOR on the line will do: the phase of the class in equilibrium at (p_i, T_r)
            phases, _ = fl.flash(z, w['p_i'], w['T_r'], properties=False)
            if len(phases) != 2:
                raise ValueError(f'p_sat {sat["p"]:.1f} bar above p_i, and one phase at (p_i, T_r)')
            z = np.array(phases[0 if w['fluid_class'] == 'gas_condensate' else 1][5])
            sep = fl.separation(z)
            moved, gor, rule = sep['gor'] / gor, sep['gor'], 'equilibrium_phase'
        sat = fl.saturation(z, w['T_r'], 1.3 * w['p_i'])
    t_sat = time.time() - t0 - t_match
    field = fl.field(z, w['p_sep'], w['T_sep'])
    # No two phases anywhere from 1 bar to 1.3 p_i at T_r: the fluid has no saturation pressure above 1 bar there
    p_sat, kind = (sat['p'], sat['kind']) if sat and sat['kind'] != 'above_grid' else (1.0, 'none')
    if sat and sat['kind'] == 'above_grid':
        raise ValueError(f'two phases at 1.3 p_i at T_r after the rule {rule}')
    hook = dict(p_sat=p_sat, R_i=gor, rho_o=sep['rho_o'], rho_g=sep['rho_g'])
    if (w['source'] != 'synthetic' or rule == 'equilibrium_phase') and field is not None:
        hook |= dict(api=field['api'], sg_gas=field['sg_gas'])
    return dict(
        well=w['well'],
        status='ok',
        fluid_class=w['fluid_class'],
        source=w['source'],
        composition=rec,
        saturation_kind=kind,
        saturation_rule=rule,
        gor_moved=moved,
        p_sat=p_sat,
        R_i=gor,
        field=field,
        single_stage=dict(api=_api_of(sep['rho_o']), sg_gas=sep['rho_g'] / RHO_AIR, gor=gor),
        c7_mol=float(np.sum(z[len(LIGHT) :])),
        c1_mol=float(z[2]),
        hook=hook,
        drawn={k: w[k] for k in ('api', 'sg_gas', 'R_si', 'p_sat')},
        stream=dict(
            fractions=fl.fractions,
            z=[float(v) for v in z],
            sep=dict(
                rho_o=sep['rho_o'],
                rho_g=sep['rho_g'],
                gor=sep['gor'],
                M_o=sep['M_o'],
                y=[float(v) for v in sep['y']],
                x=[float(v) for v in sep['x']],
            ),
        ),
        seconds=dict(match=t_match, saturation=t_sat),
    )


# --- The table ----------------------------------------------------------------------------------------------------------
def well_grid(T_s, T_r, p_i) -> tuple:  # spec: SMP-116
    """The well's grid: the temperatures (K), every 10 K from below T_s - 15 K to above T_r, and the top of the
    pressures (bar), 1.05 p_i rounded up to 10 bar, plus a step."""
    T_lo = 10 * math.floor((T_s - 15) / 10)
    T_hi = 10 * math.ceil((T_r + 1) / 10)
    return np.arange(T_lo, T_hi + 1e-9, 10.0), 10 * math.ceil(1.05 * p_i / 10) + 10


def _pressure_axis(p_top, p_sat=()):  # spec: SMP-116
    """The pressures of a table (bar): 1, 2.5, 5 and 7.5 bar, then every 10 bar to p_top, with the saturation
    pressures p_sat, rounded to 1e-3 bar, as nodes."""
    base = np.unique(np.r_[1.0, 2.5, 5.0, 7.5, np.arange(10.0, p_top + 1e-9, 10.0)])
    return np.unique(np.r_[base, [round(p, 3) for p in p_sat]])


def _tabulate(fl, z, sep_gor, T_axis, p_top):  # spec: SMP-116
    """
    The table of stream z (PVT-TAB-6) on T_axis and the pressures to p_top with every temperature's saturation
    pressure as a node, one row per node. Two-phase nodes from the flash, with the vapour's mass fraction X at most
    1 - DELTA; all-liquid nodes with the incipient vapour flashed at (p, T); all-vapour nodes with X = 1 - DELTA and
    the liquid of the nearest two-phase node of the temperature. sep_gor is the separation's rho_o and rho_g and the
    stream's GOR.
    """
    rho_o, rho_g, gor = sep_gor
    m = rho_o + gor * rho_g
    R_s = P_REF / (rho_g * T_REF)
    sat = {T: fl.saturation(z, T, p_top) for T in T_axis}
    p_axis = _pressure_axis(p_top, [s['p'] for s in sat.values() if s and s['kind'] != 'above_grid'])
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
                if liquid:  # Above a bubble point: no free gas; the gas is the incipient vapour
                    X, rho_L, mu_o = 0.0, ph[0][1], ph[0][3]
                    y_b = s['lo'][0][0][5]
                    gas, _ = fl.flash(y_b, p, T)
                    rho_V = sum(q[0] for q in gas) / sum(q[0] / q[1] for q in gas)
                    mu_g = gas[0][3]
                    sigma = flashes[near][2]
                else:  # All vapour: the liquid held at the nearest two-phase node's, or none at this temperature
                    X, rho_V, mu_g = 1.0, ph[0][1], ph[0][3]
                    if near is not None:
                        _, (_, rho_L, _, mu_o, _, _) = flashes[near][1]
                        sigma = flashes[near][2]
                    else:
                        rho_L, mu_o, sigma = math.nan, math.nan, math.nan
            X = min(X, 1 - DELTA)
            rows.append(dict(p=p, T=T, X=X, rho_L=rho_L, rho_V=rho_V, mu_o=mu_o, mu_g=mu_g, sigma=sigma))
    df = pd.DataFrame(rows)
    for col in ('rho_L', 'mu_o', 'sigma'):  # A temperature with no two-phase node takes its neighbour's
        df[col] = df.groupby('p')[col].transform(lambda c: c.ffill().bfill())
    df['R_so'] = gor - df.X * m / rho_g
    df['B_o'] = (1 - df.X) * m / df.rho_L
    df['Z'] = CF_BAR * df.p / (df.rho_V * R_s * df['T'])
    return df, sat


def _falling_pz_pressures(df) -> list:  # spec: SMP-116
    """The pressures to drop from a table, one row per node with the columns p, T and Z, so that p/Z rises strictly
    with p at every temperature (PVT-TAB-1): at each temperature the nodes off the longest run along which p/Z rises
    (of the longest, the one that ends highest, so that a spike goes), dropped at every temperature, so that the grid
    stays rectangular. They are artefacts of the flash near a hot fluid's near-critical bubble point."""
    drop = set()
    for _, g in df.sort_values('p').groupby('T'):
        pz = (g.p / g.Z).to_numpy()
        n = len(pz)
        best, prev = [1] * n, [-1] * n
        for i in range(n):
            for j in range(i):
                if pz[j] < pz[i] and best[j] + 1 > best[i]:
                    best[i], prev[i] = best[j] + 1, j
        i, keep = n - 1 - int(np.argmax(best[::-1])), set()
        while i >= 0:
            keep.add(i)
            i = prev[i]
        drop |= {float(p) for i, p in enumerate(g.p) if i not in keep}
    return sorted(drop)


def _repair(df, gor) -> tuple:  # spec: SMP-116
    """A table of one GOR without the pressures at which p/Z does not rise (_falling_pz_pressures), and those
    pressures. A table that would lose more than MAX_DROPPED of its pressures fails, with a ValueError."""
    bad = _falling_pz_pressures(df)
    if len(bad) > MAX_DROPPED * df.p.nunique():
        raise ValueError(f'p/Z falls at {len(bad)} of {df.p.nunique()} pressures at GOR {gor:.6g}')
    return (df[~df.p.isin(bad)] if bad else df), bad


def _write_tables(path, w, fluid, frames):  # spec: SMP-116
    """A well's table set as one CSV file in PVT-TAB-1's format, with a gor column (PVTTableSet's): frames holds one
    frame per GOR, one row per node, with the columns gor, p, T, R_so, B_o, Z, mu_o, mu_g and sigma. The header gives
    the stream's rho_o and rho_g to the bit, and the fluid's source. Returns the nodes written."""
    df = pd.concat(frames, ignore_index=True)
    src = fluid['source'] if fluid['source'] != 'synthetic' else f'synthetic, draw index {w["well"]}'
    sep = fluid['stream']['sep']
    header = [
        '# ManyWells PVT table (specs/model/pvt/table.md)',
        f'# name: k{w["well"]:06d}_{w["fluid_class"]}',
        f'# source: {SOURCE}; NeqSim {NEQSIM_VERSION}, Peng-Robinson with Peneloux; {src}',
        '# separation: single stage at 1.01325 bar and 288.15 K',
        f'# rho_o: {sep["rho_o"]!r}',
        f'# rho_g: {sep["rho_g"]!r}',
    ]
    with open(path, 'w') as fh:
        fh.write('\n'.join(header) + '\n')
        df[['gor', 'p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma']].to_csv(fh, index=False, float_format='%.10g')
    return len(df)


def build_tables(w, fluid, axis, path) -> tuple:  # spec: SMP-116
    """The well's table set at the GORs of its axis, each on the well's own grid with its saturation pressures as
    nodes (_tabulate) and repaired (_repair), written to path (_write_tables). Returns the nodes written and the
    pressures dropped, by GOR."""
    fl = _Flash(fluid['stream']['fractions'])
    sep = fluid['stream']['sep']
    T_axis = np.array(w['T_axis'], dtype=float)
    frames, dropped = [], {}
    for g in axis['gors']:
        df, _ = _tabulate(fl, fl.recombine(sep, g), (sep['rho_o'], sep['rho_g'], g), T_axis, w['p_top'])
        df, bad = _repair(df, g)
        if bad:
            dropped[f'{g:.6g}'] = bad
        frames.append(df.assign(gor=g))
    return _write_tables(path, w, fluid, frames), dropped


# --- The build ----------------------------------------------------------------------------------------------------------
def _write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def build_task(task) -> tuple:  # spec: SMP-116
    """One well's next step of the build: its fluid, or, once its axis exists, its table set and manifest. task is
    (DIR, the well's record with the dataset's kind and seed, its axis or None). A step that fails is written with the
    status 'failed', its step and its error. Returns the draw index, the step ('fluid', 'tables' or 'skip'), the
    status and the seconds taken."""
    outdir, w, axis = task
    outdir, k = Path(outdir), w['well']
    fpath, mpath = outdir / 'fluids' / f'k{k:06d}.json', outdir / f'k{k:06d}.json'
    t0 = time.time()
    if not fpath.exists():
        try:
            rec = build_fluid(w)
        except MemoryError:
            raise  # The process's, not the fluid's: nothing is recorded, and a rerun builds it
        except Exception:  # noqa: BLE001 - a failed fluid is recorded, and its well is not solved
            rec = dict(well=k, status='failed', step='fluid', error=traceback.format_exc()[-1500:])
        rec['seconds'] = rec.get('seconds', {}) | dict(fluid=time.time() - t0)
        _write_json(fpath, rec)
        return k, 'fluid', rec['status'], time.time() - t0
    fluid = json.loads(fpath.read_text())
    if mpath.exists() or (axis is None and fluid['status'] == 'ok'):
        return k, 'skip', fluid['status'], 0.0
    man = {key: v for key, v in fluid.items() if key != 'stream'} | dict(kind=w['kind'], seed=w['seed'])
    if fluid['status'] == 'ok' and 'error' in axis:
        man |= dict(status='failed', step='axis', error=axis['error'])
    elif fluid['status'] == 'ok':
        try:
            nodes, dropped = build_tables(w, fluid, axis, outdir / f'k{k:06d}.csv')
            man |= dict(csv=f'k{k:06d}.csv', gors=axis['gors'], nodes=nodes, n_T=len(w['T_axis']), p_top=w['p_top'])
            man['dropped_pressures'] = dropped
        except MemoryError:
            raise
        except Exception:  # noqa: BLE001 - a failed table set is recorded, and its well is not solved
            man |= dict(status='failed', step='tables', error=traceback.format_exc()[-1500:])
    man['seconds'] = fluid['seconds'] | dict(tables=time.time() - t0)
    _write_json(mpath, man)
    return k, 'tables', man['status'], time.time() - t0


def build(outdir, processes=4):  # spec: SMP-116
    """The next step of every well in DIR/wells.json, in a pool of processes: the fluids, and, once DIR/axes.json
    exists, the table sets of its wells. NeqSim must be the version the tables are specified with."""
    try:
        found = metadata.version('neqsim')
    except metadata.PackageNotFoundError:
        found = 'none'
    if found != NEQSIM_VERSION:
        raise RuntimeError(f'the tables are made with NeqSim {NEQSIM_VERSION}, in its own environment; found {found}')
    outdir = Path(outdir)
    (outdir / 'fluids').mkdir(parents=True, exist_ok=True)
    spec = json.loads((outdir / 'wells.json').read_text())
    axes = json.loads((outdir / 'axes.json').read_text()) if (outdir / 'axes.json').exists() else {}
    tasks = [
        (str(outdir), w | dict(kind=spec['kind'], seed=spec['seed']), axes.get(str(w['well']))) for w in spec['wells']
    ]
    t0, done = time.time(), {}
    with multiprocessing.get_context('forkserver').Pool(processes) as pool:
        for k, step, status, sec in pool.imap_unordered(build_task, tasks):
            done[k] = (step, status)
            if step != 'skip':
                print(f'well {k}: {step} {status} in {sec:.1f} s', flush=True)
    steps = pd.Series([f'{a} {b}' for a, b in done.values()]).value_counts().to_dict()
    print(f'{len(tasks)} wells in {time.time() - t0:.0f} s on {processes} processes: {steps}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outdir', help='the folder with wells.json, and axes.json once the fluids are built')
    ap.add_argument('--processes', type=int, default=4, help='each holds a JVM, about 1 GB')
    args = ap.parse_args()
    build(args.outdir, args.processes)


if __name__ == '__main__':
    main()
