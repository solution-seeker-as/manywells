"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Wells and checks for comparing the Rust core with the CasADi backend (plans/manywells-v2-plan.md, Step 9;
specs/features/015-rust-develop-model.md). develop's model has no reference root sets, so the core is checked
against the CasADi backend, which is independent code:

- the rows at the same states, in every configuration of the matrix (test_backend_comparison.py, fast);
- the root sets on the comparison set, wells drawn with the ported sampler with each option switched on by an
  overlay (test_backend_comparison.py, slow; scripts/verification/compare_backends.py for larger runs).

A configuration is a base well with overlays, each of which switches one option on or off. The sampler draws
deviated and L-shaped wells, black oil, real gas and friction from roughness (SMP-40 to SMP-44), but not the others,
so the overlays add them: lift-gas temperature, a fixed rate, PI inflow, a Bernoulli choke, and the options of
develop's model switched off one at a time.

This module is a helper, not a test module: test_backend_comparison.py and the script import it.
"""

import time
import traceback
import warnings
from dataclasses import dataclass, field, replace

import casadi as ca
import numpy as np

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import DEVELOP, V1, v1_well
from manywells.discretization import DIM_X, build_system
from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.pvt import api_from_density, density_from_api
from manywells.sampling.conditions import sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.slip import SlipModel
from manywells.solvers.roots import TOL_X, state_distance

ROW_REL = 1e-10       # rows at the same state: the same arithmetic, to rounding amplified by the rows' cancellations
ROOT_ROW = 1e-8       # a core-only root zeroes every CasADi row but the choke row to this (bar, K, kg/(m² s), m/s)
ROOT_CHOKE_REL = 1e-6  # and the choke row to this fraction of the wellhead rate, or to what p_0's resolution allows,
ROOT_P0_ULPS = 8       # this many ulp of p_r times |dR/dp_0|, if more (steep at the trickle roots next to p_r)
SEED = 20261002       # the comparison set's seed (SMP-31)
FEATURES = ('001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011')


def features(wp) -> frozenset:
    """The feature specs (specs/features/) whose options a well uses beyond the v1.0.0 configuration's."""
    geo, fluid, thermal, out = wp.geometry, wp.fluid, wp.thermal, set()
    if not np.allclose(geo.delta_md, geo.L / geo.n_cells, rtol=1e-9, atol=0):
        out.add('001')
    if not np.allclose(geo.cos_incl, 1.0, rtol=0, atol=1e-12):
        out |= {'001', '002'}
    if wp.slip != SlipModel():
        out.add('002')
    if fluid.surface_tension_model != 'liquid':
        out.add('003')
    if type(wp.friction) is RoughnessFriction:
        out.add('004')
    if not fluid.ideal_gas:
        out.add('006')
    if fluid.oil_model == 'black_oil':
        out |= {'007', '008'}
    if thermal.frictional_heating or thermal.gravity_term:
        out.add('009')
    if thermal.lift_gas_mixing:
        out.add('010')
    if type(wp.inflow) is FixedFlowRate:
        out.add('011')
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# Overlays: one option each, as fn(wp, bc) -> (wp, bc)

def _survey(md, tvd, wp):
    return WellGeometry.from_survey(md, tvd, n_cells=wp.geometry.n_cells, D=wp.geometry.D)


def _depth(wp):
    return wp.geometry.tvd[0]


def non_uniform(wp, bc):
    """A vertical well on a grid whose cells grow towards the bottomhole."""
    s = np.linspace(0.0, 1.0, wp.geometry.n_cells + 1) ** 1.3 * _depth(wp)
    return replace(wp, geometry=WellGeometry(md_survey=s, tvd_survey=s, D=wp.geometry.D)), bc


def deviated(wp, bc, kickoff=0.3, theta=40.0):
    """Vertical to the kickoff, then straight at theta degrees to the same true vertical depth."""
    L = _depth(wp)
    k = kickoff * L
    return replace(wp, geometry=_survey([0.0, k, k + (L - k) / np.cos(np.radians(theta))], [0.0, k, L], wp)), bc


def l_shaped(wp, bc, horizontal=600.0):
    """Vertical to the bottomhole's depth, then horizontal."""
    L = _depth(wp)
    return replace(wp, geometry=_survey([0.0, L, L + horizontal], [0.0, L, L], wp)), bc


def fluid(**kw):
    return lambda wp, bc: (replace(wp, fluid=replace(wp.fluid, **kw)), bc)


def black_oil(wp, bc):
    """Black oil with dissolved gas; an oil outside the correlations' API range of 10 to 40 becomes API 30."""
    rho_o = wp.fluid.rho_o if 10 <= api_from_density(wp.fluid.rho_o) <= 40 else density_from_api(30.0)
    return replace(wp, fluid=replace(wp.fluid, oil_model='black_oil', rho_o=rho_o)), bc


def bubble_point(wp, bc):
    """Black oil whose solution gas-oil ratio is capped at a bubble point halfway between p_s and p_r."""
    wp, bc = black_oil(wp, bc)
    return replace(wp, fluid=replace(wp.fluid, p_bubble=bc.p_s + 0.5 * (bc.p_r - bc.p_s))), bc


def thermal(**kw):
    return lambda wp, bc: (replace(wp, thermal=replace(wp.thermal, **kw)), bc)


def lift_gas_temperature(wp, bc):
    """Lift gas at least 1 kg/s, injected cold (30% of the way from T_s to T_r), mixing with the inflow."""
    wp = replace(wp, thermal=replace(wp.thermal, lift_gas_mixing=True))
    return wp, replace(bc, w_lg=max(bc.w_lg, 1.0), T_lg=bc.T_s + 0.3 * (bc.T_r - bc.T_s))


def fixed_rate(wp, bc):
    """The liquid rate the inflow gives at 60% of the way from p_s to p_r, fixed."""
    w = float(wp.inflow.liquid_mass_flow_rate(bc.p_s + 0.6 * (bc.p_r - bc.p_s), bc.p_r))
    return replace(wp, inflow=FixedFlowRate(w_l_const=w)), bc


def productivity_index(wp, bc):
    """Linear inflow with the same rate as the inflow at half the drawdown."""
    d = 0.5 * (bc.p_r - bc.p_s)
    k_l = float(wp.inflow.liquid_mass_flow_rate(bc.p_r - d, bc.p_r)) / d
    return replace(wp, inflow=ProductivityIndex(k_l=k_l)), bc


def vogel(wp, bc):
    """Vogel inflow with the same rate as the inflow at half the drawdown."""
    p = bc.p_s + 0.5 * (bc.p_r - bc.p_s)
    r = p / bc.p_r
    w_l_max = float(wp.inflow.liquid_mass_flow_rate(p, bc.p_r)) / (1 - 0.2 * r - 0.8 * r ** 2)
    return replace(wp, inflow=Vogel(w_l_max=w_l_max)), bc


def choke(model):
    return lambda wp, bc: (replace(wp, choke=model(K_c=wp.choke.K_c, chk_profile=wp.choke.chk_profile)), bc)


OVERLAYS = {
    'non-uniform grid': non_uniform,
    'deviated': deviated,
    'L-shaped': l_shaped,
    'slip constants': lambda wp, bc: (replace(wp, slip=SlipModel(C_0_annular=1.02, C_0_slug=1.15, C_0_bubbly=1.25,
                                                                 v_inf_annular=0.02)), bc),
    'water': fluid(wlr=0.3),
    'frictional heating': thermal(frictional_heating=True),
    'gravity term': thermal(gravity_term=True),
    'energy terms': thermal(frictional_heating=True, gravity_term=True),
    'real gas': fluid(ideal_gas=False),
    'black oil': black_oil,
    'bubble point': bubble_point,
    'oil surface tension': fluid(surface_tension_model='oil'),
    'chen': lambda wp, bc: (replace(wp, friction=RoughnessFriction(roughness=4.5e-5, correlation='chen')), bc),
    'haaland': lambda wp, bc: (replace(wp, friction=RoughnessFriction(roughness=4.5e-5, correlation='haaland')), bc),
    'lift-gas temperature': lift_gas_temperature,
    'fixed rate': fixed_rate,
    'productivity index': productivity_index,
    'vogel': vogel,
    'bernoulli': choke(BernoulliChokeModel),
    'simpson': choke(SimpsonChokeModel),
    # develop's options switched off one at a time
    'dead oil': fluid(oil_model='dead_oil', p_bubble=None),
    'ideal gas': fluid(ideal_gas=True),
    'liquid surface tension': fluid(surface_tension_model='liquid'),
    'fixed f_D': lambda wp, bc: (replace(wp, friction=FixedFrictionFactor(f_D=0.03)), bc),
    'no frictional heating': thermal(frictional_heating=False),
    'no gravity term': thermal(gravity_term=False),
    'no lift-gas mixing': thermal(lift_gas_mixing=False),
}


# ---------------------------------------------------------------------------------------------
# The configuration matrix of the row comparison: small wells, built without the sampler

def _row_vector_well(k, n_cells):
    """Well W1 or W2 of specs/model/vectors/v1_rows.json, the v1.0.0 wells of the row vectors, on n_cells."""
    from .spec_parse import row_vectors
    w = row_vectors()['wells'][k]['params']
    inflow = Vogel(w['w_l_max']) if w['inflow'] == 'vogel' else ProductivityIndex(w['k_l'])
    model = SimpsonChokeModel if w['choke'] == 'simpson' else BernoulliChokeModel
    wp = v1_well(L=w['L'], D=w['D'], rho_l=w['rho_l'], R_s=w['R_s'], cp_g=w['cp_g'], cp_l=w['cp_l'], f_D=w['f_D'],
                 h=w['h'], f_g=w['f_g'], inflow=inflow, choke=model(K_c=w['K_c'], chk_profile=w['profile']),
                 n_cells=n_cells)
    return wp, BoundaryConditions(p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], u=w['u'], w_lg=w['w_lg'])


BASES = {
    'W1': lambda n: _row_vector_well(0, n),   # Vogel, Simpson choke with the sigmoid profile, lift gas
    'W2': lambda n: _row_vector_well(1, n),   # productivity index, Bernoulli choke, choked at its root
    'develop': lambda n: (WellProperties(geometry=WellGeometry.vertical(length=2000.0, n_cells=n)),
                          BoundaryConditions(p_r=200.0, p_s=20.0, u=0.6, w_lg=0.5)),
}

V1_OVERLAYS = ('non-uniform grid', 'deviated', 'L-shaped', 'slip constants', 'water', 'frictional heating',
               'gravity term', 'energy terms', 'real gas', 'black oil', 'bubble point', 'oil surface tension', 'chen', 'haaland',
               'lift-gas temperature', 'fixed rate')
DEVELOP_OVERLAYS = ('dead oil', 'ideal gas', 'liquid surface tension', 'fixed f_D', 'haaland', 'no frictional heating',
                    'no gravity term', 'no lift-gas mixing', 'deviated', 'L-shaped', 'non-uniform grid',
                    'slip constants', 'water', 'bubble point', 'lift-gas temperature', 'fixed rate', 'vogel', 'simpson')


@dataclass(frozen=True)
class Configuration:
    base: str
    overlays: tuple = ()

    @property
    def name(self) -> str:
        return '+'.join((self.base,) + self.overlays)

    def inputs(self, n_cells=10):
        """(wp, bc) of the configuration on its base well of the matrix"""
        wp, bc = BASES[self.base](n_cells)
        for o in self.overlays:
            wp, bc = OVERLAYS[o](wp, bc)
        return wp, bc


def matrix() -> list:
    out = []
    for base in ('W1', 'W2'):
        out += [Configuration(base)] + [Configuration(base, (o,)) for o in V1_OVERLAYS]
    out += [Configuration('develop')] + [Configuration('develop', (o,)) for o in DEVELOP_OVERLAYS]
    out += [Configuration('develop', ('deviated', 'fixed rate')), Configuration('develop', ('L-shaped', 'haaland'))]
    return out


def perturb(X):
    """A few per cent off a state, deterministic, alpha kept inside (0, 1) (make_v1_vectors.perturb)."""
    s = np.sin(np.arange(X.size).reshape(X.shape) + 1.0)
    Y = X * (1 + 0.05 * s)
    Y[:, 3] = np.clip(X[:, 3] + 0.02 * s[:, 3], 0.01, 0.99)
    return Y


def trial_state(wp, bc) -> np.ndarray:
    """
    A state far from every root but physically plausible, so that every row is far from zero: the pressure falling
    from 80% to 30% of the way from p_s to p_r, the temperature from the inflow's halfway to T_s, the void fraction
    from 0.15 to 0.85 across the three regimes, the densities and the velocities that carry the phase rates at each
    point, then perturbed.
    """
    system, fl, A = build_system(wp), wp.fluid, wp.geometry.A
    params = system.params(bc)
    n = wp.geometry.n_cells + 1
    d = bc.p_r - bc.p_s
    p = np.linspace(bc.p_s + 0.8 * d, bc.p_s + 0.3 * d, n)
    T_in = float(np.asarray(system.bottom_guess(p[0], params)).ravel()[6])
    T = np.linspace(T_in, 0.5 * (T_in + bc.T_s), n)
    alpha = np.linspace(0.15, 0.85, n)
    w_res = float(system.reservoir_rate(p[0], params))
    X = np.zeros((n, DIM_X))
    for i in range(n):
        rho_g, rho_l = float(fl.gas_density(p[i], T[i])), float(fl.liquid_density(p[i], T[i]))
        w_g, w_l = (float(ca.DM(w)) for w in fl.phase_rates(p[i], T[i], w_res, bc.w_lg))
        X[i] = p[i], w_g / (A * alpha[i] * rho_g), w_l / (A * (1 - alpha[i]) * rho_l), alpha[i], rho_g, rho_l, T[i]
    return perturb(X)


def casadi_rows(system, bc, x) -> tuple:
    return system.row_ids, np.asarray(system.residual(np.ravel(x), system.params(bc))).ravel()


def row_difference(ids, rust, casadi) -> tuple:
    """The largest relative difference between two backends' rows, and where: (difference, index, ID)."""
    rel = np.abs(rust - casadi) / np.maximum(np.abs(casadi), np.finfo(float).tiny)
    k = int(np.argmax(rel))
    return float(rel[k]), k, ids[k]


# ---------------------------------------------------------------------------------------------
# The comparison set: sampled wells with overlays

COMPARISON = {
    V1: ('', 'deviated', 'L-shaped', 'real gas', 'black oil', 'bubble point', 'oil surface tension', 'chen',
         'haaland', 'frictional heating', 'gravity term', 'lift-gas temperature', 'fixed rate'),
    DEVELOP: ('', 'dead oil', 'ideal gas', 'liquid surface tension', 'haaland', 'fixed f_D', 'no frictional heating',
              'no gravity term', 'lift-gas temperature', 'fixed rate', 'productivity index', 'bernoulli',
              'bubble point'),
}


@dataclass(frozen=True)
class Case:
    """One sampled well and operating point of the comparison set, with an overlay (or none)."""
    configuration: str      # V1 or DEVELOP, the sampler's configuration of the base well (SMP-40)
    overlay: str            # one of COMPARISON[configuration]; '' for none
    well: int
    seed: int = SEED
    n_cells: int = 100

    @property
    def group(self) -> str:
        return f'{self.configuration}{"+" + self.overlay if self.overlay else ""}'

    @property
    def name(self) -> str:
        return f'{self.group}#{self.well}'

    def inputs(self):
        """(wp, bc): the well's draws and one stationary sample (SMP-18 to SMP-22), then the overlay."""
        draw = sample_well(self.seed, self.well)
        bc, fractions = sample_conditions(draw, rng_for(self.seed, self.well, 0, 'sample'))
        wp = well_properties(draw, fractions, self.configuration, self.n_cells)
        return OVERLAYS[self.overlay](wp, bc) if self.overlay else (wp, bc)


def comparison_set(per_configuration=2, seed=SEED, configurations=None) -> list:
    """
    The comparison set: per_configuration wells for each overlay of each base configuration, from the well indices
    whose draws the configuration can take (the develop configuration needs oil, SMP-43). Well indices are taken in
    order, so a larger set contains the smaller.
    """
    out = []
    for conf, overlays in COMPARISON.items():
        for overlay in overlays:
            name = f'{conf}{"+" + overlay if overlay else ""}'
            if configurations and name not in configurations:
                continue
            k, found = 0, 0
            while found < per_configuration:
                case = Case(conf, overlay, k, seed)
                try:
                    case.inputs()
                    out.append(case)
                    found += 1
                except ValueError:
                    pass
                k += 1
    return out


@dataclass
class Comparison:
    """The comparison of the two backends' root sets for one case."""
    case: str
    casadi: list = field(default_factory=list)       # p_0 and label of each CasADi root
    rust: list = field(default_factory=list)         # p_0 and label of each core root
    missed: list = field(default_factory=list)       # CasADi roots the core does not find, by p_0
    labels: list = field(default_factory=list)       # matched roots whose labels differ, by p_0
    core_only: list = field(default_factory=list)    # core roots no CasADi root matches: (p_0, zeroes the rows)
    several_alpha: bool = False                      # the slip law has several void fractions at a root's point
    row_rel: float = 0.0                             # largest relative row difference at the perturbed roots
    row_at: str = ''
    seconds: dict = field(default_factory=dict)      # build + search per backend
    marches: int = 0                                 # the core's marches
    counts: dict = field(default_factory=dict)       # the core's work (march.rs, Counts)
    choked_differs: int = 0                          # matched roots whose CHOKED flags differ (information)
    error: str = ''                                  # a backend raised

    @property
    def ok(self) -> bool:
        """The pass rule (Step 9): every CasADi root found by the core with its label, every core-only root a root of
        the CasADi rows, and the rows equal at the perturbed roots; where the slip law has several void fractions,
        the rows only."""
        rows = self.row_rel <= ROW_REL and all(zeroes for _, zeroes in self.core_only)
        if self.error:
            return False
        if self.several_alpha:
            return rows
        return rows and not self.missed and not self.labels


def void_fractions(slip, D):
    """h(α) = α (C_0 j_m + v_inf) - j_g of the slip law at fixed superficial velocities, as a CasADi function of
    (α, j_g, j_l, rho_g, rho_l, sigma, cos_incl)."""
    alpha, j_g, j_l, rho_g, rho_l, sigma, cos = (ca.SX.sym(n) for n in ('a', 'jg', 'jl', 'rg', 'rl', 's', 'c'))
    C_0, v_inf = slip.identify_parameters(j_g / alpha, j_l / (1 - alpha), alpha, rho_g, rho_l, sigma, D, cos)
    return ca.Function('h', [alpha, j_g, j_l, rho_g, rho_l, sigma, cos], [alpha * (C_0 * (j_g + j_l) + v_inf) - j_g])


def several_void_fractions(wp, x, grid=2001) -> bool:
    """Whether the slip law has more than one void fraction, at fixed superficial velocities, at a point of state x."""
    X = np.reshape(x, (-1, DIM_X))
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    sigma = np.array([float(wp.fluid.surface_tension(p[i], T[i], rho_l[i])) for i in range(len(X))])
    cos = np.array([wp.geometry.cos_incl[max(i - 1, 0)] for i in range(len(X))])
    inputs = np.vstack([alpha * v_g, (1 - alpha) * v_l, rho_g, rho_l, sigma, cos])
    a = np.linspace(1e-9, 1 - 1e-9, grid)
    h = void_fractions(wp.slip, wp.geometry.D).map(grid * len(X))
    values = np.asarray(h(np.repeat(a, len(X)).reshape(1, -1), *(np.tile(inputs, grid)[k].reshape(1, -1)
                                                                  for k in range(6)))).reshape(grid, -1)
    return bool(np.any(np.sum(np.diff(np.sign(values), axis=0) != 0, axis=0) > 1))


def compare_root_sets(wp, bc, system, casadi_roots, rust_roots, rust_rows) -> Comparison:
    """Compare the backends' roots for one well; rust_rows(x) gives the core's rows at x."""
    c = Comparison(case='')
    c.casadi = [(r.p_0, r.label) for r in casadi_roots]
    c.rust = [(r.p_0, r.label) for r in rust_roots]
    matched = set()
    for cr in casadi_roots:
        k = next((k for k, rr in enumerate(rust_roots) if state_distance(rr.x, cr.x, bc) <= TOL_X), None)
        if k is None:
            c.missed.append(cr.p_0)
            continue
        matched.add(k)
        if rust_roots[k].label != cr.label:
            c.labels.append(cr.p_0)
        c.choked_differs += rust_roots[k].choked != cr.choked
    for k, rr in enumerate(rust_roots):
        if k in matched:
            continue
        ids, rows = casadi_rows(system, bc, rr.x)
        ids = np.array(ids)
        p, v_g, v_l, alpha, rho_g, rho_l, T = rr.state[-1]
        w_m = wp.geometry.A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
        slope = abs(rr.slope) * max(w_m, 1e-3) / (bc.p_r - bc.p_s)  # dR/dp_0 (kg/s per bar), undoing normalized_slope
        choke_bound = max(ROOT_CHOKE_REL * w_m, ROOT_P0_ULPS * np.finfo(float).eps * bc.p_r * slope)
        zeroes = (np.max(np.abs(rows[ids != 'CHK-1'])) < ROOT_ROW and abs(rows[ids == 'CHK-1'][0]) <= choke_bound)
        c.core_only.append((rr.p_0, bool(zeroes)))
    roots = list(casadi_roots) + list(rust_roots)
    c.several_alpha = any(several_void_fractions(wp, r.x) for r in roots)
    for r in roots:
        X = perturb(r.state)
        ids, casadi = casadi_rows(system, bc, X)
        rel, k, eq_id = row_difference(ids, rust_rows(X), casadi)
        if rel > c.row_rel:
            c.row_rel, c.row_at = rel, f'{eq_id} at row {k} of the root at {r.p_0:.4f} bar'
    return c


def compare_case(case: Case) -> Comparison:
    """Solve one case with both backends, timed, and compare their root sets."""
    try:
        wp, bc = case.inputs()
        t0 = time.perf_counter()
        casadi = SSDFSimulator(wp)
        casadi_roots = casadi.root_set(bc).roots
        t1 = time.perf_counter()
        rust = SSDFSimulator(wp, backend='rust')
        rs = rust.root_set(bc)
        t2 = time.perf_counter()
        c = compare_root_sets(wp, bc, casadi.system, casadi_roots, rs.roots,
                              lambda x: rust._roots.rows(bc, x)[1])
        c.seconds = {'casadi': t1 - t0, 'rust': t2 - t1}
        c.marches, c.counts = rs.search[0].iterations, dict(rust._roots.counts)
    except Exception as e:
        c = Comparison(case='', error=f'{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}')
    c.case = case.name
    return c


def report(comparisons) -> None:
    """Warn about what does not fail the pass rule but is worth Bjarne's look: core-only roots and the cases with
    several void fractions."""
    for c in comparisons:
        if c.core_only:
            warnings.warn(f'{c.case}: the core finds roots the CasADi search misses, at p_0 = '
                          + ', '.join(f'{p:.4f} bar' for p, _ in c.core_only))
        if c.several_alpha:
            warnings.warn(f'{c.case}: the slip law has several void fractions at a root; compared on rows only')
