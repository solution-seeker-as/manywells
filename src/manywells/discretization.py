"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The discretized system of a well (specs/model/discretization.md): the rows at each grid point, in the order of
DISC-11, built once per well with the operating point as parameters.

The pipe is discretized into N cells (see WellGeometry). The state x holds seven variables at each of the N + 1
grid points, from the bottomhole (point 0) to the wellhead (point N), in this order:
    [p, v_g, v_l, alpha, rho_g, rho_l, T]
(bar, m/s, m/s, -, kg/m³, kg/m³, K). Cell i lies between points i - 1 and i. Each row is defined once, below, and
build_system assembles it into the full residual and into the per-point functions the initial-guess march solves.
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from manywells.slip import regime_label
from manywells.units import CF_BAR, STD_GRAVITY

JT_T_PR_MIN = 1.05
"""The pseudo-reduced temperature of the solver's lower temperature bound with the Joule-Thomson term (System.bounds)."""

DIM_X = 7
STATE = ('p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T')  # Order is important
PARAMS = ('p_r', 'p_s', 'T_r', 'T_s', 'T_lg', 'u', 'w_lg')   # The operating point, as the system's parameters


@dataclass(frozen=True)
class PointState:
    """The seven state values at one grid point, symbolic or numeric."""
    p: object
    v_g: object
    v_l: object
    alpha: object
    rho_g: object
    rho_l: object
    T: object

    @classmethod
    def of(cls, x):
        return cls(*(x[k] for k in range(DIM_X)))

    @property
    def rho_m(self):  # spec: BAL-7
        """Mixture density (kg/m³)."""
        return self.alpha * self.rho_g + (1 - self.alpha) * self.rho_l

    @property
    def v_m(self):  # spec: BAL-8
        """Mixture velocity (m/s)."""
        return self.alpha * self.v_g + (1 - self.alpha) * self.v_l

    @property
    def gas_flux(self):
        """Gas mass flux (kg/m²/s)."""
        return self.alpha * self.rho_g * self.v_g

    @property
    def liquid_flux(self):
        """Liquid mass flux (kg/m²/s)."""
        return (1 - self.alpha) * self.rho_l * self.v_l

    @property
    def momentum_flux(self):
        """Momentum flux (Pa)."""
        return self.alpha * self.rho_g * self.v_g ** 2 + (1 - self.alpha) * self.rho_l * self.v_l ** 2


@dataclass(frozen=True)
class OperatingPoint:
    """The system's parameters (PARAMS), symbolic or numeric."""
    p_r: object
    p_s: object
    T_r: object
    T_s: object
    T_lg: object
    u: object
    w_lg: object

    @classmethod
    def of(cls, prm):
        return cls(*(prm[k] for k in range(len(PARAMS))))


# ---------------------------------------------------------------------------------------------
# Rows

def bottom_rows(wp, s, w_res, op):
    """Rows at the bottomhole, point 0: the inflow rows and the inflow temperature."""
    fluid, A = wp.fluid, wp.geometry.A
    w_g, w_l = fluid.phase_rates(s.p, s.T, w_res, op.w_lg)
    T_in = wp.thermal.inflow_temperature(w_res, op.w_lg, op.T_r, op.T_lg, fluid)
    return [
        A * s.alpha * s.rho_g * s.v_g - w_g,         # spec: INF-6
        A * (1 - s.alpha) * s.rho_l * s.v_l - w_l,   # spec: INF-7
        s.T - T_in,                                   # Inflow temperature (THM-3 or THM-5)
    ]


def cell_rows(wp, s, s_prev, w_res, op, delta_md, cos_incl, tvd_frac):
    """Balance rows of cell i, between points i - 1 (s_prev) and i (s)."""
    fluid, D, A = wp.fluid, wp.geometry.D, wp.geometry.A

    # Mass: the change in mass flux equals the change in the phase rates, which is zero without mass transfer
    w_g, w_l = fluid.phase_rates(s.p, s.T, w_res, op.w_lg)
    w_g_prev, w_l_prev = fluid.phase_rates(s_prev.p, s_prev.T, w_res, op.w_lg)
    r_g = s.gas_flux - s_prev.gas_flux - (w_g - w_g_prev) / A          # spec: DISC-7
    r_l = s.liquid_flux - s_prev.liquid_flux - (w_l - w_l_prev) / A    # spec: DISC-8

    # Momentum: friction along the flow path, gravity along the vertical
    F = wp.friction.pressure_gradient(s, fluid, D)
    G = STD_GRAVITY * s.rho_m  # spec: BAL-6
    r_p = (s.momentum_flux / CF_BAR + s.p) - (s_prev.momentum_flux / CF_BAR + s_prev.p) \
        + delta_md * (F + cos_incl * G) / CF_BAR                          # spec: DISC-9

    # Energy
    T_a = wp.thermal.ambient_temperature(tvd_frac, op.T_r, op.T_s)
    dp_dmd = CF_BAR * (s.p - s_prev.p) / delta_md
    dT_dmd = wp.thermal.temperature_gradient(s, fluid, T_a, F, dp_dmd, cos_incl, D)
    r_T = s.T - s_prev.T - delta_md * dT_dmd                              # spec: DISC-10

    return [r_g, r_l, r_p, r_T]


def choke_row(wp, s, op):  # spec: CHK-1
    """The wellhead row: the rate leaving the tubing passes through the choke."""
    A = wp.geometry.A
    w_m = A * s.alpha * s.rho_g * s.v_g + A * (1 - s.alpha) * s.rho_l * s.v_l
    return w_m - wp.choke.mass_flow_rate(op.u, op.p_s, s, A)


def closure_rows(wp, s, cos_incl):
    """Closure relations at a point: the slip law and the phase densities."""
    fluid = wp.fluid
    sigma = fluid.surface_tension(s.p, s.T, s.rho_l)
    C_0, v_inf = wp.slip.identify_parameters(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, wp.geometry.D, cos_incl)
    return [
        s.v_g - C_0 * s.v_m - v_inf,                 # spec: SLIP-1
        fluid.gas_law_row(s.p, s.T, s.rho_g),        # Gas law (PVT-GAS-1, PVT-GAS-3 or PVT-GAS-11)
        s.rho_l - fluid.liquid_density(s.p, s.T),    # Liquid density (PVT-MIX-1 or PVT-MIX-6)
    ]


def row_ids(wp) -> tuple:
    """The spec ID of every row of the system of wp, in order (DISC-11)."""
    fluid, N = wp.fluid, wp.geometry.n_cells
    bottom = ('INF-6', 'INF-7', 'THM-5' if wp.thermal.lift_gas_mixing else 'THM-3')
    cell = ('DISC-7', 'DISC-8', 'DISC-9', 'DISC-10')
    gas_law = 'PVT-GAS-1' if fluid.ideal_gas else 'PVT-GAS-11' if fluid.z_factor_model == 'dak' else 'PVT-GAS-3'
    closures = ('SLIP-1', gas_law,
                'PVT-MIX-1' if fluid.oil_model == 'dead_oil' else 'PVT-MIX-6')
    ids = bottom + closures
    for i in range(1, N + 1):
        ids += cell + (('CHK-1',) if i == N else ()) + closures
    return ids


# ---------------------------------------------------------------------------------------------
# The system of a well

@dataclass(frozen=True)
class System:
    """
    The discretized system of one well, with the operating point as parameters.

    Functions (CasADi), with params the parameter vector of params(bc):
        residual(x, params) -> r                 every row, in row_ids order
        bottom_rows(x_0, params) -> r_0          the six rows of point 0
        point_rows(x_i, x_prev, w_res, params, delta_md, cos_incl, tvd_frac) -> r_i
                                                 the seven rows of a point i > 0, without CHK-1
        cell_slope(x_i, x_prev, w_res, params, delta_md, cos_incl, tvd_frac) -> s
                                                 the slope of cell i's momentum row in p_i with the point's other
                                                 rows held at zero, positive where the cell is subsonic (SOL-8)
        reservoir_rate(p_0, params) -> w_res     the reservoir liquid rate (kg/s)
        bottom_guess(p_0, params) -> x_0         a starting point for point 0's rows at p_0
        regime_probabilities(x) -> P             the flow-regime probabilities [p_annular, p_slug, p_bubbly]
                                                 at each point, 3 x (N + 1)
        outputs(x, params) -> (choked, w_res, w_g_res)
                                                 the rest of what a root carries besides its state: CHOKED (CHK-12,
                                                 1 or 0) and the reservoir liquid and gas rates (kg/s)
    """
    wp: object
    n_x: int
    row_ids: tuple
    residual: ca.Function
    bottom_rows: ca.Function
    point_rows: ca.Function
    cell_slope: ca.Function
    reservoir_rate: ca.Function
    bottom_guess: ca.Function
    regime_probabilities: ca.Function
    outputs: ca.Function

    @property
    def n_cells(self) -> int:
        return self.wp.geometry.n_cells

    @staticmethod
    def params(bc) -> np.ndarray:
        """The parameter vector of an operating point, with T_lg = T_r when it is None."""
        T_lg = bc.T_r if bc.T_lg is None else bc.T_lg
        return np.array([bc.p_r, bc.p_s, bc.T_r, bc.T_s, T_lg, bc.u, bc.w_lg], dtype=float)

    def flow_regimes(self, x) -> tuple:
        """The flow-regime label at each point of a state (SLIP-8)."""
        P = np.asarray(self.regime_probabilities(x), dtype=float)
        return tuple(regime_label(P[:, i]) for i in range(P.shape[1]))

    def root_outputs(self, x, params) -> dict:
        """What a root carries besides its state, as Python values: flow_regime, choked, w_res and w_g_res."""
        choked, w_res, w_g_res = self.outputs(x, params)
        return {'flow_regime': self.flow_regimes(x), 'choked': bool(float(choked)), 'w_res': float(w_res),
                'w_g_res': float(w_g_res)}

    def subsonic(self, x, params) -> bool:  # spec: SOL-8
        """Whether every cell of state x is subsonic: its momentum row rises with p_i along the point's other rows."""
        geo, X = self.wp.geometry, np.reshape(x, (-1, DIM_X))
        w_res = float(self.reservoir_rate(X[0, 0], params))
        N = self.n_cells
        slopes = self.cell_slope.map(N)(X[1:].T, X[:-1].T, w_res, params, ca.DM(geo.delta_md).T,
                                        ca.DM(geo.cos_incl).T, ca.DM(geo.tvd_frac[1:]).T)
        return bool(np.all(np.asarray(slopes) > 0))

    def bounds(self, bc):
        """
        Bounds on the state for the solver: p in [p_s, p_r], alpha in [0, 1], velocities and densities
        non-negative, and T in [min(T_s, T_lg), T_r + 1]. A root must also be admissible (SOL-1).

        With the Joule-Thomson term (THM-8) on a real gas the fluid can be colder than its surroundings, and the
        lower bound is 1.05 T_pc where that is lower: the lower end of the Dranchuk-Abou-Kassem equation of state's
        range, below which the Joule-Thomson factor has a pole (PVT-GAS-10; specs/features/016-joule-thomson.md).
        """
        T_low = bc.T_s if bc.T_lg is None else min(bc.T_s, bc.T_lg)
        if self.wp.thermal.joule_thomson and not self.wp.fluid.ideal_gas:
            T_low = min(T_low, JT_T_PR_MIN * self.wp.fluid.pseudo_critical[1])
        lb = np.array([bc.p_s, 0, 0, 0, 0, 0, T_low], dtype=float)
        ub = np.array([bc.p_r, np.inf, np.inf, 1, np.inf, np.inf, bc.T_r + 1], dtype=float)
        n = self.n_cells + 1
        return np.tile(lb, n), np.tile(ub, n)


def build_system(wp) -> System:  # spec: DISC-11
    """Build the discretized system of a well (WellProperties) once."""
    geo = wp.geometry
    N = geo.n_cells
    n_x = DIM_X * (N + 1)

    x = ca.SX.sym('x', n_x)
    prm = ca.SX.sym('params', len(PARAMS))
    op = OperatingPoint.of(prm)
    S = [PointState.of(x[DIM_X * i:DIM_X * (i + 1)]) for i in range(N + 1)]

    # The reservoir liquid rate is a function of x_0 and the parameters, passed to every point's rows
    w_res = wp.inflow.liquid_mass_flow_rate(S[0].p, op.p_r)

    rows = bottom_rows(wp, S[0], w_res, op) + closure_rows(wp, S[0], geo.cos_incl[0])
    for i in range(1, N + 1):
        rows += cell_rows(wp, S[i], S[i - 1], w_res, op, geo.delta_md[i - 1], geo.cos_incl[i - 1], geo.tvd_frac[i])
        if i == N:
            rows.append(choke_row(wp, S[N], op))
        rows += closure_rows(wp, S[i], geo.cos_incl[i - 1])
    residual = ca.Function('residual', [x, prm], [ca.vertcat(*rows)], ['x', 'params'], ['r'])

    # Point 0 alone (the closures at point 0 use the inclination of cell 1)
    x_0 = ca.SX.sym('x_0', DIM_X)
    s_0 = PointState.of(x_0)
    w_res_0 = wp.inflow.liquid_mass_flow_rate(s_0.p, op.p_r)
    r_0 = bottom_rows(wp, s_0, w_res_0, op) + closure_rows(wp, s_0, geo.cos_incl[0])
    bottom = ca.Function('bottom_rows', [x_0, prm], [ca.vertcat(*r_0)], ['x_0', 'params'], ['r_0'])

    # A point i > 0, with the cell's geometry as inputs, so one function serves every cell
    x_i, x_prev = ca.SX.sym('x_i', DIM_X), ca.SX.sym('x_prev', DIM_X)
    w, d_md, cos_i, frac = ca.SX.sym('w_res'), ca.SX.sym('delta_md'), ca.SX.sym('cos_incl'), ca.SX.sym('tvd_frac')
    s_i, s_prev = PointState.of(x_i), PointState.of(x_prev)
    r_i = cell_rows(wp, s_i, s_prev, w, op, d_md, cos_i, frac) + closure_rows(wp, s_i, cos_i)
    point = ca.Function('point_rows', [x_i, x_prev, w, prm, d_md, cos_i, frac], [ca.vertcat(*r_i)],
                        ['x_i', 'x_prev', 'w_res', 'params', 'delta_md', 'cos_incl', 'tvd_frac'], ['r_i'])

    # The momentum row's slope in p_i along the curve on which the point's six other rows are zero (SOL-8):
    # ds = ∂r_p/∂p - ∂r_p/∂y (∂r_o/∂y)^-1 ∂r_o/∂p, with y the other six unknowns and r_o the other six rows
    J = ca.jacobian(ca.vertcat(*r_i), x_i)
    other = [0, 1, 3, 4, 5, 6]  # the rows of point i but DISC-9, the third of cell_rows
    slope = J[2, 0] - ca.mtimes(J[2, 1:], ca.solve(J[other, 1:], J[other, 0]))
    cell_slope = ca.Function('cell_slope', [x_i, x_prev, w, prm, d_md, cos_i, frac], [slope],
                             ['x_i', 'x_prev', 'w_res', 'params', 'delta_md', 'cos_incl', 'tvd_frac'], ['s'])

    # Reservoir rate, and a starting point for point 0: the densities and the inflow temperature at p_0, half gas
    # by volume, and the velocities that carry the phase rates
    p_0 = ca.SX.sym('p_0')
    w_p0 = wp.inflow.liquid_mass_flow_rate(p_0, op.p_r)
    reservoir_rate = ca.Function('reservoir_rate', [p_0, prm], [w_p0], ['p_0', 'params'], ['w_res'])
    fluid, A = wp.fluid, geo.A
    T_in = wp.thermal.inflow_temperature(w_p0, op.w_lg, op.T_r, op.T_lg, fluid)
    rho_g, rho_l = fluid.gas_density(p_0, T_in), fluid.liquid_density(p_0, T_in)
    w_g, w_l = fluid.phase_rates(p_0, T_in, w_p0, op.w_lg)
    alpha = 0.5
    guess = ca.vertcat(p_0, w_g / (A * alpha * rho_g), w_l / (A * (1 - alpha) * rho_l), alpha, rho_g, rho_l, T_in)
    bottom_guess = ca.Function('bottom_guess', [p_0, prm], [guess], ['p_0', 'params'], ['x_0'])

    # Outputs of a root: the flow-regime probabilities at every point (one function mapped over the points, for the
    # regime labels of SLIP-8), CHOKED and the reservoir rates
    x_pt, cos_pt = ca.SX.sym('x_pt', DIM_X), ca.SX.sym('cos_incl')
    s = PointState.of(x_pt)
    sigma = fluid.surface_tension(s.p, s.T, s.rho_l)
    classify = ca.Function('classify', [x_pt, cos_pt],
                           [wp.slip.classify(s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, cos_pt)])
    cos_points = [geo.cos_incl[max(i - 1, 0)] for i in range(N + 1)]
    P = classify.map(N + 1)(ca.reshape(x, DIM_X, N + 1), ca.DM(cos_points).T)
    regimes = ca.Function('regime_probabilities', [x], [P], ['x'], ['P'])
    choked = wp.choke.is_choked(S[N].p, op.p_s)
    outputs = ca.Function('outputs', [x, prm], [choked, w_res, fluid.reservoir_gas_rate(w_res)],
                          ['x', 'params'], ['choked', 'w_res', 'w_g_res'])

    return System(wp=wp, n_x=n_x, row_ids=row_ids(wp), residual=residual, bottom_rows=bottom, point_rows=point,
                  cell_slope=cell_slope,
                  reservoir_rate=reservoir_rate, bottom_guess=bottom_guess, regime_probabilities=regimes,
                  outputs=outputs)
