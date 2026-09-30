"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The verifier's checks (specs/verification.md), applied to a candidate's roots for one case.

Per root: Residuals, Invariants and, for roots that pass both, Stability (the label computed
from the residual graph). Per case: Operating point and Root set, against the case's reference
root set. Per convergence group: Convergence of the operating point over N, 2N and 4N.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla

from manywells_verify.cases import Case, Root
from manywells_verify.residual import DIM_X, ResidualGraph, choke_row, row_layout

GRAVITY = 9.81      # m/s², for the momentum-row scale only
W_FLOOR = 1e-3      # kg/s, floor on mass-rate scales
V_FLOOR = 1e-3      # m/s, floor on velocity scales


@dataclass(frozen=True)
class Tolerances:
    """Provisional values until Step 2 item 5 sets them (specs/verification.md)."""
    tol_r: float = 1e-6              # scaled residual, ∞-norm
    tol_x: float = 1e-4              # scaled state distance, ∞-norm
    p_slack: float = 1e-6            # bar: allowed pressure rise between neighbouring points
    flux_rel: float = 1e-6           # phase mass-rate variation along the well, relative to the total rate
    T_slack: float = 1e-6            # K: allowed temperature below the ambient profile
    choke_band: float = 5e-4         # bar: dead band around the choked/unchoked switch
    label_min: float = 1e-6          # |normalized dR/dp0| below which the label is indeterminate
    order: tuple = (0.7, 1.4)        # bounds on the observed order of convergence
    conv_noise: float = 1e-9         # relative change of an output below which it is not used
    conv_choke_margin: float = 1.0   # bar: groups this close to the choke switch are skipped


@dataclass
class Check:
    status: str                      # 'pass', 'fail', 'indeterminate' or 'n/a'
    detail: str = ''
    value: float | None = None

    @property
    def failed(self) -> bool:
        return self.status == 'fail'


@dataclass
class RootResult:
    index: int
    checks: dict
    label: str | None = None         # label from the residual graph, for roots that pass
    slope: float | None = None       # normalized dR/dp0

    @property
    def valid(self) -> bool:
        return not (self.checks['residuals'].failed or self.checks['invariants'].failed)


@dataclass
class CaseResult:
    case: Case
    roots: list
    checks: dict                     # every check, aggregated over the case's roots
    findings: list = field(default_factory=list)
    n_stable_reference: int | None = None   # stable roots in the reference root set, if there is one


# ---------------------------------------------------------------------------------------------
# Quantities derived from a state

def area(case: Case) -> float:
    return np.pi * (case.params['D'] / 2) ** 2


def grid(x, n_cells: int) -> np.ndarray:
    """State as an (N + 1, 7) array: one row per grid point, columns as in residual.STATE."""
    return np.asarray(x, dtype=float).reshape(n_cells + 1, DIM_X)


def mass_rates(X, case: Case):
    """Gas and liquid mass rates (kg/s) at every grid point."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    A = area(case)
    return A * alpha * rho_g * v_g, A * (1 - alpha) * rho_l * v_l


def ambient_temperature(case: Case) -> np.ndarray:
    N, T_r, T_s = case.n_cells, case.params['T_r'], case.params['T_s']
    return T_r - np.arange(N + 1) * (T_r - T_s) / N


def outputs(X, case: Case) -> dict:
    w_g, w_l = mass_rates(X, case)
    return {'PBH': X[0, 0], 'PWH': X[-1, 0], 'TWH': X[-1, 6], 'WLIQ': w_l[0], 'WGAS incl. lift gas': w_g[0]}


def row_points(n_cells: int) -> np.ndarray:
    """Grid point that each row of r belongs to."""
    rows_left, rows_clo, rows_cell, row_chk = row_layout(n_cells)
    points = np.empty(DIM_X * (n_cells + 1), dtype=int)
    points[rows_left] = 0
    points[rows_clo] = np.arange(n_cells + 1)[:, None]
    points[rows_cell] = np.arange(1, n_cells + 1)[:, None]
    points[row_chk] = n_cells
    return points


def row_scales(X, case: Case, names, points) -> np.ndarray:
    """Scale of each residual row, so that rows of different kinds and units are comparable."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    A = area(case)
    G = alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l        # total mass flux, kg/(m² s)
    rho_m = alpha * rho_g + (1 - alpha) * rho_l
    dz = case.params['L'] / case.n_cells
    dT = max(case.params['T_r'] - case.params['T_s'], 1.0)
    per_point = {
        'gas inflow': np.maximum(A * G, W_FLOOR), 'liquid inflow': np.maximum(A * G, W_FLOOR),
        'inflow temperature': np.full_like(p, dT),
        'slip': np.maximum(np.abs(v_g), V_FLOOR),
        'gas EOS': np.maximum(p, 1.0),
        'liquid density': np.full_like(p, case.params['rho_l']),
        'gas mass': np.maximum(G, W_FLOOR / A), 'liquid mass': np.maximum(G, W_FLOOR / A),
        'momentum': np.maximum(rho_m * GRAVITY * dz / 1e5, 1e-3),
        'energy': np.full_like(p, dT),
        'choke': np.maximum(A * G, W_FLOOR),
    }
    return np.array([per_point[name][k] for name, k in zip(names, points)])


def state_distance(x, reference, case: Case) -> float:
    """Scaled ∞-norm distance between two states on the same grid."""
    X, Y = grid(x, case.n_cells), grid(reference, case.n_cells)
    scale = np.column_stack([
        np.full(len(Y), max(case.params['p_r'] - case.params['p_s'], 1.0)),     # p
        np.maximum(np.abs(Y[:, 1]), V_FLOOR), np.maximum(np.abs(Y[:, 2]), V_FLOOR),  # v_g, v_l
        np.ones(len(Y)),                                                        # alpha
        np.maximum(Y[:, 4], 1e-3), np.maximum(Y[:, 5], 1e-3),                   # rho_g, rho_l
        np.full(len(Y), max(case.params['T_r'] - case.params['T_s'], 1.0)),     # T
    ])
    return float(np.max(np.abs(X - Y) / scale))


# ---------------------------------------------------------------------------------------------
# Per-root checks

def check_residuals(r, scale, names, points, n_cells: int, tol: Tolerances) -> Check:
    c = choke_row(n_cells)
    if np.isnan(r[c]):
        return Check('fail', 'choke row is NaN: the wellhead pressure is below the pressure the choke sees downstream')
    if not np.all(np.isfinite(r)):
        bad = np.flatnonzero(~np.isfinite(r))
        return Check('fail', f'{len(bad)} non-finite rows, first: {names[bad[0]]} at point {points[bad[0]]}')
    scaled = np.abs(r) / scale
    k = int(np.argmax(scaled))
    detail = f'{scaled[k]:.1e} at the {names[k]} row of point {points[k]} ({r[k]:+.2e} unscaled)'
    return Check('pass' if scaled[k] <= tol.tol_r else 'fail', detail, float(scaled[k]))


def check_invariants(X, case: Case, choked, tol: Tolerances) -> Check:
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    prm = case.params
    problems = []
    if alpha.min() < 0 or alpha.max() > 1:
        problems.append(f'alpha outside [0, 1] ({alpha.min():.3g} to {alpha.max():.3g})')
    for name, values in (('rho_g', rho_g), ('rho_l', rho_l), ('v_g', v_g), ('v_l', v_l)):
        if values.min() <= 0:
            problems.append(f'{name} not positive (min {values.min():.3g})')
    w_g, w_l = mass_rates(X, case)
    w_tot = w_g[0] + w_l[0]
    if w_tot > 0:
        drift = max(np.abs(w_g - w_g[0]).max(), np.abs(w_l - w_l[0]).max()) / w_tot
        if drift > tol.flux_rel:
            problems.append(f'phase mass rates vary along the well by {drift:.1e} of the total rate')
    rise = np.diff(p)
    if rise.max() > tol.p_slack:
        k = int(np.argmax(rise)) + 1
        problems.append(f'pressure rises by {rise.max():.2e} bar at point {k}')
    if not p[-1] > prm['p_s']:
        problems.append(f'wellhead pressure {p[-1]:.4f} bar not above p_s = {prm["p_s"]:.4f} bar')
    if not p[0] < prm['p_r']:
        problems.append(f'bottomhole pressure {p[0]:.4f} bar not below p_r = {prm["p_r"]:.4f} bar')
    below = ambient_temperature(case) - T
    if below.max() > tol.T_slack:
        problems.append(f'temperature {below.max():.2e} K below ambient at point {int(np.argmax(below))}')
    if choked is not None:
        margin = prm['cpr'] * p[-1] - prm['p_s']       # v1's is_choked: p_s <= cpr p_N
        if abs(margin) > tol.choke_band and choked != (margin >= 0):
            problems.append(f'CHOKED = {choked}, but cpr p_N - p_s = {margin:+.4f} bar')
    if problems:
        return Check('fail', '; '.join(problems[:3]) + (f' (+{len(problems) - 3} more)' if len(problems) > 3 else ''))
    return Check('pass')


def stability_slope(J, X, case: Case):
    """
    Normalized d(choke row)/dp0 along the manifold where every other row holds, and a
    condition estimate of the reduced Jacobian. Positive is unstable, negative is stable.
    """
    N = case.n_cells
    c, n = choke_row(N), J.shape[0]
    rows = np.delete(np.arange(n), c)
    cols = np.arange(1, n)
    J = J.tocsr()
    J_rr = J[rows][:, cols].tocsc()
    lu = spla.splu(J_rr)
    dx = -lu.solve(J[rows, 0].toarray().ravel())
    slope = J[c, 0] + (J[c][:, cols] @ dx).item()
    w_g, w_l = mass_rates(X, case)
    normalized = float(slope) * (case.params['p_r'] - case.params['p_s']) / max(w_g[-1] + w_l[-1], W_FLOOR)
    inverse = spla.LinearOperator(J_rr.shape, matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans='T'),
                                  dtype=float)
    cond = spla.onenormest(J_rr) * spla.onenormest(inverse)
    return normalized, float(cond)


def label_of(slope: float, tol: Tolerances) -> str:
    if slope > tol.label_min:
        return 'unstable'
    if slope < -tol.label_min:
        return 'stable'
    return 'indeterminate'


def check_root(graph: ResidualGraph, case: Case, root: Root, index: int, tol: Tolerances) -> RootResult:
    N = case.n_cells
    if root.x.shape != (DIM_X * (N + 1),):
        failed = Check('fail', f'state has length {root.x.size}, expected {DIM_X * (N + 1)} for N = {N}')
        return RootResult(index, {'residuals': failed, 'invariants': failed,
                                  'stability': Check('n/a', 'root is not valid')})
    X = grid(root.x, N)
    P = graph.param_vector(case.params)
    r, J = graph.evaluate(root.x, P, N, case.variant)
    names, points = graph.row_names(N), row_points(N)
    checks = {'residuals': check_residuals(r, row_scales(X, case, names, points), names, points, N, tol),
              'invariants': check_invariants(X, case, root.choked, tol)}
    result = RootResult(index, checks)
    if not result.valid:
        checks['stability'] = Check('n/a', 'root is not valid')
        return result

    try:
        slope, cond = stability_slope(J, X, case)
    except RuntimeError as e:            # singular reduced Jacobian
        checks['stability'] = Check('indeterminate', f'reduced Jacobian is singular ({e})')
        result.label = 'indeterminate'
        return result
    result.label, result.slope = label_of(slope, tol), slope
    detail = f'graph label {result.label} (normalized dR/dp0 {slope:+.2e}, cond {cond:.1e})'
    if result.label == 'indeterminate':
        checks['stability'] = Check('indeterminate', detail, slope)
    elif root.label is None:
        checks['stability'] = Check('n/a', detail + '; candidate reports no label', slope)
    else:
        status = 'pass' if root.label == result.label else 'fail'
        checks['stability'] = Check(status, f'candidate label {root.label}, {detail}', slope)
    return result


# ---------------------------------------------------------------------------------------------
# Per-case checks

def check_operating_point(case: Case, roots, results, reference, tol: Tolerances) -> Check:
    if reference is None:
        return Check('n/a', 'no reference root set')
    stable = [r for r in reference if r.label == 'stable']
    chosen = [k for k, r in enumerate(roots) if r.operating_point]
    if len(chosen) > 1:
        return Check('fail', f'{len(chosen)} roots are marked as the operating point')
    if len(stable) > 1:
        return Check('indeterminate', f'reference has {len(stable)} stable roots')
    if not stable:
        if chosen:
            return Check('fail', 'reference has no stable root (the well cannot flow), but an operating point is reported')
        return Check('pass', 'no stable reference root and no operating point reported')
    if not chosen:
        return Check('fail', 'no operating point reported, but the reference has a stable root')
    k = chosen[0]
    if not results[k].valid:
        return Check('fail', 'the operating point fails Residuals or Invariants')
    d = state_distance(roots[k].x, stable[0].x, case)
    if d <= tol.tol_x:
        return Check('pass', f'distance {d:.1e} to the stable reference root', d)
    nearest = min(reference, key=lambda r: state_distance(roots[k].x, r.x, case))
    return Check('fail', f'distance {d:.1e} to the stable reference root; nearest reference root is '
                         f'{nearest.label} ({state_distance(roots[k].x, nearest.x, case):.1e}); '
                         f'p0 {roots[k].x[0]:.3f} bar vs {stable[0].x[0]:.3f} bar', d)


def check_root_set(case: Case, roots, results, reference, tol: Tolerances):
    """Root set check and findings about valid candidate roots missing from the reference."""
    if reference is None:
        return Check('n/a', 'no reference root set'), []
    if len(roots) <= 1 and all(r.label is None for r in roots):
        return Check('n/a', 'candidate reports no root set (at most one unlabelled root)'), []
    valid = [k for k, res in enumerate(results) if res.valid]
    matched, missing = set(), []
    for ref in reference:
        # An unlabelled candidate root matches by distance alone; Stability checks the labels it does report
        eligible = [k for k in valid if roots[k].label is None or ref.label == 'indeterminate'
                    or roots[k].label == ref.label]
        dist = {k: state_distance(roots[k].x, ref.x, case) for k in eligible}
        best = min(dist, key=dist.get, default=None)
        if best is None or dist[best] > tol.tol_x:
            missing.append(f'{ref.label} root at p0 = {ref.x[0]:.3f} bar')
        else:
            matched.add(best)
    findings = [f'candidate root {k} (p0 = {roots[k].x[0]:.3f} bar) is valid but not in the reference root set'
                for k in valid if k not in matched]
    if missing:
        return Check('fail', 'missing: ' + '; '.join(missing)), findings
    return Check('pass', f'{len(reference)} reference roots matched'), findings


def aggregate(results, name: str) -> Check:
    """One check over all roots: fail if any root fails, then indeterminate, then pass."""
    checks = [(res.index, res.checks[name]) for res in results]
    if not checks:
        return Check('n/a', 'no roots reported')
    for status in ('fail', 'indeterminate'):
        hits = [(k, c) for k, c in checks if c.status == status]
        if hits:
            k, c = hits[0]
            return Check(status, f'root {k}: {c.detail}', c.value)
    passed = [(k, c) for k, c in checks if c.status == 'pass']
    if not passed:
        return Check('n/a', checks[0][1].detail)
    worst = max(passed, key=lambda kc: kc[1].value if kc[1].value is not None else 0.0)
    return Check('pass', f'root {worst[0]}: {worst[1].detail}' if worst[1].detail else '', worst[1].value)


def verify_case(graph: ResidualGraph, case: Case, roots, reference=None, tol: Tolerances = Tolerances()) -> CaseResult:
    """Run every per-root and per-case check on a candidate's roots for one case."""
    roots = list(roots or [])
    results = [check_root(graph, case, root, k, tol) for k, root in enumerate(roots)]
    checks = {name: aggregate(results, name) for name in ('residuals', 'invariants', 'stability')}
    checks['operating_point'] = check_operating_point(case, roots, results, reference, tol)
    checks['root_set'], findings = check_root_set(case, roots, results, reference, tol)
    findings += [f'root {res.index}: {res.checks["stability"].detail}'
                 for res in results if res.checks['stability'].status == 'indeterminate']
    if checks['operating_point'].status == 'indeterminate':
        findings.append(checks['operating_point'].detail)
    n_stable = None if reference is None else sum(r.label == 'stable' for r in reference)
    return CaseResult(case, results, checks, findings, n_stable)


# ---------------------------------------------------------------------------------------------
# Per-group check

def check_convergence(members, tol: Tolerances) -> Check:
    """
    Observed order of each output over N, 2N and 4N. members: (case, operating point or None),
    one per grid. Implicit Euler is first order, so the order should be close to 1.
    """
    members = sorted(members, key=lambda m: m[0].n_cells)
    Ns = [c.n_cells for c, _ in members]
    if len(members) != 3 or Ns[1] != 2 * Ns[0] or Ns[2] != 2 * Ns[1]:
        return Check('n/a', f'group has grids {Ns}, expected N, 2N and 4N')
    if any(root is None for _, root in members):
        return Check('fail', 'no valid operating point on every grid')
    grids = [grid(root.x, c.n_cells) for c, root in members]
    margin = min(abs(c.params['cpr'] * X[-1, 0] - c.params['p_s']) for (c, _), X in zip(members, grids))
    if margin < tol.conv_choke_margin:
        return Check('n/a', f'within {margin:.2f} bar of the choke switch')
    outs = [outputs(X, c) for (c, _), X in zip(members, grids)]
    orders, skipped = {}, []
    for name in outs[0]:
        o1, o2, o4 = (o[name] for o in outs)
        d1, d2 = o1 - o2, o2 - o4
        if abs(d2) <= tol.conv_noise * max(abs(o4), 1.0) or d2 == 0:
            skipped.append(name)
            continue
        orders[name] = float(np.log2(abs(d1 / d2)))
    if not orders:
        return Check('n/a', 'every output changes at the noise level')
    lo, hi = tol.order
    bad = {k: q for k, q in orders.items() if not lo <= q <= hi}
    detail = ', '.join(f'{k} {q:.2f}' for k, q in orders.items())
    if skipped:
        detail += f' (not used: {", ".join(skipped)})'
    return Check('fail' if bad else 'pass', f'observed orders {detail}', min(orders.values()))
