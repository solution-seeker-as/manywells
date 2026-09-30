"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The verifier's checks (specs/verification.md), applied to a candidate's roots for one case.

The verifier holds no model. It compares a candidate's roots with the case's reference root
set, computed from v1.0.0: Invariants per root; Operating point, Root set and Stability per
case, by matching each candidate root to the nearest reference root; Convergence of the
operating point over N, 2N and 4N per group.
"""

from dataclasses import dataclass, field

import numpy as np

from manywells_verify.cases import DIM_X, Case

V_FLOOR = 1e-3      # m/s, floor on velocity scales in the state distance


@dataclass(frozen=True)
class Tolerances:
    """Provisional values until Step 2 item 5 sets them (specs/verification.md)."""
    tol_x: float = 1e-4              # scaled state distance to a reference root, ∞-norm
    p_slack: float = 1e-6            # bar: allowed pressure rise between neighbouring points
    flux_rel: float = 1e-6           # phase mass-rate variation along the well, relative to the total rate
    T_slack: float = 1e-6            # K: allowed temperature below the ambient profile
    choke_band: float = 5e-4         # bar: dead band around the choked/unchoked switch
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
    invariants: Check
    match: int | None = None         # index of the reference root within tol_x, if any
    distance: float | None = None    # scaled distance to the nearest reference root

    @property
    def valid(self) -> bool:
        return not self.invariants.failed


@dataclass
class CaseResult:
    case: Case
    roots: list
    checks: dict                     # invariants, operating_point, root_set, stability
    findings: list = field(default_factory=list)
    n_stable_reference: int | None = None


# ---------------------------------------------------------------------------------------------
# Quantities derived from a state

def area(case: Case) -> float:
    return np.pi * (case.params['D'] / 2) ** 2


def grid(x, n_cells: int) -> np.ndarray:
    """State as an (N + 1, 7) array: one row per grid point, columns as in cases.STATE."""
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


def state_distance(x, reference, case: Case) -> float:
    """
    Scaled ∞-norm distance between two states on the same grid: p by p_r - p_s, velocities and
    densities relative to the reference, alpha absolute, T by T_r - T_s.
    """
    X, Y = grid(x, case.n_cells), grid(reference, case.n_cells)
    scale = np.column_stack([
        np.full(len(Y), max(case.params['p_r'] - case.params['p_s'], 1.0)),
        np.maximum(np.abs(Y[:, 1]), V_FLOOR), np.maximum(np.abs(Y[:, 2]), V_FLOOR),
        np.ones(len(Y)),
        np.maximum(Y[:, 4], 1e-3), np.maximum(Y[:, 5], 1e-3),
        np.full(len(Y), max(case.params['T_r'] - case.params['T_s'], 1.0)),
    ])
    return float(np.max(np.abs(X - Y) / scale))


# ---------------------------------------------------------------------------------------------
# Per-root check

def check_invariants(x, case: Case, choked, tol: Tolerances) -> Check:
    N = case.n_cells
    if np.size(x) != DIM_X * (N + 1):
        return Check('fail', f'state has length {np.size(x)}, expected {DIM_X * (N + 1)} for N = {N}')
    X = grid(x, N)
    if not np.all(np.isfinite(X)):
        return Check('fail', 'state has non-finite values')
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
        problems.append(f'pressure rises by {rise.max():.2e} bar at point {int(np.argmax(rise)) + 1}')
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
        extra = f' (+{len(problems) - 3} more)' if len(problems) > 3 else ''
        return Check('fail', '; '.join(problems[:3]) + extra)
    return Check('pass')


# ---------------------------------------------------------------------------------------------
# Per-case checks

def check_operating_point(case: Case, roots, results, reference, tol: Tolerances) -> Check:
    stable = [k for k, r in enumerate(reference) if r.label == 'stable']
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
    k, s = chosen[0], stable[0]
    res = results[k]
    if not res.valid:
        return Check('fail', f'the operating point fails Invariants: {res.invariants.detail}')
    d = state_distance(roots[k].x, reference[s].x, case)
    if d <= tol.tol_x:
        return Check('pass', f'distance {d:.1e} to the stable reference root', d)
    nearest = min(range(len(reference)), key=lambda j: state_distance(roots[k].x, reference[j].x, case))
    return Check('fail', f'distance {d:.1e} to the stable reference root; nearest reference root is '
                         f'{reference[nearest].label} ({state_distance(roots[k].x, reference[nearest].x, case):.1e}); '
                         f'p0 {roots[k].x[0]:.3f} bar vs {reference[s].x[0]:.3f} bar', d)


def check_root_set(roots, results, reference) -> Check:
    if len(roots) <= 1 and all(r.label is None for r in roots):
        return Check('n/a', 'candidate reports no root set (at most one unlabelled root)')
    matched = {res.match for res in results if res.valid and res.match is not None}
    missing = [f'{r.label} root at p0 = {r.x[0]:.3f} bar' for j, r in enumerate(reference) if j not in matched]
    if missing:
        return Check('fail', 'missing: ' + '; '.join(missing))
    return Check('pass', f'{len(reference)} reference roots matched')


def check_stability(roots, results, reference) -> Check:
    """Each label the candidate reports must equal the label of the reference root it matches."""
    compared = [(k, res.match) for k, res in enumerate(results)
                if roots[k].label is not None and res.valid and res.match is not None]
    if not compared:
        return Check('n/a', 'no labelled root matches a reference root')
    wrong = [(k, j) for k, j in compared if roots[k].label != reference[j].label]
    if wrong:
        k, j = wrong[0]
        return Check('fail', f'root {k} is labelled {roots[k].label}, the reference root at '
                             f'p0 = {reference[j].x[0]:.3f} bar is {reference[j].label}')
    return Check('pass', f'{len(compared)} labels match the reference')


def aggregate_invariants(results) -> Check:
    if not results:
        return Check('n/a', 'no roots reported')
    failed = [res for res in results if res.invariants.failed]
    if failed:
        return Check('fail', f'root {failed[0].index}: {failed[0].invariants.detail}')
    return Check('pass', f'{len(results)} roots')


def verify_case(case: Case, roots, reference, tol: Tolerances = Tolerances()) -> CaseResult:
    """Run every per-root and per-case check on a candidate's roots for one case."""
    roots, reference = list(roots or []), list(reference or [])
    results = []
    for k, root in enumerate(roots):
        res = RootResult(k, check_invariants(root.x, case, root.choked, tol))
        if res.valid and reference:
            dist = [state_distance(root.x, ref.x, case) for ref in reference]
            j = int(np.argmin(dist))
            res.distance = dist[j]
            res.match = j if dist[j] <= tol.tol_x else None
        results.append(res)

    checks = {'invariants': aggregate_invariants(results),
              'operating_point': check_operating_point(case, roots, results, reference, tol),
              'root_set': check_root_set(roots, results, reference),
              'stability': check_stability(roots, results, reference)}
    findings = [f'root {res.index} (p0 = {roots[res.index].x[0]:.3f} bar) passes Invariants but matches no '
                f'reference root' + (f' (nearest at distance {res.distance:.1e})' if res.distance is not None else '')
                for res in results if res.valid and res.match is None]
    if checks['operating_point'].status == 'indeterminate':
        findings.append(checks['operating_point'].detail)
    return CaseResult(case, results, checks, findings, sum(r.label == 'stable' for r in reference))


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
        if d2 == 0 or abs(d2) <= tol.conv_noise * max(abs(o4), 1.0):
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
