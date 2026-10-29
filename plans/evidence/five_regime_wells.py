"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The five-regime slip closure of specs/features/019-five-flow-regimes.md on wells drawn by develop's sampler (seed
2026), each at one sampled operating point, on the CasADi backend. A prototype (five_regime_slip.py): the closure is
a subclass of SlipModel. Each well is solved in five variants of develop's default:

  current    develop's fitted three-regime classifier (SLIP-6 to SLIP-11)
  proposal   the five-regime closure: w = 0.02, f = 0.02 in (A-17), lambda floor 0.1 v_inf_b, sqrt guard 1e-6
  no_floor   the proposal with lambda = j_g / j_m
  no_guard   the proposal with sqrt(lambda) in (A-17)
  cell_fd    the proposal with the cell's Darcy factor f_D in (A-17)

Per variant it records whether there is an operating point, the number of roots, PBH, the reservoir liquid rate
and the time of the root search. Along the proposal's operating point it records the five-regime labels; along the
cell_fd variant's it records f_D, the labels with f_D and with 0.02, and whether the slip law has several void
fractions when f_D depends on alpha (the rest of the point fixed).

From the repository root, on one process (about 40 minutes for 40 wells, most of it in the no_floor and no_guard
variants):

    uv run python plans/evidence/five_regime_wells.py 40 five_regime_wells.csv
"""

import sys
import time
from collections import Counter
from dataclasses import replace

import casadi as ca
import numpy as np
import pandas as pd

from five_regime_slip import (MERGED, Hasan5SlipModel, label, margins, parameters, patch_closure_rows,
                              probabilities, terms)
from manywells.configurations import DEVELOP
from manywells.discretization import DIM_X, PointState
from manywells.sampling.conditions import sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026
VARIANTS = {'current': None, 'proposal': {}, 'no_floor': {'floor': 0.0}, 'no_guard': {'sqrt_eps': 0.0},
            'cell_fd': {'f': None}}
AG = np.linspace(1e-3, 1 - 1e-3, 400)


def solve(wp, bc):
    t0 = time.perf_counter()
    try:
        rs = SSDFSimulator(wp).root_set(bc)
    except (SimError, ValueError, RuntimeError):
        rs = None
    return rs, time.perf_counter() - t0


def points(wp, x):
    """(j_g, j_l, rho_g, rho_l, sigma, cos) at each point of a state."""
    X = np.reshape(x, (-1, DIM_X))
    for i, (p, vg, vl, a, rg, rl, T) in enumerate(X):
        yield i, X[i], (a * vg, (1 - a) * vl, rg, rl, float(wp.fluid.surface_tension(p, T, rl)),
                        float(wp.geometry.cos_incl[max(i - 1, 0)]))


def main(n_wells, out):
    patch_closure_rows()
    rows, shares, merged_diff, n_pts = [], Counter(), 0, 0
    fd, fd_changes, fd_multi, fd_pts = [], Counter(), 0, 0
    for well in range(n_wells):
        draw = sample_well(SEED, well)
        bc, fr = sample_conditions(draw, rng_for(SEED, well, 'conditions'))
        try:
            wp = well_properties(draw, fr, DEVELOP)
        except ValueError:
            continue
        D = wp.geometry.D
        ops = {}
        for name, kw in VARIANTS.items():
            wpv = wp if kw is None else replace(wp, slip=Hasan5SlipModel(D=D, **kw))
            rs, t = solve(wpv, bc)
            op = rs.operating_point if rs else None
            ops[name] = op
            rows.append(dict(well=well, variant=name, trajectory=draw.trajectory[0], gas_lift=bc.w_lg > 0,
                             roots=len(rs.roots) if rs else 0, op=op is not None,
                             pbh=op.x[0] if op else np.nan, w_res=op.w_res if op else np.nan, seconds=t))
        if ops['proposal'] is not None and ops['current'] is not None:
            for (i, _, pt), cur in zip(points(wp, ops['proposal'].x), ops['current'].flow_regime):
                lab = label(probabilities(margins(terms(pt[0], pt[1], *pt[2:5], D, pt[5]), pt[0]), 0.02))[0]
                shares[lab] += 1
                merged_diff += MERGED[lab] != cur
                n_pts += 1
        if ops['cell_fd'] is not None:
            xs = ca.SX.sym('x', DIM_X)
            F = ca.Function('f_D', [xs], [wp.friction.friction_factor(PointState.of(xs), wp.fluid, D)])
            for i, x_i, (jg, jl, rg, rl, sigma, cos) in points(wp, ops['cell_fd'].x):
                f_i = float(F(x_i))
                fd.append(f_i)
                a, b = [label(probabilities(margins(terms(jg, jl, rg, rl, sigma, D, cos, f), jg), 0.02))[0]
                        for f in (0.02, f_i)]
                fd_changes[(a, b)] += a != b
                XA = np.tile(x_i[:, None], (1, AG.size))
                XA[1], XA[2], XA[3] = jg / AG, jl / (1 - AG), AG
                t = terms(jg, jl, rg, rl, sigma, D, cos, np.asarray(F.map(AG.size)(XA)).ravel())
                C_0, v_inf = parameters(probabilities(margins(t, jg), 0.02), t)
                h = AG * (C_0 * (jg + jl) + v_inf) - jg
                fd_multi += int(np.sum(np.sign(h[:-1]) * np.sign(h[1:]) < 0) > 1)
                fd_pts += 1
        print(f'well {well:3d}: ' + ', '.join(f"{r['variant']} {r['pbh']:.2f} bar {r['seconds']:.1f} s"
                                               for r in rows[-len(VARIANTS):]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    wide = df.pivot(index='well', columns='variant')
    ok = wide['op'].astype(bool)
    print(f'\n{wide.shape[0]} wells')
    for name in VARIANTS:
        d = df[df.variant == name]
        print(f'{name:>9}: operating point {d.op.sum()}, roots per well {dict(Counter(d.roots))}, '
              f'time {d.seconds.sum():.0f} s in total, {d.seconds.median():.1f} s median, {(d.seconds > 20).sum()} over 20 s')
    for a, b in [('proposal', 'current'), ('cell_fd', 'proposal'), ('no_floor', 'proposal'), ('no_guard', 'proposal')]:
        both = ok[a] & ok[b]
        dp = (wide.pbh[a] - wide.pbh[b])[both]
        dw = ((wide.w_res[a] - wide.w_res[b]) / wide.w_res[b])[both]
        print(f'{a} - {b} ({both.sum()} wells): PBH median {dp.median():+.3f}, mean |.| {dp.abs().mean():.3f}, '
              f'max |.| {dp.abs().max():.3f} bar; rate median {dw.median():+.2%}, max |.| {dw.abs().max():.2%}')
    print(f'proposal regimes along its operating points ({n_pts} points): '
          + ', '.join(f'{k} {v / n_pts:.1%}' for k, v in shares.most_common()))
    print(f'points whose merged label differs from the current one (same point index): {merged_diff / n_pts:.1%}')
    fd = np.array(fd)
    print(f'cell f_D along the cell_fd operating points ({fd_pts} points): 5th percentile {np.percentile(fd, 5):.4f}, '
          f'median {np.median(fd):.4f}, 95th {np.percentile(fd, 95):.4f}')
    print(f'labels with f_D instead of 0.02: {sum(fd_changes.values())} points change '
          f'({", ".join(f"{a} -> {b} {n}" for (a, b), n in fd_changes.items() if n)}); '
          f'several void fractions under f_D(alpha): {fd_multi} of {fd_pts}')


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
