"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Why a v1-only sweep of the choke row over p0 is not a completeness check (v2 plan, Step 2.4).

Evaluates R(p0) (cellwise march at fixed p0, then v1's choke row) on a p0 grid and prints
the sign pattern: '+', '-', or '?' where the march fails or the row is NaN. Near p_r the
tubing cannot lift the liquid column, p_L < p_s, and the choke row takes the square root
of a negative number; at high rates the cellwise march fails. For well 977 the sweep finds
only the stable root and misses the trickle root at p0 = 208.02 bar.

Runs against v1.0.0, from the root of a v1.0.0 checkout (or set MANYWELLS_V1 to it):

    python <repo>/plans/evidence/choke_sweep.py --config manywells-sol-1_config.zip 41 977
"""

import argparse
import contextlib
import io
import time

import numpy as np

from stability_label import SimError, load_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[2])
    parser.add_argument('--config', required=True, help='path to manywells-sol-1_config.zip')
    parser.add_argument('n_points', type=int, help='number of p0 grid points')
    parser.add_argument('well_ids', type=int, nargs='+')
    args = parser.parse_args()

    for wid in args.well_ids:
        sim = load_graph(args.config, wid)
        bc = sim.bc
        t = time.time()
        vals = []
        for p0 in np.linspace(bc.p_s + 1e-3, bc.p_r - 1e-3, args.n_points):
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    R, _ = sim.R_of_p0(p0)
                except (SimError, RuntimeError):
                    R = np.nan
            vals.append((p0, R))
        signs = ''.join('+' if R > 0 else '-' if R < 0 else '?' for _, R in vals)
        changes = [(round(vals[i][0], 2), round(vals[i + 1][0], 2)) for i in range(len(vals) - 1)
                   if np.sign(vals[i][1]) * np.sign(vals[i + 1][1]) < 0]
        print(f'well {wid}: {time.time() - t:.0f}s  signs {signs}\n  sign changes in {changes}', flush=True)


if __name__ == '__main__':
    main()
