"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Small fixtures for the verifier's unit tests, from v1.0.0 (verification/tests/data/fixtures.npz):

- sol-1 well 977, a two-root well: its trickle root and its stable root;
- a one-root sol-1 well;
- well 977 with p_r lowered past the fold, where v1 finds no root from any start;
- the one-root well's root at N = 50, 100 and 200, as a convergence group.

Each root is labelled from v1's Jacobian with plans/evidence/stability_label.py. Runs in the v1.0.0
environment, from the repository root:

    .worktrees/v1.0.0/.venv/bin/python verification/build/make_test_fixtures.py --data-dir <dir>
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, params_of, variant_of  # noqa: E402  (sets up the v1.0.0 imports)
from manywells.simulator import SimError  # noqa: E402
from root_sets import GUESSES  # noqa: E402  (plans/evidence)
from scripts.load_well_from_dataset import load_well  # noqa: E402
from stability_label import Graph  # noqa: E402

OUT = REPO / 'verification' / 'tests' / 'data' / 'fixtures.npz'
FOLD_P_R = 185.0  # bar; well 977's two roots merge near p_r = 188.6 bar


def roots_of(wp, bc, n_cells):
    """Distinct roots v1 reaches from its default guess and cellwise guesses (root_sets.GUESSES)."""
    sim = Graph(wp, bc, n_cells=n_cells).build()
    found = {}
    for f in GUESSES:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.x_guess = None if f is None else sim._simulate_cellwise(bc.p_s + f * (bc.p_r - bc.p_s), bc.T_r)
                x = np.array(sim.simulate())
        except SimError:
            continue
        found.setdefault(round(x[0], 3), x)
    roots = []
    for p0, x in sorted(found.items()):
        slope, _ = sim.dR_dp0(x)
        roots.append({'x': x, 'label': 'unstable' if slope > 0 else 'stable', 'v1_slope': float(slope)})
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--data-dir', type=Path, required=True, help='folder with manywells-sol-1_config.zip')
    args = parser.parse_args()
    df = pd.read_csv(args.data_dir / 'manywells-sol-1_config.zip', compression='zip')

    # First well that fails the two-root criterion p_r - p_s < rho_l g L / 1e5
    one_root = int(df.loc[df['bc.p_r'] - df['bc.p_s'] >= df['wp.rho_l'] * 9.81 * df['wp.L'] / 1e5, 'ID'].iloc[0])
    specs = [('sol1-977', 977, None, 100, ''), ('sol1-one-root', one_root, None, 100, ''),
             ('sol1-977-past-fold', 977, FOLD_P_R, 100, '')]
    specs += [(f'conv-one-root-N{n}', one_root, None, n, 'conv-one-root') for n in (50, 100, 200)]

    meta, arrays = [], {}
    for case_id, well_id, p_r, n_cells, group in specs:
        well = load_well(well_id, df)
        if p_r is not None:
            well.bc.p_r = p_r
        roots = roots_of(well.wp, well.bc, n_cells)
        entry = {'case_id': case_id, 'config_id': well_id, 'n_cells': n_cells, 'group': group,
                 'params': params_of(well.wp, well.bc), 'variant': variant_of(well.wp), 'roots': []}
        for k, root in enumerate(roots):
            key = f'{case_id}.root{k}'
            arrays[key] = root['x']
            entry['roots'].append({'key': key, 'label': root['label'], 'v1_slope': root['v1_slope']})
        meta.append(entry)
        print(f'{case_id}: {len(roots)} roots, ' + ', '.join(f'p0 {arrays[r["key"]][0]:.3f} {r["label"]}'
                                                            for r in entry['roots']))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, meta=np.array(json.dumps(meta)), **arrays)
    print(f'Wrote {OUT.relative_to(REPO)}')


if __name__ == '__main__':
    main()
