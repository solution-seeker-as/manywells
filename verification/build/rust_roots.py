"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Rust roots for the case set (method B of the reference root sets, v2 plan Step 2 item 4), and
fold brackets for near-fold cases. Runs in the Rust environment (README.md), from the repository root:

    .worktrees/rust/.venv/bin/python verification/build/rust_roots.py fold --data-dir <dir>
    .worktrees/rust/.venv/bin/python verification/build/rust_roots.py roots

`fold` picks two-root sol-1 wells and bisects p_r down to where the two roots merge and the well
stops flowing; it writes data/full/fold.csv for make_cases.py. `roots` solves every case in
data/full/cases.json and writes data/full/rust_roots.{json,npz}. Rust's roots are not accurate
enough to be reference roots themselves; label_cases.py refines them with Newton on the frozen
graph. Rust returns at most two roots per case.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, quiet, read_json, seed_for, well_of, write_json  # noqa: E402
from manywells.simulator import SimError  # noqa: E402  (v1.0.0 API; manywells_rs raises it)
from scripts.load_well_from_dataset import load_well  # noqa: E402
import manywells_rs  # noqa: E402

N_FOLD_WELLS = 40


def rust_roots(wp, bc, n_cells=100):
    """Rust roots as flat states, highest p0 first; [] when there is none."""
    try:
        with quiet():
            return [np.asarray(x) for x in manywells_rs.SSDFSimulator(wp, bc, n_cells=n_cells).simulate()]
    except SimError:
        return []


def fold(args):
    df = pd.read_csv(args.data_dir / 'manywells-sol-1_config.zip', compression='zip')
    two_root = df[df['bc.p_r'] - df['bc.p_s'] < df['wp.rho_l'] * 9.81 * df['wp.L'] / 1e5]
    ids = np.random.default_rng(seed_for('fold')).permutation(two_root['ID'].to_numpy())
    rows = []
    for well_id in ids:
        well = load_well(int(well_id), df)
        wp, bc = well.wp, well.bc
        if len(rust_roots(wp, bc)) != 2:
            continue
        p_config = p_hi = p_lo = bc.p_r
        step = 0.02 * (bc.p_r - bc.p_s)
        while rust_roots(wp, bc) and bc.p_r - step > bc.p_s + 1.0:   # step down until no root
            p_hi = bc.p_r
            bc.p_r -= step
            p_lo = bc.p_r
        if rust_roots(wp, bc):
            continue                                                    # still flowing near p_s
        while p_hi - p_lo > 1e-4:                                       # bisect the fold
            bc.p_r = 0.5 * (p_hi + p_lo)
            if rust_roots(wp, bc):
                p_hi = bc.p_r
            else:
                p_lo = bc.p_r
        rows.append({'config_id': int(well_id), 'p_r_config': p_config, 'p_r_fold': p_hi, 'p_r_no_root': p_lo})
        print(f'well {well_id}: fold between p_r = {p_lo:.4f} and {p_hi:.4f} bar')
        if len(rows) == N_FOLD_WELLS:
            break
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / 'fold.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} fold brackets to {DATA / "fold.csv"}')


def roots(args):
    cases = read_json(DATA / 'cases.json')
    meta, arrays = {}, {}
    for case in cases:
        wp, bc = well_of(case)
        found = rust_roots(wp, bc, case['n_cells'])
        meta[case['case_id']] = [float(x[0]) for x in found]
        for k, x in enumerate(found):
            arrays[f'{case["case_id"]}.B{k}'] = x
    write_json(meta, DATA / 'rust_roots.json')
    np.savez_compressed(DATA / 'rust_roots.npz', **arrays)
    counts = pd.Series([len(v) for v in meta.values()]).value_counts().sort_index()
    print('Rust roots per case: ' + ', '.join(f'{n}: {c} cases' for n, c in counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='command', required=True)
    p_fold = sub.add_parser('fold', help='fold brackets for near-fold cases')
    p_fold.add_argument('--data-dir', type=Path, required=True, help='folder with manywells-sol-1_config.zip')
    sub.add_parser('roots', help='Rust roots for every case in data/full/cases.json')
    args = parser.parse_args()
    {'fold': fold, 'roots': roots}[args.command](args)


if __name__ == '__main__':
    main()
