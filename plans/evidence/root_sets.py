"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Root sets and stability labels with v1.0.0 on sampled sol-1 configs (v2 plan, "Evidence").

For each sampled well, runs v1 from its default guess plus five cellwise guesses spread
over (p_s, p_r), collects the distinct roots, and labels each with stability_label.Graph.
Wells are sampled with seed 1: n2 that meet the two-root criterion and n1 that do not.

Runs against v1.0.0, from the root of a v1.0.0 checkout (or set MANYWELLS_V1 to it):

    python <repo>/plans/evidence/root_sets.py --config manywells-sol-1_config.zip 40 10 out.csv
"""

import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

from stability_label import Graph, SimError, load_well

GUESSES = [None, 0.5, 0.7, 0.85, 0.95, 0.995]  # None = v1 default; else fraction of (p_s, p_r)


def root_set(sim):
    """Distinct roots (p0 rounded to 1e-3 bar) reached from GUESSES, and the default guess's root."""
    bc = sim.bc
    roots, default = {}, None
    for f in GUESSES:
        with contextlib.redirect_stdout(io.StringIO()):  # v1 prints on failure
            try:
                sim.x_guess = None if f is None else sim._simulate_cellwise(bc.p_s + f * (bc.p_r - bc.p_s), bc.T_r)
                xs = np.asarray(sim.simulate())
            except SimError:
                continue
        d, _ = sim.dR_dp0(xs)
        p0 = round(xs[0], 3)
        X = xs.reshape(-1, 7)
        roots.setdefault(p0, dict(d=d, twh=X[-1, 6], drawdown=bc.p_r - xs[0], min_v=X[:, 1:3].min()))
        if f is None:
            default = p0
    return roots, default


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[2])
    parser.add_argument('--config', required=True, help='path to manywells-sol-1_config.zip')
    parser.add_argument('n2', type=int, help='number of wells meeting the two-root criterion')
    parser.add_argument('n1', type=int, help='number of wells not meeting it')
    parser.add_argument('out', help='output CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.config, compression='zip')
    two = (df['bc.p_r'] - df['bc.p_s']) < df['wp.rho_l'] * 9.81 * df['wp.L'] / 1e5
    rng = np.random.default_rng(1)
    ids = list(rng.choice(df.loc[two, 'ID'], args.n2, replace=False)) + \
        list(rng.choice(df.loc[~two, 'ID'], args.n1, replace=False))
    print(f'two-root criterion true for {two.sum()} of {len(df)} configs; gas lift in {(df["bc.w_lg"] > 0).sum()}')

    rows = []
    for wid in ids:
        well = load_well(int(wid), df)
        sim = Graph(well.wp, well.bc).build()
        t = time.time()
        roots, default = root_set(sim)
        stable = [p for p, r in roots.items() if r['d'] < 0]
        unstable = [p for p, r in roots.items() if r['d'] > 0]
        rows.append(dict(
            id=int(wid), crit_two=bool(two[df['ID'] == wid].iloc[0]), n_roots=len(roots),
            n_stable=len(stable), n_unstable=len(unstable),
            v1_default='failed' if default is None else ('unstable' if roots[default]['d'] > 0 else 'stable'),
            twh_stable=roots[stable[0]]['twh'] if stable else np.nan,
            twh_unstable=roots[unstable[0]]['twh'] if unstable else np.nan,
            drawdown_unstable=roots[unstable[0]]['drawdown'] if unstable else np.nan,
            unstable_above_stable=(min(unstable) > max(stable)) if (stable and unstable) else None,
            secs=time.time() - t))
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.groupby('crit_two')[['n_roots', 'n_stable', 'n_unstable']].value_counts())
    print(out.groupby('crit_two')['v1_default'].value_counts())
    print(out[['twh_stable', 'twh_unstable', 'drawdown_unstable', 'secs']].describe())


if __name__ == '__main__':
    main()
