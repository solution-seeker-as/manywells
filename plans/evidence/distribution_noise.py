"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The noise and the bias of the verifier's Distributions check (specs/verification.md, Distributions; model-lock plan,
Step L6). Regenerations of the published sol-1 wells (scripts/verification/regenerate_distributions.py) with other
seeds give the gaps' spread from draw to draw; subsamples of the reference's own rows give the spread a candidate
with no bias would have; and a replay of the same draws with v1.0.0's solver, as its generator solved them, says
which regenerated rows the reference lacks because v1.0.0 failed them or put them on the trickle signature.

  regen      run the regeneration for each seed, writing DIR/<backend>-n<N>-seed<S>.parquet and its log
  gaps       each regeneration's gaps per feature: the CDF gap, its percentile and sign, and the largest
             rank-correlation gap in the feature's row of the matrix, to a CSV
  reference  the same for draws of N rows per well from the reference's own rows (the published sol-1 rows off the
             trickle signature), weighted as a candidate is
  summary    mean, standard deviation and maximum of each feature's gaps over the seeds, per source
  draws      every draw of a seed, for the v1.0.0 replay (develop's sampler, SMP-18 to SMP-22 and SMP-31)
  v1         solve the draws with v1.0.0 as its generator did: from the well's root at u = 0.5, which v1.0.0 solves
             from its default guess. Runs in the v1.0.0 environment and writes v1.0.0's row of each draw
  drivers    where a regeneration's gap in one feature (PWH by default) sits: its percentiles, the wells it comes
             from, and the gaps with the draws that v1.0.0 fails, or puts on the trickle signature, left out
  roots      every root set of a seed's draws on the Rust core: the stable root's row and, where there is one, the
             trickle root's (the unstable root with the highest p_0), for power
  power      what the check catches: candidates that take the trickle root instead of the stable one in a share of
             the draws that have one, spread over the wells as a solver's defect would be
  inputs     what the check catches of a sampler error: one drawn input's column of a regeneration redrawn from a
             wrong range (the solved features, which would move too, are left as they are)

From the repository root, with the sol-1 files of the public Hugging Face dataset solution-seeker-as/manywells:

    uv run python plans/evidence/distribution_noise.py regen DIR --backend rust --samples 5 --seeds 20261001 1 2
    uv run python plans/evidence/distribution_noise.py gaps DIR gaps.csv
    uv run python plans/evidence/distribution_noise.py reference manywells-sol-1.zip reference.csv --samples 5
    uv run python plans/evidence/distribution_noise.py summary gaps.csv reference.csv
    uv run python plans/evidence/distribution_noise.py draws draws.json --config manywells-sol-1_config.zip
    MANYWELLS_V1=.worktrees/v1.0.0 .worktrees/v1.0.0/.venv/bin/python plans/evidence/distribution_noise.py \
        v1 draws.json v1.json --config manywells-sol-1_config.zip
    uv run python plans/evidence/distribution_noise.py drivers DIR/rust-n5-seed20261001.parquet --v1 v1.json \
        --data manywells-sol-1.zip [--feature PWH]
    uv run python plans/evidence/distribution_noise.py roots roots.parquet --config manywells-sol-1_config.zip
    uv run python plans/evidence/distribution_noise.py power roots.parquet
    uv run python plans/evidence/distribution_noise.py inputs DIR/rust-n5-seed20261001.parquet \
        --config manywells-sol-1_config.zip

The draws of regen, draws and v1 do not depend on the backend (SMP-31), so a row's ID and sample index k identify
it in every file.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEED = 20261001  # the regeneration's default seed (scripts/verification/regenerate_distributions.py)
SIGNATURE_DP = 0.01  # bar, the trickle-root signature (TRICKLE_DP in manywells_verify.distributions)


# ---------------------------------------------------------------------------------------------
# Gaps (develop environment)


def reference_summary():
    from manywells_verify.distributions import REFERENCE

    return json.loads(REFERENCE.read_text())['datasets']['sol-1']


def feature_gaps(reference: dict, df: pd.DataFrame) -> list:
    """
    Per feature, the check's CDF gap with its percentile and sign (candidate minus reference), and the largest
    rank-correlation gap in the feature's row, with the candidate weighted as the check weights it.
    """
    from manywells_verify.distributions import FEATURES, PERCENTILES, weighted_cdf, weighted_spearman, well_weights

    w = well_weights(reference, df)
    df, w = df[w > 0], w[w > 0]
    diff = np.abs(weighted_spearman(df, w) - np.array(reference['spearman']))
    out = []
    for i, f in enumerate(FEATURES):
        q, F = np.array(reference['features'][f]['values']), np.array(reference['features'][f]['cdf'])
        d = weighted_cdf(df[f].to_numpy(dtype=float), w, q) - F
        j, c = int(np.argmax(np.abs(d))), int(np.argmax(diff[i]))
        out.append(
            dict(
                feature=f,
                cdf=float(abs(d[j])),
                sign=int(np.sign(d[j])),
                percentile=int(PERCENTILES[j]),
                corr=float(diff[i, c]),
                corr_with=FEATURES[c],
            )
        )
    return out


def regen(args):
    args.dir.mkdir(parents=True, exist_ok=True)
    for n, seed in [(n, seed) for n in args.samples for seed in args.seeds]:
        name = f'{args.backend}-n{n}-seed{seed}'
        cmd = [sys.executable, '-m', 'scripts.verification.regenerate_distributions', '--samples', str(n)]
        cmd += ['--backend', args.backend, '--seed', str(seed), '--processes', str(args.processes)]
        cmd += ['--out', str(args.dir / f'{name}.parquet')] + (['--config', str(args.config)] if args.config else [])
        t0 = time.perf_counter()
        with open(args.dir / f'{name}.log', 'w') as log:
            subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'{name}: {time.perf_counter() - t0:.0f} s', flush=True)


def gaps(args):
    reference = reference_summary()
    rows = []
    for path in sorted(args.dir.glob('*-n*-seed*.parquet')):
        backend, n, seed = path.stem.split('-')
        df = pd.read_parquet(path)
        meta = dict(source=backend, samples=int(n[1:]), seed=int(seed[4:]), rows=len(df))
        rows += [meta | g for g in feature_gaps(reference, df)]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'{len(rows)} rows to {args.out}')


def reference_noise(args):
    from manywells_verify.distributions import trickle_signature

    reference = reference_summary()
    stable = pd.read_csv(args.data, compression='zip')
    stable = stable[~trickle_signature(stable)].reset_index(drop=True)
    rows = []
    for n in args.samples:
        for draw in range(args.draws):
            rng = np.random.default_rng([SEED, n, draw])
            df = stable.iloc[rng.permutation(len(stable))].groupby('ID').head(n)
            meta = dict(source='reference', samples=n, seed=draw, rows=len(df))
            rows += [meta | g for g in feature_gaps(reference, df)]
        print(f'{n} rows per well: {args.draws} draws', flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


def summary(args):
    df = pd.concat([pd.read_csv(p) for p in args.gaps], ignore_index=True)
    keys = ['source', 'samples']
    runs = df.groupby(keys + ['seed']).agg(cdf=('cdf', 'max'), corr=('corr', 'max'), rows=('rows', 'first'))
    print("The check's statistics, the largest gap over the features, per run:")
    agg = runs.groupby(keys).agg(
        runs=('cdf', 'size'),
        rows=('rows', 'mean'),
        cdf_mean=('cdf', 'mean'),
        cdf_sd=('cdf', 'std'),
        cdf_max=('cdf', 'max'),
        corr_mean=('corr', 'mean'),
        corr_sd=('corr', 'std'),
        corr_max=('corr', 'max'),
    )
    print(agg.round(4).to_string(), '\n')
    per = df.groupby(keys + ['feature'], sort=False).agg(
        cdf_mean=('cdf', 'mean'),
        cdf_sd=('cdf', 'std'),
        cdf_max=('cdf', 'max'),
        corr_mean=('corr', 'mean'),
        corr_sd=('corr', 'std'),
        corr_max=('corr', 'max'),
    )
    for key, block in per.groupby(level=[0, 1], sort=False):
        print(f'{key[0]}, {key[1]} per well, per feature:')
        print(block.droplevel([0, 1]).round(4).to_string(), '\n')
    if args.out:
        per.reset_index().to_csv(args.out, index=False)


# ---------------------------------------------------------------------------------------------
# The v1.0.0 replay


def write_draws(args):
    from manywells.sampling.conditions import sample_conditions
    from manywells.sampling.wells import rng_for, well_from_config

    wells = pd.read_csv(args.config, compression='zip').to_dict('records')
    out = {}
    for row in wells:
        draw, ks = well_from_config(row), []
        for k in range(args.samples):
            bc, (f_g, f_o, f_w) = sample_conditions(draw, rng_for(args.seed, row['ID'], k, 'sample'))
            ks.append([k, bc.u, bc.w_lg, bc.p_s, bc.p_r, f_g, f_o, f_w])
        out[str(row['ID'])] = ks
    args.out.write_text(json.dumps({'seed': args.seed, 'samples': args.samples, 'draws': out}))
    print(f'{sum(len(v) for v in out.values())} draws of {len(out)} wells to {args.out}')


def _v1_row(sim, x, well, new, pvt):
    """v1.0.0's dataset row of a solution (generate_well_data.simulate_well)."""
    df_x = sim.solution_as_df(x)
    w_g = float((sim.wp.A * df_x['alpha'] * df_x['rho_g'] * df_x['v_g']).iloc[-1])
    w_l = float((sim.wp.A * (1 - df_x['alpha']) * df_x['rho_l'] * df_x['v_l']).iloc[-1])
    f_g, f_o, f_w = new.fractions
    wlf = f_w / (f_o + f_w)
    w_w, w_o, w_lg = w_l * wlf, w_l * (1 - wlf), sim.bc.w_lg
    rho_g = pvt.gas_density(sim.wp.R_s)
    q_g, q_lg, q_l = 3600 * w_g / rho_g, 3600 * w_lg / rho_g, 3600 * w_l / sim.wp.rho_l
    q_o, q_w = 3600 * w_o / well.oil.rho, 3600 * w_w / well.water.rho
    pwh = float(df_x['p'].iloc[-1])
    return {
        'CHK': sim.bc.u,
        'PBH': float(df_x['p'].iloc[0]),
        'PWH': pwh,
        'PDC': sim.bc.p_s,
        'TBH': sim.bc.T_r,
        'TWH': float(df_x['T'].iloc[-1]),
        'WGL': w_lg,
        'WGAS': w_g - w_lg,
        'WLIQ': w_l,
        'WOIL': w_o,
        'WWAT': w_w,
        'WTOT': w_g + w_l,
        'QGL': q_lg,
        'QGAS': q_g - q_lg,
        'QLIQ': q_l,
        'QOIL': q_o,
        'QWAT': q_w,
        'QTOT': q_g + q_l,
        'FGAS': f_g,
        'FOIL': f_o,
        'FWAT': f_w,
    }


def _v1_well(task):
    """One well's draws solved as v1.0.0's sol-1 generator solved its samples."""
    import contextlib
    import io

    import manywells.pvt as pvt
    from manywells.simulator import SimError, SSDFSimulator
    from scripts.load_well_from_dataset import load_well

    config, draws = task
    well = load_well(config['ID'], pd.DataFrame([config]))
    sim = SSDFSimulator(well.wp, well.bc)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):  # v1.0.0 prints Ipopt's status on every failed solve
        try:
            sim.bc.u = 0.5
            sim.x_guess = sim.simulate()
        except SimError:
            return [{'ID': config['ID'], 'k': k, 'v1': 'no u = 0.5 root'} for k, *_ in draws]
        for k, u, w_lg, p_s, p_r, f_g, f_o, f_w in draws:
            new = well.copy()  # Well.sample_new_conditions with develop's draws
            new.bc.u, new.bc.p_s, new.bc.p_r = u, p_s, p_r
            if well.has_gas_lift:
                new.bc.w_lg = w_lg
            new.fractions = (f_g, f_o, f_w)
            mix = pvt.liquid_mix(well.oil, well.water, f_o / (f_o + f_w))
            new.wp.rho_l, new.wp.cp_l, new.wp.inflow.f_g = mix.rho, mix.cp, f_g
            sim.wp, sim.bc = new.wp, new.bc
            rec = {'ID': config['ID'], 'k': k}
            try:
                x = sim.simulate()
            except SimError:
                out.append(rec | {'v1': 'failed'})
                continue
            r = _v1_row(sim, x, well, new, pvt)
            if r['WLIQ'] < 0 or r['WGAS'] + r['WGL'] < 0:
                status = 'invalid'
            elif r['WTOT'] < 0.1:
                status = 'dropped'
            elif r['PWH'] - r['PDC'] < SIGNATURE_DP:
                status = 'signature'
            else:
                status = 'row'
            out.append(rec | r | {'v1': status})
    return out


def v1_replay(args):
    import multiprocessing

    v1 = os.environ.get('MANYWELLS_V1')
    if not v1:
        sys.exit('set MANYWELLS_V1 to a v1.0.0 checkout and run this in its environment')
    sys.path.insert(0, str(Path(v1).resolve()))
    draws = json.loads(args.draws.read_text())['draws']
    configs = pd.read_csv(args.config, compression='zip').to_dict('records')
    tasks = [(c, draws[str(c['ID'])]) for c in configs if str(c['ID']) in draws][: args.wells]
    t0 = time.perf_counter()
    with multiprocessing.Pool(args.processes) as pool:
        rows = [r for rs in pool.map(_v1_well, tasks, chunksize=1) for r in rs]
    args.out.write_text(json.dumps(rows))
    counts = pd.Series([r['v1'] for r in rows]).value_counts().to_dict()
    print(f'{len(rows)} draws of {len(tasks)} wells in {time.perf_counter() - t0:.0f} s: {counts}')


# ---------------------------------------------------------------------------------------------
# What drives a gap


def drivers(args):
    from manywells_verify.distributions import FEATURES, PERCENTILES, weighted_cdf, well_weights

    f, reference = args.feature, reference_summary()
    rows = pd.read_parquet(args.rows)
    q, F_ref = np.array(reference['features'][f]['values']), np.array(reference['features'][f]['cdf'])
    w = well_weights(reference, rows)
    d = weighted_cdf(rows[f].to_numpy(dtype=float), w, q) - F_ref
    j = int(np.argmax(np.abs(d)))
    print(f'{args.rows.name}: {len(rows)} rows; {f} CDF gap {d[j]:+.4f} at percentile {PERCENTILES[j]} ({q[j]:.4g})')
    print('  candidate minus reference CDF, every fifth percentile:')
    print('    ' + ', '.join(f'P{p} {g:+.4f}' for p, g in zip(PERCENTILES, d) if p % 5 == 0))

    if args.data:
        # The gap at its percentile, well by well: n_ref(k) / N (F_k,cand - F_k,ref), with the reference's own rows
        from manywells_verify.distributions import trickle_signature

        pub = pd.read_csv(args.data, compression='zip', usecols=sorted({'ID', 'PWH', 'PDC', f}))
        sig = trickle_signature(pub)
        stable = pub[~sig]
        n_ref = stable.groupby('ID').size()
        N = n_ref.sum()
        F_ref_k = stable.assign(b=stable[f] <= q[j]).groupby('ID')['b'].mean()
        F_cand_k = rows.assign(b=rows[f] <= q[j]).groupby('ID')['b'].mean()
        ids = n_ref.index
        c = (n_ref / N * (F_cand_k.reindex(ids) - F_ref_k)).fillna(0.0)
        missing = ids.difference(F_cand_k.index)
        cut = pub[sig].groupby('ID').size().reindex(ids, fill_value=0) > 0
        print(
            f'  by well at P{PERCENTILES[j]}: sum {c.sum():+.4f}; wells without candidate rows {len(missing)} '
            f'(reference share {n_ref[missing].sum() / N:.4f}); the {cut.sum()} wells that lost signature rows '
            f'{c[cut].sum():+.4f}, the others {c[~cut].sum():+.4f}; largest single well {c.abs().max():.4f}'
        )

    if args.v1:
        v1 = pd.DataFrame(json.loads(args.v1.read_text()))
        status = rows.merge(v1[['ID', 'k', 'v1']], on=['ID', 'k'], how='left')['v1'].fillna('not replayed').to_numpy()
        print(f'  v1.0.0 on the {len(rows)} regenerated rows, with their {f} quartiles:')
        for s, block in rows.groupby(status):
            print(f'    {s:>16s} {len(block):5d}: {np.percentile(block[f], [25, 50, 75]).round(3).tolist()}')
        failed = v1[v1['v1'] != 'row'].merge(rows[['ID', 'k']], on=['ID', 'k'], how='left', indicator=True)
        print(
            "  v1.0.0's draws without a row: "
            + ', '.join(f'{k} {v}' for k, v in failed['v1'].value_counts().items())
            + f'; of them {(failed["_merge"] == "left_only").sum()} without a row in the regeneration either'
        )

        def report(name, df):
            g = pd.DataFrame(feature_gaps(reference, df)).set_index('feature')
            worst = g['cdf'].idxmax()
            print(
                f'  {name:<56s} {len(df):5d} rows: {f} {g.loc[f, "cdf"]:.4f}, largest CDF {g["cdf"].max():.4f} '
                f'({worst}), rank correlation {g["corr"].max():.4f}'
            )

        report('all regenerated rows', rows)
        report('without the draws v1.0.0 fails (failed, invalid, dropped)', rows[np.isin(status, ['row', 'signature'])])
        report('... and without those it puts on the signature', rows[status == 'row'])
        mine = v1[v1['v1'] == 'row']
        report("v1.0.0's own rows of these draws, off the signature", mine[list(FEATURES) + ['ID', 'k']])
        both = rows.merge(mine[['ID', 'k', 'PBH']], on=['ID', 'k'], suffixes=('', '_v1'))
        other = (both['PBH'] - both['PBH_v1']).abs() > 0.01
        print(f'  draws where v1.0.0 has a row off the signature at another PBH (> 0.01 bar): {other.sum()}')


# ---------------------------------------------------------------------------------------------
# What the check catches


def _roots_well(task):
    """The stable and the trickle row of each of one well's draws, from the Rust core's root sets."""
    from manywells.configurations import V1
    from manywells.datasets.rows import sample_row
    from manywells.sampling.conditions import sample_conditions
    from manywells.sampling.wells import rng_for, well_from_config, well_properties
    from manywells.simulator import SimError, SSDFSimulator

    config, seed, n = task
    draw, out = well_from_config(config), []
    for k in range(n):
        bc, fractions = sample_conditions(draw, rng_for(seed, config['ID'], k, 'sample'))
        try:
            rs = SSDFSimulator(well_properties(draw, fractions, V1), backend='rust').root_set(bc)
        except (SimError, ValueError):
            continue
        if rs.operating_point is None:
            continue
        out.append(sample_row(rs.operating_point, draw, fractions, bc) | {'ID': config['ID'], 'k': k, 'root': 'stable'})
        unstable = [r for r in rs.roots if r.label == 'unstable']
        if unstable:
            r = sample_row(max(unstable, key=lambda r: r.p_0), draw, fractions, bc)
            out.append(r | {'ID': config['ID'], 'k': k, 'root': 'trickle'})
    return out


def roots(args):
    import multiprocessing

    wells = pd.read_csv(args.config, compression='zip').to_dict('records')
    t0 = time.perf_counter()
    with multiprocessing.Pool(args.processes) as pool:
        rows = pool.map(_roots_well, [(w, args.seed, args.samples) for w in wells], chunksize=1)
    df = pd.DataFrame([r for rs in rows for r in rs])
    df.to_parquet(args.out, index=False)
    print(f'{(df.root == "stable").sum()} stable and {(df.root == "trickle").sum()} trickle rows in ', end='')
    print(f'{time.perf_counter() - t0:.0f} s')


def power(args):
    from manywells_verify.distributions import FEATURES, compare, well_weights

    reference, F = reference_summary(), list(FEATURES) + ['ID', 'k']
    df = pd.read_parquet(args.roots)
    stable = df[(df.root == 'stable') & (df.WTOT >= 0.1)].set_index(['ID', 'k'])
    trickle = df[df.root == 'trickle'].set_index(['ID', 'k'])
    trickle = trickle[trickle.index.isin(stable.index)]
    print(f'{len(stable)} draws with a stable row, {len(trickle)} of them with a trickle root')
    for share in args.shares:
        for rep in range(args.draws):
            rng = np.random.default_rng([SEED, rep, int(1000 * share)])
            take = trickle[rng.random(len(trickle)) < share]
            kept = take[take.WTOT >= 0.1]  # a trickle root below 0.1 kg/s is dropped, as the generator drops it
            cand = pd.concat([stable.drop(take.index), kept]).reset_index()[F]
            weighted, equal = compare(reference, cand, well_weights(reference, cand)), compare(reference, cand)
            print(
                f'  share {share:4.2f}, draw {rep}: {len(take)} trickle roots, {len(kept)} rows '
                f'({len(kept) / len(cand):.3f} of the rows, {len(take) - len(kept)} dropped); '
                f'weighted CDF {weighted["worst_cdf"][1]:.4f} ({weighted["worst_cdf"][0]}), rank correlation '
                f'{weighted["worst_corr"][2]:.4f}; unweighted {equal["worst_cdf"][1]:.4f} ({equal["worst_cdf"][0]}), '
                f'{equal["worst_corr"][2]:.4f}'
            )


def inputs(args):
    from manywells_verify.distributions import compare, well_weights

    reference, rows = reference_summary(), pd.read_parquet(args.rows)
    cfg = pd.read_csv(args.config, compression='zip').set_index('ID').loc[rows['ID']]
    rng, n = np.random.default_rng(SEED), len(rows)
    lift = cfg['has_gas_lift'].to_numpy(dtype=bool)
    errors = [
        ('choke U(0.05, 1), redrawn (no error: the control)', 'CHK', rng.uniform(0.05, 1, n)),
        ('choke U(0.1, 1) (SMP-18: U(0.05, 1))', 'CHK', rng.uniform(0.1, 1, n)),
        ('lift gas U(0, 4) (SMP-19: U(0, 5))', 'WGL', np.where(lift, rng.uniform(0, 4, n), 0.0)),
        ('separator pressure +-20% (SMP-20: +-10%)', 'PDC', cfg['bc.p_s'].to_numpy() * rng.uniform(0.8, 1.2, n)),
        (
            'gas fraction +-10% (SMP-22: +-5%)',
            'FGAS',
            np.minimum(0.99, cfg['fraction.gas'].to_numpy() * rng.uniform(0.9, 1.1, n)),
        ),
    ]
    for name, f, x in errors:
        r = compare(reference, rows.assign(**{f: x}), well_weights(reference, rows))
        print(f'{name:50s} {f} CDF gap {r["cdf_gaps"][f]:.4f}; largest {r["worst_cdf"][1]:.4f} ({r["worst_cdf"][0]})')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('regen')
    p.add_argument('dir', type=Path)
    p.add_argument('--backend', choices=('casadi', 'rust'), required=True)
    p.add_argument('--samples', type=int, nargs='+', default=[5])
    p.add_argument('--seeds', type=int, nargs='+', default=[SEED])
    p.add_argument('--processes', type=int, default=12)
    p.add_argument('--config', type=Path)
    p.set_defaults(run=regen)
    p = sub.add_parser('gaps')
    p.add_argument('dir', type=Path)
    p.add_argument('out', type=Path)
    p.set_defaults(run=gaps)
    p = sub.add_parser('reference')
    p.add_argument('data', type=Path, help='manywells-sol-1.zip')
    p.add_argument('out', type=Path)
    p.add_argument('--samples', type=int, nargs='+', default=[5])
    p.add_argument('--draws', type=int, default=20)
    p.set_defaults(run=reference_noise)
    p = sub.add_parser('summary')
    p.add_argument('gaps', type=Path, nargs='+')
    p.add_argument('--out', type=Path)
    p.set_defaults(run=summary)
    p = sub.add_parser('draws')
    p.add_argument('out', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--samples', type=int, default=5)
    p.set_defaults(run=write_draws)
    p = sub.add_parser('v1')
    p.add_argument('draws', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--processes', type=int, default=12)
    p.add_argument('--wells', type=int, help='only the first WELLS wells')
    p.set_defaults(run=v1_replay)
    p = sub.add_parser('drivers')
    p.add_argument('rows', type=Path)
    p.add_argument('--feature', default='PWH')
    p.add_argument('--v1', type=Path, help="the v1 command's output for the rows' seed")
    p.add_argument('--data', type=Path, help='manywells-sol-1.zip, for the decomposition by well')
    p.set_defaults(run=drivers)
    p = sub.add_parser('roots')
    p.add_argument('out', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--processes', type=int, default=12)
    p.set_defaults(run=roots)
    p = sub.add_parser('power')
    p.add_argument('roots', type=Path)
    p.add_argument('--shares', type=float, nargs='+', default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument('--draws', type=int, default=3)
    p.set_defaults(run=power)
    p = sub.add_parser('inputs')
    p.add_argument('rows', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.set_defaults(run=inputs)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
