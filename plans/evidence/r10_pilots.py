"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The measurements of R10's pilots (plans/sampler-redesign/r10-measurements.md): a sol-2 and an nsol-2 dataset written
by scripts/data_generation/v2/generate_dataset.py, each with its companion logs and its run folder, measured as R9's
pilots were (r9-measurements.md): the outcomes, the rates of the core wells by class and well type, the flow regimes,
the decline of an nsol-2 life by drive, the two datasets' states by stage (D-105), and the cost.

    (ulimit -v 8000000; OMP_NUM_THREADS=1 .venv/bin/python plans/evidence/r10_pilots.py SOL_OUT NSOL_OUT)

SOL_OUT and NSOL_OUT are the script's --out folders.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OILS = ('black_oil', 'volatile_oil')
YEAR = 52.1775  # weeks


def load(out, kind):
    out = Path(out)
    ds, logs = out / f'manywells-{kind}-2', out / f'manywells-{kind}-2-logs'
    t = {n: pd.read_parquet(ds / f'{n}.parquet') for n in ('wells', 'samples')}
    t |= {n: pd.read_parquet(logs / f'{n}.parquet') for n in ('drawn_wells', 'redraws')}
    t['metadata'] = json.loads((ds / 'metadata.json').read_text())
    return t


def q(s, ps=(10, 50, 90), fmt='{:.3g}'):
    return ' / '.join(fmt.format(v) for v in np.percentile(np.asarray(s, dtype=float), ps))


def pct(x):
    return f'{100 * float(x):.1f}%'


def oil_equivalent(s):
    return 24 * (s.QOIL + s.QGAS / 1000)  # Sm3/d, 1000 Sm3 of gas = 1 Sm3 of oil


def outcomes(t, kind):
    d = t['drawn_wells']
    kept = d.outcome == 'accepted'
    print(f'\n== {kind}-2: {kept.sum()} kept of {len(d)} drawn ({pct(kept.mean())})')
    for tail, g in d.groupby('coverage_tail', observed=True):
        print(f'  {tail}: {pct((g.outcome == "accepted").mean())} of {len(g)}')
    print('  outcomes: ' + ', '.join(f'{k} {v}' for k, v in d.outcome.value_counts().items()))
    core = d[d.coverage_tail == 'core']
    print(f'  redraw limit among core drawn wells: {pct((core.outcome == "redraw_limit").mean())}')
    by = d.groupby('fluid_class', observed=True).outcome.agg(lambda o: f'{(o == "accepted").sum()}/{len(o)}')
    print('  kept by class: ' + ', '.join(f'{k} {v}' for k, v in by.items()))
    s, r = t['samples'], t['redraws']
    accepted_redraws = r[r.ID.notna()] if 'ID' in r else r
    print(f'  redraws per sample: {len(accepted_redraws) / len(s):.3f}')


def rates(t):
    w, s = t['wells'].set_index('ID'), t['samples']
    s = s.join(w[['fluid_type', 'well_type', 'coverage_tail', 'lift']], on='ID')
    s = s.assign(oe=oil_equivalent(s), gas=24 * s.QGAS)
    med = s.groupby('ID').agg(oe=('oe', 'median'), gas=('gas', 'median'), gor=('GOR', 'median'))
    med = med.join(w[['fluid_type', 'well_type', 'coverage_tail', 'lift']])
    core = med[med.coverage_tail == 'core']
    off_oil = core[core.fluid_type.isin(OILS) & core.well_type.isin(['platform', 'subsea']) & (core.gor <= 1068)]
    gas = core[~core.fluid_type.isin(OILS)]
    print('\n== sol-2 rates, core wells, median-sample rate')
    print(
        f'  offshore oil (GOR <= 1068), Sm3/d oe, P10 / P50 / P90: {q(off_oil.oe)}; above 1018: '
        f'{pct((off_oil.oe > 1018).mean())}; mean {off_oil.oe.mean():.0f} ({len(off_oil)} wells)'
    )
    print(
        f'  gas and condensate, Sm3/d gas, P10 / P50 / P90: {q(gas.gas)}; above 270,000: '
        f'{pct((gas.gas > 270_000).mean())} ({len(gas)} wells)'
    )
    for wt, g in core.groupby('well_type', observed=True):
        o, gg = g[g.fluid_type.isin(OILS)], g[~g.fluid_type.isin(OILS)]
        print(
            f'  {wt}: oil wells {len(o)}, P50 {o.oe.median():.0f}, above 1018 {pct((o.oe > 1018).mean())}, '
            f'below 16 {pct((o.oe < 16).mean())}, lifted {pct((o.lift == "gas_lift").mean())}; gas wells {len(gg)}, '
            f'P50 {gg.gas.median():.3g}, above 270,000 {pct((gg.gas > 270_000).mean())}'
        )


def regimes(t, kind):
    s = t['samples'].join(t['wells'].set_index('ID')[['coverage_tail']], on='ID')
    for col, where in (('FRBH', 'bottom'), ('FRWH', 'top')):
        share = s[col].astype(str).value_counts(normalize=True)
        print(f'  {kind}-2 regimes at the {where}: ' + ', '.join(f'{k} {pct(v)}' for k, v in share.items()))


def decline(t):
    w, s = t['wells'].set_index('ID'), t['samples']
    s = s.join(w[['T_end', 'drive', 'fluid_type', 'coverage_tail']], on='ID')
    s = s[s.coverage_tail == 'core'].assign(oe=lambda x: oil_equivalent(x), years=lambda x: x.WEEKS / YEAR)
    s['tf'] = s.years / s.T_end
    s['grp'] = np.where(s.fluid_type.isin(OILS), 'oil', 'gas') + ', ' + s.drive.astype(str)
    rows = []
    for ID, g in s.groupby('ID'):
        first = g.loc[g.years <= 1.0, 'oe'].mean()
        if not first > 0:
            continue
        fifths = [
            g.loc[(g.tf > i / 5) & (g.tf <= (i + 1) / 5) if i else (g.tf <= 0.2), 'oe'].mean() / first for i in range(5)
        ]
        rows.append(dict(ID=ID, grp=g.grp.iloc[0], **{f'f{i + 1}': v for i, v in enumerate(fifths)}))
    r = pd.DataFrame(rows)
    print('\n== nsol-2 decline, core wells, mean rate per fifth of the life over the first year, medians')
    print('  all: ' + ', '.join(f'{r[f"f{i}"].median():.2f}' for i in range(1, 6)))
    for grp, g in r.groupby('grp'):
        print(f'  {grp} ({len(g)} wells): last fifth {g.f5.median():.2f}')


def stages(sol, nsol):
    w = sol['wells']
    w = w[w.coverage_tail == 'core'].set_index('ID')
    ss = sol['samples'].groupby('ID').agg(p=('PRES', 'median'), wlr=('WLR', 'median'))
    ss = ss.join(w[['p_i', 'drive', 'fluid_type', 'life_stage']], how='inner')
    ss['r_p'] = ss.p / ss.p_i
    n = nsol['samples'].join(nsol['wells'].set_index('ID')[['p_i', 'drive', 'fluid_type', 'coverage_tail']], on='ID')
    n = n[n.coverage_tail == 'core'].assign(r_p=lambda x: x.PRES / x.p_i)
    print("\n== the two datasets by stage (core), p_r/p_i and the oils' water cut, medians, nsol-2 / sol-2")
    for (stage, drive), g in n.groupby(['STAGE', 'drive'], observed=True):
        h = ss[(ss.life_stage == stage) & (ss.drive == drive)]
        go, ho = g[g.fluid_type.isin(OILS)], h[h.fluid_type.isin(OILS)]
        print(
            f'  {stage}, {drive}: {g.r_p.median():.2f} / {h.r_p.median():.2f}; '
            f'{go.WLR.median():.2f} / {ho.wlr.median():.2f}'
        )


def cost(t, kind, attempts_dir):
    d, md = t['drawn_wells'], t['metadata']
    cpu = d.cpu_seconds.sum()
    kept = (d.outcome == 'accepted').sum()
    att = pd.concat([pd.read_parquet(f) for f in sorted(Path(attempts_dir).glob('*.parquet'))])
    att = att.join(d.set_index('draw_index')[['fluid_class']], on='draw_index')
    print(f'\n== {kind}-2 cost')
    print(
        f'  CPU per accepted well: {cpu / kept:.1f} s; projected for 10,000 wells: {cpu / kept * 1e4 / 3600:.0f} CPU-hours'
    )
    print(f'  table builds: {d.table_seconds.fillna(0).sum() / 3600:.2f} CPU-hours over {len(d)} drawn wells')
    t_solve = att.t_solve.dropna()
    print(
        f'  solve, median / mean / P95: {t_solve.median():.3f} / {t_solve.mean():.3f} / {t_solve.quantile(0.95):.3f} s'
    )
    by = att.groupby('fluid_class', observed=True).t_solve.mean()
    print('  mean solve by class: ' + ', '.join(f'{k} {v:.3f}' for k, v in by.items()))
    print(f'  code: {md.get("code", md.get("code_version"))}')


def main():
    sol_out, nsol_out = sys.argv[1:3]
    sol, nsol = load(sol_out, 'sol'), load(nsol_out, 'nsol')
    outcomes(sol, 'sol')
    outcomes(nsol, 'nsol')
    rates(sol)
    print('\n== flow regimes')
    regimes(sol, 'sol')
    regimes(nsol, 'nsol')
    decline(nsol)
    stages(sol, nsol)
    cost(sol, 'sol', Path(sol_out) / 'run' / 'attempts')
    cost(nsol, 'nsol', Path(nsol_out) / 'run' / 'attempts')


if __name__ == '__main__':
    main()
