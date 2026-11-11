"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 11 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The checks of R11's pilot (plans/sampler-redesign/r11-measurements.md): r10_pilots.py's measurements, and the
spec's checks that R11's pilot judges: SMP-106's targets by class (wells dropped at the limits, table failures among
them; nsol-2's dead dates; solver failures among the attempts and the wells), with the sign changes D-138 counts at
the edge; the core wells' reservoir temperatures and pressure gradients against Nadeau, Sun and Ehrenberg's (2023)
classes of reservoirs (Table 1, Fig. 20); the gas wells' separator pressures and rates after D-134; and nsol-2's rate
by fifth of the life, by fluid and drive, after D-135 and D-140.

    (ulimit -v 8000000; OMP_NUM_THREADS=1 .venv/bin/python plans/evidence/r11_pilot.py SOL_OUT NSOL_OUT)

SOL_OUT and NSOL_OUT are generate_dataset.py's --out folders.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import r10_pilots as r10  # noqa: E402

LIMIT_DROPS = ('redraw_limit', 'slot_limit', 'solver_failure_limit', 'dead_date', 'table_failed')
CLASSES = ('black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas')
NADEAU_T = ((331.15, 'below 58 C', 0.220), (395.15, '58 to 122 C', 0.651), (np.inf, 'above 122 C', 0.129))  # Table 1
NADEAU_SG = ((0.8, 'below 0.8', 0.060), (1.3, '0.8 to 1.3', 0.778), (np.inf, 'above 1.3', 0.162))  # Fig. 20
HYDROSTATIC = 0.0979  # bar/m, fresh water (Nadeau et al. 2023, Methods)


def pct(x):
    return f'{100 * float(x):.1f}%'


def attempts(out):
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(out, 'run', 'attempts').glob('*.parquet'))])


def targets(t, kind, out):
    """SMP-106's targets: at most 1% of a class's drawn wells dropped at the limits; at most 0.5% of nsol-2's wells
    as dead_date; solver failures in at most 0.5% of the attempts and of the wells of every class."""
    d = t['drawn_wells']
    att = attempts(out).join(d.set_index('draw_index')[['fluid_class', 'coverage_tail']], on='draw_index')
    print(f'\n== {kind}-2: SMP-106 targets, by class (drawn wells; core / tail)')
    for c in CLASSES:
        g = d[d.fluid_class == c]
        if not len(g):
            continue
        drops = g.outcome.isin(LIMIT_DROPS)
        by = {k: v for k, v in g[drops].outcome.value_counts().items() if v}
        a = att[att.fluid_class == c]
        sf_att = (a.outcome == 'solver_failure').mean()
        sf_wells = (g.n_redraws_solver > 0).mean() if 'n_redraws_solver' in g else float('nan')
        above = [
            n for n, x in (('drops', drops.mean() > 0.01), ('solver failures', max(sf_att, sf_wells) > 0.005)) if x
        ]
        flag = f' ABOVE: {", ".join(above)}' if above else ''
        print(
            f'  {c}: {len(g)} drawn ({(g.coverage_tail == "core").sum()} / {(g.coverage_tail == "tail").sum()}); '
            f'dropped at the limits {drops.sum()} ({pct(drops.mean())}) {by}; solver failures: '
            f'{pct(sf_att)} of {len(a)} attempts, {pct(sf_wells)} of the wells{flag}'
        )
    if kind == 'nsol':
        print(
            f'  dead dates: {(d.outcome == "dead_date").sum()} of {len(d)} ({pct((d.outcome == "dead_date").mean())})'
        )
    reasons = att[att.outcome != 'operating_point'].reason.value_counts().to_dict()
    edge = int((att.get('rejected_at_edge', pd.Series(dtype=float)).fillna(0) > 0).sum())
    print(f'  attempts without an operating point by reason: {reasons}; with a rejection at the edge (D-138): {edge}')
    print(f'  wells by outcome: {dict((k, v) for k, v in d.outcome.value_counts().items() if v)}')
    hi = d[d.outcome == 'accepted'].n_redraws
    print(
        f"  accepted wells' redraws, P50 / P90 / P99 / max: {hi.quantile([0.5, 0.9, 0.99]).round(0).tolist()} / {hi.max()}"
    )


def reservoirs(t):
    """The core wells' T_r and pressure gradient against Nadeau et al.'s (2023) classes, by reservoirs of the world's
    largest fields: a comparison, not a target (SMP's Checks)."""
    w = t['wells']
    core = w[w.coverage_tail == 'core']
    print('\n== sol-2 core wells: reservoir classes against Nadeau et al. (2023), sampler / published')
    lo = 0.0
    for hi, name, share in NADEAU_T:
        m = (core.T_r >= lo) & (core.T_r < hi)
        print(f'  T_r {name}: {pct(m.mean())} / {pct(share)}')
        lo = hi
    depth = core.z + np.where(core.well_type == 'onshore', 0.0, core.water_depth.fillna(0.0))
    sg = core.p_i / (HYDROSTATIC * depth)
    lo = 0.0
    for hi, name, share in NADEAU_SG:
        m = (sg >= lo) & (sg < hi)
        print(f'  gradient over fresh water, {name}: {pct(m.mean())} / {pct(share)} (from sea level offshore)')
        lo = hi


def gas_separators(t):
    """D-134: the core offshore gas wells' separator and back-pressure."""
    w = t['wells']
    g = w[(w.coverage_tail == 'core') & (w.well_type != 'onshore') & ~w.fluid_type.isin(r10.OILS)]
    o = w[(w.coverage_tail == 'core') & (w.well_type != 'onshore') & w.fluid_type.isin(r10.OILS)]
    print('\n== sol-2 core offshore wells: p_sep and p_s_nom (bar), P10 / P50 / P90')
    print(f'  gas classes ({len(g)}): p_sep {r10.q(g.p_sep)}; p_s_nom {r10.q(g.p_s_nom)}')
    print(f'  oils ({len(o)}): p_sep {r10.q(o.p_sep)}; p_s_nom {r10.q(o.p_s_nom)}')


def decline_by_fifth(t):
    """D-140: the core wells' rate by fifth of the life over the first year's, medians by fluid and drive; the oils'
    oil rate as well as the oil equivalent."""
    w, s = t['wells'].set_index('ID'), t['samples']
    s = s.join(w[['T_end', 'drive', 'fluid_type', 'coverage_tail']], on='ID')
    s = s[s.coverage_tail == 'core'].assign(oe=r10.oil_equivalent, oil=24 * s.QOIL, tf=s.WEEKS / r10.YEAR / s.T_end)
    s['grp'] = np.where(s.fluid_type.isin(r10.OILS), 'oil', 'gas') + ', ' + s.drive.astype(str)
    rows = []
    for ID, g in s.groupby('ID'):
        first = g[g.WEEKS / r10.YEAR <= 1.0]
        fifth = [g[(g.tf > i / 5) & (g.tf <= (i + 1) / 5) if i else (g.tf <= 0.2)] for i in range(5)]
        rows.append(
            dict(
                grp=g.grp.iloc[0],
                **{f'{c}{i}': f[c].mean() / first[c].mean() for c in ('oe', 'oil') for i, f in enumerate(fifth)},
            )
        )
    r = pd.DataFrame(rows)
    groups = [*r.groupby('grp'), ('oil, both drives', r[r.grp.str.startswith('oil')])]
    print('\n== nsol-2 core wells: rate by fifth of the life over the first year, medians')
    for grp, g in groups:
        cols = ('oe', 'oil') if grp.startswith('oil') else ('oe',)
        for c in cols:
            print(f'  {grp} ({len(g)} wells), {c}: ' + ' '.join(f'{g[f"{c}{i}"].median():.2f}' for i in range(5)))


def main():
    sol_out, nsol_out = sys.argv[1:3]
    sol, nsol = r10.load(sol_out, 'sol'), r10.load(nsol_out, 'nsol')
    r10.outcomes(sol, 'sol')
    r10.outcomes(nsol, 'nsol')
    targets(sol, 'sol', sol_out)
    targets(nsol, 'nsol', nsol_out)
    r10.rates(sol)
    gas_separators(sol)
    reservoirs(sol)
    print('\n== flow regimes')
    r10.regimes(sol, 'sol')
    r10.regimes(nsol, 'nsol')
    r10.decline(nsol)
    decline_by_fifth(nsol)
    r10.stages(sol, nsol)
    r10.cost(sol, 'sol', Path(sol_out) / 'run' / 'attempts')
    r10.cost(nsol, 'nsol', Path(nsol_out) / 'run' / 'attempts')


if __name__ == '__main__':
    main()
