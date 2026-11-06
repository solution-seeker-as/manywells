"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 2 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The measurements of Step R6 of plans/sampler-redesign-plan.md on the tables that sampler_prototype.py writes
(plans/sampler-redesign/sampling-draft.md, What R6 must measure; r6-measurements.md quotes them):

  report DIR [--figures FIG] [--sol1 ZIP] [--processes P]
                       the outcomes, redraws, CasADi retries, the redraws' bias (criterion 4), coverage, realism
                       against the sources and Bjarne's goals, the cost, and with --sol1 the features next to the
                       published sol-1's (its zip, read by its exact path)
  resolve DIR N [--backend rust|casadi] [--processes P] [--seed S]
                       re-solve N samples drawn at random from the tables, built from the rows as datasets-draft.md
                       (Section 3) builds them: with rust, the re-solve check (D-92); with casadi, the backends'
                       comparison
  validity DIR N [--processes P]
                       re-solve N samples with the Rust core and measure each correlation's inputs along the operating
                       points against its published range, as validity_ranges.py does (specs/model/validity.md)

From the repository root, with the sandbox off for the pool (resolve, validity):

    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 3600 .venv/bin/python plans/evidence/sampler_prototype_report.py \\
        report OUT/sol-2 --figures FIG --sol1 SOL1_ZIP)
"""

import argparse
import json
import math
import multiprocessing
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import ks_2samp, spearmanr

pa.set_cpu_count(2)  # Arrow's thread pools reserve address space that the processes' memory cap does not allow
pa.set_io_thread_count(2)

OILS = ('black_oil', 'volatile_oil')
OP = ('operating_point', 'recovered')
CLASSES = ('black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas')
REGIMES = ('bubbly', 'dispersed-bubbly', 'slug', 'churn', 'annular')
# EIA (2025), Table B32: Gulf of Mexico oil wells by annual-average rate, 2024, in Sm3/d of oil equivalent
# (sources-wells-production.md, Section 5)
EIA_EDGES = (0, 1.6, 2.4, 16, 64, 254, 1018, 2035, np.inf)
EIA_SHARES = (9.8, 4.8, 37.8, 16.4, 13.9, 11.8, 4.2, 1.3)
GOR_OIL_WELL = 1068.0  # Sm3/Sm3: EIA's oil wells, GOR at most 6000 cf/b
COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def pct(x):
    return f'{100 * x:.1f}%'


def q(s, ps=(5, 50, 95), fmt='{:.4g}'):
    s = pd.Series(s).dropna()
    return ' / '.join(fmt.format(v) for v in np.percentile(s, ps)) if len(s) else '-'


def table(df, title=None):
    if title:
        print(f'\n{title}')
    print(df.to_string())


def load(out):
    out = Path(out)
    t = {n: pd.read_parquet(out / f'{n}.parquet') for n in ('wells', 'samples', 'drawn_wells', 'attempts', 'redraws')}
    runs = sorted(out.glob('run_*.json'))
    t['run'] = [json.load(open(p)) for p in runs]
    t['kind'] = 'nsol' if 'WEEKS' in t['samples'].columns else 'sol'
    return t


# --- Outcomes and redraws ---------------------------------------------------------------------------------------------
def outcomes(t):
    d = t['drawn_wells']
    print(
        f'\n== Wells: {len(d)} drawn, {(d.outcome == "accepted").sum()} accepted ({pct((d.outcome == "accepted").mean())})'
    )
    print(d.outcome.value_counts().to_string())
    print('not producible, by reason: ' + d.not_producible_reason.value_counts().to_dict().__repr__())
    for col in ('fluid_class', 'well_type', 'geometry_class', 'coverage_tail', 'lift'):
        if col not in d:
            continue
        ct = pd.crosstab(d[col], d.outcome, margins=True)
        ct['kept'] = (ct.get('accepted', 0) / ct['All']).map(pct)
        table(ct, f'Outcome by {col} (wells)')


def slot_attempts(t):
    """The slot attempts of the wells that reached their slots, with the well's columns joined."""
    a = t['attempts']
    a = a[a.kind == 'slot'].copy()
    d = t['drawn_wells'].set_index('draw_index')
    cols = ['fluid_class', 'fluid_type', 'well_type', 'geometry_class', 'coverage_tail', 'lift', 'p_i', 'w_lg_max']
    cols += [c for c in ('p_s_nom', 'p_r_nom', 'gor_nom', 'wlr_nom') if c in d]
    a = a.join(d[[c for c in cols if c not in a]], on='draw_index')
    if 'p_s_nom' in d and 'p_s_nom' in a:  # sol-2's slot attempts carry no p_s_nom of their own
        a['p_s_nom'] = a['p_s_nom'].fillna(a.draw_index.map(d['p_s_nom']))
    a['w_lg_frac'] = np.where(a.lift == 'gas_lift', a.WGL / a.w_lg_max, np.nan)
    a['r_p'] = a.PRES / a.p_i
    a['no_op'] = a.outcome == 'no_operating_point'
    a['solver'] = a.outcome == 'solver_failure'
    return a


def redraws(t):
    a = slot_attempts(t)
    acc = a[a.well_outcome == 'accepted']
    n_s = (acc.outcome.isin(OP)).sum()
    print(
        f'\n== Attempts of accepted wells: {len(acc)} for {n_s} samples; redraws per sample {(len(acc) - n_s) / n_s:.4f}'
    )
    print(acc.outcome.value_counts().to_string())
    print('reasons: ' + repr(acc.reason.value_counts().to_dict()))
    print(f'Rust reasons of all attempts (all wells): {a.rust_reason.value_counts().to_dict()}')
    rows = []
    for col in ('fluid_class', 'well_type', 'geometry_class', 'coverage_tail', 'lift'):
        for key, g in acc.groupby(col):
            ns = g.outcome.isin(OP).sum()
            rows.append(
                dict(
                    by=col,
                    value=key,
                    wells=g.draw_index.nunique(),
                    attempts=len(g),
                    redraws_per_sample=(len(g) - ns) / ns,
                    no_op=pct(g.no_op.mean()),
                    solver_failure=pct(g.solver.mean()),
                    recovered=pct((g.outcome == 'recovered').mean()),
                )
            )
    table(pd.DataFrame(rows).set_index(['by', 'value']), 'Redraws of accepted wells by group (shares of attempts)')
    per_well = acc.groupby('draw_index').apply(lambda g: (~g.outcome.isin(OP)).sum(), include_groups=False)
    per_slot = acc.groupby(['draw_index', 'slot']).size()
    print(
        f'Redraws per well: P50 / P90 / P99 / max {q(per_well, (50, 90, 99, 100))}; wells with none {pct((per_well == 0).mean())}'
    )
    print(f'Attempts per slot: P99 / max {q(per_slot, (99, 100))}')
    print(f"Share of a well's draws that flow: P1 / P5 / P50 {q(200 / (200 + per_well), (1, 5, 50))}")
    # By decile of the attempt's inputs (all slot attempts, also of wells dropped at a limit)
    bins = {
        'u': (a.CHK, np.r_[0.05, np.linspace(0.145, 0.905, 9), 0.9999, 1.0]),
        'w_lg/w_lg,max': (a.w_lg_frac, np.r_[-1e-9, 1e-9, np.linspace(0.1, 1.0, 10)]),
        'p_r/p_i (deciles)': (a.r_p, np.unique(np.nanquantile(a.r_p, np.linspace(0, 1, 11)))),
    }
    out = {}
    for name, (x, edges) in bins.items():
        cut = pd.cut(x, edges, include_lowest=True)
        g = a.groupby(cut, observed=True)
        df = pd.DataFrame(dict(attempts=g.size(), no_op=g.no_op.mean(), solver_failure=g.solver.mean()))
        out[name] = df
        table(
            df.assign(no_op=df.no_op.map(pct), solver_failure=df.solver_failure.map(pct)),
            f'No operating point by {name}',
        )
    return out


def retries(t):
    a = t['attempts']
    r = a[a.retried.fillna(False).astype(bool)]
    rust = a[a.t_rust.notna()]
    print(f'\n== CasADi retries (D-54): {len(r)} of {len(a)} attempts ({pct(len(r) / len(a))})')
    if len(r):
        g = r.groupby('rust_reason')
        df = pd.DataFrame(
            dict(
                retries=g.size(),
                recovered=g.apply(lambda x: (x.outcome == 'recovered').sum(), include_groups=False),
                casadi_no_root=g.apply(lambda x: (x.casadi_reason == 'no_root').sum(), include_groups=False),
                median_s=g.t_casadi.median(),
                total_s=g.t_casadi.sum(),
            )
        )
        table(df, "By the Rust core's reason")
        print(f'by kind: {r.groupby("kind").outcome.value_counts().to_dict()}')
    t_c, t_all = r.t_casadi.sum(), a.t_total.sum()
    print(
        f'Retry time {t_c:.0f} s of {t_all:.0f} s in attempts ({pct(t_c / t_all)}); Rust solve median '
        f'{rust.t_rust.median():.3f} s, retry median {r.t_casadi.median() if len(r) else math.nan:.2f} s'
    )
    rec = (r.outcome == 'recovered').sum()
    slot = a[a.kind == 'slot']
    sf = (slot.rust_outcome == 'solver_failure').sum()
    print(
        f'Gain: {rec} attempts recovered of {len(r)} retried; slot attempts that are Rust failures {pct(sf / len(slot))}, '
        f'left after the retry {pct((slot.outcome == "solver_failure").mean())}; '
        f'{t_c / max(rec, 1):.0f} s of retry per recovered attempt'
    )
    return r


def ks(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 5 or len(y) < 5:
        return math.nan
    return float(ks_2samp(x, y).statistic)


def bias(t):
    """Criterion 4: the inputs over all draws against over the accepted samples, split into the shift of the
    no-operating-point redraws and of the solver-failure redraws (the largest CDF gap, Kolmogorov-Smirnov D)."""
    a = slot_attempts(t)
    a = a[a.well_outcome == 'accepted']
    nominal = {'PDC': 'p_s_nom', 'PRES': 'p_r_nom', 'GOR': 'gor_nom', 'WLR': 'wlr_nom'}
    if t['kind'] == 'sol':
        inputs = {
            'CHK': a.CHK,
            'WGL/w_lg,max': a.w_lg_frac,
            **{f'{c}/{n}': a[c] / a[n] for c, n in nominal.items()},
        }
    else:
        inputs = {'CHK': a.CHK, 'WGL/w_lg,max': a.w_lg_frac, 'PDC/p_s_nom': a.PDC / a.p_s_nom, 'PRES/p_i': a.r_p}
    acc, no_op, sf = a.outcome.isin(OP), a.no_op, a.solver
    rows = []
    for name, x in inputs.items():
        x = x.replace([np.inf, -np.inf], np.nan)
        rows.append(
            dict(
                input=name,
                draws=x.notna().sum(),
                D_all_vs_accepted=ks(x, x[acc]),
                D_no_op_only=ks(x, x[~no_op]),
                D_solver_only=ks(x, x[~sf]),
                mean_all=x.mean(),
                mean_accepted=x[acc].mean(),
            )
        )
    df = pd.DataFrame(rows).set_index('input')
    table(df.round(4), '== Criterion 4: all draws against accepted samples (KS D; accepted wells)')
    by = []
    for cls, g in a.groupby('fluid_class'):
        acc_g = g.outcome.isin(OP)
        by.append(dict(fluid_class=cls, D_CHK=ks(g.CHK, g.CHK[acc_g]), D_WGL=ks(g.w_lg_frac, g.w_lg_frac[acc_g])))
    table(pd.DataFrame(by).set_index('fluid_class').round(4), 'By fluid class')
    # The wells: all drawn wells against the accepted ones
    d = t['drawn_wells']
    d = d[d.outcome != 'crash']
    accw = d.outcome == 'accepted'
    cols = ['z', 'L', 'p_i', 'T_r', 'permeability', 'net_pay', 'skin', 'design_rate', 'D', 'R_si', 'api', 'K_c']
    cols += [c for c in ('p_r_nom', 'gor_nom', 'wlr_nom', 'p_s_nom', 'w_lg_max') if c in d]
    rows = []
    for c in cols:
        x = d[c].astype(float)
        row = dict(input=c, D_drawn_vs_accepted=ks(x, x[accw]))
        for reason in d.outcome.unique():
            if reason != 'accepted':
                row[f'D_without_{reason}'] = ks(x, x[d.outcome != reason])
        rows.append(row)
    table(
        pd.DataFrame(rows).set_index('input').round(4),
        'Wells: drawn against accepted (KS D), and the shift of each drop',
    )
    cats = []
    for c in ('fluid_class', 'well_type', 'geometry_class', 'coverage_tail', 'pressure_class', 'inflow'):
        if c in d:
            s_all, s_acc = d[c].value_counts(normalize=True), d.loc[accw, c].value_counts(normalize=True)
            cats.append(dict(label=c, largest_share_change=float((s_acc - s_all).abs().max())))
    table(pd.DataFrame(cats).set_index('label').round(4), 'Labels: largest change of a share, drawn to accepted wells')
    return inputs, a


# --- Coverage and realism ---------------------------------------------------------------------------------------------
def coverage(t):
    s, w = t['samples'], t['wells']
    s = s.join(w.set_index('ID')[['fluid_class', 'fluid_type', 'coverage_tail', 'well_type', 'lift']], on='ID')
    print(f'\n== Coverage: {len(s)} samples of {s.ID.nunique()} wells')
    for pos in ('FRBH', 'FRWH'):
        ct = pd.crosstab(s.fluid_type, s[pos], normalize='index').reindex(columns=REGIMES, fill_value=0)
        ct.loc['all'] = s[pos].value_counts(normalize=True).reindex(REGIMES, fill_value=0)
        ct.loc['core'] = s[s.coverage_tail == 'core'][pos].value_counts(normalize=True).reindex(REGIMES, fill_value=0)
        ct.loc['tail'] = s[s.coverage_tail == 'tail'][pos].value_counts(normalize=True).reindex(REGIMES, fill_value=0)
        table((100 * ct).round(1), f'Flow regime at the {"bottom" if pos == "FRBH" else "top"} (%)')
    ch = s.groupby('ID').CHOKED.mean()
    print(
        f'\nChoked samples {pct(s.CHOKED.mean())}; per well P10 / P50 / P90 {q(ch, (10, 50, 90))}; wells never choked '
        f'{pct((ch == 0).mean())}, always choked {pct((ch == 1).mean())}, choked in more than 80% {pct((ch > 0.8).mean())}'
    )
    for tail in ('core', 'tail'):
        print(f'  {tail}: choked {pct(s[s.coverage_tail == tail].CHOKED.mean())}')
    print(f'Gas lift: {pct((w.lift == "gas_lift").mean())} of wells; samples with lift gas {pct((s.WGL > 0).mean())}')
    print(
        f'  lifted wells by class: {w.groupby("fluid_class").lift.apply(lambda x: pct((x == "gas_lift").mean())).to_dict()}'
    )
    rates = {
        'QOIL (Sm3/d)': 24 * s.QOIL,
        'QGAS (Sm3/d)': 24 * s.QGAS,
        'QLIQ (Sm3/d)': 24 * s.QLIQ,
        'QWAT (Sm3/d)': 24 * s.QWAT,
        'QGL (Sm3/d)': 24 * s.QGL,
        'WTOT (kg/s)': s.WTOT,
        'PBH (bar)': s.PBH,
        'PWH (bar)': s.PWH,
        'PDC (bar)': s.PDC,
        'PWH - PDC (bar)': s.PWH - s.PDC,
        'TBH (K)': s.TBH,
        'TWH (K)': s.TWH,
        'CHK': s.CHK,
    }
    rows = []
    for name, x in rates.items():
        r = dict(quantity=name)
        for tail in ('core', 'tail'):
            xs = x[s.coverage_tail == tail]
            r |= {
                f'{tail} {k}': v
                for k, v in zip(('min', 'P5', 'P50', 'P95', 'max'), np.nanpercentile(xs, (0, 5, 50, 95, 100)))
            }
        rows.append(r)
    table(pd.DataFrame(rows).set_index('quantity').map(lambda v: f'{v:.4g}'), 'Ranges, core and tail')
    by = s.groupby('fluid_class')
    table(
        pd.DataFrame(
            {
                'QOIL P50 (Sm3/d)': 24 * by.QOIL.median(),
                'QGAS P50 (Sm3/d)': 24 * by.QGAS.median(),
                'QLIQ P50': 24 * by.QLIQ.median(),
                'PBH P50': by.PBH.median(),
                'PWH P50': by.PWH.median(),
                'TWH P50': by.TWH.median(),
            }
        ).round(1),
        'Medians by class',
    )
    cv = s.groupby('ID').QTOT.agg(lambda x: x.std() / x.mean())
    print(
        f'Wells whose QTOT varies by under 5% (coefficient of variation): {pct((cv < 0.05).mean())} '
        f'({(cv < 0.05).sum()} wells); CV P5 / P50 / P95 {q(cv)}'
    )
    low = w.set_index('ID').loc[cv[cv < 0.05].index]
    if len(low):
        print(
            f'  their classes {low.fluid_class.value_counts().to_dict()}, choked share '
            f'{q(ch.loc[low.index], (5, 50, 95))}, K_c/A {q(low.K_c / (math.pi * low.D**2 / 4), (5, 50, 95))}'
        )
    return s


def realism(t):
    s, w, d = t['samples'], t['wells'], t['drawn_wells']
    print('\n== Realism')
    s = s.assign(q_oe=24 * (s.QOIL + s.QGAS / 1000))
    med = s.groupby('ID').q_oe.median()
    w = w.set_index('ID').assign(q_oe_med=med)
    gor_col = 'gor_nom' if 'gor_nom' in w else 'R_si'
    off_oil = w[w.fluid_type.isin(OILS) & w.well_type.isin(['platform', 'subsea']) & (w[gor_col] <= GOR_OIL_WELL)]
    rows = []
    for name, sub in (('EIA 2024, GoM oil wells', None), ('core, median sample', 'core'), ('all, median sample', 'all'),
                      ('core, best rate', 'best')):  # fmt: skip
        if sub is None:
            rows.append(dict(source=name, n=1869, **{lab: v for lab, v in zip(bracket_labels(), EIA_SHARES)}))
            continue
        x = off_oil[off_oil.coverage_tail == 'core'] if sub in ('core', 'best') else off_oil
        vals = x.best_rate if sub == 'best' else x.q_oe_med
        h = np.histogram(vals, EIA_EDGES)[0] / len(vals) * 100
        rows.append(dict(source=name, n=len(vals), **{lab: v for lab, v in zip(bracket_labels(), h)}))
    eia = pd.DataFrame(rows).set_index('source')
    table(eia.round(1), 'Offshore oil wells (GOR <= 1068 Sm3/Sm3) by rate bracket, Sm3/d of oil equivalent (%)')
    x = off_oil[off_oil.coverage_tail == 'core'].q_oe_med
    print(
        f'  core median-sample rate P10 / P50 / P90 {q(x, (10, 50, 90))} Sm3/d; mean {x.mean():.0f} '
        f'(EIA mean 167 Sm3/d); above 509 Sm3/d {pct((x > 509).mean())} (EIA 12%)'
    )
    gas = w[~w.fluid_type.isin(OILS) & (w.coverage_tail == 'core')]
    gq = 24 * s.groupby('ID').QGAS.median().reindex(gas.index)
    print(
        f'  core gas and condensate wells, median-sample gas rate P10 / P50 / P90 {q(gq, (10, 50, 90))} Sm3/d '
        f'(EIA GoM gas wells: mean 32,700 Sm3/d, 2.3% above 270,000 Sm3/d); above 270,000: {pct((gq > 2.7e5).mean())}'
    )
    # gradients and temperatures by class, accepted core wells; BOEM's by wells: [own analysis of the BOEM sands atlas]
    core = w[w.coverage_tail == 'core']
    grad = (core.p_i - 1.01325) / (core.z + core.water_depth)
    rows = []
    for cls in CLASSES:
        m = core.fluid_class == cls
        rows.append(dict(fluid_class=cls, wells=m.sum(), grad_P50=grad[m].median(), T_r_P50=core.T_r[m].median()))
    table(
        pd.DataFrame(rows).set_index('fluid_class').round(3),
        'Initial gradient and T_r, accepted core wells (compare with BOEM: [own analysis of the BOEM sands atlas])',
    )
    dcore = d[(d.coverage_tail == 'core') & d.p_i.notna()]
    g_all = (dcore.p_i - 1.01325) / (dcore.z + dcore.water_depth)
    print(
        f'  above 0.6 psi/ft: drawn {pct((g_all > 0.1357).mean())}, accepted {pct((grad > 0.1357).mean())} (BOEM: [own analysis of the BOEM sands atlas])'
    )
    # Bjarne's goals
    print("\nBjarne's goals (accepted wells and their samples)")
    if 'p_r_nom' in w:
        below = w.p_r_nom < w.p_h
        print(
            f'  1. p_r below hydrostatic: {pct(below.mean())} of wells (core {pct(below[w.coverage_tail == "core"].mean())});'
            f' below 0.9 of it {pct((w.p_r_nom < 0.9 * w.p_h).mean())}; drawn {pct((d.p_r_nom < d.p_h).mean())}'
        )
    else:
        below = s.PRES < s.ID.map(w.p_h)
        print(
            f'  1. samples with p_r below hydrostatic: {pct(below.mean())}; wells that reach it {pct(below.groupby(s.ID).any().mean())}'
        )
    print(
        f'  2. samples with PDC below 10 bar: {pct((s.PDC < 10).mean())}; wells with any {pct((s.groupby("ID").PDC.min() < 10).mean())}'
    )
    gw = (w.p_i - 1.01325) / (w.z + w.water_depth)
    over = gw > 0.1357
    top = w.q_oe_med > w.q_oe_med.quantile(0.75)
    print(
        f'  3. overpressured (above 0.6 psi/ft): {pct(over.mean())} of wells; of these in the top quartile of rates '
        f'{pct((over & top).sum() / max(over.sum(), 1))}; overpressured and above 1000 Sm3/d {pct((over & (w.q_oe_med > 1000)).mean())}'
    )
    print(f'  4. samples fully open: {pct((s.CHK == 1).mean())} (drawn 5%); below 0.1: {pct((s.CHK < 0.1).mean())}')
    lifted = w[w.lift == 'gas_lift']
    ql = 24 * s.groupby('ID').QLIQ.median().reindex(lifted.index)
    if len(lifted) > 5:
        rho = spearmanr(lifted.w_lg_max, ql).statistic
        print(
            f'  5. lifted wells {pct(len(lifted) / len(w))}; Spearman(w_lg_max, median QLIQ) {rho:+.2f}; '
            f'w_lg_max P5 / P50 / P95 {q(lifted.w_lg_max)} kg/s'
        )
        oils = w[w.fluid_type.isin(OILS)]
        gq_ = pd.qcut(oils[gor_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        print(
            '     lifted share of oil wells by GOR quartile: '
            + ', '.join(f'{k} {pct((oils.lift[gq_ == k] == "gas_lift").mean())}' for k in ['Q1', 'Q2', 'Q3', 'Q4'])
        )
        sl = s[s.ID.isin(lifted.index)]
        glr = (24 * sl.QGL) / (24 * sl.QLIQ).replace(0, np.nan)
        print(f'     injected GLR of lifted samples with lift gas, P5 / P50 / P95 {q(glr[sl.WGL > 0])} Sm3/Sm3')
    return eia


def bracket_labels():
    e = EIA_EDGES
    return [f'{e[i]:g}-{e[i + 1]:g}' if np.isfinite(e[i + 1]) else f'>{e[i]:g}' for i in range(len(e) - 1)]


# --- Cost -----------------------------------------------------------------------------------------------------------
def cost(t, processes=12):
    a, d = t['attempts'], t['drawn_wells']
    rust = a[a.t_rust.notna()]
    n_acc = (d.outcome == 'accepted').sum()
    print('\n== Cost')
    print(
        f'Rust solve (t_rust, one attempt): median {rust.t_rust.median():.3f} s, mean {rust.t_rust.mean():.3f} s, '
        f'P95 {rust.t_rust.quantile(0.95):.3f} s; in the core: median {rust.t_core.median():.3f} s'
    )
    by = rust.join(d.set_index('draw_index')[['fluid_class']], on='draw_index').groupby('fluid_class').t_rust
    print('  by class, mean: ' + ', '.join(f'{k} {v:.3f}' for k, v in by.mean().items()))
    t_att = a.t_total.sum()
    t_core = a.t_core.sum()
    t_cas = a.t_casadi.sum()
    cpu = d.cpu_seconds.sum()
    outside = t_att - t_core - t_cas
    print(
        f'Attempt time {t_att:.0f} s: in the core {t_core:.0f} s, CasADi retries {t_cas:.0f} s, outside both {outside:.0f} s;'
        f" worker CPU {cpu:.0f} s, of which outside the attempts {cpu - t_att:.0f} s (drawing, life scans' setup)"
    )
    share_out = (cpu - t_core - t_cas) / cpu
    print(
        f'Time outside the core per solve: {1e3 * outside / len(rust):.2f} ms (median attempt '
        f'{1e3 * (rust.t_total - rust.t_core - rust.t_casadi).median():.2f} ms); share of all CPU outside the core and '
        f'the retries {pct(share_out)}'
    )
    per_acc = cpu / n_acc
    slot = a[a.kind == 'slot']
    print(
        f"CPU per accepted well {per_acc:.1f} s (all drawn wells' CPU over the accepted); solves per accepted well "
        f'{len(a) / n_acc:.1f} ({len(slot) / n_acc:.1f} slot attempts)'
    )
    cpu_h = 10_000 * per_acc / 3600
    for run in t['run']:
        if run.get('wells') is None:
            wall, p = run['wall_seconds'], run['processes']
            eff = d.seconds.sum() / (p * wall)
            print(
                f'Run: wall {wall:.0f} s on {p} processes; pool efficiency (worker busy time / (processes x wall)) {eff:.2f}'
            )
    print(
        f'Projection, 10,000 wells: {cpu_h:.0f} CPU-hours, {cpu_h / processes:.1f} h wall on {processes} processes '
        f'at full efficiency'
    )
    print(
        f'Batch API (backlog 4.3): at most {pct(share_out)} of the CPU is outside the core and the retries, so its '
        f'wall-time gain is at most that, {"above" if share_out > 0.2 else "below"} the 20% the draft set'
    )


# --- nsol-2 ---------------------------------------------------------------------------------------------------------
def nsol(t):
    d, w, s, a = t['drawn_wells'], t['wells'], t['samples'], slot_attempts(t)
    print('\n== nsol-2')
    print(f'Lift start rule (accepted wells): {w.lift_start_rule.fillna("natural").value_counts().to_dict()}')
    print(f'Lift start rule (drawn wells): {d.lift_start_rule.fillna("natural").value_counts().to_dict()}')
    later = w[(w.lift_start > 0)]
    print(f'  later lift starts at t/T_life {q(later.lift_start / later.T_life)}')
    print(
        f'T_end / T_life: {q(w.T_end / w.T_life, (5, 25, 50))}; lives cut by the floor {pct((w.T_end < w.T_life - 1e-9).mean())}'
    )
    print(
        f'short_life {pct((d.outcome == "short_life").mean())}, dead_date {pct((d.outcome == "dead_date").mean())} of drawn wells'
    )
    acc = a[a.well_outcome == 'accepted']
    ac = acc.assign(kind_=np.where(acc.TEST.notna(), 'test', 'operating'))
    g = ac.groupby('kind_')
    print('Redraws per sample by kind: ' + ', '.join(
        f'{k} {(len(x) - x.outcome.isin(OP).sum()) / x.outcome.isin(OP).sum():.4f}' for k, x in g))  # fmt: skip
    first = acc[acc.attempt == 0].set_index(['draw_index', 'slot'])
    red = acc[acc.attempt > 0].join(first[['CHK']].rename(columns={'CHK': 'u0'}), on=['draw_index', 'slot'])
    if len(red):
        print(f'Redrawn controls against the planned: |u - u_plan| P50 / P90 {q((red.CHK - red.u0).abs(), (50, 90))}')
    tf = s.WEEKS / 52.1775 / s.ID.map(w.set_index('ID').T_end)
    st = s.join(w.set_index('ID')[['p_i', 'drive', 'fluid_type', 'coverage_tail']], on='ID')
    st = st[st.coverage_tail == 'core']
    st['r_p'] = st.PRES / st.p_i
    table(st.groupby(['STAGE', 'drive']).r_p.median().unstack().round(3), 'p_r/p_i by per-sample stage (core), median')
    q_oe = 24 * (s.QOIL + s.QGAS / 1000)
    bins = pd.cut(tf, np.linspace(0, 1, 6), include_lowest=True)
    rel = q_oe / q_oe.groupby(s.ID).transform('first').replace(0, np.nan)
    print(
        'Oil equivalent over the life, relative to the first sample, median by fifth of the life: '
        + ', '.join(f'{v:.2f}' for v in rel.groupby(bins, observed=True).median())
    )


# --- Figures --------------------------------------------------------------------------------------------------------
def style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#c3c2b7')
    ax.grid(color='#e6e5e0', linewidth=0.6)
    ax.set_axisbelow(True)


def figures(t, fig_dir, sol1=None, deciles=None, eia=None):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = 'sol2' if t['kind'] == 'sol' else 'nsol2'
    made = []
    # 1. Criterion 4: the inputs over all draws and over the accepted samples
    a = slot_attempts(t)
    a = a[a.well_outcome == 'accepted']
    acc = a.outcome.isin(OP)
    xs = {'CHK': a.CHK, 'WGL / w_lg,max (lifted wells)': a.w_lg_frac, 'PDC / p_s,nom': a.PDC / a.p_s_nom}
    xs['PRES / p_r,nom' if t['kind'] == 'sol' else 'PRES / p_i'] = a.PRES / (a.p_r_nom if t['kind'] == 'sol' else a.p_i)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE)
    for ax, (name, x) in zip(axs, xs.items()):
        for sel, lab, c in ((slice(None), 'all draws', COLOURS[0]), (acc, 'accepted samples', COLOURS[1])):
            v = np.sort(x[sel].dropna())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=c, linewidth=2, label=lab)
        style(ax, f'{name}: KS D {ks(x, x[acc]):.3f}', name, 'CDF' if ax is axs[0] else None)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    made.append(save(fig, fig_dir / f'{tag}_bias_cdf.png'))
    # 2. No operating point by input bin
    if deciles:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), facecolor=SURFACE)
        for ax, (name, df) in zip(axs, deciles.items()):
            x = np.arange(len(df))
            ax.bar(x - 0.2, 100 * df.no_op, 0.38, color=COLOURS[0], label='no operating point')
            ax.bar(x + 0.2, 100 * df.solver_failure, 0.38, color=COLOURS[1], label='solver failure')
            ax.set_xticks(x, [str(i) for i in df.index], rotation=60, fontsize=6)
            style(
                ax, f'Attempts without an operating point, by {name}', name, '% of attempts' if ax is axs[0] else None
            )
        axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        made.append(save(fig, fig_dir / f'{tag}_redraws_by_bin.png'))
    s = t['samples']
    # 3. Regimes at the bottom and the top
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), facecolor=SURFACE)
    for ax, pos in zip(axs, ('FRBH', 'FRWH')):
        sh = s[pos].value_counts(normalize=True).reindex(REGIMES, fill_value=0) * 100
        ax.bar(range(5), sh.values, color=COLOURS[0], width=0.6)
        for i, v in enumerate(sh.values):
            ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=8, color=INK)
        ax.set_xticks(range(5), REGIMES, fontsize=8)
        style(ax, f'Flow regime at the {"bottom" if pos == "FRBH" else "top"} ({tag})', '', '% of samples')
    fig.tight_layout()
    made.append(save(fig, fig_dir / f'{tag}_regimes.png'))
    # 4. Rates, pressures and temperatures next to sol-1
    qs = {
        'log10 QOIL (Sm3/d)': lambda d: np.log10(24 * d.QOIL.where(d.QOIL > 0)),
        'log10 QGAS (Sm3/d)': lambda d: np.log10(24 * d.QGAS.where(d.QGAS > 0)),
        'log10 QWAT (Sm3/d), samples with water': lambda d: np.log10(24 * d.QWAT.where(d.QWAT > 0)),
        'PBH (bar)': lambda d: d.PBH,
        'PWH (bar)': lambda d: d.PWH,
        'PDC (bar)': lambda d: d.PDC,
        'TWH (K)': lambda d: d.TWH,
        'WTOT (kg/s)': lambda d: np.log10(d.WTOT.where(d.WTOT > 0)),
    }
    fig, axs = plt.subplots(2, 4, figsize=(14, 6), facecolor=SURFACE)
    for ax, (name, f) in zip(axs.ravel(), qs.items()):
        v = f(s).replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5)
        if sol1 is not None:
            v1 = f(sol1).replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = min(lo, np.nanpercentile(v1, 0.5)), max(hi, np.nanpercentile(v1, 99.5))
            ax.hist(
                v1, bins=60, range=(lo, hi), density=True, histtype='step', linewidth=2, color=COLOURS[1], label='sol-1'
            )
        ax.hist(
            v,
            bins=60,
            range=(lo, hi),
            density=True,
            histtype='step',
            linewidth=2,
            color=COLOURS[0],
            label=f'{tag} prototype',
        )
        style(ax, name.replace('WTOT (kg/s)', 'log10 WTOT (kg/s)'), '')
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    made.append(save(fig, fig_dir / f'{tag}_features_vs_sol1.png'))
    # 5. EIA brackets
    if eia is not None:
        fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
        x = np.arange(len(EIA_SHARES))
        for i, (name, c) in enumerate(zip(eia.index[:3], COLOURS)):
            ax.bar(
                x + (i - 1) * 0.27,
                eia.loc[name, bracket_labels()].values,
                0.25,
                color=c,
                label=f'{name} (n={eia.loc[name, "n"]})',
            )
        ax.set_xticks(x, bracket_labels(), fontsize=8)
        style(ax, 'Offshore oil wells by rate bracket', 'Sm3/d of oil equivalent', '% of wells')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        made.append(save(fig, fig_dir / f'{tag}_eia_brackets.png'))
    # 6. nsol-2: a few wells' rates over their life
    if t['kind'] == 'nsol':
        w = t['wells'].set_index('ID')
        fig, axs = plt.subplots(2, 3, figsize=(13, 6), facecolor=SURFACE)
        for ax, i in zip(axs.ravel(), w.index[:6]):
            si = s[s.ID == i]
            ops = si[si.TEST.isna()]
            ax.plot(
                ops.WEEKS / 52.1775, 24 * (ops.QOIL + ops.QGAS / 1000), color=COLOURS[0], linewidth=2, label='operating'
            )
            te = si[si.TEST.notna()]
            ax.scatter(
                te.WEEKS / 52.1775, 24 * (te.QOIL + te.QGAS / 1000), s=14, color=COLOURS[1], label='tests', zorder=3
            )
            style(
                ax,
                f'well {i}: {w.loc[i, "fluid_class"]}, {w.loc[i, "drive"]}, {w.loc[i, "lift"]}',
                'years',
                'Sm3/d o.e.',
            )
        axs[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        made.append(save(fig, fig_dir / f'{tag}_histories.png'))
    print('\nFigures: ' + ', '.join(str(p) for p in made))


def save(fig, path):
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# --- sol-1 --------------------------------------------------------------------------------------------------------------
def read_sol1(path):
    cols = ['ID', 'CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH', 'WGL', 'WTOT', 'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH']  # fmt: skip
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(z.open('manywells-sol-1.csv'), usecols=cols)


def compare_sol1(t, sol1):
    s = t['samples']
    print(f'\n== Next to sol-1 ({len(sol1)} samples of {sol1.ID.nunique()} wells): P5 / P50 / P95, and KS D')
    rows = []
    for c, f in (('QOIL (Sm3/d)', lambda d: 24 * d.QOIL), ('QGAS (Sm3/d)', lambda d: 24 * d.QGAS),
                 ('QLIQ (Sm3/d)', lambda d: 24 * d.QLIQ), ('QWAT (Sm3/d)', lambda d: 24 * d.QWAT),
                 ('QGL (Sm3/d)', lambda d: 24 * d.QGL), ('WTOT (kg/s)', lambda d: d.WTOT),
                 ('PBH (bar)', lambda d: d.PBH), ('PWH (bar)', lambda d: d.PWH), ('PDC (bar)', lambda d: d.PDC),
                 ('TBH (K)', lambda d: d.TBH), ('TWH (K)', lambda d: d.TWH), ('CHK', lambda d: d.CHK),
                 ('FGAS', lambda d: d.FGAS), ('FWAT', lambda d: d.FWAT)):  # fmt: skip
        x, y = f(s), f(sol1)
        rows.append(dict(feature=c, prototype=q(x), sol_1=q(y), KS_D=ks(x, y)))
    table(pd.DataFrame(rows).set_index('feature'))
    print(
        f'CHOKED: prototype {pct(s.CHOKED.mean())}, sol-1 {pct(sol1.CHOKED.mean())}; lift gas in samples: '
        f'{pct((s.WGL > 0).mean())} against {pct((sol1.WGL > 0).mean())}'
    )
    for pos in ('FRBH', 'FRWH'):
        print(
            f'{pos}: prototype {(100 * s[pos].value_counts(normalize=True)).round(1).to_dict()}; '
            f'sol-1 {(100 * sol1[pos].value_counts(normalize=True)).round(1).to_dict()}'
        )


# --- Re-solving samples from the tables -----------------------------------------------------------------------------------
def table_from_rows(rows):
    """A table well's PVTTableSet from its `pvt` rows, through PVTTable.read_csv as datasets-draft.md has it."""
    import tempfile

    from manywells.pvt.table import PVTTable

    r0 = rows.iloc[0]
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as fh:
        fh.write(f'# rho_o: {float(r0.rho_o)!r}\n# rho_g: {float(r0.rho_g)!r}\n')
        rows[['gor', 'p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma']].to_csv(
            fh, index=False, float_format='%.17g'
        )
    try:
        return PVTTable.read_csv(fh.name)
    finally:
        Path(fh.name).unlink()


def case_from_rows(w, s, survey, pvt_rows=None):
    """A sample's well and boundary conditions from its rows, as datasets-draft.md (Section 3) builds them: the black
    oil, or a table well's table set from its `pvt` rows; Vogel, a productivity index or Forchheimer's gas inflow
    (R9). Tables of the interim mapping (before R9) have no table and gas wells on a productivity index (k_l)."""
    from manywells.choke import SimpsonChokeModel
    from manywells.configurations import V2, V2_SLIP, check
    from manywells.friction import RoughnessFriction
    from manywells.geometry import WellGeometry
    from manywells.inflow import Forchheimer, ProductivityIndex, Vogel
    from manywells.pvt.black_oil import BlackOil
    from manywells.pvt.fluid import FluidModel
    from manywells.simulator import BoundaryConditions, WellProperties
    from manywells.thermal import ThermalModel

    nsol = 'HTC' in s
    if w.get('pvt_model') == 'table':
        hc = table_from_rows(pvt_rows)
    else:
        hc = BlackOil(rho_o=w.rho_o, rho_g=w.rho_g, gas_law='dak', p_sep=w.p_sep, T_sep=w.T_sep)
    k_l, w_l_max = (s.KL, s.WLMAX) if nsol else (w.k_l, w.w_l_max)
    if w.inflow == 'forchheimer':
        a, b = (s.FORCHA, s.FORCHB) if nsol else (w.forchheimer_a, w.forchheimer_b)
        inflow = Forchheimer(a=a, b=b)
    else:
        inflow = Vogel(w_l_max=w_l_max) if w.inflow == 'vogel' else ProductivityIndex(k_l=k_l)
    lifted = bool(s.LIFT) if nsol else w.lift == 'gas_lift'
    wp = WellProperties(
        geometry=WellGeometry.from_survey(survey.md.values, survey.tvd.values, n_cells=int(w.n_cells), D=w.D),
        fluid=FluidModel(pvt=hc, rho_w=w.rho_w, gor=s.GOR, wlr=s.WLR, cp_g=w.cp_g, cp_o=w.cp_o, cp_w=w.cp_w),
        friction=RoughnessFriction(roughness=s.ROUGH if nsol else w.roughness, correlation='chen'),
        thermal=ThermalModel(h=s.HTC if nsol else w.h),
        slip=V2_SLIP,
        inflow=inflow,
        choke=SimpsonChokeModel(K_c=w.K_c, chk_profile=w.chk_profile),
        injection_md=w.injection_md if lifted else None,
    )
    bc = BoundaryConditions(
        p_r=s.PRES, p_s=s.PDC, T_r=w.T_r, T_s=w.T_s, T_lg=w.T_lg if lifted else None, u=s.CHK, w_lg=s.WGL
    )
    check(wp, V2)
    return wp, bc


def _resolve_task(task):
    from sampler_prototype import features
    from validity_ranges import point_rows

    from manywells.simulator import SSDFSimulator

    w, s, survey, backend, with_points, pvt_rows = task
    t0 = time.perf_counter()
    out = dict(ID=int(s.ID), slot=int(s.slot), backend=backend)
    try:
        wp, bc = case_from_rows(w, s, survey, pvt_rows)
        rs = SSDFSimulator(wp, backend=backend).root_set(bc)
    except Exception as e:  # noqa: BLE001
        out.update(found=False, error=f'{type(e).__name__}: {e}'[:200], seconds=time.perf_counter() - t0)
        return out, []
    op = rs.operating_point
    out.update(
        seconds=time.perf_counter() - t0, n_roots=len(rs.roots), n_stable=sum(r.label == 'stable' for r in rs.roots)
    )
    out['p0_roots'] = [float(r.p_0) for r in rs.roots]
    out['labels'] = [r.label for r in rs.roots]
    if op is None:
        out['found'] = False
        return out, []
    row = features(op, bc, wp.fluid)
    out.update(found=True, **{f'new_{k}': v for k, v in row.items()})
    points = []
    if with_points:
        fl, geo = wp.fluid, wp.geometry
        q_l = op.w_res / fl.rho_l
        q_lg = bc.w_lg / fl.rho_g
        from manywells.units import CF_RS

        out.update(
            api=fl.api, sg_gas=fl.sg_gas, gor=fl.gor, wlr=fl.wlr, gor_scf=fl.gor / CF_RS, glr_scf=fl.glr / CF_RS,
            p_r=bc.p_r, T_r=bc.T_r, md=geo.md[0], eps_D=wp.friction.roughness / geo.D,
            q_l_stbd=q_l * 86400 * 6.28981, bhp_psig=(op.p_0 - 1.01325) * 14.503774,
            glr_total_scf=(fl.glr + q_lg / q_l) / CF_RS if q_l > 0 else np.nan,
        )  # fmt: skip
        points = [dict(ID=int(s.ID), slot=int(s.slot), **r) for r in point_rows(wp, op)]
    return out, points


def resolve(out_dir, n, backend, processes, seed, with_points=False):
    out_dir = Path(out_dir)
    w = pd.read_parquet(out_dir / 'wells.parquet').set_index('ID')
    s = pd.read_parquet(out_dir / 'samples.parquet')
    geo = pd.read_parquet(out_dir / 'geometry.parquet')
    pvt = pd.read_parquet(out_dir / 'pvt.parquet')
    pick = s.sample(n=min(n, len(s)), random_state=seed)
    tasks = [(w.loc[r.ID], r, geo[geo.ID == r.ID], backend, with_points) for r in pick.itertuples(index=False)]
    tasks = [
        (wr, pd.Series(r._asdict()), g, b, p, pvt[pvt.pvt_table_id == wr.get('pvt_table_id')])
        for wr, r, g, b, p in tasks
    ]
    t0 = time.perf_counter()
    with multiprocessing.get_context('forkserver').Pool(processes) as pool:
        res = pool.map(_resolve_task, tasks, chunksize=1)
    print(f'{len(res)} samples re-solved with {backend} in {time.perf_counter() - t0:.0f} s on {processes} processes')
    cases = pd.DataFrame([r for r, _ in res])
    points = pd.DataFrame([p for _, ps in res for p in ps])
    a = pd.read_parquet(out_dir / 'attempts.parquet')
    rec = a[(a.kind == 'slot') & (a.outcome == 'recovered')][['ID', 'slot']].assign(recovered=True)
    cases = cases.merge(pick, on=['ID', 'slot']).merge(rec, on=['ID', 'slot'], how='left')
    cases['recovered'] = cases.recovered.fillna(False)
    return cases, points


def compare_resolved(cases, backend):
    found = cases[cases.found]
    print(
        f'Operating point found in {len(found)} of {len(cases)} ({cases.recovered.sum()} of them recovered by the '
        f'CasADi retry in the run)'
    )
    if not len(found):
        return
    d = pd.DataFrame(
        {
            'PBH': (found.new_PBH - found.PBH).abs() / (found.PRES - found.PDC),
            'PWH': (found.new_PWH - found.PWH).abs() / (found.PRES - found.PDC),
            'TWH': (found.new_TWH - found.TWH).abs() / (found.TBH - found.TWH).abs().clip(lower=1),
            'WTOT': (found.new_WTOT - found.WTOT).abs() / found.WTOT,
            'QOIL': (found.new_QOIL - found.QOIL).abs() / found.QTOT,
        }
    )
    exact = (d.max(axis=1) == 0) & (found.new_CHOKED == found.CHOKED) & (found.new_FRWH == found.FRWH)
    print(f'Identical features: {exact.sum()} of {len(found)}; within 1e-4 (D-92): {(d.max(axis=1) <= 1e-4).sum()}')
    print('Largest relative difference by feature: ' + ', '.join(f'{k} {v:.2e}' for k, v in d.max().items()))
    diff = found[d.max(axis=1) > 1e-4]
    if len(diff):
        print(f'{len(diff)} samples differ beyond 1e-4:')
        cols = ['ID', 'slot', 'CHK', 'PBH', 'new_PBH', 'WTOT', 'new_WTOT', 'n_roots', 'p0_roots', 'labels', 'recovered']
        print(diff[cols].to_string())
    regimes = (found.new_FRBH == found.FRBH).mean(), (found.new_FRWH == found.FRWH).mean()
    print(
        f'Same regimes: bottom {pct(regimes[0])}, top {pct(regimes[1])}; same CHOKED {pct((found.new_CHOKED == found.CHOKED).mean())}'
    )
    miss = cases[~cases.found]
    if len(miss):
        cols = ['ID', 'slot', 'CHK', 'PRES', 'PDC', 'PBH', 'n_roots', 'p0_roots', 'labels', 'error', 'recovered']
        print(f'No operating point in {len(miss)}:\n' + miss.reindex(columns=cols).to_string())
    print(f'Time per sample: median {cases.seconds.median():.2f} s, mean {cases.seconds.mean():.2f} s ({backend})')


def validity_summary(cases, points):
    from validity_ranges import RANGES

    solved = cases[cases.found]
    rows = []
    for name, col, tab, lo, hi in RANGES:
        df = solved if tab == 'cases' else points
        if col not in df:
            continue
        x = df[col]
        out = pd.Series(False, index=df.index)
        if lo is not None:
            out |= x < lo
        if hi is not None:
            out |= x > hi
        share = out.mean() if tab == 'cases' else out.groupby([df.ID, df.slot]).any().mean()
        rows.append(dict(row=name, variable=col, low=lo, high=hi, P5=x.quantile(0.05), P50=x.median(),
                         P95=x.quantile(0.95), samples_out=round(100 * share, 1)))  # fmt: skip
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('samples_out: % of the samples with an operating point that have a grid point (or input) outside the range')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report')
    r.add_argument('dir')
    r.add_argument('--figures')
    r.add_argument('--sol1')
    for name in ('resolve', 'validity'):
        p = sub.add_parser(name)
        p.add_argument('dir')
        p.add_argument('n', type=int)
        p.add_argument('--processes', type=int, default=12)
        p.add_argument('--seed', type=int, default=6)
        if name == 'resolve':
            p.add_argument('--backend', choices=('rust', 'casadi'), default='rust')
    args = ap.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    if args.cmd == 'report':
        t = load(args.dir)
        outcomes(t)
        dec = redraws(t)
        retries(t)
        bias(t)
        coverage(t)
        eia = realism(t)
        if t['kind'] == 'nsol':
            nsol(t)
        cost(t)
        sol1 = read_sol1(args.sol1) if args.sol1 else None
        if sol1 is not None:
            compare_sol1(t, sol1)
        if args.figures:
            figures(t, args.figures, sol1, dec, eia)
    elif args.cmd == 'resolve':
        cases, _ = resolve(args.dir, args.n, args.backend, args.processes, args.seed)
        cases.to_parquet(Path(args.dir) / f'resolve_{args.backend}.parquet', index=False)
        compare_resolved(cases, args.backend)
    else:
        cases, points = resolve(args.dir, args.n, 'rust', args.processes, args.seed, with_points=True)
        validity_summary(cases, points)


if __name__ == '__main__':
    main()
