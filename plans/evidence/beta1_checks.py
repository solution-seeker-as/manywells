"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 14 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The checks and measurements of the v2 beta's full generation (plans/sampler-redesign/beta1-analysis.md), on a
dataset's generation folder: the dataset, its logs and its run (manywells.sampling.v2). Every number is ManyWells' own,
from synthetic wells.

    python plans/evidence/beta1_checks.py checks GEN          # the checks at generation: counts, rows, configuration, split
    python plans/evidence/beta1_checks.py report GEN          # drops by group, thresholds, shifts, redraws, datasheet numbers
    python plans/evidence/beta1_checks.py resolve GEN OUT     # a seeded sample of rows solved again: to the bit, and CasADi
    python plans/evidence/beta1_checks.py validity OUT        # the re-solved samples against the correlations' ranges
    python plans/evidence/beta1_checks.py rules GEN           # the four rule points left for after the beta
    python plans/evidence/beta1_checks.py limit GEN 3000 RULE # wells of a lift rule dropped at the redraw limit, again

GEN is a generation's folder, which holds the dataset (manywells-<kind>-2-beta1), its logs (the same name with -logs)
and its run (run/, with the attempt log); OUT is a prefix for the re-solve's files. From the project root, with the
generator's interpreter (Python, numpy and scipy as in metadata.json, so that the re-solve can be exact), capped:
(ulimit -v 8000000; OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 .venv/bin/python plans/evidence/beta1_checks.py ...).
"""

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from manywells.datasets.v2 import FEATURES, _sample_features, sample_case
from manywells.sampling.v2 import generate
from manywells.simulator import SSDFSimulator
from manywells.units import CF_BAR, CF_PSI, CF_RS

pa.set_cpu_count(2)  # pyarrow's pools would otherwise start a thread per core, which the address-space cap refuses
pa.set_io_thread_count(2)

LIMITS = (  # SMP-106's limits, and the generator's own failures
    'best_rate_solver_failure',
    'invalid_input',
    'slot_limit',
    'redraw_limit',
    'solver_failure_limit',
    'dead_date',
    'table_failed',
    'crash',
)
FILTERS = ('not_producible', 'short_life')  # SMP-101 to SMP-103: wells that would not be produced
CLASS_GOR = {'volatile_oil': (356.0, 588.0), 'gas_condensate': (588.0, 8900.0)}  # Sm3/Sm3, SMP-55
DRY_GAS_GOR = 1e6 / 2.8  # The GOR at SMP-55's dry-gas yield
MASS = ('WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT')
VOLUME = ('QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT')
FRACTIONS = ('FGAS', 'FOIL', 'FWAT')
STB_PER_M3 = 6.28981
PSI_PER_BAR = CF_BAR / CF_PSI
# A well's columns that the generation sets, not the draws: left out of the shifts
OUTPUTS = {
    'ID',
    'draw_index',
    'n_cells',
    'best_rate',
    'best_rate_w_lg',
    'scan_best_rate',
    'T_end',
    'T_scan',
    'lift_start',
    'p_change',
    'seconds',
    'cpu_seconds',
    'table_seconds',
    'n_attempts',
    'n_several_stable',
    'n_dead_dates',
}
LABELS_OUT = {
    'split',
    'configuration',
    'outcome',
    'not_producible_reason',
    'error',
    'best_rate_outcomes',
    'scan_outcome',
    'scan_best_rate',
    'flags',
}


# --- Reading -----------------------------------------------------------------------------------------------------------
def folders(gen):
    """The dataset's folder, its logs' and the run's, in a generation's folder."""
    gen = Path(gen)
    data = next(p for p in sorted(gen.glob('manywells-*')) if not p.name.endswith('-logs'))
    return data, data.with_name(data.name + '-logs'), gen / 'run'


def load(gen, *tables, columns=None) -> dict:
    """A dataset's tables; `pvt`'s text columns as categories, which as Python strings would not fit the cap."""
    data, logs, _ = folders(gen)
    out = {}
    for t in tables:
        path = (logs if t in ('drawn_wells', 'redraws') else data) / f'{t}.parquet'
        text = ['name', 'source', 'separation'] if t == 'pvt' else None
        out[t] = pq.read_table(path, columns=columns, read_dictionary=text).to_pandas()
    return out


def metadata(gen) -> dict:
    return json.loads((folders(gen)[0] / 'metadata.json').read_text())


def attempts(gen, columns) -> pd.DataFrame:
    """The run's attempt log of the accepted wells' slots (every record, with the samples' names)."""
    parts = sorted((folders(gen)[2] / 'attempts').glob('part-*.parquet'))
    cols = list(dict.fromkeys(['kind', 'well_outcome', *columns]))
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in parts], ignore_index=True)
    return df[(df['kind'] == 'slot') & (df['well_outcome'] == 'accepted')]


def share(n, total) -> str:
    return f'{n} of {total} ({100 * n / total:.2f}%)' if total else '0 of 0'


def quantiles(x, q=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return ', '.join(f'P{int(100 * v)} {np.quantile(x, v):.4g}' for v in q) if len(x) else 'none'


def lift_rule(df, kind) -> pd.Series:
    """The rule that started a well's lift, natural where none did; nsol-2 wells without a life scan apart."""
    rule = df['lift_start_rule'].astype(object)
    out = rule.where(rule.notna(), 'natural')
    if kind == 'nsol':
        out = out.where(df['T_scan'].notna() | rule.notna(), 'no life scan')
    return out


# --- The checks at generation (datasets-draft.md, Checks at generation) -----------------------------------------------
def _case_check(task):
    """configurations.check, through sample_case, of one sample's well."""
    ID, slot, well, sample, survey, pvt = task
    try:
        sample_case(pd.Series(well), pd.Series(sample), survey, pvt)
        return ID, slot, None
    except Exception as e:  # The check's ValueError, or a row that does not build
        return ID, slot, f'{type(e).__name__}: {e}'


def checks(gen, processes=8):
    meta = metadata(gen)
    t = load(gen, 'wells', 'samples', 'geometry', 'pvt')
    wells, samples, geometry, pvt = t['wells'], t['samples'], t['geometry'], t['pvt']
    nsol = 'WEEKS' in samples
    n, n_slots = meta['n_wells'], meta['n_slots']
    print(f'{meta["dataset"]} ({meta["kind"]}, seed {meta["seed"]}): {len(wells)} wells, {len(samples)} samples')

    # The counts: n wells, n_slots samples each, slots 0 to n_slots - 1 once each, ID dense
    per = samples.groupby('ID')['slot'].agg(['size', 'min', 'max', 'nunique'])
    print(
        f'  counts: {len(wells) == n} wells; ID dense {bool((wells["ID"].to_numpy() == np.arange(len(wells))).all())};'
        f' every well {n_slots} samples {bool((per["size"] == n_slots).all())}, slots 0 to {n_slots - 1} once each'
        f' {bool(((per["min"] == 0) & (per["max"] == n_slots - 1) & (per["nunique"] == n_slots)).all())};'
        f' samples of unknown wells {int((~samples["ID"].isin(wells["ID"])).sum())}'
    )
    print(
        f'  geometry: {len(geometry)} rows, every well {bool(geometry["ID"].drop_duplicates().sort_values().to_numpy().tolist() == list(range(n)))};'
        f' pvt: {len(pvt)} rows of {pvt["pvt_table_id"].nunique()} tables; table wells'
        f' {int((wells["pvt_model"] == "table").sum())}, with their tables'
        f' {bool(wells.loc[wells["pvt_model"] == "table", "pvt_table_id"].isin(pvt["pvt_table_id"]).all())}'
    )

    # The rows: SMP-28's row check and v1's relations between the rates
    bad = {}
    for c in MASS + VOLUME:
        bad[f'{c} < 0'] = int((samples[c] < 0).sum())
    for c in FRACTIONS:
        bad[f'{c} outside [0, 1]'] = int(((samples[c] < 0) | (samples[c] > 1)).sum())
    for c in FEATURES:
        if samples[c].dtype.kind == 'f':
            bad[f'{c} not finite'] = int((~np.isfinite(samples[c])).sum())
    rel = {
        'WLIQ = WOIL + WWAT': (samples['WLIQ'], samples['WOIL'] + samples['WWAT']),
        'WTOT = WLIQ + WGAS + WGL': (samples['WTOT'], samples['WLIQ'] + samples['WGAS'] + samples['WGL']),
        'QLIQ = QOIL + QWAT': (samples['QLIQ'], samples['QOIL'] + samples['QWAT']),
        'QTOT = QLIQ + QGAS + QGL': (samples['QTOT'], samples['QLIQ'] + samples['QGAS'] + samples['QGL']),
        'FGAS + FOIL + FWAT = 1': (samples['FGAS'] + samples['FOIL'] + samples['FWAT'], 1.0),
    }
    worst = {k: float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))) for k, (a, b) in rel.items()}
    print(f'  rows failing the row check: {sum(bad.values())}', {k: v for k, v in bad.items() if v})
    print('  relations, largest relative difference:', {k: f'{v:.1e}' for k, v in worst.items()})

    # Every well in the v2.0.0 configuration: sample_case checks it, at its first sample and, in nsol-2, its first
    # lifted one, where the lift's injection point is set
    first = samples.loc[samples['slot'] == 0]
    picks = [first]
    if nsol:
        picks.append(samples.loc[samples['LIFT'].astype(bool)].sort_values(['ID', 'slot']).groupby('ID').head(1))
    pick = pd.concat(picks, ignore_index=True)
    surveys = {ID: g.sort_values('station') for ID, g in geometry.groupby('ID')}
    tables = {k: g for k, g in pvt.groupby('pvt_table_id')}
    by_id = wells.set_index('ID', drop=False)

    def tasks():
        for _, s in pick.iterrows():
            w = by_id.loc[s['ID']]
            yield (
                int(s['ID']),
                int(s['slot']),
                w.to_dict(),
                s.to_dict(),
                surveys[s['ID']],
                tables[w['pvt_table_id']] if w['pvt_model'] == 'table' else None,
            )

    t0 = time.time()
    with Pool(processes) as pool:
        out = list(pool.imap_unordered(_case_check, tasks(), chunksize=20))
    failed = [o for o in out if o[2]]
    print(
        f'  configurations.check(wp, v2.0.0): {len(out)} samples of {pick["ID"].nunique()} wells built,'
        f' {len(failed)} refused ({time.time() - t0:.0f} s)'
    )
    for o in failed[:10]:
        print('    ', o)

    # The split: 20% test per stratum, ranks 5, 10, ... (SMP-110)
    strata = ['fluid_type', 'well_type', 'geometry_class', 'coverage_tail']
    g = wells.assign(test=wells['split'] == 'test').groupby(strata, observed=True)['test'].agg(['size', 'sum'])
    off = (g['sum'] - g['size'] // 5).abs()
    print(
        f'  split: train {int((wells["split"] == "train").sum())}, test {int((wells["split"] == "test").sum())};'
        f' {len(g)} strata, every one with floor(n/5) test wells {bool((off == 0).all())};'
        f' test share in strata of 50 wells or more {quantiles(g.loc[g["size"] >= 50, "sum"] / g.loc[g["size"] >= 50, "size"], (0, 0.5, 1))}'
    )


# --- The report: drops by group, dead dates, shifts, redraws ----------------------------------------------------------
def _group_table(dw, key, kind):
    """Drawn wells by group: those the filters drop, those the limits drop, by outcome."""
    rows = []
    for g, d in dw.groupby(key, observed=True, dropna=False):
        c = d['outcome'].value_counts()
        filt = int(sum(c.get(o, 0) for o in FILTERS))
        lim = int(sum(c.get(o, 0) for o in LIMITS))
        rows.append(
            dict(
                group=g,
                drawn=len(d),
                kept=int(c.get('accepted', 0)),
                filtered=filt,
                limits=lim,
                limits_of_drawn=100 * lim / len(d),
                limits_past_filters=100 * lim / max(len(d) - filt, 1),
                **{o: int(c.get(o, 0)) for o in LIMITS if c.get(o, 0)},
            )
        )
    return pd.DataFrame(rows).fillna(0)


def _shift_rows(ref, kept, columns, min_n=200):
    """Each column's shift between a reference set of wells and the kept ones: the KS distance, and where the kept
    wells' median lies in the reference's distribution (0.5 is no shift). A column with fewer than min_n values in
    either set is left out."""
    rows = []
    for c in columns:
        a, b = ref[c].dropna().to_numpy(float), kept[c].dropna().to_numpy(float)
        if len(a) < min_n or len(b) < min_n or np.ptp(a) == 0:
            continue
        rows.append(
            dict(
                column=c,
                n_ref=len(a),
                n_kept=len(b),
                ks=ks_2samp(a, b).statistic,
                median_at=float((a < np.median(b)).mean()),
                ref_median=float(np.median(a)),
                kept_median=float(np.median(b)),
            )
        )
    names = [
        'column',
        'n_ref',
        'n_kept',
        'ks',
        'median_at',
        'ref_median',
        'kept_median',
    ]  # Also with no row, as in a pilot
    return pd.DataFrame(rows, columns=names).sort_values('ks', ascending=False)


def _label_shift(ref, kept, columns):
    rows = []
    for c in columns:
        a = ref[c].astype(str).value_counts(normalize=True)
        b = kept[c].astype(str).value_counts(normalize=True)
        d = (b.reindex(a.index.union(b.index), fill_value=0) - a.reindex(a.index.union(b.index), fill_value=0)) * 100
        rows.append(
            dict(
                column=c, max_pp=float(d.abs().max()), value=d.abs().idxmax(), change_pp=float(d.loc[d.abs().idxmax()])
            )
        )
    return pd.DataFrame(rows).sort_values('max_pp', ascending=False)


THRESHOLDS = dict(overall=2.0, decile=10.0, group=10.0)  # Percent of the wells past the filters (SMP-106, D-162)
GROUPS = ('fluid_class', 'well_type', 'lift_rule', 'tail', ['fluid_class', 'lift_rule'])


def _thresholds(dw, past, dec, kind):
    """The thresholds for the final datasets (SMP-106, D-162), each on the drawn wells past the filters: the share
    dropped at the limits overall, in the worst tenth of a drawn input that 1000 wells or more hold (dec, report's
    table of the deciles), and in the worst group of 100 drawn wells or more."""
    overall = 100 * past['outcome'].isin(LIMITS).mean()
    groups = pd.concat([_group_table(dw, key, kind).assign(by=str(key)) for key in GROUPS])
    # nsol-2's wells dropped before their life scan have no lift rule: a group by its outcome, not by its draws
    groups = groups[(groups['drawn'] >= 100) & ~groups['group'].astype(str).str.contains('no life scan')]
    groups = groups.sort_values('limits_past_filters', ascending=False)
    verdict = lambda value, bound: 'holds' if value <= bound else 'EXCEEDED'  # noqa: E731
    print('\n-- thresholds for the final datasets (SMP-106), of the wells past the filters:')
    print(
        f'   overall: {overall:.2f}% dropped at the limits, at most {THRESHOLDS["overall"]}%: {verdict(overall, THRESHOLDS["overall"])}'
    )
    if len(dec):
        worst = 100 * dec['largest'].iloc[0]
        print(
            f'   a tenth of a drawn input: the largest {worst:.1f}% ({dec["column"].iloc[0]}), at most {THRESHOLDS["decile"]}%:'
            f' {verdict(worst, THRESHOLDS["decile"])}'
        )
    else:
        print('   a tenth of a drawn input: no input with 1000 wells')
    if len(groups):
        g = groups.iloc[0]
        print(
            f'   a group of 100 drawn wells or more: the largest {g["limits_past_filters"]:.1f}% ({g["by"]}: {g["group"]},'
            f' {int(g["drawn"])} drawn), at most {THRESHOLDS["group"]}%: {verdict(g["limits_past_filters"], THRESHOLDS["group"])}'
        )
        print(f'   groups past it: {int((groups["limits_past_filters"] > THRESHOLDS["group"]).sum())} of {len(groups)}')
    else:
        print('   a group of 100 drawn wells or more: none')


def report(gen):
    meta = metadata(gen)
    kind = meta['kind']
    t = load(gen, 'drawn_wells', 'wells', 'redraws')
    dw, wells, redraws = t['drawn_wells'], t['wells'], t['redraws']
    samples = load(gen, 'samples')['samples']
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 200)
    fmt = {'float_format': lambda v: f'{v:.4g}'}
    print(f'== {meta["dataset"]}: seed {meta["seed"]}, code {meta["code"]["commit"][:7]}, created {meta["created"]}')

    # Outcomes and the share kept
    c = dw['outcome'].value_counts()
    filt, lim = int(sum(c.get(o, 0) for o in FILTERS)), int(sum(c.get(o, 0) for o in LIMITS))
    print(f'drawn {len(dw)}, kept {share(int(c["accepted"]), len(dw))}; outcomes {c.to_dict()}')
    print(
        f'  dropped by the filters {share(filt, len(dw))}; at the limits {share(lim, len(dw))}, {share(lim, len(dw) - filt)} past the filters'
    )

    # The drops at the limits by group (D-145)
    dw = dw.assign(lift_rule=lift_rule(dw, kind), tail=dw['coverage_tail'].astype(str))
    for key in ('fluid_class', 'well_type', 'lift_rule', 'tail', ['fluid_class', 'lift_rule']):
        print(f'\n-- by {key}:')
        print(_group_table(dw, key, kind).to_string(index=False, **fmt))
    oils = dw[dw['fluid_type'].isin(['black_oil', 'volatile_oil'])]
    print('\n-- oils, by lift rule:')
    print(_group_table(oils, 'lift_rule', kind).to_string(index=False, **fmt))
    # Oils natural where the filter applies that do not pass it: lifted by SMP-103, sol-2's at their date and nsol-2's
    # from their start (D-161; the first beta's nsol-2 dropped them), where a dead date's lift has a later start
    rule = oils['lift_start_rule'].astype(str)
    nnf = oils[(rule == 'no_natural_flow') & ((oils['lift_start'] == 0) if kind == 'nsol' else True)]
    print(
        f'   lifted by the filter because they cannot flow naturally (no_natural_flow): kept {share(int((nnf["outcome"] == "accepted").sum()), len(nnf))}'
    )
    if kind == 'nsol':
        failed = oils['not_producible_reason'].astype(str).isin(['no_operating_point', 'below_floor'])
        dropped = (rule != 'from_start') & (oils['outcome'] == 'not_producible') & failed
        print(
            f'   not lifted from their start by their draws, and dropped there without flow: {share(int(dropped.sum()), len(oils))}'
        )

    # Dead dates (nsol-2, SMP-106)
    if kind == 'nsol':
        acc = wells
        dd = acc[acc['n_dead_dates'] > 0]
        cut = 1 - acc['T_end'] / acc['T_scan']
        print(
            f'\n-- dead dates: counts {meta["counts"]["dead_dates"]}; kept wells with one {share(len(dd), len(acc))},'
            f' by number {dd["n_dead_dates"].value_counts().sort_index().to_dict()};'
            f' drawn wells with one {share(int((dw["n_dead_dates"] > 0).sum()), len(dw))}'
        )
        print(
            f'  kept wells whose lift a dead date started (no_natural_flow): {share(int((acc["lift_start_rule"] == "no_natural_flow").sum()), len(acc))}'
        )
        print(
            f'  kept wells whose life a dead date cut (T_end < T_scan): {share(int((cut > 1e-12).sum()), len(acc))}; the share cut: {quantiles(cut[cut > 1e-12])}'
        )
        print(f'  T_end (years): {quantiles(acc["T_end"])}; T_life: {quantiles(acc["T_life"])}')

    # The shift of each drawn input: all drawn wells against the kept ones, and the wells past the filters against
    # the kept ones (the limits' own shift, acceptance criterion 4), which the share dropped bounds
    num = [c for c in dw.columns if dw[c].dtype.kind in 'fiu' and c not in OUTPUTS and not c.startswith('n_redraws')]
    lab = [c for c in dw.columns if dw[c].dtype.kind in 'OSU' or isinstance(dw[c].dtype, pd.CategoricalDtype)]
    lab = [c for c in lab if c not in LABELS_OUT and c not in ('lift_rule', 'tail')]
    kept = dw[dw['outcome'] == 'accepted']
    past = dw[~dw['outcome'].isin(FILTERS)]
    dropped = dw[dw['outcome'].isin(LIMITS)]
    # Which wells the limits drop: the dropped against the kept, where the dropped wells' median lies among the kept
    names = dict(n_ref='n_kept', n_kept='n_dropped', ref_median='kept_median', kept_median='dropped_median')
    s = _shift_rows(kept, dropped, num, min_n=30).rename(columns=names)
    print(f'\n-- the wells dropped at the limits ({len(dropped)}) against the kept, the ten largest KS:')
    print(s.head(10).to_string(index=False, **fmt))
    # How hard they hit the tails: the share dropped at the limits in each decile of an input, past the filters
    lim = past['outcome'].isin(LIMITS)
    rows = []
    for c in num:
        x = past[c].dropna()
        if len(x) < 1000 or x.nunique() < 20:
            continue
        decile = pd.qcut(x.rank(method='first'), 10, labels=False)
        d = lim.loc[x.index].groupby(decile).mean()
        rows.append(dict(column=c, n=len(x), largest=d.max(), decile=int(d.idxmax()), smallest=d.min()))
    dec = pd.DataFrame(rows, columns=['column', 'n', 'largest', 'decile', 'smallest'])
    dec = dec.sort_values('largest', ascending=False)
    if len(dec):
        print(
            f'   share dropped in a decile of an input: the largest {dec["largest"].iloc[0]:.4f} ({dec["column"].iloc[0]},'
            f' decile {dec["decile"].iloc[0]}); inputs with a decile above 5%: {int((dec["largest"] > 0.05).sum())},'
            f' above 10%: {int((dec["largest"] > 0.10).sum())}, of {len(dec)}'
        )
        print(dec.head(6).to_string(index=False, **fmt))
    else:
        print('   share dropped in a decile of an input: no input with 1000 wells past the filters')
    _thresholds(dw, past, dec, kind)
    for name, ref in (('past the filters', past), ('drawn', dw)):
        s = _shift_rows(ref, kept, num)
        print(f'\n-- shift, {name} ({len(ref)}) against kept ({len(kept)}), the ten largest KS of {len(s)} inputs:')
        print(s.head(10).to_string(index=False, **fmt))
        print(f'   inputs with KS above 0.02: {int((s["ks"] > 0.02).sum())}, above 0.05: {int((s["ks"] > 0.05).sum())}')
        ls = _label_shift(ref, kept, lab)
        print('   labels, the five largest changes of a share (percentage points):')
        print(ls.head(5).to_string(index=False, **fmt))

    # Redraws: per sample by outcome and reason, the share of attempts redrawn, and the samples' shift
    r = redraws[redraws['ID'].notna()]
    print(
        f'\n-- redraws of the kept wells: {len(r)} ({len(r) / len(samples):.4f} per sample); share of the attempts'
        f' redrawn {share(len(r), len(r) + len(samples))}; rows of the redraw log {len(redraws)}, of dropped wells'
        f' {int(redraws["ID"].isna().sum())}'
    )
    print('   by outcome and reason:', r.groupby(['outcome', 'reason'], observed=True, dropna=False).size().to_dict())
    per = r.groupby('ID').size().reindex(wells['ID'], fill_value=0) / meta['n_slots']
    print(
        f'   redraws per sample by well: {quantiles(per, (0.5, 0.9, 0.99, 1))}; wells with none {share(int((per == 0).sum()), len(per))}'
    )
    inputs = [
        c
        for c in ('CHK', 'PDC', 'WGL', 'PRES', 'GOR', 'WLR', 'HTC', 'ROUGH', 'KL', 'WLMAX', 'FORCHA', 'FORCHB')
        if c in r
    ]
    every = pd.concat([samples[inputs], r[inputs]], ignore_index=True)
    s = _shift_rows(every, samples, inputs)
    print('   inputs, every attempt of the kept wells against their samples:')
    print(s.to_string(index=False, **fmt))
    lifted = samples['WGL'] > 0
    ref = pd.concat([samples.loc[lifted, ['WGL']], r.loc[r['WGL'] > 0, ['WGL']]])
    s = _shift_rows(ref, samples.loc[lifted, ['WGL']], ['WGL'])
    print('   WGL where lift gas is injected:', s.to_string(index=False, header=False, **fmt))

    # Several stable roots, and the samples off their table's grid, from the run's attempt log
    a = attempts(gen, ['ID', 'slot', 'attempt', 'outcome', 'off_table', 'several_stable', 't_solve'])
    a = a[a['outcome'] == 'operating_point'].merge(samples[['ID', 'slot', 'attempt']], on=['ID', 'slot', 'attempt'])
    a = a.merge(wells[['ID', 'fluid_class', 'pvt_model']], on='ID')
    print(
        f'\n-- the samples in the attempt log: {len(a)} (of {len(samples)}); several stable roots {share(int(a["several_stable"].sum()), len(a))},'
        f' drawn_wells.n_several_stable {int(kept["n_several_stable"].sum())}'
    )
    tab = a[a['pvt_model'] == 'table']
    off = tab['off_table'].astype(bool)
    print(
        f"   off their table's grid: {share(int(off.sum()), len(a))} of all samples, {share(int(off.sum()), len(tab))} of the table wells' samples, in {tab.loc[off, 'ID'].nunique()} wells; by class",
        tab.loc[off, 'fluid_class'].value_counts().to_dict(),
    )
    print(f'   the solve (s): {quantiles(a["t_solve"], (0.5, 0.95, 0.99, 1))}')

    # The datasheets' numbers
    data, logs, _ = folders(gen)
    sizes = {p.name: p.stat().st_size for folder in (data, logs) for p in sorted(folder.glob('*.parquet'))}
    print('\n-- datasheet numbers')
    print(
        '   rows:',
        {k: v['rows'] for k, v in meta['files'].items()},
        '; logs: drawn_wells',
        len(dw),
        'redraws',
        len(redraws),
    )
    print('   sizes (MB):', {k: round(v / 1e6, 1) for k, v in sizes.items()})
    for col in ('coverage_tail', 'fluid_type', 'fluid_class', 'well_type', 'geometry_class'):
        k_ = wells[col].astype(str).value_counts(normalize=True).round(4).to_dict()
        d_ = dw[col].astype(str).value_counts(normalize=True).round(4).to_dict()
        print(f'   {col}: kept {k_}; drawn {d_}')
    print(
        f'   split: {wells["split"].value_counts().to_dict()}; seed {meta["seed"]}; CPU hours {meta["counts"]["cpu_seconds"] / 3600:.0f}'
    )


# --- The re-solve check (datasets-draft.md, Section 3, D-92) -----------------------------------------------------------
def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _value(v):
    v = v.item() if hasattr(v, 'item') else v
    return None if v is pd.NA else v


def _points(wp, op) -> list:
    """Each correlation's inputs at the grid points of an operating point (validity_ranges.point_rows), and the
    mixture's velocity and no-slip density for the erosional check."""
    fl, geo = wp.fluid, wp.geometry
    ppc, tpc = fl.pseudo_critical
    cos = np.asarray(geo.cos_incl)
    rows = []
    for i, (p, v_g, v_l, alpha, rho_g, rho_l, T) in enumerate(op.state):
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        v_m = alpha * v_g + (1 - alpha) * v_l
        mu_m = float(fl.mixture_viscosity(p, T, alpha, rho_g, rho_l, v_g, v_l))
        v_sg, v_sl = alpha * v_g, (1 - alpha) * v_l
        rho_ns = (v_sg * rho_g + v_sl * rho_l) / (v_sg + v_sl)
        rows.append(
            dict(
                i=i,
                cos_incl=cos[max(i - 1, 0)],
                p=p,
                T=T,
                p_psia=p * PSI_PER_BAR,
                T_F=(T - 273.15) * 1.8 + 32,
                T_C=T - 273.15,
                ppr=p * CF_BAR / ppc,
                tpr=T / tpc,
                rs=float(fl.rs(p, T)),
                Re=rho_m * abs(v_m) * geo.D / mu_m,
                C_e=(v_sg + v_sl) * math.sqrt(rho_ns) / 1.22,  # API RP 14E's C at the point, SI form
                regime=op.flow_regime[i],
            )
        )
    return rows


def _resolve_task(task):
    k, well, sample, survey, pvt, with_casadi = task
    well, sample = pd.Series(well), pd.Series(sample)
    rec = dict(k=k, ID=int(well['ID']), slot=int(sample['slot']))
    t0 = time.time()
    wp, bc = sample_case(well, sample, survey, pvt)
    rs = SSDFSimulator(wp, backend='rust').root_set(bc)
    rec['t_rust'] = time.time() - t0
    op = rs.operating_point
    if op is None:
        return rec | dict(found=False), []
    fl = wp.fluid
    feat = _sample_features(op, bc, fl)
    diff = [key for key in FEATURES if not _same(feat[key], _value(sample[key]))]
    q_l = sample['QLIQ'] * 24 * STB_PER_M3  # STB/d
    rec |= dict(
        found=True,
        differ=';'.join(diff),
        md=float(wp.geometry.md[0]),
        eps_D=wp.friction.roughness / wp.geometry.D,
        api=fl.api,
        sg_gas=fl.sg_gas,
        wlr=sample['WLR'],
        p_r=bc.p_r,
        q_l_stbd=q_l,
        bhp_psig=(sample['PBH'] - 1.01325) * PSI_PER_BAR,
        glr_scf=sample['QGAS'] / sample['QLIQ'] / CF_RS if sample['QLIQ'] > 0 else math.inf,
        glr_total_scf=(sample['QGAS'] + sample['QGL']) / sample['QLIQ'] / CF_RS if sample['QLIQ'] > 0 else math.inf,
    )
    points = [dict(k=k, **p) for p in _points(wp, op)]
    if with_casadi:  # The CasADi backend's operating point, within the verifier's tol_x applied to the features
        t0 = time.time()
        try:
            sim = SSDFSimulator(wp, backend='casadi')
            cop = sim.root_set(bc).operating_point
            start = 'own'
            # As in the review before the run: where CasADi's own search finds no operating point, or another one
            # than the core's, it starts again from the core's root as well
            if cop is None:
                cop, start = sim.root_set(bc, x_guess=op).operating_point, 'core root'
            elif not _deviation(_sample_features(cop, bc, fl), feat, bc)['ok']:
                rec['own_PBH'] = cop.p_0
                again = sim.root_set(bc, x_guess=op).operating_point
                cop, start = (again, 'core root, own search differed') if again is not None else (cop, start)
        except Exception as e:
            cop, start = None, f'{type(e).__name__}: {e}'[:200]
        rec['t_casadi'] = time.time() - t0
        rec['casadi_start'] = start if cop is not None else f'none ({start})'
        if cop is not None:
            rec |= _deviation(_sample_features(cop, bc, fl), feat, bc)
    return rec, points


def _deviation(cf, feat, bc) -> dict:
    """CasADi's features against the core's, by the verifier's tol_x applied to them (D-92)."""
    dev = dict(
        dev_p=max(abs(cf[x] - feat[x]) for x in ('PBH', 'PWH')) / (bc.p_r - bc.p_s),
        dev_T=max(abs(cf[x] - feat[x]) for x in ('TBH', 'TWH')) / (bc.T_r - bc.T_s),
        dev_w=max(abs(cf[x] - feat[x]) for x in MASS) / feat['WTOT'],
        same_choked=cf['CHOKED'] == feat['CHOKED'],
        same_regimes=(cf['FRBH'], cf['FRWH']) == (feat['FRBH'], feat['FRWH']),
        regimes=f'{feat["FRBH"]}/{feat["FRWH"]} vs {cf["FRBH"]}/{cf["FRWH"]}',
    )
    dev['ok'] = max(dev['dev_p'], dev['dev_T'], dev['dev_w']) <= 1e-4 and dev['same_choked'] and dev['same_regimes']
    return dev


def resolve(gen, out, n=1000, n_casadi=100, processes=8):
    meta = metadata(gen)
    t = load(gen, 'wells', 'samples', 'geometry', 'pvt')
    wells, samples, geometry, pvt = t['wells'], t['samples'], t['geometry'], t['pvt']
    rng = np.random.default_rng(meta['seed'])  # The sample of rows: seeded by the dataset's seed
    rows = rng.choice(len(samples), size=n, replace=False)
    by_id = wells.set_index('ID', drop=False)
    wanted = samples.iloc[rows]
    surveys = {ID: g.sort_values('station') for ID, g in geometry[geometry['ID'].isin(wanted['ID'])].groupby('ID')}
    table_ids = by_id.loc[wanted['ID'].unique(), 'pvt_table_id'].dropna().unique()
    tables = {k: g for k, g in pvt[pvt['pvt_table_id'].isin(table_ids)].groupby('pvt_table_id')}
    del pvt

    def tasks():
        for k, (_, s) in enumerate(wanted.iterrows()):
            w = by_id.loc[s['ID']]
            table = tables[w['pvt_table_id']] if w['pvt_model'] == 'table' else None
            yield k, w.to_dict(), s.to_dict(), surveys[s['ID']], table, k < n_casadi

    t0 = time.time()
    with Pool(processes) as pool:
        res = list(pool.imap_unordered(_resolve_task, tasks(), chunksize=1))
    cases = pd.DataFrame([r for r, _ in res]).sort_values('k')
    cases = cases.merge(
        wells[['ID', 'fluid_class', 'pvt_model', 'inflow', 'well_type', 'coverage_tail', 'flags', 'tubing']], on='ID'
    )
    points = pd.DataFrame([p for _, ps in res for p in ps])
    cases.to_parquet(f'{out}_cases.parquet', index=False)
    points.to_parquet(f'{out}_points.parquet', index=False)
    print(f'{meta["dataset"]}: {n} rows re-solved in {time.time() - t0:.0f} s on {processes} processes')
    summarize_resolve(out)


def summarize_resolve(out):
    cases = pd.read_parquet(f'{out}_cases.parquet')
    found = cases['found']
    same = found & (cases['differ'] == '')
    print(
        f'  Rust core: operating point found {share(int(found.sum()), len(cases))}; every feature the same to the bit {share(int(same.sum()), len(cases))}'
    )
    for _, r in cases[found & (cases['differ'] != '')].iterrows():
        print(f'    ID {r["ID"]} slot {r["slot"]}: differ in {r["differ"]}')
    print(f'  solve (s): {quantiles(cases["t_rust"], (0.5, 0.95, 1))}')
    c = cases[cases['casadi_start'].notna()] if 'casadi_start' in cases else cases.iloc[:0]
    if len(c):
        ok = c['ok'].fillna(False).astype(bool)
        print(
            f'  CasADi, {len(c)} rows: operating point {share(int(c["dev_p"].notna().sum()), len(c))}; starts',
            c['casadi_start'].value_counts().to_dict(),
            f'; within tol_x with the same choked state and regimes {share(int(ok.sum()), len(c))}',
        )
        if 'own_PBH' in c:
            for _, r in c[c['own_PBH'].notna()].iterrows():
                print(
                    f"    ID {r['ID']} slot {r['slot']}: CasADi's own search took PBH {r['own_PBH']:.4f} bar; from the core's root, within tol_x {bool(r['ok'])}"
                )
        print(
            f'    largest deviations: p {c["dev_p"].max():.2e}, T {c["dev_T"].max():.2e}, w {c["dev_w"].max():.2e}; time (s) {quantiles(c["t_casadi"], (0.5, 0.95, 1))}'
        )
        for _, r in c[~ok].iterrows():
            print(
                f'    ID {r["ID"]} slot {r["slot"]} {r["fluid_class"]}: {r["casadi_start"]}; dev p {r["dev_p"]:.2e} T {r["dev_T"]:.2e}'
                f' w {r["dev_w"]:.2e}; choked same {r["same_choked"]}; regimes {r["regimes"]}'
            )


# --- The correlations' ranges, on the re-solved samples (specs/model/validity.md) --------------------------------------
# (row, column, table, low, high, wells): the table is 'cases' for a sample's input or operating point, 'points' for a
# grid point of it; the wells are those whose model uses the correlation. None is an open end.
RANGES = [
    ('Beggs and Robinson, Table 1', 'api', 'cases', 16, 58, 'black_oil'),
    ('Beggs and Robinson, Table 1', 'p_psia', 'points', None, 5250 + 14.696, 'black_oil'),
    ('Beggs and Robinson, Table 1', 'T_F', 'points', 70, 295, 'black_oil'),
    ('Beggs and Robinson, Table 1', 'rs', 'points', 20 * CF_RS, 2070 * CF_RS, 'black_oil'),
    ('Abdul-Majeed, dead oil', 'T_C', 'points', 15.56, 54.44, 'black_oil'),
    ('Abdul-Majeed, dead oil', 'api', 'cases', 15, 50, 'black_oil'),
    ('Abdul-Majeed, live oil', 'rs', 'points', 3, 208, 'black_oil'),
    ('Sutton', 'sg_gas', 'cases', 0.571, 1.679, 'black_oil'),
    ('Dranchuk and Abou-Kassem', 'ppr', 'points', 0.2, 30, 'black_oil'),
    ('Dranchuk and Abou-Kassem', 'tpr', 'points', 1.0, 3.0, 'black_oil'),
    ('Lee, Gonzalez and Eakin', 'p_psia', 'points', 100, 8000, 'black_oil'),
    ('Lee, Gonzalez and Eakin', 'T_F', 'points', 100, 340, 'black_oil'),
    ('Hasan et al., Table 2', 'glr_total_scf', 'cases', 104, 5615, 'all'),
    ('Hasan et al., Table 2', 'q_l_stbd', 'cases', 100, 27760, 'all'),
    ('Hasan et al., Table 2', 'wlr', 'cases', 0, 0.98, 'all'),
    ('Hasan et al., Table 2', 'bhp_psig', 'cases', 108, 13755, 'all'),
    ('Hasan et al., Table 2', 'md', 'cases', 1635 * 0.3048, 30650 * 0.3048, 'all'),
    ('Chen', 'Re', 'points', 4000, 4e8, 'all'),
    ('Chen', 'eps_D', 'cases', 5e-7, 0.05, 'all'),
    ('Forchheimer, pressure squared', 'p_r', 'cases', None, 2000 / PSI_PER_BAR, 'forchheimer'),
]


def validity(out):
    cases = pd.read_parquet(f'{out}_cases.parquet')
    points = pd.read_parquet(f'{out}_points.parquet')
    cases = cases[cases['found']]
    points = points.merge(cases[['k', 'pvt_model', 'inflow']], on='k')
    print(
        f'{len(cases)} re-solved samples, {len(points)} grid points; black oil {int((cases["pvt_model"] == "black_oil").sum())}, tables {int((cases["pvt_model"] == "table").sum())}'
    )
    rows = []
    for name, col, table, lo, hi, which in RANGES:
        df = cases if table == 'cases' else points
        if which == 'black_oil':
            df = df[df['pvt_model'] == 'black_oil']
        elif which == 'forchheimer':
            df = df[df['inflow'] == 'forchheimer']
        x = df[col]
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= x < lo
        if hi is not None:
            bad |= x > hi
        shares = bad if table == 'cases' else bad.groupby(df['k']).any()
        rows.append(
            dict(
                row=name,
                variable=col,
                low=lo,
                high=hi,
                wells=which,
                samples=len(shares),
                P0=x.min(),
                P50=x.median(),
                P100=x.max(),
                out=round(100 * shares.mean(), 1),
            )
        )
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('out: % of the samples (that use the correlation) with the input, or a grid point, outside the range')
    # The erosional velocity (API RP 14E), against the tubing flag (D-100): the largest C along each sample's well
    ce = points.groupby('k')['C_e'].max().rename('C_e')
    c = cases.join(ce, on='k')
    flagged = c['flags'].fillna('').str.contains('design_rate_above_tubing_capacity')
    for name, d in (('flagged', c[flagged]), ('not flagged', c[~flagged]), ('all', c)):
        print(
            f'  {name:12s} {len(d):5d} samples: C at its largest {quantiles(d["C_e"], (0.5, 0.95, 1))}; above 100 {share(int((d["C_e"] > 100).sum()), len(d))}, above 150 {share(int((d["C_e"] > 150).sum()), len(d))}'
        )


# --- The rule points for after the beta (preflight-measurements.md, For after the beta) --------------------------------
def rules(gen):
    meta = metadata(gen)
    kind = meta['kind']
    t = load(gen, 'wells', 'drawn_wells')
    wells, dw = t['wells'], t['drawn_wells']
    cols = ['ID', 'slot', 'GOR', 'PDC', 'WGL', 'QGL', 'QGAS', 'QLIQ'] + (['LIFT', 'STAGE'] if kind == 'nsol' else [])
    samples = load(gen, 'samples', columns=cols)['samples']
    print(f'== {meta["dataset"]}: {len(wells)} kept wells, {len(dw)} drawn')

    # 1. Table fluids outside their class (SMP-115, D-127). McCain's classes are by the initial producing GOR as the
    # field reports it (SMP-55), which is the built fluid's `field_gor`; `R_si` is the table's single-stage GOR
    print("\n-- 1. table fluids outside their class, by the built fluid's field GOR")
    bins, names = (
        [0, 356, 588, 8900, DRY_GAS_GOR, np.inf],
        ['black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas'],
    )
    for name, df in (('kept', wells), ('drawn, with a table', dw[dw['R_si'].notna()])):
        tab = df[df['pvt_model'] == 'table']
        by_gor = pd.cut(tab['field_gor'], bins, labels=names).astype(str)
        print(f'  {name}: {len(tab)} table wells; drawn class against the class of the field GOR')
        print(pd.crosstab([tab['fluid_class'].astype(str), tab['saturation_rule'].astype(str)], by_gor).to_string())
        for cls in CLASS_GOR:
            d = tab['fluid_class'] == cls
            out_ = d & (by_gor != cls)
            print(
                f'     {cls}: outside its class {share(int(out_.sum()), int(d.sum()))}; field GOR over the drawn GOR, outside: {quantiles((tab["field_gor"] / tab["R_si_drawn"])[out_], (0, 0.5, 1))}'
            )
    tab = wells[wells['pvt_model'] == 'table']
    moved = pd.cut(tab['field_gor'], bins, labels=names).astype(str) != tab['fluid_class'].astype(str)
    print(
        f"  kept wells whose class would change: {share(int(moved.sum()), len(wells))}; the classes' shares, now and by the field GOR:"
    )
    after = wells['fluid_class'].astype(str).copy()
    after.loc[tab.index[moved]] = pd.cut(tab.loc[moved, 'field_gor'], bins, labels=names).astype(str)
    print(
        '    ',
        wells['fluid_class'].astype(str).value_counts(normalize=True).round(4).to_dict(),
        '->',
        after.value_counts(normalize=True).round(4).to_dict(),
    )
    for cls, (lo, hi) in CLASS_GOR.items():
        ids = tab.loc[(tab['fluid_class'] == cls), 'ID']
        g = samples.loc[samples['ID'].isin(ids), 'GOR']
        print(
            f"  kept {cls}: samples' GOR (producing, at their dates) below the class {share(int((g < lo).sum()), len(g))}, above it {share(int((g > hi).sum()), len(g))}"
        )

    # 2. The downstream pressure's nominal value in the tail
    print('\n-- 2. p_s,nom and PDC by tail and well type')
    for tail in ('core', 'tail'):
        for wt in ('platform', 'onshore', 'subsea'):
            d = wells[(wells['coverage_tail'].astype(str) == tail) & (wells['well_type'].astype(str) == wt)]
            if not len(d):
                continue
            if kind == 'sol':
                p = d['p_s_nom']
                print(
                    f'  {tail} {wt:8s} {len(d):5d} kept: p_s_nom {quantiles(p)}; at 1.5 bar {share(int((np.abs(p - 1.5) < 1e-9).sum()), len(d))}; life stage {d["life_stage"].astype(str).value_counts().to_dict()}'
                )
            s = samples.loc[samples['ID'].isin(d['ID']), 'PDC']
            print(
                f"  {tail} {wt:8s} samples' PDC: {quantiles(s)}; at most 1.5 bar {share(int((s <= 1.5 + 1e-9).sum()), len(s))}"
            )
    if kind == 'nsol':
        tl = wells.loc[wells['coverage_tail'].astype(str) == 'tail', 'ID']
        s = samples[samples['ID'].isin(tl)]
        for stage, g in s.groupby(s['STAGE'].astype(str)):
            print(f'  tail samples at stage {stage}: PDC {quantiles(g["PDC"], (0.05, 0.5, 0.95, 1))}')
    # 3. The tubing flag (D-100)
    print('\n-- 3. design_rate_above_tubing_capacity')
    for name, df in (('drawn', dw), ('kept', wells)):
        f = df['flags'].fillna('').str.contains('design_rate_above_tubing_capacity')
        print(
            f'  {name}: flagged {share(int(f.sum()), len(df))}',
            {
                k: f'{int(v.sum())} of {len(v)} ({100 * v.mean():.1f}%)'
                for k, v in f.groupby([df['coverage_tail'].astype(str), df['fluid_type'].astype(str)])
            },
        )
        print(
            '     by well type:',
            {k: f'{100 * v.mean():.1f}%' for k, v in f.groupby(df['well_type'].astype(str))},
            '; their tubing:',
            df.loc[f, 'tubing'].astype(str).value_counts().to_dict(),
        )
    f = dw['flags'].fillna('').str.contains('design_rate_above_tubing_capacity')
    print(
        f'  kept of the drawn: flagged {share(int((dw.loc[f, "outcome"] == "accepted").sum()), int(f.sum()))}, not flagged {share(int((dw.loc[~f, "outcome"] == "accepted").sum()), int((~f).sum()))}'
    )

    # 4. The lift capacity's bounds (SMP-80, D-31)
    print('\n-- 4. w_lg_max at the bounds')
    lifted = wells[wells['lift'] == 'gas_lift'] if kind == 'sol' else wells[wells['lift_start'].notna()]
    cap = lifted['w_lg_max']
    at_low, at_high = np.abs(cap - 0.1) < 1e-12, np.abs(cap - 3.5) < 1e-12
    print(
        f'  kept lifted wells {len(lifted)}: w_lg_max at 0.1 kg/s {share(int(at_low.sum()), len(lifted))}, at 3.5 kg/s {share(int(at_high.sum()), len(lifted))}; {quantiles(cap)}'
    )
    print(
        '     at 0.1, by fluid type:',
        {k: f'{int(v.sum())} of {len(v)}' for k, v in at_low.groupby(lifted['fluid_type'].astype(str))},
        '; by well type:',
        {k: f'{int(v.sum())} of {len(v)}' for k, v in at_low.groupby(lifted['well_type'].astype(str))},
    )
    print(
        f'     glr_cap at 0.1: {quantiles(lifted.loc[at_low, "glr_cap"])}; elsewhere {quantiles(lifted.loc[~at_low & ~at_high, "glr_cap"])}'
    )
    flag = lifted['flags'].fillna('').str.contains('lift_capacity_at_bound')
    print(
        f'     flagged lift_capacity_at_bound: {share(int((flag & at_low).sum()), int(at_low.sum()))} of those at 0.1, {int((flag & ~at_low).sum())} others'
    )
    s = samples[samples['ID'].isin(lifted['ID']) & (samples['WGL'] > 0) & (samples['QLIQ'] > 0)]
    s = s.merge(lifted[['ID', 'fluid_type']], on='ID')
    s['glr'] = s['QGL'] / s['QLIQ']
    s['low'] = s['ID'].isin(lifted.loc[at_low, 'ID'])
    oil = s['fluid_type'].astype(str).isin(['black_oil', 'volatile_oil'])
    for name, d in (
        ('oils at 0.1', s[s['low'] & oil]),
        ('gas wells at 0.1', s[s['low'] & ~oil]),
        ('oils above it', s[~s['low'] & oil]),
        ('gas wells above it', s[~s['low'] & ~oil]),
    ):
        print(
            f'  {name:18s} {d["ID"].nunique():5d} wells, {len(d):7d} lifted samples: injected GLR (Sm3/Sm3) {quantiles(d["glr"], (0.05, 0.5, 0.95))};'
            f' above 1000 {share(int((d["glr"] > 1000).sum()), len(d))}; liquid (Sm3/d) {quantiles(d["QLIQ"] * 24, (0.05, 0.5, 0.95))}; lift gas (Sm3/d) {quantiles(d["QGL"] * 24, (0.05, 0.5, 0.95))}'
        )
        if 'gas' in name:
            print(
                f'      formation gas (Sm3/d) {quantiles(d["QGAS"] * 24, (0.05, 0.5, 0.95))}; lift gas over formation gas {quantiles(d["QGL"] / d["QGAS"], (0.05, 0.5, 0.95))}'
            )


# --- A higher redraw limit (SMP-106) -------------------------------------------------------------------------------------
def _limit_task(task):
    kind, seed, k, tables, limit = task
    generate._REDRAW_LIMIT = limit  # Read by generate._solve_slots when it runs
    t0 = time.process_time()
    res = (generate.generate_nsol2_well if kind == 'nsol' else generate.generate_sol2_well)(seed, k, tables)
    return k, res['info']['outcome'], time.process_time() - t0


def limit(gen, new_limit, rule, processes=8):
    """The wells that the redraw limit dropped, those with the lift rule given, generated again with a higher limit:
    which are kept, and their CPU time against the run's."""
    meta = metadata(gen)
    kind, seed = meta['kind'], meta['seed']
    dw = load(gen, 'drawn_wells', columns=['draw_index', 'outcome', 'lift_start_rule', 'cpu_seconds'])['drawn_wells']
    d = dw[(dw['outcome'] == 'redraw_limit') & (dw['lift_start_rule'] == rule)]
    tables = str(Path(gen) / 'tables')
    with Pool(processes) as pool:
        out = pool.map(_limit_task, [(kind, seed, int(k), tables, new_limit) for k in d['draw_index']], chunksize=1)
    res = pd.DataFrame(out, columns=['draw_index', 'outcome', 'cpu']).merge(d, on='draw_index')
    print(
        f'{meta["dataset"]}: {len(res)} wells with lift rule {rule} dropped at the redraw limit of'
        f' {meta["limits"]["redraws"]}; at {new_limit}:',
        res['outcome_x'].value_counts().to_dict(),
        f'; CPU {res["cpu"].sum():.0f} s against {res["cpu_seconds"].sum():.0f} s in the run;'
        f' every well at the limit took {dw.loc[dw["outcome"] == "redraw_limit", "cpu_seconds"].sum():.0f} s',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('limit')
    p.add_argument('gen')
    p.add_argument('limit', type=int)
    p.add_argument('rule')
    p.add_argument('--processes', type=int, default=8)
    for name in ('checks', 'report', 'rules'):
        p = sub.add_parser(name)
        p.add_argument('gen')
        if name == 'checks':
            p.add_argument('--processes', type=int, default=8)
    p = sub.add_parser('resolve')
    p.add_argument('gen')
    p.add_argument('out')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--casadi', type=int, default=100)
    p.add_argument('--processes', type=int, default=8)
    p = sub.add_parser('validity')
    p.add_argument('out')
    args = parser.parse_args()
    if args.cmd == 'checks':
        checks(args.gen, args.processes)
    elif args.cmd == 'report':
        report(args.gen)
    elif args.cmd == 'resolve':
        resolve(args.gen, args.out, args.n, args.casadi, args.processes)
    elif args.cmd == 'validity':
        validity(args.out)
    elif args.cmd == 'limit':
        limit(args.gen, args.limit, args.rule, args.processes)
    else:
        rules(args.gen)


if __name__ == '__main__':
    main()
