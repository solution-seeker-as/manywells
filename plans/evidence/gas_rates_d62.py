"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 06 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Gas wells' rates under the two draws of the non-Darcy coefficient, with feature 022's gas inflow, for the checkpoint's
item D2 (plans/sampler-redesign/checkpoint.md; D-62, D-23, D-61, D-100). It draws the sampler draft's wells k = 0, 1,
2, ... (sampler_draft_draws.py, seed 2026) until it has N gas and condensate wells (wet gas, dry gas and gas
condensate, core and tail), and solves each under both options of sampler_draft_draws.NONDARCY: 'r3', D-23's
LogU(2e-4, 2e-3) d/Mscf, and 'd62', D-62's estimate from Economides et al.'s Eq. 4-34 at a perforated share, times its
error factor. Both options draw the same wells; only D and what follows from it differ (b, the design rate, and so the
tubing and the bean).

Each well is mapped to v2.0.0 by the prototype's interim mapping (sampler_prototype.Builder) with one change: the
inflow is feature 022's Forchheimer.from_gas_rate(a_gas, b_gas, rho_g), the draw's coefficients in bar2 per Sm3/d of
separator gas with mu_g Z at p_r (not D-61's fit to the pseudo-pressure, which R9's sampler makes). Condensates keep
the interim black oil at their API, gas gravity and GOR (their tables come with R9), and their rates carry it. Every
well is solved on the Rust core only:

- the best rate (D-50): a fully open choke at the nominal state, lift as drawn (a lifted well at 0, 0.5 and 1 of its
  lift-gas capacity); a well without an operating point there, or below half its floor (D-34), is not producible, and
  the prototype would drop it;
- a producible well's first 10 sol-2 samples (sampler_prototype.sol2_inputs with the well's own generators, j = 0 to
  9), without redraws: a sample without an operating point is left out of the well's median.

The rate is the formation gas, 24 QGAS in Sm3/d; a well's median-sample rate is the median over its samples with an
operating point. Also recorded: whether a sample is choked; the wellhead's no-slip mixture velocity over the erosional
velocity 1.22 c_e / sqrt(rho_ns) (API RP 14E with the well's c_e, as the draws size the tubing); the non-Darcy share of
the pressure-squared drop, b q^2 / (p_r^2 - p_wf^2), at the design rate and at the best rate's operating point; D; and
D-25's flag design_rate_above_tubing_capacity, a check (D-100).

The elasticity: a subset of the wells, the first M of each well type in draw order, is solved again with the
permeability lowered: 'inflow_half', the well as drawn with only a and b from half the permeability, all else equal,
the tubing too; 'draw_half', 'draw_quarter' and 'draw_eighth', the permeability times 1/2, 1/4 or 1/8 inside the draw
(PERM's medians and bounds scaled, so that every uniform is the same and k scales exactly), so that the design rate,
the tubing and the bean follow as the draft draws them. Per well, x = ln(q(k) / q(k s)) / ln(1/s) of the median-sample
rate: rate proportional to k^x. With it, the gas classes' permeability medians by well type can be set without
another run.

The report compares the rates with EIA's figures for the Gulf of Mexico's gas wells, a mean of 32,700 Sm3/d and 2.3%
above 270,000 Sm3/d (EIA 2025, Table B32, as [WP §5] quotes them). Per-well medians by region and well type are not
in this script: they are kept outside the repo [own analysis of BOEM's OGOR-A production, ANP's production by well
and NSTA's production by well], and no value computed from real-well data is in this script or its output.

Feature 022 is on develop since 2026-11-06 (#38). --draws names the directory of the draft's scripts (by default
this script's; a copy keeps a run apart from edits made while it runs). From the repository root,
with the sandbox off (the pool needs Unix sockets), each process capped:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 7200 .venv/bin/python plans/evidence/gas_rates_d62.py 700 \\
        --draws DIR --processes 10 --out OUT)
    ... --report OUT     the report again, without solving

    N                gas and condensate wells to draw (default 700)
    --elasticity M   wells per well type in the elasticity subset (default 70; 0 leaves it out)
    --out OUT        the directory for wells.parquet (one row per well, option and variant), samples.parquet and
                     run.json
"""

import argparse
import json
import math
import multiprocessing
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.inflow import Forchheimer
from manywells.sampling.wells import rng_for

OPTIONS = ('r3', 'd62')
SCALES = {'base': 1.0, 'inflow_half': 0.5, 'draw_half': 0.5, 'draw_quarter': 0.25, 'draw_eighth': 0.125}  # k's factor
LOWERED = tuple(SCALES)[1:]
N_SAMPLES = 10  # sol-2 samples per well, j = 0 to 9
GAS = ('gas', 'gas_condensate')
WELL_TYPES = ('platform', 'subsea', 'onshore')
CLASSES = ('wet_gas', 'dry_gas', 'gas_condensate')
BRACKET = 270_000.0  # Sm3/d, EIA's top bracket for gas wells (10 MMcf/d)
EIA_MEAN, EIA_ABOVE = 32_700.0, 0.023  # Gulf of Mexico gas wells: mean rate and share above BRACKET [WP §5]
FLAG = 'design_rate_above_tubing_capacity'

_DRAFT = {}


def _draft(draws):
    """The draft's modules (sampler_draft_draws, sampler_prototype) from the directory draws, imported once."""
    if draws not in _DRAFT:
        sys.path.insert(0, str(Path(draws).resolve()))
        import sampler_draft_draws  # noqa: E402
        import sampler_prototype  # noqa: E402

        _DRAFT[draws] = sampler_draft_draws, sampler_prototype
    return _DRAFT[draws]


def draw(dd, seed, k, option, scale=1.0):
    """Well k under the non-Darcy option, with the permeability times scale: PERM's medians and bounds scaled, so that
    the same uniforms give k times scale, and every other draw is the same."""
    perm, nondarcy = dd.PERM, dd.NONDARCY
    dd.NONDARCY = option
    dd.PERM = {t: (m * scale, s, lo * scale, hi * scale) for t, (m, s, lo, hi) in perm.items()}
    try:
        return dd.draw_well(seed, k)
    finally:
        dd.PERM, dd.NONDARCY = perm, nondarcy


def gas_builder(proto):
    class GasBuilder(proto.Builder):
        """The prototype's Builder with feature 022's Forchheimer inflow from the well's a_gas and b_gas."""

        def case(self, inp):
            wp, bc = super().case(inp)
            return replace(wp, inflow=Forchheimer.from_gas_rate(self.w['a_gas'], self.w['b_gas'], wp.fluid.rho_g)), bc

    return GasBuilder


def wellhead_ratio(root, c_e):
    """The operating point's no-slip mixture velocity at the wellhead over the erosional velocity 1.22 c_e/sqrt(rho)."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = root.state[-1]
    v_m = alpha * v_g + (1 - alpha) * v_l
    rho_ns = (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l) / v_m
    return float(v_m / (1.22 * c_e / math.sqrt(rho_ns)))


def solve_well(dd, proto, k, option, variant, seed):
    """One well under an option and a variant: its row and its samples' rows."""
    w, stations = draw(dd, seed, k, option, 1.0 if variant == 'inflow_half' else SCALES[variant])
    if variant == 'inflow_half':  # only the inflow's coefficients from half the permeability
        half, _ = draw(dd, seed, k, option, 0.5)
        w = dict(w, a_gas=half['a_gas'], b_gas=half['b_gas'], perm=half['perm'], q_g_ref=half['q_g_ref'])
    lifted = w['lift'] == 'gas_lift'
    w_max = w['w_lg_max'] if lifted else 0.0
    builder = gas_builder(proto)(w, stations, design=w if lifted else None)
    fixed = proto.well_fixed(w)
    p_r, b, dd_ = w['p_r'], w['b_gas'], w['delta_d']
    nominal = dict(fixed, p_r=p_r, gor=w['R_go'], wlr=w['wlr'], wgr=w['wgr'], p_s_nom=w['p_s'])
    row = dict(draw=k, option=option, variant=variant, scale=SCALES[variant])
    row |= {c: w[c] for c in ('tail', 'well_type', 'fluid_class', 'geometry_family', 'stage', 'drive', 'lift')}
    row |= {c: w.get(c, math.nan) for c in ('perm', 'h_net', 'skin', 'D_nondarcy', 'D_eq434', 'perf_share')}
    row |= {c: w[c] for c in ('a_gas', 'b_gas', 'a_ratio', 'b_ratio', 'q_g_ref', 'AOF', 'p_r', 'p_s', 'D', 'c_e')}
    row['q_cap_gas'] = w['q_cap'] * 1e6 / (1e6 / w['R_i'] + 10)  # the tubing's erosional capacity at the design state
    row['flagged'] = FLAG in w['flags']
    row['nd_share_design'] = b * w['q_g_ref'] ** 2 / (p_r**2 - ((1 - dd_) * p_r) ** 2)
    log = []
    br = proto.best_rate(builder, nominal, lifted, w_max, proto.FLOOR[w['well_type']], log, {})
    row['status'] = br['status']
    ops = [r for r in log if proto.has_op(r)]
    if ops:
        top = max(ops, key=lambda r: r['q_oe'])
        q = 24 * top['QGAS']
        row.update(best_q_g=q, best_q_oe=top['q_oe'], best_choked=bool(top['CHOKED']), best_PBH=top['PBH'])
        row.update(
            best_v_ratio=wellhead_ratio(top['_root'], w['c_e']), nd_share_best=b * q**2 / (p_r**2 - top['PBH'] ** 2)
        )
    samples = []
    if br['status'] == 'pass':
        for j in range(N_SAMPLES):
            inp = dict(fixed, lift_on=lifted, **proto.sol2_inputs(rng_for(seed, k, j, 'v2', 'sample'), w))
            rec = proto.attempt(builder, inp)
            s = dict(draw=k, option=option, variant=variant, j=j, outcome=rec['outcome'], reason=rec['reason'])
            s |= {c: inp[c] for c in ('u', 'w_lg', 'p_s', 'p_r', 'gor', 'wlr')}
            if proto.has_op(rec):
                s.update(q_g=24 * rec['QGAS'], choked=bool(rec['CHOKED']), PBH=rec['PBH'], PWH=rec['PWH'])
                s['v_ratio'] = wellhead_ratio(rec['_root'], w['c_e'])
            samples.append(s)
    ok = [s for s in samples if 'q_g' in s]
    row['n_samples'], row['n_samples_op'] = len(samples), len(ok)
    if ok:
        q = np.array([s['q_g'] for s in ok])
        row.update(median_q_g=float(np.median(q)), mean_q_g=float(q.mean()))
        row['n_choked'] = sum(s['choked'] for s in ok)
        row['n_erosional'] = sum(s['v_ratio'] > 1 for s in ok)
    return row, samples


def task(args):
    k, option, variant, seed, draws = args
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        dd, proto = _draft(draws)
        row, samples = solve_well(dd, proto, k, option, variant, seed)
    except Exception:  # noqa: BLE001 - recorded, not raised
        row, samples = dict(draw=k, option=option, variant=variant, error=traceback.format_exc()[-1500:]), []
    row.update(seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0)
    return row, samples


def cases(n, seed, draws):
    """The draw indices and labels of the first n gas and condensate wells (the labels do not depend on the option)."""
    dd, _ = _draft(draws)
    out, k = [], 0
    while len(out) < n:
        w, _ = draw(dd, seed, k, 'r3')
        if w['fluid_type'] in GAS:
            out.append((k, w['well_type'], w['tail']))
        k += 1
    return out, k


def run(n, seed, draws, processes, m, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    wells, n_drawn = cases(n, seed, draws)
    subset = [k for wt in WELL_TYPES for k in [i for i, t, _ in wells if t == wt][:m]]
    todo = [(k, o, 'base', seed, draws) for o in OPTIONS for k, _, _ in wells]
    todo += [(k, o, v, seed, draws) for o in OPTIONS for v in LOWERED for k in subset]
    print(f'{len(wells)} gas and condensate wells in draws 0 to {n_drawn - 1}; {len(subset)} in the elasticity subset; '
          f'{len(todo)} tasks', flush=True)  # fmt: skip
    rows, samples = [], []
    with multiprocessing.get_context('forkserver').Pool(processes, maxtasksperchild=40) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(task, todo, chunksize=1)):
            rows.append(r)
            samples += s
            if (i + 1) % 100 == 0:
                print(f'{i + 1} of {len(todo)} tasks, {time.perf_counter() - t0:.0f} s', flush=True)
    wall = time.perf_counter() - t0
    df, sm = pd.DataFrame(rows).sort_values(['option', 'variant', 'draw']), pd.DataFrame(samples)
    df.to_parquet(out / 'wells.parquet', index=False)
    sm.to_parquet(out / 'samples.parquet', index=False)
    meta = dict(n=n, seed=seed, draws=str(draws), processes=processes, elasticity=m, n_drawn=n_drawn, tasks=len(todo))
    meta |= dict(wall_seconds=wall, cpu_seconds=float(df.cpu_seconds.sum()), python=sys.version)
    (out / 'run.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(todo)} tasks in {wall:.0f} s wall on {processes} processes, {meta["cpu_seconds"]:.0f} s CPU in tasks')
    return df, sm


# --- Report -------------------------------------------------------------------------------------------------------------
def pct(s, ps):
    s = pd.Series(s, dtype=float).dropna()
    return np.percentile(s, ps) if len(s) else np.full(len(ps), np.nan)


def summary(g):
    """One group's measurements: rates in thousands of Sm3/d, shares in %."""
    n = len(g)
    prod = g[g.status == 'pass']
    q = prod.median_q_g.dropna()
    p = pct(q, (10, 25, 50, 75, 90)) / 1e3
    n_op = prod.n_samples_op.sum()
    d = pct(g.D_nondarcy, (10, 50, 90))
    return pd.Series(
        {
            'n': n,
            'no_op%': 100 * (g.status == 'no_operating_point').mean(),
            'floor%': 100 * (g.status == 'below_floor').mean(),
            'fail%': 100 * (g.status == 'solver_failure').mean(),
            'P10': p[0],
            'P25': p[1],
            'P50': p[2],
            'P75': p[3],
            'P90': p[4],
            'mean': q.mean() / 1e3,
            '>270k%': 100 * (q > BRACKET).mean() if len(q) else np.nan,
            'best_P50': pct(prod.best_q_g, (50,))[0] / 1e3,
            'choked%': 100 * prod.n_choked.sum() / n_op if n_op else np.nan,
            'v>ve%': 100 * prod.n_erosional.sum() / n_op if n_op else np.nan,
            'nd_P50': pct(g.nd_share_design, (50,))[0],
            'nd>0.5%': 100 * (g.nd_share_design > 0.5).mean(),
            'D_P10': d[0] * 1e4,
            'D_P50': d[1] * 1e4,
            'D_P90': d[2] * 1e4,
            'flag%': 100 * g.flagged.mean(),
        }
    )


def table(df, keys):
    t = df.groupby(keys, observed=True).apply(summary, include_groups=False)
    return t.round(1).astype({'n': int})


def elasticity(df):
    """Per well of the subset: x between k and each lowered permeability, from the median-sample rates."""
    base = df[df.variant == 'base'].set_index(['option', 'draw'])
    rows = []
    for (option, variant), g in df[df.variant != 'base'].groupby(['option', 'variant']):
        g = g.set_index('draw')
        b = base.loc[option].loc[g.index]
        ok = (b.status == 'pass') & (g.status == 'pass')
        x = np.log(b.median_q_g / g.median_q_g) / math.log(1 / SCALES[variant])
        rows.append(
            pd.DataFrame(
                dict(option=option, variant=variant, well_type=b.well_type, tail=b['tail'], x=x.where(ok)),
                index=g.index,
            ).assign(lost=(b.status == 'pass') & (g.status != 'pass'), q=g.median_q_g.where(g.status == 'pass'))
        )
    return pd.concat(rows).reset_index()


def report(df, sm):
    df = df[df.error.isna()] if 'error' in df else df
    base = df[df.variant == 'base'].copy()
    base['core'] = np.where(base['tail'], 'tail', 'core')
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 200)
    print(
        'Rates: the median-sample formation gas rate (thousands of Sm3/d) over producible wells, P10 to P90 and mean;'
    )
    print(
        '>270k%: wells above 270,000 Sm3/d; no_op%, floor%, fail%: wells not producible at the best rate (no operating'
    )
    print(
        'point, below half the floor, a solver failure); best_P50: the best rate; choked% and v>ve%: samples choked and'
    )
    print(
        'above the erosional velocity at the wellhead; nd_P50, nd>0.5%: the non-Darcy share of the drop at the design'
    )
    print('rate; D_P10 to D_P90: D in 1e-4 d/Mscf; flag%: wells flagged design_rate_above_tubing_capacity.\n')
    for option in OPTIONS:
        b = base[base.option == option]
        print(f'=== {option} ===')
        print(table(b, ['core']).to_string(), '\n')
        print(table(b, ['core', 'well_type']).to_string(), '\n')
        print(table(b, ['core', 'fluid_class']).to_string(), '\n')
        print(table(b, ['core', 'well_type', 'fluid_class']).to_string(), '\n')
        off = b[(b.core == 'core') & (b.well_type != 'onshore') & (b.status == 'pass')].median_q_g.dropna()
        print(
            f'Core offshore wells: mean {off.mean():,.0f} Sm3/d and {100 * (off > BRACKET).mean():.1f}% above 270,000 '
            f"Sm3/d, against the Gulf of Mexico gas wells' {EIA_MEAN:,.0f} Sm3/d and {100 * EIA_ABOVE:.1f}% (EIA, "
            'calendar-day averages of a mature shelf; [WP §5])\n'
        )
    both = base.pivot_table(index='draw', columns='option', values='median_q_g')
    r = (both['d62'] / both['r3']).dropna()
    print(f'd62 / r3, median-sample rate per well, P10 / P50 / P90: {np.round(pct(r, (10, 50, 90)), 2)} (n={len(r)})')
    r3 = base[base.option == 'r3']
    ratio = r3.D_nondarcy / r3.D_eq434
    print(
        f"R3's D over Eq. 4-34 perforated over the net pay, P5 / P50 / P95: {np.round(pct(ratio, (5, 50, 95)), 2)}; "
        f'core {np.round(pct(ratio[~r3["tail"]], (5, 50, 95)), 2)}\n'
    )
    if (df.variant != 'base').any():
        el = elasticity(df)
        print('Elasticity: rate proportional to k^x, x = ln(q(k)/q(sk))/ln(1/s) of the median-sample rate, per well;')
        print("lost%: producible at k, not at sk. P50 at sk: the subset's median-sample rate (thousands of Sm3/d).")
        for option in OPTIONS:
            print(f'=== {option} ===')
            sub = base[(base.option == option) & base.draw.isin(el.draw)]
            for wt in WELL_TYPES:
                s0 = sub[(sub.well_type == wt) & (sub.status == 'pass')].median_q_g
                print(
                    f'  {wt}: subset {int((sub.well_type == wt).sum())} wells, P50 at k {np.median(s0.dropna()) / 1e3:.0f}'
                )
                for variant in LOWERED:
                    e = el[(el.option == option) & (el.variant == variant) & (el.well_type == wt)]
                    for label, ee in (('all', e), ('core', e[~e['tail']])):
                        x = ee.x.dropna()
                        print(
                            f'    {variant:12s} {label:4s}: x P25 / P50 / P75 {np.round(pct(x, (25, 50, 75)), 2)}, '
                            f'mean {x.mean():.2f} (n={len(x)}); lost {100 * ee.lost.mean():.1f}%; '
                            f'P50 at sk {pct(ee.q, (50,))[0] / 1e3:.0f}'
                        )
            steps = ('draw_half', 'draw_quarter', 'draw_eighth')
            for hi, lo in zip(steps, steps[1:]):
                a = el[(el.option == option) & (el.variant == hi)].set_index('draw')
                z = el[(el.option == option) & (el.variant == lo)].set_index('draw')
                xs = np.log(a.q / z.q) / math.log(2)
                for wt in WELL_TYPES:
                    v = xs[z.well_type == wt].dropna()
                    print(
                        f'  {wt}: x from k{SCALES[hi]:g} to k{SCALES[lo]:g} (draw) P25 / P50 / P75 '
                        f'{np.round(pct(v, (25, 50, 75)), 2)} (n={len(v)})'
                    )
            print()
    print('Cost per task (one well, 11 to 13 solves) on one process, s: P50 / P95 '
          f'{np.round(pct(df.seconds, (50, 95)), 1)}; total CPU in tasks {df.cpu_seconds.sum():.0f} s')  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int, nargs='?', default=700)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--draws', default=str(Path(__file__).parent))
    ap.add_argument('--processes', type=int, default=10)
    ap.add_argument('--elasticity', type=int, default=70)
    ap.add_argument('--out', default='gas_rates_d62')
    ap.add_argument('--report', help='report a saved run (its OUT directory), without solving')
    args = ap.parse_args()
    if args.report:
        report(
            pd.read_parquet(Path(args.report) / 'wells.parquet'), pd.read_parquet(Path(args.report) / 'samples.parquet')
        )
        return
    report(*run(args.n, args.seed, args.draws, args.processes, args.elasticity, args.out))


if __name__ == '__main__':
    main()
