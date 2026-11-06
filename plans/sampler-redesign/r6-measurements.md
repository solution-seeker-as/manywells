# Sampler redesign — R6: the prototype and its measurements

*Step R6 of `plans/sampler-redesign-plan.md`, 2026-11-02. Evidence for Bjarne's checkpoint (R8). Plans are drafts, not specs. The draft procedure is `sampling-draft.md` (R3, R4, revised by R7), the dataset design `datasets-draft.md` (R7), and R6's changes to the draft are D-100 to D-103 in `decisions.md`.*

R6 built a prototype of the draft, ran a pilot of each dataset, measured what the draft asks R6 to measure (`sampling-draft.md`, What R6 must measure), changed the draft where the measurements showed it failing a goal or an acceptance criterion, and ran the pilots again. This file gives the numbers, with the script and seed behind each table, a summary per acceptance criterion and per goal, and what needs Bjarne's ruling.

**The interim mapping.** R5's gas-rate inflow and PVT tables do not exist yet. Every result here uses the draft's interim mapping (`sampling-draft.md`, Mapping): volatile oils and gas condensates are the black oil with their API gravity, gas gravity, separator and GOR, and gas and condensate wells flow in on a productivity index that gives the liquid the gas carries at the design drawdown. Every number on volatile oils, condensates and gas wells, and every number over all wells, depends on it. R5's D-69 also proposes a CasADi-only prototype of the gas-rate inflow on a subset; R6 did not run it.

## What was run

The prototype is `plans/evidence/sampler_prototype.py` (the generator) and `sampler_prototype_report.py` (the measurements). It draws wells and samples with R3's and R4's draw-only scripts (`sampler_draft_draws.py`, `sampler_draft_samples.py`), maps each well to `v2.0.0` and checks it with `configurations.check`, solves every attempt with the Rust core, retries the core's solver failures with the CasADi backend (D-54), and applies the producibility filter, the `nsol-2` life scan and the redraw rules with their limits (D-50, D-53 to D-56, D-80 to D-84). It writes the tables of `datasets-draft.md` (`wells`, `samples`, `geometry`, `pvt` empty, `redraws`, `drawn_wells`, `metadata.json`), and the run's own log of every attempt with its search counts and times (`attempts.parquet`).

Two revisions of the draft were run. *Draft* is the draft as R3, R4 and R7 left it, with the geometry ruled on 2026-11-02 (task 1 below). *R6* adds D-100 to D-103. `--revision draft` reproduces the first.

| Run | Revision | Seed | Wells drawn | Accepted | Samples | Solves | Wall time | Processes |
|---|---|---|---|---|---|---|---|---|
| `sol-2` | draft | 2026 | 442 | 400 | 80,000 | 85,627 | 27.2 min | 12 |
| `nsol-2` | draft | 2027 | 106 | 100 | 20,000 | 25,871 | 9.8 min + 1.0 min | 11, then 12 |
| `sol-2` | R6 | 2026 | 450 | 400 | 80,000 | 85,942 | 25.6 min | 11 |
| `nsol-2` | R6 | 2027 | 109 | 100 | 20,000 | 24,666 | 7.8 min | 11 |

*Wells drawn* counts the draw indices up to the 400th (or 100th) accepted well; the pool solved a few more, which the tables leave out. Two `nsol-2` draft wells crashed on a date one ulp past the life span, where R4's pressure trend is complex; after the fix (the date clamped to $T_\text{life}$) they were solved again with the same draws (the second wall time). Every pool process ran with `OMP_NUM_THREADS=1` under a 2.6 GB address-space cap and a timeout, the reports under 4 GB; none hit either. At most 12 processes ran at once.

```console
(ulimit -v 2600000; OMP_NUM_THREADS=1 timeout 4500 .venv/bin/python plans/evidence/sampler_prototype.py run sol 400 OUT/sol-2-r6 --seed 2026 --processes 11 --revision r6)
(ulimit -v 2600000; OMP_NUM_THREADS=1 timeout 3600 .venv/bin/python plans/evidence/sampler_prototype.py run nsol 100 OUT/nsol-2-r6 --seed 2027 --processes 11 --revision r6)
(ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 1200 .venv/bin/python plans/evidence/sampler_prototype_report.py report OUT/sol-2-r6 --figures FIG --sol1 SOL1_ZIP)
.venv/bin/python plans/evidence/sampler_prototype_report.py resolve OUT/sol-2-r6 200 --backend casadi --seed 6
.venv/bin/python plans/evidence/sampler_prototype_report.py validity OUT/sol-2-r6 2000 --seed 7
```

The pool needs the command sandbox off (Python 3.14's forkserver uses Unix sockets). Unless a table says otherwise, its numbers are from the R6 runs and the report of that run.

### Task 1: the geometry as ruled

`sampler_draft_draws.py` now draws the four classes of D-9 at 0.25 each, for every well type and in core and tail alike: the slightly deviated class as `build_and_hold` with $\theta_h \sim U(5°, 20°)$, the deviated class as two thirds `build_and_hold` with $\theta_h \sim U(20°, 60°)$ and one third `s_shaped` (D-57), and the highly deviated class as half `high_angle` with $\theta_\text{heel} \sim U(60°, 75°)$ and half `build_and_hold` with $\theta_h \sim U(60°, 75°)$. The geometry block keeps its own generator. Rerun on 2000 wells (seed 2026) and compared column by column with the run before the change:

- unchanged in every well: the labels other than the geometry, depth, water depth and $L$, salinity, every pressure and temperature, the separator, the fluid ($R_{si}$, $p_\text{sat}$, API, gas gravity), the reservoir state, permeability, net pay, skin, drawdown, the inflow's label, the choke's $\beta$, $C_D$ and profile, the lift label and its draws, the heat capacities and `configurations.check`;
- changed, all downstream of the survey: the survey's parameters, MD, the inclinations, the productivity's factor and with it $J$, $k_l$, $w_{l,\max}$, the design rate, the tubing (164 wells), $K_c$, $h$, the valve's MD and $w_{lg,\max}$; and the roughness of 42 wells, because the completion block draws one more number when the design rate fits the tubing. That branch is R3's; R10 should draw every number of a block whatever its branch.

The draws: 23.2% / 25.7% / 25.7% / 25.5% vertical / slightly deviated / deviated / highly deviated; MD/TVD of the deviated classes 1.004 / 1.094 / 1.758 and their largest inclination 7° / 36° / 72° (P5 / P50 / P95); no cell above 75°; no well falls back to vertical (R3's placeholder shares had 3 of 2000).

## Summary by acceptance criterion

1. **Implausible wells.** Every drawn well keeps every bound of the draft and passes `configurations.check(wp, 'v2.0.0')`: no well was dropped as `invalid_input` in any run. Initial gradients and reservoir temperatures by fluid class match BOEM's medians by wells within 0.012 bar/m and 11 K [own analysis of BOEM's sands atlas]. The draft's rates were the implausible part: 35% of the core offshore oil wells made more than 1018 Sm³/d of oil equivalent at their median sample, against 5.5% of the Gulf of Mexico's oil wells. D-100 lowers the permeability: 23% now, still four times EIA's share (Realism). Gas wells make 0.9 million Sm³/d at the median, far above the Gulf's gas wells, in the interim mapping.
2. **Wells that would not be produced.** The producibility filter drops 10.2% of the drawn `sol-2` wells (5.5% of core wells, 30% of tail wells) and 5.5% of the drawn `nsol-2` wells. Before D-101 it also dropped 9 core oil wells (2.4%) that an operator would lift, because R3's static check left them natural.
3. **Missing real conditions.** The pilots reach depleted reservoirs (44.5% of `sol-2` wells below hydrostatic), separators below 10 bar (16.5% of samples), overpressure (26% of the accepted wells, core and tail, above 0.6 psi/ft at their initial pressure), low rates (oil P5 of 2.1 Sm³/d in core wells), water cuts to 0.93 (P95), all five regimes at the bottom and the top, choked and unchoked flow in most wells, and lift-gas rates from off to past the optimum. Thin: churn flow (1.5% of samples at the bottom, 2.0% at the top), dispersed bubbly flow at the top (0.5%), and wells near the economic floor (16% of core offshore oil wells below 16 Sm³/d against EIA's 52%).
4. **Solver-shaped distributions.** In the R6 `sol-2` pilot no attempt of an accepted well was redrawn for a solver failure; in `nsol-2` one of 20,808 was. The largest shift of an input's distribution by solver-failure redraws is a Kolmogorov–Smirnov $D$ of 0.0000 (`sol-2`) and 0.0001 (`nsol-2`). The redraws for no operating point shift the lift gas by $D$ = 0.054 (`sol-2`) and 0.034 (`nsol-2`), the choke by 0.005 and 0.003, and the rest by at most 0.013: physics, logged. The well drops at the limits shift no well input by more than $D$ = 0.010; the producibility filter shifts them by up to 0.07 (and the lift capacity by 0.16), on purpose. Under D-53 as drafted, 4 `sol-2` tail wells were dropped at the solver-failure limit, 3 of them only for scans with non-finite points, no root and no rejected sign change, where CasADi found no root either; D-102 counts these as no operating point.

## Summary by goal

| Goal | `sol-2` (R6) | `nsol-2` (R6) |
|---|---|---|
| 1. Depleted reservoirs, $p_r$ below hydrostatic | 44.5% of wells (core 41.5%), 28.0% below 0.9 of it; 50.0% of drawn wells | 56.4% of samples; 69% of wells reach it |
| 2. Separator pressures below 10 bar | 16.5% of samples; 19.5% of wells have one | 16.2% of samples; 36% of wells |
| 3. Overpressured, high-rate wells | 26.0% of the accepted wells, core and tail, above 0.6 psi/ft at their initial pressure (core alone 22.5%, Realism); 38.5% of these in the top quartile of rates; 11.5% of all accepted wells both overpressured and above 1000 Sm³/d | 30.0% of the accepted wells, core and tail, at their initial pressure; 26.7%; 9.0% |
| 4. Modest weight at a fully open choke | 5.1% of samples fully open (5% drawn), 4.6% below 0.1 | 5.1%, 5.0% |
| 5. Lift gas scaled to the well, also in gassier wells | 22.2% of wells lifted; Spearman 0.81 between $w_{lg,\max}$ and the well's median liquid rate; accepted oil wells, core and tail, lifted by quartile of their nominal GOR 48 / 30 / 25 / 21% | 38% lifted at some time in the life; Spearman 0.88 |
| Labelled tail of 10% to 20% | 14.5% of accepted wells (9.25% before D-103) | 19% (13% before) |
| Geometry as ruled | 27.0 / 24.3 / 23.3 / 25.5% vertical / slightly / deviated / highly deviated among accepted wells | 21 / 28 / 24 / 27% |

Goal 5: lift gas reaches the gassier wells, but less often: the static check lifts heavy, low-GOR columns, so the top GOR quartile of oil wells is lifted half as often as the bottom one. Whether that meets "gas lift in gassier wells too" is Bjarne's call (For Bjarne's ruling).

## Outcomes

Wells by outcome, `sol-2`, R6 (`report`, seed 2026):

| | Drawn | Kept | Not producible | Redraw limit | Solver-failure limit |
|---|---|---|---|---|---|
| black oil | 218 | 85.3% | 29 (13.3%) | 2 | 1 |
| volatile oil | 71 | 84.5% | 10 (14.1%) | 1 | 0 |
| gas condensate | 73 | 93.2% | 5 (6.8%) | 0 | 0 |
| wet gas | 80 | 97.5% | 2 (2.5%) | 0 | 0 |
| dry gas | 8 | 100% | 0 | 0 | 0 |
| platform | 223 | 93.3% | 12 (5.4%) | 2 | 1 |
| subsea | 158 | 83.5% | 26 (16.5%) | 0 | 0 |
| onshore | 69 | 87.0% | 8 (11.6%) | 1 | 0 |
| vertical | 121 | 89.3% | 12 | 0 | 1 |
| slightly deviated | 114 | 85.1% | 15 | 2 | 0 |
| deviated | 103 | 90.3% | 10 | 0 | 0 |
| highly deviated | 112 | 91.1% | 9 | 1 | 0 |
| core | 363 | 94.2% | 20 (5.5%) | 1 | 0 |
| tail | 87 | 66.7% | 26 (29.9%) | 2 | 1 |
| all | 450 | 88.9% | 46 (10.2%) | 3 (0.7%) | 1 (0.2%) |

No well was dropped as `invalid_input`, `best_rate_solver_failure` or `slot_limit`. The not-producible wells: 38 without an operating point at their best rate and 8 below half the floor. D-101 lifted 22 natural oil wells that could not flow at their best rate: 8 of them were kept (6 core), 13 failed with lift too, and 1 reached the redraw limit. The draft's run kept 90.5% (core 95.5%, tail 59.7%): fewer wells were weak before D-100 lowered the productivity.

`nsol-2`, R6 (seed 2027): 109 drawn, 100 kept (91.7%; core 94.2%, tail 82.6%); 6 not producible at the start, 1 `short_life`, 1 `redraw_limit`, 1 `dead_date`. The draft's run: 106 drawn, 100 kept; 2 not producible, 2 at the redraw limit, 1 at the solver-failure limit (well 62, below), 1 `short_life`.

**The limits' targets.** D-81 sets at most 1% of a class's wells dropped at the limits, and D-83 at most 0.5% of the `nsol-2` wells as `dead_date`. Measured: black oils 1.4% and volatile oils 1.4% in `sol-2` (3 of 218 and 1 of 71), every other class 0%; in `nsol-2` one `dead_date` of 109 wells (0.9%) and one redraw-limit well of 46 black oils (2.2%). The samples are small; the counts are above the targets in the oil classes. The wells at the redraw limit are lifted wells that flow only at high lift-gas rates, or late natural oil wells at their floor.

## Redraws

Attempts of the accepted wells and the redraws per sample (`report`):

| | `sol-2` draft | `sol-2` R6 | `nsol-2` draft | `nsol-2` R6 |
|---|---|---|---|---|
| Attempts for 80,000 / 20,000 samples | 82,373 | 82,867 | 21,394 | 20,808 |
| Redraws per sample | 0.030 | 0.036 | 0.070 | 0.040 |
| `no_operating_point` | 2365 | 2867 | 1393 | 807 |
| `solver_failure` (after the retry) | 8 | 0 | 1 | 1 |
| `recovered` by the retry | 0 | 0 | 2 | 0 |

Redraws per sample in the R6 `sol-2` pilot by group: lifted wells 0.118, natural wells 0.012; black oil 0.059, volatile oil 0.022, wet gas 0.021, condensate 0.006, dry gas 0.007; platform 0.042, subsea 0.031, onshore 0.026; vertical 0.054, highly deviated 0.050, slightly deviated 0.028, deviated 0.008; core and tail both 0.036. Per well: 85.2% of the wells need no redraw, P90 10 and P99 146 redraws, at most 377; per slot at most 11 attempts (P99 2). At the 1st percentile a kept well flows in 58% of its draws.

Attempts without an operating point by the attempt's inputs (`sol-2`, R6; all slot attempts, including those of wells dropped at a limit; `sampler_prototype_report.py report`, figure `sol2_redraws_by_bin.png`):

| Choke $u$ | 0.05–0.15 | 0.15–0.24 | 0.24–0.34 | 0.34–0.43 | 0.43–0.53 | 0.53–0.62 | 0.62–0.72 | 0.72–0.81 | 0.81–0.91 | 0.91–1 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no operating point | 10.0% | 6.5% | 6.2% | 5.4% | 5.1% | 4.9% | 4.8% | 4.6% | 4.8% | 4.4% | 5.1% |

| $w_{lg}/w_{lg,\max}$ (lifted wells) | 0 | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no operating point | 37.9% | 33.0% | 25.8% | 19.1% | 15.7% | 11.9% | 9.1% | 8.3% | 6.4% | 5.1% | 4.3% |

| $p_r/p_i$ decile | 0.15–0.43 | 0.43–0.64 | 0.64–0.79 | 0.79–0.85 | 0.85–0.89 | 0.89–0.93 | 0.93–0.96 | 0.96–0.98 | 0.98–0.99 | 0.99–1.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| no operating point | 20.0% | 13.2% | 2.2% | 10.2% | 1.8% | 4.3% | 2.3% | 0.9% | 1.0% | 0.4% |

Solver failures stay at or below 0.6% in every bin (0.0% in most). In the draft's run they reached 0.8% at the smallest openings and 1.7% in the lowest decile of $p_r/p_i$: the incomplete scans that D-102 reclassifies. A well that needs lift dies with its lift gas off or low, so most of its no-operating-point attempts lie there; that is R4's expectation (D-38) and the redraw log keeps them.

`nsol-2`, R6: redraws per sample 0.040 in the operating samples and 0.041 in the tests; a redrawn choke lies 0.12 from its planned position at the median, 0.51 at P90.

## Criterion 4: how far the redraws shift the inputs

The inputs over all draws of the accepted wells (the samples with their redraws) against over the accepted samples: the largest gap between their CDFs (Kolmogorov–Smirnov $D$), and the gap left when only the `no_operating_point` or only the `solver_failure` attempts are removed (`report`, figures `sol2_bias_cdf.png`, `nsol2_bias_cdf.png`):

| Input | `sol-2` all vs accepted | of which no operating point | of which solver failure | `nsol-2` all vs accepted | of which solver failure |
|---|---|---|---|---|---|
| CHK | 0.0049 | 0.0049 | 0.0000 | 0.0032 | 0.0000 |
| WGL / $w_{lg,\max}$ (lifted wells) | 0.0536 | 0.0536 | 0.0000 | 0.0336 | 0.0001 |
| PDC / $p_{s,\text{nom}}$ | 0.0018 | 0.0018 | 0.0000 | 0.0020 | 0.0000 |
| PRES / $p_{r,\text{nom}}$ (`nsol-2`: / $p_i$) | 0.0023 | 0.0023 | 0.0000 | 0.0129 | 0.0000 |
| GOR / its nominal | 0.0010 | 0.0010 | 0.0000 | – | – |
| WLR / its nominal | 0.0021 | 0.0021 | 0.0000 | – | – |

The mean lift-gas fraction rises from 0.444 over all draws to 0.479 over the samples; by class the lift gas's $D$ is 0.072 for black oils. The draft's run gave the same picture ($D$ of 0.0051 for the choke and 0.0461 for the lift gas, solver-failure shifts at most 0.0001).

The wells: all drawn wells against the accepted ones, and the shift that each kind of drop alone causes (`sol-2`, R6):

| Well input | Drawn vs accepted | Without the not-producible | Without the redraw limit | Without the solver-failure limit |
|---|---|---|---|---|
| $z$ | 0.052 | 0.048 | 0.004 | 0.002 |
| $p_i$ | 0.048 | 0.048 | 0.003 | 0.002 |
| $T_r$ | 0.038 | 0.038 | 0.004 | 0.002 |
| permeability | 0.029 | 0.031 | 0.006 | 0.002 |
| design rate | 0.060 | 0.060 | 0.003 | 0.001 |
| $p_{r,\text{nom}}$ | 0.072 | 0.065 | 0.006 | 0.002 |
| $w_{lg,\max}$ | 0.149 | 0.162 | 0.010 | 0.006 |

Among the labels, the tail's share changes most, by 4.8 points (20% drawn, 14.5% accepted), then the inflow's by 2.7 and the well type's by 2.4. The producibility filter removes deep, low-pressure, low-rate and weakly lifted wells, as criterion 2 asks; the limits move nothing by more than 0.010.

## The CasADi retry (D-54)

| | `sol-2` draft | `nsol-2` draft | `sol-2` R6 | `nsol-2` R6 |
|---|---|---|---|---|
| Retries | 52 | 41 | 11 | 12 |
| Recovered | 0 | 2 | 0 | 0 |
| Retry time | 353 s (2.0% of the attempts' time) | 245 s (4.8%) | 29 s (0.2%) | 72 s (1.6%) |
| Median retry | 4.4 s | 4.4 s | 2.4 s | 3.6 s |
| Rust reason of the retried | incomplete scan 52 | incomplete scan 28, unstable only 13 | incomplete scan 11 | incomplete scan 8, unstable only 4 |

The Rust solve takes 0.16 s at the median, so a retry costs 15 to 28 Rust solves. Over the four runs the retry ran 116 times and recovered 2 attempts, both in the draft's `nsol-2` run, both scans with non-finite points and no rejected sign change, which D-102 now counts as no operating point. Under R6's rules it recovered none of 23. Its measured gain is therefore 2 attempts of about 222,000, at about 700 s of CPU: no measurable gain on the pilots. It is a fallback under the constitution's principle 7, so Bjarne decides whether it stays; R6 recommends dropping it from the generator and keeping the solver failures as redraws with their reason in the log, where they stay countable.

The draft's `nsol-2` well 62, a tail black oil late in life, is a solver finding: its searches returned one unstable root and no stable one at 15 dates, on both backends, with rejected sign changes in 64 of its scans. A continuous $R(p_0)$ cannot do that (D-53), so either both backends miss a stable root or $R$ is not continuous there. It was dropped at the solver-failure limit. R6 did not investigate it further.

## Coverage

Flow regimes at the bottom and the top, shares of samples (`sol-2`, R6; figure `sol2_regimes.png`):

| Bottom | bubbly | dispersed bubbly | slug | churn | annular |
|---|---|---|---|---|---|
| black oil | 57.4 | 14.1 | 20.4 | 0.0 | 8.1 |
| volatile oil | 16.3 | 28.6 | 27.6 | 5.0 | 22.5 |
| gas condensate | 0.6 | 1.3 | 3.7 | 4.2 | 90.2 |
| wet and dry gas | 1.1 | 0.0 | 0.3 | 0.0 | 98.6 |
| core | 31.8 | 11.1 | 15.5 | 1.1 | 40.5 |
| tail | 16.0 | 10.8 | 7.3 | 3.4 | 62.6 |
| all | 29.5 | 11.1 | 14.3 | 1.5 | 43.7 |

| Top | bubbly | dispersed bubbly | slug | churn | annular |
|---|---|---|---|---|---|
| black oil | 9.6 | 0.5 | 44.3 | 2.7 | 43.0 |
| volatile oil | 1.9 | 1.3 | 23.5 | 4.4 | 68.9 |
| gas condensate | 0.2 | 0.2 | 1.8 | 0.5 | 97.3 |
| wet and dry gas | 0.0 | 0.0 | 1.4 | 0.0 | 98.6 |
| core | 4.8 | 0.5 | 26.3 | 2.2 | 66.2 |
| tail | 4.5 | 0.2 | 15.4 | 0.6 | 79.3 |
| all | 4.8 | 0.5 | 24.7 | 2.0 | 68.1 |

`nsol-2` (R6): bottom 22.1 / 6.3 / 12.7 / 1.9 / 57.0% and top 1.9 / 0.8 / 23.0 / 0.7 / 73.6%, in the same order. The tail adds churn flow at the bottom (3.4% against 1.1% in the core) and annular flow; it does not add dispersed bubbly flow at the top.

**Choked flow.** 34.8% of the `sol-2` samples are choked (draft: 42.3%), 34.3% of core and 38.2% of tail samples; per well P10 / P50 / P90 0 / 0.22 / 1. 24.2% of the wells are never choked, 14.2% always, and 18.0% in more than 80% of their samples, which SMP-28's filter would have dropped. `nsol-2`: 36.7%.

**Gas lift.** 22.2% of the `sol-2` wells are lifted (black oils 33.9%, volatile oils 21.7%, wet gas 9.0%, condensates 7.4%, dry gas 12.5%); 20.5% of the samples carry lift gas. The injected gas–liquid ratio of the samples with lift gas is 2.6 / 78 / 988 Sm³/Sm³ (P5 / P50 / P95).

**Ranges** of the samples, core wells and tail wells, P5 / P50 / P95 and the extremes (`sol-2`, R6):

| Quantity | Core min | Core P5 / P50 / P95 | Core max | Tail P5 / P50 / P95 | Tail max |
|---|---|---|---|---|---|
| QOIL (Sm³/d) | 0.0004 | 2.1 / 112 / 2375 | 9127 | 0.23 / 63 / 3042 | 6947 |
| QGAS (Sm³/d) | 0.04 | 570 / 128,000 / 2.67·10⁶ | 1.01·10⁷ | 631 / 489,000 / 7.1·10⁶ | 1.23·10⁷ |
| QLIQ (Sm³/d) | 0.002 | 4.0 / 169 / 3315 | 9774 | 1.9 / 220 / 4744 | 9984 |
| QWAT (Sm³/d) | 0 | 0 / 20 / 1036 | 3882 | 0.70 / 34 / 3424 | 6247 |
| WTOT (kg/s) | 0.0002 | 0.098 / 4.3 / 53.7 | 136 | 0.11 / 13.6 / 118 | 172 |
| PBH (bar) | 11.6 | 50.5 / 196 / 416 | 731 | 20.6 / 171 / 492 | 679 |
| PWH (bar) | 1.5 | 12.2 / 47.0 / 178 | 565 | 6.2 / 55.3 / 182 | 360 |
| PDC (bar) | 1.5 | 5.7 / 28.4 / 74.6 | 123 | 1.8 / 24.0 / 113 | 132 |
| PWH − PDC (bar) | 0 | 0.003 / 9.1 / 141 | 510 | 0.003 / 10.4 / 142 | 351 |
| TBH (K) | 302 | 321 / 359 / 424 | 449 | 297 / 356 / 444 | 447 |
| TWH (K) | 270 | 283 / 324 / 390 | 427 | 274 / 305 / 406 | 432 |

The tail widens the gas rates, the water rates and the low end of PBH and PDC; it adds little at the low end of the oil rates, which the core already reaches.

**QTOT that the choke barely moves.** 5.0% of the `sol-2` wells (20, all natural) have a coefficient of variation of QTOT below 0.05, against 2.2% in the draft's run; `nsol-2` 3.0%. This is D-51's check on R3's beans (D-27). The choke holds back little of the rate: the median sample makes 95% of the well's best rate at the median well (93% in the draft's run), and samples below $u$ = 0.3 make 70% of the rate of samples above 0.9. Lowering the productivity (D-100) made the beans relatively larger. For Bjarne's ruling below.

## Realism

**Oil-well rates against EIA's brackets.** Offshore oil wells (platform and subsea, black and volatile oils with a nominal GOR up to 1068 Sm³/Sm³, EIA's oil wells) by their median sample's oil equivalent, against the Gulf of Mexico's oil wells in 2024 (EIA 2025, Table B32 [WP §5]; figures `sol2_eia_brackets.png` in both folders):

| Sm³/d of oil equivalent | 0–1.6 | 1.6–2.4 | 2.4–16 | 16–64 | 64–254 | 254–1018 | 1018–2035 | above 2035 |
|---|---|---|---|---|---|---|---|---|
| EIA 2024, 1869 wells | 9.8% | 4.8% | 37.8% | 16.4% | 13.9% | 11.8% | 4.2% | 1.3% |
| draft, core, 208 wells | 0.0 | 0.5 | 4.8 | 12.5 | 19.2 | 28.4 | 10.1 | 24.5 |
| R6, core, 189 wells | 2.6 | 0.0 | 13.8 | 13.8 | 26.5 | 20.6 | 11.1 | 11.6 |
| R6, core, best rate | 2.1 | 0.5 | 13.2 | 13.8 | 24.9 | 21.2 | 10.6 | 13.8 |
| R6 `nsol-2`, core, 42 wells | 0.0 | 0.0 | 7.1 | 23.8 | 26.2 | 28.6 | 4.8 | 9.5 |

R6's core offshore oil wells: 8.9 / 197 / 2321 Sm³/d (P10 / P50 / P90), mean 822 Sm³/d, 31% above 509 Sm³/d; the draft's: median 457, mean 1538, 47% above 509. EIA: mean 167 Sm³/d and 12% above 509 Sm³/d. EIA's numbers are calendar-day averages of a mature shelf, and its lowest brackets hold wells shut in for part of the year, which no operating point represents; the share above 1018 Sm³/d (R6 23%, EIA 5.5%) is the fairer comparison, and it still fails by a factor of four. R3's design liquid rate, which R3 flagged at a median of 451 Sm³/d, is now 139 Sm³/d (draws of 2000 wells, seed 2026).

**Gas wells.** The core gas and condensate wells' median-sample gas rate is 0.14 / 0.90 / 3.1 million Sm³/d (P10 / P50 / P90; draft 0.27 / 1.8 / 6.6), and 83% lie above 270,000 Sm³/d, against the Gulf's gas wells' mean of 32,700 Sm³/d and 2.3% above 270,000 [WP §5]. The interim productivity index carries these numbers (R5 measures it within a factor 0.57 to 2.05 of the gas-rate inflow, D-69), and R5's D-62 would raise the rates further. The Gulf's gas wells are mostly old shelf wells; no public per-well rate distribution of offshore gas wells elsewhere is in hand [WP §5, Gaps].

**Pressures and temperatures against BOEM.** Accepted core wells of the R6 `sol-2` pilot, medians, to compare with BOEM's medians by wells and class [own analysis of BOEM's sands atlas]; each lies within 0.012 bar/m and 11 K of BOEM's:

| Class | Wells | Initial gradient below sea level (bar/m) | $T_r$ (K) |
|---|---|---|---|
| black oil | 177 | 0.117 | 358 |
| volatile oil | 46 | 0.119 | 359 |
| gas condensate | 53 | 0.127 | 382 |
| wet gas | 64 | 0.116 | 351 |
| dry gas | 2 | 0.104 | 310 |

22.5% of the accepted core wells start above 0.6 psi/ft, by their initial pressure over the TVD below sea level (21.2% of the drawn core wells; BOEM's share by wells is in [own analysis of BOEM's sands atlas]). The filters do not move them. Over all accepted wells, tail included, the share is 26.0% (Summary by goal, goal 3): the tail draws its overpressure ratio flat, and 27 of its 58 accepted wells lie above 0.6 psi/ft, against 77 of 342 core wells.

**Validity ranges.** 2000 samples of the R6 `sol-2` pilot re-solved with the Rust core, each correlation's inputs along the operating point against its published range, as `validity_ranges.py` measures them (`validity`, seed 7); share of samples with a grid point or an input outside:

| Correlation (`validity.md`) | Variable | Range | P5 / P50 / P95 | Samples outside |
|---|---|---|---|---|
| Beggs and Robinson | pressure (psia) | to 5265 | 364 / 1653 / 4659 | 10.6% |
| Beggs and Robinson | temperature (°F) | 70 to 295 | 74 / 161 / 293 | 26% |
| Beggs and Robinson | $R_s$ (Sm³/Sm³) | 3.6 to 369 | 14 / 81 / 302 | 8.5% |
| Whitson and Brulé, condensates and leaner | GOR (scf/STB) | to 3000 | 648 / 3622 / 714,700 | 52.8% |
| Valkó and McCain | GOR | 1.8 to 395 Sm³/Sm³ | 115 / 645 / 127,300 | 60.8% |
| Abdul-Majeed, dead oil | temperature (°C) | 15.6 to 54.4 | 23 / 71 / 145 | 93% |
| Abdul-Majeed, live oil | pressure (bar) | 13.8 to 207 | 25 / 114 / 321 | 49.2% |
| Dranchuk and Abou-Kassem | $p_{pr}$ | 0.2 to 30 | 0.55 / 2.5 / 7.0 | 3.6% |
| Lee, Gonzalez and Eakin | temperature (°F) | 100 to 340 | 74 / 161 / 293 | 38.6% |
| Chen | Reynolds number | 4000 to $4\cdot10^8$ | 2127 / 393,000 / 5.7·10⁶ | 12.8% |
| Hasan et al., Table 2 | GLR (scf/STB) | 104 to 5615 | 187 / 2442 / 181,200 | 42.8% |
| Hasan et al., Table 2 | liquid rate (STB/d) | 100 to 27,760 | 22 / 1114 / 23,140 | 18.8% |
| Hasan et al., Table 2 | water cut | 0 to 0.98 | 0 / 0.25 / 0.93 | 2.0% |
| Hasan et al., Table 2 | MD (m) | 498 to 9342 | 1713 / 3189 / 6667 | 0% |
| Cells | inclination | at most 87° | – | 0% |

API, gas gravity, $T_r$ and the relative roughness stay inside every range. Against v1's sampler (`validity.md`, Where the sampler leaves the ranges): GORs above 534 Sm³/Sm³ fall from 69% of the cases to 53% of the samples, and they are now the gas and condensate classes, not oils; Hasan et al.'s GLR is left by 43% against 42%; no cell is horizontal, against 24%. The water-cut exits are gas wells, whose water–liquid ratio is not capped.

**Next to `sol-1`.** The R6 `sol-2` samples against the published `sol-1` (1,000,000 samples of 2000 wells, read from its zip by its exact path; figure `sol2_features_vs_sol1.png`): P5 / P50 / P95 and the KS $D$:

| Feature | Prototype `sol-2` | `sol-1` | $D$ |
|---|---|---|---|
| QOIL (Sm³/d) | 1.1 / 101 / 2552 | 15.7 / 617 / 2447 | 0.42 |
| QGAS (Sm³/d) | 572 / 155,900 / 3.63·10⁶ | 14,820 / 404,900 / 1.98·10⁶ | 0.24 |
| QLIQ (Sm³/d) | 3.4 / 175 / 3695 | 52.7 / 974 / 3003 | 0.47 |
| QWAT (Sm³/d) | 0 / 21 / 1539 | 1.2 / 164 / 1394 | 0.34 |
| WTOT (kg/s) | 0.098 / 4.8 / 66.3 | 1.5 / 16.8 / 41.9 | 0.44 |
| PBH (bar) | 43.9 / 194 / 429 | 147 / 260 / 391 | 0.28 |
| PWH (bar) | 11.0 / 47.9 / 179 | 22.9 / 61.8 / 152 | 0.17 |
| PDC (bar) | 3.7 / 27.9 / 79.1 | 13.8 / 34.9 / 89.3 | 0.18 |
| TBH (K) | 316 / 358 / 428 | 338 / 379 / 418 | 0.26 |
| TWH (K) | 283 / 322 / 393 | 291 / 347 / 384 | 0.39 |
| CHK | 0.10 / 0.55 / 1 | 0.12 / 0.54 / 0.95 | 0.05 |
| FGAS | 0.029 / 0.29 / 0.97 | 0.031 / 0.26 / 0.81 | 0.20 |
| FWAT | 0 / 0.086 / 0.82 | 0.001 / 0.14 / 0.70 | 0.13 |

Choked: 34.8% against 34.0%; samples with lift gas 20.5% against 17.6%. Regimes at the bottom, `sol-1`'s three against the five: `sol-1` bubbly 33.3%, slug-churn 33.7%, annular 33.1%; the prototype bubbly 29.5%, dispersed bubbly 11.1%, slug 14.3%, churn 1.5%, annular 43.7%. The prototype spans about two more decades of rate at the low end, reaches low bottomhole pressures (depleted reservoirs) and low separator pressures that `sol-1` lacks, and has no counterpart to `sol-1`'s cluster of wellhead temperatures at 275 to 285 K, which is v1's trickle roots (`plans/evidence/README.md`, `trickle_signature.py`): every prototype sample is the stable root.

## The `nsol-2` life

From the R6 run (`report`): lift starts from the start in 11 accepted wells, at the drawn time in 17, at the end of natural flow (D-49's rule 2, D-84) in 10, by the static check in none (one dropped well), and 62 stay natural; a later start lies at 0.11 / 0.55 / 0.76 of the life span. The floor ends 8% of the lives early ($T_\text{end}/T_\text{life}$ at P5 0.40). The oil equivalent over the life, relative to the first sample, falls to 0.93, 0.69, 0.55, 0.42 and 0.35 at the medians of the five fifths of the life: a decline like Arps's (figure `nsol2_histories.png`, six wells with their tests). By the per-sample stage, core wells' $p_r/p_i$ medians are 0.92 / 0.65 / 0.48 (initial, mid-life, late) under depletion and 0.98 / 0.93 / 0.89 under pressure support, against R3's `sol-2` states of 0.99 / 0.71 / 0.38 under depletion (R4's sanity check).

## Cost

Measured on the pilots, each attempt on one process (`report`, Cost):

| | `sol-2` draft | `sol-2` R6 | `nsol-2` draft | `nsol-2` R6 |
|---|---|---|---|---|
| Rust solve, median / mean / P95 (s) | 0.159 / 0.203 / 0.470 | 0.159 / 0.189 / 0.415 | 0.162 / 0.189 / 0.384 | 0.162 / 0.184 / 0.352 |
| Solves per accepted well | 214 | 215 | 259 | 247 |
| CPU per accepted well (s), all drawn wells' CPU | 44.4 | 40.7 | 51.3 | 46.1 |
| Time outside the core per solve | 0.18 ms | 0.16 ms | 0.19 ms | 0.17 ms |
| Share of CPU outside the core and the retries | 0.1% | 0.1% | 0.1% | 0.1% |
| Pool efficiency (busy time over processes × wall) | 0.91 | 0.97 | 0.80 | 0.89 |
| Projected CPU-hours, 10,000 wells | 123 | 113 | 143 | 128 |
| Projected wall time on 12 processes | 11.3 h | 9.7 h | 14.9 h | 12.0 h |

The projections take the pilot's CPU per accepted well, which includes the dropped wells' solves, the retries and the life scans, and divide by 12 processes at the measured pool efficiency. Both R6 datasets together: about 240 CPU-hours, 22 hours on 12 processes; within the 85 to 120 CPU-hours per dataset that R7 estimated (`datasets-draft.md`, Cost). The solve's mean is 1.2 times its median; gas classes and volatile oils are the slowest (0.21 to 0.27 s on average). A process with the CasADi backend loaded used 404 MB resident in a single-process check, and every pool process ran within its 2.6 GB cap, so 12 processes fit in 32 GB.

**The batch API (backlog §4.3).** Building a well, the root set's Python side and the row take 0.16 to 0.19 ms per solve, 0.1% of the CPU; drawing the wells and samples takes another 0.03%. A batch API could at most remove that, so its wall-time gain on the generation is about 0.1%, far below the 20% that D-93 set. The process pool is enough.

**File sizes.** The R6 `sol-2` samples table is 16.9 MB for 80,000 rows of 30 columns (float64), so about 420 MB for 2 million rows; `nsol-2`'s is 4.4 MB for 20,000 rows of 39 columns, about 440 MB. The wells tables are 0.28 MB and 0.16 MB per 400 and 100 wells (about 7 and 16 MB at 10,000 wells), the geometry 13,171 stations for 400 wells, and the redraw log 5208 rows for 400 `sol-2` wells (about 130,000 rows at 10,000).

## Backends and the re-solve check

**The CasADi backend** on 200 samples of the R6 `sol-2` pilot drawn at random (seed 6), each re-solved from its rows without an extra start (`resolve --backend casadi`): CasADi finds the operating point in 196, and every one agrees with the Rust core's within a relative $4\cdot10^{-8}$ in PBH, $3\cdot10^{-9}$ in PWH, $3\cdot10^{-8}$ in TWH and $8\cdot10^{-8}$ in WTOT, with the same regimes at both ends and the same choked state. In the other 4 (two highly deviated wet-gas wells at bottomhole pressures of 14 to 41 bar) CasADi finds only the unstable root and no operating point, where the core finds a stable root below it: CasADi's multi-start misses the low-pressure root of these gas wells. CasADi takes 3.7 s per sample at the median, 23 times the core.

**The re-solve check (D-92).** 1000 samples of the R6 `sol-2` pilot and 500 of `nsol-2`, drawn at random (seed 6), rebuilt from the `wells`, `samples` and `geometry` rows as `datasets-draft.md` (Section 3) builds them and solved again with the Rust core: every feature identical, in all 1500 (also 300 and 200 of the draft's runs). The tables carry every input a re-solve needs, the interim gas wells' $k_l$ included.

## Changes to the draft (D-100 to D-103)

| Entry | Change | Why | Effect in the R6 pilots |
|---|---|---|---|
| D-100 | Permeability's median 50 → 15 mD offshore, 10 → 3 mD onshore | Rates far above EIA's brackets (criterion 1, D-24's check) | Core offshore oil wells above 1018 Sm³/d: 35% → 23%; median 457 → 197 Sm³/d; gas wells' median 1.8 → 0.9 million Sm³/d; more weak wells dropped (core 3.7% → 5.5%) |
| D-101 | `sol-2`: a natural oil well that cannot flow at its best rate is lifted | 2.4% of core wells dropped that an operator would lift (criterion 2); as `nsol-2` does | 22 wells lifted, 8 kept |
| D-102 | No root and no rejected sign change is no operating point, whatever the non-finite scan points | D-53 counted 211 such attempts as solver failures; CasADi found no root in 70 of the 72 it retried; 4 tail wells dropped at the solver-failure limit (criterion 4) | Solver failures of accepted wells 8 → 0 (`sol-2`); retries 52 → 11 |
| D-103 | The tail's drawn share 0.15 → 0.20 | The accepted tail was 9.25%, below the Goals' 10% | 14.5% (`sol-2`), 19% (`nsol-2`) |

Each is an entry in `decisions.md`, with its alternatives, and `sampling-draft.md` carries it in the section it changes. The R6 pilots rerun everything the changes touch; the draft's pilots stay reproducible with `--revision draft`.

## For Bjarne's ruling

- **D-100 to D-103**, each a working assumption.
- **The rates' target.** After D-100, 23% of the core offshore oil wells still lie above 1018 Sm³/d against EIA's 5.5%, and gas wells make 0.9 million Sm³/d at the median. How close the population should come to the Gulf's 2024 brackets, a mature shelf in calendar-day averages, against a global offshore population with Norwegian wells of 10,000 bbl/d, is his call. A permeability median of 10 mD would give about 12%. The gas rates wait for R5's inflow.
- **The beans (D-27).** The median sample makes 95% of its well's best rate, and 5.0% of the wells vary their QTOT by under 5%. D-51 calls that the sign of beans that are too large. R6 recommends a smaller bean, for example $\beta \sim U(0.15, 0.45)$ in place of $U(0.25, 0.6)$, closer to the textbook beans of 24/64 and 32/64 in in 2-7/8 to 3-1/2 in tubing ($\beta$ of 0.13 to 0.2), measured in the rework; it is not changed here, because the choke's sources are thin and the change moves every rate again.
- **The CasADi retry (D-54).** No measurable gain: 2 attempts recovered of 116 retries over the four runs, none under R6's rules, at 0.2% to 4.8% of the CPU. R6 recommends dropping it (principle 7).
- **The limits' targets (D-81, D-83).** Black and volatile oils each lose 1.4% of their `sol-2` wells at the limits (target 1%), and `nsol-2` lost one well of 109 as `dead_date` (target 0.5%). The pilots are small; the full generation will show whether the targets or the limits need to move.
- **Gas lift and the GOR (goal 5).** The top GOR quartile of oil wells is lifted 21% of the time against 48% in the bottom quartile.
- **Thin regimes.** Churn flow is 1.5% of the samples at the bottom and 2.0% at the top, dispersed bubbly flow 0.5% at the top. The tail does not add them; a coverage device for regimes would need its own draw.
- **The tail's pressures.** The tail draws $p_{s,\text{nom}}$ and $p_r$ independently, and a third of its wells cannot flow. D-103 compensates with the share; linking the two would need the separator drawn after the reservoir state.
- **Solver findings.** `nsol-2` well 62 (draft run): root sets with only an unstable root on both backends. CasADi misses the stable root of 2% of the compared samples (wet gas at low pressure). Neither shapes the pilots' distributions measurably, but both are findings for the backlog.
- **Not done in R6:** R5's gas-rate inflow prototype on CasADi (D-69), and the comparison with today's (v1's) sampler beyond `sol-1`'s published data.

## Figures

PNG files in the session's scratchpad, `r6/figures/` (R6) and its `draft/` subfolder (the draft's runs), not in the repo:

| File | Shows |
|---|---|
| `sol2_bias_cdf.png`, `nsol2_bias_cdf.png` | CDFs of the choke, the lift gas, PDC and PRES over all draws and over the accepted samples, with the KS $D$ |
| `sol2_redraws_by_bin.png`, `nsol2_redraws_by_bin.png` | Attempts without an operating point by bin of $u$, $w_{lg}/w_{lg,\max}$ and $p_r/p_i$, split into no operating point and solver failure |
| `sol2_regimes.png`, `nsol2_regimes.png` | The five regimes at the bottom and the top |
| `sol2_features_vs_sol1.png`, `nsol2_features_vs_sol1.png` | Histograms of the rates, pressures and temperatures next to `sol-1`'s (the draft's `nsol-2` figure has no `sol-1` overlay) |
| `sol2_eia_brackets.png`, `nsol2_eia_brackets.png` | Offshore oil wells by rate bracket next to EIA's |
| `nsol2_histories.png` | Six `nsol-2` wells' oil equivalent over their life, operating samples and tests |

## Files

- The repo: `plans/evidence/sampler_prototype.py`, `plans/evidence/sampler_prototype_report.py`, the geometry and D-100, D-103 in `plans/evidence/sampler_draft_draws.py`, their rows in `plans/evidence/README.md`; this file; `decisions.md`; the changes in `sampling-draft.md`.
- The scratchpad's `r6/` folder holds each run's tables in the schema of `datasets-draft.md`, for Bjarne's private comparisons: `sol-2-r6/` and `nsol-2-r6/` (R6), `sol-2/` and `nsol-2/` (the draft), each with `wells.parquet`, `samples.parquet`, `geometry.parquet`, `pvt.parquet` (empty in the interim mapping), `redraws.parquet`, `drawn_wells.parquet`, `metadata.json` and the run log `attempts.parquet`; the reports as text (`*_report.txt`, `backends_sol2.txt`, `validity_sol2.txt`); and the raw results per well (`raw/`).
