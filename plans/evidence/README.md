# Evidence for the v2 plan

Scripts and results behind the numbers in `../manywells-v2-plan.md`: the "Known v1 defect" bullet, the "Multiple roots and stability" section, and Step 2.4. Feature specs 016 to 021 have their own sections below. The v2 plan's scripts were run on 2026-09-30 against v1.0.0 (Python 3.11.13, casadi 3.6.4, numpy 1.24.3, pandas 1.5.3), using the published `manywells-sol-1` data and configs.

## Setup

The scripts use v1.0.0's API, not `develop`'s. Run them from the root of a v1.0.0 checkout, or set `MANYWELLS_V1` to one:

```console
git worktree add --detach ../manywells-v1 v1.0.0
cd ../manywells-v1
uv sync --frozen    # --locked re-resolves, and fails, if a global uv exclude-newer is set
.venv/bin/python <repo>/plans/evidence/stability_label.py --config <data>/manywells-sol-1_config.zip 977
```

`manywells-sol-1.zip` and `manywells-sol-1_config.zip` are in `data/` of the Hugging Face dataset `solution-seeker-as/manywells`.

## Scripts and results

| Script | What it shows | Result |
|---|---|---|
| `stability_label.py 977` | The stability label from v1's residual graph: drop the choke row, treat `p_0` as a parameter, and take the sign of d(choke row)/d`p_0` from one linear solve | v1's default guess lands on the trickle root (`p_0` = 208.02 bar, TWH 285.5 K, d/d`p_0` = +285: unstable). Other guesses reach `p_0` = 169.00 bar (TWH 332.2 K, −2.43: stable). |
| `root_sets.py 40 10` → `root_sets_sol1_50.csv` | Root sets from v1 run from six starting points, on 40 wells that meet the two-root criterion and 10 that do not (seed 1) | Every two-root well has one stable and one unstable root, the unstable one at higher `p_0`; every one-root well's root is stable. v1's default solve returns the trickle root in 1 of the 40 (well 1701). Median 3.1 s per well for the six starts. |
| `choke_sweep.py 41 977` | Why a v1-only sweep of the choke row over `p_0` cannot check completeness | The row is NaN where `p_L < p_s`, which is next to the trickle root, and the cellwise march fails at high rates. The sweep finds only the stable root. |
| `trickle_signature.py` | The trickle-root signature (PWH within 10 mbar of PDC) in the published data | 3.53% of samples, from 131 of 2,000 wells, with median TWH 279.4 K. They make up 33.5k of the 44.2k samples between 275 and 285 K. |

## Solver versions matter

The same batch was first run with casadi 3.8.1, the environment on `rust_implementation`. The roots agree with the v1.0.0 run to within 1e-7 K in TWH. The one difference is well 1847, where v1's default solve failed under 3.8.1 and converges to the stable root under 3.6.4. That is why Step 2 pins the CasADi version along with `manywells==1.0.0`.

## Feature 018: PVT tables

Scripts behind the measurements in `../../specs/features/018-pvt-tables.md`, run on 2026-10-28 on `develop` at `8b4f1e6` (Python 3.11, casadi 3.8.1), with 24 CPUs shared with other work, before issue #8's in-situ mixing of oil and water. `pvt_tables.py` is the prototype: it adds a table by a subclass of `FluidModel` as it was before the spec's interface, so it runs on the CasADi backend only, and on `develop` up to `162eb47`. The bug it found, issue #10, has its own section below. `pvt_table_eos.py` writes the synthetic equation-of-state tables of `tests/data/pvt/` and measures the GOR axis; it needs NeqSim 3.4.0 and Java, not ManyWells.

| Script | What it shows | Result |
|---|---|---|
| `pvt_tables.py points` | The error of tables of develop's black oil, on four grids and at one temperature, against the black oil at random points of 40 sampled fluids; and interpolating $\rho_g$ in place of $Z$ | 10 bar by 10 K: $R_{so}$ within 0.45 Sm³/Sm³, $\rho_l$ within 0.03%, $\rho_g$ within 0.21% (2.1% interpolating $\rho_g$). Refining a coarse table by monotone cubics rounds off the bubble point's kink |
| `pvt_tables.py wells 88 OUT.csv` | The same tables on the 263 cases of `jt_wells.py`: operating points found, PBH, PWH and TWH against the black oil, and the time | See the feature spec's Measurements |
| `pvt_tables.py interp 12 OUT.csv` | Bilinear interpolation against CasADi's cubic B-spline on the 10 bar by 10 K grid | See the feature spec's design choice 1 |
| `pvt_tables.py write DIR` | Wrote `vazquez_beggs.csv` and `synthetic_eos_oil_360K.csv` in the prototype's format; `specs/tools/make_pvt_tables.py` writes them now | |
| `pvt_tables.py eos 88 TABLE OUT.csv` | The synthetic fluid's table against develop's black oil at the same stock-tank densities and GOR, on the sampled cases whose $p_r$ and $T_r$ the table covers | See the feature spec's Measurements |
| `pvt_tables.py scan TABLE EOS.csv` | The cases `eos` lost against the capped black oil: the shooting residual after a march from 80 values of $p_0$, and the table with 0.01 kg/s of lift gas | 16 of 17 have an all-liquid bottomhole with the table; 12 solve with the lift gas (the feature spec's Findings 1) |
| `pvt_table_eos.py OUT.csv` | Writes `synthetic_eos_oil.csv` (PVT-TAB-6 from NeqSim's flash) | A rerun reproduces the file byte for byte |
| `pvt_table_eos.py gor` | One table read at 0.6 to 2 times its GOR by the black-oil rule, and a GOR axis, against the recombined stream's own tables | Within 10% of the GOR: no-slip density within 1.2–1.7% at the median, 4.7% at most; a GOR axis: 0.01–0.3%, 2.2% at most |
| `pvt_table_eos.py gor_table OUT.csv` | Writes `synthetic_eos_oil_gor.csv` | |

`pvt_table_backends.py` measures the tables as implemented, on both backends (the feature spec's acceptance 9). It was run on 2026-10-28 on the branch `pvt-tables` at `1e463fd` (Python 3.14, casadi 3.8.1), each case on one process, 12 processes sharing 24 CPUs with other work.

| Script | What it shows | Result |
|---|---|---|
| `pvt_table_backends.py wells 88 OUT.csv` | The 263 cases of `jt_wells.py`, with develop's black oil and a table of it on 10 bar by 10 K, on each backend: operating points, PBH, PWH and TWH, and the time per case | Each fluid has an operating point in the same 260 cases on each backend: the table loses none. PBH within 0.013 bar of the black oil's. The table costs the Rust core nothing (1.00 times the black oil's time), and CasADi 1.76 times at the median |
| `pvt_table_backends.py eos 88 OUT.csv` | The synthetic fluid's table against develop's black oil capped at the table's bubble point, on the cases the table covers, on each backend | 148 and 141 operating points, the same cases on both backends. In the 7 the table lacks, it has no root at all (the core's shooting residual is positive across $(p_s, p_r)$). The two backends agree to $10^{-6}$ bar |

## Feature 017: calibration

Scripts behind the measurements in `../../specs/features/017-calibration.md` that the twin study (`scripts/calibration/twins.py`) does not make, run on 2026-10-02 on the branch `calibration`, with 24 CPUs. Run from the project root as `uv run python plans/evidence/calibration_measurements.py <part>`.

| Part | What it shows | Result |
|---|---|---|
| `rows` | The Rust core's time for a row (the full search) on four sampled wells at 20, 50 and 100 cells, and 16 rows in 1 to 16 threads | 32 to 810 ms per row; 16 threads are 12 times faster than one |
| `step` | The forward difference of the predicted observations in the log of each parameter, for steps from $10^{-1}$ to $10^{-8}$ | Four digits from $10^{-4}$, five from $10^{-5}$ to $10^{-8}$: the Jacobian's step of $10^{-6}$ is safe |
| `valley` | Well 3 of seed 2026, periodic tests: the fit from the medians alone, with $\lvert z\rvert \le 4$, and from the start search | From the medians it stops at a roughness of 143 m (cost 119.6); with bounds at the bounds (cost 23,446); from the start search in the truth's basin (cost 27.6, 798 solves against 2,793) |
| `backends` | The root sets of both backends at the truth, at the rows of the backend comparison's wells 7 and 1 | Well 7: at $u = 0.2$ the CasADi search has two stable roots where the core has one. Well 1: it has no root at two rows where the core has a stable one |

## Feature 016: Joule–Thomson cooling

Scripts behind the measurements in `../../specs/features/016-joule-thomson.md`, run on 2026-10-02 against `develop`. `jt_wells.py` is the prototype that chose the design: it adds the term by a subclass of `ThermalModel`, so it runs on the CasADi backend only, with Papay's gas law. `dak_jt_default.py` measures the implemented feature.

| Script | What it shows | Result |
|---|---|---|
| `jt_factor.py` | The gas's Joule–Thomson factor $J = T(\partial\ln Z/\partial T)_ p$ from Papay's z-factor and from the Dranchuk–Abou-Kassem equation of state, against CoolProp's reference equations of state for methane and two natural gases, over 5 to 460 bar and 280 to 425 K. Needs CoolProp: `uv run --no-project --with CoolProp --with numpy --with scipy python plans/evidence/jt_factor.py` | DAK follows the reference to 460 bar (within 0.07 for methane); Papay's factor has the wrong sign above about 300 bar. Papay's $Z$ at 460 bar is 11% to 29% above the reference. |
| `jt_wells.py 88 OUT.csv` | The term on 88 wells of `develop`'s sampler (seed 2026), three operating points each, with DAK's or Papay's factor, and with the cell's pressure gradient in place of $F + \rho_m g\cos\theta$ | See the feature spec's Measurements. About 20 minutes on 24 cores with one BLAS thread per process. |
| `dak_jt_default.py OUT.csv` | The implemented feature on the same cases, solved by the Rust core in four variants of `develop`'s default: Papay without the term (the old default), DAK without it, Papay with it, and DAK with it (the new default); with the core's time and work counters on one process | Each variant has an operating point in 259 of 263 cases. The term lowers TWH by 7.5 K at the median and by up to 45.9 K; the gas law alone moves PWH by up to 6.2 bar. About 5 minutes. |

## Issue #8: oil and water mixed by in-situ volume

Measurements behind the ruling on PVT-MIX-6 and PVT-MIX-8 in `../../specs/features/007-black-oil.md`, run on 2026-10-28. `liquid_density_wells.py` solves wells of `develop`'s sampler (seed 2026) at their nominal operating point and two sampled ones, with the model of the tree it runs in. It was run on `develop` at 8b4f1e6 (before) and on the branch with the change (after), and `compare` joins the two runs.

| Run | What it shows | Result |
|---|---|---|
| `run 500 OUT.csv --backend rust` | 1493 cases, Rust core | Operating point in 1457 cases before and 1458 after. The one lost case is a solver finding: well 30, sample 2, where the core brackets the sign change of R but rejects the refined root at its acceptance test, and the CasADi backend finds it (156.683 bar). The two gained cases (wells 179 and 428) have a stable and an unstable root after the change in both backends, and no root before in either. The bottomhole liquid density falls by 1.0% at the median, 2.8% at the 10th percentile and 6.4% at most ($B_o$ up to 1.75). PBH changes by less than 0.3 bar in 80% of the cases, from −5.1 to +0.6 bar. The liquid rate changes by −0.01% at the median and by up to +87%, in wells with a few bar of drawdown. |
| `run 88 OUT.csv --backend casadi` | 263 of the same cases, CasADi backend | Operating point in 255 cases before and 257 after, none lost. The density falls by 1.3% at the median and 5.4% at most; PBH changes from −3.4 to +0.6 bar. After the change the backends agree within $10^{-4}$ bar in PBH on the 256 cases both solve, and disagree on whether there is an operating point in 4 cases, against 5 before. |
| Timing, one process each, before and after side by side in the same environment | 179 cases (Rust), 48 cases (CasADi) | Rust: 8% less time in total, 0.93 times at the median case. CasADi: 0.97 times at the median case and 11% more in total, from a few cases where the multi-start search does more work (well 4, sample 1: 9.4 s to 33.2 s). |
| `scripts/verification/compare_backends.py --configurations develop --per-configuration 60` | The backends on the comparison set's `develop` wells | 59 of 60 cases pass the rule of `tests/test_backend_comparison.py` before and after; the failing case is the same (`develop#17`, two roots the core misses). |

In 117 of the 1456 cases with water that both runs solve, $R_{so}$ at the bottomhole exceeds the gas–oil ratio, so the density assumes more dissolved gas than PVT-OIL-13 carries (007, open items). There PBH falls by 0.2 bar at the median.

## Issue #16: the black oil capped at the gas–oil ratio

Measurements behind PVT-OIL-15 and the cap of PVT-OIL-13 in Sm³/Sm³ (`../../specs/features/007-black-oil.md`), run on 2026-10-29. `gor_cap_wells.py` solves wells of `develop`'s sampler (seed 2026) at their nominal operating point and two sampled ones; it was run with `develop` at `0112290` (before) and on the branch `fix/temperature-bounds-live-oil` with the change (after), each in its own environment, and `compare` joins the two runs. Both runs shared the machine with other work, so their times compare only roughly.

| Run | What it shows | Result |
|---|---|---|
| `run 500 OUT.csv --backend rust` | 1492 cases, Rust core | Operating point in 1448 cases before and 1444 after; none gained. The lost cases, (1, 0) and well 442 at all three points, have no root after the change: the core's $R(p_0)$ is positive on 4000 samples from $p_s$ to $p_r$, and the CasADi backend finds none either. In the 109 cases undersaturated at the bottomhole, the bottomhole liquid density rises by 2.4% at the median and up to 14.3%, PBH by 0.33 bar at the median and up to 10.8 bar, and the reservoir liquid rate falls by 1.3% at the median and up to 23%. In the other 1335 cases, PBH changes by at most $10^{-4}$ bar and the rate by at most 0.013%. |
| `run 88 OUT.csv --backend casadi` | 263 of the same cases, CasADi backend | Operating point in 254 cases before and 253 after: (1, 0) lost, as on the core. In the 20 undersaturated cases PBH rises by up to 10.7 bar. Joined with the core's runs, the two backends find an operating point in the same cases before and after, and agree within $2.8\cdot10^{-8}$ bar in PBH before and $1.7\cdot10^{-7}$ bar after. |

The same script measures the CasADi backend's new temperature bounds (issue #13, `../../specs/features/012-root-search.md`): `run 88 OUT.csv --backend casadi` on the branch with the bounds, joined with the run with the cap alone, finds the same 253 operating points within $6\cdot10^{-7}$ bar, in 2% more time in total. On the verifier's case set, `scripts.verification.develop_candidate --log` gives Ipopt's iterations per start for v1.0.0's bounds, no upper bound, and margins of 10 and 50 K (012).

## Issue #10: the march with an all-liquid bottomhole

Measurements behind the retry of point 0 in `../../specs/features/012-root-search.md`, run on 2026-10-28 on `develop` at `ae754d8` with the retry added (Python 3.14, casadi 3.8.1), with 24 CPUs shared with other work. `all_liquid_bottomhole.py` compares the variants of the search: the Rust core, the CasADi search with the retry, without it (`old`), and with the rejected no-slip guess at every start (`guess`). The problem was found with the prototype of feature 018 (PVT tables, a draft on the branch `pvt-tables`).

| Run | What it shows | Result |
|---|---|---|
| `repro` | Well 87 of seed 2026 at its nominal point, `gor=40` | Without the retry, no operating point, and point 0 fails from 0.5 at four of the seven starts; with it, 304.40 bar, the core's |
| `cases 88 OUT.csv` | The 263 cases of `jt_wells.py` at their GOR and at half and a quarter of it (789), every variant | Without the retry the search misses 15 of the 116 operating points with an all-liquid bottomhole; with it, none, and none lost. 9 misses with gas-rich bottomholes in every variant |
| `attempts 20 OUT.csv` | The 60 cases of the first 20 wells at their GOR, one process: each start's outcome and Ipopt iterations | The retry changes 4 of 420 starts, each from a failed march to a root, and no other: 0.16% more iterations. The `guess` variant changes the paths of others: 11% more (20% in a first measurement whose starts differed by rounding) |
| `scripts.verification.develop_candidate`, with and without the retry, twice each | The verifier's case set | PASS, 100% for both; the same 200 roots to the bit; 1.66 and 1.68 s per case against 1.68 and 1.64 s |

## Feature 019: five flow regimes

Scripts behind `../../specs/features/019-five-flow-regimes.md`, run on 2026-10-28 against `develop` at b313b9d. `five_regime_slip.py` is the prototype, a subclass of `SlipModel`, so it runs on the CasADi backend only; `five_regime_wells.py` imports it. Both need `develop` before the feature was implemented (b313b9d): the prototype's `classify` has the old signature, and `five_regime_wells.py`'s `current` variant is the default model of that time. `slip_models.py` runs on either, as it pins the fitted classifier.

| Script | What it shows | Result |
|---|---|---|
| `slip_models.py` | `develop`'s slip closure against Hasan et al. (2010) and Shi et al. (2005), point by point, at two fluid states and four inclinations, and on a 40 × 40 grid of superficial velocities | See the feature spec's Motivation. About a minute. |
| `five_regime_slip.py` | The prototype's closure on a 60 × 60 grid, with $w$ = 0.01, 0.02 and 0.05, against the crisp hierarchy, Hasan et al.'s model and `develop`'s fitted classifier; `ranges [CSV]`, the sampler's gas–liquid ratios and the solved rates against Hasan et al.'s Table 2 | With $w = 0.02$, 99.9% of the labels are the crisp hierarchy's, and $\alpha$ is 0.005 to 0.007 from Hasan et al.'s at the mean, against 0.016 to 0.035 for the fitted classifier. 38% of 2000 sampled wells have a GLR above Table 2's range. A few minutes. |
| `five_regime_wells.py 40 OUT.csv` | 39 wells of `develop`'s sampler (seed 2026), one sampled operating point each, in five variants: the fitted classifier, the proposal, and the proposal without the floor on $\lambda_g$, without the guard $\delta$, and with the cell's $f_D$ | The proposal finds an operating point in all 39 wells, the fitted classifier in 38. Against the fitted classifier PBH changes by −0.73 bar at the median and up to 14.0 bar, the rate by +2.6% and up to 19.8%. Without the floor the search takes 690 s in total against 157 s, without the guard 1107 s, with the same operating points; the cell's $f_D$ moves PBH by at most 0.013 bar. About 40 minutes on one process, on a loaded machine. |
| `five_regime_default.py OUT.csv --casadi-wells 30` | The implemented feature on the 263 cases of `dak_jt_default.py`, with the fitted classifier and the five regimes, on the Rust core, and the 90 cases of the first 30 wells on the CasADi backend; on `develop` with the feature | The core finds an operating point in 257 cases with the five regimes against 260 with the fitted classifier, the CasADi backend in 88 of 90 against 89: in each lost case the model has no root. PBH changes by −0.56 bar at the median, from −14.6 to +4.0 bar. The backends' operating points agree within $9 \cdot 10^{-9}$ bar. About 15 minutes on one process. |

## Feature 020: a scale on the profile parameter

Scripts behind `../../specs/features/020-profile-scale.md`, run on 2026-10-29 on the branch `profile-scale` (`develop` at `c12fd51` with the feature; Python 3.14, casadi 3.8.1), with 24 CPUs shared with other work. Every well is synthetic, drawn by `develop`'s sampler.

| Script | What it shows | Result |
|---|---|---|
| `profile_scale_wells.py OUT.csv --casadi-wells 30` | The cases of `five_regime_default.py` with $\kappa_0$ = 1, 0.5 and 0: 263 on the Rust core, the 90 of the first 30 wells on the CasADi backend too | No operating point lost; 2 and 4 gained on the core at $\kappa_0$ = 0.5 and 0. At $\kappa_0 = 0$, PBH changes by −0.19 bar at the median and up to −15.3 bar, the rate by +0.6% and up to +75%. In one case (well 14, its second sampled point) the CasADi search finds a second stable root at $\kappa_0 \le 0.5$ that the core misses. About 20 minutes on one process. |
| `profile_scale_twins.py OUT.json --wells 20 --detail 8 --workers 12` | The twin study's wells 0 to 19 with `profile_scale` free besides `K_c`, `w_l_max`, `roughness` and `h`, in every instrumentation; and on wells 0 to 7, data with `profile_scale` drawn calibrated with friction alone, the scale alone or both, held out at higher rates | Every fit converges. `profile_scale` is recovered where the data depend on it, and stays at its prior on the six wells where they do not. Its errors correlate with `roughness`'s by −0.22 to +0.28. 017's twin criteria hold in 89% to 100% of the fits, below 95% in `no_downhole` from well 1's false minimum and well 6. With the slip fixed, friction absorbs the holdup error and errs at higher rates (well 6: roughness −3.8 sd, held-out RMS 1.87 against 1.04). About 40 minutes on 12 threads. |

## Feature 021: the lift gas's injection depth

Scripts behind `../../specs/features/021-lift-gas-injection-depth.md`, run on 2026-10-29 on the branch `lift-gas-injection-depth` (`validity-refusals` at `8c66a60` with the feature; Python 3.14, casadi 3.8.1), with 24 CPUs shared with other work. Every well is synthetic, drawn by `develop`'s sampler.

| Script | What it shows | Result |
|---|---|---|
| `injection_depth_rows.py dump OUT MODE`, `compare A B` | Acceptance 2: both backends' rows, parameters, bounds and root sets on the backend comparison's matrix and six sampled wells, before the feature and with it, with the lift gas at the bottomhole by `injection_md=None` and by its measured depth | The same to the bit: 31,192 row values and 202 roots, in both modes. A few minutes per dump |
| `injection_depth_wells.py run 1000 OUT.csv --casadi-cases 90` | The 290 gas-lifted cases of 1000 wells (the sampled operating points of the wells with gas lift, `five_regime_default.py`'s draw), with the lift gas at the bottomhole and at a valve at 75% and 50% of the measured depth, on the core, and the first 90 on the CasADi backend | Operating points: 287, 286 and 283 on the core, 90, 90 and 89 on the CasADi backend, in the same cases. With the valve at 50% PBH rises by 1.44 bar at the median and up to 23.9 bar, and the rate falls by 5.7%; L-shaped wells gain. The core's time does not change; the CasADi backend's rises by 6 to 11% per case at the median. About 5 minutes on 12 processes |
| `injection_depth_wells.py summary OUT.csv` | The tables of the feature spec's Measurements | |
| `injection_depth_wells.py scan OUT.csv` | The core's $R(p_0)$ on 4000 samples for each case that loses its operating point with the valve | In all five (case, valve) pairs $R$ stays positive: the model has no root |
| `injection_depth_wells.py run 1000 OUT.csv --casadi-cases 0 --processes 1`, on the core with each adaptation of its temperature solve at the injection cell undone in turn | The adaptations' effect (principle 7) | The chord iteration's slope saves 2 to 3% of the chord fallbacks and no measurable time; the widened bracket changes nothing on these cases. About 4 minutes per variant |

## Validity ranges

The measurements behind `../../specs/model/validity.md`, run on 2026-10-29 on `develop` at `8874b7f` (Python 3.14), on the Rust core with 8 processes. `validity_ranges.py run` solves wells of `develop`'s sampler (seed 2026) at their nominal operating point and two sampled ones, as `gor_cap_wells.py` does, and records each correlation's inputs at every grid point of each operating point; `summary` compares them with the published ranges in its `RANGES`.

| Run | What it shows | Result |
|---|---|---|
| `run 500 OUT --backend rust` | 1494 cases of 500 wells: the wells' inputs, and the state and each correlation's inputs along the 1453 operating points (146,753 grid points) | The solves take 436 s in total, a few minutes on 8 processes |
| `summary OUT` | Per row of the table: the sampler's range, and the share of cases outside the published range | See the table in `validity.md` |

Rerun on 2026-10-29 on the branch `independent-checks`, after the rulings on PVT-GAS-5 and PVT-MIX-9 (Independent checks, below) and, for comparison, with PVT-MIX-9's in-situ fraction restored: the same 1453 operating points. The flowing fraction moves one case below Beggs and Robinson's 5265 psia (101 cases above it before, 100 after: 7.0% to 6.9%). Below Chen's Re = 4000 lie 16 cases both before and after, 1.1%, where the table had 1.2% from `8874b7f`: the change is between that commit and the branch's base. No other share changes at the table's precision.

## Independent checks

The checks behind `../../specs/model/README.md`, Test vectors, Independent checks (`../model-lock-plan.md`, Step L3, which takes over Step V1 of the validation plan's draft), run on 2026-10-29 on `develop` at `f7f1218` (Python 3.14, casadi 3.8.1, numpy 2.5.3), with `fluids` 1.3.1 and scipy 1.18.1 in an overlay environment, so that neither is a dependency, and rerun on the branch `independent-checks` after each of Bjarne's two rulings of the same day (below):

```console
uv run --with fluids==1.3.1 python plans/evidence/independent_checks.py
```

It compares the CasADi backend's component functions; the vectors carry the result to the Rust core. It takes a few seconds. The worked examples are in `tests/test_worked_examples.py`, which runs with the test suite.

| Check | Grid | Result |
|---|---|---|
| FRIC-4 against `fluids.friction.Chen_1979(Re, eD)` | Re from 4000 to $10^{8}$ (81 points, logarithmic), $\varepsilon/D$ 0 and $10^{-7}$ to 0.05 (42) | largest relative difference $5.9\cdot10^{-16}$ |
| FRIC-5 against `fluids.friction.Haaland(Re, eD)` | the same | $6.5\cdot10^{-16}$ |
| FRIC-4 and FRIC-5 against the exact Colebrook–White solution, `fluids.friction.Clamond(Re, eD)` | the same | FRIC-4 −0.32% to +0.33%, FRIC-5 −1.42% to +1.31% |
| FRIC-4 against Chen's form of Colebrook's equation (3.7065, 2.5226), solved here | his range: Re from 4000 to $4\cdot10^{8}$ (61), $\varepsilon/D$ from $5\cdot10^{-7}$ to 0.05 (31) | Colebrook exceeds FRIC-4 by −0.28% to +0.4651%; his Table I gives at most +0.4651% and at least −0.2286% over 36 points |
| CHK-4 against `fluids.compressible.P_critical_flow(1, k)` | $k$ from 1.05 to 1.67 | identical |
| PVT-MIX-9 against `fluids.two_phase_voidage.gas_liquid_viscosity(x, mu_l, mu_g, Method='Cicchitti')` | $\alpha$ from 0 to 1, four gas and three liquid densities and viscosities, three pairs of phase velocities | identical. `fluids` is given `develop`'s gas mass fraction of the flow, so this checks the mixing rule; the worked example checks the fraction. Before the ruling the script gave it the in-situ fraction, with the same result |
| PVT-GAS-9, PVT-GAS-11 against Dranchuk and Abou-Kassem's Eqs. (2) and (3), transcribed and solved by bisection | $p_{pr}$ from 0.2 to 15 (33), $T_{pr}$ from 1.05 to 3 (40) | $\rho_r$ within $7.7\cdot10^{-15}$, $Z$ within $1.3\cdot10^{-15}$ (relative) |
| PVT-GAS-10 against a central difference of that transcription's $\ln Z$ in $T_{pr}$ along an isobar | the same | within $1.8\cdot10^{-7}$, the difference's truncation error |
| PVT-GAS-5 against Sutton's Eqs. (14) and (15) as printed | $\gamma_g$ from 0.571 to 1.679, his data | $T_{pc}$ within $1.2\cdot10^{-16}$, $p_{pc}$ within $1.9\cdot10^{-16}$. Before the ruling, $p_{pc}$ was lower by $5.9\cdot10^{-5}$ to $2.2\cdot10^{-4}$, from 131.07 for 131.0 |
| PVT-OIL-3 against Abdul-Majeed and Abu Al-Soof's Eqs. (1) and (3) | API 15 to 50, 0 to 150 °C | within $4.5\cdot10^{-16}$; against the 18 measurements of their Table 1, within 0.78 dyn/cm, 0.27 dyn/cm on average |
| PVT-OIL-12 against their Eqs. (4) and (5) | $R_{so}$ from 0.5 to 20 and 80 to 400 Sm³/Sm³, where the blend has no weight | within $1.9\cdot10^{-7}$. At 50 Sm³/Sm³ the blend gives 0.400, between the two equations' 0.425 and 0.375 |

The first run of Step V1 (2026-10-02) found the same agreement with `fluids`. Rerun the script when a vector of these IDs is regenerated.

### The rulings that follow the sources

The checks found two values that `develop` did not reproduce, and Bjarne ruled on 2026-10-29 to follow the source in each: PVT-GAS-5's coefficient 131.0 for 131.07 (`../../specs/model/pvt/gas.md`), and PVT-MIX-9's gas mass fraction of the flow for the in-situ one (`../../specs/model/pvt/mixture.md`). `source_rulings.py` measures each on the wells of `gor_cap_wells.py`: `develop`'s sampler (seed 2026), each well at its nominal operating point and two sampled ones, 1492 cases of 500 wells on the Rust core and 263 of 88 on the CasADi backend. `gor_cap_wells.py run` was run before the changes (`develop` at `ce1bb0e` with the branch's checks), after the first and after the second, each with 8 processes on 24 CPUs shared with other work, and `source_rulings.py compare` joins two runs.

| Change | Backend | Operating points | PBH | PWH | Reservoir liquid rate |
|---|---|---|---|---|---|
| PVT-GAS-5, 131.0 for 131.07 | Rust core | 1444 before and after, the same cases | within $7.5\cdot10^{-4}$ bar; median $4.8\cdot10^{-5}$ | within $5.7\cdot10^{-3}$ bar | within 0.0074% |
| PVT-GAS-5, 131.0 for 131.07 | CasADi | 253 before and after, the same cases | within $7.5\cdot10^{-4}$ bar; median $5.3\cdot10^{-5}$ | within $3.7\cdot10^{-3}$ bar | within 0.0032% |
| PVT-MIX-9, the flowing fraction | Rust core | 1444 before and after, the same cases | falls in every case, by at most 0.19 bar; median $9.5\cdot10^{-4}$ | rises, by at most 0.27 bar but in one case, 13.9 bar | rises, by at most 0.35% but in one case, 109% |
| PVT-MIX-9, the flowing fraction | CasADi | 253 before and after, the same cases | falls in every case, by at most 0.15 bar; median $2.0\cdot10^{-3}$ | rises, by at most 0.21 bar | rises, by at most 0.35% |

After each change the two backends find an operating point in the same 253 cases, as before, and agree within $6.3\cdot10^{-7}$ bar in PBH after the first and $1.1\cdot10^{-7}$ bar after the second. The time per case is unchanged.

With PVT-MIX-9's flowing fraction, the friction falls wherever the gas slips past the liquid, so PBH falls and PWH and the rate rise. The one large change is well 455's second sampled point: 0.02 kg/s of a gas-rich stream (GOR 2372 Sm³/Sm³) through a nearly closed choke ($u = 0.076$), of an oil of 21.8 °API whose Beggs–Robinson viscosity at the 274 K wellhead, below the correlation's range, is 290 Pa s. Its friction is laminar ($\text{Re} \approx 0.2$ at the wellhead), where $f_D = 64/\text{Re}$ is proportional to $\mu_m$, and its slip makes the flowing fraction 0.65 against an in-situ 0.04, so $\mu_m$ falls by a factor of 2.7. Its drawdown rises from 0.026 to 0.054 bar, its rate doubles and its PWH rises from 29.9 to 43.7 bar. Both backends agree there, before and after.

`source_rulings.py zero_flow` backs the note on the flowing fraction in `pvt/mixture.md`, Safeguards, where it is $0/0$. On well 0's CasADi system, at states through which nothing flows at any point, DISC-10 and CHK-1 are NaN with the in-situ fraction, with the flowing one and with a guarded one (0 where $w_m \le 0$), and DISC-9 is NaN too with the unguarded flowing fraction, which the code takes. At `v1.0.0+chen#6`, the start that reached a state with a supersonic cell with the in-situ fraction ends with Ipopt's `Invalid_Number_Detected` with the flowing fraction, guarded or not, and the root set is the same single root (264.00 bar before, 263.95 after). Four tests pinned operating points of the old PVT-MIX-9. The Joule–Thomson gas wells of `tests/test_rust_backend.py` were re-solved: their $p_0$ fell by 0.004, 0.066 and 0.246 bar, and both backends agree to $10^{-6}$ bar. `tests/test_roots.py` showed SOL-8's rejection of a supersonic state at `v1.0.0+chen#6`; a scan of the comparison set's overlays, wells 0 to 7, found two cases where a start now reaches one, and the test takes `develop+dead oil#6`, whose one root, 261.24 bar, the core finds too.

The `develop` vectors that the first change alters are those of PVT-GAS-3, PVT-GAS-4 and PVT-GAS-5, and PVT-GAS-11, by at most $10^{-4}$ (relative), and THM-8, by at most $1.2\cdot10^{-6}$ K/m, which is 2% of the third row's, near where the gas's Joule–Thomson factor changes sign. `tests/data/pvt/vazquez_beggs.csv`, regenerated by `specs/tools/make_pvt_tables.py`, changes in $Z$ only, by at most $5.9\cdot10^{-5}$ (relative). The second alters FRIC-3's, whose $f_D$ and $F$ fall by 0.18%, 8.3% and 0.41%, and PVT-MIX-9's, which take the phase velocities as inputs: its rows without gas and without slip keep their values, and the two with slip fall by 10% and 73%.

## Step L6: the Distributions check's noise and bias

Measurements behind Step L6 of `../model-lock-plan.md`, run on 2026-10-30 on the branch `distribution-margin` (`develop` at `e44bb54`; Python 3.14, casadi 3.8.1), with 12 processes on 24 CPUs shared with other work (the CasADi backend's seeds 1 and 2 with 20). The v1.0.0 replay ran in v1.0.0's environment (Python 3.11, casadi 3.6.4). `distribution_noise.py regen` runs `scripts/verification/regenerate_distributions.py`, 5 samples per published `sol-1` well unless stated, with the `--seed` option this step added; its default, 20261001, reproduces Step 7's draws. The check's statistic below is the largest CDF gap over the 21 features (bound 0.02) and the largest rank-correlation gap (bound 0.05), with the candidate weighted as the check weights it.

| Run | What it shows | Result |
|---|---|---|
| `regen DIR --backend casadi` | Today's CasADi backend at the default seed | 9,880 rows, PASS: PWH 0.0191, rank correlation 0.030 (Step 7: 9,879 rows, 0.0196 and 0.031). Every published well now has an operating point at u = 0.5 (Step 7: 2 without), and 120 samples fail. 1,322 s |
| `regen DIR --backend rust` | Today's Rust core at the default seed | 9,920 rows, FAIL: PWH 0.0210, 0.035, as in Step 8. 80 samples fail. 97 s |
| `regen DIR --backend rust --seeds 1 2 … 10` | The spread over 11 seeds | 0.0218 ± 0.0030 (mean ± standard deviation), at most 0.0285; correlation 0.0365 ± 0.0042, at most 0.0418. It passes 0.02 at 3 of the 11 seeds. CHK is the largest gap at 7 of them. 83 s each |
| `regen DIR --backend casadi --seeds 1 2` | The CasADi backend at two more seeds | Seed 1: 9,859 rows, 0.0163 (PWH), 0.039; seed 2: 9,875 rows, 0.0187 (PWH), 0.037. Over the 3 seeds 0.0180 ± 0.0015, at most 0.0191; correlation 0.0354 ± 0.0048, at most 0.0393. It passes 0.02 at all three, by 0.0009 to 0.0037. 1,093 and 1,125 s on 20 processes, with other work on the machine |
| `reference manywells-sol-1.zip OUT --samples 5 10 20 50 --draws 30` | The reference's own rows, drawn per well as a regeneration draws and weighted: what a candidate without bias gives | 5 per well: 0.0085 ± 0.0021, at most 0.0138; correlation 0.0166 ± 0.0049, at most 0.0296 (PWH: 0.0064 ± 0.0019). 10, 20 and 50 per well: 0.0061, 0.0046 and 0.0025; correlation 0.0109, 0.0090 and 0.0053. 71 s |
| `regen DIR --backend rust --samples 10 20 50` | Whether more samples shrink the gap | 0.0203, 0.0204 and 0.0191 (CHK; PWH 0.0189); correlation 0.035, 0.035 and 0.034. The gap does not fall: it is a bias. 154, 291 and 707 s |
| `draws`, `v1`, `drivers --v1 --data` | v1.0.0's generator on the default seed's draws, and where the gap comes from | Below. The replay takes 344 s |
| `roots`, `power` | What the check catches: candidates that take the trickle root in a share of the draws that have one | Below. 69 s |
| `inputs DIR/rust-n5-seed20261001.parquet` | What the check catches of a sampler error in a drawn input | Below. Seconds |

Per feature, over the seeds: the CDF gap, and the largest rank-correlation gap in the feature's row of the matrix, each as mean ± standard deviation (maximum); the headers give the number of seeds, or of draws of the reference's rows.

| Feature | Rust core, CDF (11) | Rust core, correlation | CasADi backend, CDF (3) | CasADi backend, correlation | Reference's rows, CDF (30) | Reference's rows, correlation |
|---|---|---|---|---|---|---|
| CHK | 0.0210 ± 0.0035 (0.0285) | 0.0332 ± 0.0056 (0.0418) | 0.0170 ± 0.0013 (0.0184) | 0.0283 ± 0.0047 (0.0316) | 0.0076 ± 0.0027 (0.0138) | 0.0164 ± 0.0052 (0.0296) |
| PBH | 0.0042 ± 0.0008 (0.0053) | 0.0178 ± 0.0035 (0.0245) | 0.0037 ± 0.0005 (0.0042) | 0.0148 ± 0.0014 (0.0161) | 0.0034 ± 0.0005 (0.0044) | 0.0095 ± 0.0056 (0.0296) |
| PWH | 0.0203 ± 0.0026 (0.0248) | 0.0337 ± 0.0057 (0.0416) | 0.0180 ± 0.0015 (0.0191) | 0.0314 ± 0.0028 (0.0346) | 0.0064 ± 0.0019 (0.0102) | 0.0126 ± 0.0038 (0.0216) |
| PDC | 0.0040 ± 0.0007 (0.0056) | 0.0334 ± 0.0045 (0.0410) | 0.0038 ± 0.0007 (0.0046) | 0.0346 ± 0.0061 (0.0393) | 0.0034 ± 0.0006 (0.0047) | 0.0094 ± 0.0046 (0.0201) |
| TBH | 0.0000 ± 0.0000 (0.0000) | 0.0151 ± 0.0051 (0.0231) | 0.0002 ± 0.0001 (0.0003) | 0.0132 ± 0.0084 (0.0228) | 0.0000 ± 0.0000 (0.0000) | 0.0094 ± 0.0057 (0.0295) |
| TWH | 0.0112 ± 0.0020 (0.0151) | 0.0314 ± 0.0063 (0.0416) | 0.0087 ± 0.0001 (0.0088) | 0.0292 ± 0.0013 (0.0302) | 0.0051 ± 0.0015 (0.0096) | 0.0102 ± 0.0053 (0.0268) |
| WGL | 0.0041 ± 0.0016 (0.0067) | 0.0128 ± 0.0046 (0.0199) | 0.0029 ± 0.0006 (0.0035) | 0.0157 ± 0.0074 (0.0228) | 0.0031 ± 0.0008 (0.0049) | 0.0078 ± 0.0034 (0.0163) |
| WGAS | 0.0097 ± 0.0012 (0.0112) | 0.0334 ± 0.0036 (0.0395) | 0.0088 ± 0.0020 (0.0111) | 0.0346 ± 0.0061 (0.0393) | 0.0046 ± 0.0015 (0.0109) | 0.0091 ± 0.0045 (0.0201) |
| WLIQ | 0.0107 ± 0.0025 (0.0150) | 0.0292 ± 0.0036 (0.0347) | 0.0101 ± 0.0007 (0.0105) | 0.0288 ± 0.0014 (0.0296) | 0.0050 ± 0.0010 (0.0072) | 0.0094 ± 0.0029 (0.0168) |
| WOIL | 0.0085 ± 0.0021 (0.0122) | 0.0263 ± 0.0049 (0.0343) | 0.0078 ± 0.0003 (0.0080) | 0.0264 ± 0.0038 (0.0294) | 0.0050 ± 0.0012 (0.0079) | 0.0089 ± 0.0033 (0.0185) |
| WWAT | 0.0062 ± 0.0011 (0.0080) | 0.0231 ± 0.0030 (0.0272) | 0.0052 ± 0.0011 (0.0063) | 0.0239 ± 0.0043 (0.0269) | 0.0042 ± 0.0011 (0.0069) | 0.0088 ± 0.0030 (0.0176) |
| WTOT | 0.0112 ± 0.0021 (0.0147) | 0.0341 ± 0.0059 (0.0418) | 0.0095 ± 0.0007 (0.0104) | 0.0334 ± 0.0030 (0.0358) | 0.0057 ± 0.0016 (0.0099) | 0.0094 ± 0.0028 (0.0186) |
| QGL | 0.0044 ± 0.0016 (0.0076) | 0.0128 ± 0.0046 (0.0199) | 0.0027 ± 0.0004 (0.0031) | 0.0158 ± 0.0075 (0.0229) | 0.0031 ± 0.0009 (0.0050) | 0.0077 ± 0.0033 (0.0162) |
| QGAS | 0.0102 ± 0.0015 (0.0129) | 0.0334 ± 0.0034 (0.0392) | 0.0092 ± 0.0016 (0.0110) | 0.0344 ± 0.0060 (0.0390) | 0.0045 ± 0.0012 (0.0094) | 0.0094 ± 0.0045 (0.0212) |
| QLIQ | 0.0107 ± 0.0025 (0.0144) | 0.0291 ± 0.0038 (0.0351) | 0.0100 ± 0.0004 (0.0103) | 0.0289 ± 0.0014 (0.0300) | 0.0051 ± 0.0011 (0.0072) | 0.0093 ± 0.0028 (0.0164) |
| QOIL | 0.0085 ± 0.0021 (0.0129) | 0.0263 ± 0.0050 (0.0343) | 0.0074 ± 0.0007 (0.0082) | 0.0263 ± 0.0038 (0.0294) | 0.0050 ± 0.0014 (0.0086) | 0.0088 ± 0.0033 (0.0188) |
| QWAT | 0.0062 ± 0.0011 (0.0080) | 0.0231 ± 0.0030 (0.0272) | 0.0052 ± 0.0011 (0.0063) | 0.0239 ± 0.0043 (0.0269) | 0.0042 ± 0.0011 (0.0069) | 0.0088 ± 0.0030 (0.0176) |
| QTOT | 0.0119 ± 0.0020 (0.0158) | 0.0322 ± 0.0046 (0.0410) | 0.0114 ± 0.0015 (0.0130) | 0.0338 ± 0.0070 (0.0391) | 0.0050 ± 0.0016 (0.0108) | 0.0098 ± 0.0039 (0.0200) |
| FGAS | 0.0023 ± 0.0002 (0.0027) | 0.0127 ± 0.0060 (0.0234) | 0.0023 ± 0.0002 (0.0025) | 0.0099 ± 0.0018 (0.0117) | 0.0022 ± 0.0004 (0.0038) | 0.0095 ± 0.0053 (0.0283) |
| FOIL | 0.0027 ± 0.0007 (0.0040) | 0.0131 ± 0.0032 (0.0186) | 0.0025 ± 0.0008 (0.0033) | 0.0104 ± 0.0016 (0.0117) | 0.0026 ± 0.0003 (0.0031) | 0.0089 ± 0.0044 (0.0197) |
| FWAT | 0.0021 ± 0.0003 (0.0026) | 0.0112 ± 0.0026 (0.0158) | 0.0023 ± 0.0002 (0.0024) | 0.0119 ± 0.0032 (0.0145) | 0.0020 ± 0.0003 (0.0031) | 0.0087 ± 0.0034 (0.0188) |

**The gap is v1.0.0's, not `develop`'s** (Step 8's gap 5, measured). `v1` solves each of the default seed's 10,000 draws as v1.0.0's generator did, from the well's root at u = 0.5. It gives 9,280 rows, fails 352 draws, drops 48 ($w_m < 0.1$ kg/s) and puts 305 on the trickle signature; 3 wells (15 draws) have no root at u = 0.5 in the replay, although v1.0.0 solved them when it made the data. Its 9,280 rows match the reference within the noise above (0.0063 and 0.0136), so the replay stands in for the published data. Where v1.0.0 and `develop` reach the same root, in 9,150 draws, their rows agree within $2\cdot10^{-3}$ bar in PBH on the Rust core and $3\cdot10^{-7}$ bar on the CasADi backend. What remains is two v1.0.0 defects:

1. **Failed draws.** v1.0.0's generator drew again after a failed solve, so the published data lack the draws it failed. They lie at small openings (u between 0.074 and 0.141 for the middle half, median 0.099) and high PWH (median 135 bar), and `develop` solves 276 of the 352 on the Rust core and 234 on the CasADi backend. In the published rows 13.0% of the choke positions are below 0.2, where U(0.05, 1) gives 15.8%: the reference's CHK CDF lies up to 0.025 below the uniform's, so even CHK, an input, fails 0.02 when a solver solves every draw and the noise is small. Without the draws v1.0.0 fails or drops, `develop`'s PWH gap falls from 0.021 to 0.007 (CasADi: from 0.019 to 0.008) and CHK's from 0.020 to 0.009.
2. **Trickle rows that the signature misses.** In 126 draws (1.3%), v1.0.0's row is off the signature but on another root: PBH 34 bar higher at the median, the flow 4% of the stable root's, TWH 281.5 K against 330.7 K, and PWH − PDC from 0.012 to 0.39 bar (10th to 90th percentile). They stay in the reference. Without them, v1.0.0's own rows show the gap `develop` shows: WTOT 0.017 and correlation 0.043. A wider signature would not separate them cleanly: 78 of the 9,150 stable rows have PWH − PDC below 0.5 bar.

With every row, the failed draws dominate (PWH, CHK); without them, the missed trickle rows do (WTOT 0.016 to 0.017, correlation 0.044 to 0.045 on the two backends). The PWH gap sits at P62 (74.7 bar), with the regeneration's CDF more than 0.005 below the reference's from P15 to P95: the regeneration has more rows at high wellhead pressures. It is spread over the wells: none contributes more than 0.0003, and the 127 wells that lost signature rows give −0.003 of the −0.021. The Rust core's gap is larger than the CasADi backend's because it solves 50 draws the CasADi search does not, at small openings (median u 0.11, PWH 158 bar); the CasADi backend has 10 the core lacks, and in 15 shared draws the two pick different stable roots (Step 8's gaps 2 and 3). The replay of seed 1's draws gives the same picture: v1.0.0 fails 339 draws, drops 58, puts 307 on the signature and 117 on a missed trickle root; its own rows give 0.0095 and 0.0122; and without its failed draws the Rust core's PWH gap falls from 0.019 to 0.005.

**The weights blunt the check against the trickle root.** The calibration in `specs/verification.md`, 0.034 (TWH) and 0.092 for the published rows with the trickle rows left in, is unweighted. Weighted, as the check weights since 2026-10-01, the same rows give 0.0050 and 0.020 and pass: the signature rows sit in the wells the reference keeps fewest rows of (a median 66% of such a well's rows are on the signature), so their weight falls from 3.53% of the rows to 0.73%. The replay as a candidate, its signature rows kept, passes as well (0.0082 and 0.027, against 0.032 and 0.079 unweighted), closer to the reference than `develop`. Spread over all wells, as a solver's defect would be, trickle roots are caught (`power`, three draws each, weighted):

| Trickle roots taken | Rows on the trickle root | CDF gap | Rank-correlation gap |
|---|---|---|---|
| 5% of the 5,214 draws with one | 1.1–1.4% | 0.019–0.021 | 0.028–0.033 |
| 10% | 2.1–2.6% | 0.024–0.026 | 0.044–0.053 |
| 15% | 3.4–3.6% | 0.032–0.034 | 0.069–0.074 |
| 20% | 4.8–5.1% | 0.039–0.043 | 0.093–0.109 |
| 25% | 5.9–6.1% | 0.048–0.049 | 0.122–0.125 |

More than half of the trickle roots carry less than 0.1 kg/s and are dropped, as the generator drops them, so the rows on the trickle root are fewer than the draws.

**Sampler errors are caught coarsely** (`inputs`, which redraws one input's column of the default seed's Rust rows from a wrong range, leaving the solved features as they are). The control, CHK redrawn from SMP-18's U(0.05, 1), gives 0.022: the reference's own bias. A choke from U(0.1, 1) gives 0.037, and lift gas from U(0, 4) 0.034. A separator pressure within ±20% gives 0.010 and gas fractions within ±10% give 0.004: the wells' nominal values differ so much that a per-well range barely moves the marginal.

## Sampler redesign: the draft well draws

The draw-only check of Step R3 of `../sampler-redesign-plan.md` (`../sampler-redesign/sampling-draft.md`), run on 2026-10-30 on the branch `sampler-redesign`, in about 2 s for 2000 wells on one process. It draws wells without solving them; R6 extends it to solve them. The geometry shares in it are placeholders, pending the geometry evidence.

| Script | What it shows | Result |
|---|---|---|
| `sampler_draft_draws.py 2000 [--seed 2026] [--csv OUT]` | The draft's labels, marginals and rank correlations, the shares behind Bjarne's goals, every truncation and bound, and `configurations.check` of each well in the draft's interim mapping to `v2.0.0` | No bound violated and every well in `v2.0.0`. Gradients and temperatures by fluid class within 0.008 bar/m and 6 K of BOEM's medians by wells; rank correlations of black oils within 0.12 of the Gulf's but for $R_{si}$–$T_r$ (0.32 against 0.56). 46.7% of core wells below hydrostatic, 15.6% with a nominal separator pressure below 10 bar, 20.4% above 0.6 psi/ft. Oils' design liquid rates 33 / 451 / 6411 Sm³/d (P10 / P50 / P90), which R6 checks against the Gulf's brackets |
