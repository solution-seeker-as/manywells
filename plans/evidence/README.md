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
