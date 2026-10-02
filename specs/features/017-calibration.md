# 017 · Calibration to a well's production data

*Feature spec for Step C1 of `plans/calibration-plan.md`, 2026-10-02. Status: implemented on the branch `calibration`, with Steps C2 to C7 of the plan; signed off by Bjarne on 2026-10-02 (Sign-off, at the end). Not a model feature: it adds `specs/calibration.md` and changes no equation in `specs/model/`.*

## Decisions (Bjarne, 2026-10-02)

- **Parameters:** the choke coefficient, the inflow productivity, friction and the heat-transfer coefficient, each switchable.
- **Forward model:** every row is solved for the whole well's operating point; the objective covers whatever was measured.
- **Model error:** measurement noise only, no discrepancy term.
- **Result:** the MAP point estimate only, no covariance.
- **Tests:** simulate wells, then check that the calibration gets close to, if not identifies, the parameters they were simulated with.

## Motivation

ManyWells' second purpose is "flow prediction after calibration to a well's data" (`specs/goals.md`), and v2.0.0 needs calibration, checked privately against real-well data. The calibration functions on `develop` fitted one component at a time to columns that are not measured: `calibrate_simpson_choke_model` needed the gas mass fraction and densities upstream of the choke, which only a simulation gives. They had no priors, no missing values and no noise model.

## Delta

- **`specs/calibration.md`** (new): CAL-1 to CAL-13, the data, noise, parameters and priors, the predicted observations, the objective, the fit, the result and the synthetic data, with its checks. `tests/spec_parse.py` reads it, so the traceability test checks the CAL tags as it checks SMP.
- **`specs/model/`:** no change. The `v1.0.0` configuration and the verifier's reports are untouched.
- **`specs/architecture.md`:** the module row of `calibration/` and the calibration contract (Interface contracts, Calibration).
- **`datasets/rows.py`:** `root_features(root, bc, rho_o, rho_w, rho_g, x_w)` is extracted from `sample_row`, so that the datasets and the calibration define the features once; the dataset rows do not change.
- **Code** (`src/manywells/calibration/`): `data.py` (`CalibrationData`, `Noise`), `parameters.py` (`Parameter`, `apply`, `default_prior`, `darcy_productivity_index`, `vogel_maximum_rate`), `objective.py`, `fit.py` (`calibrate`, `evaluate`, `CalibrationResult`, `CalibrationError`), `synthetic.py` (`synthetic_data`). `choke_cal.py` and `inflow_cal.py` are removed.
- **`pyproject.toml`:** `scipy` is declared. It was already in `uv.lock` as scikit-learn's dependency (1.17.1, or 1.18.1 on Python ≥ 3.12); the lock gains the declaration only.
- **Docs and scripts:** `docs/calibration.md`; `scripts/sim_examples/calibrate_well.py`, run by `tests/test_examples.py`; `scripts/calibration/twins.py`, the twin study.

## The method

Each row of one well's data is solved for the operating point (`simulate`) of its own `WellProperties`: the free parameters applied with `dataclasses.replace`, and the row's fluid. The predicted observations are PBH, PWH, TWH and the reservoir mass rate, with the dataset features' definitions. The MAP estimate minimizes the noise-scaled residuals plus the priors' residuals, in the standardized log $z$, by `scipy.optimize.least_squares` with a forward-difference Jacobian. On the Rust core every row of every evaluation, the Jacobian's columns included, is solved in one pool of threads, since the core releases the GIL. A row with no operating point takes a barrier; where some row has none at the prior medians, the fit starts from the feasible start with the lowest cost, one or two prior standard deviations along one parameter (CAL-9).

## Measurements

On 24 CPUs, 2026-10-02. The scripts are `scripts/calibration/twins.py` (items 6 to 10) and `plans/evidence/calibration_measurements.py` (items 1 to 3, the false minimum of item 7 and the root sets of item 10).

1. **Cost of a row** on the Rust core, the full search (`simulate`), for wells drawn with `manywells.sampling` (seed 1) at 20, 50 and 100 cells: vertical wells 32, 65 and 117 ms (well 1), 166–200, 258–336 and 296–437 ms (well 3); a deviated well 334–373, 347–354 and 413–420 ms (well 0); an L-shaped well 318–326, 575–581 and 809–812 ms (well 2).
2. **Threads.** 16 rows of well 1 at 50 cells: 1.05 s on 1 thread, 0.27 s on 4, 0.14 s on 8, 0.09 s on 16, a speed-up of 12 on 16 threads.
3. **The Jacobian's step.** The forward difference of PBH, PWH, TWH and $\log W$ in $\log\theta$, for each of `K_c`, `w_l_max`, `roughness` and `h` on well 1 at 50 cells and $u = 0.6$, with steps $10^{-1}$ to $10^{-8}$: the quotients agree to four digits from $10^{-4}$ and to five from $10^{-5}$ to $10^{-8}$. So a step of $10^{-6}$ in $z$ (a change of $s \cdot 10^{-6}$ in $\log\theta$) is well inside the range where the solves' tolerance does not matter. For example, $d\,\text{PWH}/d\log K_c$ is $-2.62129$ at $10^{-4}$ and $-2.62149$ from $10^{-6}$ to $10^{-8}$.
4. **Identification**, on wells 0 to 7 of seed 1 (vertical, deviated and L-shaped, 20 cells, 10 rows), noise-free rows with the noise scaled down by 1000, $z^* = (1.5, -1, 1, -1.5)$: every parameter within $2\cdot10^{-7}$ of $z^*$ in every well, in 280 to 400 solves, 1 to 23 s per fit.
5. **Getting close** on the same wells with noise (items 4 and 5 were measured with the fit started at the medians, before item 7's start search; the recovery tests pass with it): the distance to $z^*$ after the fit is 0.04 to 1.11, against 2.55 at the medians; `K_c`, `w_l_max` and `h` are each within 0.27 in every instrumentation tested. The largest errors are the roughness's on well 1, whose tubing is almost smooth (Findings, 1).

6. **The twin study** (`uv run python -m scripts.calibration.twins twins.json --wells 30`).
   - **Setup.** Wells 0 to 29 of seed 2026 from `manywells.sampling` on `develop`'s model, 20 cells: vertical, deviated and L-shaped wells, Vogel inflow, Simpson chokes of every profile, black oil, roughness friction. Well 11 has no operating point at its nominal conditions and is left out. Each well has $z^*$ drawn from the priors, $N(0, 1)$ per parameter clipped to ±2.5, with all four parameters free.
   - **Recovery.** 20 noisy rows in each instrumentation; held out, 10 new rows with full instrumentation. Errors $|\hat z - z^*|$ in prior standard deviations:

   | Instrumentation | fits | converged | closer than the medians | median error (`K_c`, `w_l_max`, `roughness`, `h`) | largest error | held-out RMS, median |
   |---|---|---|---|---|---|---|
   | `full` | 29 | 100% | 100% | 0.01, 0.01, 0.04, 0.02 | 0.04, 0.04, 1.00, 0.18 | 1.03 |
   | `no_downhole` | 29 | 100% | 100% | 0.01, 0.02, 0.10, 0.02 | 0.45, 0.32, 1.31, 0.15 | 1.41 |
   | `periodic_tests` | 29 | 100% | 100% | 0.02, 0.01, 0.08, 0.02 | 0.51, 0.05, 1.31, 0.15 | 1.02 |
   | `pressures_only` | 29 | 100% | 97% | 0.03, 0.01, 0.14, 0.04 | 0.13, 0.06, 1.45, 0.22 | 1.05 |
   | `random_missing` | 29 | 100% | 100% | 0.02, 0.00, 0.05, 0.03 | 0.26, 0.07, 0.83, 0.14 | 1.06 |

   - **Not closer.** The one fit that ended no closer than the medians (well 6, `pressures_only`) has its truth 0.24 from the medians and ends 0.24 from it, within the noise.
   - **The held-out RMS of `no_downhole`** is 1.41: its held-out rows have PBH, which it was not calibrated on.
7. **The start search** (CAL-9, principle 7), the same 145 recovery fits with the fit started at the medians alone (falling back to the axis starts only where a row cannot flow there, the first version) and with the cheapest feasible of the $4p + 1$ starts (the rule as specified):
   - **Without the start search.** 20 of the 145 fits (4 of the 29 wells, in every instrumentation) cannot start at the medians at all: some row has no operating point there, typically at a small choke opening. With a start at the medians alone they raise `CalibrationError`.
   - **Without the cheapest start.** Two fits (well 3, deviated, `periodic_tests` and `pressures_only`) converged to $z_\text{roughness} = 13.0$ with "success". That is a roughness of 143 m in a 0.14 m tubing, where the friction correlation has a spurious valley: cost 119.6 against 27.6 in the truth's basin. With the medians 2.4 standard deviations from the truth's productivity (cost 134,000 there), the first steps crossed ground where the well cannot flow. Bounding $z$ to ±4 does not help: the fit stops at the bounds, at a cost of 13,000 to 23,000. Started at the cheapest of the $4p + 1$ starts, both reach the truth's basin (errors 1.28 and 1.45, from the roughness).
   - **The cost.** The start search raises the median number of solves per fit from 700 to 940, the total over 145 fits from 126,564 to 155,004 (+22%), and the median time per fit from 7.6 s to 10.8 s on 24 CPUs. No other fit changed by more than 0.3 in $z$.

8. **More data, closer** (wells 0 to 7, full instrumentation; median error in $z$):

   | Data | `K_c` | `w_l_max` | `roughness` | `h` |
   |---|---|---|---|---|
   | 10 rows | 0.015 | 0.006 | 0.150 | 0.019 |
   | 40 rows | 0.012 | 0.003 | 0.070 | 0.012 |
   | 160 rows | 0.002 | 0.002 | 0.059 | 0.010 |
   | 20 rows, noise / 4 | 0.003 | 0.002 | 0.015 | 0.005 |

   A fit of 160 rows takes 40 to 65 s.
9. **Misspecified twins** (wells 0 to 7, 20 rows, full instrumentation): data from a model the calibration lacks, calibrated with the shared parameters' priors. Bias of $\hat z$ from $z^*$:

   | Model error | median bias (`K_c`, `w_l_max`, `h`) | largest bias | held-out RMS, median and largest |
   |---|---|---|---|
   | a Simpson choke, calibrated as a Bernoulli one | +0.06, +0.00, −0.00 | −0.53, +0.06, −0.19 | 1.08, 5.01 |
   | another choke profile | −1.49, −0.02, −0.03 | −3.08, +1.73, −0.77 | 14.11, 48.36 |
   | the energy terms (frictional heating, gravity, Joule–Thomson) left out | +0.01, +0.00, +0.41 | +0.56, +0.04, +1.45 | 1.08, 4.79 |
   | roughness friction, calibrated with a fixed `f_D` | −0.00, −0.00, +0.02 | +0.56, −0.16, −0.06 | 1.07, 1.57 |
   | slip $C_0$ 10% low | −0.02, −0.03, +0.01 | +0.94, −0.33, +0.52 | 3.44, 14.03 |

10. **The backends**, on wells 0 to 7 at 10 cells: 4 rows from the Rust core, `K_c` and `h` free, calibrated by each backend on the same data.
    - **Same estimate.** In 7 wells the estimates agree to $2.5\cdot10^{-8}$ in $z$ or better.
    - **Well 1 (L-shaped).** The CasADi backend cannot start: its search finds no root at two of the four rows, where the core finds a stable one (`plans/improvements.md` §2.9).
    - **Time.** A fit takes 2 to 8 s on the core and 45 to 330 s on the CasADi backend, which builds a system per row and parameter value. These were measured with another run on the same CPUs, so they are upper bounds.
    - **A first run that mixed the two effects.** It generated each backend's data with that backend, and on two wells the CasADi data differed from the core's at small choke openings:
      - on well 7 its search returned another stable root (two stable roots at $p_0$ = 356.62 and 356.88 bar, where the core has one at 356.47);
      - on well 1 it returned no root at two rows.

      On the same data, well 7 agrees to $4\cdot10^{-10}$.

## Acceptance

- `tests/test_calibration.py` (fast): the data's validation, the noise, the parameters and priors, the predicted observations, the residuals and the synthetic data's instrumentation.
- `tests/test_calibration_recovery.py` (slow): the Checks of `specs/calibration.md` (identification, getting close in every instrumentation, determinism), the result and its diagnostics, the start off the medians and a row that cannot flow anywhere, on seeded wells.
- The twin study: in every instrumentation, at least 95% of the fits converge, end closer to $z^*$ than the medians are where $z^*$ is more than 0.5 from them, have `K_c`, `w_l_max` and `h` within 0.5 of $z^*$, and have every RMS of scaled residuals below 2. Measured: 100%, 100%, 96.6% to 100% (28 of 29 in `periodic_tests`, whose largest `K_c` error is 0.51) and 100%.
- `tests/test_spec_traceability.py` passes with the CAL namespace; `tests/test_sampling.py` passes unchanged after the extraction; `scripts/sim_examples/calibrate_well.py` runs (`tests/test_examples.py`).

## Findings

1. **Roughness is the weak parameter.** In smooth tubing the friction factor depends on the Reynolds number alone, so the roughness barely changes the pressure drop. On well 1 of seed 1 ($\varepsilon \approx 6\cdot10^{-6}$ m) a unit change of $\log\varepsilon$ moves PBH by 0.1 bar. Its median error is 0.04 to 0.14, and the largest 0.8 to 1.45, where the other parameters stay within 0.51 (Measurements, item 6). The prior keeps it where the data do not determine it, as intended.
2. **The plan's identifiability table was wrong about productivity.** It expected the productivity to be weakly determined without a downhole gauge. With $p_r$ given and rates on some rows it is well determined: largest error 0.32 without PBH, 0.06 with rates on two rows only. The plan and `docs/calibration.md` have the measured table. The case without any rates, and an unknown $p_r$, were not tested.
3. **The medians may not flow.** For 4 of 29 wells, some row has no operating point at the prior medians, although the well flows at its own parameters. A start at the medians cannot proceed there, because the barrier is flat (Measurements, item 7).
4. **A correlation outside its range can trap the fit.** From medians far from the truth, the fit stepped into a valley of the friction correlation at a roughness of 143 m and reported success. The start search avoids both 3 and 4 (CAL-9). A check that a parameter stays in its correlation's validity range would catch this kind of trap directly; it is not in this version.
5. **Model error is sometimes absorbed silently.** With the energy terms left out, $h$ absorbs them, with a median bias of +0.41 and up to +1.45, while the held-out RMS stays near 1 in most wells. A Bernoulli choke for a Simpson one does the same in `K_c` (up to −0.53). A wrong choke profile or slip law shows in the held-out residuals (median RMS 14 and 3.4) and biases `K_c` by up to 3 standard deviations. This is the evidence for the discrepancy decision deferred in `plans/calibration-plan.md`: with measurement noise only, a parameter shift beyond 2 (CAL-12) and the residuals against `CHK` are the only signs of model error, and they miss some of it.
6. **A fixed `f_D` is an adequate stand-in for roughness friction** at these wells: no bias beyond 0.56, and a held-out RMS of at most 1.57.
7. **The CasADi backend calibrates as the core does, where its search finds the core's roots.** It cannot calibrate a well where it misses them (Measurements, item 10; `plans/improvements.md` §2.9), and it is 20 to 100 times slower here, so the core is the calibration's backend. Its cost, a system built per row and parameter value, would fall if the fluid's fractions and the parameters were symbols of the CasADi system (`plans/improvements.md` §4.5).

## Breaking change

`calibration.choke_cal` and `calibration.inflow_cal` are removed; `calibrate` with one free parameter replaces them (CHANGELOG).

## Out of scope

The items under After this plan in `plans/calibration-plan.md`: posterior uncertainty (a Laplace approximation from the fit's Jacobian), model discrepancy, uncertain inputs such as $p_r$, change over time and sequential recalibration, hierarchical priors across wells, more parameters, faster solves (a tracked operating point on the core, or the parameters as symbols of the CasADi system) and modular fits. The private real-well checks (Step C8) wait for `plans/validation-plan.md`.

## Sign-off (Bjarne, 2026-10-02)

- **`specs/calibration.md`:** the data's columns (CAL-1, CAL-2), the rate observation (CAL-3), the noise defaults (CAL-4), the priors (CAL-6), the barrier and the start search (CAL-9), the stopping rule (CAL-11), and the Checks' tolerances: identification within $10^{-3}$; getting close within 0.5 for `K_c`, `w_l_max` and `h`, with residual RMS below 2; and the twin study's share of 95%. Signed off.
- **The start search** (constitution, principle 7; Measurements, item 7). It lets 20 of 145 twin fits start at all and removes 2 false minima, for 22% more solves in total. Signed off.
- **The calibration contract** in `specs/architecture.md`, **the removal** of `choke_cal` and `inflow_cal` (an API break, in the CHANGELOG), and **declaring `scipy`**. Signed off.
- **The added line in `AGENTS.md`** that puts `specs/calibration.md` among the specs that need sign-off. Signed off.
- **Old files.** Bjarne asked for the old calibration files to be deleted: `choke_cal.py` and `inflow_cal.py` with this branch, and his untracked `scripts/wellbore_cal.py`, which no longer ran on `develop`'s API.
