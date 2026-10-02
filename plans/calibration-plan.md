# ManyWells calibration — plan

*Draft 2026-10-02. Owner: Bjarne Grimstad. Status: proposal. Bjarne made the scope decisions below on 2026-10-02; everything else is a draft for his sign-off.*

This plan makes it easy to calibrate ManyWells to a well's production data: rates from well tests or multiphase flow meters (MPFMs), and pressures and temperatures from whatever sensors the well has. Rows may have missing values, and instrumentation differs between wells; not every well has a downhole gauge. The first version calibrates four parameters per well (choke, inflow, friction, heat transfer), each with a physics-based prior, and returns the posterior mode (MAP). It works with both backends, the Rust core by default.

The foundation plan (`plans/manywells-v2-plan.md`) defers calibration until after its Step 10 and lists it under After this plan. v2.0.0 needs it, checked privately against real-well data (`specs/goals.md`).

## Starting point

- **Existing code.**
  - `src/manywells/calibration/choke_cal.py` fits `K_c` of a Bernoulli or Simpson choke by least squares, and `inflow_cal.py` fits `k_l` or `w_l_max`.
    - The choke fit needs the mixture density or gas mass fraction upstream of the choke, which is not measured; only a simulation gives it.
    - Neither has priors or handles a missing value.
    - Only `tests/test_calibration.py` uses them. `plans/improvements.md` §2.8 lists their bare `Exception` and `print`.
  - `scripts/wellbore_cal.py` (untracked, March 2026) fits the pipe roughness to PWH from marches anchored at the measured PBH. It cannot run on `develop`: it sets fields on frozen inputs and calls `_simulate_cellwise`, which `d8f015f` removed.
- **What a calibration can build on.**
  - `SSDFSimulator(wp, backend=...).simulate(bc)` returns the operating point as a `Root` (`solution.py`), or raises `NoOperatingPoint`.
    - From a `Root` come PBH (`p_0`), TBH, PWH and TWH (first and last points of `state`), and the reservoir rates `w_res` and `w_g_res`.
    - `datasets/rows.py` (`sample_row`) turns a root into the dataset features of `docs/datasets.md`. It takes a sampler draw, not a `WellProperties`.
  - Inputs are frozen. A new parameter value means a new `WellProperties`, made with `dataclasses.replace`.
  - Only the seven operating-point values are parameters of the built CasADi system (`discretization.py`, `PARAMS`). Well parameters such as `K_c`, `k_l`, `f_D` and `h` are constants in it.
    - On the CasADi backend a new value therefore means a rebuild (median 0.30 s, Step 7).
    - On the Rust core it costs a new `_core.Well`, about 0.03 ms.
  - The Rust core's root search takes a median of 0.15 s per case on `develop`'s model; the CasADi backend takes 1.4 s (`specs/features/015-rust-develop-model.md`). The core's `root_set` releases the GIL, so rows can be solved in threads.
  - Nothing computes derivatives with respect to well parameters. The stability slope is an implicit-function step with respect to `p_0` only on the CasADi backend, and a central difference on the core.
- **The parameters' physics.**
  - The choke rate is linear in `K_c σ(u)` at a fixed wellhead state (CHK-2), so only the product is identifiable. `K_c` is in m²: the discharge coefficient times the throat area.
  - `h` and the heat capacities enter THM-1 only as `h/C`, so they cannot be calibrated together.
  - TBH is set by `T_r` (THM-3) and lift-gas mixing (THM-5). None of the first version's parameters affects it.
- **v1.0.0's calibration** (paper §6.1).
  - Method: friction factor, heat capacity and choke parameters fitted by nonlinear least squares on part of one real well's data. Then 100 points were simulated, with the measured PDC, choke position and gas fraction as boundary conditions.
  - Errors (paper Table 4), the baseline of the private accuracy check:

    | Output | RMSE | MAPE |
    |---|---|---|
    | PBH | 2.69 bar | 1.31% |
    | PWH | 1.63 bar | 2.57% |
    | TWH | 1.71 K | 0.38% |
    | Liquid rate | 1.58 kg/s | 5.64% |
    | Gas rate | 0.24 kg/s | 5.64% |
- **Real-well data is confidential.** Agents never access it (`specs/constitution.md`). Everything the tests and twin experiments need is synthetic, drawn with `manywells.sampling`. Bjarne runs the real-well check, and only aggregates are published.
- **The validation plan.** `plans/validation-plan.md` (draft, 2026-10-02) plans the private field checks: a reader, a mapping and metrics for the private datasets in a `validation/` workspace member, and a private runner (its V4). Its V6 adds calibrated checks once calibration exists. Step C8 below builds on it rather than on a separate script.

## The problem

Following Kennedy and O'Hagan (2001), take one well with rows k = 1..n. A row is a steady period: a well test, an MPFM average, or a period with sensors only.

- **Control inputs x_k**, known for each row: choke position CHK, PDC (`p_s`), lift gas WGL, `p_r`, `T_r`, `T_s`, and the fluid's `gor` and `wlr`.
- **Calibration parameters θ**: the first version's parameters. They are unknown but the same for every row in the window.
- **Observations y_kj**: any subset of PBH, PWH, TWH and the reservoir mass rate. A missing value is simply absent.
- **The model** y_kj = η_j(x_k, θ) + ε_kj, where η is the well's operating point at x_k under θ, and ε_kj ~ N(0, σ_kj²) is measurement noise.

This is Kennedy and O'Hagan's model with ρ = 1 and no discrepancy term δ, using the simulator itself rather than a Gaussian-process emulator. They drop the emulator themselves when the code is cheap (their §6.1), and here a row costs about 0.15 s on the core. The MAP estimate is

  θ̂ = argmin_θ ½ Σ_k Σ_{j ∈ obs(k)} ((y_kj − η_j(x_k, θ)) / σ_kj)² + ½ Σ_i ((log θ_i − log m_i) / s_i)²

with a log-normal prior (median m_i, log-sd s_i) on each free parameter. Both terms are sums of squares, so a least-squares solver minimizes them directly.

Two cautions from the literature shape the design:

- **θ̂ is a best fit, not the physical value.** A calibrated θ is the best fit under the model (Kennedy and O'Hagan, §4.3). With no discrepancy term, θ absorbs model error, so a θ̂ far from its prior points to model inadequacy rather than to a discovery (Challenor, in the discussion).
- **Wellhead data alone cannot split friction from inflow.** With only wellhead data, friction and inflow productivity act on PWH together, and the prior decides between them (Seman et al. 2020, §2.2; Nikoofard et al. 2017).

## Scope decisions

Bjarne, 2026-10-02:

- **Parameters.** The first version calibrates:
  - the choke coefficient `K_c`;
  - the inflow productivity (`k_l` or `w_l_max`);
  - friction (`f_D`, or the roughness);
  - the heat-transfer coefficient `h`.

  Each one is switchable: the user frees the ones to fit, and the others keep their values. The heat capacities (confounded with `h`), the choke profile, the critical pressure ratio and the slip parameters stay fixed.
- **Joint full-well forward model.** Each row is simulated as an operating point of the whole well, and the objective covers whatever was measured. One code path serves every instrumentation, and a missing sensor just drops terms. Modular fits (choke from wellhead data, tubing from marches anchored at PBH) are not in the first version.
- **Measurement noise only.** No discrepancy term: no extra model-error variance, no offset per well test, and no Gaussian-process δ.
- **MAP point estimate only.** Priors act as physics-based regularization. There is no posterior covariance, Laplace approximation or sampling in the first version.

Proposed alongside these, for sign-off:

- **Per well, over a window.** A calibration covers one well and one set of rows, with θ constant across them. A table with several wells is calibrated well by well. Drift over time is handled by recalibrating on a recent window.
- **Inputs are known.** `p_r`, `T_r`, `T_s` and the fluid fractions are given per row, not calibrated. `p_r` is the main risk: a 1% error in it significantly corrupts a productivity estimate (Nikoofard et al. 2017, Table VII). Calibrating it is the first item after this plan.
- **The Rust core by default.** The calibration calls only `SSDFSimulator(wp, backend=...)`, so it works with both backends.
  - Each row has its own fluid and therefore its own `WellProperties`. The CasADi backend builds once per row and θ, which is fine for cross-checks on small cases and too slow for routine use.

## Design

### Data

A calibration takes a `CalibrationData` for one well: a table with one row per steady period. Units are those at ManyWells' interfaces (bar, K), and columns use the dataset feature names of `docs/datasets.md` where one exists.

| Role | Columns | Missing values |
|---|---|---|
| Inputs | CHK, PDC, WGL; `p_r`, `T_r`, `T_s`, GOR, WLR per row or per well | not allowed |
| Observations | PBH, PWH, TWH; phase rates QOIL, QGAS, QWAT (Sm³/h) | allowed (NaN) |
| Noise | the source of each row's rates (well test or MPFM); optional σ columns that override the defaults | — |

- **One rate observation per row.** When a row's GOR and WLR come from the same measurements as its rates (a well test, an MPFM), the phase split is an input and says nothing about θ. Counting each phase rate would count the same information three times. So the phase rates are converted, with the fluid's standard densities, into one observation: the reservoir mass rate WLIQ + WGAS (WTOT without lift gas). It works for oil and gas wells alike, and vendor studies find total mass rate a reliable tuning quantity (Bikmukhametov and Jäschke 2020, §3.4).
- **TBH is not an observation** in the first version, because no parameter affects it. A measured TBH can be used to set `T_r`.
- **Rows are flowing periods.** CHK > 0, and every row needs an operating point at the starting θ (see the forward model).

### Parameters and priors

Each parameter is a field of a component of `WellProperties`, applied with `dataclasses.replace`. Each is a plain field in `solvers/rust.py`'s `core_well`, so both backends take it.

Every prior is log-normal, which keeps the parameter positive. The fit works in the standardized log, z = (log θ − log m)/s, so a parameter's prior residual is z itself. The table below is a proposal, for sign-off.

**Choke coefficient**
- **Field:** `choke.K_c` (m²); CHK-2, CHK-5, CHK-6.
- **Median:**
  - with a known full-open throat area A_c: C_D·A_c with C_D = 0.6;
  - otherwise 0.12 A, with A the tubing area.
- **Log-sd (95% range):**
  - with A_c known: 0.2, which gives C_D from 0.40 to 0.89;
  - otherwise 0.35, which gives 0.06 A to 0.24 A.
- **Basis:**
  - `K_c` = C_D·A_c.
  - With Simpson's multiplier (CHK-5), tuned C_D is 0.61–0.69 in the lab and 0.47–0.67 in the field (Haug 2012, Table 1).
  - The sharp-edge contraction coefficient is 0.61, and Mwalyepelo and Stanko (2016) find 0.65 best.
  - 0.12 A is the paper's reference value (C_D = 0.6 for a throat of A/5), and 0.06 A to 0.24 A is SMP-5.
  - A valve's Cv curve gives `K_c σ(u)` directly, about 1.70·10⁻⁵ Cv m² for Cv in US gpm/psi^½ (Schüller et al. 2003, Eq. 1); the spec checks the conversion.

**Inflow productivity**
- **Productivity index:** `inflow.k_l` (kg/(s·bar)); INF-2.
  - **Median:** the radial-inflow (Darcy) index from permeability, net pay, viscosity, formation volume factor and skin, when those are known; otherwise a well-test or pressure-transient estimate.
  - **Log-sd (95% range):** 1.15, a factor of 10 either way.
- **Vogel maximum rate:** `inflow.w_l_max` (kg/s); INF-1.
  - **Median:** the user's estimate of the absolute open-flow rate.
  - **Log-sd (95% range):** 1.15.
- **Basis for both:** a well's productivity has no generic value, so the prior carries what the user knows (Ahmed 2006; Guo et al. 2008).

**Friction**
- **Friction factor:** `friction.f_D`; FRIC-1, FRIC-2.
  - **Median:** 0.02.
  - **Log-sd (95% range):** 0.5, which gives 0.0075 to 0.053.
  - **Basis:** Moody gives 0.009 to 0.025 for steel tubing. The upper tail is wider because in a vertical well `f_D` also absorbs holdup error. SMP-3 samples U(0.01, 0.08).
- **Pipe roughness:** `friction.roughness` (m); FRIC-3 to FRIC-6.
  - **Median:** 4.6·10⁻⁵ m, commercial steel (Moody 1944).
  - **Log-sd (95% range):** 1.15, which gives 4.6·10⁻⁶ to 4.6·10⁻⁴ m.
  - **Basis:** the range runs from drawn to corroded tubing. SMP-42 samples LogUniform(1.5·10⁻⁶, 1.5·10⁻⁴).

**Heat-transfer coefficient**
- **Field:** `thermal.h` (W/(m²K)); THM-1.
- **Median:** 15.
- **Log-sd (95% range):** 0.5, which gives 5.6 to 40.
- **Basis:**
  - `h` acts on T − T_a, with T_a the undisturbed geotherm. It is therefore Hasan and Kabir's (2012) effective coefficient U·k_e/(k_e + r_ti·U·T_D), about 5 to 25.
  - Wiktorski et al. (2019) give 9.5 to 22 for U before the formation's transient resistance.
  - SMP-4 samples U(10, 40).

The shape of the choke profile σ(u) is not calibrated. With rows at several openings, a wrong profile shows up as residuals that trend with CHK, and the diagnostics report that.

### Forward model and observations

- **The forward solve.** For a given θ, each row becomes a `WellProperties` (θ applied, plus the row's fluid) and a `BoundaryConditions`. `simulate(bc)` solves it for its operating point: the stable root, or the one with the lowest `p_0` if there are several (SOL-6).
- **Predicted observations.** They come from the root, with the same definitions as the dataset features. One function, extracted from `sample_row`, takes the well, the boundary conditions and the root. The datasets and calibration both call it, and the dataset rows do not change.
- **No operating point.** A row was observed flowing, so a θ under which it has no operating point is inconsistent with it.
  - Following Seman et al.'s extreme barrier, the row's residuals then take a large fixed value, so the solver rejects the step.
  - The fit starts at the prior medians. A row with no operating point there is reported and the fit stops, instead of dropping the row silently.
  - This rule is a spec decision, for sign-off.
- **Cost.** Each objective evaluation is n solves, and a finite-difference Jacobian with p free parameters adds p more evaluations.
  - Example: 50 rows, four parameters and about 20 iterations make about 5,000 solves. At 0.15 s each, that is about 13 minutes on one core, or under 2 minutes on 8 threads. Step C4 measures it.
  - Speed-up option: a solve that tracks the operating point from the previous θ, with a local search on the core's `residual(p_0)` (about 2 ms per call) in place of the full scan. That is new solver machinery, so it comes only with a measured gain (principle 7).

### Noise and the objective

- **The noise model.** Each observation's σ comes from its sensor or source: absolute for pressures and temperatures, relative for the rate. Rates enter as log residuals, so a relative σ applies directly.
- **Defaults.** For sign-off, and overridable per well and per column:

  | Observation | Default σ | Basis |
  |---|---|---|
  | PBH | 0.3 bar | downhole quartz gauges, about 0.02% of full scale, plus drift |
  | PWH | 0.3 bar | wellhead transmitters, 0.02–0.1% of full scale, drift up to 0.1% a year |
  | TWH | 1 K | sensor ±0.3–0.5 K, but an installed bias of several kelvin (Ausen et al. 2017) |
  | Rate, well test | 2.5% | test separator: oil 1–5%, gas 2–5%, at 95% (NFOGM 2005) |
  | Rate, MPFM | 10% | about 10% MAPE (Grimstad et al. 2021; Hotvedt et al. 2022) |
- **The σ's also weight the outputs.** With measurement noise only, small sensor σ's let the pressures dominate, and model error then moves θ. Step C6's misspecified twins measure how much.
- **The solver.** The fit uses `scipy.optimize.least_squares` (trust region, finite-difference Jacobian in z), started from z = 0.
  - **No new package.** scipy is already in `uv.lock` (1.17.1, or 1.18.1 on Python ≥ 3.12), as a requirement of scikit-learn, which `slip.py` uses.
  - **Declared anyway.** `pyproject.toml` does not list scipy, and nothing in `src/` imports it yet. Calibration would be its first direct use, so `pyproject.toml` declares it; otherwise calibration would break if scikit-learn were ever dropped.
- **The result.** It holds θ̂, the residual of every observation, and two diagnostics that need no covariance:
  - the normalized residuals per output, against CHK and against time;
  - each parameter's shift from its prior median, in prior standard deviations. A shift beyond 2 suggests model error rather than a better estimate.

### What the data can identify

What we expect, from the model's structure and the literature. Steps C5 and C6 check it.

| Instrumentation | K_c | Productivity | Friction | h |
|---|---|---|---|---|
| PBH, PWH, TWH and rates | yes | yes, from p_r − PBH against rate | yes, from PBH − PWH against rate | yes |
| PWH, TWH and rates (no downhole gauge) | yes | weakly: confounded with friction through PWH | weakly | yes |
| Wellhead sensors on every row, rates from periodic tests | yes, from the test rows | weakly | weakly | yes |
| PBH, PWH, TWH, no rates | a one-parameter family: fix friction, or add rates on some rows, and the rest follows | same | same | same |

- **Choked flow.** PDC then carries no information, and `K_c` absorbs any error in r_c.
- **Wide-open choke.** With a small pressure drop, sensor drift dominates the choke fit (Ausen et al. 2017).

### Module layout

```
src/manywells/calibration/
  __init__.py      calibrate, CalibrationData, Parameter, CalibrationResult
  data.py          CalibrationData: columns and their roles, units, noise; validated in __post_init__
  parameters.py    Parameter (a component field and its log-normal prior), applying θ to a WellProperties,
                   the default priors with their sources
  objective.py     the predicted observations of a row; the residual vector (noise-scaled observation
                   residuals and prior residuals); the barrier
  fit.py           calibrate(): rows solved in threads, scipy's least_squares, the result
  synthetic.py     synthetic calibration data: a well's rows simulated at known parameters, with seeded
                   noise and an instrumentation mask (for the tests, the twin study and the example)
```

- **Layering.** The module sits on top of the simulator (`specs/architecture.md`). It imports nothing from `scripts/`, holds no model equation, and calls the shared feature function in `datasets/rows.py`.
- **Retiring the old functions.** `choke_cal.py` and `inflow_cal.py` are retired. Calling `calibrate` with only `K_c`, or only the productivity, free does their job from measured data, without the unmeasured upstream state. The CHANGELOG lists the break with an old→new snippet.

## Steps

### Step C1 — Write the calibration spec

- **Goal.** A decided contract to derive the code from.
- **Work.**
  - **`specs/calibration.md`.** It sits outside `specs/model/` because it is not physics, like `specs/sampling.md`, and its IDs are CAL-n. It covers:
    - the data and the roles of its columns, and the rate observation;
    - the parameters, their transform and the default priors with their sources;
    - the predicted observations, the noise model and the objective;
    - the barrier, the stopping rule, and the result with its diagnostics.

    Add the CAL namespace to `tests/spec_parse.py`, so the traceability test checks the `# spec: CAL-n` tags as it checks SMP.
  - **`specs/features/NNN-calibration.md`**, at the next free number (016 is Step 10's Joule–Thomson). It covers:
    - motivation;
    - the delta: a new spec and no change to `specs/model/`, so the `v1.0.0` configuration is untouched;
    - acceptance: Step C5's recovery tests and Step C6's twin study, with their tolerances;
    - out of scope: After this plan.
  - **The calibration contract in `specs/architecture.md`:** the module row and the interface.
- **Done when.** Bjarne has signed off the spec, including the priors and noise defaults, and approved the feature spec.
- **Who.** Agent drafts; Bjarne decides.

### Step C2 — Data and observations

- **Work.**
  - `CalibrationData`, validated in `__post_init__`: units, column roles, no missing inputs, at least one observation per row, positive rates.
  - The conversion of the phase rates into the reservoir mass rate.
  - The feature function extracted from `sample_row`, leaving the dataset rows unchanged; the sampling tests pin them.
  - The predicted observations of a root.
- **Tests.** Named after the modules, as `docs/testing.md` describes.
  - A row's predicted observations equal the dataset features of the same root.
  - Each kind of malformed input is rejected with a clear `ValueError`.
- **Done when.** The tests pass, and `tests/test_sampling.py` passes unchanged.

### Step C3 — Parameters and priors

- **Work.**
  - `Parameter`, and applying θ to the frozen inputs.
  - The default priors with their sources, and the choke prior from a throat area when one is given.
- **Tests.**
  - Applying θ changes only the named field, and the caller's inputs are not mutated.
  - The Rust core receives the value (`core_well`).
  - The defaults match the spec's table.
- **Can run in parallel with C2.**

### Step C4 — The objective and its cost

- **Work.**
  - The residual vector for a θ, with the rows solved in threads on the core (it releases the GIL) and one by one on the CasADi backend. The barrier.
  - Measurements on synthetic wells:
    - the time per evaluation at n = 10, 50 and 200 rows, on both backends;
    - the accuracy of the finite-difference Jacobian, comparing steps h and h/2 in z. The root's tolerance has to stay well below the change a step makes.
- **Done when.** The measurements are in the feature spec, and the Jacobian is smooth on the twins. If the cost is too high for interactive use, the agent proposes the tracked solve with its measured gain, under principle 7.

### Step C5 — The fit, and recovering known parameters

- **Goal.** A fit that, on wells simulated with known parameters, gets back to those parameters, or close to them where the data cannot identify them.
- **Work.**
  - `calibrate(wp, data, free=[...], priors=..., backend='rust')` returns a `CalibrationResult`: θ̂, the calibrated `WellProperties`, residuals, diagnostics and solver status.
  - `least_squares` starts from z = 0.
  - Failures raise a `CalibrationError` and are reported through logging, not `print` (`plans/improvements.md` §2.8).
  - **Synthetic wells** (`synthetic.py`). For a well and known parameters θ*, simulate rows at operating points spread over CHK, WGL and PDC. Then add noise at the default σ's, from a seeded generator (constitution, Determinism), and mask the observations by instrumentation:
    - full;
    - no downhole gauge;
    - wellhead sensors on every row, with rates on one row in five;
    - pressures only, with rates on two rows;
    - 20% of observations missing at random.
- **Recovery tests.** Slow tests in `tests/`, on a few seeded wells.
  - **Setup.**
    - Each well has about 10 rows on a coarse grid, on the Rust core.
    - θ* is placed one to two prior standard deviations from the medians, so the fit has to move.
    - Errors are measured in prior standard deviations: |z(θ̂) − z(θ*)| per parameter, and its norm over the free parameters.
  1. **Identification.** With noise-free rows, full instrumentation and all four parameters free, θ̂ is within a tight tolerance of θ*. The only error left is the prior's pull, which the rows make small.
  2. **Getting close.** With noisy rows, in each instrumentation pattern:
     - the distance to θ* is smaller after the fit than at the prior medians;
     - each parameter the pattern identifies (table above) is within a tolerance of θ*.

     Where the data fix only a combination, such as friction and productivity without a downhole gauge, the fit can only move along that ridge. It still moves towards θ*: in a linearized model with noise-free rows, the MAP is the point of the ridge nearest the prior median, and θ* lies on the ridge, so the distance cannot grow. Noise can make it grow a little, which the tolerance allows for.
- **Done when.**
  - The recovery tests pass at the tolerances Bjarne set in Step C1.
  - The fit is deterministic: the same inputs and seed give the same θ̂.

### Step C6 — The twin study

- **Goal.** Run Step C5's recovery checks across many sampled wells. Find the wells where the fit fails, and measure what a wrong model does to θ̂.
- **Work.**
  1. **Twins.**
     - Wells from `manywells.sampling`, on `develop`'s model.
     - θ* drawn from the priors.
     - 10 to 100 rows per well.
     - Each of Step C5's instrumentation patterns.
  2. **Recovery statistics**, per pattern and parameter:
     - the distribution of |z(θ̂) − z(θ*)|, and the share of wells where the fit got closer to θ* than the prior medians are;
     - how the error falls as rows are added (10, 40, 160 rows) and as the noise shrinks;
     - the RMS of the held-out rows' normalized residuals;
     - every well where the fit failed, with the reason.

     The identifiability table above is confirmed or corrected.
  3. **Misspecified twins.** Generate data from a model the calibration lacks, and record θ̂'s bias and the held-out error:
     - Chen friction, calibrated with a fixed `f_D`;
     - a Simpson choke, calibrated as a Bernoulli one;
     - a different choke profile;
     - slip parameters off by 10%;
     - `develop`'s energy terms, calibrated without them.

     These are the evidence for the discrepancy decision under After this plan. They have no pass/fail.
  4. **Backends.** On a few small twins, the core and the CasADi backend give the same θ̂ within a tolerance.
- **Output.** `scripts/calibration/twins.py`, research code like `scripts/verification/`, with its results in the feature spec.
- **Done when.** The share of wells that meet Step C5's criteria, in each pattern, is at least what Bjarne set in Step C1, and the results are recorded.
- **Who.** Agent runs; Bjarne sets the tolerances.

### Step C7 — Retire, document, example

- **Work.**
  - Remove `choke_cal.py`, `inflow_cal.py` and their tests, with a CHANGELOG entry.
  - Add `scripts/sim_examples/calibrate_well.py`, on a synthetic well; `tests/test_examples.py` runs it headless.
  - Write `docs/calibration.md`: preparing the data, choosing free parameters and priors, reading the diagnostics, and the identifiability table.
  - Update the layout line in `AGENTS.md`, and declare `scipy` in `pyproject.toml`.
  - `scripts/wellbore_cal.py` is Bjarne's untracked file. Freeing the roughness supersedes it, and he decides whether to delete it.
- **Done when.** The full suite passes, the example included.

### Step C8 — Calibrated real-well checks (private)

- **Work.** The agent connects `calibrate` to the private checks:
  - the v2 plan's real-well accuracy check (paper §6.1);
  - the calibrated variants of `plans/validation-plan.md`, V6.

  Each one:
  - reuses the validation plan's reader, mapping, metrics and private runner (its V4);
  - calibrates on a training part only and evaluates on the held-out part, since nothing is fitted to the validation targets (validation plan, principle 6);
  - reports aggregates only: RMSE and MAPE for PBH, PWH, TWH and the liquid and gas rates, next to v1.0.0's Table 4.

  The pipeline is tested on synthetic files in the private datasets' format. Bjarne runs it.
  - If the validation plan is not approved, the same check is a script under `scripts/calibration/` that reads a local directory, which is never committed.
- **Done when.** The aggregate results are recorded, and Bjarne has set the acceptance bound relative to v1.0.0's errors (`specs/goals.md`, Decided elsewhere).
- **Who.** Agent writes; Bjarne runs it and decides.

## Sequencing

```
C1 ─┬─▶ C2 ─┬─▶ C4 ─▶ C5 ─▶ C6 ─┬─▶ C7
    └─▶ C3 ─┘                   └─▶ C8
```

C2 to C7 touch `calibration/`, the feature function in `datasets/rows.py`, tests, scripts and docs, but not the model or the solvers. So they do not conflict with model work that runs at the same time (Step 10). The foundation plan defers calibration until after Step 10; whether C1 starts earlier is Bjarne's call. C8 also waits for the validation plan's V4.

## After this plan

In rough order of value:

- **Posterior uncertainty.** A Laplace approximation from the fit's own Jacobian, (JᵀJ)⁻¹ in z. With it comes an identifiability report per well (posterior-to-prior standard deviation, correlations), checked against an MCMC sampler on twins.
- **Model discrepancy.** Decided from C6's misspecified twins. Candidates:
  - an extra model-error variance per output;
  - a shared offset per well test, since errors within one test are correlated and MPFM errors are "often systematic" (NFOGM 2005);
  - for data-rich wells, a Gaussian-process δ restricted so that it does not absorb θ (Plumlee 2017; Gu and Wang 2018; Brynjarsdóttir and O'Hagan 2014).
- **Uncertain inputs** as parameters with priors: `p_r` first, then the fluid fractions and `T_r`.
- **Change over time.** Sequential recalibration that carries the posterior forward as the next prior, which needs the Laplace step, and parameters that drift (Hotvedt et al. 2022).
- **Many wells.** Hierarchical priors that pool information across similar wells (Notz et al., in the discussion of Kennedy and O'Hagan; Sandnes et al. 2021).
- **More parameters.** The choke profile's exponent, r_c, the slip parameters, and friction per section.
- **Speed.** A tracked operating-point solve on the core, or sensitivities from the implicit-function theorem with the parameters as symbols of the CasADi system. The second changes the `System` contract, as in `plans/improvements.md` §4.5. Each needs a measured gain (principle 7).
- **Modular fits** that cut a suspect module's influence on the others (Liu, Bayarri and Berger 2009), if C6 shows contamination.
- **Data preparation.** Detecting steady periods and averaging MPFM data, as Nemoto et al. (2023) do for well tests.

## Needs Bjarne's sign-off

- `specs/calibration.md`: the priors, the noise defaults, the rate observation, the barrier and the stopping rule.
- The feature spec, and the tolerances of the recovery tests (C5) and the twin study (C6).
- The calibration contract in `specs/architecture.md`.
- Retiring `choke_cal.py` and `inflow_cal.py` (an API break), and declaring `scipy`.
- Any solver machinery from Step C4, with its measured gain.
- The accuracy bound of the private check (Step C8).

## Open decisions

- Whether C1 starts before Step 10 is done.
- Whether `p_r` is a free parameter already in the first version. Without PBH it is confounded with the productivity.
- The default σ's. With measurement noise only, they also weight the outputs against each other and against the priors.

## References

In `manywells-papers/papers/`:
- Kennedy and O'Hagan (2001), with the discussion.
- Seman et al. (2020).
- Ausen et al. (2017).
- Nemoto et al. (2023).
- Nikoofard et al. (2017).
- Bikmukhametov and Jäschke (2020).
- Grimstad et al. (2025), §6.1.
- Grimstad et al. (2021).
- Hotvedt et al. (2022).
- Sandnes et al. (2021).
- Haug (2012).
- Schüller et al. (2003).
- Mwalyepelo and Stanko (2016).
- Wiktorski et al. (2019).
- Hasan and Kabir (2012).
- Moody (1944).
- Guo et al. (2008).
- Ahmed (2006).

Not in the folder yet:
- Brynjarsdóttir and O'Hagan (2014), "Learning about physical parameters: the importance of model discrepancy", *Inverse Problems* 30, 114007.
- Higdon, Kennedy, Cavendish, Cafeo and Ryne (2004), "Combining field data and computer simulations for calibration and prediction", *SIAM J. Sci. Comput.* 26(2), 448–466.
- Liu, Bayarri and Berger (2009), "Modularization in Bayesian analysis, with emphasis on analysis of computer models", *Bayesian Analysis* 4(1), 119–150.
- Tuo and Wu (2015), "Efficient calibration for imperfect computer models", *Annals of Statistics* 43(6), 2331–2352.
- Plumlee (2017), "Bayesian calibration of inexact computer models", *JASA* 112, 1274–1285.
- Gu and Wang (2018), "Scaled Gaussian stochastic process for computer model calibration and prediction", *SIAM/ASA J. Uncertainty Quantification*.
- Hotvedt, Grimstad, Ljungquist and Imsland (2022), "On gray-box modeling for virtual flow metering", *Control Engineering Practice* 118, 104974. MAP with physical priors on a mechanistic choke model.
- Lorentzen, Nævdal and Lage (2003), *International Journal of Multiphase Flow* 29(8), 1283–1309. Ensemble Kalman filter tuning of a drift-flux well model.
- NFOGM, *Handbook of Multiphase Flow Metering*, Rev. 2 (2005).
