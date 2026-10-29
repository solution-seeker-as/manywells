# 020 · A scale on the profile parameter, free in calibration

*Feature spec, 2026-10-29. Status: approved by Bjarne on 2026-10-29 (Approval, at the end); implemented on the branch `profile-scale`. The implemented feature's measurements await sign-off.*

## Decisions (Bjarne, 2026-10-29)

- **The formulation.** The profile parameter's departure from 1 is scaled: $C_0^\ast = 1 + \kappa_0(C_0 - 1)$, with $\kappa_0 \ge 0$. $\kappa_0 = 1$ is the slip model's own $C_0$, and $\kappa_0 = 0$ is $C_0 = 1$: the scale moves $C_0$ towards no slip but never past it.
- **One scale for every regime** (Design choice), not one per regime.
- **The name** is `SlipModel.profile_scale`, symbol $\kappa_0$.
- **Both classifiers** take it: the five regimes (SLIP-15) and the fitted classifier (SLIP-2).
- **The default** is 1, so `develop`'s results do not change.
- **The drift velocity** stays as it is.
- **In calibration,** `profile_scale` is a free parameter, with the prior of CAL-6 below: a median of 0.5 and $s = 0.7$.

## Motivation

- **Friction is the only free parameter that can absorb a holdup error.**
  - In a calibration (`specs/calibration.md`), the tubing's pressure drop is gravity, friction and acceleration.
  - Gravity, the largest term in an oil well, depends on the holdup, which the slip law sets. But CAL-5 holds the slip parameters fixed.
  - CAL-6 says as much: the prior of `f_D` "is wider because in a vertical well $f_D$ also absorbs holdup error".
- **A friction factor that absorbs a holdup error loses its meaning and extrapolates wrongly.**
  - Friction grows with the mixture velocity squared. A holdup error has no such dependence, and may even fall with velocity as the flow approaches homogeneous.
  - A calibrated `f_D` or `roughness` then fits the rates it was calibrated on, but is not a property of the tubing, and errs at other rates.
- **$C_0$ is the least certain constant of the slip law.**
  - Published values for upward bubbly and slug flow lie between about 1.0 and 1.2.
  - Zuber and Findlay (1965) give 1.2 for fully developed flow in round tubes. Ishii (1977) gives $1.2 - 0.2\sqrt{\rho_g/\rho_l}$, which falls towards 1.1 at high pressure.
  - Pipes as large as a well's tubing give lower values (Hibiki and Ishii 2003). Dispersed bubbly flow is treated as homogeneous, $C_0 = 1$, in mechanistic models (Ansari et al. 1994).
  - Hasan et al.'s Table 1, which SLIP-15 uses, has 1.2 in bubbly, dispersed bubbly and slug flow: the top of that range.
- **Why one scale.** A single parameter, which a calibration can identify, moves every regime's $C_0$ within the published range. Holding it at no slip or above keeps it physical.

## Design choice: one scale for every regime

Each regime has its own $C_0$: Table 1 has 1.0 for annular, 1.15 for churn, and 1.2 for dispersed bubbly, slug and bubbly flow; the fitted classifier has 1.0, 1.175 and 1.2. One scale, with one prior, acts on all of them:

1. **Scaling the blend scales each regime.** $C_0 = \sum_r p_r C_{0,r}$ with $\sum_r p_r = 1$ (SLIP-2, SLIP-15), so
   $$1 + \kappa_0(C_0 - 1) = \sum_r p_r\left(1 + \kappa_0(C_{0,r} - 1)\right).$$
   Every regime's departure from 1 is multiplied by the same $\kappa_0$, and their ratios are kept. Annular flow, with $C_{0,r} = 1$, does not move.
2. **The prior on $\kappa_0$ is a prior on each regime's $C_0$, fully correlated.**
   - In bubbly, dispersed bubbly and slug flow, $C_0^\ast = 1 + 0.2\kappa_0$; in churn flow, $1 + 0.15\kappa_0$.
   - Those three share one value and fill most of an oil well's tubing, so in practice $\kappa_0$ moves that value, and the churn value follows in proportion.
   - The assumption is that the regimes' $C_0$ are in error by a common factor.
3. **The common factor has a physical basis.** Ishii (1977) gives one $C_0$ for bubbly, slug and churn flow in round tubes, $1.2 - 0.2\sqrt{\rho_g/\rho_l}$.
   - Its density-ratio term multiplies the departure from 1 by the same factor, $1 - \sqrt{\rho_g/\rho_l}$, in all three.
   - The large-pipe values of Hibiki and Ishii (2003) are below 1.2 in every regime they cover.
   - $\kappa_0$ stands for such a factor, which Table 1's constants leave out.
4. **One parameter per regime would not be identified.**
   - A calibration's observations (CAL-2) see the tubing's holdup through one pressure drop and one temperature per row, while the tubing passes through several regimes at once.
   - Three free $C_0$ would be confounded with each other, and with friction. Their priors, not the data, would set them.
5. **What one scale cannot express** is an error that changes with the velocity within a regime, such as dispersed bubbly flow becoming homogeneous at high velocity. That is a change to SLIP-15 itself, not a calibration parameter (Out of scope).

## Delta

### `specs/model/slip.md`

**SLIP-18 · Scaled profile parameter.** The profile parameter of SLIP-2 or SLIP-15 is scaled about 1:

$$C_0^\ast = 1 + \kappa_0\thinspace(C_0 - 1), \qquad \kappa_0 \ge 0,$$

and $C_0^\ast$ takes $C_0$'s place in SLIP-1. $v_\infty$ is unchanged.
- $\kappa_0 = 1$ (the default) is the classifiers' own values, exactly.
- $\kappa_0 = 0$ gives $C_0^\ast = 1$ at every point.
- For $0 \le \kappa_0 \le 1$, $C_0^\ast$ lies between 1 and $C_0$.
- In annular flow, where both classifiers have $C_0 = 1$, it changes nothing.

The rest of `slip.md`:

- **Interface.** `SlipModel.profile_scale` ($\kappa_0$, default 1), validated in `__post_init__`: finite and $\ge 0$. `identify_parameters` returns $C_0^\ast$.
- **Options.** SLIP-18 belongs to both options' rows. At $\kappa_0 = 1$ it is the identity.
- **Safeguards.** None new. With `regime_model='hasan'`, $C_0^\ast$ still does not depend on $\alpha$, so the slip law keeps its one root (Open question).

### `specs/calibration.md`

- **CAL-5.** A new row: `profile_scale` | `slip.profile_scale` | any `SlipModel` | SLIP-18. The sentence "the slip parameters stay fixed" becomes "the other slip parameters stay fixed".
- **CAL-6.** A new row, proposed:

| Name | Median $m$ | $s$ (95% range) | Basis |
|---|---|---|---|
| `profile_scale` | 0.5 | 0.7 (0.13–2.0) | the middle of the published range of $C_0$ for upward bubbly and slug flow, 1.0 to 1.2 (Motivation) |

Through SLIP-18 (Design choice, item 2), the prior is one on each regime's $C_0$:

| Prior | Bubbly, dispersed bubbly and slug | Churn | Annular |
|---|---|---|---|
| median 0.5, $s = 0.7$ (proposed) | 1.1 (1.025–1.39) | 1.075 (1.019–1.30) | 1 |
| median 0.5, $s = 1.0$ | 1.1 (1.014–1.71) | 1.075 (1.011–1.53) | 1 |
| median 1, $s = 0.5$ | 1.2 (1.075–1.53) | 1.15 (1.056–1.40) | 1 |

- **The fitted classifier.** Its bubbly flow's 1.2 moves as in the first column, and its slug-churn flow's 1.175 becomes $1 + 0.175\kappa_0$.
- **Median 0.5** is the middle of the published range.
- **A median of 1** centres the prior on Table 1, the model's default. That is the top of the published range, so a calibration would pay to move to its middle.
- **The width sets the cost near no slip.** $\kappa_0 = 0.1$, $C_0^\ast = 1.02$, is 2.3 prior standard deviations below a median of 0.5 at $s = 0.7$, and 1.6 at $s = 1.0$.

### Other files

- **`specs/model/nomenclature.md`:** $\kappa_0$, `profile_scale`.
- **`specs/model/README.md`:** SLIP-18 in the slip row of both configurations' column for `develop`. The `v1.0.0` configuration takes no new ID.
- **`specs/architecture.md`:** the `SlipModel` row of the interface table.
- **`docs/calibration.md`:** the new parameter, and when to free it.
  - Free it when the tubing's pressure drop is in error and friction calibrates below its prior's range. Hold friction at its prior then, since the two are hard to tell apart from PBH − PWH alone.
  - Report its posterior correlation with the friction parameter.

### Code

| Where | Python | Rust |
|---|---|---|
| slip | the field and its check; SLIP-18 in `identify_parameters`, after SLIP-2 or SLIP-15 | `profile_scale` in `Slip`, applied in `parameters`; `_component` for the vectors |
| harness | `solvers/rust.py` passes the field. `configurations.check` already flags any `SlipModel` that is not `v1_well`'s, so the `v1.0.0` configuration takes $\kappa_0 = 1$ only. | `lib.rs`: `Well::new`'s argument; test wells with $\kappa_0 \ne 1$ in `input.rs` |
| calibration | `FIELDS['profile_scale'] = ('slip', 'profile_scale', SlipModel)`, `default_prior` | – |

## Off in the `v1.0.0` configuration

$\kappa_0 = 1$ is the identity, and `configurations.check` flags any other value in the `v1.0.0` configuration. The row vectors and both verifier reports do not change.

## Acceptance

1. **The `v1.0.0` configuration is unchanged.** The row vectors pass. Both verifier candidates report PASS with 0 failures, 0 expected failures and a stable-root rate of 100%.
2. **Vectors.** A table for SLIP-18, checked in both backends by `tests/test_spec_vectors.py`: $C_0^\ast$ and $v_\infty$ at $\kappa_0$ = 0, 0.5, 1 and 1.5, in each classifier and in each of the five regimes.
3. **`tests/test_slip.py`** and `slip.rs`:
   - $\kappa_0 = 1$ gives the unscaled $C_0$ to the bit.
   - $\kappa_0 = 0$ gives $C_0^\ast = 1$.
   - $C_0^\ast$ is monotone in $\kappa_0$, and $1 \le C_0^\ast \le C_0$ on $[0, 1]$, on random states across the sampler's range.
   - $v_\infty$ does not depend on $\kappa_0$.
   - The field is validated.
4. **`tests/test_calibration.py`.** `profile_scale` applies to any `SlipModel`, and its default prior is CAL-6's row.
5. **Backend comparison** (`tests/backend_cases.py`):
   - feature `020`, with an overlay $\kappa_0 = 0.5$;
   - matrix entries on a vertical and a deviated well;
   - a comparison-set group.
   The rows agree in every configuration, as for the other options.
6. **The Rust core.** `cargo test --no-default-features` passes.
7. **Measurements** (`plans/evidence/`; sampled wells only):
   - **Sampled wells.** The option's effect at $\kappa_0$ = 0, 0.5 and 1, on both backends: operating points found, and the change in PBH, PWH and rate. A fall in operating points found is a solver finding.
   - **Twins.** The twin study of 017 (`scripts/calibration/twins.py`), with `profile_scale` free besides the four parameters, and data generated at $z^\ast$ drawn from all five priors. Per instrumentation:
     - the share of fits that converge;
     - the recovery error of each parameter in $z$;
     - the correlation of the errors of `profile_scale` and `roughness`, which measures how far the two can be told apart.
   - **Misspecified twins.** Data generated with $\kappa_0 \ne 1$, calibrated with `roughness` free and `profile_scale` held at 1, against `profile_scale` free and `roughness` held at its prior. This shows where a holdup error goes when the slip is fixed.
   - The tolerances of the twin checks are proposed with the measurements, for sign-off, as in 017.
8. **The suites.** The full test suite passes.

Done on the branch, 2026-10-29, on `develop` at `c12fd51` with the feature (Python 3.14, casadi 3.8.1):

- **Items 1 to 6 hold.**
  - Both verifier candidates report PASS, with 0 failures, 0 expected failures, 141 cases and a stable-root rate of 100%.
  - The SLIP-18 vectors are two tables, `SLIP-18` (the five regimes) and `SLIP-18 (fitted)`, since a develop table's inputs are numbers. They have 20 and 12 rows and are checked in both backends.
  - The backend comparison has the overlays `profile scale` ($\kappa_0 = 0.5$) and `no profile slip` ($\kappa_0 = 0$). The matrix has the entries `develop+profile scale`, `develop+no profile slip`, `W1+deviated+five regimes+profile scale` and `W2+L-shaped+no profile slip`. The comparison-set group `develop` takes both overlays.
  - The Rust core has two new test wells, `w1_profile_scale` and `w2_profile_scale`. `cargo test --no-default-features` passes 50 tests.
- **Item 8.** 940 passed and 1 skipped. Its one warning is the known extra root at `v1.0.0+L-shaped#0`, from 016.
- **Item 7** is under Measurements.

## Measurements

The scripts are in `plans/evidence/` (`README.md` there). The wells are all synthetic: `develop`'s sampler, seed 2026.

**Sampled wells** (`profile_scale_wells.py`). The 263 cases of `five_regime_default.py`, with $\kappa_0$ = 1, 0.5 and 0, on the Rust core, and the 90 cases of the first 30 wells on the CasADi backend too:

| | Operating points, core (of 263) | CasADi (of 90) | PBH change, median (10th percentile, largest) | PWH change, median (largest) | Rate change, median (largest) |
|---|---|---|---|---|---|
| $\kappa_0 = 1$ | 257 | 88 | – | – | – |
| $\kappa_0 = 0.5$ | 259 | 89 | −0.09 bar (−2.2, −6.8) | +0.14 bar (+6.0) | +0.3% (+38%) |
| $\kappa_0 = 0$ | 261 | 90 | −0.19 bar (−4.8, −15.3) | +0.28 bar (+13.3) | +0.6% (+75%) |

*The changes are from the core, against $\kappa_0 = 1$, over the cases with an operating point in both.*

- **No operating point is lost.** The gains are cases whose column becomes light enough to flow.
- **A quarter of the cases do not change at all.** They flow in annular flow, where $C_0 = 1$, or without free gas.
- **Cost.** The time per case does not change on the core. On the CasADi backend the total falls from 698 s to 426 s at $\kappa_0 = 0$.
- **Solver finding: the core misses a root.**
  - At $\kappa_0$ = 0.5 and 0, the CasADi search finds a second stable root in one case, well 14 at its second sampled point. It is on a gas-rich branch, with mean void fraction 0.92 and PWH 129 bar against 89 bar.
  - The core's search misses that root, so the backends pick different operating points there: 0.77 bar apart in PBH.
  - At $\kappa_0 = 1$ the model has the one root, and the backends agree to $10^{-8}$ bar in all 88 cases.
  - This is the kind of miss already known at `develop#17`, where the core misses two roots (`plans/evidence/README.md`). Not changed.

**Twins** (`profile_scale_twins.py --wells 20 --detail 8`). The twin study's wells 0 to 19; well 11 has no operating point and is left out. All five parameters are free, and $z^\ast$ is drawn from their priors. Errors $\lvert\hat z - z^\ast\rvert$, median and largest:

| Instrumentation | Fits | Converged | Closer than the medians | `K_c` | `w_l_max` | `roughness` | `h` | `profile_scale` | Error correlation, `profile_scale` with `roughness` | Meet 017's twin criteria |
|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 19 | 100% | 95% | 0.02, 0.83 | 0.00, 0.49 | 0.05, 0.37 | 0.04, 1.87 | 0.33, 2.26 | +0.23 | 95% |
| `no_downhole` | 19 | 100% | 95% | 0.01, 0.83 | 0.02, 0.49 | 0.10, 0.99 | 0.02, 1.87 | 0.58, 1.73 | +0.28 | 89% |
| `periodic_tests` | 19 | 100% | 100% | 0.02, 0.08 | 0.01, 0.05 | 0.06, 0.97 | 0.02, 0.09 | 0.18, 1.73 | −0.22 | 100% |
| `pressures_only` | 19 | 100% | 100% | 0.04, 0.12 | 0.01, 0.04 | 0.18, 0.62 | 0.06, 0.13 | 0.11, 1.73 | +0.20 | 100% |
| `random_missing` | 19 | 100% | 95% | 0.02, 0.83 | 0.01, 0.47 | 0.09, 0.37 | 0.03, 1.87 | 0.23, 2.26 | +0.27 | 95% |

017's twin criteria: the fit converges, ends closer to $z^\ast$ than the medians where $z^\ast$ is more than 0.5 from them, has `K_c`, `w_l_max` and `h` within 0.5 of $z^\ast$, and has every RMS of scaled residuals below 2.

- **Where the data depend on $\kappa_0$, it is recovered.** On wells 2, 5, 6, 8, 13 and 16 the error is 0.0 to 0.6 in most instrumentations; well 16 recovers $z^\ast = -2.5$ to within 0.7.
- **On a third of these wells the data do not depend on $\kappa_0$.** On wells 4, 9, 10, 12, 14 and 15, $\hat z$ stays at 0, the prior median, so the error is $\lvert z^\ast\rvert$. These sampled wells are gas-rich and flow mostly in annular flow (019, Validity ranges), where $C_0 = 1$. A parameter the data do not determine stays at its prior, as CAL-5 intends.
- **`profile_scale` and `roughness` are told apart on these wells.** Their errors correlate by −0.22 to +0.28.
- **Well 1 (L-shaped) ends in a false minimum.**
  - In `full` and `random_missing` the fit stops at $z_\text{profile\_scale} = -1.00$, one of CAL-9's starts, with `h` 1.9 and `K_c` 0.8 from $z^\ast$. The truth is $+1.26$.
  - Along $\kappa_0$ alone, with the other parameters at the truth, the cost is smooth, has its minimum near $z^\ast$, and puts no row at CAL-9's barrier.
  - So the minimum is in the joint space, as 017's well 3 was for `roughness`.
  - It is behind the failures in three instrumentations. `no_downhole`'s other failure is well 6, which without the gauge ends at $+0.45$ against $z^\ast = -0.70$.

**Misspecified twins** (wells 0 to 7). The truth has `roughness` at its median and `profile_scale` drawn. Each fit has 20 rows at choke positions 0.2–0.5. Each is checked on 10 held-out rows at 0.7–1.0, higher rates than the fit saw:

| Well | $z^\ast$ `profile_scale` | Friction absorbs it: $z$ `roughness`, held-out RMS | Scale free: $z$ `profile_scale`, held-out RMS | Both free: held-out RMS |
|---|---|---|---|---|
| 0 | −1.13 | −0.06, 0.93 | −0.09, 0.93 | 0.93 |
| 1 | +1.26 | +0.00, 4.54 | −1.00, 7.82 (did not converge) | 7.82 (did not converge) |
| 2 | −0.04 | −0.15, 1.63 | +0.04, 1.12 | 1.11 |
| 3 | −0.06 | −0.45, 1.40 | +0.10, 1.06 | 1.07 |
| 4 | −0.11 | +0.01, 1.08 | +0.00, 1.07 | 1.08 |
| 5 | +1.23 | +0.64, 1.04 | +1.22, 0.88 | 0.86 |
| 6 | −0.70 | −3.76, 1.87 | −0.68, 1.04 | 1.68 |
| 7 | +0.93 | +0.07, 1.04 | −0.53, 1.01 | 1.03 |

- **With the slip fixed, friction absorbs a holdup error.** On well 6 the roughness falls 3.8 prior standard deviations to fit the low-rate rows. At higher rates that fit errs: held-out RMS 1.87, against 1.04 with the scale free. Wells 2, 3 and 5 show the same, less strongly. This is the motivation, measured on twins.
- **Where the data do not depend on $\kappa_0$ (wells 0, 4, 7)**, all three fits are equal.
- **Well 1's false minimum** is the exception.

## Sign-off

For Bjarne, on the implemented feature:

- **The changed results** where $\kappa_0 \ne 1$ (Measurements, Sampled wells). The default $\kappa_0 = 1$ changes no result of `develop`.
- **The solver finding**: the core misses a stable root that the CasADi search finds, in 1 of 90 cases at $\kappa_0 \le 0.5$.
- **The twin checks' tolerances.** Proposed:
  - 017's criteria stay as they are for fits with `profile_scale` free.
  - The criteria hold in 95%, 89%, 100%, 100% and 95% of the fits. `no_downhole` is below 95% by well 1's false minimum and by well 6.
  - No recovery tolerance on `profile_scale` itself, since on a third of the sampled wells the data do not determine it.
  - Well 1's false minimum is a finding for `plans/improvements.md`, as 017's start search was.

## Out of scope

- **A scale on the drift velocity**, or one scale per regime. A later option, if one $\kappa_0$ is not enough.
- **Changing the default.** $\kappa_0 = 1$ keeps `develop`'s model. Whether to change $C_0$ itself is a separate decision, made on evidence.
- **Fitting $\kappa_0$ outside the calibration**, for instance in the sampler. SMP's ranges are unchanged.

## Approval (Bjarne, 2026-10-29)

- **The spec**, as written, before the implementation starts: one scale for every regime, the name, both classifiers, the default $\kappa_0 = 1$ and the drift velocity unchanged. Approved.
- **The edits** to `specs/model/slip.md` (SLIP-18) and to `specs/calibration.md` (CAL-5 and CAL-6). Signed off.
- **The prior** of CAL-6: a median of 0.5 and $s = 0.7$. Chosen of the three in CAL-6's table.

The implemented feature's measurements (Acceptance, item 7), with the twin checks' tolerances, come back for sign-off.

## Sources

- Zuber and Findlay (1965), "Average volumetric concentration in two-phase flow systems", *Journal of Heat Transfer* 87, 453–468. Not consulted.
- Ishii (1977), "One-dimensional drift-flux model and constitutive equations for relative motion between phases in various two-phase flow regimes", ANL-77-47: $C_0$ for round tubes, as tabulated by Ishii et al. (2007).
- Hibiki and Ishii (2003), "One-dimensional drift–flux model for two-phase flow in a large diameter pipe", *International Journal of Heat and Mass Transfer* 46, 1773–1790: as tabulated by Ishii et al. (2007).
- Ishii, Paranjape, Sawant and Ozar (2007), "Void fraction in large diameter pipes: literature search for existing database and correlations", Purdue PU/NE-07-06 for the US NRC (ML081280202).
- Ansari, Sylvester, Sarica, Shoham and Brill (1994), "A comprehensive mechanistic model for upward two-phase flow in wellbores", *SPE Production & Facilities* 9, 143–151: dispersed bubbly flow without slip. Not consulted.
- Hasan, Kabir and Sayarpour (2010), Table 1, as for SLIP-15.
