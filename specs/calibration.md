# Calibration

*Step C1 of `plans/calibration-plan.md`. Owner: Bjarne Grimstad. Status: decided, 2026-10-02. The scope decisions (the four parameters, a joint full-well forward model, measurement noise only, a MAP point estimate) are Bjarne's, and he signed off the rest, the priors, the noise defaults, the start search and the tolerances of the checks included, on the same day. Implemented in `manywells.calibration` (feature spec `specs/features/017-calibration.md`). Any change to this file needs his sign-off.*

How ManyWells calibrates a well to its production data: rates from well tests or multiphase flow meters (MPFMs), and pressures and temperatures from whatever sensors the well has, with missing values. This is not physics, so it lives outside `specs/model/`, like `specs/sampling.md`; it changes no equation of the model, and the `v1.0.0` configuration is untouched.

## Conventions

- **One well, one window.** A calibration covers one well and a set of its rows, each a steady period: a well test, an MPFM average, or a period with sensors only. The free parameters $\theta$ are the same for every row. A table with several wells is calibrated well by well; drift over time is handled by recalibrating on a recent window.
- **The model.** Following Kennedy and O'Hagan (2001), the observation $j$ of row $k$ is $y_{kj} = \eta_j(x_k, \theta) + \varepsilon_{kj}$ with $\varepsilon_{kj} \sim N(0, \sigma_{kj}^2)$, where $x_k$ are the row's inputs and $\eta$ the observations predicted from the well's operating point (SOL-4 to SOL-6). This is their model with $\rho = 1$, no discrepancy term $\delta$ and the simulator itself in place of a Gaussian-process emulator: measurement noise only (Bjarne, 2026-10-02).
- **Units** as at ManyWells' interfaces: bar, K, kg/s; phase rates at standard conditions (PVT-GAS-2) in Sm³/h, as in the datasets (`docs/datasets.md`).

## Data

### CAL-1 · Inputs

A row's inputs are known, with no missing values:

| Column | Meaning | Unit | If absent |
|---|---|---|---|
| `CHK` | choke position $u \in (0, 1]$ | – | required |
| `PDC` | pressure downstream of the choke, $p_s$ | bar | required |
| `WGL` | lift-gas rate $w_{lg}$ | kg/s | the well's boundary conditions |
| `p_r`, `T_r`, `T_s`, `T_lg` | reservoir pressure; reservoir, surface and lift-gas temperatures | bar, K | the well's boundary conditions |
| `gor`, `wlr` | the fluid's gas-oil ratio and water-liquid ratio (PVT-MIX-10) | Sm³/Sm³, – | the well's fluid |

Row $k$ is solved at the well's boundary conditions with the row's inputs, and with the well's fluid at the row's `gor` and `wlr`. Every row is a flowing period: $u > 0$.

### CAL-2 · Observations

A row's observations, each of which may be missing (NaN): `PBH`, `PWH` (bar) and `TWH` (K), and the phase rates `QOIL`, `QGAS`, `QWAT` (Sm³/h) together. A row has at least one. `TBH` is not an observation: no free parameter affects it (THM-3, THM-5); a measured `TBH` can set `T_r`.

### CAL-3 · Rate observation

The phase rates of a row give one observation, the reservoir mass rate

$$W_\text{res} = \frac{\rho_{o,sc}\,Q_\text{OIL} + \rho_{g,sc}\,Q_\text{GAS} + \rho_{w,sc}\,Q_\text{WAT}}{3600} \quad [\text{kg/s}],$$

with the row's fluid's densities at standard conditions. When a row's `gor` and `wlr` come from the same rates, the phase split is an input and says nothing about $\theta$, so counting each phase rate would count the same information three times. The total mass rate suits oil and gas wells alike.

### CAL-4 · Noise

Each observation's standard deviation: absolute for the pressures and the temperature, relative for the rate. Defaults (`calibration.Noise`):

| Observation | $\sigma$ | Basis |
|---|---|---|
| `PBH` | 0.3 bar | downhole quartz gauges, about 0.02% of full scale, plus drift |
| `PWH` | 0.3 bar | wellhead transmitters, 0.02–0.1% of full scale, drift up to 0.1% a year |
| `TWH` | 1 K | sensors ±0.3–0.5 K, but an installed bias of several kelvin (Ausen et al. 2017) |
| rate, well test | 2.5% | test separator: oil 1–5%, gas 2–5%, at 95% (NFOGM 2005) |
| rate, MPFM | 10% | about 10% MAPE (Grimstad et al. 2021; Hotvedt et al. 2022) |

A row's `RATE_SOURCE` (`test`, the default, or `mpfm`) picks the rate's default; its columns `PBH_SD`, `PWH_SD`, `TWH_SD` and `RATE_SD` (relative) override the defaults row by row. With measurement noise only, the $\sigma$ also weight the observations against each other and against the priors.

## Parameters

### CAL-5 · Free parameters

A free parameter is a field of a component of `WellProperties`, applied with `dataclasses.replace`:

| Name | Field | Component class | Spec IDs |
|---|---|---|---|
| `K_c` | `choke.K_c` (m²) | any choke | CHK-2, CHK-5, CHK-6 |
| `k_l` | `inflow.k_l` (kg/(s bar)) | `ProductivityIndex` | INF-2 |
| `w_l_max` | `inflow.w_l_max` (kg/s) | `Vogel` | INF-1 |
| `f_D` | `friction.f_D` | `FixedFrictionFactor` | FRIC-1, FRIC-2 |
| `roughness` | `friction.roughness` (m) | `RoughnessFriction` | FRIC-3 to FRIC-6 |
| `h` | `thermal.h` (W/(m² K)) | `ThermalModel` | THM-1 |

Each has a log-normal prior, $\log\theta_i \sim N(\log m_i, s_i^2)$, which keeps it positive. The fit works in the standardized log $z_i = (\log\theta_i - \log m_i)/s_i$: $z = 0$ is the prior medians, and $z_i$ is the parameter's shift from its median in prior standard deviations. The parameters that are not free keep the well's values. The heat capacities are not parameters: they enter THM-1 only through $h/C$, so they are confounded with $h$. The choke profile, the critical pressure ratio and the slip parameters stay fixed.

### CAL-6 · Default priors

| Name | Median $m$ | $s$ (95% range) | Basis |
|---|---|---|---|
| `K_c` | $0.6\,A_c$, with $A_c$ the choke's full-open throat area, if given; otherwise $0.12\,A$, with $A$ the tubing's area | 0.2 with $A_c$ ($C_D$ 0.40–0.89); otherwise 0.35 ($0.06A$ to $0.24A$) | $K_c = C_D A_c$. With Simpson's multiplier (CHK-5), tuned $C_D$ is 0.61–0.69 in the lab and 0.47–0.67 in the field (Haug 2012, Table 1); the sharp-edge contraction coefficient is 0.61; 0.65 is best in Mwalyepelo and Stanko (2016). $0.12A$ is the paper's reference ($C_D = 0.6$ for a throat of $A/5$), and $0.06A$ to $0.24A$ is SMP-5. |
| `k_l`, `w_l_max` | the well's value: the user's estimate, from CAL-7 or a well-test analysis | 1.15 (a factor of 10 either way) | a well's productivity has no generic value; the prior carries what the user knows |
| `f_D` | 0.02 | 0.5 (0.0075–0.053) | Moody gives 0.009–0.025 for steel tubing; the upper tail is wider because in a vertical well $f_D$ also absorbs holdup error. SMP-3 samples $U(0.01, 0.08)$. |
| `roughness` | $4.6\cdot10^{-5}$ m, commercial steel (Moody 1944) | 1.15 ($4.6\cdot10^{-6}$ to $4.6\cdot10^{-4}$ m) | from drawn to corroded tubing; SMP-42 samples LogUniform($1.5\cdot10^{-6}$, $1.5\cdot10^{-4}$) |
| `h` | 15 W/(m² K) | 0.5 (5.6–40) | $h$ acts on $T - T_a$ with $T_a$ the undisturbed geotherm (THM-1, THM-4), so it is Hasan and Kabir's (2012) effective coefficient $U k_e/(k_e + r_{ti} U T_D)$, about 5–25; Wiktorski et al. (2019) give 9.5–22 for $U$ before the formation's transient resistance; SMP-4 samples $U(10, 40)$ |

A user's prior replaces a default parameter by parameter.

### CAL-7 · Productivity from the reservoir

A physics-based median for the productivity, where the reservoir is known: the pseudo-steady radial-inflow (Darcy) index

$$k_l = \rho_{l,sc}\,\frac{2\pi k h_\text{net}}{\mu B\left(\ln(r_e/r_w) - 3/4 + S\right)}\cdot 10^5 \quad [\text{kg/(s bar)}],$$

with permeability $k$ (m²), net pay $h_\text{net}$ (m), the liquid's viscosity $\mu$ (Pa s) and formation volume factor $B$ at reservoir conditions, drainage and wellbore radii $r_e > r_w$ (m), skin $S$, and the liquid's density at standard conditions $\rho_{l,sc}$ (Ahmed 2006; Guo et al. 2008). For Vogel inflow, $w_{l,\max} = k_l\,p_r/1.8$, whose slope at $p_0 = p_r$ is $k_l$ (INF-1).

## Forward model

### CAL-8 · Predicted observations

At $\theta$, row $k$ is solved for its operating point (SOL-4 to SOL-6, `simulate`), on its own well: $\theta$ applied, with the row's fluid (CAL-1). The predicted observations come from the root with the definitions of the dataset features (SMP-30, `datasets.rows.root_features`): `PBH` $= p_0$, `PWH` and `TWH` the last point's $p$ and $T$, and the rate WLIQ + WGAS $= w_\text{res} + w_{g,\text{res}}$, the reservoir's liquid and gas without the lift gas.

### CAL-9 · No operating point

A row was observed flowing, so a $\theta$ under which it has no operating point is inconsistent with it: each of the row's scaled residuals (CAL-10) is then $10^3$, which makes the solver reject the step (Seman et al. 2020, an extreme barrier). The barrier is flat, so the fit must start where every row flows. It starts at the start with the lowest cost (CAL-10) at which every row has an operating point, among the prior medians, $z = 0$, and $z = \pm 1$ and $\pm 2$ along each parameter in turn: $4p + 1$ starts for $p$ parameters, solved together; the medians win a tie. If no start has an operating point at every row, a `CalibrationError` names the rows that have none at the medians, rather than dropping them.

Two failures of a start at the medians alone, found in the twin study (feature spec 017, Findings): a well whose parameters lie far from the medians can fail to flow at the medians at some rows, typically at small choke openings, although it flows at its own parameters; and from medians far from the truth, the fit can step past ground where the well cannot flow into a spurious valley of a correlation outside its range, such as a roughness of 140 m.

## Fit

### CAL-10 · Objective

The MAP estimate minimizes

$$\tfrac12 \sum_k \sum_{j \in \text{obs}(k)} r_{kj}^2 + \tfrac12 \sum_i z_i^2, \qquad r_{kj} = \frac{y_{kj} - \eta_j}{\sigma_{kj}} \text{ for PBH, PWH, TWH}, \quad r_{k,\text{W}} = \frac{\log W_{\text{res},k} - \log \eta_{\text{W}}}{\sigma_{k,\text{W}}},$$

a sum of squares in which the priors' residuals are $z$ itself. Its Jacobian in $z$ is a forward difference with step $10^{-6}$: the predicted observations are smooth in $\log\theta$, and the difference quotients of PBH, PWH, TWH and $\log W$ agree to five digits for steps from $10^{-4}$ to $10^{-8}$ (feature spec 017, Measurements).

### CAL-11 · The fit

`scipy.optimize.least_squares`, trust-region reflective, from the start of CAL-9, with the Jacobian of CAL-10. It stops when a step changes $z$ by less than $10^{-6}$ (relative), the cost by less than $10^{-10}$ (relative), or the scaled gradient falls below $10^{-10}$, or after 100 evaluations of the residuals; the last is not convergence.

### CAL-12 · Result and diagnostics

The result holds the estimate $\hat\theta$, the calibrated `WellProperties`, the solver's status, the start, and, for every row, each observation's observed and predicted value and scaled residual. It reports no covariance (Bjarne, 2026-10-02). Two diagnostics need none:

- the root mean square of each observation's scaled residuals, near 1 where the model fits within the noise; plotted against `CHK` or time, a trend points to a wrong choke profile or to drift;
- each parameter's shift from its prior median, $z_i$: beyond 2 prior standard deviations it is flagged, as a sign of model error rather than of a better estimate (Kennedy and O'Hagan 2001, §4.3; Challenor in the discussion).

`calibration.evaluate` gives the same table for any well on any rows, so a calibrated well can be checked on rows it was not calibrated on.

## Synthetic data and checks

### CAL-13 · Synthetic data

For a well and known parameters $\theta^*$: operating points around the well's boundary conditions, spread evenly over the choke position (0.2 to 1 by default), uniformly within ±10% of $p_s$ and over a lift-gas range if given; each solved for its operating point (an operating point without one is left out); noise at the $\sigma$ of CAL-4, Gaussian on PBH, PWH and TWH, and one log-normal factor per row on the three phase rates together, so the fractions stay the well's; then the observations of an instrumentation:

| Instrumentation | Observations |
|---|---|
| `full` | PBH, PWH, TWH and the rates on every row |
| `no_downhole` | no PBH |
| `periodic_tests` | PBH, PWH, TWH on every row, the rates on one row in five |
| `pressures_only` | PBH, PWH, TWH, the rates on two rows |
| `random_missing` | each observation missing with probability 0.2, the phase rates together; every row keeps one |

Every draw comes from `numpy.random.default_rng(seed)`.

### Checks

The calibration is checked on wells simulated with known parameters (`tests/test_calibration_recovery.py`), drawn with `manywells.sampling` in the `develop` configuration on 20 cells, with $z^* = (1.5, -1, 1, -1.5)$ for (`K_c`, `w_l_max`, `roughness`, `h`), all four free, from the start of CAL-9. Bjarne signed off the tolerances on 2026-10-02.

1. **Identification.** With noise-free rows, full instrumentation and the $\sigma$ scaled down by 1000, so that the priors' pull is negligible, every parameter is within $10^{-3}$ of $z^*$.
2. **Getting close.** With noise, in each instrumentation: the fit converges; $\lVert\hat z - z^*\rVert < \lVert z^*\rVert$, closer to the truth than the medians; `K_c`, `w_l_max` and `h` are each within 0.5 of $z^*$; and every observation's RMS of scaled residuals is below 2. Where the data fix only a combination of parameters, the fit can only move along that ridge; it still moves towards $z^*$, since in a linearized model with noise-free rows the MAP is the point of the ridge nearest the medians, and $z^*$ lies on the ridge.
3. **Determinism.** The same inputs give the same $\hat\theta$.

4. **Twin study** (`scripts/calibration/twins.py`): the same checks on 30 sampled wells, with $z^*$ drawn from the priors and 20 rows per well. In every instrumentation, at least 95% of the fits converge, end closer to $z^*$ than the medians where $z^*$ is more than 0.5 from them, have `K_c`, `w_l_max` and `h` within 0.5 of $z^*$, and have every RMS of scaled residuals below 2. The study also measures how the error falls with rows and noise and what a wrong model does to $\hat\theta$, and compares the backends; its results are in the feature spec.

## Coverage

| ID | Paper | Code | Checked by |
|---|---|---|---|
| CAL-1 | — | `calibration/data.py`; `objective.py` `row_conditions`, `row_fluids` | property: tests/test_calibration.py (validation, row inputs) |
| CAL-2 | — | `calibration/data.py` | property: tests/test_calibration.py (missing values) |
| CAL-3 | — | `calibration/data.py` `observations` | property: tests/test_calibration.py (reservoir mass rate) |
| CAL-4 | — | `calibration/data.py` `Noise`, `noise_sd` | property: tests/test_calibration.py (defaults and overrides) |
| CAL-5 | — | `calibration/parameters.py` | property: tests/test_calibration.py (transform, apply) |
| CAL-6 | — | `calibration/parameters.py` `default_prior` | property: tests/test_calibration.py (the table) |
| CAL-7 | — | `calibration/parameters.py` | property: tests/test_calibration.py (Darcy index, Vogel slope) |
| CAL-8 | — | `calibration/objective.py` `predicted` | property: tests/test_calibration.py (dataset features) |
| CAL-9 | — | `calibration/objective.py`, `fit.py` `start` | property: tests/test_calibration.py (barrier); property: tests/test_calibration_recovery.py (start off the medians, no operating point at the start) |
| CAL-10 | — | `calibration/objective.py` | property: tests/test_calibration.py (scaled residuals); property: tests/test_calibration_recovery.py (identification) |
| CAL-11 | — | `calibration/fit.py` | property: tests/test_calibration_recovery.py (identification, getting close, determinism) |
| CAL-12 | — | `calibration/fit.py` `CalibrationResult`, `evaluate` | property: tests/test_calibration_recovery.py (result and diagnostics) |
| CAL-13 | — | `calibration/synthetic.py` | property: tests/test_calibration.py (instrumentations, seeded noise) |
