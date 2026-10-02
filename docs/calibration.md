# Calibrating a well

This guide shows how to calibrate a ManyWells well to its production data with `manywells.calibration`, and how to
use the calibrated well:
1. describe the well;
2. prepare its data;
3. choose which parameters to fit and their priors;
4. run the fit and read its result;
5. check the calibrated well and predict with it.

[`specs/calibration.md`](../specs/calibration.md) specifies the method, and
[`scripts/sim_examples/calibrate_well.py`](../scripts/sim_examples/calibrate_well.py) is a complete example to run.

## What calibration does

A calibration fits a few parameters of one well so that its simulated operating points match what was measured:
- the choke coefficient;
- the inflow productivity;
- friction;
- heat transfer.

The measurements are rates from well tests or a multiphase flow meter (MPFM), and pressures and temperatures from
whatever sensors the well has. Each row of data is one steady period, and a value may be missing.

For given parameter values, the fit solves every row for the well's operating point: the stable root, at the row's
choke position, downstream pressure and lift-gas rate. It compares the predicted observations with the measured
ones, each scaled by its measurement noise, and adds a physics-based prior for each parameter. The answer is the set
of parameters that minimises the sum: the posterior mode, or MAP estimate. In Kennedy and O'Hagan's (2001) terms, it
calibrates the simulator itself, without an emulator and without a model-discrepancy term.

The workflow in short:

```python
from manywells.calibration import CalibrationData, calibrate

data = CalibrationData(df)                                  # one row per steady period
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'h'])
print(result.summary())                                     # the estimate, its shifts from the priors, the fit
well = result.well                                          # a WellProperties with the fitted values
```

`wp` is the well as you know it, `bc` its boundary conditions, and `df` its rows. The sections below explain each.

## 1. Describe the well

The well is a `WellProperties` and a `BoundaryConditions`, as for a simulation (`docs/simulate.md`). The
calibration changes only the free parameters, and keeps everything else as you give it. So set every part to what
you know:

```python
from manywells.choke import SimpsonChokeModel
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Vogel
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.thermal import ThermalModel

geometry = WellGeometry.from_survey([0, 800, 3100], [0, 800, 2600], n_cells=50, D=0.1)  # MD and TVD survey (m)
fluid = FluidModel(rho_o=density_from_api(32), rho_g=gas_density_from_sg(0.7), gor=150.0, wlr=0.2)
wp = WellProperties(geometry=geometry, fluid=fluid,
                    inflow=Vogel(w_l_max=40.0),                   # your estimate: it is the prior's median
                    choke=SimpsonChokeModel(K_c=0.12 * geometry.A, chk_profile='linear'),
                    friction=RoughnessFriction(roughness=4.6e-5),
                    thermal=ThermalModel(h=15.0))
bc = BoundaryConditions(p_r=280.0, T_r=368.0, T_s=280.0)          # bar and K
```

Things to get right:
- **Geometry.** The geometry runs from where the reservoir fluid enters, the bottom of the modelled tubing, to the
  wellhead. `D` is the tubing's inner diameter.
- **Fluid.**
  - The densities are at standard conditions; `density_from_api` and `gas_density_from_sg` convert from the usual
    field values.
  - `gor` and `wlr` are the well's default gas-oil and water-liquid ratios, and a row can override them.
- **Choke profile.** It maps the choke position to an opening and is not calibrated: pick the one closest to the
  valve's characteristic (`linear`, `sigmoid`, `convex` or `concave`).
- **Reservoir and ambient.**
  - `p_r` and `T_r` are the reservoir's pressure and temperature, and `T_s` the ambient temperature at the surface
    (or the seabed).
  - They are inputs, not fitted. An error in `p_r` goes straight into the productivity, so use the best value you
    have, and give it per row if it changes over the data's period.
- **Units.** Pressures are absolute bar and temperatures kelvin everywhere in ManyWells: add 1.01325 to a gauge
  pressure in barg, and 273.15 to degrees Celsius.

## 2. Prepare the data

One table per well, one row per steady period: a `pandas.DataFrame` that `CalibrationData` checks.

| Column | Role | Meaning | Unit |
|---|---|---|---|
| `CHK` | input, required | choke position, in (0, 1] | – |
| `PDC` | input, required | pressure downstream of the choke | bar |
| `WGL` | input | lift-gas rate (0 if absent) | kg/s |
| `p_r`, `T_r`, `T_s`, `T_lg` | input | reservoir pressure; reservoir, surface and lift-gas temperatures (`bc`'s if absent) | bar, K |
| `gor`, `wlr` | input | gas-oil and water-liquid ratios at standard conditions (the fluid's if absent) | Sm³/Sm³, – |
| `PBH`, `PWH`, `TWH` | observation | bottomhole and wellhead pressure, wellhead temperature | bar, K |
| `QOIL`, `QGAS`, `QWAT` | observation | phase rates at standard conditions, all three or none | Sm³/h |
| `RATE_SOURCE` | noise | `test` (the default) or `mpfm`: the source of the row's rates | – |
| `PBH_SD`, `PWH_SD`, `TWH_SD` | noise | the row's own standard deviation of a pressure or temperature | bar, K |
| `RATE_SD` | noise | the row's own relative standard deviation of the rate | – |

Rules:
- **Inputs** may not be missing.
- **Observations** may be missing (NaN). A well without a downhole gauge has no `PBH` column, and rows between well
  tests have no rates.
- **At least one observation per row**, and every row is a flowing period: average a steady period, and leave out
  transients, start-ups and shut-ins.
- **The three phase rates are one observation.** The fit compares the reservoir's total mass rate, because the split
  between the phases is an input (`gor`, `wlr`). Give the row the ratios of the same test, so that they agree with
  its rates.
- **Other columns, such as a time stamp, are kept**, and appear in the result's residual table.

Converting a typical well-test export:

```python
import pandas as pd

from manywells.calibration import CalibrationData

tests = pd.read_csv('well_tests.csv')                     # your export, one row per test or averaged period
df = pd.DataFrame({
    'TIME': pd.to_datetime(tests['date']),
    'CHK': tests['choke_percent'] / 100,
    'PDC': tests['p_downstream_barg'] + 1.01325,          # barg to bar (absolute)
    'PWH': tests['p_wellhead_barg'] + 1.01325,
    'PBH': tests['p_downhole_barg'] + 1.01325,            # leave out if the well has no downhole gauge
    'TWH': tests['t_wellhead_degC'] + 273.15,
    'QOIL': tests['oil_Sm3_per_day'] / 24,                # Sm³/d to Sm³/h
    'QGAS': tests['gas_Sm3_per_day'] / 24,
    'QWAT': tests['water_Sm3_per_day'] / 24,
})
df['gor'] = df['QGAS'] / df['QOIL']
df['wlr'] = df['QWAT'] / (df['QOIL'] + df['QWAT'])
df[['gor', 'wlr']] = df[['gor', 'wlr']].ffill()           # rows between tests keep the latest test's ratios
data = CalibrationData(df)
```

What `CalibrationData` rejects, each with a `ValueError` that says why:
- a missing or non-numeric input;
- `CHK` outside (0, 1], or negative pressures, temperatures or rates;
- some but not all of the three phase rates;
- a row without observations;
- an unknown `RATE_SOURCE`.

### Noise

Each observation's measurement noise is a standard deviation. The defaults (`Noise`) are:

| Observation | Default | Typical source |
|---|---|---|
| `PBH` | 0.3 bar | downhole quartz gauge, with drift |
| `PWH` | 0.3 bar | wellhead pressure transmitter |
| `TWH` | 1 K | wellhead temperature sensor, with its installed bias |
| rate, well test | 2.5% | test separator |
| rate, MPFM | 10% | multiphase flow meter |

Change them for a well with `CalibrationData(df, noise=Noise(TWH=3.0))`, or for single rows with the `*_SD`
columns. The calibration has no term for the model's own error, so these values also weight the observations against
each other: a smaller one makes the fit follow that sensor more closely.
- **A sensor you trust less.** Give it a larger value, for example a wellhead temperature sensor with a known bias.
- **A downhole gauge some way above the inflow.** It reads less than the model's `PBH`, which is at the bottom of
  the modelled tubing. Correct the reading by the hydrostatic head between the two, or raise `PBH_SD`.

## 3. Choose the free parameters and their priors

| Name | Field | Default prior: median, log-sd | The well needs |
|---|---|---|---|
| `K_c` | `choke.K_c` (m²) | 0.12 A, 0.35; with `A_c=` the throat area, 0.6 A_c, 0.2 | any choke |
| `w_l_max` | `inflow.w_l_max` (kg/s) | the well's value, 1.15 | `Vogel` inflow |
| `k_l` | `inflow.k_l` (kg/(s bar)) | the well's value, 1.15 | `ProductivityIndex` inflow |
| `roughness` | `friction.roughness` (m) | 4.6·10⁻⁵, 1.15 | `RoughnessFriction` |
| `f_D` | `friction.f_D` | 0.02, 0.5 | `FixedFrictionFactor` |
| `h` | `thermal.h` (W/(m² K)) | 15, 0.5 | |

**How to read a prior.** Each prior is log-normal: the parameter's logarithm is normal, with that median and
standard deviation (log-sd). About 95% of the prior lies within a factor of $e^{2s}$ of the median:
- a factor of 2 for s = 0.35;
- 2.7 for s = 0.5;
- 10 for s = 1.15.

The defaults are physics-based (`specs/calibration.md`, CAL-6):
- **`K_c`** is a discharge coefficient times the choke's throat area.
- **`h`** is an effective heat-transfer coefficient to the undisturbed formation.
- **`roughness`** spans drawn to corroded steel tubing.
- **Productivity.** A well's productivity has no generic value, so its prior's median is whatever `wp` holds, and
  the wide log-sd lets the data decide.

Replace a default with what you know of the well:

```python
from manywells.calibration import (CENTIPOISE, MILLIDARCY, Parameter, calibrate, darcy_productivity_index,
                                   vogel_maximum_rate)

# The choke's throat area from its data sheet: K_c's prior becomes C_D A_c with C_D = 0.6
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'h'], A_c=1.2e-3)

# A productivity from the reservoir: pseudo-steady radial inflow, then Vogel's maximum rate with that slope
k_l = darcy_productivity_index(k=150 * MILLIDARCY, h_net=25.0, mu=1.5 * CENTIPOISE, B=1.25, r_e=600.0, r_w=0.108,
                               skin=2.0, rho_sc=wp.fluid.rho_l)
priors = {'w_l_max': Parameter('w_l_max', vogel_maximum_rate(k_l, bc.p_r), 0.7, 'Darcy, from core data'),
          'h': Parameter('h', 20.0, 0.3, 'completion with insulated tubing')}
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'h'], priors=priors)
```

**Free only what the data can determine.** On 29 sampled wells with 20 rows each (the twin study,
`specs/features/017-calibration.md`), the errors were as follows, in prior standard deviations, median and largest:

| Instrumentation | `K_c` | `w_l_max` | `roughness` | `h` |
|---|---|---|---|---|
| everything on every row | 0.01, 0.04 | 0.01, 0.04 | 0.04, 1.00 | 0.02, 0.18 |
| no downhole gauge | 0.01, 0.45 | 0.02, 0.32 | 0.10, 1.31 | 0.02, 0.15 |
| rates on one row in five | 0.02, 0.51 | 0.01, 0.05 | 0.08, 1.31 | 0.02, 0.15 |
| rates on two rows | 0.03, 0.13 | 0.01, 0.06 | 0.14, 1.45 | 0.04, 0.22 |

- **Well determined.** With the reservoir pressure given and rates on some rows, the choke coefficient, the
  productivity and the heat transfer are well determined, downhole gauge or not.
- **Roughness is the hardest.** In smooth tubing the friction factor depends on the Reynolds number alone, so the
  roughness barely changes the pressure drop.
- **No rates at all.** Then nothing but the tubing model ties the pressures to a rate, and friction, the choke
  coefficient and the productivity trade off. The twin study did not test this case; fix friction if you calibrate
  without rates.

A parameter the data do not determine stays near its prior, which is what you want. A good first choice is
`['K_c', 'w_l_max', 'h']` (or `k_l`), adding friction where the well has a downhole gauge. Section 7 shows how to
check what your well's instruments determine before you trust a fit.

## 4. Run the calibration

```python
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'roughness', 'h'], backend='rust', workers=None)
```

What it does:
- **It starts** at the cheapest of the prior medians and of points one and two prior standard deviations along each
  parameter, among those at which every row has an operating point.
- **It runs** a least-squares solver (`scipy.optimize.least_squares`) in the parameters' standardized logarithms.
- **It solves every row** for its operating point at each trial. On the Rust core (the default) the rows run in
  parallel threads, `workers` of them (by default the number of CPUs).

On 24 CPUs, a fit of 20 rows and four parameters on a 20-cell grid takes about 11 s at the median (3 to 50 s
across the twin study's wells), and 160 rows about a minute. A finer grid costs more: a row of a 100-cell L-shaped
well takes 0.8 s against 0.3 s at 20 cells. It is deterministic:
the same inputs give the same result.

The CasADi backend (`backend='casadi'`) gives the same estimate where its root search finds the same roots, but it
builds a system for every row and parameter value: it is 20 to 100 times slower, and it misses some roots the core
finds. Use it for a well with a component the Rust core does not have, such as your own subclass of `InflowModel`.

`calibrate` raises `CalibrationError` if no start has an operating point at every row (section 8), and
`ValueError` for a parameter the well cannot take, such as `k_l` with Vogel inflow. It logs to
`logging.getLogger('manywells')` and prints nothing.

## 5. Read the result

```python
print(result.summary())
```

```text
parameter         value  prior median  shift (sd)
K_c           0.0011862    0.00094248        0.66
w_l_max          45.227            60       -0.25
roughness    0.00014398       4.6e-05        0.99
h                22.077            15        0.77
RMS of scaled residuals: PBH 0.82, PWH 0.77, TWH 0.76, WRES 1.17
converged: Both `ftol` and `xtol` termination conditions are satisfied.
```

| Attribute | What it holds |
|---|---|
| `result.values` | the estimate, `{name: value}` |
| `result.well` | `wp` with the estimate applied: a `WellProperties` to simulate with |
| `result.z` | each parameter's shift from its prior median, in prior standard deviations |
| `result.flagged()` | the parameters shifted by more than 2 |
| `result.rms()` | each observation's root mean square of scaled residuals |
| `result.residuals` | the data's rows with `<obs>_obs`, `<obs>_pred` and `<obs>_res` for `PBH`, `PWH`, `TWH` and `WRES` (the mass rate) |
| `result.success`, `result.message` | whether the solver met its stopping rule, and why it stopped |
| `result.start`, `result.parameters`, `result.n_solves` | where the fit started, the priors used, and the solves it took |

How to judge a fit:
- **The RMS of scaled residuals** should be near 1: the model then fits within the noise. Much larger means the
  model cannot match the data, or the noise is set too small for what the model can do.
- **A flagged parameter**, pushed more than 2 prior standard deviations from where physics puts it, is more likely
  absorbing a model error than measuring a property of the well. Check the inputs, the choke profile and the fluid
  before you trust it.
- **Plot the scaled residuals against `CHK`.** A trend points to a wrong choke profile, which the twin study found
  biases `K_c` by up to 3 standard deviations. Plotted against time, a trend points to drift, and to calibrating on a
  recent window instead.
- **Silent errors.** Some model errors leave the residuals clean. With an energy-balance error, for instance, `h`
  absorbs it. So a good fit shows that the model can reproduce the data, not that each parameter is physically
  right.

## 6. Check the calibrated well and predict with it

Keep some rows back and check the calibrated well on them; that is the test that matters for prediction:

```python
from manywells.calibration import evaluate

train, test = df[df['TIME'] < '2026-06-01'], df[df['TIME'] >= '2026-06-01']
result = calibrate(wp, CalibrationData(train), bc, free=['K_c', 'w_l_max', 'h'])
table = evaluate(result.well, CalibrationData(test), bc)        # the same columns as result.residuals
print(table[['TIME', 'PWH_obs', 'PWH_pred', 'WRES_obs', 'WRES_pred']])
```

To predict an operating point you have not measured, for example at a new choke position, simulate the calibrated
well. `root_features` gives the dataset features: rates at standard conditions in Sm³/h, pressures in bar and
temperatures in K.

```python
from dataclasses import replace

from manywells.datasets.rows import root_features
from manywells.simulator import SSDFSimulator

well = result.well                                        # or replace(result.well, fluid=...) for the current ratios
new_bc = replace(bc, u=0.8, p_s=21.0)
op = SSDFSimulator(well, backend='rust').simulate(new_bc)     # NoOperatingPoint if the well cannot flow there
f = well.fluid
pred = root_features(op, new_bc, f.rho_o, f.rho_w, f.rho_g, 1 - f.f_o_in_liquid)
print(pred['QOIL'], pred['QGAS'], pred['QWAT'], pred['PWH'], pred['TWH'])
```

### Several wells, and change over time

A calibration covers one well and one window of rows, with the same parameters for every row. Calibrate a field well
by well, each with its own `WellProperties` and its own instruments:

```python
results = {name: calibrate(wells[name], CalibrationData(rows), conditions[name], free=['K_c', 'w_l_max', 'h'])
           for name, rows in field.groupby('WELL')}   # wells and conditions: dicts of WellProperties and BoundaryConditions
```

Wells change:
- **What changes.** Reservoir pressure falls, the productivity changes and chokes erode.
- **What to do.** Recalibrate on a recent window, for example the last six months, as new well tests arrive.
- **Before you trust a recalibration**, compare the new estimate with the last one.

## 7. Check what your instruments can determine

Before calibrating real data, calibrate a synthetic copy of your well:
1. Choose parameters you pretend not to know, and simulate the well's rows at them with your instruments and noise.
2. Calibrate from the priors, and see how close the fit gets.

```python
from manywells.calibration import synthetic_data

truth = {'K_c': 1.3 * wp.choke.K_c, 'w_l_max': 0.6 * wp.inflow.w_l_max, 'h': 25.0}
twin = synthetic_data(wp, bc, truth, n_rows=20, seed=1, instrumentation='no_downhole')
check = calibrate(wp, twin, bc, free=list(truth))
for name in truth:
    print(f'{name}: true {truth[name]:.4g}, calibrated {check.values[name]:.4g}')
```

`synthetic_data` takes these instrumentations:
- `full`: everything on every row;
- `no_downhole`: no `PBH`;
- `periodic_tests`: rates on one row in five;
- `pressures_only`: rates on two rows;
- `random_missing`: each observation missing with probability 0.2.

Its other arguments:
- `noise=` sets the noise;
- `noisy=False` leaves it out;
- `u=`, `w_lg=` and `p_s_spread=` choose the operating points.

A parameter that this check does not recover will not be recovered from real data either.

## 8. Troubleshooting

| Symptom | Likely cause, and what to do |
|---|---|
| `CalibrationError`: rows have no operating point at the prior medians, nor at any start | The well cannot flow at those rows under any start near the priors. Check the row's inputs: `PDC` above `p_r`, gauge rather than absolute pressures, a wrong `p_r`, or a fluid far from the well's. Otherwise widen the productivity's prior, or centre it nearer the truth. |
| `ValueError: parameter k_l needs inflow to be a ProductivityIndex` | The parameter does not belong to the well's component: free `w_l_max` for Vogel inflow, `f_D` for a fixed friction factor. |
| A parameter is flagged | Section 5: probably a model error. Check the inputs, the choke profile and the fluid; fix the parameter at its prior if the data cannot determine it. |
| The RMS of one observation is far above 1 | A biased sensor or a model error in that part. Raise that observation's noise if you trust it less, and look at its residuals against `CHK` and time. |
| `result.success` is `False` | The solver stopped after 100 evaluations. Check the result as above. Too many free parameters for the data is the usual reason; free fewer. |
| `ValueError: the Rust core cannot solve this well` | The well has a component the core does not have. Use `backend='casadi'`, which is much slower. |

## Limits of this version

- **One well and one window.** The parameters are the same for every row; recalibrate as the well changes.
- **Inputs are known.** The reservoir pressure, the temperatures and the fluid's ratios are taken as given and not
  fitted.
- **No uncertainty.** The result is the posterior mode, without a covariance.
- **Measurement noise only.** The model's own error is not represented, so the parameters absorb it.
- **Four parameters.** The choke profile, the critical pressure ratio, the slip parameters and the heat capacities
  are not calibrated.

`plans/calibration-plan.md` lists what is planned after this version.

## Reference

| Name | What it is |
|---|---|
| `CalibrationData(rows, noise=Noise())` | one well's rows (section 2), validated on construction |
| `Noise(PBH=0.3, PWH=0.3, TWH=1.0, rate_test=0.025, rate_mpfm=0.10)` | the default noise per observation |
| `calibrate(wp, data, bc, free, priors=None, A_c=None, backend='rust', workers=None)` | the fit; returns a `CalibrationResult` |
| `evaluate(wp, data, bc, backend='rust', workers=None)` | the residual table of a well on rows |
| `CalibrationResult` | the result (section 5) |
| `CalibrationError` | a fit that cannot start, or ends where a row cannot flow; a `SimError` |
| `Parameter(name, median, log_sd, basis='')` | a free parameter's log-normal prior |
| `default_prior(name, wp, A_c=None)` | the default prior of a parameter for a well |
| `FIELDS` | the parameters' names, with the field each sets |
| `apply(wp, values)` | a copy of `wp` with parameter values set |
| `darcy_productivity_index(k, h_net, mu, B, r_e, r_w, skin, rho_sc)`; `MILLIDARCY`, `CENTIPOISE` | a productivity index from the reservoir (SI units; the constants convert mD and cP) |
| `vogel_maximum_rate(k_l, p_r)` | Vogel's maximum rate with the slope `k_l` at `p_r` |
| `synthetic_data(wp, bc, values, n_rows, seed, instrumentation='full', noise=None, noisy=True, rate_source='test', backend='rust', workers=None, **spread)` | a well's rows simulated at known parameters (section 7) |
| `INSTRUMENTATIONS` | the instrumentations `synthetic_data` takes |
