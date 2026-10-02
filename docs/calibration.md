# Calibrating a well

This guide shows how to calibrate a well to its production data with `manywells.calibration`: how to prepare the
data, choose the parameters to fit and their priors, run the fit and read its result.
[`specs/calibration.md`](../specs/calibration.md) specifies it, and
[`scripts/sim_examples/calibrate_well.py`](../scripts/sim_examples/calibrate_well.py) is a complete example.

## The idea

A calibration fits a few parameters of one well (its choke coefficient, inflow productivity, friction and heat
transfer) so that the well's operating point matches what was measured: rates from well tests or a multiphase flow
meter (MPFM), and pressures and temperatures from whatever sensors the well has. Each row of data is one steady
period. The fit solves every row for the well's operating point, compares the predicted observations with the
measured ones, scaled by each measurement's noise, and adds a physics-based prior for each parameter. It returns the
parameters that minimise the sum: the posterior mode, or MAP estimate (Kennedy and O'Hagan 2001, without their
model-discrepancy term).

```python
from manywells.calibration import CalibrationData, calibrate

data = CalibrationData(df)                              # one row per steady period
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'roughness', 'h'])
print(result.summary())
calibrated = result.well                                # a WellProperties with the fitted values
```

`wp` is the well as you know it, and `bc` its boundary conditions; each row replaces the ones it has columns for.

## Preparing the data

One table per well, one row per steady period, in bar, K and Sm³/h. Average a period that is steady, and leave out
transients, shut-ins and rows whose choke is closed.

| Column | | Meaning |
|---|---|---|
| `CHK`, `PDC` | required | choke position in (0, 1], pressure downstream of the choke (bar) |
| `WGL` | optional input | lift-gas rate (kg/s) |
| `p_r`, `T_r`, `T_s`, `T_lg` | optional inputs | reservoir pressure (bar) and temperatures (K), if they change between rows |
| `gor`, `wlr` | optional inputs | the fluid's gas-oil ratio (Sm³/Sm³) and water-liquid ratio, for example from the latest well test |
| `PBH`, `PWH`, `TWH` | observations | bottomhole and wellhead pressure (bar), wellhead temperature (K) |
| `QOIL`, `QGAS`, `QWAT` | observations | phase rates at standard conditions (Sm³/h), all three or none |
| `RATE_SOURCE` | optional | `test` (default) or `mpfm` |
| `PBH_SD`, `PWH_SD`, `TWH_SD`, `RATE_SD` | optional | a row's own noise: absolute for pressures and temperature, relative for the rate |

Inputs may not be missing; observations may (NaN). A well without a downhole gauge has no `PBH` column, and rows
between well tests have no rates. The three phase rates count as one observation, the reservoir's total mass rate,
because the split between them is an input (`gor`, `wlr`). Other columns, such as a time stamp, are kept and appear
in the result's residual table. `CalibrationData` checks the table and raises `ValueError` on what it cannot use.

The default noise (`Noise`) is 0.3 bar on the pressures, 1 K on the wellhead temperature, 2.5% on well-test rates
and 10% on MPFM rates. Pass `CalibrationData(df, noise=Noise(...))` to change it for a well. With measurement noise
only, these also weight the observations: a smaller σ makes the fit follow that sensor more closely.

## Choosing the parameters and their priors

| Name | Field | Default prior: median, log-sd | Needs |
|---|---|---|---|
| `K_c` | `choke.K_c` (m²) | 0.12 A, 0.35; or 0.6 A_c, 0.2 with `A_c=` the choke's throat area | any choke |
| `w_l_max` | `inflow.w_l_max` (kg/s) | the well's value, 1.15 | `Vogel` |
| `k_l` | `inflow.k_l` (kg/(s bar)) | the well's value, 1.15 | `ProductivityIndex` |
| `roughness` | `friction.roughness` (m) | 4.6·10⁻⁵, 1.15 | `RoughnessFriction` |
| `f_D` | `friction.f_D` | 0.02, 0.5 | `FixedFrictionFactor` |
| `h` | `thermal.h` (W/(m² K)) | 15, 0.5 | |

A prior is log-normal: the parameter's log is normal with that median and standard deviation, so a log-sd of 0.5
puts 95% of the prior within a factor of e¹ ≈ 2.7 of the median. `calibrate(..., priors={'h': Parameter('h', 20.0,
0.3)})` replaces a default with what you know of the well. The productivity's default median is the value in `wp`,
so set it to your best estimate; `darcy_productivity_index` gives one from the reservoir's permeability, net pay,
viscosity, formation volume factor and skin, and `vogel_maximum_rate` converts it to Vogel's maximum rate.

Free only what the data can determine. On 29 sampled wells with 20 rows each (the twin study,
`specs/features/017-calibration.md`), errors in prior standard deviations, median and largest:

| Instrumentation | `K_c` | `w_l_max` | `roughness` | `h` |
|---|---|---|---|---|
| everything on every row | 0.01, 0.04 | 0.01, 0.04 | 0.04, 1.00 | 0.02, 0.18 |
| no downhole gauge | 0.01, 0.45 | 0.02, 0.32 | 0.10, 1.31 | 0.02, 0.15 |
| rates on one row in five | 0.02, 0.51 | 0.01, 0.05 | 0.08, 1.31 | 0.02, 0.15 |
| rates on two rows | 0.03, 0.13 | 0.01, 0.06 | 0.14, 1.45 | 0.04, 0.22 |

The choke coefficient, the productivity and the heat transfer are well determined as long as some rows have rates,
because the reservoir pressure is given. The roughness is the hardest: in smooth tubing the friction factor depends
on the Reynolds number alone, so the roughness barely changes the pressure drop. Without any rates, nothing but the
tubing model ties the pressures to a rate, so expect friction, the choke coefficient and the productivity to trade
off (the twin study did not test this case). A parameter the data do not determine stays near its
prior, which is what you want.

## Reading the result

```text
parameter         value  prior median  shift (sd)
K_c           0.0011862    0.00094248        0.66
w_l_max          45.227            60       -0.25
roughness    0.00014398       4.6e-05        0.99
h                22.077            15        0.77
RMS of scaled residuals: PBH 0.82, PWH 0.77, TWH 0.76, WRES 1.17
converged: Both `ftol` and `xtol` termination conditions are satisfied.
```

- **The shift** is how far each parameter moved from its prior median, in prior standard deviations. Beyond 2 it is
  flagged (`result.flagged()`): a parameter pushed far from where physics puts it is more likely absorbing a model
  error than measuring a property of the well.
- **The RMS of scaled residuals** is near 1 when the model fits within the noise. Much larger means the model cannot
  match the data, or the noise is set too small.
- **The residual table** (`result.residuals`) has, for every row, each observation's observed and predicted value
  and scaled residual (`PWH_obs`, `PWH_pred`, `PWH_res`, ...). Plot the residuals against `CHK`: a trend points to a
  wrong choke profile. Against time, it points to drift, and a recalibration on a recent window.

`evaluate(result.well, other_data, bc)` gives the same table for rows the well was not calibrated on, which is the
check that matters for prediction.

## Synthetic wells

`synthetic_data(wp, bc, values, n_rows, seed, instrumentation=...)` simulates a well at parameters `values`, with
noise and the observations of an instrumentation (`full`, `no_downhole`, `periodic_tests`, `pressures_only`,
`random_missing`). Calibrating such data from the priors shows what a well's instruments can determine before real
data is at hand; the recovery tests and the twin study are built on it.

## Limits of this version

- **One well and one window.** The parameters are the same for every row; recalibrate as the well changes.
- **Inputs are known.** The reservoir pressure, temperatures and fluid ratios are taken as given. An error in the
  reservoir pressure goes straight into the productivity.
- **No uncertainty.** The result is the posterior mode, without a covariance.
- **Measurement noise only.** The model's own error is not represented, so the parameters absorb it.
- **Speed.** Each row is a full solve; on the Rust core the rows run in parallel threads, and a fit of a few tens of
  rows takes seconds to a minute. The CasADi backend works too, but builds a system per row and parameter value.
