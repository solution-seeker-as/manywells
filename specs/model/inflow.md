# Inflow

## Purpose

The rates at which reservoir fluid enters the well at the bottomhole and lift gas at its injection point, and the two rows that impose them on the state at point 0 (paper §2.3, after Nikoofard et al., 2017).

## Interface

| Function | Inputs | Outputs | Symbolic |
|---|---|---|---|
| reservoir inflow | $p_0$ (bar), $p_r$ (bar), the model's parameters | $w_l$, $w_{g,\text{res}}$ (kg/s) | yes, in $p_0$ |
| inflow rows | state at point 0, $w_{lg}$ | two rows (kg/s) | yes |
| injection point | $z_\text{inj}$ (m), the grid's measured depths | $k$ | no |

In `develop` an inflow model returns the liquid rate only, `liquid_mass_flow_rate(p_0, p_r)`, which is $w_\text{res}$ in `discretization.md`; the fluid model gives the reservoir gas rate (INF-4, `FluidModel.reservoir_gas_rate`) and the phase rates at the bottomhole (INF-5 with dead oil, PVT-OIL-13 with black oil, `FluidModel.phase_rates`). The injection depth is the well's `WellProperties.injection_md`, and `WellProperties.injection_point` is $k$ (INF-9); the discretization gives each point's phase rates the lift gas from point $k$ up.

## Equations

### INF-1 · Vogel inflow performance

$$w_l = w_{l,\max}\left[1 - 0.2\frac{p_0}{p_r} - 0.8\left(\frac{p_0}{p_r}\right)^2\right]$$

$w_{l,\max} \ge 0$ is the liquid rate at $p_0 = 0$. Used by the published datasets.

### INF-2 · Productivity index

$$w_l = k_l\thinspace(p_r - p_0)$$

with $k_l \ge 0$ in kg/(s bar). Not in the paper; v1.0.0's default inflow model, with $k_l = 0.5$ and $f_g = 0.1379$.

### INF-3 · Fixed rate

$$w_l = w_{l,\text{const}}, \qquad w_{g,\text{res}} = w_{g,\text{const}}$$

Both rates are constants, independent of $p_0$; INF-4 does not apply. Not in the paper. v1.0.0 marks it untested, and no dataset uses it.

### INF-4 · Reservoir gas rate

For INF-1 and INF-2:

$$w_{g,\text{res}} = \frac{f_g}{1 - f_g}w_l$$

where $f_g \in (0, 1)$ is the gas mass fraction of the reservoir inflow.

### INF-5 · Bottomhole rates

$$w_g(z_0) = w_{g,\text{res}} + w_{lg}, \qquad w_l(z_0) = w_l$$

The lift gas is the produced gas, injected at the injection point of INF-9, the bottomhole by default, at the rate $w_{lg} \ge 0$.

### INF-6 · Gas inflow row

$$r = A\alpha_0\rho_{g,0}v_{g,0} - w_g(z_0) \quad \text{[kg/s]}$$

### INF-7 · Liquid inflow row

$$r = A\thinspace(1 - \alpha_0)\rho_{l,0}v_{l,0} - w_l(z_0) \quad \text{[kg/s]}$$

### INF-8 · Fixed liquid rate

$$w_l = w_{l,\text{const}}$$

independent of $p_0$, with the gas from INF-4, as for INF-1 and INF-2. `develop`'s `FixedFlowRate`, which replaces INF-3: the gas–liquid split is a property of the fluid, not of the inflow (`plans/develop_model_changes.md`, change 11). Useful for calibration, where the rate is known.

### INF-9 · Lift-gas injection point

The lift gas enters at point $k$, the grid point nearest $z_\text{inj}$, the injection point's measured depth from the surface; of two grid points equally near, the deeper. It enters the phase rates (INF-5, PVT-OIL-13) at points $i \ge k$ only:

$$w_{lg,i} = \begin{cases} 0 & i < k \\ w_{lg} & i \ge k \end{cases}$$

Below point $k$ the tubing carries the reservoir fluid alone. Without an injection depth, $k = 0$, and this is INF-5. The lift gas still does not dissolve into the oil (PVT-OIL-13). Above the bottomhole, the lift gas mixes with the flow in cell $k$'s energy row (THM-9, DISC-12), and point 0's inflow temperature row is THM-3, since no lift gas enters there.

## Options

| Option | IDs | Used by |
|---|---|---|
| Vogel | INF-1, INF-4 | the published datasets |
| Productivity index | INF-2, INF-4 | v1.0.0's default |
| Fixed rate | INF-3 | none; `develop` has INF-8 in its place |
| Fixed liquid rate | INF-8, INF-4 | `develop`'s `FixedFlowRate`; not in the `v1.0.0` configuration |

The lift gas's injection point is a well option of `develop`: at the bottomhole (`injection_md=None`, the default) it is INF-5, and above it INF-9. The `v1.0.0` configuration injects at the bottomhole.

All of them use INF-5 to INF-7 (with black oil, PVT-OIL-13 in place of INF-5). On `develop`, the inflow model returns $w_l$ only and the fluid model supplies $f_g$ (PVT-MIX-10).

## Safeguards

None. Above $p_0 = p_r$ the inflow rates are negative, which no admissible state has (SOL-1). v1.0.0 asserts $w_{l,\max} \ge 0$, $k_l \ge 0$ and $0 < f_g < 1$; for INF-3 it asserts $w_{l,\text{const}} \ge 0$ twice and never checks $w_{g,\text{const}}$. `develop` raises `ValueError` for negative $w_{l,\max}$, $k_l$ and $w_{l,\text{const}}$.

## Sources

- Paper (10) and §2.3.
- Vogel (1968), "Inflow performance relationships for solution-gas drive wells", *Journal of Petroleum Technology* 20, 83–92.
- Guo, Lyons and Ghalambor (2007), *Petroleum production engineering: a computer-assisted approach*, Gulf Professional Publishing: Example Problem 3.2, a worked example of INF-1.
- Nikoofard, Aarsnes, Johansen and Kaasa (2017), "State and parameter estimation of a drift-flux model for underbalanced drilling operations", *IEEE Transactions on Control Systems Technology* 25.

## Test vectors

The rows INF-6 and INF-7, and INF-5, are in the row vectors (`discretization.md`): W1 has Vogel inflow with gas lift, W2 productivity-index inflow without. INF-8 is checked by `tests/test_inflow.py` and the example `scripts/sim_examples/fixed_rate.py`. The vectors below pin INF-9: the injection point on a uniform grid and on one whose cells grow towards the bottomhole, with $z_\text{inj}$ halfway between two points (1990 and 1970 m on the uniform grid) to pin the deeper of two equally near; and the phase rates of a black oil (API 35, $\gamma_g$ = 0.65, $R_{go}$ = 150) below, at and above the injection point $k$ of a vertical well of 1000 m on 10 cells, injected at $1000 - 100k$ m.

**Independent checks**, on `develop` at `f7f1218` (2026-10-29; `README.md`, Test vectors). Both backends reproduce the twelve rates of Guo, Lyons and Ghalambor's Example Problem 3.2 (2007) by INF-1, at the precision printed (`tests/test_worked_examples.py`). The other IDs are definitions and balances, with no independent source.

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### INF-1

| w_l_max | p | p_r | → w_l |
|---|---|---|---|
| 100.0 | 0.0 | 200.0 | 100.0 |
| 100.0 | 60.0 | 200.0 | 86.8 |
| 100.0 | 160.0 | 200.0 | 32.79999999999998 |
| 100.0 | 200.0 | 200.0 | 0.0 |
| 35.5 | 120.3 | 180.7 | 18.185894796997104 |

### INF-2

| k_l | p | p_r | → w_l |
|---|---|---|---|
| 0.5 | 150.0 | 170.0 | 10.0 |
| 1.2 | 80.5 | 200.0 | 143.4 |
| 0.5 | 170.0 | 170.0 | 0.0 |

### INF-3

| w_l_const | w_g_const | p | p_r | → w_l | → w_g |
|---|---|---|---|---|---|
| 20.0 | 3.0 | 100.0 | 200.0 | 20.0 | 3.0 |

### INF-4

| f_g | w_l | → w_g |
|---|---|---|
| 0.05 | 61.38775510204081 | 3.23093447905478 |
| 0.5 | 61.38775510204081 | 61.38775510204081 |
| 0.9 | 61.38775510204081 | 552.4897959183675 |
<!-- vectors:end -->

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### INF-9

| L | n_cells | exponent | z_inj | → k |
|---|---|---|---|---|
| 2000.0 | 100.0 | 1.0 | 1000.0 | 50.0 |
| 2000.0 | 100.0 | 1.0 | 1234.5 | 38.0 |
| 2000.0 | 100.0 | 1.0 | 1990.0 | 0.0 |
| 2000.0 | 100.0 | 1.0 | 1970.0 | 1.0 |
| 2000.0 | 100.0 | 1.0 | 2000.0 | 0.0 |
| 2000.0 | 100.0 | 1.0 | 0.0 | 100.0 |
| 3000.0 | 20.0 | 1.3 | 1500.0 | 8.0 |
| 3000.0 | 20.0 | 1.3 | 2900.0 | 1.0 |
| 3000.0 | 20.0 | 1.3 | 50.0 | 19.0 |

### INF-9 (phase rates)

| k | i | p | T | w_res | w_lg | → w_g | → w_l |
|---|---|---|---|---|---|---|---|
| 5.0 | 4.0 | 150.0 | 350.0 | 20.0 | 2.0 | 1.6228018123261068 | 21.19101557896899 |
| 5.0 | 5.0 | 150.0 | 350.0 | 20.0 | 2.0 | 3.6228017272789224 | 21.19101557896899 |
| 5.0 | 9.0 | 60.0 | 320.0 | 20.0 | 2.0 | 4.359212439215064 | 20.454604855375294 |
| 0.0 | 0.0 | 200.0 | 360.0 | 15.0 | 1.0 | 1.8990432884777912 | 16.211319771097855 |
| 8.0 | 3.0 | 250.0 | 370.0 | 10.0 | 3.0 | 0.3905595035973581 | 11.016349755130243 |
| 8.0 | 8.0 | 100.0 | 340.0 | 10.0 | 3.0 | 4.024232636855253 | 10.382676043888667 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| INF-1 | (10) | `inflow.py` `Vogel.mass_flow_rates` | vectors; rows |
| INF-2 | — | `inflow.py` `ProductivityIndex.mass_flow_rates` | vectors; rows |
| INF-3 | — | `inflow.py` `FixedFlowRate.mass_flow_rates` | vectors; spec-only: not implemented on develop, which has INF-8 in its place |
| INF-4 | (10) | `inflow.py` `Vogel.mass_flow_rates`, `ProductivityIndex.mass_flow_rates` | vectors; rows |
| INF-5 | (10) | `simulator.py` `_left_boundary_eqs` (`w_g += bc.w_lg`) | rows |
| INF-6 | (10) | `simulator.py` `_left_boundary_eqs` (`g1`) | rows |
| INF-7 | (10) | `simulator.py` `_left_boundary_eqs` (`g2`) | rows |
| INF-8 | — | — | property: tests/test_inflow.py |
| INF-9 | — | — | vectors; property: tests/test_model_properties.py (the total mass rate below and above the injection point) |
