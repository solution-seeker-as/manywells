# Inflow

## Purpose

The rates at which reservoir fluid enters the well at the bottomhole and lift gas at its injection point, and the two rows that impose them on the state at point 0 (paper §2.3, after Nikoofard et al., 2017). An oil well's inflow gives the liquid rate; a gas well's gives the gas rate, and its liquid comes with the gas (INF-10, INF-11).

## Interface

| Function | Inputs | Outputs | Symbolic |
|---|---|---|---|
| reservoir inflow | $p_0$ (bar), $p_r$ (bar), the model's parameters; the fluid's $f_g$ for INF-10 | $w_l$, $w_{g,\text{res}}$ (kg/s) | yes, in $p_0$ |
| inflow rows | state at point 0, $w_{lg}$ | two rows (kg/s) | yes |
| injection point | $z_\text{inj}$ (m), the grid's measured depths | $k$ | no |

In `develop` an inflow model returns the liquid rate only, `liquid_mass_flow_rate(p_0, p_r, fluid=None)`, which is $w_\text{res}$ in `discretization.md`; the fluid model gives the reservoir gas rate (INF-4, `FluidModel.reservoir_gas_rate`) and the phase rates at the bottomhole (INF-5 with dead oil, PVT-OIL-13 with black oil, `FluidModel.phase_rates`). The liquid inflows ignore the fluid. The gas-rate inflow, `Forchheimer`, needs it: it returns INF-11 of its INF-10 (`FluidModel.reservoir_liquid_rate`), so that every row downstream sees the same $w_\text{res}$, and INF-4 gives its gas rate back; `Forchheimer.gas_mass_flow_rate(p_0, p_r)` is INF-10 alone. Every caller passes the well's fluid. The injection depth is the well's `WellProperties.injection_md`, and `WellProperties.injection_point` is $k$ (INF-9); the discretization gives each point's phase rates the lift gas from point $k$ up.

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

### INF-10 · Forchheimer gas inflow

Forchheimer's law for stabilised radial flow of a real gas, in the pressure squared (Golan and Whitson 1991, Eqs. 2.44 and 2.51 to 2.53; Economides et al. 1994, Eqs. 4-32 and 4-33): with $\Delta = p_r^2 - p_0^2$ (bar²), the reservoir gas's mass rate at standard conditions is the root of $\Delta = a\thinspace w_{g,\text{res}} + b\thinspace w_{g,\text{res}}^2$ that is zero at $\Delta = 0$:

$$w_{g,\text{res}} = \frac{2\Delta}{a + \sqrt{a^2 + 4b\thinspace\lvert\Delta\rvert}}$$

with the Darcy coefficient $a > 0$ in bar² s/kg and the non-Darcy coefficient $b \ge 0$ in bar² s²/kg², constants of the well. For $\Delta \ge 0$ this is the positive root of the quadratic, $(-a + \sqrt{a^2 + 4b\Delta})/(2b)$, written without the cancellation that form has at small $b\Delta$, and without its division by zero at $b = 0$, where it is $\Delta/a$. For $\Delta < 0$, above $p_r$, it is the odd extension (Safeguards). The code evaluates $\Delta$ as $(p_r - p_0)(p_r + p_0)$, which keeps its precision next to $p_r$, where the Rust core's search brackets roots down to $10^{-6}$ bar of drawdown.

Its slope, $dw_{g,\text{res}}/dp_0 = -2p_0/\sqrt{a^2 + 4b\thinspace\lvert\Delta\rvert}$, is finite and negative for every $p_0 > 0$, and $-2p_r/a$ at $p_0 = p_r$: the inflow is smooth in $p_0$ below $p_r$, continuously differentiable through it, and needs no smoothing function. The pressure-squared form holds where $\mu_g Z$ is nearly constant between $p_0$ and $p_r$, below about 138 bar (`validity.md`, note 12); the coefficients are inputs, which a sampler or a user fits to the well's pressure range. `Forchheimer(a, b)`; `Forchheimer.from_gas_rate(a_q, b_q, rho_g)` converts coefficients in bar² per Sm³/d of separator gas, $a = (86\thinspace 400/\rho_{g,\text{sc}})\thinspace a_q$ and $b = (86\thinspace 400/\rho_{g,\text{sc}})^2 b_q$, with the fluid's $\rho_{g,\text{sc}}$ (`specs/features/022-gas-rate-inflow.md`, Units). Not in the paper.

### INF-11 · Reservoir liquid rate from the gas rate

For INF-10, the liquid comes with the gas in the fluid's ratios at standard conditions, INF-4 read backwards:

$$w_\text{res} = \frac{1 - f_g}{f_g}\thinspace w_{g,\text{res}} = \frac{\rho_o + \rho_w\thinspace\text{WOR}}{\rho_{g,\text{sc}} R_{go}}\thinspace w_{g,\text{res}}$$

with $f_g$ of PVT-MIX-10 and $\text{WOR} = \text{wlr}/(1 - \text{wlr})$. In a gas well's terms, with the condensate–gas ratio $\text{CGR} = 1/R_{go}$ and the water–gas ratio $\text{WGR} = \text{WOR}/R_{go}$ (both Sm³/Sm³), $w_\text{res} = (\rho_o\thinspace\text{CGR} + \rho_w\thinspace\text{WGR})\thinspace w_{g,\text{res}}/\rho_{g,\text{sc}}$. It needs $f_g > 0$, a fluid with gas. INF-4 then gives back $w_{g,\text{res}}$, and INF-5, PVT-OIL-13 and INF-9 split the stream at each point as for any inflow: a gas well and an oil well with the same fluid and the same $w_\text{res}$ have the same rows. `FluidModel.reservoir_liquid_rate`. Not in the paper.

## Options

| Option | IDs | Used by |
|---|---|---|
| Vogel | INF-1, INF-4 | the published datasets |
| Productivity index | INF-2, INF-4 | v1.0.0's default |
| Fixed rate | INF-3 | none; `develop` has INF-8 in its place |
| Fixed liquid rate | INF-8, INF-4 | `develop`'s `FixedFlowRate`; not in the `v1.0.0` configuration |
| Forchheimer gas rate | INF-10, INF-11 | `develop`'s `Forchheimer`, accepted by `v2.0.0`'s check; not in the `v1.0.0` configuration |

The lift gas's injection point is a well option of `develop`: at the bottomhole (`injection_md=None`, the default) it is INF-5, and above it INF-9. The `v1.0.0` configuration injects at the bottomhole.

All of them use INF-5 to INF-7 (with black oil, PVT-OIL-13 in place of INF-5). On `develop`, the inflow model returns $w_l$ only and the fluid model supplies $f_g$ (PVT-MIX-10); the gas-rate inflow takes $f_g$ from the fluid too, for INF-11.

## Safeguards

None for INF-1 to INF-9. Above $p_0 = p_r$ the inflow rates are negative, which no admissible state has (SOL-1). v1.0.0 asserts $w_{l,\max} \ge 0$, $k_l \ge 0$ and $0 < f_g < 1$; for INF-3 it asserts $w_{l,\text{const}} \ge 0$ twice and never checks $w_{g,\text{const}}$. `develop` raises `ValueError` for negative $w_{l,\max}$, $k_l$ and $w_{l,\text{const}}$.

Above $p_r$ ($\Delta < 0$) INF-10 is the odd extension, $-w_{g,\text{res}}(\lvert\Delta\rvert)$: defined, finite and strictly increasing in $\Delta$ everywhere, with the slope $1/a$ at $\Delta = 0$ from both sides. Without it the quadratic's root is not real above $p_0^2 = p_r^2 + a^2/(4b)$, which a trial state of Ipopt within its bound relaxation, or a central difference at a root next to $p_r$, could reach. Its second derivative jumps at $p_0 = p_r$, where no root lies (SOL-2). `develop` raises `ValueError` for $a \le 0$, $b < 0$ or a coefficient that is not finite (`Forchheimer.__post_init__`), and for a gas-rate inflow on a fluid with $f_g = 0$, outside INF-11's range (`WellProperties.__post_init__`, which holds both). The Rust core refuses the same (`WellSpec::check`).

## Sources

- Paper (10) and §2.3.
- Vogel (1968), "Inflow performance relationships for solution-gas drive wells", *Journal of Petroleum Technology* 20, 83–92.
- Guo, Lyons and Ghalambor (2007), *Petroleum production engineering: a computer-assisted approach*, Gulf Professional Publishing: Example Problem 3.2, a worked example of INF-1.
- Nikoofard, Aarsnes, Johansen and Kaasa (2017), "State and parameter estimation of a drift-flux model for underbalanced drilling operations", *IEEE Transactions on Control Systems Technology* 25.
- Forchheimer (1901), "Wasserbewegung durch Boden", *Zeitschrift des Vereines deutscher Ingenieure* 45, 1781–1788: not in hand; as Golan and Whitson (1991, Eq. 2.36) give it.
- Golan and Whitson (1991), *Well performance*, 2nd ed., Prentice Hall: §2.4, pp. 132–149, Eqs. 2.36, 2.37, 2.44 and 2.51 to 2.62, Examples 2.4 and 2.6. INF-10.
- Economides, Hill and Ehlig-Economides (1994), *Petroleum production systems*, Prentice Hall: §4-3 and §4-4, Eqs. 4-23 to 4-34 and 4-51 to 4-54, Examples 4-7 and 4-8. INF-10.
- Rawlins and Schellhardt (1936), *Back-pressure data on natural-gas wells and their application to production practices*, US Bureau of Mines Monograph 7: the backpressure law, not taken (`specs/features/022-gas-rate-inflow.md`, design choice 1).

## Test vectors

The rows INF-6 and INF-7, and INF-5, are in the row vectors (`discretization.md`): W1 has Vogel inflow with gas lift, W2 productivity-index inflow without; the fifth well of `vectors/v2_rows.json` has the gas-rate inflow, INF-10 and INF-11. INF-8 is checked by `tests/test_inflow.py` and the example `scripts/sim_examples/fixed_rate.py`. The vectors below pin INF-9: the injection point on a uniform grid and on one whose cells grow towards the bottomhole, with $z_\text{inj}$ halfway between two points (1990 and 1970 m on the uniform grid) to pin the deeper of two equally near; and the phase rates of a black oil (API 35, $\gamma_g$ = 0.65, $R_{go}$ = 150) below, at and above the injection point $k$ of a vertical well of 1000 m on 10 cells, injected at $1000 - 100k$ m. INF-10's pin its closed form at $p_r$, at 0.1 bar of drawdown, at a typical drawdown and at the AOF (1.01325 bar), in the linear limit $b = 0$, where $b\thinspace w_{g,\text{res}} \gg a$, and above $p_r$, with coefficients at about the 5th, 50th and 95th percentiles of the sampler draft's gas wells; INF-11's the liquid at four gas mass fractions, as INF-4's, and in a gas well's terms for a wet gas, a condensate-like fluid with water, and a gas with much water.

**Independent checks**, on `develop` at `f7f1218` (2026-10-29; `README.md`, Test vectors). Both backends reproduce the twelve rates of Guo, Lyons and Ghalambor's Example Problem 3.2 (2007) by INF-1, at the precision printed (`tests/test_worked_examples.py`). INF-10 (feature 022), in both backends: the Medicine Lodge well of Golan and Whitson's Example 2.4 (1991, Table E2.4; $A$ = 0.1095 psia²/(scf/D), $B$ = 3.58·10⁻⁹ psia²/(scf/D)², $p_R$ = 1623 psia), its AOF of 15.8 MMscf/D and its 1.99 MMscf/D at 1550 psia, the two of the table's seven rows that the feature spec quotes; and the AOF of Economides et al.'s Example 4-7 (1994; $a$ = 1.5·10⁴ and $b$ = 68.5 in psi² and Mscf/d at $\bar p^2$ = 2.13·10⁷ psi²), printed as 460 Mscf/d to two figures. The other IDs are definitions and balances, with no independent source.

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

### INF-10

| a | b | p | p_r | → w_g |
|---|---|---|---|---|
| 3500.0 | 810.0 | 270.0 | 270.0 | 0.0 |
| 3500.0 | 810.0 | 269.9 | 270.0 | 0.015371034954400302 |
| 3500.0 | 810.0 | 200.0 | 270.0 | 4.568920525353095 |
| 3500.0 | 810.0 | 1.01325 | 270.0 | 7.569175544226396 |
| 3500.0 | 0.0 | 200.0 | 270.0 | 9.4 |
| 90.0 | 17.0 | 30.0 | 60.0 | 10.23045839940567 |
| 110000.0 | 32000.0 | 500.0 | 660.0 | 1.2399831009234341 |
| 90.0 | 32000.0 | 1.01325 | 270.0 | 1.5079296615610196 |
| 3500.0 | 810.0 | 270.1 | 270.0 | -0.015376708865097002 |
| 3500.0 | 810.0 | 300.0 | 270.0 | -2.916793311490393 |

### INF-11

| f_g | w_g | → w_l |
|---|---|---|
| 0.05 | 3.8 | 72.19999999999999 |
| 0.5 | 3.8 | 3.8 |
| 0.9 | 3.8 | 0.4222222222222226 |
| 0.99 | 30.0 | 0.3030303030303033 |

### INF-11 (gas well)

| gor | wlr | rho_o | rho_g | rho_w | w_g | → w_l |
|---|---|---|---|---|---|---|
| 20000.0 | 0.5 | 760.0 | 0.75 | 1025.0 | 3.8 | 0.4522000000000001 |
| 3000.0 | 0.3 | 780.0 | 0.85 | 1025.0 | 3.8 | 1.8169747899159663 |
| 100000.0 | 0.9 | 750.0 | 0.7 | 1025.0 | 30.0 | 4.275 |
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
| INF-10 | — | — | vectors; rows (`vectors/v2_rows.json`); property: tests/test_inflow.py, tests/test_model_properties.py (the reservoir rates at $p_0$ of solved gas wells) |
| INF-11 | — | — | vectors; rows (`vectors/v2_rows.json`); property: tests/test_inflow.py, tests/test_model_properties.py |
