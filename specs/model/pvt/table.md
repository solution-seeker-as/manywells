# PVT table

## Purpose

The hydrocarbon PVT of a fluid from a table (feature 018, `specs/features/018-pvt-tables.md`): the oil's solution gas–oil ratio and formation volume factor, the free gas's z-factor, and optionally the viscosities and the gas–oil surface tension, on a grid of pressures and temperatures, as a PVT report or a PVT program gives them. The table is one of the PVT models of `FluidModel` (`FluidModel(pvt=PVTTable(...))`), beside dead oil (PVT-OIL-1) and black oil (PVT-OIL-4). A fluid from an equation of state enters as a table (PVT-TAB-6); ManyWells holds no equation of state and no flash.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| a column at a point | $p$ (bar), $T$ (K) | $\hat y(p, T)$ | yes |
| solution gas–oil ratio, formation volume factor | $p$ (bar), $T$ (K) | $R_{so}$ (Sm³/Sm³), $B_o$ (–) | yes |
| gas law, gas density | $p$ (bar), $T$ (K), $\rho_g$ (kg/m³) | row (bar), $\rho_g$ (kg/m³) | yes |
| oil and gas viscosity, surface tension | $p$ (bar), $T$ (K), $\rho_g$ (kg/m³) | $\mu_o$, $\mu_g$ (Pa s), $\sigma_{go}$ (J/m²) | yes |
| the table from a flash's phase split | the separation, and $X$, $\rho_L$, $\rho_V$ at the nodes | the table | no |
| the table at the fluid's GOR | the tables at several GORs, $R_{go}$ | the table | no |

In code, `PVTTable` (`pvt/table.py`) is a `PVTModel` whose `gas_law` is `'table'`; `PVTTableSet` holds the tables of one stream at several GORs. In the Rust core, the fluid's `Pvt::Table` (`rust/src/pvt/fluid.rs`, `table.rs`) holds the table at the fluid's GOR, which `solvers/rust.py` passes it; the core never sees a GOR axis. `PVTTable.read_csv` reads either from a file, and `write_csv` writes one: `# key: value` lines (`rho_o` and `rho_g`, and the free-text `name`, `source`, `separation` and `gor`), then the columns `[gor,] p, T, R_so, B_o, Z[, mu_o, mu_g, sigma]`, one row per node, in Sm³/Sm³, bar, K and SI units.

## Equations

### PVT-TAB-1 · The table

A grid $p_1 < \dots < p_m$ (bar, $m \ge 2$) and $T_1 < \dots < T_n$ (K, $n \ge 1$), every node $(p_j, T_k)$ present, with $R_{so}$ (Sm³/Sm³), $B_o$ (–) and $Z$ (–) at each, and optionally $\mu_o$, $\mu_g$ (Pa s) and $\sigma_{go}$ (J/m²); and the densities at standard conditions $\rho_o$ and $\rho_{g,\text{sc}}$ (kg/m³) that they refer to.

- **Meaning, by the role in the balances.** Per Sm³ of stock-tank oil, the oil at $(p, T)$ carries the mass $\rho_o + R_{so}\rho_{g,\text{sc}}$ in the volume $B_o$ (PVT-OIL-9), and the free gas has the density $c_\text{bar}\thinspace p/(Z R_s T)$ with $R_s = p_\text{ref}/(\rho_{g,\text{sc}}T_\text{ref})$ (PVT-GAS-2, PVT-GAS-6). A table made from a flash may therefore have $R_{so} < 0$ and $B_o < 1$ where the free gas carries components that are oil at standard conditions (PVT-TAB-6). Above the fluid's bubble point $R_{so}$ is the table's GOR, and $Z$ is that of the gas that would appear first, so that the column is continuous.
- **Checks**, raising `ValueError` when the table is built: finite values; strictly increasing grids; $B_o > 0$, $Z > 0$, and $\mu_o, \mu_g, \sigma_{go} > 0$ where given; $R_{so}$ nondecreasing in $p$ at every $T$, to rounding (a fall of at most $10^{-12}\max(1, \lvert R_{so}\rvert)$, which PVT-TAB-7's arithmetic can make where $R_{so}$ is flat); and $R_{so} \ge 0$ unless the table has $\mu_o$ and $\sigma_{go}$, because the live-oil corrections PVT-OIL-11 and PVT-OIL-12 are undefined below $R_{so} = -17.8$ Sm³/Sm³ and untested below 0.
- **GOR axis** (optional). Tables of one stream at GORs $R_1 < \dots < R_l$, each with a grid of its own, all with the same $\rho_o$ and $\rho_{g,\text{sc}}$: the stream recombined from the same separator gas and oil, so that every GOR is on the same tie line at standard conditions.

### PVT-TAB-2 · Interpolation

A column $y$ at $(p, T)$ is

$$\hat y(p, T) = \text{bilinear interpolation of } y \text{ on the grid at } \tilde p = \min(\max(p, p_1), p_m), \quad \tilde T = \min(\max(T, T_1), T_n),$$

and with $n = 1$, linear in $\tilde p$ and independent of $T$. With $p_j \le \tilde p \le p_{j+1}$, $T_k \le \tilde T \le T_{k+1}$, $a = (\tilde p - p_j)/(p_{j+1} - p_j)$ and $b = (\tilde T - T_k)/(T_{k+1} - T_k)$:

$$\hat y = (1 - a)\big((1 - b)\thinspace y_{j,k} + b\thinspace y_{j,k+1}\big) + a\big((1 - b)\thinspace y_{j+1,k} + b\thinspace y_{j+1,k+1}\big).$$

It is exact at the nodes and continuous, and its derivatives jump at grid lines. The CasADi backend evaluates it with `casadi.interpolant(..., 'linear', ...)` on the clamped inputs, which agrees with the formula to rounding. The Rust core (`rust/src/pvt/table.rs`) does the interpolant's arithmetic, corner by corner in its order, so that the two backends agree to the bit (`tests/test_pvt_table.py`).

### PVT-TAB-3 · Oil

$$R_{so} = \hat R_{so}(p, T), \qquad B_o = \hat B_o(p, T),$$

in PVT-OIL-9, PVT-MIX-6 and PVT-OIL-13 as for the black oil: gas dissolves up to the smaller of $R_{so}$ and the gas available. Unlike the black oil's (PVT-OIL-15), the table's $R_{so}$ is not capped at the fluid's gas–oil ratio: $\hat R_{so}$ and $\hat B_o$ go together, and above the bubble point of the stream a table was made for, $\hat R_{so}$ is its GOR.

### PVT-TAB-4 · Gas law

$$c_\text{bar}\thinspace p = \hat Z(p, T)\thinspace\rho_g R_s T,$$

with the row $p - \rho_g\hat Z R_s T/c_\text{bar}$ (bar), PVT-GAS-3's form, and $R_s$ of $\rho_{g,\text{sc}}$ (PVT-GAS-6). The density at $(p, T)$ is explicit, $c_\text{bar}\thinspace p/(\hat Z R_s T)$. The Joule–Thomson factor is the Dranchuk–Abou-Kassem equation of state's (PVT-GAS-10) with the pseudo-critical properties of $\rho_{g,\text{sc}}$ (PVT-GAS-5), as for every real gas.

### PVT-TAB-5 · Viscosities and surface tension

$$\mu_o = \hat\mu_o(p, T), \qquad \mu_g = \hat\mu_g(p, T), \qquad \sigma_{go} = \hat\sigma_{go}(p, T)$$

where the table has the column; otherwise PVT-OIL-10 with PVT-OIL-11 at the API gravity of $\rho_o$ and the table's $R_{so}$, PVT-GAS-7 at the state's $(T, \rho_g)$ and the molar mass of $\rho_{g,\text{sc}}$, and PVT-OIL-3 at $\rho_o$ with PVT-OIL-12 at the table's $R_{so}$. They enter PVT-MIX-7 to PVT-MIX-9 as for the black oil. $\hat\mu_g$ is read at the point's pressure, not at its gas density.

### PVT-TAB-6 · A table from a flash

Python only; no row uses it. Given the stream's separation (stock-tank oil density $\rho_o$, gas density at standard conditions $\rho_{g,\text{sc}}$ and GOR $R_{go}$) and, at each node, the flash's vapour mass fraction $X$ of the hydrocarbon stream and the phase densities $\rho_V$ and $\rho_L$, with $m = \rho_o + R_{go}\rho_{g,\text{sc}}$ the stream's mass per Sm³ of stock-tank oil:

$$R_{so} = R_{go} - \frac{X m}{\rho_{g,\text{sc}}}, \qquad B_o = \frac{(1 - X) m}{\rho_L}, \qquad Z = \frac{c_\text{bar}\thinspace p}{\rho_V R_s T}.$$

At the GOR $R_{go}$, PVT-TAB-3 and PVT-TAB-4 then give the flash's phase mass rates and phase densities exactly at the nodes: without water or lift gas, PVT-OIL-13 gives $w_g/(w_g + w_l) = (R_{go} - R_{so})\rho_{g,\text{sc}}/m = X$. Where the stream is all liquid, the caller gives $\rho_V$ of the gas that would appear first (PVT-TAB-1).

### PVT-TAB-7 · The table at the fluid's GOR

Python only, when `FluidModel` is built. With a GOR axis and the fluid's GOR $R_{go}$ in $[R_i, R_{i+1}]$, each of the two GORs' columns is interpolated (PVT-TAB-2) onto the union of their two grids, and the result is

$$\hat y = (1 - w)\thinspace\hat y_i + w\thinspace\hat y_{i+1}, \qquad w = \frac{R_{go} - R_i}{R_{i+1} - R_i},$$

a table whose GOR is $R_{go}$, exact at the axis's GORs. A column only one of the two tables has is dropped. A GOR outside $[R_1, R_l]$ raises `ValueError`. `dataclasses.replace(fluid, gor=...)`, which calibration does for every row, reads the axis again.

## Options

| Option | IDs | Used by |
|---|---|---|
| PVT table | PVT-TAB-1 to PVT-TAB-5 | `develop` with `pvt=PVTTable(...)`, on both backends; neither configuration |
| A table from a flash | PVT-TAB-6 | `PVTTable.from_phase_split` |
| A GOR axis | PVT-TAB-7 | `pvt=PVTTableSet(...)` |

## Safeguards

- The checks of PVT-TAB-1.
- The clamping of PVT-TAB-2 keeps every property finite and positive at trial states far outside the well's pressures and temperatures, where linear extrapolation could make $B_o$ or $Z$ negative.
- `SSDFSimulator.root_set` refuses boundary conditions whose $p_r$ or $T_r$ lies outside the grid (`ValueError`), and a `Root` records whether any of its points lies off the grid (`off_table`), with a warning on the `manywells` logger for an operating point that does (specs/features/018-pvt-tables.md, ruling 3).

## Sources

- Feature spec `specs/features/018-pvt-tables.md`: the rulings, the design choices and their measurements.
- Whitson and Brulé (2000), *Phase behavior*, SPE Monograph 20: black-oil properties from an equation of state.

## Test vectors

The vectors are on two tables of `specs/tools/make_develop_vectors.py`: `TAB`, 3 × 3 nodes at 50, 150 and 300 bar and 300, 350 and 400 K with all six columns, $\rho_o = 850$ and $\rho_{g,\text{sc}} = 0.9$ kg/m³, and $R_{so} < 0$ at 50 bar and 400 K, whose nodes are the first nine rows of PVT-TAB-2; and `TAB_350`, its $R_{so}$, $B_o$ and $Z$ at 350 K alone. PVT-TAB-3 and PVT-TAB-4 are `FluidModel`'s with `pvt=TAB` and the water's defaults; PVT-TAB-5 is the PVT model's, on `TAB` and on `TAB_350`, whose missing columns the correlations stand in for.

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### PVT-TAB-2

| p | T | → R_so | → B_o | → Z | → mu_o | → mu_g | → sigma |
|---|---|---|---|---|---|---|---|
| 50.0 | 300.0 | 22.0 | 1.09 | 0.9 | 0.004 | 1.2e-05 | 0.02 |
| 50.0 | 350.0 | 15.5 | 1.11 | 0.93 | 0.0022 | 1.3e-05 | 0.017 |
| 50.0 | 400.0 | -2.0 | 0.99 | 0.95 | 0.0016 | 1.45e-05 | 0.014 |
| 150.0 | 300.0 | 78.0 | 1.21 | 0.78 | 0.0024 | 1.6e-05 | 0.011 |
| 150.0 | 350.0 | 66.0 | 1.23 | 0.85 | 0.0014 | 1.6e-05 | 0.009 |
| 150.0 | 400.0 | 54.0 | 1.24 | 0.9 | 0.001 | 1.7e-05 | 0.007 |
| 300.0 | 300.0 | 120.0 | 1.29 | 0.85 | 0.0019 | 2.5e-05 | 0.006 |
| 300.0 | 350.0 | 120.0 | 1.32 | 0.89 | 0.0011 | 2.2e-05 | 0.005 |
| 300.0 | 400.0 | 112.0 | 1.33 | 0.93 | 0.0008 | 2.1e-05 | 0.004 |
| 100.0 | 325.0 | 45.375 | 1.1600000000000001 | 0.8650000000000001 | 0.0025 | 1.425e-05 | 0.01425 |
| 225.0 | 380.0 | 87.0 | 1.2810000000000001 | 0.8970000000000001 | 0.00104 | 1.9e-05 | 0.0060999999999999995 |
| 60.0 | 390.0 | 6.99 | 1.0364 | 0.9404000000000001 | 0.0016560000000000001 | 1.4460000000000002e-05 | 0.013880000000000002 |
| 150.0 | 375.0 | 60.0 | 1.2349999999999999 | 0.875 | 0.0012000000000000001 | 1.65e-05 | 0.008 |
| 20.0 | 325.0 | 18.75 | 1.1 | 0.915 | 0.0031000000000000003 | 1.2499999999999999e-05 | 0.018500000000000003 |
| 450.0 | 360.0 | 118.4 | 1.322 | 0.8980000000000001 | 0.0010400000000000001 | 2.18e-05 | 0.0048000000000000004 |
| 225.0 | 250.0 | 99.0 | 1.25 | 0.815 | 0.00215 | 2.05e-05 | 0.0085 |
| 10.0 | 420.0 | -2.0 | 0.99 | 0.95 | 0.0016 | 1.45e-05 | 0.014 |

### PVT-TAB-2 (one temperature)

| p | T | → R_so | → B_o | → Z |
|---|---|---|---|---|
| 50.0 | 350.0 | 15.5 | 1.11 | 0.93 |
| 150.0 | 350.0 | 66.0 | 1.23 | 0.85 |
| 300.0 | 350.0 | 120.0 | 1.32 | 0.89 |
| 100.0 | 350.0 | 40.75 | 1.17 | 0.89 |
| 100.0 | 300.0 | 40.75 | 1.17 | 0.89 |
| 225.0 | 420.0 | 93.0 | 1.275 | 0.87 |
| 20.0 | 350.0 | 15.5 | 1.11 | 0.93 |
| 450.0 | 280.0 | 120.0 | 1.32 | 0.89 |

### PVT-TAB-3

| gor | wlr | p | T | w_res | w_lg | → w_g | → w_l | → rho_l |
|---|---|---|---|---|---|---|---|---|
| 120.0 | 0.0 | 100.0 | 325.0 | 20.0 | 0.0 | 1.580294275847693 | 20.960882352938928 | 767.9633620689654 |
| 120.0 | 0.3 | 225.0 | 380.0 | 20.0 | 2.0 | 2.4647213022439134 | 21.225174074857186 | 793.4653630818082 |
| 100.0 | 0.0 | 300.0 | 300.0 | 20.0 | 0.0 | 0.0005000000041986414 | 22.11764705881513 | 742.6356589147287 |
| 120.0 | 0.0 | 50.0 | 400.0 | 20.0 | 0.0 | 2.5835295085329273 | 19.957647058822154 | 856.7676767676768 |
| 150.0 | 0.2 | 20.0 | 450.0 | 10.0 | 1.0 | 2.2438909072586477 | 9.983633015843724 | 885.4637096774195 |

### PVT-TAB-4

| p | T | rho_g | → Z | → row |
|---|---|---|---|---|
| 100.0 | 325.0 | 80.0 | 0.8650000000000001 | 12.129128347504192 |
| 225.0 | 380.0 | 150.0 | 0.8970000000000001 | 25.23344612181154 |
| 50.0 | 400.0 | 12.0 | 0.95 | 32.183584938400145 |
| 450.0 | 250.0 | 300.0 | 0.85 | 200.9218289085546 |
| 300.0 | 300.0 | 0.0 | 0.85 | 300.0 |

### PVT-TAB-5

| p | T | rho_g | → mu_o | → mu_g | → sigma |
|---|---|---|---|---|---|
| 100.0 | 325.0 | 80.0 | 0.0025 | 1.425e-05 | 0.01425 |
| 225.0 | 380.0 | 150.0 | 0.00104 | 1.9e-05 | 0.0060999999999999995 |
| 450.0 | 250.0 | 300.0 | 0.0019 | 2.5e-05 | 0.006 |

### PVT-TAB-5 (correlations)

| p | T | rho_g | → mu_o | → mu_g | → sigma |
|---|---|---|---|---|---|
| 100.0 | 325.0 | 80.0 | 0.0017379325048626469 | 1.4061387246398283e-05 | 0.013249859064624465 |
| 225.0 | 380.0 | 150.0 | 0.0004917941780190083 | 1.9887756149387046e-05 | 0.004257895706659146 |
| 300.0 | 350.0 | 200.0 | 0.0005801547595244463 | 2.2915770294592255e-05 | 0.0035559733005370467 |
| 20.0 | 300.0 | 10.0 | 0.008936193297739157 | 1.0845744639213252e-05 | 0.02127850005549101 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| PVT-TAB-1 | — | — | property: tests/test_pvt_table.py (each check raises, and the CSV round trip) |
| PVT-TAB-2 | — | — | vectors; property: tests/test_pvt_table.py (exact at the nodes, the formula between them, clamped outside, one temperature) |
| PVT-TAB-3 | — | — | vectors; property: tests/test_pvt_table.py (mass conservation of the phase rates) |
| PVT-TAB-4 | — | — | vectors; property: tests/test_pvt_table.py (the row is zero at the density) |
| PVT-TAB-5 | — | — | vectors; property: tests/test_pvt_table.py (the correlations where columns are missing) |
| PVT-TAB-6 | — | — | property: tests/test_pvt_table.py (a phase split from the black oil gives back its columns) |
| PVT-TAB-7 | — | — | property: tests/test_pvt_table.py (exact at the axis's GORs, linear between, refused outside) |
