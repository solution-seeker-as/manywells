# Oil

## Purpose

Oil properties: the dead-oil model of `v1.0.0`, API gravity, and the dead-oil surface tension correlation; for `develop`, the black-oil model (Vazquez–Beggs), oil viscosity, the live-oil surface tension, and the gas that dissolves into the oil.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| API gravity | $\rho$ (kg/m³) | API (degrees) | no |
| dead-oil surface tension | $\rho$ (kg/m³), $T$ (K) | $\sigma_{od}$ (J/m²) | yes, in $T$, and in $\rho$ where $\rho$ is a state (PVT-MIX-5) |
| solution gas–oil ratio, formation volume factor | $p$ (bar), $T$ (K) | $R_{so}$ (Sm³/Sm³), $B_o$ (–) | yes |
| oil viscosity | $T$ (K); $R_{so}$ for live oil | $\mu_o$ (Pa s) | yes |
| phase rates | $p$ (bar), $T$ (K), $w_\text{res}$, $w_{lg}$ (kg/s) | $(w_g, w_l)$ (kg/s) | yes |

The black-oil correlations are in field units: $p$ in psia, $T$ in °F, $R_{so}$ in scf/STB ($1\ \text{scf/STB} = 0.178108$ Sm³/Sm³), viscosities in cP. In `develop`, `FluidModel(oil_model='black_oil')` evaluates them through `BlackOilPVT`, whose methods take Pa; the oil is given by its density at standard conditions $\rho_o$ (API by PVT-OIL-2, between 10 and 40), the gas by $\rho_{g,\text{sc}}$ (gravity $\gamma_g$ by PVT-GAS-6), and the separator by $p_\text{sep}$ and $T_\text{sep}$ (bar and K; standard conditions by default).

## Equations

### PVT-OIL-1 · Dead oil

The oil is incompressible, with a constant density $\rho_o$ and heat capacity $c_{po}$, and no gas dissolves in it (BAL-3). v1.0.0's simulator has no oil phase of its own: the oil enters only through the liquid mixing of `pvt/mixture.md`, which the sampler applies (`specs/sampling.md`). Used by `v1.0.0`.

### PVT-OIL-2 · API gravity

$$\text{API} = \frac{141.5}{\text{SG}} - 131.5, \qquad \text{SG} = \frac{\rho}{\rho_{w,\text{ref}}}, \qquad \rho_{w,\text{ref}} = 999.1\ \text{kg/m³}.$$

$\rho$ is a density at standard conditions. The inverse is $\rho = 141.5\,\rho_{w,\text{ref}}/(\text{API} + 131.5)$.

### PVT-OIL-3 · Dead-oil surface tension

Abdul-Majeed and Abu Al-Soof (2000), Eqs. (1)–(3), with $T$ in °C ($T_C = T - 273.15$) and the result converted from dyn/cm to J/m²:

$$\sigma_{od}(\rho, T) = 10^{-3}\,\big(1.11591 - 0.00305\,T_C\big)\big(38.085 - 0.259\,\text{API}(\rho)\big).$$

The coefficients are the source's, which gives the correlation in °C. It was fitted to dead-oil data at 15.6, 37.8 and 54.4 °C and API gravities 15 to 50; ManyWells evaluates it up to the reservoir temperature, 150 °C at most (`specs/sampling.md`, SMP-14), and down to API 10 (PVT-MIX-5). Which density it is evaluated at is set by PVT-MIX-5 or PVT-MIX-7.

### PVT-OIL-4 · Black oil

Gas dissolves into the oil up to the solution gas–oil ratio $R_{so}(p, T)$ of PVT-OIL-6, capped at the bubble point by PVT-OIL-7, and the oil swells by the formation volume factor $B_o(p, T)$ of PVT-OIL-8. With dead oil (PVT-OIL-1) $R_{so} = 0$ and $B_o = 1$.

### PVT-OIL-5 · Separator gas gravity

The gas gravity corrected to a reference separator at 114.7 psia, Vazquez and Beggs (1980):

$$\gamma_{gs} = \gamma_g \left[1 + 5.912\cdot10^{-5}\,\text{API}\; T_\text{sep}\, \log_{10}\!\left(\frac{p_\text{sep}}{114.7}\right)\right],$$

with $T_\text{sep}$ in °F and $p_\text{sep}$ in psia; at the reference separator, 114.7 psia, $\gamma_{gs} = \gamma_g$. At standard separator conditions (14.7 psia, 59 °F) and API 35 it is $0.891\,\gamma_g$. Decided by Bjarne, 2026-10-01: $\log_{10}$, as in the source. Until then `develop` had the natural log, which gave $0.75\,\gamma_g$ there and an $R_{so}$ 16% below the source's (Step 7 finding; `specs/features/007-black-oil.md`).

### PVT-OIL-6 · Solution gas–oil ratio

Vazquez and Beggs (1980):

$$R_{so} = C_1\,\gamma_{gs}\, p^{C_2} \exp\!\left(\frac{C_3\,\text{API}}{T + 460}\right)\ \text{scf/STB},$$

with $p$ in psia and $T$ in °F; $(C_1, C_2, C_3) = (0.0362, 1.0937, 25.7240)$ for API $\le 30$ and $(0.0178, 1.1870, 23.9310)$ above.

### PVT-OIL-7 · Bubble-point cap

If a bubble-point pressure $p_b$ is given, $R_{so} = \operatorname{smin}\big(R_{so}(p), R_{so}(p_b)\big)$ (SMO-2, $\epsilon = 10^{-6}$ in (scf/STB)²), so that no more gas dissolves above $p_b$. Without $p_b$ there is no cap: gas dissolves up to the gas available (PVT-OIL-13).

### PVT-OIL-8 · Oil formation volume factor

Vazquez and Beggs (1980), with $R_{so}$ in scf/STB (capped by PVT-OIL-7) and $T$ in °F:

$$B_o = 1 + C_4 R_{so} + (C_5 + C_6 R_{so})\,(T - 60)\,\frac{\text{API}}{\gamma_{gs}},$$

with $(C_4, C_5, C_6) = (4.677\cdot10^{-4}, 1.751\cdot10^{-5}, -1.811\cdot10^{-8})$ for API $\le 30$ and $(4.670\cdot10^{-4}, 1.100\cdot10^{-5}, 1.337\cdot10^{-9})$ above.

### PVT-OIL-9 · Live-oil density

$$\rho_{lo} = \frac{\rho_o + R_{so}\,\rho_{g,\text{sc}}}{B_o}$$

with $R_{so}$ in Sm³/Sm³: the stock-tank oil and its dissolved gas, in the swollen volume.

### PVT-OIL-10 · Dead-oil viscosity

Beggs and Robinson (1975), with $T$ in °F:

$$\mu_{od} = 10^{X} - 1\ \text{cP}, \qquad X = 10^{\,3.0324 - 0.02023\,\text{API}}\; T^{-1.163}.$$

The code's stated range is API 10 to 58 and 100 to 295 °F (38 to 146 °C); wellhead temperatures are often colder.

### PVT-OIL-11 · Live-oil viscosity

Beggs and Robinson (1975), with $R_{so}$ in scf/STB:

$$\mu_o = A\,\mu_{od}^{\,B}, \qquad A = 10.715\,(R_{so} + 100)^{-0.515}, \qquad B = 5.44\,(R_{so} + 150)^{-0.338},$$

in cP. With dead oil, $\mu_o = \mu_{od}$.

### PVT-OIL-12 · Live-oil surface tension

Abdul-Majeed and Abu Al-Soof (2000), Eqs. (4) and (5), with $R_{so}$ in Sm³/Sm³:

$$\sigma_{lo} = (1 - b)\,\frac{\sigma_{od}}{1 + 0.02549\,R^{1.0157}} + b\cdot 32.0436\,\sigma_{od}\,R^{-1.1367}, \qquad b = \frac{1}{1 + e^{-0.5\,(R_{so} - 50)}},$$

where $R = \operatorname{smax}(R_{so}, 10^{-6})$ (SMO-1) and $\sigma_{od}$ is PVT-OIL-3. The source switches from (4) to (5) at 50 Sm³/Sm³; the sigmoid (SMO-4) blends them over a few Sm³/Sm³. The two branches do not meet there: $\sigma_{lo}/\sigma_{od}$ is 0.425 by (4) and 0.375 by (5), a step of 12% that the blend smooths (`plans/develop_model_changes.md`, change 3; the code's docstring said they meet, and is corrected).

### PVT-OIL-13 · Dissolved gas and phase rates

With the reservoir liquid rate $w_\text{res}$ (INF-1, INF-2 or INF-8), the reservoir gas rate $w_{g,\text{res}}$ of INF-4 and the lift-gas rate $w_{lg}$, the gas dissolved at $(p, T)$ and the phase mass rates are

$$w_d = \operatorname{smin}\!\left(R_{so}\,\frac{\rho_{g,\text{sc}}}{\rho_o}\, x_o\, w_\text{res},\ w_{g,\text{res}}\right), \qquad w_g = \operatorname{smax}\big(w_{g,\text{res}} + w_{lg} - w_d,\ 0\big), \qquad w_l = w_\text{res} + w_d,$$

with $R_{so}$ in Sm³/Sm³, $x_o$ the oil mass fraction of the liquid at standard conditions (PVT-MIX-10), and SMO-1 and SMO-2 with $\epsilon = 10^{-6}$ (kg/s)². Only reservoir gas dissolves, not the lift gas. With dead oil the phase rates are INF-5's exactly, with no smoothing: the smooth min of PVT-OIL-13 would be $-\epsilon/(4 w_{g,\text{res}})$ at $R_{so} = 0$, not 0 (`plans/develop_model_changes.md`, change 8). The total $w_g + w_l$ is $w_{g,\text{res}} + w_{lg} + w_\text{res}$ wherever neither smoothing is active.

### PVT-OIL-14 · Bubble-point pressure

Standing (1947), with $R_{so}$ in scf/STB and $T$ in °F:

$$p_b = 18.2\left[\left(\frac{R_{so}}{\gamma_g}\right)^{0.83} 10^{\,0.00091\,T - 0.0125\,\text{API}} - 1.4\right]\ \text{psia},$$

or the given $p_b$ if there is one. Not used by the simulator.

## Options

| Option | IDs | Used by |
|---|---|---|
| Dead oil | PVT-OIL-1 | `v1.0.0`; `develop` with `oil_model='dead_oil'` |
| Black oil: Vazquez–Beggs $R_{so}$ and $B_o$, live-oil density, dissolved gas | PVT-OIL-4 to PVT-OIL-9, PVT-OIL-13 | `develop` default |
| Oil viscosity, for friction | PVT-OIL-10, and PVT-OIL-11 with black oil | `develop`'s `RoughnessFriction` |
| Live-oil surface tension | PVT-OIL-12 | `develop` default (PVT-MIX-7) |

## Safeguards

- Neither v1.0.0 function checks that the API gravity or the temperature is in the correlation's range; $\sigma_{od}$ is positive for $T_C < 365$ °C and API $< 147$.
- `BlackOilPVT` raises `ValueError` for API gravities outside 10 to 40, the Vazquez–Beggs range, and for $\gamma_g \le 0$. Nothing checks $p$ or $T$ against the correlations' ranges, or caps $R_{so}$ (`specs/sampling.md`, SMP-43).
- The smooth maxima and minima of PVT-OIL-7, PVT-OIL-12 and PVT-OIL-13.

## Sources

- Paper §4.2 (oil density range) and Appendix A.1 (surface tension).
- Abdul-Majeed and Abu Al-Soof (2000), "Estimation of gas–oil surface tension", *Journal of Petroleum Science and Engineering* 27, 197–200 (`papers/`).
- Vazquez and Beggs (1980), "Correlations for fluid physical property prediction", *Journal of Petroleum Technology* 32, 968–970.
- Beggs and Robinson (1975), "Estimating the viscosity of crude oil systems", *Journal of Petroleum Technology* 27, 1140–1141.
- Standing (1947), "A pressure-volume-temperature correlation for mixtures of California oils and gases", *Drilling and Production Practice*, API, 275–287.
- Feature specs `specs/features/007-black-oil.md` and `008-dissolved-gas.md`.

## Test vectors

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### PVT-OIL-2

| rho | → API |
|---|---|
| 825.0 | 39.860787878787875 |
| 850.0 | 34.82076470588237 |
| 925.0 | 21.335297297297302 |
| 999.1 | 10.0 |

### PVT-OIL-3

| rho | T | → sigma |
|---|---|---|
| 850.0 | 293.15 | 0.030662459169966468 |
| 825.0 | 373.15 | 0.022511717871813944 |
| 925.0 | 330.0 | 0.030687576200264993 |
| 999.1 | 280.0 | 0.038867646162500005 |
<!-- vectors:end -->

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### PVT-OIL-5

| api | sg_gas | p_sep | T_sep | → sg_gas_corr |
|---|---|---|---|---|
| 35.0 | 0.65 | 1.01325 | 288.15 | 0.5791873523243499 |
| 25.0 | 0.8 | 7.0 | 310.0 | 0.7938397327000366 |
| 40.0 | 0.6 | 20.0 | 300.0 | 0.6459273955521256 |

### PVT-OIL-6, PVT-OIL-8

| api | sg_gas | p | T | → R_so | → B_o |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 100.0 | 350.0 | 39.23670154718769 | 1.1781815984812276 |
| 35.0 | 0.65 | 250.0 | 380.0 | 104.83573364079275 | 1.3919288441591005 |
| 25.0 | 0.8 | 150.0 | 340.0 | 60.77433418919686 | 1.1950400219480732 |
| 15.0 | 0.7 | 50.0 | 320.0 | 11.297046521590264 | 1.050381834524833 |

### PVT-OIL-7

| api | sg_gas | p_bubble | p | T | → R_so |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 150.0 | 100.0 | 350.0 | 39.23670154686072 |
| 35.0 | 0.65 | 150.0 | 150.0 | 350.0 | 63.49100570561024 |
| 35.0 | 0.65 | 150.0 | 250.0 | 350.0 | 63.49109475946041 |

### PVT-OIL-9

| api | sg_gas | p | T | → rho_lo |
|---|---|---|---|---|
| 35.0 | 0.65 | 100.0 | 350.0 | 747.1961049371042 |
| 25.0 | 0.8 | 250.0 | 380.0 | 770.4809662042255 |

### PVT-OIL-10

| api | T | → mu_od |
|---|---|---|
| 35.0 | 320.0 | 0.005847118155861538 |
| 25.0 | 350.0 | 0.0061514549464027355 |
| 15.0 | 380.0 | 0.008733133689549666 |

### PVT-OIL-11

| mu_od | R_so_scf | → mu_o |
|---|---|---|
| 0.005 | 0.0 | 0.0050013935968335425 |
| 0.005 | 500.0 | 0.0010595700009963913 |
| 0.02 | 1000.0 | 0.0013102756841238225 |

### PVT-OIL-12

| sigma_od | R_so_scf | → sigma_lo |
|---|---|---|
| 0.03 | 0.0 | 0.029999735704159265 |
| 0.03 | 112.29141868978374 | 0.01955225422144745 |
| 0.03 | 252.6556920520134 | 0.013464465673189444 |
| 0.03 | 280.7285467244593 | 0.012000160003766185 |
| 0.03 | 308.80140139690525 | 0.010252515325955499 |
| 0.03 | 842.185640173378 | 0.003230722247543799 |

### PVT-OIL-13

| api | sg_gas | gor | wlr | p | T | w_res | w_lg | → w_g | → w_l |
|---|---|---|---|---|---|---|---|---|---|
| 35.0 | 0.65 | 150.0 | 0.0 | 100.0 | 350.0 | 20.0 | 0.0 | 2.0777847635751927 | 20.736032593985804 |
| 35.0 | 0.65 | 150.0 | 0.3 | 250.0 | 380.0 | 20.0 | 2.0 | 2.5632072227295097 | 21.30732037378895 |
| 35.0 | 0.65 | 20.0 | 0.0 | 250.0 | 380.0 | 20.0 | 1.0 | 1.0000004070927844 | 20.375175474539187 |
| 25.0 | 0.8 | 300.0 | 0.5 | 30.0 | 320.0 | 5.0 | 0.0 | 0.7440738546852541 | 5.02875439561121 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| PVT-OIL-1 | §2.2, §4.2 | none in the simulator; `pvt.py` `LiquidProperties` | spec-only: v1.0.0's simulator has no oil phase, and the oil enters through PVT-MIX-2 to PVT-MIX-4, which have vectors |
| PVT-OIL-2 | §4.2 | `pvt.py` `api_from_density` | vectors |
| PVT-OIL-3 | App. A.1 | `pvt.py` `dead_oil_surface_tension` | vectors |
| PVT-OIL-4 | — | — | property: tests/test_black_oil.py; spec-only: the composition of PVT-OIL-6 to PVT-OIL-8, which have vectors |
| PVT-OIL-5 | — | — | vectors; property: tests/test_black_oil.py (hand values from the source's formula) |
| PVT-OIL-6 | — | — | vectors; property: tests/test_black_oil.py |
| PVT-OIL-7 | — | — | vectors; property: tests/test_black_oil.py |
| PVT-OIL-8 | — | — | vectors; property: tests/test_black_oil.py |
| PVT-OIL-9 | — | — | vectors; property: tests/test_black_oil_consistency.py |
| PVT-OIL-10 | — | — | vectors; property: tests/test_pvt.py |
| PVT-OIL-11 | — | — | vectors; property: tests/test_pvt.py |
| PVT-OIL-12 | — | — | vectors; property: tests/test_pvt.py |
| PVT-OIL-13 | — | — | vectors; property: tests/test_fluid.py (mass conservation, exact dead-oil rates) |
| PVT-OIL-14 | — | — | property: tests/test_black_oil.py |
