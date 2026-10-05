# Gas

## Purpose

The gas phase's equation of state, the standard conditions that volumetric rates and gas densities at standard conditions refer to, and, for `develop`, the real-gas z-factor (the Dranchuk–Abou-Kassem equation of state or Papay's correlation), the gas's Joule–Thomson factor and the gas viscosity.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| gas density | $p$ (bar), $T$ (K), $R_s$ (J/(kg K)) | $\rho_g$ (kg/m³) | yes, in $p$ and $T$ |
| density at standard conditions | $R_s$ | $\rho_{g,\text{sc}}$ (kg/m³) | no |
| Joule–Thomson factor | $T$ (K), $\rho_g$ (kg/m³) | $J$ (–) | yes |

v1.0.0's code takes $p$ in Pa in `pvt.gas_density`; the vectors below are in bar.

In `develop` the gas is parameterized by its density at standard conditions $\rho_{g,\text{sc}}$ (`FluidModel.rho_g`), from which PVT-GAS-6 gives the specific gravity and $R_s$; `FluidModel.ideal_gas` chooses PVT-GAS-1 or a real gas, and `FluidModel.z_factor_model` the real gas's law: `'dak'` (PVT-GAS-11, the default) or `'papay'` (PVT-GAS-3 with PVT-GAS-4). `FluidModel.jt_factor(T, rho_g)` gives PVT-GAS-10. The correlations of PVT-GAS-4, PVT-GAS-5 and PVT-GAS-7 take $p$ in Pa internally; `FluidModel`'s methods take bar.

## Equations

### PVT-GAS-1 · Ideal gas law

$$c_\text{bar}\thinspace p = \rho_g R_s T$$

with $p$ in bar. $R_s$ is constant per well. Its row in DISC-6 is $p - \rho_g R_s T / c_\text{bar}$ (bar). Used by `v1.0.0`.

### PVT-GAS-2 · Standard conditions

Standard conditions are $p_\text{ref} = 101\thinspace 325$ Pa and $T_\text{ref} = 288.15$ K (ISO 13443), and the gas density there is

$$\rho_{g,\text{sc}} = \frac{p_\text{ref}}{R_s T_\text{ref}}.$$

It is not part of the discretized system. The datasets use it for standard volumetric gas rates (`docs/datasets.md`), and `develop` parameterizes the gas by $\rho_{g,\text{sc}}$ instead of $R_s$, so it converts between the two (`specs/sampling.md`).

### PVT-GAS-3 · Real gas law

$$c_\text{bar}\thinspace p = Z(p, T) \rho_g R_s T$$

with the z-factor of PVT-GAS-4 and $p$ in bar. Its row in DISC-11 is $p - \rho_g Z R_s T / c_\text{bar}$ (bar), the canonical form of PVT-GAS-1 with $Z$; with $Z = 1$ it is PVT-GAS-1. The form matters to the solver, not to the roots: in bar, like the momentum row, it lets Ipopt converge tightly on long grids, where the density form $\rho_g - c_\text{bar} p/(Z R_s T)$ left the phase mass rates drifting by up to $1.3\cdot10^{-6}$ of the total rate at $N = 400$ (verifier case set, 2026-10-01).

### PVT-GAS-4 · Papay z-factor

Papay (1968):

$$Z = 1 - \frac{3.52 p_{pr}}{10^{0.9813 T_{pr}}} + \frac{0.274 p_{pr}^2}{10^{0.8157 T_{pr}}}, \qquad p_{pr} = \frac{p}{p_{pc}}, \quad T_{pr} = \frac{T}{T_{pc}},$$

explicit and smooth, with the pseudo-critical properties of PVT-GAS-5. Valid for $p_{pr} < 6$ and $T_{pr} > 1.05$.

### PVT-GAS-5 · Sutton pseudo-critical properties

Sutton (1985), from the gas specific gravity $\gamma_g$:

$$p_{pc} = 756.8 - 131.07\gamma_g - 3.6\gamma_g^2\ \text{psia}, \qquad T_{pc} = 169.2 + 349.5\gamma_g - 74.0\gamma_g^2\ \text{°R},$$

converted to Pa ($1\ \text{psi} = 6894.76$ Pa) and K ($T_{pc}/1.8$).

### PVT-GAS-6 · Gas gravity and gas constant

$$\gamma_g = \frac{\rho_{g,\text{sc}} R_u T_\text{ref}}{p_\text{ref}\thinspace M_\text{air}}, \qquad M_g = M_\text{air}\thinspace\gamma_g, \qquad R_s = \frac{R_u}{M_g},$$

with $R_u = 8314.46$ J/(kmol K) and $M_\text{air} = 28.97$ kg/kmol. It agrees with PVT-GAS-2: $R_s = p_\text{ref}/(\rho_{g,\text{sc}} T_\text{ref})$.

### PVT-GAS-7 · Gas viscosity

Lee, Gonzalez and Eakin (1966), with $T$ in °R, $\rho_g$ in g/cm³ and $M_g$ in kg/kmol:

$$\mu_g = 10^{-7} K \exp\negthinspace\left(X \rho_g^{Y}\right)\ \text{Pa s}, \qquad K = \frac{(9.4 + 0.02 M_g) T^{1.5}}{209 + 19 M_g + T}, \quad X = 3.5 + \frac{986}{T} + 0.01 M_g, \quad Y = 2.4 - 0.2X.$$

The source gives $10^{-4}K\exp(\cdot)$ in cP. Used by friction (PVT-MIX-9).

### PVT-GAS-8 · Gas formation volume factor

$$B_g = \frac{Z}{Z_\text{ref}}\frac{p_\text{ref}\thinspace T}{T_\text{ref}\thinspace p},$$

the reservoir volume of a unit standard volume. Not used by the simulator; the black-oil consistency tests use it.

### PVT-GAS-9 · Dranchuk–Abou-Kassem equation of state

Dranchuk and Abou-Kassem (1975), Eq. (2), a generalized Starling equation of state with its eleven constants fitted to the Standing–Katz z-factor chart. $Z$ is an explicit function of the reduced density $\rho_r$ and the pseudo-reduced temperature $t = T_{pr}$:

$$Z(\rho_r, t) = 1 + c_1\rho_r + c_2\rho_r^2 - c_3\rho_r^5 + c_4\rho_r^2(1 + A_{11}\rho_r^2)e^{-A_{11}\rho_r^2},$$

$$c_1 = A_1 + \frac{A_2}{t} + \frac{A_3}{t^3} + \frac{A_4}{t^4} + \frac{A_5}{t^5}, \quad c_2 = A_6 + \frac{A_7}{t} + \frac{A_8}{t^2}, \quad c_3 = A_9\left(\frac{A_7}{t} + \frac{A_8}{t^2}\right), \quad c_4 = \frac{A_{10}}{t^3},$$

with $A_1, \dots, A_{11}$ = 0.3265, −1.0700, −0.5339, 0.01569, −0.05165, 0.5475, −0.7361, 0.1844, 0.1056, 0.6134, 0.7210, and the reduced density of their Eq. (3), $\rho_r = Z_c p_{pr}/(Z t)$ with $Z_c = 0.270$. With $Z = c_\text{bar}\thinspace p/(\rho_g R_s T)$,

$$\rho_r = \frac{Z_c\rho_g R_s T_{pc}}{p_{pc}},$$

a scaled gas density, with the pseudo-critical properties of PVT-GAS-5 ($p_{pc}$ in Pa). Fitted to 1,500 points of the Standing–Katz chart: an average absolute error in $Z$ of 0.585% against the original chart and 0.307% against the smoothed one (their Table 1), and 0.486% with $Z$ as a function of $t$ and $\rho_r$, the form used here. Recommended for $0.2 \le p_{pr} < 30$ with $1.0 < T_{pr} \le 3.0$, and for $p_{pr} < 1.0$ with $0.7 < T_{pr} \le 1.0$; its accuracy is unacceptable at $T_{pr} = 1.0$ with $p_{pr} \ge 1.0$.

### PVT-GAS-10 · Joule–Thomson factor

$$J = T\left(\frac{\partial \ln Z}{\partial T}\right)_p \quad [-].$$

For an ideal gas (PVT-GAS-1), $J = 0$. For a real gas, whatever its z-factor, $J$ is PVT-GAS-9's at the state's $\rho_g$ and $T$:

$$J = \frac{t Z_t - \rho_r Z_\rho}{Z + \rho_r Z_\rho},$$

$$Z_\rho = \frac{\partial Z}{\partial \rho_r} = c_1 + 2c_2\rho_r - 5c_3\rho_r^4 + 2c_4\rho_r\thinspace(1 + A_{11}\rho_r^2 - A_{11}^2\rho_r^4)e^{-A_{11}\rho_r^2},$$

$$t Z_t = t\frac{\partial Z}{\partial t} = \tilde c_1\rho_r + \tilde c_2\rho_r^2 - \tilde c_3\rho_r^5 + \tilde c_4\rho_r^2(1 + A_{11}\rho_r^2)e^{-A_{11}\rho_r^2},$$

with $\tilde c_1 = -A_2/t - 3A_3/t^3 - 4A_4/t^4 - 5A_5/t^5$, $\tilde c_2 = -A_7/t - 2A_8/t^2$, $\tilde c_3 = -A_9(A_7/t + 2A_8/t^2)$ and $\tilde c_4 = -3A_{10}/t^3$. Along an isobar, $p_{pr} = \rho_r t Z/Z_c$ gives $d\rho_r/dt = -\rho_r(Z + tZ_t)/\big(t(Z + \rho_r Z_\rho)\big)$, and $J = (t/Z)(Z_t + Z_\rho\thinspace d\rho_r/dt)$ simplifies to the form above. The Joule–Thomson coefficient is $\mu_{JT} = J/(\rho_g c_{pg})$ (Hasan and Kabir 2018, §6.4.2, with $V = ZR_sT/p$). With PVT-GAS-11 the factor is the gas law's own; with Papay's PVT-GAS-4 it is DAK's at Papay's density, whose derivative has the wrong sign above about 300 bar (`specs/features/016-joule-thomson.md`).

### PVT-GAS-11 · Real gas law with the Dranchuk–Abou-Kassem equation of state

$$c_\text{bar}\thinspace p = Z(\rho_r, T_{pr}) \rho_g R_s T$$

with $Z$ and $\rho_r$ of PVT-GAS-9 and $p$ in bar. $Z$ is explicit in the state's $\rho_g$ and $T$, so its row in DISC-11, $p - \rho_g Z R_s T / c_\text{bar}$ (bar), the canonical form of PVT-GAS-3, needs no inner solve. The density at a given $(p, T)$ is the root of $\rho_r t Z(\rho_r, t) = Z_c p_{pr}$, by Newton's method from the ideal-gas density $\rho_r = Z_c p_{pr}/t$: it converges to $10^{-12}$ in at most 17 steps for $1.05 \le T_{pr} \le 3$ and $p_{pr} \le 30$. The CasADi backend takes 20 steps, unrolled, so that the density accepts symbols; the Rust core steps until the step is at most $10^{-13}$ of $\rho_r$, at most 50 steps. Used by `develop`'s default.

## Options

| Option | IDs | Used by |
|---|---|---|
| Ideal gas | PVT-GAS-1 | `v1.0.0`; `develop` with `ideal_gas=True` |
| Real gas, Dranchuk–Abou-Kassem equation of state (Sutton pseudo-critical properties) | PVT-GAS-5, PVT-GAS-9, PVT-GAS-11 | `develop` default (`z_factor_model='dak'`) |
| Real gas with Papay's z-factor (Sutton pseudo-critical properties) | PVT-GAS-3 to PVT-GAS-5 | `develop` with `z_factor_model='papay'` |

PVT-GAS-2 and PVT-GAS-6 apply to every option, PVT-GAS-7 to the friction of `develop` (FRIC-3), and PVT-GAS-10 to the Joule–Thomson term of the energy balance (THM-8).

The heat capacity $c_{pg}$ is a constant parameter (THM-1). The lift gas is the same gas as the produced gas (paper §2.3).

## Safeguards

None in the equations. Nothing checks that $p_{pr}$ and $T_{pr}$ are in PVT-GAS-4's range; at high $p_{pr}$ its $Z$ has a minimum and then rises steeply, and at 460 bar it is 11% (methane) to 29% (gravity 0.80) above the reference equations of state (`specs/features/016-joule-thomson.md`). Nothing checks PVT-GAS-9's range either: below $T_{pr} = 1.05$, near the critical point, $Z + \rho_r Z_\rho$ (proportional to $(\partial p/\partial \rho_g)_ T$) can change sign, which gives PVT-GAS-10 a pole and can stop PVT-GAS-11's Newton iteration from converging; the Rust core then returns NaN, which fails the state.

## Sources

- Paper (9) and Table 3 (standard reference conditions).
- Papay (1968), "A termelési technológiai paraméterek változása a gáztelepek művelése során", *OGIL Műszaki Tudományos Közlemények*, 267–273.
- Sutton (1985), "Compressibility factors for high-molecular-weight reservoir gases", SPE 14265.
- Lee, Gonzalez and Eakin (1966), "The viscosity of natural gases", *Journal of Petroleum Technology* 18, 997–1000.
- Dranchuk and Abou-Kassem (1975), "Calculation of Z factors for natural gases using equations of state", *Journal of Canadian Petroleum Technology* 14(3), 34–36, doi:10.2118/75-03-03: Eqs. (2) and (3), Table 1 and the recommended range.
- Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., Society of Petroleum Engineers, §6.4.2, for the Joule–Thomson coefficient of a real gas.
- Setzmann and Wagner (1991), "A new equation of state and tables of thermodynamic properties for methane covering the range from the melting line to 625 K at pressures up to 1000 MPa", *Journal of Physical and Chemical Reference Data* 20, 1061–1155, through CoolProp 8.0.0: the reference values of `tests/test_pvt.py`.
- Feature specs `specs/features/005-fluid-model.md`, `006-real-gas.md` and `016-joule-thomson.md`.

## Test vectors

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### PVT-GAS-1

| p | T | R_s | → rho_g |
|---|---|---|---|
| 1.01325 | 288.15 | 518.3 | 0.6784483329203721 |
| 150.0 | 373.15 | 518.3 | 77.55800052268924 |
| 35.0 | 330.0 | 320.0 | 33.14393939393939 |

### PVT-GAS-2

| R_s | → rho_g_sc |
|---|---|
| 518.3 | 0.6784483329203721 |
| 320.0 | 1.0988742842269652 |
| 420.0 | 0.8372375498872117 |
<!-- vectors:end -->

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### PVT-GAS-3

| p | T | rho_g_sc | → rho_g |
|---|---|---|---|
| 1.01325 | 288.15 | 0.8 | 0.8025111150985167 |
| 100.0 | 350.0 | 0.8 | 72.63595630996694 |
| 250.0 | 380.0 | 0.68 | 130.09617761579543 |
| 30.0 | 290.0 | 1.0 | 33.344112095524885 |

### PVT-GAS-4, PVT-GAS-5

| p | T | sg_gas | → p_pc | → T_pc | → Z |
|---|---|---|---|---|---|
| 1.01325 | 288.15 | 0.554 | 47.09688673190624 | 188.95067555555553 | 0.9975929753172854 |
| 100.0 | 350.0 | 0.65 | 46.200649124600005 | 202.8388888888889 | 0.895826953447206 |
| 250.0 | 380.0 | 0.75 | 45.262203340999996 | 216.5 | 0.9409016075663112 |
| 300.0 | 330.0 | 0.9 | 43.845226739599994 | 235.45 | 0.9075327281455864 |

### PVT-GAS-9

| r | t | → Z |
|---|---|---|
| 0.05 | 1.5 | 0.9733647730404927 |
| 0.3 | 1.2 | 0.7714024426949179 |
| 0.8 | 1.6 | 0.8236809977281965 |
| 1.5 | 2.0 | 1.4411784463734922 |
| 2.0 | 1.3 | 1.6527260493786702 |

### PVT-GAS-10

| r | t | → J |
|---|---|---|
| 0.05 | 1.5 | 0.09051321008840159 |
| 0.3 | 1.2 | 1.1970493175757286 |
| 0.8 | 1.6 | 0.8597373961969282 |
| 1.5 | 2.0 | -0.1953037059866058 |
| 2.0 | 1.3 | -0.460766640637631 |

### PVT-GAS-11

| p | T | rho_g_sc | → rho_g | → Z |
|---|---|---|---|---|
| 1.01325 | 288.15 | 0.8 | 0.8020816013038209 | 0.9974047512118002 |
| 100.0 | 350.0 | 0.8 | 72.92845002685182 | 0.8913062885985571 |
| 250.0 | 380.0 | 0.68 | 131.37576835398573 | 0.9683939493287386 |
| 450.0 | 420.0 | 0.75 | 200.94491951713266 | 1.1372330376689157 |
| 30.0 | 290.0 | 1.0 | 33.109262072807454 | 0.8885375108222243 |

### PVT-GAS-6

| rho_g_sc | → sg_gas | → M_g | → R_s |
|---|---|---|---|
| 0.6783 | 0.5536169542027575 | 16.038283163253883 | 518.4133435834127 |
| 0.8 | 0.6529464298425562 | 18.915858072538853 | 439.5497136907862 |
| 1.1 | 0.8978013410335149 | 26.00930484974093 | 319.6725190478444 |

### PVT-GAS-7

| T | rho_g | M_g | → mu_g |
|---|---|---|---|
| 288.15 | 0.68 | 16.04 | 1.1127036746568543e-05 |
| 350.0 | 80.0 | 18.8 | 1.5377485303874373e-05 |
| 380.0 | 200.0 | 22.0 | 2.377124205445594e-05 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| PVT-GAS-1 | (9) | `simulator.py` `_closure_relations` (`g2`); `pvt.py` `gas_density` | vectors; rows |
| PVT-GAS-2 | Table 3 | `pvt.py` `P_REF`, `T_REF`, `gas_density`, `specific_gas_constant` | vectors |
| PVT-GAS-3 | — | — | vectors; rows (in the v1.0.0 configuration, as PVT-GAS-1) |
| PVT-GAS-4 | — | — | vectors; property: tests/test_pvt.py |
| PVT-GAS-5 | — | — | vectors |
| PVT-GAS-6 | — | — | vectors; property: tests/test_fluid.py |
| PVT-GAS-7 | — | — | vectors; property: tests/test_pvt.py |
| PVT-GAS-8 | — | — | property: tests/test_black_oil_consistency.py |
| PVT-GAS-9 | — | — | vectors; property: tests/test_pvt.py |
| PVT-GAS-10 | — | — | vectors; property: tests/test_pvt.py |
| PVT-GAS-11 | — | — | vectors; property: tests/test_pvt.py |
