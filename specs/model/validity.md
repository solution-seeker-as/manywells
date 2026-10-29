# Validity ranges

*Owner: Bjarne Grimstad. Status: signed off by Bjarne on 2026-10-29, the rules and decisions 1 to 4 (Decisions); decision 5, on horizontal sections, is open until the sampling redesign decides it with the sampling distribution. Any change to a rule or a range needs his sign-off.*

Each correlation in the model was fitted to, or tested on, data over a range of conditions. This file gives that range for every correlation in one table, with the range today's sampler reaches and a rule for what the model does outside it. It is the starting point of the sampling redesign, which chooses realistic ranges for the v2 datasets. A new option adds its rows here (`AGENTS.md`, Adding an equation). The equations are in the component files; this file defines no IDs.

## Rules

Each row takes one of three rules for a value outside its range.

- **Refuse.** The input's `__post_init__` raises `ValueError`: `BlackOil` for the API gravity (note 2), `WellProperties` for the relative roughness (note 3). The Rust core checks both again when its well is built (`WellSpec::check`), so that a well given to it directly is refused as the dataclasses refuse it. Only for inputs fixed before the solve (well and fluid parameters, boundary conditions), and only where the correlation is known to fail outside the range or the input lies outside what the option models.
- **Extrapolate.** The correlation is evaluated as it stands. This is the rule for every quantity that depends on the state: $p$, $T$, the Reynolds number, $R_{so}$, the pseudo-reduced properties and the flow rates. The solvers evaluate the rows at trial states far from any root, where a refusal would stop the search, and clamping would change the model at its roots and put kinks in the rows. It is also the rule for an input whose correlation degrades gradually outside its range.
- **Clamp.** The value is moved to the nearest end of the range. Only for a PVT table (PVT-TAB-2), whose linear extrapolation can make $B_o$ or $Z$ negative. The table also refuses $p_r$ and $T_r$ off its grid and marks a root that leaves it (`off_table`, `pvt/table.md`).

There is no fourth rule that flags a root outside a range (decision 1).

Two constraints hold for every row:

- **The `v1.0.0` configuration refuses nothing that v1.0.0 accepts**, so that the verifier's cases and the published datasets' wells stay valid. Its correlations extrapolate, as in v1.0.0.
- **A refusal is checked once, when the input is built.** In a calibration, a step into a refused region then gives rows without an operating point, which the barrier rejects (CAL-9), instead of a valley of a correlation outside its range (`specs/features/017-calibration.md`, Findings 4). The calibration builds the well at each step and takes a refused one as rows without an operating point (`Objective.wells`).

## The table

Published ranges are in the source's units, with the spec's units in brackets. *Sampler* is `develop`'s sampler today (`specs/sampling.md`) on 500 wells (seed 2026), each at its nominal operating point and at two sampled ones: 1494 cases, of which 1453 have an operating point. It gives the range of an input over those cases, and the range of a state quantity over every grid point of their operating points. *Outside* is the share of the 1453 cases with the input, or any point of the operating point, outside the published range. `plans/evidence/validity_ranges.py` measures both. The notes are below the table.

| Correlation | IDs | Input | Published range | Sampler | Outside | Rule |
|---|---|---|---|---|---|---|
| Vazquez–Beggs $R_{so}$ and $B_o$ | PVT-OIL-5, PVT-OIL-6, PVT-OIL-8 | API gravity, $\gamma_g$, $p$, $T$, $R_{so}$ | not stated in the source; Beggs–Robinson's below until it is checked (note 1) | as below | as below | API gravity: Refuse outside 16 to 58 (note 2). The rest: Extrapolate |
| Beggs–Robinson oil viscosity | PVT-OIL-10, PVT-OIL-11 | API gravity | 16 to 58 | 21.4 to 39.8 | 0% | Refuse, with black oil (note 2) |
| | | $p$ | 0 to 5250 psig [1.0 to 363 bar] | 12 to 439 bar | 7.0% | Extrapolate |
| | | $T$ | 70 to 295 °F [294 to 419 K] | 278 to 423 K | 6.3% | Extrapolate |
| | | $R_{so}$ | 20 to 2070 scf/STB [3.6 to 369 Sm³/Sm³] | 1.6 to 232 Sm³/Sm³ | 0.4% | Extrapolate |
| Black oil, no oil in the gas | PVT-OIL-4, PVT-OIL-15 | gas–oil ratio $R_{go}$ | black and volatile oils; gas condensates have 3000 to 150,000 scf/STB [534 to 26,700 Sm³/Sm³] and 40 to 60 °API (Whitson and Brulé 2000, §2) | 1.6 to 356,000 Sm³/Sm³, median 1149 | 69% above 534 | Extrapolate (decision 4) |
| Abdul-Majeed dead-oil surface tension | PVT-OIL-3 | $T$ | 15.6, 37.8 and 54.4 °C [289 to 328 K] | 278 to 423 K | 100% | Extrapolate (note 4) |
| | | API gravity | 15 to 50 | `develop`: 21.4 to 39.8, at $\rho_o$; `v1.0.0`: down to 10.0, at $\rho_l$ (PVT-MIX-5) | 0% | Extrapolate |
| Abdul-Majeed live-oil correction | PVT-OIL-12 | $R_{so}$ | about 3 to 208 Sm³/Sm³, at 37.8 °C and 13.8 to 206.8 bar | 1.6 to 232 Sm³/Sm³ | 0.4% | Extrapolate (note 4) |
| Water viscosity | PVT-WAT-3 | $T$ | no source in hand (note 5) | 278 to 423 K | – | Extrapolate |
| Sutton pseudo-critical properties | PVT-GAS-5 | $\gamma_g$ | 0.571 to 1.679 | 0.552 to 0.895 | 9.1% | Extrapolate (note 6) |
| Dranchuk–Abou-Kassem $Z$ and $J$ | PVT-GAS-9, PVT-GAS-10, PVT-GAS-11 | $p_{pr}$, $T_{pr}$ | $0.2 \le p_{pr} < 30$ with $1.0 < T_{pr} \le 3.0$ | $p_{pr}$ 0.26 to 9.6, $T_{pr}$ 1.24 to 2.22 | 0% | Extrapolate |
| Papay $Z$ (option) | PVT-GAS-4 | $p_{pr}$, $T_{pr}$ | not found (note 7) | $p_{pr}$ up to 9.6 | 33% above $p_{pr} = 6$ | Extrapolate |
| Lee–Gonzalez–Eakin gas viscosity | PVT-GAS-7 | $p$ | 100 to 8000 psia [6.9 to 552 bar] | 12 to 439 bar | 0% | Extrapolate |
| | | $T$ | 100 to 340 °F [311 to 444 K] | 278 to 423 K | 9.0% | Extrapolate |
| Five regimes, Hasan et al. (2010); the fitted classifier is fitted to the same hierarchy | SLIP-12 to SLIP-16; SLIP-2 to SLIP-8 | gas–liquid ratio | 104 to 5615 scf/STB [18.5 to 1000 Sm³/Sm³] | 1.6 to 132,700 Sm³/Sm³ | 42% | Extrapolate (note 8) |
| | | liquid rate | 100 to 27,760 STB/D [16 to 4413 Sm³/d] | 11 to 5811 Sm³/d | 1.9% | Extrapolate |
| | | water cut | 0 to 98% | 0 to 99.4% | 0.8% | Extrapolate |
| | | bottomhole pressure, measured depth | 108 to 13,755 psig [8.5 to 949 bar]; 1635 to 30,650 ft [498 to 9342 m] | 55 to 439 bar; 1513 to 6490 m | 0% | – |
| | | inclination | vertical and deviated, no range stated; no horizontal or stratified flow | 0 to 60°, and 90° in the horizontal sections of L-shaped wells | 24% with horizontal cells | Extrapolate (note 9); horizontal sections are open (decision 5) |
| Chen friction factor | FRIC-4 | Reynolds number | 4000 to $4\cdot10^{8}$ | 17 to $1.6\cdot10^{7}$ | 1.2% | Extrapolate (note 11) |
| | | $\varepsilon/D$ | $5\cdot10^{-7}$ to 0.05 | $9.1\cdot10^{-6}$ to $1.8\cdot10^{-3}$ | 0% | Refuse above 0.05 (note 3) |
| Haaland friction factor (option) | FRIC-5 | Reynolds number | 4000 to $10^{8}$ | as Chen | 1.2% | Extrapolate |
| | | $\varepsilon/D$ | 0 to 0.05 | as Chen | 0% | Refuse above 0.05 (note 3) |
| Vogel inflow | INF-1 | the reservoir | solution-gas drive below the bubble point; not where oil, gas and water all flow | every sampled well, with water cuts up to 99% and gas–oil ratios up to 356,000 Sm³/Sm³ | – | Extrapolate (note 10) |
| Simpson choke multiplier | CHK-5 | $x_g$, $\rho_l/\rho_g$ | not in hand | – | – | Extrapolate |
| PVT table | PVT-TAB-2 | $p$, $T$ | the table's grid | – | – | Clamp; refuse $p_r$ and $T_r$ off the grid (decided, `specs/features/018-pvt-tables.md`) |

The rest of the model has no published range: the balances, the ideal gas, the mixing rules and the mixture viscosity, the rise velocities (SLIP-4, SLIP-5, SLIP-10), the heat loss and the other thermal terms (Zhang et al. 2006 state no range for $h$), the choke equation, its profiles and the critical pressure ratio, and the productivity index. Their inputs carry only the physical checks of the component files' Safeguards sections.

## Notes

1. **Vazquez–Beggs's range.** Their paper (1980) states no range: more than 600 PVT analyses from fields worldwide, more than 6000 measurements of $R_{so}$, $B_o$ and $\mu_o$ together, and 5008 points for $R_{so}$. The ranges commonly quoted for it disagree with one another ($R_{so}$ up to 2070 or 2199 scf/STB, API gravity 16 to 58 or 15 to 60) and trace to Vazquez's thesis (University of Tulsa, 1976), which is not in hand. Until it is checked, this table takes Beggs and Robinson's ranges (1975, Table 1), which the most common quotation matches.
2. **API gravity.** Until 2026-10-29 `BlackOilPVT` refused API gravities outside 10 to 40, which `pvt/oil.md` called the Vazquez–Beggs range; no source in hand gives that range. Beggs and Robinson's data run from 16 to 58, and the black oil refuses API gravities outside 16 to 58 (decision 2). This refuses heavy oils from 10 to 16, which were accepted and are never sampled, and accepts light oils from 40 to 58, which a gas condensate's stock-tank liquid needs (40 to 60 °API). The coefficients' split at 30 °API stays. `BlackOil`, the PVT model a fluid takes, checks it, and `BlackOilPVT`, which evaluates the Vazquez–Beggs correlations for it, no longer does: the range is the black oil's as a whole, set by Beggs and Robinson's viscosity, which `BlackOil` applies, and `BlackOilPVT` evaluates the correlations at any API gravity, as the correlation functions do. A dead oil (PVT-OIL-1) checks no API gravity, and nor does the `v1.0.0` configuration. Heavy oils do occur: Valkó and McCain's 1745 oils run from 6.0 to 63.7 °API (2003, Table 2.1). A heavy-oil well needs a PVT table.
3. **Relative roughness.** Chen's and Haaland's $1/\sqrt{f}$ is a logarithm of a sum that $(\varepsilon/D)/3.7$ dominates at large $\varepsilon/D$. So $f$ has a pole at $\varepsilon/D \approx 3.7$ and falls again beyond it, to 0.042 at $\varepsilon/D = 1000$ (Re from $10^{5}$ to $10^{6}$): a plausible friction factor at an impossible roughness. This is the valley in which 017's fit stopped, at a roughness of 143 m. `WellProperties`, which holds both $\varepsilon$ and $D$, refuses $\varepsilon/D$ above 0.05 with `RoughnessFriction` (decision 3). The sampler reaches 0.0018, and corroded tubing ($1.5\cdot10^{-4}$ m) in the smallest sampled tubing (3 inches) gives 0.002. In a calibration the bound lies 3.8 to 4.5 prior standard deviations above the roughness's median (CAL-6), so it constrains no sensible fit. Below $5\cdot10^{-7}$, Chen's lower end, the pipe is smooth, and the correlations extrapolate there without harm.
4. **The surface tension's temperature.** Abdul-Majeed and Abu Al-Soof fitted the dead-oil tension at 15.6, 37.8 and 54.4 °C, and the live-oil correction at 37.8 °C only, with $R_{so}$ computed by Lasater's correlation rather than measured. Every sampled case has points above 54.4 °C, up to 150 °C. The formula is linear in $T$ and reaches zero at 366 °C. The tension enters the slip only through fourth roots (SLIP-4, SLIP-12): at 150 °C it is 0.69 times its value at 54.4 °C, and the rise velocities are 0.91 times theirs. The live-oil correction depends on the pressure only through $R_{so}$, which stays in range, so its pressure range (61% of the cases have points above 206.8 bar) matters less.
5. **Water viscosity.** `pvt/water.md` gives no source for PVT-WAT-3, and no source in hand has its constants. It enters only friction, through the Reynolds number. A source is a follow-up.
6. **Sutton's gas gravity.** Sutton's data run from 0.571 to 1.679, from gas condensates rich in heptanes-plus, although the paper calls the relations suitable for all reservoir gases. 9.1% of the sampled cases lie below 0.571, down to 0.552, just below methane's 0.554 (SMP-11's $R_s$ up to 520 J/(kg K)). At 0.554 Sutton gives 683 psia and 340 °R, against methane's critical point at 667 psia and 343 °R. Realistic separator gases run from 0.555 to 1.685 (Valkó and McCain, Table 2.1).
7. **Papay's range.** `pvt/gas.md` says Papay's z-factor is valid for $p_{pr} < 6$ and $T_{pr} > 1.05$. Papay (1968) is not in hand, and no source in hand states a range for it. Feature 016 measured its $Z$ at 11% to 29% above the reference equations of state at 460 bar, which is why the default is Dranchuk–Abou-Kassem. A source for the range in `pvt/gas.md`, or a rewording, is a follow-up.
8. **Hasan et al.'s gas–liquid ratio.** 40% of the cases lie above 5615 scf/STB and 1.6% below 104. The gas-rich wells flow mostly in annular flow, which the model treats without slip ($C_0 = 1$, $v_\infty = 0$), as Kabir and Hasan (2006) do for gas-condensate wells (`specs/features/019-five-flow-regimes.md`, Validity ranges).
9. **Horizontal cells.** No classifier has stratified flow. In a horizontal cell SLIP-10 sets the Taylor-bubble rise velocity to zero, and SLIP-12's bubbly–slug threshold $\alpha_t$ to zero. At the sampled horizontal cells the five regimes give annular flow at 54% of the points, slug flow at 20%, churn flow at 13% and dispersed bubbly flow at 13%. The model extrapolates; whether the redesign draws horizontal sections is decision 5, which is open.
10. **Vogel's inflow.** Vogel derived it by simulation for solution-gas drive reservoirs below the bubble point, with bubble points of 147 and 207 bar over 21 reservoir conditions. He gives an error of about 10% in most applications and at most about 20%, and warns that it does not hold where oil, gas and water all flow. Every sampled well uses it (SMP-8), wells with water and gas wells included. It is a well option, so the model extrapolates; which inflow a sampled fluid gets is the redesign's choice. The model has no inflow for gas wells.
11. **Reynolds number.** Below 4000, at 0.2% of the sampled points, FRIC-6 blends Chen's factor into the laminar $64/\text{Re}$. Its weight is 0.99 at Re = 4000 and 0.5 at Re = 3000. Neither is valid in transitional flow, and the blend is a smoothing choice.

## Where the sampler leaves the ranges

For the sampling redesign:

1. **Gas–oil ratios.** 69% of the cases have a gas–oil ratio above 3000 scf/STB (534 Sm³/Sm³), the lower end of gas condensates, and 5.0% above 150,000 scf/STB (26,700 Sm³/Sm³), their upper end. 76% lie above the largest solution gas–oil ratio among Valkó and McCain's 1745 oils, 2216 scf/STB (395 Sm³/Sm³). Their API gravity comes from the oil density (SMP-9), 21 to 40, where a gas condensate's is 40 to 60. The black oil's correlations stay in range ($R_{so}$ at most 232 Sm³/Sm³), because the oil is saturated wherever the gas–oil ratio exceeds $R_{so}(p, T)$. What the black oil misses for a gas condensate is the oil carried in the gas, which drops out as the pressure falls. SMP-43 accepted these wells until the redesign.
2. **Gas–liquid ratios** above Hasan et al.'s range in 40% of the cases (note 8).
3. **Horizontal sections** in 24% of the cases (note 9).
4. **Temperatures.** The surface tension's range is left in every case (note 4). 6.3% of the cases leave Beggs and Robinson's 70 to 295 °F: 2.1% have wellheads below 70 °F, and 4.3% have bottomholes above 295 °F, from reservoir temperatures up to 150 °C (SMP-14). 9.0% have points below Lee, Gonzalez and Eakin's 100 °F.
5. **Pressures.** 7.0% of the cases have points above Beggs and Robinson's 5250 psig (363 bar): the deepest wells, with $p_r$ up to 454 bar.
6. **Gas gravity** below Sutton's 0.571 in 9.1% of the cases (note 6).

For comparison, Valkó and McCain's oils (2003, Table 2.1) have reservoir temperatures from 289 to 445 K, API gravities from 6.0 to 63.7, separator-gas gravities from 0.555 to 1.685 and bubble points up to 462 bar. The sampler's reservoir temperatures, 333 to 423 K, lie inside.

One limit is the CasADi backend's rather than a correlation's. Without the Joule–Thomson term, the gravity term (THM-7) cools the flow by up to about 5 K per km of depth, so where the geothermal gradient is below that, a root can lie below the backend's lower temperature bound, and only the Rust core finds it (`specs/features/012-root-search.md`, A known limit). The sampler draws 32 to 37 K/km; the redesign should stay well above 5 K/km.

## Decisions

Signed off by Bjarne, 2026-10-29, but for decision 5.

1. **The rules.** Refuse, extrapolate and clamp, with the two constraints. No fourth rule, *flag*, which would extrapolate and record on the root which ranges its points leave, as `off_table` does for a table, in Python after the solve so that both backends get it. Every sampled case leaves the surface tension's temperature range (note 4), and 42% Hasan et al.'s gas–liquid ratios (note 8), so the flag would be on in most cases and say little; no user needs one yet. The evidence script measures the same for a dataset.
2. **API gravity**: refuse outside 16 to 58 instead of 10 to 40 (note 2).
3. **Relative roughness**: refuse above 0.05 with `RoughnessFriction`, checked in `WellProperties`, which holds both $\varepsilon$ and $D$ (note 3).
4. **The black oil's gas–oil ratio**: extrapolate, as before. The black oil takes any gas–oil ratio. As the ratio rises, the liquid's share of the flow, and with it the oil's effect on the pressure drop, falls, and the ratio takes no correlation out of its range (finding 1). A gas condensate whose dropout matters enters as a PVT table from a flash, which carries the oil in the gas as $R_{so} < 0$ (PVT-TAB-6). The alternative, to refuse above a bound such as 3000 scf/STB, would make every gas condensate and gas well need a table, and the sampler draw tables to have gas wells at all.
5. **Horizontal sections. Open.** Bjarne decides it with the sampling distribution, in the sampling redesign; until then the model extrapolates in them (note 9). The redesign either keeps drawing L-shaped wells with this known limit, or stops until a model with stratified flow exists, which would need its own feature spec.

Bjarne also signed off the implementation the same day:
- the checks' places: the API gravity in `BlackOil`, the PVT model a fluid takes, so that `BlackOilPVT` evaluates the correlations at any API gravity; the relative roughness in `WellProperties`; and both again in the Rust core's `WellSpec::check`;
- the develop vectors of PVT-OIL-6 and PVT-OIL-8, whose API 15 row moved to API 16, with a new row at API 58;
- the edits that follow from decisions 2 and 3 in the component files, `specs/sampling.md` (SMP-43), `specs/architecture.md` (design point 1) and `specs/calibration.md` (CAL-9, where a refused step gives rows without an operating point).

## Follow-ups

Done on 2026-10-29, with the sign-off:

- `pvt/oil.md`: PVT-OIL-10 gives Beggs and Robinson's range, 16 to 58 °API and 70 to 295 °F, where it said "the code's stated range is API 10 to 58 and 100 to 295 °F", and so does the code's docstring. The Safeguards section takes decision 2.
- `friction.md`: the Safeguards section takes decision 3.
- `pvt/gas.md` no longer states a range for Papay's z-factor, which it gave without a source (note 7).
- The refusals of decisions 2 and 3, in `BlackOil` and `WellProperties` and again in the Rust core, with tests. Neither refuses a well in the `v1.0.0` configuration, a verifier case or a sampled well: of the table's 500 wells at three operating points each, every case builds in the `v1.0.0` configuration and every case with oil in `develop`'s (1494; SMP-43 refuses the other 6), and both verifier candidates pass with a stable-root rate of 100%.
- The develop vectors of PVT-OIL-6 and PVT-OIL-8 pin the correlations at API 16 and 58, the ends of the range, in place of a row at API 15.

Open:

- Vazquez's thesis (1976), for the Vazquez–Beggs range (note 1).
- A source for the constants of the water viscosity, PVT-WAT-3 (note 5; `pvt/water.md`).
- A source for Papay's range, or for Papay (1968) itself (note 7).

## Sources

- Vazquez and Beggs (1980), "Correlations for fluid physical property prediction", *Journal of Petroleum Technology* 32, 968–970: pp. 969–970.
- Beggs and Robinson (1975), "Estimating the viscosity of crude oil systems", *Journal of Petroleum Technology* 27, 1140–1141: Table 1.
- Whitson and Brulé (2000), *Phase behavior*, SPE Monograph 20: §2, Table 2.1 and the classification of reservoir fluids.
- Valkó and McCain (2003), "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities", *Journal of Petroleum Science and Engineering* 37, 153–169: Table 2.1.
- Abdul-Majeed and Abu Al-Soof (2000), "Estimation of gas–oil surface tension", *Journal of Petroleum Science and Engineering* 27, 197–200: Tables 1 and 2, Fig. 2.
- Sutton (1985), "Compressibility factors for high-molecular-weight reservoir gases", SPE 14265: p. 3.
- Dranchuk and Abou-Kassem (1975), "Calculation of Z factors for natural gases using equations of state", *Journal of Canadian Petroleum Technology* 14(3), 34–36: the conclusions.
- Lee, Gonzalez and Eakin (1966), "The viscosity of natural gases", *Journal of Petroleum Technology* 18, 997–1000: the abstract and Table 1.
- Chen (1979), "An explicit equation for friction factor in pipe", *Industrial & Engineering Chemistry Fundamentals* 18, 296–297: Table I. Haaland (1983), "Simple and explicit formulas for the friction factor in turbulent pipe flow", *Journal of Fluids Engineering* 105, 89–90.
- Hasan, Kabir and Sayarpour (2010), "Simplified two-phase flow modeling in wellbores", *Journal of Petroleum Science and Engineering* 72, 42–49: Table 2.
- Vogel (1968), "Inflow performance relationships for solution-gas drive wells", *Journal of Petroleum Technology* 20, 83–92.
- Zhang, Wang, Sarica and Brill (2006), "Unified model of heat transfer in gas–liquid pipe flow", *SPE Production & Operations* 21, 114–122.
- The measurements: `plans/evidence/validity_ranges.py` (`plans/evidence/README.md`, Validity ranges), on `develop` at `8874b7f`.
