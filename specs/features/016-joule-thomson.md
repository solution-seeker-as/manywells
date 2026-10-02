# 016 · The Dranchuk–Abou-Kassem gas law and Joule–Thomson cooling

*Feature spec for Step 10 of `plans/manywells-v2-plan.md`, 2026-10-02. Status: approved by Bjarne on 2026-10-02 with the rulings below, implemented on the branch `step10-joule-thomson` and merged into `develop`; signed off by Bjarne on 2026-10-02 (Sign-off, at the end).*

## Rulings (Bjarne, 2026-10-02)

- **The factor.** The gas's Joule–Thomson factor is the Dranchuk–Abou-Kassem (DAK) equation of state's, in closed form at the state (PVT-GAS-10), for every real gas.
- **The gas law.** DAK is also the gas law (PVT-GAS-11), folded into this spec. Papay's z-factor stays as an option, and DAK is `develop`'s default.
- **The default.** The Joule–Thomson term is on in `develop`'s default and off in the `v1.0.0` configuration.
- **The tolerance** of the reference test of $J$ is $0.03 + 0.1\,|J_\text{ref}|$.
- **Two roots of an energy row** (found in the implementation): only the root where a cell's energy row rises in $T_i$ is physical. `solution.md` has a new SOL-9, and the CasADi search rejects a state with a falling energy row; the core takes the rising root by design (design choice 5).

## Motivation

`develop`'s gas law and energy balance do not hold over the range the v2 datasets will sample, and the energy balance treats the gas as ideal.

- **The energy balance.** `develop`'s gas is real, but `docs/thermal_energy_modeling.md` derived BAL-12 from the ideal gas's enthalpy $h_g = c_{pg}T$, which does not depend on pressure; THM-6 gives the gas no term because "an ideal gas's enthalpy does not change". A real gas's enthalpy does: $dh_g = c_{pg}\,dT - (R_s T^2/p)(\partial Z/\partial T)_p\,dp$, with $p$ in Pa. The missing term is the Joule–Thomson effect: expanding gas cools while $Z$ rises with $T$, below the inversion pressure.
  - Hasan and Kabir (2012), Eq. (7), the energy balance that `docs/thermal_energy_modeling.md` cites, has the term as $C_J\,dp/dz$.
  - Hasan and Kabir (2018), §6.4.2, give $c_p C_J = -[V - T(\partial V/\partial T)_p]$, which for a real gas, $V = ZRT/p$, is $(RT^2/p)(\partial Z/\partial T)_p$; in two-phase flow they weight the phases by mass. They do not recommend leaving it out, "because gas in most wells is rarely ideal" (§6.4.1).
- **The gas law.** Papay's z-factor (PVT-GAS-4) is valid for $p_{pr} < 6$. `develop`'s sampler draws bottomhole pressures up to 457 bar ($p_{pr}$ up to 10.4), temperatures from 277 to 423 K ($T_{pr}$ 1.18 to 2.24) and gas gravities from 0.55 to 0.90 (SMP-1, SMP-11, SMP-13 to SMP-15, SMP-21), and 34.5% of the sampled cases below have a bottomhole $p_{pr}$ above 6. At 460 bar and 360 K, Papay's $Z$ is 11% above the reference equation of state for methane, 20% for a natural gas of gravity 0.67 and 29% for gravity 0.80; DAK's is within 2%. Papay's derivative $(\partial Z/\partial T)_p$ has the wrong sign above about 300 bar.
- **Range and simplicity.** The v2 datasets will simulate operating points of a wide range of sampled wells, so the model must hold over the whole sampled range; and the implementation should be simple (Bjarne, 2026-10-02). The design choices below trade the two, each measured.

## Delta

### `specs/model/pvt/gas.md`

- **PVT-GAS-9 · Dranchuk–Abou-Kassem equation of state.** Dranchuk and Abou-Kassem (1975), Eq. (2): $Z(\rho_r, t) = 1 + c_1\rho_r + c_2\rho_r^2 - c_3\rho_r^5 + c_4\rho_r^2(1 + A_{11}\rho_r^2)e^{-A_{11}\rho_r^2}$, with eleven constants, $t = T_{pr}$ and the reduced density $\rho_r = Z_c\,p_{pr}/(Zt) = Z_c\,\rho_g R_s T_{pc}/p_{pc}$, $Z_c = 0.270$ (their Eq. 3), from Sutton's pseudo-critical properties (PVT-GAS-5). Recommended for $0.2 \le p_{pr} < 30$ with $1.0 < T_{pr} \le 3.0$, not at $T_{pr} = 1.0$ with $p_{pr} \ge 1.0$.
- **PVT-GAS-10 · Joule–Thomson factor.** $J = T(\partial \ln Z/\partial T)_p = (tZ_t - \rho_r Z_\rho)/(Z + \rho_r Z_\rho)$, DAK's in closed form at the state's $\rho_g$ and $T$; 0 for an ideal gas. $\mu_{JT} = J/(\rho_g c_{pg})$. The denominator is proportional to $(\partial p/\partial\rho_g)_T$: positive for $T_{pr} \ge 1.05$ and $p_{pr} \le 15$ (minimum 0.078, at $T_{pr} = 1.05$), negative near the critical point at $T_{pr} = 1.0$ (minimum −0.072), where $J$ has a pole.
- **PVT-GAS-11 · Real gas law with DAK.** $c_\text{bar}\,p = Z(\rho_r, T_{pr})\,\rho_g R_s T$. Its row, $p - \rho_g Z R_s T/c_\text{bar}$, is explicit in the state. The density at $(p, T)$ is a Newton solve (Design choices, item 3).
- **Options.** `FluidModel.z_factor_model`: `'dak'` (default) or `'papay'` (PVT-GAS-3 with PVT-GAS-4). `FluidModel.jt_factor(T, rho_g)` gives $J$, DAK's for both.

### `specs/model/thermal.md`

**THM-8 · Joule–Thomson term.**

$$\Phi_{JT} = \frac{\alpha\, v_g\, J\, (F + \rho_m g\cos\theta)}{C} \quad [\text{K/m}],$$

with $J$ of PVT-GAS-10 at the point, $F$ the viscous pressure gradient (FRIC-1), $\theta$ the cell's inclination and $C$ the heat-capacity flux of THM-1.

- **Derivation.** The gas's extra enthalpy flux is $(w_g/A)\,c_{pg}\mu_{JT}\,dp/dz = \alpha v_g J\, dp/dz$, using $\rho_g = p/(Z R_s T)$, and $dp/dz = -(F + \rho_m g\cos\theta)$ neglects acceleration as THM-6 and THM-7 do. It is the gas counterpart of THM-6.
- **Sign.** $F + \rho_m g\cos\theta > 0$ at every admissible state, so $\Phi_{JT}$ has the sign of $J$: it cools where $Z$ rises with $T$, everywhere in the sampled range below about 300 bar, and heats above the inversion pressure.
- **Limits.** Zero for an ideal gas and for pure liquid. For pure gas, $\Phi_{JT} = \mu_{JT}(F + \rho_g g\cos\theta)$.
- **Switch.** `ThermalModel.joule_thomson`, on by default.

### Other files

- `specs/model/balances.md`: **BAL-13**, $dT/dz = -H + \Phi_f - \Phi_g - \Phi_{JT}$; with $\Phi_{JT} = 0$ it is BAL-12.
- `specs/model/solution.md`: **SOL-9**, rising energy rows: a state with a cell whose energy row falls in $T_i$, at fixed $p_i$ along the point's mass rows and closures, is not a root; SOL-2's root set holds only states that satisfy it.
- `specs/model/discretization.md`: DISC-10's row is unchanged, and its text adds $-\Phi_{JT,i}$; DISC-11's gas-law closure is PVT-GAS-1, PVT-GAS-3 or PVT-GAS-11.
- `specs/model/README.md`: `develop`'s default gas is PVT-GAS-9 and PVT-GAS-11, its energy balance BAL-13 with THM-8.
- `specs/model/nomenclature.md`: $\Phi_{JT}$, $J$ (`jt_factor`), $\rho_r$ (`reduced_density`), `z_factor_model`.
- `specs/architecture.md`: the interface table (`gas_law_row`, `jt_factor`), the note on `dp_dmd` and the extension-point table, which said Joule–Thomson cooling would come through `dp_dmd`, and the Rust core's design point 4 (the temperature bracket).
- `docs/thermal_energy_modeling.md`: §4, the Joule–Thomson term, and the real gas in the derivation.

### Code

| Where | Python | Rust |
|---|---|---|
| gas | `pvt/gas.py`: `dak_z_factor`, `dak_jt_factor`, `dak_reduced_density` | `pvt/gas.rs`: the same |
| fluid | `FluidModel.z_factor_model`; `z_factor`, `gas_density`, `gas_law_row` with DAK; `reduced_density`, `jt_factor` | `ZFactorModel`, `GasLaw::Dak`; `Fluid::jt_factor` |
| thermal | `ThermalModel.joule_thomson`, THM-8 | `Thermal::joule_thomson`, THM-8; `is_linear` is false with it and a real gas |
| method | none | the temperature bracket's lower end steps out (item 4); the cell solve descends where the state at $p_s$ cannot be computed, the scan refines the edges of the finite region, and the temperature solve takes the upper root of a U-shaped row (item 5) |
| harness | `configurations.py` (off in `v1_well`, `z_factor_model` checked for `develop`), `discretization.row_ids`, the solver's lower temperature bound (`System.bounds`, item 6), SOL-9 in the CasADi search (`System.energy_slope`, `energy_rows_rise`), `solvers/rust.py` | `lib.rs`: `Well::new`'s arguments, `dak_z_factor`, `dak_jt_factor`, `jt_factor` components, the `lower_step_outs`, `temperature_minima` and `edge_refinements` counts; test wells `w1_joule_thomson`, `w2_joule_thomson` |

No interface changes: `temperature_gradient` already receives the state, the fluid, $F$ and $\cos\theta$ in both backends, and `dp_dmd` stays unused.

## Design choices

1. **The pressure gradient: $F + \rho_m g\cos\theta$, not the cell's $c_\text{bar}(p_i - p_{i-1})/\Delta\text{MD}_i$.** It is the substitution THM-6 and THM-7 make, so the energy balance has one approximation, the term needs no interface change, and the energy row stays a function of point $i$'s state and $T_{i-1}$. Against the cell's gradient it changes TWH by at most 0.38 K (median 0.01 K) and PWH by at most 0.02 bar (prototype).
2. **The factor: DAK's closed form at the state.** Papay's own derivative is simpler, 4 lines against DAK's 11 constants and about 15 lines in each backend, but it has the wrong sign above about 300 bar (methane at 460 bar and 360 K: −0.48 against +0.09 from the reference). DAK's follows the reference to 460 bar (Measurements), and its closed form needs no inner solve because the state carries $\rho_g$. In the prototype, with Papay's gas law, the two factors' TWH differ by more than 1 K in 33% of cases and by up to 8.4 K, and Papay's factor lost 5 of 254 operating points where DAK's lost none.
3. **The gas law: DAK in its pressure-explicit form.** $Z$ is explicit in $(\rho_g, T)$, so the CasADi rows need no inner solve. The density at $(p, T)$ is needed only where the state is built from $p$ and $T$: the CasADi backend's starting guess at point 0, and the Rust core's state at each trial $(p_i, T_i)$. Both take Newton's method on $\rho_r t Z(\rho_r, t) = Z_c\,p_{pr}$ from the ideal-gas density $Z_c\,p_{pr}/t$, which converges to $10^{-12}$ in at most 17 steps for $1.05 \le T_{pr} \le 3$ and $p_{pr} \le 30$ (11 for $T_{pr} \ge 1.4$). The CasADi backend unrolls 20 steps, so that the density accepts symbols; the Rust core steps until the step is at most $10^{-13}$ of $\rho_r$, at most 50 steps, and returns NaN otherwise, which fails the state.
4. **The Rust core's temperature bracket.** 015's lower end, $\min(T_{i-1}, T_a) - \Delta\text{MD}\,g\cos\theta/\min(c_p)$, is proven only for the existing terms, and $\Phi_{JT} \ge 0$ has no simple bound over all states. Where the row is still positive at the lower end, the lower end steps out by doubling steps, at most 30, as the upper end does. Heating by $\Phi_{JT} < 0$ is handled by the existing upper step-out. On the 263 cases of the measurements, it stepped out 725,631 times with the term (`new`), and never without it.
5. **The Rust core where the term makes the cell's rows harder.** In a gas well the term breaks three assumptions of the core's method, all at trial states far from the root or near a choked wellhead. Each was found by a case of the measurements that the CasADi backend solves and the core did not:
   - **States that cannot be computed.** At trial pressures far below the cell's, the expanding gas's cooling grows faster than its temperature falls, and the energy row has no root (at well 44, the row at $p_s$ rose from +6.6 to +12,000 as the bracket stepped down, until the state failed). The cell solve gave up where the state at $p_s$ could not be computed (`CellStep::Unsolved(p_prev)`), which left the cell's pressure at $p_{i-1}$; the pressures with a state need not form one interval, so a search over $[p_s, p_{i-1}]$ that counts the others as $+\infty$ can head the wrong way. Where the state at $p_s$ cannot be computed, the cell solve now descends from $p_{i-1}$: steps that double from $(p_{i-1} - p_s)/1024$ to the first pressure where the row is negative, then Brent; where a step reaches a pressure without a state first, the edge of the stretch with states above it is bisected to $10^{-9}\,p_{i-1}$, and the row's minimum on that stretch decides, as in the cell solve.
   - **A residual that is not finite next to a root.** The scan of $R(p_0)$ left out samples whose march fails, and a root next to such a region lost its bracket: at well 44, $R$ is finite only above 224.9 bar, and the root is at 225.68, between two samples 2.2 bar apart. Where a finite sample neighbours one that is not, the edge of the finite region is now bisected, to the width of the scan's existing refinement ($10^{-6}(p_r - p_s)$), and sampled.
   - **An energy row with two roots in $T$.** Near a choked wellhead the cooling can make the energy row U-shaped in $T$ (at well 22, roots at 279.3 and 293.4 K between the bracket's ends, both of which it is positive at). The root on the rising side of the minimum continues the root without the term, and the CasADi backend's root has it (293.35 K); the lower comes from $J$'s growth towards the critical point. Where the row is positive at both ends of the bracket, one sample 0.01 K below the lower end decides: where the row rises there, the end is on the U's falling side, and the row's minimum between the ends, narrowed to $10^{-3}$ K, brackets the upper root; otherwise the root lies below, and the lower end steps out (item 4). Step 9's assumption that the energy row has one root in its bracket does not hold with the term: its check passes on the test wells, which carry too little gas.

   All three are solver machinery (principle 7). On the 263 cases (Measurements), before them the core found an operating point in 243 cases with the term (242 with Papay's gas law and the term), against 259 without it, and the CasADi prototype had one in each case the core missed; with them it finds one in 259 in every variant, and no operating point of any variant moved by more than $10^{-6}$ bar. They cost the default before 016 (`old`), where they find nothing new, 2.9% at the median and 17.7% in total, on one process.
6. **The CasADi backend's lower temperature bound.** It was $\min(T_s, T_{lg})$, and with the term the fluid can be colder than its surroundings: in the backend comparison the core found two unstable roots whose coldest point is 0.005 and 0.1 K below $T_s$, which the CasADi backend could not reach. With the term on a real gas the bound is now $\min(T_s, T_{lg}, 1.05\,T_{pc})$, which is $1.05\,T_{pc}$ for every gas the sampler draws (198 to 247 K): the lower end of DAK's range, below which $J$ has its pole, so that the bound still keeps Ipopt away from the pole. Without the term it is unchanged. In the 263 cases of the measurements, the coldest point of any operating point is 3.4 K above $T_s$ with the term and 4.6 K above it without.

## Off in the `v1.0.0` configuration

- `joule_thomson=False` in `v1_well`; `configurations.check` flags every boolean switch of `ThermalModel` that is on. With `v1.0.0`'s ideal gas $J = 0$, so the term is zero even when on, and the gas law is PVT-GAS-1 whatever `z_factor_model` says.
- The row vectors and both verifier reports are unchanged: `develop`'s and the Rust core's candidates report PASS, a stable-root rate of 100% and no expected failures.

## Measurements

The scripts are in `plans/evidence/` (`README.md` there).

**The factor $J$** (`jt_factor.py`; reference / Papay / DAK). The references are CoolProp 8.0.0's equations of state: Setzmann and Wagner (1991) for methane, and CoolProp's mixture model for the natural gases.

| Gas | $p$ (bar) | 300 K | 360 K | 425 K |
|---|---|---|---|---|
| methane | 100 | 0.736 / 0.648 / 0.696 | 0.356 / 0.333 / 0.334 | 0.196 / 0.164 / 0.184 |
| methane | 270 | 0.574 / 0.714 / 0.586 | 0.429 / 0.279 / 0.414 | 0.265 / 0.079 / 0.248 |
| methane | 460 | −0.001 / −0.392 / 0.015 | 0.089 / −0.484 / 0.109 | 0.085 / −0.454 / 0.093 |
| gravity 0.67 | 100 | 1.336 / 0.870 / 1.047 | 0.569 / 0.460 / 0.465 | 0.302 / 0.239 / 0.250 |
| gravity 0.67 | 460 | −0.089 / −0.376 / −0.087 | 0.066 / −0.515 / 0.067 | 0.107 / −0.524 / 0.097 |
| gravity 0.80 | 100 | 2.688 / 1.173 / 1.686 | 0.958 / 0.630 / 0.669 | 0.471 / 0.340 / 0.345 |
| gravity 0.80 | 460 | −0.203 / −0.366 / −0.198 | −0.003 / −0.543 / −0.013 | 0.098 / −0.594 / 0.075 |

- DAK is within 0.07 of methane's reference everywhere on the script's grid (5 to 460 bar, 280 to 425 K). Both correlations underestimate the richer gases near 280 to 300 K at 50 to 100 bar, close to their critical region: for gravity 0.80 at 280 K and 100 bar the reference is 3.67, DAK 2.59 and Papay 1.49.
- DAK's $Z$ is within 1.3% of methane's reference at the eleven points of `tests/test_pvt.py`, where Papay's is up to 21.9% off (460 bar, 300 K).

**The prototype** (`jt_wells.py`): the term added by a subclass of `ThermalModel` on the CasADi backend, with Papay's gas law, on 88 wells of `develop`'s sampler (seed 2026), each at its nominal operating point and two sampled ones: 263 cases, 254 with an operating point. It chose items 1 and 2.

**The implemented feature** (`dak_jt_default.py`): the same 263 cases on the Rust core, in four variants of `develop`'s default: `old` (Papay, no term: the default before 016), `dak` (DAK, no term), `papay_jt` (Papay with the term) and `new` (DAK with the term: the default after 016). Every variant finds an operating point in 259 of the 263 cases.

| Change against `old` | Median | 10th percentile | Largest |
|---|---|---|---|
| TWH, `new` | −7.5 K | −20.1 K | −45.9 K |
| PWH, `new` | −0.97 bar | −3.0 bar | −6.6 bar |
| PBH, `new` | −0.04 bar | −0.81 bar | −2.4 / +3.4 bar |
| TWH, `dak` (the gas law alone) | 0.00 K | −0.07 K | −1.0 / +0.3 K |
| PWH, `dak` | −0.06 bar | −0.74 bar | −6.2 / +0.7 bar |
| PBH, `dak` | +0.02 bar | −0.07 bar | −0.24 / +0.95 bar |

- **The term dominates TWH**, and the gas law matters most at the wellhead pressure of deep gas wells. With the term on, the two gas laws' TWH differ by −0.03 to +0.23 K between the 10th and 90th percentiles, and by up to 7.6 K.
- **Colder than the surroundings.** With the term, the fluid is colder than the ambient profile somewhere above the bottomhole in 31 of the 259 cases, by up to 4.6 K; without it, never.
- **The backends.** In the 254 cases both solve, the core's `papay_jt` variant agrees with the CasADi prototype, the same model, to $2\cdot10^{-8}$ bar in PBH and $4\cdot10^{-8}$ K in TWH; the core solves 5 cases the prototype does not.
- **Work, on one process.** Without the term (`old`) the core takes 410 ms per case at the median and 128.7 s in total; with it (`new`), 267 ms and 99.9 s, because in gas wells more marches stop where a state cannot be computed (7,558 scan samples without a finite $R$, against 1,925).

**The backends** (`scripts/verification/compare_backends.py`). 20 sampled wells for each group 016 touches (`v1.0.0` with `real gas`, which is DAK now, `papay` and `joule-thomson`; `develop`, whose default has both, `develop+papay` and `develop+no joule-thomson`): all 120 cases pass Step 9's rule. No label differs; the only roots the core misses are in cases where the slip law has several void fractions or a root lies on another branch of a point's rows, which the rule allows; the one root only the core finds (`develop#15`, and the same well with `papay`) zeroes the CasADi rows, a root the CasADi search misses (`plans/improvements.md` §2.9). The core is 28× faster than the CasADi backend at the median and 39× in total, on 24 processes. Before item 6, the core found two roots in `v1.0.0+joule-thomson` that the CasADi backend's temperature bound cut off.

## Acceptance

Done on the branch, 2026-10-02:

1. **The `v1.0.0` configuration** is unchanged: the row vectors, and both verifier reports (above).
2. **Component vectors.** Develop vectors for PVT-GAS-9, PVT-GAS-10, PVT-GAS-11 and THM-8, from `specs/tools/make_develop_vectors.py`, including states at $p_{pr}$ up to 10, checked in both backends by `tests/test_spec_vectors.py`. The tables of Papay's PVT-GAS-3 to PVT-GAS-5 now pin `z_factor_model='papay'`; their values did not change.
3. **`tests/test_pvt.py`.** The density solves its gas law to $10^{-12}$ for $1.05 \le T_{pr} \le 3$, $p_{pr} \le 30$; PVT-GAS-10 equals a central difference of $\ln Z$ along an isobar to $10^{-6}$; $J = 0$ for an ideal gas and is the same for both z-factor models; against methane's reference at eleven points, $J$ within $0.03 + 0.1\,|J_\text{ref}|$ and $Z$ within 2%; $Z + \rho_r Z_\rho > 0$ for $1.05 \le T_{pr} \le 3$, $p_{pr} \le 15$. The core's `pvt/gas.rs` tests the density and the factor in the same way.
4. **`tests/test_thermal.py`** and the core's `thermal.rs`: $\Phi_{JT} = 0$ for an ideal gas and for $\alpha = 0$; for pure gas, $\Phi_{JT} = \mu_{JT}(F + \rho_g g\cos\theta)$; its sign is $J$'s, with both signs at 300 K.
5. **`tests/test_model_properties.py`**, whose wells now have DAK and the term: every check passes on both backends, including `test_heat_flows_from_the_fluid_to_the_surroundings`, as these wells carry too little gas for the term to cool them below their surroundings. A new gas-rich well (gas mass fraction 0.4) is 8 K colder at the wellhead with the term than without, on both backends.
6. **Backend comparison** (`tests/backend_cases.py`): feature `016`; overlays `papay`, `joule-thomson` (with a real gas) and `no joule-thomson`; the rows agree at the same states in every configuration of the matrix, including `W1+papay+joule-thomson` and `W2+energy terms+joule-thomson`; the comparison-set groups as above.
7. **The Rust core's method.** Step 9's assumption checks (`march.rs`, `the_cell_rows_have_the_shapes_the_cell_solve_assumes`) run on the two new test wells too. They found one gap: the energy row's crossings were counted over samples where the row is NaN, below 0 °F, where the dead-oil viscosity correlation has none; the check now counts finite samples only. `tests/test_rust_backend.py` checks item 5 on the three sampled gas wells that showed it (wells 44 and 22): the core finds the CasADi backend's operating point at each, and takes the new paths.
8. **Temperature bounds.** In the backend comparison, every root of either backend is found by both, but for the known exceptions above; the two roots that the old bound cut off are found by both since item 6.
9. **SOL-9.** `tests/test_roots.py` checks the slope with heat loss alone against $1 + \Delta\text{MD}\,4h/(D\,C)$; `tests/test_model_properties.py` checks SOL-8 and SOL-9 at every root of both backends; `tests/test_backend_comparison.py` checks that the core's marches satisfy it; and `tests/test_rust_backend.py` that the core's operating point at well 22, where the row is U-shaped, does. After the ruling, `develop`'s verifier report and the backend comparison above are unchanged: the CasADi search rejected no root.
10. **The suites.** `cargo test --no-default-features` passes (40 tests), and the full test suite passes: 692 passed and 1 skipped, before the gas-well test of item 7 was added, which passes on its own. Its one warning, a root only the core finds in `v1.0.0+L-shaped#0` at 433.06 bar, an unstable root 0.035 K above $T_s$ without the term, comes from none of the new paths (`plans/improvements.md` §2.9).

## Gaps found (Step 10)

Step 10 records every step that needed more than `AGENTS.md` and the specs described. `AGENTS.md` now has a section, "Adding an equation", with the steps this feature took.

- **Evidence.** Nothing said where a feature spec's measurements live. This spec put its scripts in `plans/evidence/`.
- **Validity ranges.** The gas law's range ($p_{pr} < 6$) had not been checked against the sampler's ($p_{pr}$ up to 10.4). Nothing required an option's correlations to cover the sampled wells. The new section asks for that check in the feature spec.
- **A changed default.** Making DAK and the term the default changed what earlier tests and tables meant: the develop vector tables of Papay, the `real gas` overlay of the backend comparison (now DAK), the single-term thermal tests, and the default row IDs. Each now pins the option it tests.
- **The Rust core's temperature bracket.** Its lower end was proven from the sign and a bound of each energy term; a new cooling term broke the proof (item 4). `specs/architecture.md` now says that a new energy term must keep the bracket valid.
- **The core's assumptions in gas wells.** The cell solve and the scan assumed that every trial state can be computed, and the temperature solve that the energy row has one root in its bracket. The term broke all three in gas wells, and the Rust core lost 16 of 259 operating points before item 5. Step 9's assumption checks did not catch it: their test wells carry too little gas. The measurement on sampled wells did, by comparing the variants' counts of operating points, and the CasADi backend's roots showed what the core missed; `AGENTS.md`'s section asks for that measurement.
- **Checks that assume heat flows outwards.** The plan's checks for a new option (New model versions) list "temperature not below the ambient profile", and `test_heat_flows_from_the_fluid_to_the_surroundings` asserts it. With the term the fluid can be colder than its surroundings in gas-rich wells, so the check holds only for wells with little gas.
- **The CasADi backend's temperature bounds** are solver bounds that a new cooling term can reach, and the term reached them (item 6). They are documented in feature spec 010 and in `System.bounds`, not in `specs/model/`. Only the backend comparison showed it, as roots only the core found.
- **Worktrees.** Other sessions share the `develop` checkout; the work went into a worktree, which the new section describes.

## Out of scope

- **Retiring Papay**: it stays as an option (ruling).
- **Heat capacities** that vary with $(p, T)$: $c_{pg}$ stays constant, although a real gas's rises with pressure.
- **Other energy terms:** the liquid's thermal expansion (its Joule–Thomson term with $\beta_l \ne 0$), the heat of solution of dissolved gas, kinetic energy in the energy balance, and Joule–Thomson cooling across the choke, which is downstream of the model.
- **The range below $T_{pr} = 1.05$**, near the critical point, where DAK is not recommended: nothing checks it (`specs/model/pvt/gas.md`, Safeguards).

## Sign-off (Bjarne, 2026-10-02)

- **The Rust core's four method changes** (principle 7), with their measured cost and gain and their constants: the lower step-out of the temperature bracket (item 4); the cell solve's descent where the state at $p_s$ cannot be computed, the scan's refinement of the edges of the finite region, and the temperature solve's choice of the rising root of a U-shaped row (item 5). Signed off.
- **The density solve's constants:** 20 unrolled Newton steps in the CasADi backend; a relative step of $10^{-13}$ and at most 50 steps in the Rust core (item 3). Signed off.
- **The CasADi backend's lower temperature bound** with the term, $1.05\,T_{pc}$ (item 6), and **the tolerance of the $Z$ test** against methane's reference, 2%. Signed off.
- **Two roots of an energy row:** ruled; SOL-9 (Rulings).
- **The edits to `specs/model/`** (PVT-GAS-9 to PVT-GAS-11, THM-8, BAL-13, SOL-9 and the text of DISC-10, DISC-11 and the configuration table) and to `specs/architecture.md`; **the changed default of `develop`**, which changes its results (Measurements); and **the `AGENTS.md` section**. Signed off.

## Sources

- Hasan and Kabir (2012), "Wellbore heat-transfer modeling and applications", *Journal of Petroleum Science and Engineering* 86–87, 127–136, Eq. (7).
- Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., SPE, §6.4.1 and §6.4.2.
- Dranchuk and Abou-Kassem (1975), "Calculation of Z factors for natural gases using equations of state", *Journal of Canadian Petroleum Technology* 14(3), 34–36, doi:10.2118/75-03-03: Eqs. (2) and (3), the coefficients, Table 1 and the recommended range. Ahmed (2006), *Reservoir engineering handbook*, 3rd ed., Eqs. 2-40 and 2-41, pp. 56–58, gives the same equation and coefficients.
- Setzmann and Wagner (1991), "A new equation of state and tables of thermodynamic properties for methane covering the range from the melting line to 625 K at pressures up to 1000 MPa", *Journal of Physical and Chemical Reference Data* 20, 1061–1155, through CoolProp 8.0.0, for the reference values.
- Alves, Alhanati and Shoham (1992), "A unified model for predicting flowing temperature distribution in wellbores and pipelines", *SPE Production Engineering*: the energy balance with the Joule–Thomson term at all inclinations. Not consulted.
