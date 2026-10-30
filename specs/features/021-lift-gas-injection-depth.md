# 021 · Lift-gas injection depth

*Feature spec, 2026-10-29. Status: approved by Bjarne on 2026-10-29, with each design choice as proposed (Approval, at the end); implemented on the branch `lift-gas-injection-depth`. Bjarne signed off the implemented feature on 2026-10-29 (Sign-off, at the end). Step L1 of `plans/model-lock-plan.md`: Bjarne put the injection depth in v2.0's model on 2026-10-29 (`specs/goals.md`).*

## Motivation

The model injects the lift gas at the bottomhole: INF-5 adds $w_{lg}$ to the gas rate at point 0, and THM-5 mixes its temperature in there. A gas-lifted well injects through a valve in the tubing above the packer, often hundreds of metres above the perforations. Below the valve the tubing carries the reservoir fluid alone, so the column there is heavier than the model makes it. Bottomhole injection therefore overstates what a given lift-gas rate does: it lightens the whole column instead of the part above the valve.

The sampler gives gas lift to half the wells whose gas mass fraction is at most 0.2, at 0 to 5 kg/s (SMP-17, SMP-19), and the lift-gas rate is a control that the datasets' users vary. For those wells to be realistic, the lift gas has to enter where a valve would be.

## Design choices (decided by Bjarne, 2026-10-29: each as proposed)

1. **Where the depth lives.**
   - *Proposed:* on `WellProperties`, as `injection_md`, the measured depth of the injection point (m); `None` means the bottomhole. The depth is a property of the completion, like the trajectory. A well's system is built once (`discretization.build_system`), and which point the lift gas enters at is part of its structure.
   - *Alternative:* on `BoundaryConditions`, with the operating point. Each sample could then move the valve, but the injection point would have to become a parameter of the system. The lift gas would then enter through a smoothed step in the point's depth, a new smoothing in every point's mass rows.
2. **The injection point on the grid.**
   - *Proposed:* the grid point nearest $z_\text{inj}$ in measured depth, the deeper of two equally near. The error is at most half a cell, 8 to 32 m on the sampler's grids of 100 cells over 1513 to 6490 m of measured depth; the discretization's error is of first order anyway. A sampler can draw the depth on grid points.
   - *Alternative:* a grid point at $z_\text{inj}$ exactly, added as a station of the survey (GEO-3 allows a non-uniform grid). Exact, but a well's grid would then depend on its valve, and `WellGeometry.from_survey`'s uniform grid would gain a point.
3. **The lift gas's temperature when it is not given.**
   - *Proposed:* the ambient temperature at the injection point, $T_a$ of THM-4 there: lift gas that has come down the annulus has had time to approach the formation's temperature. At the bottomhole that is $T_r$, so `T_lg=None` keeps its meaning there and nothing changes for today's wells.
   - *Alternative:* $T_r$ at any depth, as today. Simpler, but a gas at the reservoir's temperature higher up the well is not physical.
4. **The mixing.** The lift gas mixes with the flow at the injection point by heat-capacity rates, as THM-5 mixes it at the bottomhole. Above the bottomhole, the injection cell's energy row takes the mixing (THM-9, DISC-12), so the state keeps one temperature per point. The flow's heat-capacity rate is the one arriving from below, without the lift gas.

## Delta

### `specs/model/inflow.md`

- **INF-9 · Lift-gas injection point.**
  - The lift gas enters at point $k$, the grid point nearest $z_\text{inj}$, the injection point's measured depth (design choice 2).
  - It enters the phase rates (INF-5, PVT-OIL-13) at points $i \ge k$ only: below point $k$ the tubing carries the reservoir fluid alone.
  - Without an injection depth, $k = 0$, and this is INF-5.

  The lift gas still does not dissolve into the oil (PVT-OIL-13).
- **INF-5:** its sentence "The lift gas is the produced gas, injected at the bottomhole" says "at the injection point of INF-9, the bottomhole by default".

### `specs/model/thermal.md`

- **THM-9 · Lift-gas mixing at the injection point.** At the injection point $k > 0$, the flow arriving from below, at $\tilde T_k$, mixes with the lift gas at $T_{lg}$:

  $$T_k = \tilde T_k + \frac{H_{lg}\thinspace(T_{lg} - \tilde T_k)}{H_k + H_{lg}}, \qquad H_k = c_{pg}\big(w_g(p_k, T_k) - w_{lg}\big) + c_{pl}\thinspace w_l(p_k, T_k), \qquad H_{lg} = c_{pg} w_{lg},$$

  with the phase rates of PVT-OIL-13 (INF-5 with dead oil) at point $k$, which include the lift gas (INF-9).
  - When $T_{lg}$ is not given, it is the ambient temperature at point $k$ (THM-4; design choice 3), which at $k = 0$ is $T_r$, as THM-5 has it.
  - At $k = 0$ the mixing is THM-5's, unchanged.
  - The thermal model's switch `lift_gas_mixing` turns both off: then $T_k = \tilde T_k$.

### `specs/model/discretization.md`

- **DISC-12 · Energy row at the injection point.** For $k > 0$, cell $k$'s energy row is DISC-10's step followed by THM-9's mixing:

  $$r = T_k - \tilde T_k - \frac{H_{lg}\thinspace(T_{lg} - \tilde T_k)}{H_k + H_{lg}}, \qquad \tilde T_k = T_{k-1} + \Delta\text{MD}_k\left(\frac{dT}{d\text{MD}}\right)_k \quad \text{[K]}$$

  with $(dT/d\text{MD})_ k$ as in DISC-10, at point $k$. Without lift gas ($w_{lg} = 0$) it is DISC-10.
- **DISC-11:** at point $k > 0$ the energy row is DISC-12 instead of DISC-10. `discretization.row_ids` gives it. The mass rows need no change: over cell $k$ the rate difference in DISC-7 holds the step $w_{lg}/A$, the lift gas's source, and the momentum row DISC-9 accelerates the injected gas, which enters with no axial momentum.

### `specs/model/balances.md`

BAL-10: with an injection point above the bottomhole, $w_g(z)$ steps by $w_{lg}$ at $z_\text{inj}$, a point source of gas.

### Other files

- `nomenclature.md`: $z_\text{inj}$ (`injection_md`, m), the injection point $k$, and $T_{lg}$'s default ("the ambient temperature at the injection point").
- `README.md`'s configuration table: the lift-gas injection point is a well option (INF-5 at the bottomhole, or INF-9); the `v1.0.0` configuration injects at the bottomhole.
- `specs/architecture.md`: `WellProperties.injection_md` (design choice 1). The Rust core takes it as the injection point's index.
- `specs/model/validity.md`: no correlation is added; the mixing is an energy balance.
- `docs/simulate.md` and an example in `scripts/sim_examples/` (`gl_temp.py` or a new one): a gas-lifted well with the valve above the perforations.

### Code

| Part | Python | Rust |
|---|---|---|
| input | `WellProperties.injection_md`, validated in `__post_init__` (finite, $0 \le z_\text{inj} \le$ the well's measured depth); the injection point from the geometry | `Well::new`'s `injection_point`, passed by `solvers/rust.py`, validated |
| phase rates | `discretization.py`: each point's rows take $w_{lg}$ where $i \ge k$, 0 below | `march.rs`, `discretization.rs`: the same per point |
| energy | `thermal.ThermalModel`: THM-9's mixing; `discretization.cell_rows`: DISC-12 at point $k$ | the temperature solve at cell $k$ with the mixing |
| default $T_{lg}$ | `System.params`: $T_a$ at point $k$ when `T_lg` is `None` | `solvers/rust.py` |
| harness | `configurations.py`: `v1_well` injects at the bottomhole, and `check` requires it for `v1.0.0`; `System.bounds`'s temperatures take the default $T_{lg}$ | test wells with the valve above the bottomhole in `input.rs`'s `all()` |

## Off in the `v1.0.0` configuration

`injection_md=None`: the lift gas enters at the bottomhole, and the rows are today's, exactly. `configurations.check` flags any other value in the `v1.0.0` configuration. The row vectors and both verifier reports do not change.

## Validity ranges

The option adds no correlation, so it has no published range to check against the sampler's (`AGENTS.md`, Adding an equation, step 1). The ported sampler does not draw an injection depth; the redesign will. Its realistic range, the operating valve's depth as a share of the well's depth, is a question for the redesign's sources.

## Implementation (2026-10-29)

On the branch `lift-gas-injection-depth`, as the Delta's code table, with these details and differences:

- **Point 0's temperature row above the bottomhole.** With $k > 0$ no lift gas enters at point 0, so its inflow temperature row is THM-3, $T_0 - T_r$, with that ID, also where the thermal model mixes. THM-5 without lift gas has the same value, except at $p_0 = p_r$, where its denominator is zero (thermal.md, Safeguards). INF-9 and DISC-11 say so. A difference from the Delta, which did not name point 0's row.
- **Python.**
  - `WellProperties.injection_md` is validated in `__post_init__` (a real number from 0 to the well's measured depth; booleans, strings, NaN and infinities are refused) and stored as a float. `WellProperties.injection_point` is $k$, from `WellGeometry.nearest_point`.
  - `discretization.lifts(wp)` says where the lift gas flows, `lift_gas_rate` puts it into a point's phase rates, `inflow_temperature` gives point 0's temperature, and `lift_gas_temperature(wp, bc)` gives $T_{lg}$, its default $T_a$ at point $k$ and exactly $T_r$ at $k = 0$. THM-9 is `ThermalModel.injection_temperature`, a static method; whether to mix is the caller's (`lift_gas_mixing`).
  - `System.params(bc)` is an instance method now, since $T_{lg}$'s default depends on the well, and `solvers.rust.operating_point(wp, bc)` takes the well (it was `operating_point(bc)`). `System.bounds` takes the default $T_{lg}$ in.
  - `point_rows`, `cell_slope` and `energy_slope` take two more inputs, `lift` and `lift_prev`; `System.cell_inputs()` gives every cell's inputs after `params`, and the march passes them. With the lift gas at the bottomhole the two inputs are not used, and the functions' expressions are those before the feature. Where they are symbols, DISC-12 is chosen by `ca.if_else`; in the residual, built point by point, by the point's index.
  - `configurations.check` lists `injection_md` for `v1.0.0` whenever it is not `None`, even at the bottomhole.
- **Rust.** `WellSpec.injection_point`, refused beyond $N$, passed by `solvers/rust.py` and `Well::new(..., injection_point=0)`; `geometry::Cell.index`; `discretization::lift_gas_rate`, `inflow_temperature` and `mixes_lift_gas`; `energy_row` takes $w_\text{res}$, for DISC-12's phase rates at point $k$; `thermal::injection_temperature` (THM-9) and `lift_gas_share` ($\lambda$, below). The components `injection_point`, `point_phase_rates`, `injection_temperature` and `energy_row` serve the vectors.
- **The core's temperature solve at the injection cell** (`specs/architecture.md`, Rust core, design point 4). DISC-12's row is $(1 - \lambda)$ times DISC-10's row plus $\lambda\thinspace(T_k - T_{lg})$, with $\lambda = H_{lg}/(H_k + H_{lg})$ in $[0, 1]$. So:
  - the closed-form step of a linear row (heat loss alone, no mass transfer) is not used at cell $k$; the general solve takes that one cell;
  - the bracket's ends become $\min(T_\text{lo}, T_{lg})$ and $\max(T_{k-1}, T_{a,k}, T_{lg})$, where both parts of the row have the signs the bracket needs, and the step-outs are as before;
  - the chord iteration's slope is $(1 - \lambda)$ times the heat loss's plus $\lambda$, and the test of a solved temperature uses the same slope.

  These are adaptations of the existing solve to the new row, not a new fallback; their measured effect is under Measurements, for sign-off.
- **$R(p_0)$ below the separator** is unchanged: the rate at the last point the march reached, with the lift gas. That is the rate leaving the well, so $R$ does not jump where the march stops below the injection point.
- **Test wells** in `input.rs`'s `all()`: `w1_valve` (develop's model, cold lift gas at a valve halfway down), `w1_hot_valve` (heat loss alone, lift gas at 420 K, hotter than the reservoir, at three quarters of the depth), `w2_valve` (L-shaped, Joule–Thomson cooling, 1.5 kg/s at 290 K at three quarters) and `w1_valve_no_mixing` (the mixing off).
- **Vectors.** INF-9 has two tables, since a `develop` table's inputs are numbers: `INF-9`, the injection point on a uniform and a non-uniform grid, with two depths halfway between grid points, and `INF-9 (phase rates)`. THM-9 and DISC-12 have one each.
- **Example.** `scripts/sim_examples/gl_valve.py` solves a gas-lifted well with the valve at the bottomhole, at 75% and at 50% of its measured depth; `docs/simulate.md` describes `injection_md`.
- **No other solver machinery.** The CasADi search, the march's fallbacks and the core's scan are unchanged.

## Acceptance

1. **The `v1.0.0` configuration is unchanged.** The row vectors pass, and both verifier candidates report PASS with 0 failures, 0 expected failures and a stable-root rate of 100%.
2. **At the bottomhole nothing changes.** With `injection_md=None` or $z_\text{inj}$ at the bottomhole, every row of `develop`'s test wells is the same to the bit as before, in both backends.
3. **Vectors** for INF-9 (the injection point on a grid, and the phase rates below and above it), THM-9 and DISC-12, in `specs/tools/make_develop_vectors.py`, checked in both backends by `tests/test_spec_vectors.py`.
4. **Property tests** (`tests/test_model_properties.py`) on gas-lifted wells with the valve at half and three quarters of the depth, on both backends. In each:
   - the total mass rate is $w_\text{res} + w_{g,\text{res}}$ below point $k$, and $w_{lg}$ more from point $k$ up;
   - the energy balance holds across the injection point;
   - the solution converges at first order as the grid is refined;
   - the two-root stability property holds.
5. **Backend comparison** (`tests/backend_cases.py`): feature `021`; overlays with the valve at 50% and 75% of the measured depth; matrix entries; comparison-set groups. The rows agree in every configuration, and the comparison set passes.
6. **The Rust core.** `cargo test --no-default-features` passes, with the march's assumption checks on the new test wells, the temperature solve's bracket included (`specs/architecture.md`, Rust core, design point 4).
7. **Measurements** (`plans/evidence/`; sampled wells only). The sampler's gas-lifted wells, with the valve at the bottomhole, at 75% and at 50% of the measured depth, on both backends:
   - the operating points found;
   - the change in PBH, PWH, TWH and the reservoir rate;
   - the share of wells that flow only with the lift gas at the bottomhole.

   A fall in the operating points found is a solver finding unless the model has no root, which the core's scan of $R(p_0)$ shows.
8. **The suites.** The full test suite passes.

Done on the branch, 2026-10-29, on `validity-refusals` at `8c66a60` with the feature (Python 3.14, casadi 3.8.1):

1. **The `v1.0.0` configuration is unchanged.** The row vectors pass. Both verifier candidates report PASS, with 0 failures, 0 expected failures, 141 cases and a stable-root rate of 100%. Each has its one finding at `fold-1503`, an unstable root the reference lacks; the CasADi candidate's 201 roots are the same to the bit as a run of it before the feature.
2. **At the bottomhole nothing changes** (`plans/evidence/injection_depth_rows.py`). On the 74 configurations of the backend comparison's matrix that the base branch has (10 cells) and on 6 sampled wells of the comparison set (100 cells, 4 of them with at least 1 kg/s of lift gas), with `injection_md=None` and with `injection_md` at the bottomhole: both backends' rows at the trial states and at the perturbed roots (31,192 values), the parameters, the bounds, and both backends' root sets (202 roots, with their labels and slopes) are the same to the bit as on the base branch.
3. **Vectors.** `INF-9` (9 rows, two of them halfway between grid points), `INF-9 (phase rates)` (6), `THM-9` (5) and `DISC-12` (4), in both backends. No other develop vector changed.
4. **Property tests** (`tests/test_model_properties.py`): two wells with the valve above the bottomhole, a deviated black-oil well with cold lift gas at a valve halfway down ($k$ = 50 of 100) and an L-shaped one with the lift gas at three quarters of the measured depth, in its build section, at the default $T_{lg}$. Every property of the file holds on both backends, and the backends find the same roots. The total mass rate is $w_\text{res} + w_{g,\text{res}}$ below point $k$, to SMO-1's floor of free gas where the oil is undersaturated, and $w_{lg}$ more from point $k$ up, constant to $10^{-6}$. At point $k$ the heat-capacity flux of the state balances that of the flow from below at $\tilde T_k$ and the lift gas's at $T_{lg}$, to $10^{-9}$. The solution converges at first order on 40, 80 and 160 cells. A two-root well keeps its stable and its unstable root with 5 g/s of lift gas at a valve at half and at three quarters of the depth; more lift gas leaves one root. `tests/test_thermal.py` checks THM-9's limits and that THM-9 at the bottomhole is THM-5.
5. **Backend comparison** (`tests/backend_cases.py`): feature `021`; the overlays `valve at 50%` and `valve at 75%`, each with at least 1 kg/s of lift gas; the matrix entries `W1`, `W2` and `develop` with each overlay, `W1+lift-gas temperature+valve at 75%`, `W2+energy terms+joule-thomson+lift-gas temperature+valve at 50%`, `develop+lift-gas temperature+valve at 50%`, `develop+deviated+valve at 75%` and `develop+no lift-gas mixing+valve at 50%`; the comparison-set groups `v1.0.0+valve at 50%`, `develop+valve at 50%` and `develop+valve at 75%`. The rows agree in every configuration of the matrix, to $3 \cdot 10^{-13}$ on the new groups, and the comparison set passes (Measurements, The backends).
6. **The Rust core.** `cargo test --no-default-features` passes 53 tests, with the march's assumption checks, the temperature solve's included, on the four new test wells.
7. **Measurements**: below.
8. **The suites.** The full test suite passes: 1058 passed and 1 skipped. Its one warning is the known extra root at `v1.0.0+L-shaped#0`, from 016.

## Measurements

The scripts are in `plans/evidence/` (`README.md` there). The wells are all synthetic: `develop`'s sampler, seed 2026, in `develop`'s default configuration.

**Sampled wells** (`injection_depth_wells.py run 1000`). The cases of `five_regime_default.py`'s draw from 1000 wells that carry lift gas: the nominal operating point has none (SMP-28), so these are the two sampled operating points of each of the 145 wells with gas lift, 290 cases, at 0.004 to 5.0 kg/s of lift gas (median 2.7) and 1550 to 7241 m of measured depth. Each is solved with the lift gas at the bottomhole and at a valve at 75% and at 50% of the measured depth from the surface, $T_{lg}$ not given, so the ambient temperature at the valve. The Rust core solves all 290, the CasADi backend the first 90.

| Lift gas at | Operating points, core (of 290) | CasADi (of 90) | PBH change, median (10th and 90th percentile; range) | PWH change, median (range) | TWH change, median (range) | Rate change, median (range) |
|---|---|---|---|---|---|---|
| the bottomhole | 287 | 90 | – | – | – | – |
| 75% of MD | 286 | 90 | +0.53 bar (−0.39, +2.85; −5.7 to +10.1) | −0.54 bar (−44.3 to +5.4) | −1.0 K (−11.6 to +3.0) | −2.0% (−67% to +27%) |
| 50% of MD | 283 | 89 | +1.44 bar (−0.00, +6.71; −3.2 to +23.9) | −1.6 bar (−85.5 to +6.0) | −2.7 K (−27.3 to +2.1) | −5.7% (−99.8% to +28%) |

*The changes are the core's, against the bottomhole, over the cases with an operating point in both: 286 and 283.*

- **The valve lifts less, as the motivation expects.** In the 137 vertical cases PBH rises by 0.78 bar at the median with the valve at 75% and by 1.88 bar at 50%, by up to 10.1 and 23.9 bar, and the rate falls with it. The wellhead is cooler, by 2.7 K at the median at 50%: the lift gas enters at the ambient temperature at the valve, below $T_r$, and the rate is lower.
- **L-shaped wells gain.** In the 71 L-shaped cases PBH falls by 0.25 bar at the median with the valve at 75%, by up to 5.7 bar, and the rate rises by up to 27%. Their horizontal section is 11 to 56% of the measured depth (28% at the median). There the lift gas adds friction and lifts nothing, so a valve that takes it out of all or part of that section helps. PBH falls in 71 cases at 75% and in 30 at 50%.
- **Wells that flow only with the lift gas at the bottomhole:** 1 of the 287 cases with an operating point (0.3%) with the valve at 75%, (864, 2) as (well, operating point), and 4 (1.4%) at 50%: (302, 2), (577, 1), (864, 2) and (958, 1). None is gained. In each the model has no root: the core's $R(p_0)$ (`injection_depth_wells.py scan`) is positive on 4000 samples from $p_s$ to $10^{-6}$ bar below $p_r$, at least 0.027 to 0.87 kg/s, with its least value within 0.5 bar of $p_r$, the no-flow limit. The CasADi backend loses (302, 2) too, the one of the four among its 90.
- **The backends agree.** They find an operating point in the same cases at each depth, and their PBH agree to $1.0 \cdot 10^{-8}$, $4.8 \cdot 10^{-8}$ and $2.9 \cdot 10^{-7}$ bar. Case (302, 1) with the valve at 50% has four roots in both: stable at 156.21 bar, unstable at 158.11, stable at 159.96 and unstable at 162.30; its operating point is the first (SOL-6).
- **Cost.** On one process the core takes 85.3, 75.7 and 73.1 s for the 290 cases with the lift gas at the bottomhole, at 75% and at 50%, 0.14, 0.13 and 0.13 s at the median: no more. The CasADi backend takes 1.06 and 1.11 times as long per case at the median with the valve at 75% and at 50%, and 19% and 22% longer in total (638, 758 and 775 s, on 12 processes; a case's three depths ran one after another in one process).

**The core's temperature solve at the injection cell** (Implementation; principle 7). The same 290 cases on the core on one process, with each adaptation undone in turn:

| Variant | Operating points (75%, 50%) | Chord fallbacks at 75% / 50% | Temperature solves at 75% / 50% |
|---|---|---|---|
| as implemented | 286, 283 | 219,414 / 206,078 | 31.64 / 31.21 million |
| the heat loss's slope in the chord iteration, not $(1 - \lambda)$ of it plus $\lambda$ | 286, 283, PBH within $3 \cdot 10^{-13}$ bar | 224,378 / 212,192 (+2.3%, +3.0%) | the same to 0.003% |
| the bracket without $T_{lg}$ | the same to the bit | the same | the same |

- The slope saves 2 to 3% of the chord fallbacks, a few thousand temperature solves in 31 million, and no measurable time: the variants' times differ by 3 to 4%, as much as with the lift gas at the bottomhole, whose code path they share.
- The widened bracket changes nothing on these cases: the chord iteration converged at every injection cell, so the bracket was never needed there. It is what keeps the bracket's ends of the right sign by construction (design point 4), and costs nothing.

**The backends** (`scripts/verification/compare_backends.py`, 12 processes): with 20 sampled wells per group, all 60 cases pass Step 9's rule; with 60 per group, 179 of 180, with the groups `v1.0.0+lift-gas temperature` and `develop+lift-gas temperature`, the lift gas at the bottomhole, as controls:

| Group | Cases | CasADi roots | Core roots | Missed by the core | Core only (zero the CasADi rows) | Several $\alpha$ | Largest row difference |
|---|---|---|---|---|---|---|---|
| `v1.0.0+valve at 50%` | 60 | 59 | 62 | 0 | 3 (3) | 2 | $2.4 \cdot 10^{-13}$ |
| `develop+valve at 50%` | 60 | 60 | 59 | 1 | 0 | 0 | $2.9 \cdot 10^{-13}$ |
| `develop+valve at 75%` | 60 | 60 | 60 | 0 | 0 | 0 | $3.0 \cdot 10^{-14}$ |
| `v1.0.0+lift-gas temperature` | 60 | 59 | 58 | 1, on another branch | 0 | 2 | $6.8 \cdot 10^{-14}$ |
| `develop+lift-gas temperature` | 60 | 60 | 60 | 0 | 0 | 0 | $6.0 \cdot 10^{-14}$ |

The core is 25 times faster than the CasADi backend at the median on the first run's 60 cases.

- **Solver finding: the CasADi search misses three operating points in the `v1.0.0` configuration with a valve.** In `v1.0.0+valve at 50%` #17, #22 and #27, the core finds a stable root that zeroes the CasADi rows, and the CasADi search finds none: from every start the march fails or Ipopt reports the problem infeasible. With the lift gas at the bottomhole it finds each of them. #22 and #27 have several void fractions under the fitted classifier, and #17 a nearly closed choke ($u$ = 0.08); at #17 the march from $p_0$ at the root fails at point $k$, where the lift gas takes $\alpha$ from 0.65 to 0.93 in one cell, and from the core's root Ipopt converges. In `develop`'s default configuration the CasADi search missed none of the 210 cases with a valve (here and in the sampled wells). No solver machinery is added for it.
- **Solver finding: the core misses an unstable root.** At `develop+valve at 50%` #22, a gas-rich well ($R_{go}$ = 936 Sm³/Sm³), the CasADi search finds an unstable root at 265.65 bar besides the operating point at 74.66 bar, which both find. At that root the wellhead lies at 24.74 bar, just above $p_s$ = 24.70 bar, with $\alpha$ = 0.86: on a second solution of the wellhead point's other rows, along which the last cell's momentum row rises again near $p_s$. The core's cell solve takes the momentum row's first sign change below $p_{N-1}$, at 77.66 bar, where $R$ is −145 kg/s. The comparison rule's test of another branch, the sign of the point's Jacobian, does not flag it. The kind of miss known at `develop#17` (`plans/evidence/README.md`, issue #8); the operating point is the same. Not changed.

## Out of scope

- **Annulus flow**, the casing pressure, and the valve's or orifice's flow equation: the lift-gas rate stays a boundary condition. Annulus flow is a non-goal (`specs/goals.md`).
- **Several injection points**, such as unloading valves.
- **Casing heading** and other transients of gas lift: the model is steady state, a non-goal.
- **The lift gas's temperature profile in the annulus.** $T_{lg}$ is given, or the ambient temperature at the injection point.
- **The lift gas dissolving into the oil** (PVT-OIL-13 keeps it free, as today).
- **Drawing an injection depth** in the sampler: the redesign's.

## Approval (Bjarne, 2026-10-29)

- **The spec**, as written, before the implementation starts, with the four design choices as proposed: the depth on `WellProperties` as `injection_md`, the nearest grid point, the ambient temperature at the injection point as $T_{lg}$'s default, and the mixing in the injection cell's energy row. Approved.
- **The new IDs** INF-9, THM-9 and DISC-12, and **the edits** to `specs/model/` listed in the Delta. Approved.

The implemented feature's measurements (Acceptance, item 7), and any solver machinery the implementation needs, come back for sign-off, as in 016, 019 and 020.

## Sign-off (Bjarne, 2026-10-29)

On the implemented feature, each item signed off:

- **The changed results** where the lift gas enters above the bottomhole (Measurements, Sampled wells): the column below the valve is heavier, PBH rises and the rate falls in vertical and deviated wells, and L-shaped wells gain where the valve leaves the horizontal section. 1 and 4 of 287 cases lose their operating point with the valve at 75% and 50%, where the model has no root. Nothing changes where the lift gas enters at the bottomhole, `develop`'s default (Acceptance, item 2).
- **The implementation's differences from the Delta** (Implementation): point 0's temperature row is THM-3 where the injection point is above the bottomhole, stated in INF-9 and DISC-11; `System.params` became an instance method and `solvers.rust.operating_point` takes the well; `point_rows`, `cell_slope` and `energy_slope` take `lift` and `lift_prev`; `configurations.check` refuses any `injection_md` in the `v1.0.0` configuration, the bottomhole's too.
- **The core's temperature solve at the injection cell**, with its measured effect (principle 7): the general solve in place of the closed-form step, the bracket widened to take in $T_{lg}$, and the chord iteration's slope $(1 - \lambda)$ times the heat loss's plus $\lambda$, which saves 2 to 3% of the chord fallbacks and no measurable time. The slope could be the heat loss's alone, with the same results; the widened bracket is what keeps design point 4's proof. Both kept: they are the existing chord iteration and bracket applied to DISC-12's row.
- **The CasADi backend's cost**: 6 and 11% more time per case at the median, 19 and 22% in total, with the valve at 75% and 50%.
- **The solver findings** (Measurements, The backends): the CasADi search misses 3 of 60 operating points in `v1.0.0+valve at 50%`, none in `develop`'s configuration; the core misses one unstable root, at `develop+valve at 50%` #22, of a kind known at `develop#17`. Neither is changed; both are recorded in `plans/improvements.md` §2.9.

