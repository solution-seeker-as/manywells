# ManyWells v2 architecture

*Step 6 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: decided, 2026-09-30; Bjarne ruled on its five open decisions (Decisions, below). Step 7 implemented the Python side (2026-10-01); its changes to this file are marked "Step 7", and Bjarne signed them off on 2026-10-01.*

This file fixes the module boundaries, the interfaces between modules and the extension points of ManyWells v2, and the design of the Rust core for `develop`'s model. It does not define physics (`specs/model/`), the sampling procedure (`specs/sampling.md`), the checks (`specs/verification.md`), the v2 dataset schema (release work, `specs/goals.md`) or the calibration (`specs/calibration.md`), whose interface it gives under Interface contracts. Step 7 implements the Python side, and the Rust core follows in Steps 8 and 9.

A **module** is one part of the code with one job: for a model part, its spec file, its Python module and, in the core, its Rust module of the same name. The solvers are the exception, because the two backends solve differently (`solvers/` in Python, `march.rs` and `shoot.rs` in the core). A feature belongs to exactly one module (feature map, below).

## Layers

```
inputs          WellProperties, BoundaryConditions, configurations      Python
components      geometry  pvt  slip  friction  thermal  inflow  choke   ┐
discretization  the rows of each point, in DISC-6 order                  ├ backend: CasADi or the Rust core
solvers         march, root search, stability label; Ipopt adapter      ┘
solution        Root, RootSet, operating point (SOL-4 to SOL-6)          Python
simulator       SSDFSimulator: the public API over one backend           Python
on top          sampling, datasets, calibration
```

The **backend** is the part that holds the model equations: components, discretization and solvers. There are two: `develop`'s Python/CasADi code, and the Rust core, which covers the `v1.0.0` configuration since Step 8 and the whole model from Step 9 on. Both are kept and developed together (`specs/goals.md`, 2026-10-01). Everything else is Python and is shared by both backends, so switching backends changes no input, output or selection rule.

Dependency rules, checked in review:

- Components import only `units`, `ca_functions` and, within `pvt/`, each other. A component that needs another one's output receives it as an argument or as the component object; it does not import it.
- `discretization` imports the components. `solvers` import `discretization` and `solution` and know nothing about the physics.
- Nothing in the backend imports `simulator`, `sampling`, `datasets` or `calibration`. Nothing in `src/` imports `scripts/` or `verification/`, and `verification/` imports nothing from `manywells` (`specs/verification.md`).
- No model equation is in the shared Python layer: each is in both backends, under the same spec ID. The backend returns what the Python layer would otherwise have to recompute (below).

## Modules

| Module | Owns | Spec | On `develop` today |
|---|---|---|---|
| `simulator.py` | `SSDFSimulator`, `WellProperties`, `BoundaryConditions`, `SimError`, `NoOperatingPoint`; backend choice | this file | exists; its assembly moves to `discretization`, `thermal` and `solvers` |
| `configurations.py` | named configurations (`v1.0.0`, `develop`); the map from v1.0.0's parameters to `develop`'s inputs; the check that a well is in a configuration | `model/README.md`, `sampling.md` SMP-40 | new (Step 7) |
| `geometry.py` | trajectory, grid, cross-section | `model/geometry.md` | exists |
| `pvt/` | `FluidModel`: phase densities, viscosities, surface tension, heat capacities, the reservoir gas–liquid split, phase mass rates (mass transfer) | `model/pvt/*.md` | exists; the split and the phase rates move in from `simulator.py` |
| `slip.py` | drift-flux law, rise velocities, flow-regime classifier | `model/slip.md` | exists |
| `friction.py` | viscous pressure gradient, friction-factor options | `model/friction.md` | exists; becomes a component in `WellProperties` |
| `thermal.py` | heat loss, ambient profile, frictional heating, gravity term, inflow and lift-gas temperature | `model/thermal.md` | new; the terms move out of `simulator.py` |
| `inflow.py` | inflow models | `model/inflow.md` | exists |
| `choke.py` | choke models, profiles, critical flow | `model/choke.md` | exists; called polymorphically |
| `ca_functions.py`, `units.py` | smoothing; constants and conversions | `model/smoothing.md`, `model/nomenclature.md` | exist |
| `discretization.py` | balances, the rows of each point, their order, the system of a well | `model/balances.md`, `model/discretization.md` | new; moves out of `simulator.py` |
| `solvers/` | `ipopt.py` (adapter), `march.py` (initial guess), `roots.py` (multi-start search, stability label) | `model/solution.md` SOL-3; the rest is numerics, not physics | new; moves out of `simulator.py` |
| `solution.py` | `Root`, `RootSet`, operating-point selection | `model/solution.md` SOL-4 to SOL-6 | new |
| `sampling/` | `wells.py`, `conditions.py`, `generate.py`: well draws, operating-point draws, the generation procedure | `sampling.md` | new (Step 7), ported from `scripts/data_generation/` |
| `datasets/` | `schema.py`, `rows.py`, `io.py`: feature definitions, the features of a solved sample, writers and readers | `docs/datasets.md`; the v2 schema is release work | new |
| `calibration/` | `data.py`, `parameters.py`, `objective.py`, `fit.py`, `synthetic.py`: one well's calibration data and noise, the free parameters and their priors, the residuals of each row's operating point, the MAP fit, synthetic data at known parameters | `calibration.md` | rewritten (`plans/calibration-plan.md`), with the contract below |

The sampler lives in `src/`, not in `scripts/`, because the traceability test looks for the SMP tags in `src/` and `rust/`. The generator scripts become thin callers of `sampling.generate`. The candidate adapter that runs `develop` on the verifier's case set imports both `manywells` and `manywells_verify`, so it is a script, `scripts/verification/develop_candidate.py` (Step 7).

## Interface contracts

Units are those of `specs/model/nomenclature.md`: SI in equations; bar and kelvin in every argument, return value, state vector, parameter vector and dataset row. `CF_BAR` converts where an equation needs Pa.

### Inputs

- **Immutable and validated.** `WellProperties`, `BoundaryConditions`, `FluidModel`, `WellGeometry` and every component model are frozen dataclasses. They validate in `__post_init__` and raise `ValueError` (`plans/improvements.md` §2.1). The simulator never writes to them (§2.5); a changed input is a new object (`dataclasses.replace`). Frozen classes also reject assignment to a field that does not exist, the silent failure of §1.3.
- **`WellProperties`** holds one object per model part: `geometry`, `fluid`, `friction`, `thermal`, `slip`, `inflow`, `choke`. `f_D` and `roughness` move into `friction`, and `h` into `thermal`. The default choke, a Bernoulli choke with $K_c = 0.1A$, is set in `__post_init__`, not by the simulator.
- **`BoundaryConditions`** is unchanged: `p_r`, `p_s` (bar), `T_r`, `T_s`, `T_lg` (K; `None` means $T_r$), `u` (–), `w_lg` (kg/s). It is the parameter vector of a well's system, so changing it never rebuilds anything.
- **Configurations.** A configuration is a choice of one option per model part (`specs/model/README.md`). The options live in the component objects, so a configuration is not an object the simulator takes: `configurations.v1_well(...)` builds a `WellProperties` in the `v1.0.0` configuration from v1.0.0's parameters, those of a verifier case (`L`, `D`, `rho_l`, `R_s`, `cp_g`, `cp_l`, `f_D`, `h`, `f_g`, the inflow and choke parameters and N), and `configurations.check(wp, 'v1.0.0')` raises and lists every option that differs. The `develop` configuration is the dataclasses' defaults.

### Components

Every function of the state accepts CasADi symbols as well as floats in the CasADi backend, and `f64` in the Rust core. `s` is a `PointState`: the seven state values of one point, in state-vector order, with $\rho_m$ and $v_m$ (BAL-7, BAL-8) as properties.

| Component | Function | Inputs | Output | Options in `v1.0.0` · other options |
|---|---|---|---|---|
| `WellGeometry` | attributes | – | `n_cells`; `md`, `tvd` (m, N + 1, bottomhole first); `delta_md`, `cos_incl` (N; entry i − 1 is cell i, between points i − 1 and i); `tvd_frac` (N + 1); `D` (m); `A` (m²) | vertical, uniform (GEO-1) · any (MD, TVD) survey |
| `FluidModel` | `gas_density(p, T)`, `gas_law_row(p, T, rho_g)` | bar, K; bar, K, kg/m³ | kg/m³; bar | ideal gas (PVT-GAS-1) · Dranchuk–Abou-Kassem (PVT-GAS-11), Papay z-factor (PVT-GAS-3, PVT-GAS-4) |
| | `jt_factor(T, rho_g)` | K, kg/m³ | $J$, – | 0 (ideal gas) · DAK's (PVT-GAS-10) |
| | `liquid_density(p, T)` | bar, K | kg/m³ | constant (PVT-MIX-1) · black oil |
| | `surface_tension(p, T, rho_l)` | bar, K, kg/m³ | N/m | dead oil at the state's $\rho_l$ (PVT-MIX-5) · at $\rho_o$, with the live-oil correction |
| | `liquid_viscosity(p, T)`, `gas_viscosity(T, rho_g)` | bar, K; K, kg/m³ | Pa s | – (unused by fixed $f_D$) |
| | `reservoir_gas_rate(w_res)` | kg/s | $w_{g,\text{res}}$, kg/s | INF-4 |
| | `phase_rates(p, T, w_res, w_lg)` | bar, K, kg/s, kg/s | $(w_g, w_l)$, kg/s | no mass transfer (BAL-3): constants, exactly · dissolved gas |
| | `cp_g`, `cp_l` | – | J/(kg K) | constants |
| `SlipModel` | `identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl)` | SI | $(C_0, v_\infty)$ | three regimes (SLIP-1 to SLIP-8) · inclination terms; four regimes (after the plan) |
| | `flow_regime(...)` | floats | regime name | |
| `FrictionModel` | `pressure_gradient(s, fluid, D)` | state, fluid, m | $F$, Pa/m (FRIC-1) | fixed $f_D$ (FRIC-2) · roughness with Chen or Haaland |
| `ThermalModel` | `ambient_temperature(tvd_frac, T_r, T_s)` | –, K, K | K | linear in depth (THM-2; implemented as THM-4) |
| | `inflow_temperature(w_res, w_lg, T_r, T_lg, fluid)` | kg/s, kg/s, K, K, fluid | K | $T_r$ (THM-3) · lift-gas mixing |
| | `temperature_gradient(s, fluid, T_a, F, dp_dmd, cos_incl, D)` | state, fluid, K, Pa/m, Pa/m, –, m | $dT/d\text{MD}$, K/m | heat loss (THM-1) · frictional heating, gravity term, Joule–Thomson term |
| `InflowModel` | `liquid_mass_flow_rate(p_0, p_r)` | bar, bar | $w_\text{res}$, kg/s | Vogel, productivity index (INF-1, INF-2) · fixed rate |
| `ChokeModel` | `mass_flow_rate(u, p_s, s, A)` | –, bar, state, m² | $w_c$, kg/s | Simpson, Bernoulli with four profiles (CHK-2 to CHK-10) |
| | `is_choked(p_N, p_s)` | bar, bar | bool | CHK-12 |

Contract changes against `develop` today, all in Step 7:

- `ChokeModel.mass_flow_rate` takes the wellhead state and each model reads what it needs, so the simulator no longer dispatches on `isinstance` (§2.3).
- `reservoir_gas_rate` and `phase_rates` move the gas–liquid split (INF-4) and the mass-transfer expression from the simulator into the fluid model, where `thermal` also finds the split for the lift-gas mixing temperature. With dead oil `phase_rates` returns $(f_g/(1-f_g)\,w_\text{res} + w_{lg},\ w_\text{res})$ exactly, not through the smooth min, so the `v1.0.0` rows need no bypass (`plans/develop_model_changes.md`, change 8).
- `surface_tension` takes the state's $\rho_l$, which the `v1.0.0` option needs (change 3).
- `temperature_gradient` returns $dT/d\text{MD}$ at point $i$, so the energy row of every thermal option is $T_i - T_{i-1} - \Delta\text{MD}_i\,(dT/d\text{MD})_i$; `v1.0.0` has $dT/d\text{MD} = -H$ (THM-1). `dp_dmd` is the cell's pressure gradient $c_\text{bar}(p_i - p_{i-1})/\Delta\text{MD}_i$, which no option uses: the Joule–Thomson term (THM-8) takes $F + \rho_m g\cos\theta$ for the pressure gradient, as THM-6 and THM-7 do, which keeps the energy row a function of point $i$'s state and $T_{i-1}$ (`specs/features/016-joule-thomson.md`).
- `FluidModel.p_sep` and `p_bubble` are in bar, like every other interface (§1.6).
- The slip parameters become dataclass fields (§2.7).

### Discretization

`build_system(wp) -> System`, once per well:

| Member | Content |
|---|---|
| `n_x` | $7(N+1)$ |
| `params(bc)` | the parameter vector $[p_r, p_s, T_r, T_s, T_{lg}, u, w_{lg}]$, with $T_{lg} = T_r$ when it is `None` |
| `residual` | `ca.Function(x, params) -> r`: every row, in DISC-6 order and canonical units |
| `row_ids` | the spec ID of each row, such as `INF-6` or `CHK-1`, so tests and the stability label find rows by ID rather than by position |
| `bottom_rows` | `ca.Function(x_0, params) -> r_0`: the six rows of point 0; the march fixes $p_0$ and solves them for the other six unknowns |
| `point_rows` | `ca.Function(x_i, x_prev, w_res, params, delta_md, cos_incl, tvd_frac) -> r_i`: the seven rows of point $i > 0$ without CHK-1, one function for every cell, used by the march |
| `bounds(bc)` | `(lbx, ubx)`: the admissible box of SOL-1 and the solver's temperature bounds |
| `regime_probabilities` | `ca.Function(x) -> P`: the flow-regime probabilities at each point, one function mapped over the points (`plans/improvements.md` §4.4), for the labels of SLIP-8 (Step 7: split from `outputs`, as it needs no parameters) |
| `outputs` | `ca.Function(x, params) -> ...`: the rest of what a `Root` carries besides its state: CHOKED (CHK-12) and the reservoir phase rates |
| `reservoir_rate`, `bottom_guess` | `ca.Function(p_0, params)`: $w_\text{res}$, and a start for point 0's rows at $p_0$, for the march (Step 7) |

- **No hidden state** (§2.4). The reservoir liquid rate $w_\text{res} = $ `inflow.liquid_mass_flow_rate(p_0, p_r)` is an expression of $x_0$ and the parameters, passed to every point's rows as an argument.
- **One row form for every trajectory.** The rows use $\Delta\text{MD}$ for friction and heat loss, $\Delta\text{MD}\cos\theta$ for gravity, and $T_a$ from the TVD fraction. On a vertical uniform grid they reduce to DISC-4, and to DISC-5 with the `v1.0.0` thermal option, so a new trajectory never needs a new row.
- **One mass-row form for every fluid** (decision 4; Step 7 writes it into `discretization.md` and `balances.md` with new IDs): $(\alpha\rho_g v_g)_i - (\alpha\rho_g v_g)_{i-1} - \big(w_g(p_i, T_i) - w_g(p_{i-1}, T_{i-1})\big)/A$, and the same for the liquid. With the dead-oil `phase_rates` the rate difference is identically zero, so this is DISC-2 and DISC-3 as functions of the state, which the `v1.0.0` configuration requires (`discretization.md`, Interface). With dissolved gas it has the same roots as `develop`'s current rows $A\alpha\rho_g v_g - w_g(p_i, T_i)$, since INF-6 fixes point 0. Mass transfer then lives in `pvt/` alone.
- **Integrator.** Implicit Euler is the only scheme. Another scheme would be a DISC option here, and it would change the order the Convergence check expects.

### Solvers

| Function | Contract |
|---|---|
| `IpoptSolver(system)` | builds the feasibility NLP once, with the operating point as NLP parameters (`nlpsol` with `p`) |
| `.solve(x0, params, lbx, ubx) -> SolveResult` | `x`, `success`, `status`, `stats`; it never prints (§2.2) |
| `Marcher(system)(params, p_0) -> x` | the initial guess: point 0 from `bottom_rows`, then each point from `point_rows`, each by a Newton rootfinder built once per well, and by Ipopt with v1.0.0's bounds where Newton fails (Step 7: a class, so its solvers are built once; the fallback is solver machinery, measured in `specs/features/012-root-search.md`) |
| `RootFinder(system).find(bc, x_guess=None) -> RootSet` | solves from each start: `x_guess` if given, the default march from $p_0 = p_r - 0.05(p_r - p_s)$, and marches from $p_0 = p_s + f(p_r - p_s)$ for $f \in \{0.5, 0.7, 0.85, 0.975, 0.995, 0.999\}$ (the verifier's method A, without its duplicate of the default, and 0.975 and 0.999; Step 7, measured in feature spec 012); accepts a solve that Ipopt reports as `Solve_Succeeded`, as the reference build does, and that is admissible (SOL-1); merges roots within `tol_x` of each other in the verifier's state distance; labels each by SOL-3 from `residual`'s Jacobian, with the CHK-1 row and the $p_0$ column found through `row_ids`, and calls a label indeterminate where the normalized slope is at most `label_min` in magnitude (decision 5). Step 7: a class, so that the solver, the march and the Jacobian are built once per well |

The library cannot import the verifier, so it defines its own copies of the state distance, `tol_x` and `label_min`, and a test in `tests/` checks that they equal the verifier's (`tests/test_roots.py`). Step 7 measured the starts on the case set (feature spec 012); `simulate` tries every start, with no early exit.

Another solver adapter (Newton or KINSOL on the square system, JIT compilation, dual warm starts; `plans/improvements.md` §4.2) goes in `solvers/` with the same `solve` contract, and is solver machinery under principle 7.

### Solution

- `Root`: `x` (array of $7(N+1)$ values in state-vector order), `label` (`stable`, `unstable` or `indeterminate`), `slope` ($dR/dp_0$ normalized by $(p_r - p_s)/w_m$, as the verifier does), `choked` (CHK-12), and per point the `flow_regime`, with the reservoir phase rates.
- `RootSet`: `roots`, sorted by $p_0$; `operating_point` (a `Root` or `None`); `several_stable` (SOL-6's flag); and the search record (starts tried and failed), which is not part of the model.
- `select_operating_point(roots)` implements SOL-4 to SOL-6 once, for every backend.

### Public API

```python
sim = SSDFSimulator(wp)                  # validates and builds the well's system once
op = sim.simulate(bc)                    # the operating point, a Root; raises NoOperatingPoint
rs = sim.root_set(bc)                    # every root found, labelled; empty if the well cannot flow
df = sim.solution_as_df(op)              # per point: state, md, tvd, flow regime
op2 = sim.simulate(bc2, x_guess=op.x)   # a warm start makes the search faster, not the answer different
```

`NoOperatingPoint` is a `SimError` and carries the root set (SOL-5). Log messages go to `logging.getLogger('manywells')`. Whether `simulate` tries every start or stops early is solver policy: Step 7 measured its stable-root rate and cost on the case set, and each backend may choose differently, because only the returned operating point is specified (principle 6). The two-argument constructor `SSDFSimulator(wp, bc)` with `simulate()` keeps working, with a `DeprecationWarning`, until v2.0.0; it returns the operating point's state as a flat list, as before. `SSDFSimulator(wp, backend='casadi' | 'rust')` chooses the backend; both take the same inputs and return the same types. Step 8 added the `backend` argument with the Rust core; `'rust'` takes wells in the `v1.0.0` configuration only and refuses others, and builds no CasADi system.

### Calibration

*Added with `specs/calibration.md` (`plans/calibration-plan.md`, Step C1), 2026-10-02; signed off by Bjarne the same day.*

```python
data = CalibrationData(df, noise=Noise())                  # one well's rows: inputs, observations (NaN where missing)
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'roughness', 'h'], priors=None, backend='rust')
result.values, result.z, result.well, result.residuals    # MAP estimate, shifts in prior sds, calibrated well, table
table = evaluate(result.well, other_data, bc)             # the same table on rows it was not calibrated on
```

- **On top of the simulator.** `calibration/` calls only `SSDFSimulator(wp, backend=...).simulate(bc)` and the datasets' feature definitions (`datasets.rows.root_features`), so it works with both backends and holds no model equation. A free parameter is a plain field of a component (CAL-5), applied with `dataclasses.replace`, so every component the core has takes it.
- **One well per row.** Each row has its own fluid (`gor`, `wlr`), so it is solved on its own `WellProperties`: on the core, a new `_core.Well` costs nothing and the rows run in threads, since the core releases the GIL; on the CasADi backend, each row and parameter value builds a system, which is fine for cross-checks on small cases and too slow for routine use. Making the parameters symbols of the CasADi system (`plans/improvements.md` §4.5) would change that, and needs a measured gain (principle 7).
- **Errors.** `CalibrationError` (a `SimError`) when a row has no operating point at the prior medians, or the fit ends at parameters where one has none (CAL-9); `ValueError` for malformed data or parameters.

## Extension points

A new option is a contributor's change, not a runtime plug-in (decision 1): new equation IDs in its component file, the option in one module of each backend, off in the `v1.0.0` configuration, test vectors, and a feature spec (`specs/features/NNN-<name>.md`) approved first. A Python subclass of a component ABC works with the CasADi backend only; the Rust core has a closed set of options per module.

| Extension | What a new option adds | Where |
|---|---|---|
| Friction | a `FrictionModel` class: another friction-factor correlation, or a gradient that depends on the regime | `friction` |
| Well trajectory | a `WellGeometry` factory, such as from a directional survey file or an L-shaped well. Any (MD, TVD) survey already works, because the rows read only `delta_md`, `cos_incl` and `tvd_frac` | `geometry` |
| Thermal | a `ThermalModel` option: another heat-loss law, a term that needs the cell's pressure gradient (`dp_dmd`), or another ambient profile (such as a seabed temperature) through `ambient_temperature`. A new energy term must keep the Rust core's temperature bracket valid (design point 4) | `thermal` |
| Fluid | a gas law, oil model or mass-transfer law behind the `FluidModel` functions | `pvt/` |
| Slip | a slip law or flow-regime classifier, such as the four-regime model | `slip` |
| Inflow, choke | an `InflowModel` or `ChokeModel` class, or a choke profile | `inflow`, `choke` |
| Integrator | a DISC option | `discretization` |
| Solver | an adapter with the `solve` contract | `solvers/` |

## Rust core

The core implements `develop`'s model, every option in `specs/model/` that a row uses, including the `v1.0.0` configuration. Step 8 built it for that configuration (`specs/features/014-rust-solver.md`), and Step 9 ported the rest of `develop`'s model (`specs/features/015-rust-develop-model.md`). The equations no row uses (the sampler's liquid mixing, conversions between inputs, water's formation volume factor) are in Python only.

### Layout

```
rust/
  Cargo.toml            crate manywells-core, library _core; pyo3 behind the feature python
  src/
    lib.rs              bindings: module manywells._core (pyo3)
    input.rs            WellSpec and OperatingPoint, built once per well from the Python dataclasses
    geometry.rs  pvt/{gas,oil,water,mixture,fluid}.rs  slip.rs  friction.rs  thermal.rs
    inflow.rs  choke.rs  smoothing.rs  units.rs
    discretization.rs   the rows of one point (DISC, BAL, boundary rows)
    march.rs            the cell solve, and the march from p_0 to the wellhead
    shoot.rs            scan and bracket R(p_0): the roots and their labels
    scalar.rs           bracketed scalar root finding (Brent)
```

The port on `rust_implementation` maps onto it: `simulator.rs` splits into `discretization.rs`, `march.rs` and `shoot.rs`; `brentq.rs` becomes `scalar.rs`, `math.rs` `smoothing.rs`, and `constants.rs` `units.rs`; the others keep their names. Rust code carries `// spec:` tags, which the traceability test already reads.

### Design

1. **Inputs.** The Python dataclasses stay the inputs and keep the validation. `solvers/rust.py` converts a well to the core's plain parameters once, when `SSDFSimulator(wp)` is built, and the bindings make a `WellSpec` of them; the port converted on every `simulate()` call. The parameters are the dataclasses' own fields, and the core derives the rest with the same arithmetic, each under its tag: the grid's cell lengths, inclinations and depth fractions (GEO-3), the fluid's derived constants (PVT-GAS-5, PVT-GAS-6, PVT-MIX-10, PVT-OIL-5) and the critical pressure ratio (CHK-4). Interpolating a survey onto the grid (GEO-4) stays in Python. Each option is an enum variant named after its spec option, dispatched by `match`. A component whose class is not exactly one of the core's (decision 1), or slip constants outside the void-fraction bracket ($C_0 \ge 1$, $v_{\infty,\text{annular}} \ge 0$), is refused with `ValueError`.
2. **Rows defined once.** `discretization.rs` defines each point's rows in DISC-6 form, as functions of $(x_{i-1}, x_i, w_\text{res})$ and the operating point. The march solves them, and a binding evaluates them, so `tests/test_spec_vectors.py` checks the core against v1.0.0's row vectors, and, through a test-only binding that calls each component function by name, against the component vectors, as it checks the CasADi backend. The tests also compare the two backends' rows at the same states, in every configuration of a matrix that switches each option on and off (`tests/backend_cases.py`). That catches an assembly error on a new code path, the gap that "New model versions" in the plan leaves to review, unless both backends make the same error, and it is more sensitive than comparing operating points.
3. **Shooting.** Fix $p_0$; the inflow gives $w_\text{res}$; march to the wellhead; $R(p_0)$ is the CHK-1 row there, in any form with the sign of $w_m - w_c$ everywhere (CHK-11), such as the port's squared row; Step 8 uses the canonical row. The roots are the sign changes of $R$ on a scan, refined by Brent. Because the march zeroes every other row, $R$ is SOL-3's shooting residual, and the sign of $dR/dp_0$ at a root is the stability label. A sample where $R$ is not finite, where a march leaves the closures' range, is left out of the scan.
4. **Temperature solve per cell.** The port's closed-form $T(z)$ holds only for v1.0.0's energy balance. `develop`'s frictional-heating and gravity terms depend on the state, and so on $p_i$, and with dissolved gas the heat flux capacity does too. The cell solve therefore finds $(p_i, T_i)$: at each trial $p_i$ it solves the energy row for $T_i$, then evaluates the momentum row, and Brent on $p_i$ zeroes that. At $(p_i, T_i)$ the closures give the other five unknowns: $\rho_g$ and $\rho_l$ from PVT, the phase rates, $\alpha$ from the slip law, and the velocities from the rates. Where the energy row is linear in $T_i$ and does not depend on $p_i$ (heat loss alone, without mass transfer, as in `v1.0.0`), its solve is one step, which is v1's recursion (19) that Step 8 adopts. Otherwise a chord iteration, Newton's method with the heat loss's slope, starts from the temperature at the cell's previous trial pressure, and a bracketed Brent takes over where it does not converge. The bracket's lower end is proven from the sign and a bound of each cooling term, and steps out where the Joule–Thomson term, which has no simple bound, cools past it; its upper end steps out where frictional heating or Joule–Thomson heating dominates (`specs/features/015-rust-develop-model.md`, `016-joule-thomson.md`). With the Dranchuk–Abou-Kassem gas law, $\rho_g$ at $(p_i, T_i)$ is a Newton solve of PVT-GAS-11. In a gas well the Joule–Thomson term can leave the energy row without a root at trial pressures far below the cell's, so not every trial state can be computed, and near a choked wellhead it can give the row two roots in $T$: the cell solve then descends from $p_{i-1}$ to the first sign change, the scan of $R(p_0)$ refines the edges of the region where $R$ is finite, and the temperature solve takes the root on the rising side of the row's minimum (`016-joule-thomson.md`, design choice 5).
5. **Phase rates along the well.** The port holds $w_g$ and $w_l$ fixed for the whole well. With dissolved gas they depend on $(p, T)$, so each point evaluates `phase_rates` at $(p_i, T_i)$; with dead oil it returns the port's constants.
6. **Outputs.** Every root as a full state in state-vector order (bar, K), with its slope, CHOKED flag, the flow regime at each point and the reservoir phase rates. The Python layer builds the `RootSet` and selects the operating point, so no model equation is needed in Python.
7. **Batches.** `root_sets(well, operating_points)` solves many operating points of one well in parallel, with the GIL released. A case's result does not depend on the thread count, which keeps runs deterministic (constitution, Determinism). This is the library-level batch API of `plans/improvements.md` §4.3.
8. **Packaging.** A maturin mixed project: `pyproject.toml` builds `manywells._core` from `rust/`, and the Python package stays in `src/manywells/`, so installing from source needs a Rust toolchain. pyo3 is an optional dependency behind the crate's `python` feature, so `cargo test --no-default-features` builds the core without Python. The verifier stays a pure-Python workspace member. Prebuilt wheels are release work.
9. **Methods and constants** (scan density, tolerances, the $\alpha$ and $T$ solves, root acceptance) are decided in the Rust feature spec under principle 7, starting from `plans/solver_improvements.md`: `specs/features/014-rust-solver.md` gives the `v1.0.0` configuration's, with their measured gains, which Bjarne signed off on 2026-10-01, and `specs/features/015-rust-develop-model.md` those Step 9 added for the rest of the model.

The two backends must agree on the operating point for the same cases, and the verifier checks each in the `v1.0.0` configuration. With `develop`'s options on, the tests compare them on the comparison set (`tests/backend_cases.py`): the core finds every CasADi root with its label, except where a root lies on another branch of a point's rows.

## Feature map

Every planned v2 feature, from `specs/goals.md` and the plan's "After this plan", and its one module:

| Feature | Module |
|---|---|
| Black-oil PVT | `pvt/` (`black_oil.py`) |
| Gas dissolving into oil | `pvt/` (`fluid.py`, `phase_rates`) |
| Real-gas z-factor | `pvt/` (`gas.py`) |
| Surface tension from the fluid model, live-oil correction | `pvt/` (`fluid.py`) |
| Deviated and L-shaped wells | `geometry` |
| Inclination in the slip model | `slip` |
| Four-regime flow-regime model | `slip` |
| Friction from roughness (Chen, Haaland); the Step 10 correlation | `friction` |
| Frictional heating, gravity term in the energy balance | `thermal` |
| Lift-gas temperature | `thermal` |
| Fixed-rate inflow | `inflow` |
| v1-compatibility configuration | `configurations` |
| Root set with stability labels, and the operating point, as `simulate` returns them | `solution` |
| Finding the roots: multi-start search and the label computation | `solvers/` (in the core, `shoot.rs`) |
| Building each well's system once, operating point as parameters | `discretization` |
| Ported v1 sampler; the sampling redesign | `sampling/` |
| v2 datasets: schema, features, stability column, writers | `datasets/` |
| Batch API | `simulator` (the core's `root_sets`) |
| Rust core with Python bindings | `rust/` (the bindings in `lib.rs`) |
| Calibration | `calibration/` |
| Closed loop | out of v2 |

Prebuilt wheels are packaging (`pyproject.toml` and a release workflow), not a module.

## Backlog items this file settles

`plans/improvements.md` tags four items to Step 6. Step 7 implements them.

- **§2.3 choke dispatch:** polymorphic `ChokeModel.mass_flow_rate(u, p_s, s, A)`.
- **§2.4 hidden `_w_l_inflow`:** $w_\text{res}$ is an explicit argument of every point's rows.
- **§2.5 mutating the caller's objects:** frozen inputs; the default choke is set in `WellProperties.__post_init__`; the operating point is passed as parameters, never written into an input.
- **§4.1 building the NLP once:** `build_system` and `IpoptSolver` once per well, with the operating point as parameters, and one parametric rootfinder for the march. This removes work per call and adds no path. Measured on `develop` at 100 cells (two wells, a scratch script, 2026-09-30; the default black-oil well and a dead-oil, ideal-gas, fixed-$f_D$ well with a Simpson choke and Vogel inflow):

  | | Default well | Dead-oil well |
  |---|--:|--:|
  | Warm re-solve (the generator's reuse of the u = 0.5 root), total | 717 ms | 304 ms |
  | of which assembly and `nlpsol` creation | 97% | 97% |
  | of which Ipopt | 15 ms | 7.7 ms |
  | Cold solve, total | 1234 ms | 533 ms |
  | of which assembly and `nlpsol` creation | 64% | 56% |
  | of which the march | 34% | 35% |

  In a separate run, the march's Newton solves took 8 ms of its 452 ms in the default well; building the rows and 100 rootfinders took the rest. Once a well's system is built, the speed-up is bounded at 48x and 39x for warm re-solves, and at about 40x for a cold solve of the default well (1234 ms against 29 ms of Newton and Ipopt). These are single runs, and Ipopt's cold time varied from 21 to 98 ms between runs. The build, under 1 s per well at 100 cells, is paid once. It also makes the multi-start search affordable: six starts then cost six marches and Ipopt solves, not six builds. Step 7 repeats the measurement on the verifier's case set before the change merges (principle 7).

## Decisions

Decided by Bjarne, 2026-09-30, each as recommended:

1. **Extension by contributors, not by runtime plug-ins.** A new option is a spec change and code in one module of each backend; the Rust core has a closed set of options per module. A user's Python subclass of a component ABC runs only on the CasADi backend. Python callbacks from the core were rejected: they would cost the core's speed and add a second code path.
2. **`closed_loop/` gets a frozen private base.** It subclasses `SSDFSimulator` and overrides `_compute_left_boundary_state`, `_initial_guess` and `simulate`, which Step 7 restructures, and no test covers it. Before restructuring, Step 7 copies today's `SSDFSimulator` verbatim into `closed_loop/` as its private base, so its behaviour does not change until `closed_loop/` is retired. Letting it break, and removing it in Step 7, were rejected. Bjarne retired `closed_loop/`, its frozen base and its generators on 2026-10-01 (`specs/goals.md`).
3. **API breaks**, all four accepted, each to be listed in the v2 CHANGELOG with an old→new snippet (`specs/goals.md`): `SSDFSimulator(wp)` with `simulate(bc)` returning a `Root`, and the two-argument constructor deprecated until v2.0.0; `friction` and `thermal` objects in `WellProperties` in place of `f_D`, `roughness` and `h`; frozen inputs; `p_sep` and `p_bubble` in bar.
4. **The flux-difference mass-row form** (Discretization, above), one form for every fluid. Two forms, v1's flux continuity for `v1.0.0` and `develop`'s local-rate rows for mass transfer, were rejected. It is a change to `specs/model/`, so Step 7's spec text with its new IDs comes to Bjarne for sign-off.
5. **The root search uses the verifier's thresholds:** roots within `tol_x` in the verifier's state distance are merged, and a label is indeterminate at a normalized slope of at most `label_min`, so the library and the verifier agree on what counts as a root and a label. Step 7 proposes the starts and any early exit from measurements.
