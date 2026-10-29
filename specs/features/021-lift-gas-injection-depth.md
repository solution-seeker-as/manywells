# 021 · Lift-gas injection depth

*Feature spec, 2026-10-29. Status: approved by Bjarne on 2026-10-29, with each design choice as proposed (Approval, at the end); not yet implemented. Step L1 of `plans/model-lock-plan.md`: Bjarne put the injection depth in v2.0's model on 2026-10-29 (`specs/goals.md`).*

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
