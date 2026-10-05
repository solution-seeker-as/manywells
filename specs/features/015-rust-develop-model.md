# 015 · `develop`'s model in the Rust core

*Feature spec for Step 9 of `plans/manywells-v2-plan.md` (2026-10-02), on the branch `step9-rust-develop-model`, one commit per feature spec. Status: Bjarne signed off on 2026-10-02: the three solver changes under principle 7, the comparison's tolerances and the temperature solve's constants, no performance target yet, and the Step 9 edits to `specs/architecture.md`. His rulings on the findings are given under each.*

## Motivation

`specs/goals.md` keeps two backends, developed together, each equation in both. After Step 8 the Rust core covered the `v1.0.0` configuration only, so `develop`'s model had no second implementation: its changes since v1.0.0 were checked by component vectors, invariants, spot checks, convergence and property checks, which do not catch a self-consistent assembly error. Comparing two backends' rows at the same states does. Step 10's equation must also go into both, which needs the core to have the rest of the model.

## Delta

No change to `specs/model/`. The core implements every option that a row of the system uses, under the same equation IDs as the CasADi backend, with `// spec:` tags:

| Feature | Equations in the core | Commit |
|---|---|---|
| 005 fluid model | the fluid's own fields; PVT-GAS-6, PVT-MIX-10 derived in the core | `d60ddac` |
| 001 survey grid | GEO-3 (from Python's MD and TVD), DISC-9, DISC-10, THM-4 | `4e083b5` |
| 009 energy terms | THM-6, THM-7, and the temperature solve per cell (below) | `ecb5114`, `1e67328`, `522259f` |
| 002 slip inclination | SLIP-10, SLIP-11, the slip constants | `b78f7fb` |
| 006 real gas | PVT-GAS-3, PVT-GAS-4, PVT-GAS-5 | `8befc20` |
| 007, 008 black oil, dissolved gas | PVT-OIL-4 to PVT-OIL-9, PVT-OIL-13, PVT-MIX-6, SMO-2; phase rates at each point | `572da32` |
| 003 surface tension | PVT-MIX-7, PVT-OIL-12, SMO-4 | `ae71439` |
| 004 friction | FRIC-3 to FRIC-6, PVT-GAS-7, PVT-OIL-10, PVT-OIL-11, PVT-WAT-3, PVT-MIX-8, PVT-MIX-9 | `f0ac7f2` |
| 010 lift-gas temperature | THM-5 | `45ab4ab` |
| 011 fixed liquid rate | INF-8 | `fba173a` |

The plan had a commit each for 007 and 008. Python's black oil always has dissolved gas, so neither can be compared on its own, and they went in together. Before the features, a first commit (`b3a0c1a`) built the harness below and changed the core with no change to any root:
- DISC-11's row IDs, and the mass rows' rate-difference form (DISC-7, DISC-8);
- phase rates at each point's $(p, T)$, and the temperature computed cell by cell;
- $T_\text{lg}$ in the operating point, which is now the CasADi system's parameter vector;
- CHK-4 computed in the core.

**Not in the core,** because no row uses them; they stay in Python:
- PVT-GAS-2 and PVT-GAS-8, conversions between inputs and the gas's formation volume factor;
- PVT-MIX-2 to PVT-MIX-4, the sampler's liquid mixing (SMP-12);
- PVT-OIL-14, the bubble point from Standing's correlation, which the fluid takes as an input instead;
- PVT-WAT-2, water's formation volume factor (water is incompressible in every configuration).

GEO-4, the interpolation of a survey onto the grid, also stays in Python. The core takes the grid points.

**Python.** `solvers/rust.py` passes the dataclasses' own fields. `SSDFSimulator(wp, backend='rust')` accepts every well whose component classes are exactly the core's (`CORE_CLASSES`). A user's subclass of a component ABC needs the CasADi backend (`specs/architecture.md`, decision 1). Slip constants outside the void-fraction bracket ($C_0 < 1$ or $v_{\infty,\text{annular}} < 0$) are refused too.

## The method

The march of 014 with three changes:

1. **Phase rates at each point.** `phase_rates(p_i, T_i, w_res, w_lg)` gives each point's rates, and the velocities carry them, so the mass rows hold by construction with dissolved gas too.
2. **Temperature per cell.** Where the energy row is linear in $T_i$ and does not depend on $p_i$ (heat loss alone, without mass transfer, as in `v1.0.0`), $T_i$ is v1's one closed-form step, as in 014. Otherwise $T_i$ is solved at each trial pressure of the cell solve:
   - **Chord iteration.** It starts from the temperature at the cell's previous trial pressure, with steps $T \leftarrow T - r_T / (1 + \Delta\text{MD}\thinspace 4h/(D c_p\text{-flux}))$, Newton's method with the heat loss's slope. It stops when the step is a few ulp, at most 10 steps.
   - **Bracketed Brent** takes over if a step does not shrink. Its lower end, $\min(T_{i-1}, T_a) - \Delta\text{MD}\thinspace g\cos\theta/\min(c_{pg}, c_{pl})$, is proven: $r_T \le 0$ there for every state, because the heat loss has the sign of $T - T_a$, frictional heating is non-negative, and the gravity term lies in $[0, g\cos\theta/\min(c_p)]$. Its upper end, $\max(T_{i-1}, T_a)$, has $r_T \ge 0$ without frictional heating; where frictional heating keeps $r_T$ negative there, it steps out by doubling steps (at most 30).
   - **Solved or not.** A temperature counts as solved if $|r_T|$ over the heat-loss slope is at most $10^{-8}$ K. At a jump in the closures (the slip law switching void fraction), the march continues at the jump and fails, as at a cell whose momentum row is not solved.
3. **Inclination.** At each point, the slip law takes the cell's inclination (point 0 takes cell 1's), as the CasADi rows do.

**Changes to the scan:**
- A sample where $R$ is not finite, where a march leaves the closures' range, is left out instead of aborting the search.
- A refined local minimum must be strict on one side.

## Solver machinery and its measured gain (principle 7; signed off 2026-10-02)

Each row adds one change to the row above; one core per case.

**The temperature solve.** Verifier's 141 cases with frictional heating and the gravity term on (overlay `energy terms`):

| Change | Cases completed | Search (median) | Search, all cases | States per T solve |
|---|--:|--:|--:|--:|
| Bracketed Brent at each trial pressure (`ecb5114`) | 107 of 141 | 0.368 s | 57.3 s | 8.5 |
| Leave non-finite samples out of the scan (`1e67328`) | **141** | 0.468 s | 89.2 s | 8.8 |
| Chord iteration first, Brent as fallback (`522259f`) | 141 | **0.163 s** | **36.8 s** | **3.5** |

- **The 34 aborted cases** are marches near $p_s$ through choked cells. There, frictional heating at 500 m/s heats the flow past the range of the dead-oil surface tension correlation, which is negative above about 639 K. Their roots are unaffected; 1,111 samples are left out in those cases.
- **The chord iteration** is 2.9× faster at the median and 2.4× in total. Brent takes over in 0.23% of the solves (51,035 of 22.3 million), and the comparison with the CasADi backend is unchanged.

**The strict minimum.** 40 fixed-rate wells of the comparison set:

| Change | Marches (median) | Marches (largest) | Search, all cases |
|---|--:|--:|--:|
| Every local minimum with neighbours ≥ R refined | 682 | 2,688 | 31.7 s |
| Strict on one side (`fba173a`) | 250 | 963 | 19.0 s |

With a fixed rate, every march that falls below $p_s$ gives the same $R$, the rate itself, so $R$ is flat there and every flat sample was refined (90 refinements at `v1.0.0+fixed rate#11`). With an inflow model, $R$ below $p_s$ varies with $p_0$, so the `v1.0.0` configuration is unchanged.

**The `v1.0.0` configuration.**
- The verifier's 141 cases take 0.066 s per case (median; 11.1 s in total, 154 marches), as in 014.
- The Rust candidate reports PASS with no expected failures and a stable-root rate of 100%.
- Its 201 roots are identical to Step 8's except for commit `4e083b5`, where the grid and the ambient profile became Python's. That moved them by rounding only: largest scaled distance $4.5 \cdot 10^{-13}$.

**Performance against the CasADi backend.**
- Comparison set, 20 wells per configuration (520 cases, 24 processes, build and search per case): core 0.205 s at the median, CasADi 4.08 s, 19.9× at the median and 57.7× in total.
- One process, 2 wells per configuration (52 cases): core 0.154 s at the median, CasADi 1.40 s, 9.1× at the median and 26.9× in total (at least 1.3× in every case). On the v1.0.0 base, 0.081 s against 0.83 s (10.2× and 14.6×); on the develop base, 0.248 s against 4.5 s (18.2× and 31.1×).

**Performance target:** none yet (Bjarne, 2026-10-02). The measurements above are the record.

## The comparison with the CasADi backend

`develop`'s model has no reference root sets, so the core is checked against the CasADi backend. The checks, strongest first:

- **Rows at the same state** (`tests/test_backend_comparison.py`, fast). On a matrix of 55 configurations (the row-vector wells W1 and W2, and `develop`'s default well, each with one option switched on or off by an overlay of `tests/backend_cases.py`, and two combinations), the core's rows equal the CasADi system's, with the same IDs in the same order. They agree to $10^{-10}$ of each row's largest value along the well, at a trial state across the three flow regimes. The core's march from a bottomhole pressure zeroes every CasADi row except the choke row.
- **Component vectors** (`tests/test_spec_vectors.py`). The core passes every vector table of v1.0.0 and `develop` through a test-only binding (`Well._component`), except the tables of the equations not in the core.
- **Root sets** (`tests/test_backend_comparison.py`, slow, 2 wells per configuration; `scripts/verification/compare_backends.py`, 20). Wells are drawn with the ported sampler for 26 configurations: `v1.0.0` and `develop` (SMP-40 to SMP-44), each with one overlay. The overlays add what the sampler does not draw: lift-gas temperature, a fixed rate, PI inflow, a Bernoulli choke, and `develop`'s options switched off one at a time. The pass rule:
  - every CasADi root is matched by a core root within `tol_x`, with the same label;
  - every root only the core finds zeroes every CasADi row: CHK-1 to the larger of $10^{-6} w_m$ and 8 ulp of $p_r$ times $|dR/dp_0|$, the others to $10^{-8}$;
  - the rows agree at the perturbed roots;
  - a case where the slip law has several void fractions at a root's point, or where a root lies on another branch of a point's rows, is compared on rows only. A root is on another branch where the sign of $\det(\partial\thinspace\text{rows of point } i / \partial x_i)$ differs from the rest of the well's; the determinant changes sign only through a fold.
- **Property checks** (`tests/test_model_properties.py`): every check runs on both backends, and the two find the same roots.

**Result**, 520 cases: all pass.
- **Roots.** The CasADi backend finds 630, the core 620. All 31 the core misses are in cases with several void fractions or on another branch.
- **Core-only roots.** The core finds 21 roots the CasADi search misses, and 20 of them zero every CasADi row (`plans/improvements.md` §2.9).
- **Labels and rows.** No label differs, with a fixed rate either. Rows agree to $8.8 \cdot 10^{-12}$.

Per configuration (CasADi roots / core roots / missed / core-only):

| | v1.0.0 base | develop base |
|---|---|---|
| none | 29 / 28 / 1 / 0 | 25 / 22 / 3 / 0 |
| deviated, L-shaped | 28 / 28 / 2 / 2; 25 / 28 / 1 / 4 | in the base (SMP-41) |
| real or ideal gas | 28 / 28 / 0 / 0 | 21 / 22 / 0 / 1 |
| black or dead oil; bubble point | 22 / 22 / 1 / 1; 22 / 22 / 1 / 1 | 27 / 28 / 0 / 1; 24 / 23 / 2 / 1 |
| oil or liquid surface tension | 29 / 28 / 1 / 0 | 21 / 22 / 0 / 1 |
| Chen; Haaland; fixed $f_D$ | 30 / 29 / 2 / 1; 30 / 29 / 2 / 1 | Chen in the base; 22 / 22 / 1 / 1; 24 / 22 / 3 / 1 |
| frictional heating; gravity term (on or off) | 29 / 28 / 1 / 0; 29 / 28 / 1 / 0 | 24 / 22 / 3 / 1; 23 / 22 / 2 / 1 |
| lift-gas temperature | 20 / 20 / 0 / 0 | 20 / 20 / 0 / 0 |
| fixed rate | 16 / 16 / 0 / 0 | 16 / 17 / 0 / 1 |
| productivity index; Bernoulli | in the base (W2) | 22 / 22 / 1 / 1; 24 / 22 / 3 / 1 |

## The method's assumptions, checked

- **The void-fraction bracket [0, 1]** holds for every inclination: $C_0 \ge 1$ and $v_\infty \ge 0$ with the deviation factor in [0, 1]. It is checked on $10^4$ random states with $\cos\theta \in [0, 1]$ (`tests/test_rust_backend.py`); the core refuses slip constants outside it.
- **The momentum row** with the temperature solved at each pressure is U-shaped on $[p_s, p_{i-1}]$ and positive at $p_{i-1}$. **The energy row**, where it depends on the pressure, has one root in $T$ at the root's pressure. Both are checked at every cell of every root of the 13 test wells (`rust/src/march.rs`, `the_cell_rows_have_the_shapes_the_cell_solve_assumes`). Together the test wells cover every option, including a horizontal section and a fixed rate.
- **Shooting with a fixed rate** (INF-8): $w_\text{res}$ does not depend on $p_0$, but $R(p_0)$ still does, through the march. On the 40 fixed-rate wells of the comparison set, the core and the CasADi backend find the same roots with the same labels, and the slope has the bracket's sign at every root of the fixed-rate test well.

## Acceptance

- `cargo test --no-default-features`: the closures, the rows, a march that zeroes every row but the choke row, every root zeroing every row, the slope's sign, and the assumptions above, on 13 test wells.
- `tests/test_spec_vectors.py`: the core against v1.0.0's row vectors to $10^{-10}$, and against every component vector table except those of the equations not in the core.
- `tests/test_backend_comparison.py`: rows and marches on the matrix (fast), and the root sets on the comparison set (slow).
- `tests/test_model_properties.py` on both backends, and `tests/test_rust_backend.py`.
- The verifier on the Rust candidate (`develop_candidate.py --backend rust`): PASS, no expected failures, 100%. On `develop`'s CasADi candidate, unchanged.
- The traceability test passes with the core's tags.

## Findings, with Bjarne's rulings

1. **Another branch of the momentum row.** At `v1.0.0+chen#6` and `v1.0.0+haaland#6`, the CasADi search finds a second stable root (318.69 bar). Its last cell drops from 106 to 10.3 bar, past the sonic point of the cell's U-shaped momentum row. Its rows are zero ($5 \cdot 10^{-9}$), so it is a root of the discretized system. The core's march takes each cell's subsonic root by design (014), so it does not find it. Neither `discretization.md` nor `solution.md` said which root of a cell is physical; a supersonic cell is not, in a steady-state model. **Ruling (2026-10-02):** only the subsonic root of a cell is physical. `solution.md` has a new SOL-8, Subsonic cells, and SOL-2's root set holds only states whose cells are all subsonic. The CasADi search now rejects a state with a supersonic cell (`System.subsonic`, from the slope of each cell's momentum row with the point's other rows held at zero); the core's cell solve takes the subsonic root by design. None of the verifier's 200 reference roots has a supersonic cell, and both candidates still report PASS, 100%. At `v1.0.0+chen#6` the CasADi search now finds only the root at 264.00 bar, as the core does (`tests/test_roots.py`). The comparison above was run before the ruling; there, the two supersonic roots count among the missed.
2. **Several void fractions.** In the cases where the slip law has several void fractions, the backends find different subsets of roots on different branches. At `develop#17`, CasADi finds four stable roots between 208.45 and 209.96 bar, and the core the lowest of them, which is the operating point by SOL-6. That is Step 8's finding 2, now on `develop`'s model in 1 to 2 of 20 wells per configuration. It stays open, with `slip.md`, to be ruled after the plan.
3. **The core accepts a jump in $R$ as a root** where the void fraction switches branch between neighbouring $p_0$ (`v1.0.0+deviated#17`, 210.36 bar). Its choke row is $-7.5 \cdot 10^{-4}$ kg/s, under the acceptance bound of $10^{-3}$ of the rate. The signed-off rule compares such cases on rows only, so it passes. A tighter acceptance, such as $|R|$ within a few ulp of $p_r$ times $|dR/dp_0|$, would reject it, but it is solver machinery and was not tried. **Ruling (2026-10-02):** a known finding, not changed; noted in `slip.md`, Open question, and revisited when that is ruled.
4. **What the label means with a fixed rate.** With INF-8 the inflow does not respond to $p_0$, so the nodal-analysis argument of SOL-3 (a small rise in rate makes the flow fall back) has no inflow side. Both backends compute the label the same way, from $dR/dp_0$, and agree. **Ruling (2026-10-02):** the label is kept, computed by SOL-3 as for any inflow; a note under SOL-3 says that with a fixed rate it comes from the tubing and the choke alone.
5. **The comparison's tolerances**, signed off (2026-10-02):
   - rows to $10^{-10}$ of each row's largest value along the well. Per row, a mass row that is small at one point where its terms cancel ($-0.0012$ against terms of 2.2, with dissolved gas) gave $9 \cdot 10^{-11}$ from a difference of $10^{-13}$;
   - core-only roots to $10^{-8}$, and the choke row to the larger of $10^{-6} w_m$ and 8 ulp of $p_r$ times $|dR/dp_0|$ (at the extreme trickle root of `v1.0.0+L-shaped#0`, 0.005 bar below $p_r$, the normalized slope is $5.5 \cdot 10^9$);
   - 2 wells per configuration in the test, 20 in the script; a 2001-point grid in $\alpha$ for several void fractions.
6. **Constants of the temperature solve**, signed off (2026-10-02): the guard of $10^{-8}$ K, at most 10 chord steps, at most 30 step-outs.

## Out of scope

The batch API (`root_sets`, `plans/improvements.md` §4.3), prebuilt wheels, the CasADi search's missed roots (§2.9), a ruling on either branch question, the four-regime flow-regime model, and making the fluid's fractions parameters of the CasADi system (§4.5), which the core makes unnecessary for dataset generation unless datasets are generated on the CasADi backend.
