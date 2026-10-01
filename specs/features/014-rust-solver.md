# 014 · The Rust core in the v1.0.0 configuration

*Feature spec for Step 8 of `plans/manywells-v2-plan.md` (2026-10-01), from the port on `rust_implementation` @ `0e9e98b` (Kajrakso, PR #6), and `plans/solver_improvements.md` items 1 to 6. Status: Bjarne signed off on 2026-10-01: the full scan, rather than the early stop; the bracketed void fraction, the drawdown ladder, the jump guard, and the tolerances and slope; no performance target yet; and the Step 8 edits to `specs/architecture.md` and `specs/verification.md`. His rulings on the findings are given under each.*

## Motivation

`specs/goals.md` makes a Rust core with Python bindings the v2 implementation, and the plan's Step 9 adds an equation to it. The port solved v1.0.0's model by shooting on the bottomhole pressure, about 100 times faster than v1.0.0, but no candidate check said it was right. On the verifier's case set its roots failed: none was labelled, its stable-root rate was 0%, and only 5 of the 200 reference roots were within `tol_x` (median distance 1.8e-3). Its temperatures came from the exact solution of the energy ODE instead of v1.0.0's rows, its void fractions stopped on the iteration's step size, its outer tolerance was 1e-6 bar on $p_0$, and its scan assumed at most two roots.

## Delta

No change to `specs/model/`. The core implements the rows of the `v1.0.0` configuration as specified: the energy rows DISC-5 with THM-1 (no exact solution), the slip law SLIP-1 without safeguards (`slip.md`, Safeguards), and the choke row CHK-1 in its canonical form with CHK-11's no flow where $\Delta p \le 0$ (the port squared it, which CHK-11 also allows).

- **`rust/`**, built as `manywells._core` by maturin from the root `pyproject.toml` (`specs/architecture.md`, Rust core, point 8). Crate `manywells-core`, one module per spec file (`slip.rs`, `choke.rs`, `inflow.rs`, `friction.rs`, `thermal.rs`, `geometry.rs`, `pvt/{gas,oil,mixture,fluid}.rs`, `smoothing.rs`, `units.rs`) with `// spec:` tags; `discretization.rs` defines each point's rows once, in DISC-6 order and v1.0.0's units; `march.rs`, `shoot.rs` and `scalar.rs` solve them. The bindings sit behind the crate's `python` feature, so `cargo test --no-default-features` needs no Python.
- **`src/manywells/solvers/rust.py`**: `RustRootFinder(wp)` converts a well in the `v1.0.0` configuration to the core's inputs once (`configurations.differences` must be empty, else `ValueError`), and builds the `RootSet` from the core's roots with the CasADi backend's `admissible` (SOL-1), `normalized_slope` and `label_of` (SOL-3) and `select_operating_point` (SOL-4 to SOL-6).
- **`SSDFSimulator(wp, backend='casadi' | 'rust')`**; with `'rust'` no CasADi system is built. `manywells.sampling.generate.Settings` has a `backend` field, and `develop_candidate.py` and `regenerate_distributions.py` take `--backend`.

## The method

Given $p_0$, the inflow gives the phase rates, which are the same at every point (BAL-3). The march then solves the rows point by point up the well:

1. **Temperature.** Each cell's energy row is linear in $T_i$, because the mass rows fix its heat flux capacity, and does not depend on pressure: $T_i = (T_{i-1} + \Delta z\,k\,T_{a,i})/(1 + \Delta z\,k)$ with $k = 4h/(D(c_{p,g}G_g + c_{p,l}G_l))$, once per march.
2. **Closures at a trial pressure.** The gas law gives $\rho_g$, and $\rho_l$ is constant. The void fraction comes from Brent on $h(\alpha) = \alpha(C_0 j_m + v_\infty) - j_g$, which is $-\alpha$ times SLIP-1. The classifier sees $\alpha$ only through $c_2$ and $c_4$, and $C_0 \ge 1$ and $v_\infty \ge 0$ for every mix of regimes. So $h(0) = -j_g < 0$ and $h(1) \ge j_l + v_\infty > 0$, and [0, 1] always brackets a root.
3. **Pressure.** The cell's momentum row is U-shaped in $p_i$, with its minimum at the sonic pressure $p^*$. Brent runs on $[p_s, p_{i-1}]$ if the row is negative at $p_s$. Otherwise a golden-section search finds $p^*$: below zero, Brent runs on $[p^*, p_{i-1}]$; at $p_s$, the root lies below the separator; above zero elsewhere, the cell is choked. A cell counts as solved only if $|row| \le 10^{-8}$ bar at Brent's answer.
4. **Residual.** Once the pressure would fall below $p_s$ it stays there, so the wellhead is at or below the critical pressure and $R = w_m$ exactly (CHK-11). Otherwise $R$ is the CHK-1 row at the wellhead.

The search scans $R$ at 101 points on $[p_s + 10^{-3}, p_r - 10^{-6}]$. It adds a ladder of samples in the top interval whose drawdown halves from half a step down to $10^{-6}$ bar, and runs Brent on every sign change. At every interior sample where $R \ge 0$ is a local minimum, a golden-section search over its neighbours, down to $10^{-6}(p_r - p_s)$, looks for $R < 0$ and then brackets both sides of it. Brent runs in the drawdown $d = p_r - p_0$ with a relative tolerance and $x_{tol} = 4\varepsilon p_r$. A root is accepted if its march did not fail and $|R| \le 10^{-3} w_m$. Its slope $dR/dp_0$ is a central difference with a step of $10^{-4}\min(p_r - p_0, p_0 - p_s)$.

## Solver machinery and its measured gain (principle 7; signed off 2026-10-01)

Measured on the verifier's 141 cases, 2026-10-01, one core, search times per case (the core needs no build). Each row adds one change to the row above; the commits on `step8-rust-core` follow this order.

| Change | Reference roots within `tol_x` | Stable-root rate | Search (median) | Marches (median) | Search, all cases |
|---|--:|--:|--:|--:|--:|
| The port, restructured (same roots to 5e-13 bar) | 5 / 200 | 0 / 136 | 5.3 ms | 22 | 1.03 s |
| v1.0.0's energy rows | 55 | 21 | 5.0 ms | 21 | 1.08 s |
| Bracketed void fraction | 194 | 134 | 9.8 ms | 22 | 2.21 s |
| Canonical choke row, stop below $p_s$, cell bracket $[p_s, p_{i-1}]$ | 187 | 134 | 9.5 ms | 19 | 1.51 s |
| Scan every bracket, refine local minima | 191 | 136 | 45.6 ms | 133 | 8.99 s |
| Brent in the drawdown, cell Brent to a few ulp, $\lvert R\rvert$ acceptance | **200** | **136** | 53.0 ms | 133 | 9.67 s |
| Slope by central difference | 200 | 136 | 55.0 ms | 135 | 9.86 s |
| Without the port's narrow first bracket in the cell | 200 | 136 | 57.7 ms | 136 | 9.84 s |
| Drawdown ladder in the top interval | 200 | 136 | 65.2 ms | 154 | 11.02 s |
| **Jump guard in the cell (the code as it is)** | **200** | **136** | **65.4 ms** | **154** | **11.03 s** |

With the last row the verifier reports PASS with no expected failures: every reference root is matched (median distance 7e-14, largest 5.3e-8; `develop`'s CasADi search reaches 2.3e-7) with its label, and the stable-root rate is 100%. v1.0.0 from its default guess: 74.3%. `develop`'s CasADi search, timed on the same machine: 1.49 s per case (build and search, median), so the core is 23× faster at the median and 31× in total (at least 8× in every case).

- **Bracketed void fraction.** This is the change that matters most for accuracy: 55 → 194 roots within `tol_x`. It doubles the search time, but has no tuning constant and needs no fallback. Not tried: Aitken acceleration of the port's iteration (`solver_improvements.md` item 2, option 3), which would need the bracketed solve as its fallback.
- **The full scan with refinement.** Without the refinement, `fold-1528` and `fold-0422` lose both roots: with accurate void fractions, their negative region is narrower than one scan step. The full scan is 4.7× the cost of the port's early stop. The alternative is the port's early stop with the refinement and the ladder: on the same code it passes the verifier the same way (200 / 200, 100%), at **18.1 ms** per case (median; 2.56 s in total, 3.6× faster). But it assumes $R$ has at most one negative region, which the case set cannot test: its cases with more than two roots (`fold-1505` and `fold-0485`, two stable roots each) were left out as disagreements. Bjarne chose the full scan (2026-10-01).
- **Coarser scans fail**, even with the refinement and before the ladder: 50 intervals 135 / 136, 25 intervals 130, 12 intervals 119.
- **The drawdown ladder** (about 20 more marches per case, +12%) changes nothing on the case set, whose lowest $u$ is 0.133. In the regenerated `sol-1` samples, at $u$ = 0.054 to 0.086, both roots of IDs 23, 48, 61, 65 and 67 lie 0.15 to 2.2 bar below $p_r$, inside the top scan interval; without the ladder the core missed all of them, and the CasADi backend found them.
- **The jump guard** costs nothing measurable and changes nothing on the case set. In the regenerated samples it stops the core from returning a non-root at ID 44, k = 4, where the slip law has three void fractions and the march's closure switched between them: the momentum rows of points 96 to 100 were 0.3 bar (Findings, 2).
- **Tolerances.** The cell's Brent has $x_{tol} = 0$ and $r_{tol} = 4\varepsilon$. The outer Brent has $x_{tol} = 4\varepsilon p_r$ on the drawdown. The acceptance bound $|R| \le 10^{-3} w_m$ sits well above the extreme trickle roots, where one ulp of $p_0$ moves the choke rate by about $5\cdot10^{-5}$ of $w_m$ (`sol1-1214-d0`), and far below a jump. The local-minimum search narrows to $10^{-6}(p_r - p_s)$, 1/100 of `tol_x` in $p_0$. The slope step is $10^{-4}$ of the distance to $p_r$ or $p_s$.
- **The slope.** It costs two marches per root and gives the CasADi backend's label rule, including `indeterminate`. It matches the reference's Jacobian slope at a median ratio of 1.000000. At a few extreme trickle roots the step crosses CHK-11's kink, so the size is off (ratio 0.036 to 1.36) but not the sign. The smallest |slope| in the case set is 8.6, against `label_min` = 1e-3. Its sign matches the bracket's orientation at every root of the case set (`tests/test_rust_backend.py`).
- **Removed, no gain:** the port's narrow first bracket in the cell solve (9.80 s without it, 9.86 s with it), and an early stop of the cell's golden-section search (9.84 s). Also removed: the port's void-fraction clamps, its +1e-6 m/s slip guard and its $\rho_g$ floor, and its march below $p_s$ from a floor of $10^{-3}$ bar.

**Performance target:** none yet (Bjarne, 2026-10-01). The measurements above are the record, and a target is set when `develop`'s model is ported to the core. For comparison, the code as it is is 23× faster than `develop`'s CasADi backend (build and search, per case, median); with the early-stop scan it would be about 80×.

## Off in the `v1.0.0` configuration

Not a model option. The core covers the `v1.0.0` configuration only, and refuses any other well. `develop`'s model in Rust comes after the plan (`specs/architecture.md`).

## Acceptance

- The verifier on the core's candidate (`develop_candidate.py --backend rust`, CI job `verify`) reports PASS with no expected failures and a 100% stable-root rate.
- The core's rows match v1.0.0's row vectors to 1e-10 (`tests/test_spec_vectors.py`), and the traceability test passes.
- `tests/test_rust_backend.py`:
  - well 977 gives v1.0.0's 169.0004 bar (stable) and 208.0218 bar (unstable); the port gave 169.13 bar;
  - the port's three test wells keep their roots;
  - every row of the CasADi system is zero at the core's roots;
  - slow: the two backends find the same roots, labels, CHOKED flags and regimes;
  - slow: the slope's sign matches the bracket's orientation on the case set;
  - slow: the slip law has one void fraction at every point of every reference root.
- `cargo test --no-default-features`: the closures, the rows, a march that zeroes the rows it solves, and every root zeroing every row.

## Findings, with Bjarne's rulings

1. **`fold-1503`'s reference misses a root.** The core finds a second, unstable root at $p_0$ = 128.0882 bar, which no reference root matches (the verifier lists it as a finding). v1.0.0 solved from it reports `Solve_Succeeded` in 5 iterations and stays (relative change 1.7e-13), with slope +458. So it is a v1.0.0 root that neither reference search found. The old port put it at 127.83 bar, from where v1.0.0 converged elsewhere. **Ruling (2026-10-01):** the reference stays unchanged, and the root is a known finding (`specs/verification.md`). Adding it would make `develop`'s CasADi candidate fail Root set there, because its search does not find it.
2. **The slip law can have several void fractions.** At sampled states with a nearly closed choke and gas lift, SLIP-1 has three roots in $\alpha$. At ID 44, k = 4, point 95, they are 0.662, 0.915 and 0.920. The root set then contains roots that differ in the branch at some points: at IDs 91 and 95 (k = 2) the CasADi backend and the core find different stable roots (248.590 and 248.386 bar; 273.642 and 273.347 bar), and each is a root of the full system. Neither `slip.md` nor `solution.md` says which branch is physical. SOL-6's lowest stable root then depends on finding every branch, which neither backend does. On the case set $\alpha$ is unique at every reference point (tested). **Ruling (2026-10-01):** an open model question, noted in `slip.md` and `solution.md`, to be ruled after the plan with the new flow-regime model of `specs/goals.md`.
3. **The CasADi backend misses roots that the core finds.** In 3 of 1,000 regenerated samples on the first 200 wells (IDs 16, 70 and 130), only the core has an operating point, and each is a genuine root: every CasADi row is about 1e-13 there, and Ipopt started there stays. At ID 67, k = 0, the core also finds an unstable root the CasADi backend misses. This is a gap in `develop`'s multi-start search (012), not changed here; the CasADi backend is retired once the core covers the whole model.
4. **The case set misses three regimes:**
   - wells whose roots both lie within one scan step of $p_r$ (low $u$; finding 2's ladder);
   - states with several void fractions;
   - wells with more than two roots, which were left out as disagreements, so an early-stop scan is untested.

   Its near-fold cases were placed with the old port's scan grid (`verification/build/rust_roots.py`, `fold`), which biases them towards that grid. Method B, the old port, misses the first kind too, so under the build's rule such cases would be left out as disagreements. **Ruling (2026-10-01):** after the plan; the case set stays as it is through Step 9.
5. **Distributions.** Regenerated with the core at 5 samples per published `sol-1` well: 9,920 rows in 49 s on 24 cores. The check fails narrowly: CDF gap 0.021 for PWH (bound 0.02), rank-correlation gap 0.035 (bound 0.05). `develop`'s CasADi regeneration (9,879 rows) passed at 0.0196. On the first 200 wells, the two backends' rows agree to 1e-6 in 986 of 988 common samples. The other two are finding 2's, and the core solves 3 samples the CasADi search does not (finding 3). So the core's extra rows are mostly samples that the CasADi search, and likely v1.0.0's generator, did not solve, and that the published reference does not have. **Ruling (2026-10-01):** a known result, decided together with Step 7's open item on the check's margin.

## Out of scope

`develop`'s model in Rust (temperature solve per cell, phase rates along the well), the batch API (`root_sets`), prebuilt wheels, and the branch's research scripts, dataset generators and notebook, which use v1's API and pick roots by position. The Rust datasets on the branch were local and never published. The regeneration above replaces them as the cross-check of the reference's shape.
