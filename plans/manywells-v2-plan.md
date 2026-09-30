# ManyWells v2 — Spec-Driven Development Plan

*Draft 2026-09-29, revised 2026-09-30 (multiple roots and stability; work continues on `develop`). Owner: Bjarne Grimstad. Status: proposal.*

## Starting point

- **v1.0.0 is released** and is the reference implementation of the model equations: Python, CasADi expression graph, Ipopt feasibility NLP. It was validated against a real well and generated the three published datasets (`manywells-sol-1`, `manywells-nsol-1`, `manywells-nscl-1`). The paper (Geoenergy Science and Engineering 257, 2026) is the de-facto model specification.
- **`develop` is ahead of v1.0.0, and v2 work continues there.** As of 2026-09-30 it is 73 commits past the tag (`git log v1.0.0..develop`). It already has work on four of the five v1 limitations below, without specs or verifier coverage: deviated and L-shaped wells (`geometry.py`); friction from pipe roughness with the Chen and Haaland correlations, with a fixed `f_D` as an option (`friction.py`); black-oil PVT with gas dissolving into the oil, and a real-gas z-factor (`pvt/`); and an energy balance with frictional heating, a gravity term and lift-gas temperature. It also has a fixed-rate inflow, an initial-guess march using `ca.rootfinder`, the `src/` layout, a test suite, a CI workflow (`.github/workflows/tests.yml`) and an `AGENTS.md`. Its defaults differ from v1.0.0 (black oil, real gas, roughness-based friction), and no configuration reproduces v1.0.0's model today: dead oil, ideal gas, a fixed `f_D` and a vertical well come close, but the extra energy terms are always on (`src/manywells/simulator.py:315-326`).
- **A fast Rust implementation exists** on the `rust_implementation` branch and is the first candidate against v1.0.0. It targets v1.0.0's model and API, not `develop`'s: its Python wrapper takes v1's `WellProperties` fields (`L`, `D`, `rho_l`, …), which `develop` has replaced with `geometry` and `fluid`. It must be shown to solve v1's model, not to reproduce v1's solutions. It finds every root of the steady-state equations and can label each one stable or unstable; v1 returns whichever root Ipopt reaches from its initial guess. On 151 sampled two-root wells, v1 landed on the unstable trickle root in 12 (`plans/solver_improvements.md`). Where the two disagree on the operating point, the stable root is correct.
- **Known v1 defect: no root selection.** The paper says the system has "multiple solutions, some which may be unphysical" and relies on a good initial guess. The generators solve each well at u = 0.5 first and reuse that solution as the initial guess for all 500 samples of the well, which likely explains why trickle samples cluster in a few wells. In `manywells-sol-1`, 3.5% of samples (from 131 of 2,000 wells) have PWH within 10 mbar of PDC, with a median TWH of 279 K against 349 K for samples with PWH more than 1 bar above PDC. They make up 33.5k of the 44k samples between 275 and 285 K, which is the low-temperature spike in the TWH histogram. The exact count comes from the relabelling in Step 3.
- **Validation data.** `solution-seeker-as/manywells-validation-data` on HuggingFace is private and stays private: it is the home for the real-well data from paper §6.1 (current contents to be confirmed in Step 3), which is confidential ("Real data is confidential and cannot be shared", paper, Data availability). Everything the verifier needs is synthetic (cases, v1 solutions, stability labels, dataset summaries), so it goes into a new public dataset instead (Step 3), and CI, agents and third parties can run the verifier without credentials.
- **Known v1 limitations** (paper §8) are the natural v2 feature candidates: incompressible liquid and no gas–liquid mass transfer; vertical wells only; a single friction factor for the whole well; a simplified thermal model; independent parameter sampling that can produce unrealistic wells. `develop` already has work on the first four; Step 7 brings it under the verifier.

## Principles

1. **The spec is the source of truth; code is derived.** Every physics expression in the code traces to a numbered equation in `specs/model.md`. No physics change without a spec change in the same PR.
2. **Verify against the equations, not against v1's numbers.** The pass/fail signal is "does this solution satisfy the model?", evaluated with v1's own residual function, so it survives solver changes and language ports and does not flag alternative valid roots as regressions.
3. **Harness before features.** Nothing agent-produced merges until the verifier (Step 2) is green on v1.0.0 itself, meaning it passes everywhere except the expected-failure list of Step 2. The model changes already on `develop` predate the verifier; Step 7 brings them under it after the fact.
4. **Small specs, versioned with the code, reviewed like code.** One spec per feature; the executable part (tests, tolerances, fixtures) is primary, prose is secondary.
5. **Humans decide physics, tolerances, and scope. Agents draft, implement, and run the loop.**
6. **The model's answer is a root set, not a root.** For each case the spec defines the steady-state roots, a stability label for each, and the operating point: the stable root. Which root a solver happens to converge to is not part of the model, so v1's choice of root is not a reference.

## The verification core

The v1.0.0 CasADi graph defines a residual function

```
r(x; θ, b, N) ∈ R^{8(N+1)}
```

over the full state `x = (α_g, α_l, ρ_g, ρ_l, v_g, v_l, p, T)(z_i)`, `i = 0..N`, for well parameters `θ`, boundary conditions and controls `b`, and grid size `N`. Its rows are the discretized mass balances (16)–(17), the implicit-Euler momentum balance (18), the energy balance (19), the 4(N+1) closure relations (drift-flux, softmax regime transitions, fluid properties), and the 4 boundary conditions (inflow (10), choke (11)/(14), temperature (15)).

The verifier holds two graphs: the frozen v1.0.0 graph, and the graph of the current model on `develop`, re-derived from `specs/model.md` and its feature specs (Step 7) and versioned with them. A candidate is checked against the graph of the model it implements. The Rust port and `develop` in its v1-compatibility configuration are checked against the v1.0.0 graph; `develop`'s full model is checked against its own.

A candidate's result for a case — one or more roots, from a refactored Python solver, the Rust port, or anything else — **passes** if all of the following hold:

| Check | Criterion | Applies to |
|---|---|---|
| **Residuals** | Scaled `‖r‖∞ ≤ tol_r`, with per-equation scaling so mass, momentum and energy rows are comparable | every root the candidate reports |
| **Invariants** | `0 ≤ α ≤ 1` and `α_g + α_l = 1`; `α_g ρ_g v_g` and `α_l ρ_l v_l` constant along the well with dead oil (with black oil, gas dissolves into the oil and only the total mass flux is constant); `p(z)` monotone decreasing from bottomhole to wellhead; choke consistent with the critical pressure ratio, `p_sc = max(p_s, r_c p_u)` and the `CHOKED` flag; densities and velocities strictly positive (rules out the zero-flow root, but not the trickle root, whose velocities are small and positive) | every root the candidate reports |
| **Stability** | The candidate's label for each root matches the label computed from the residual graph: drop the choke row, treat `p_0` as a parameter, and one linear solve with the reduced Jacobian gives d(choke row)/d`p_0` at the root, where the choke row is the rate from the tubing minus the rate the choke passes. Positive is unstable, negative is stable. A positive rescaling of the row does not change the sign, so a squared choke equation gives the same label | every reported root that passes Residuals and Invariants |
| **Operating point** | The candidate's operating point is within `tol_x` of the stable reference root | cases with a reference root set (Step 2) |
| **Root set** | The candidate reports every reference root, each within `tol_x` and with the same stability label | candidates that return root sets, on cases with a reference root set |
| **Convergence** | Outputs (PBH, PWH, TWH, rates) change at first order as `Δz → Δz/2 → Δz/4` (implicit Euler) | a fixed subset of cases, on the stable root |
| **Distributions** | Regenerated samples match the marginals and correlation structure of the three published datasets, with trickle-root samples removed (Step 3), within stated bounds | dataset-level, per release |

The choke row (11) takes the square root of `p_L − p_sc` and is NaN when `p_L < p_s`, so the verifier evaluates it only at candidate roots, where `p_L > p_s`.

Because the problem is a nonconvex feasibility NLP with multiple roots and a 95.6% success rate in v1, the verifier must also handle **cases where v1 failed or returned the unstable root**. A candidate that converges where v1 failed is judged on the checks above alone. The headline metric is the **stable-root rate**: of the cases in the validation set that have a stable root, the share where the candidate returns it as the operating point. v2 must not lower it. v1's baseline is its success rate minus the cases where it returned the trickle root.

Value-matching goldens (exact reproduction of v1 outputs) are kept **only for cases where v1 returned the stable root**, as a fast smoke test — they are not the definition of correctness.

### Multiple roots and stability

- **Why two roots.** At zero rate the tubing is full of liquid. If `p_r − p_s < ρ_l g L / 10⁵` (with `L` the true vertical depth for a deviated well), the static column cannot reach the separator. Gas lightens the column at moderate rates and friction dominates at high rates, so the tubing curve crosses the choke curve twice. This criterion split all 2,000 `sol-1` configs into two-root and one-root wells (`plans/solver_description.md` §7). Every `sol-1` config has `w_lg = 0`, so the criterion is untested with gas lift, which can remove the trickle root (well 977 at 0.3 kg/s).
- **Which root is stable.** This is the standard nodal-analysis argument. Raise the rate slightly: at the low-`p_0` root the tubing then delivers less wellhead pressure than the choke needs and the flow falls back; at the high-`p_0` (trickle) root it delivers more and the flow runs away. This is static stability only; heading and other dynamic instabilities are outside a steady-state model.
- **Evidence (v1, 2026-09-30).** The graph-based label was checked on well 977 and 50 sampled `sol-1` configs, 40 of them two-root wells. Every two-root well had exactly one stable and one unstable root, the unstable one at higher `p_0`, as the Rust bracket rule says; every one-root well's root was stable. Running v1 from its default guess plus five cellwise guesses spread over `(p_s, p_r)` found every root in all 50 wells, at a median of 3.3 s per well. v1's default solve returned the trickle root in 1 of the 40 two-root wells, and in well 977. Trickle roots pass Residuals and Invariants: at well 977, `‖r‖∞ = 9·10⁻¹⁴` and the smallest velocity is 0.097 m/s.
- **On `develop`.** `develop` still returns whichever root Ipopt reaches. Its initial guess now comes from a `ca.rootfinder` march instead of a per-cell Ipopt solve, so which root it lands on has to be measured, not assumed to match v1. The graph-based label and the multi-start search work on `develop`'s CasADi graph as they do on v1's, so `develop` can return labelled root sets without the Rust port.
- **Rust solver details** are in `plans/solver_description.md` (§7–8) and `plans/solver_improvements.md`.

### Real-well accuracy (private)

The checks above establish that a candidate solves the model and returns its stable root. They say nothing about whether the physics is accurate. That is what the §6.1 comparison against a real well measures: calibrate the uncertain parameters (friction factor, heat capacity, choke parameters) on part of the data, then compare simulated PBH, PWH and TWH against the 100 measured points, using v1.0.0's errors on the same points as the baseline, after confirming that v1 was on the stable root at each point. Because the data is confidential, this check:

- runs manually or on a private runner, never in public CI and never in an agent session;
- is required for each feature that changes the model (Steps 7 and 9) and before tagging `v2.0.0`, not on every PR;
- reports aggregates only (error metrics per output), so nothing confidential reaches the public repo, PRs, reports or changelogs.

## Steps

### Step 1 — Write the v2 goals and non-goals (1 page)

- **Goal.** Give the process the input it cannot produce itself.
- **Work.** Decide which v1 limitations v2 relaxes; whether v2 is Rust core + Python bindings, Rust and Python side by side, or Python only; which API and dataset-schema breaks are acceptable, including reporting every root with a stability label; what stays out of scope (transient flow, competing with OLGA/LedaFlow).
- **Output.** `specs/goals.md`.
- **Done when.** You would hand it to a new contributor and expect no clarifying questions about scope.
- **Who.** Bjarne. An agent can interview and draft; the decisions are yours.

### Step 2 — Build the verifier from v1.0.0

- **Goal.** A frozen, implementation-independent oracle with a hard automatic pass/fail.
- **Work.**
  1. Pin `manywells==1.0.0` and its CasADi version. Extract the residual function `r(x; θ, b, N)` as a standalone CasADi `Function`, built for each grid size `N` used in the case set (a candidate is always evaluated on its own grid); serialize it (`.casadi`) and generate C source from it, committed under `verification/residual_graph/` and tagged with the CasADi version, so it can be evaluated from Python *and* linked into Rust tests without the Python stack. Build it in a separate environment from the `v1.0.0` tag and commit only the serialized graph: `develop`'s package is also called `manywells`, so the verifier cannot import v1.0.0 next to it.
  2. Write the `manywells-verify` package: `verify(case, candidate_roots) → report` implementing the checks above, including the graph-based stability label, with per-equation residual scaling and a clear verdict.
  3. Define the validation case set: sample wells and operating points covering all regimes, choked/unchoked, gas lift on/off, and all three operating modes. Include cases where v1 failed, two-root wells, wells near the fold where the two roots merge, gas-lift cases (the two-root criterion is untested with gas lift), and wells whose first solve at u = 0.5 lands on the trickle root.
  4. Label each case by its reference root set, found two independent ways: v1 from its default guess plus cellwise guesses spread over `(p_s, p_r)`, and `manywells_rs`. Accept a root only if it passes Residuals and Invariants, and take its stability label from the residual graph. If Rust roots are not yet accurate enough to pass `tol_r` (items 2–3 of `plans/solver_improvements.md`), use each as the initial guess for a v1 solve. A case gets a reference root set when both methods find the same roots; disagreements go to Bjarne. Also record v1's outcome from its default guess: stable root, unstable root, or failed. A v1-only sweep of the choke row over `p_0` does not work as a completeness check: the row is NaN when `p_L < p_s`, which is exactly the region next to the trickle root, and the cellwise march fails at high rates.
  5. Propose initial tolerances (`tol_r`, `tol_x`, invariant slack, convergence-order bounds) and confirm them with Bjarne.
  6. Run the verifier on v1.0.0's own solutions. v1 must pass Residuals, Invariants and Convergence on every solution it returns; it reports no stability labels or root sets, so those two checks do not apply. Cases where v1 returns the unstable root fail Operating point and go on an expected-failure list as a known v1 defect. Any other failure is a real finding about v1, not about the verifier.
- **Output.** `verification/` package (including the residual graph), `specs/verification.md`, and a job in the existing `.github/workflows/tests.yml` that runs it on every PR. The job needs no secrets, so it also runs on PRs from forks (the Rust port arrived as one).
- **Done when.** v1.0.0 passes on the full case set apart from the expected-failure list; the report is readable by a human in under a minute.
- **Who.** Agent builds; Bjarne sets tolerances and adjudicates any v1 failures.

### Step 3 — Publish the public verification dataset

- **Goal.** Make the verifier's inputs public, versioned, and reusable by the Rust port, by agents, and by third parties, with no credentials needed.
- **Work.** Create a new public HuggingFace dataset (working name `solution-seeker-as/manywells-verification`) containing:
  - `cases/` — well parameters `θ`, boundary conditions and controls `b`, grid `N`, one row per case, with a stable case ID.
  - `roots/` — the reference root set per case: `p_0` and full state per root, its stability label, and which of the two methods in Step 2 found it.
  - `solutions_v1/` — full state `x(z_0..z_N)` from v1.0.0 per case, Ipopt status and iteration count, `‖r‖∞` at v1's own solution, the stability label of v1's root, and v1's outcome (stable root, unstable root, or failed).
  - `v1_sample_labels/` — for each sample in the published datasets, whether it is on the stable or the unstable root. The `sol-1` rows cannot be regenerated: each sample redraws `p_r` (±2%) and the fluid fractions, `p_r` is not stored, and the generators seeded from process ID × time. So each row's inputs are rebuilt: u from CHK, `p_s` from PDC, `w_lg` from WGL, the fractions from FGAS/FOIL/FWAT, and `p_r` by inverting the Vogel inflow at PBH and WLIQ. A label is kept only if re-solving reproduces PBH, PWH and TWH; the remaining rows are labelled by the trickle signature (PWH within 10 mbar of PDC) and marked approximate. Check what the `nsol-1` and `nscl-1` rows store before applying the same method to them.
  - `dataset_reference/` — summary statistics (marginals, correlation matrices) of the three published datasets, both as published and with trickle-root samples removed, for the distribution check, so CI does not download 3M rows.
  - A dataset card describing the schema, tolerances, and how to run the verifier against it.

  The residual graph is not in the dataset; it lives with the verifier code (Step 2), so the two are versioned together.
- **Private data.** `manywells-validation-data` stays private and holds only the confidential real-well data, which feeds only the private accuracy check. Confirm what it currently contains; anything synthetic in it moves to the public dataset.
- **Done when.** `manywells-verify` runs end-to-end from a clean clone with only the public dataset as input and no HF token set.
- **Who.** Agent prepares; Bjarne reviews and uploads.

### Step 4 — Extract the spec from paper + v1.0.0, and reconcile

- **Goal.** A structured, numbered model spec, and a settled list of where code and paper disagree.
- **Work.** An agent drafts `specs/model.md` from paper §2–§4 and the appendix (equations with IDs, closures per regime, fluid property and thermal models, inflow and choke boundary conditions, the sampling procedure), then cross-checks against v1.0.0 and produces a **discrepancy list**: clipping, smoothing at regime transitions, safeguards, unit conversions, or constants not in the paper. Bjarne adjudicates each item: paper is right, code is right, or both change. Each equation ID gets a pointer to the verifier row(s) that enforce it. `specs/model.md` describes the v1.0.0 model. The model changes on `develop` since v1.0.0 go on a separate list, as input to Step 7, where each becomes a feature spec; `docs/thermal_energy_modeling.md` and `docs/corrigendum.md` on `develop` are inputs. The spec also needs two items the paper does not define, which Bjarne decides:
  - **Solution set and operating point.** The roots on `(p_s, p_r)`; the static stability criterion (the nodal-analysis argument above, dR/d`p_0` < 0 in the shooting form); the operating point as the stable root; and "no stable root" meaning the well cannot flow at those conditions. The two-root criterion is recorded as a tested property, not a definition.
  - **The choke residual for `p_L ≤ p_s`.** Equation (11) has no real value there: v1 returns NaN and the Rust port squares the equation. The definition of the root set depends on it.
- **Output.** `specs/model.md`, `specs/discrepancies.md` (resolved), the list of model changes on `develop` since v1.0.0, and updated verifier scaling if the reconciliation changes anything.
- **Done when.** Every residual row in the verifier maps to an equation ID and vice versa.
- **Who.** Agent drafts and cross-checks; Bjarne adjudicates.

### Step 5 — Write the constitution and the agent instructions

- **Goal.** Short, stable rules that every agent session starts from.
- **Work.** `specs/constitution.md` (1–2 pages): purpose, non-goals, the principles above, SI units internally, code names follow the paper's nomenclature (`alpha_g`, `rho_l`, `v_m`, `w_g`, `f_D`), determinism given a seed, dataset reproducibility (every dataset row carries the inputs needed to re-solve it, and generators record their seeds), no confidential real-well data in the public repo or datasets. `AGENTS.md` already exists on `develop` (environment, layout, contracts, "done means"); extend it, and symlink `CLAUDE.md` to it, with: how to run the verifier, how to add a case, how to regenerate reference data, what requires human sign-off (any change to `specs/model.md`, any tolerance, any schema change to the published datasets), and that agents never access the private real-well data (the accuracy check is run by Bjarne).
- **Done when.** A fresh agent session, given only the repo, runs the verifier correctly on the first try.
- **Who.** Agent drafts; Bjarne edits.

### Step 6 — Write the v2 architecture spec (short)

- **Goal.** Module boundaries and extension points, designed around the limitations v2 relaxes.
- **Work.** Start from the modules that already exist on `develop` (`geometry.py`, `pvt/`, `slip.py`, `inflow.py`, `choke.py`, `friction.py`, `ca_functions.py`, `units.py`, `calibration/`, `closed_loop/`) and add what is missing: discretization/integrator, solver adapter (Ipopt today; others possible), sampling, dataset schema and writers. Extension points for pluggable friction models, well trajectory, and richer thermal models. Interface contracts with types and units; a solver returns a root set with stability labels, not a single solution. Decide where the Rust implementation sits. Its two shortcuts do not carry over to `develop`'s physics: it computes temperature in closed form, but the frictional-heating term depends on pressure; and it holds the phase mass rates fixed along the well, but they vary once gas dissolves into the oil. A Rust core for the full v2 model would need a temperature solve per cell and phase rates that vary along the well.
- **Output.** `specs/architecture.md`.
- **Done when.** Each planned v2 feature can be located in exactly one module.
- **Who.** Agent drafts; Bjarne reviews.

### Step 7 — Bring `develop` under the verifier

- **Goal.** Put the model changes already on `develop` under the same spec-and-verifier discipline as new features, anchored to v1.0.0.
- **Work.**
  1. Take the list of model changes since v1.0.0 from Step 4: geometry and inclination corrections in the slip model, friction, PVT, the energy balance, lift-gas temperature, fixed-rate inflow, and the initial-guess march.
  2. Add a v1-compatibility configuration in which every change is switched off: vertical well, fixed `f_D`, dead oil, ideal gas, v1 energy balance. The extra energy terms have no switch today. Keep the configuration as a supported mode; it is the regression anchor to v1.0.0.
  3. Require `develop` in that configuration to pass the v1.0.0 verifier on the full case set, including Operating point.
  4. Write a feature spec after the fact for each change (`specs/features/NNN-<name>.md`), with its spec delta against `specs/model.md`, and run the private accuracy check for each.
  5. Build the residual graph of `develop`'s full model from the updated spec, and check `develop` against it on cases that exercise the new features (deviated wells, black oil, real gas, lift-gas temperature). Invariants that hold only for dead oil are dropped for black-oil cases.
  6. Once Step 1 has decided on root reporting, return root sets with stability labels from `develop`'s simulator, using the multi-start search and the graph-based label, and measure its stable-root rate.
- **Output.** The v1-compatibility configuration and its CI job; a feature spec per change since v1.0.0; a verifier version for `develop`'s model.
- **Done when.** `develop` passes the v1.0.0 verifier in the v1-compatibility configuration, and its own verifier on the full model.
- **Who.** Agent implements and drafts the specs; Bjarne adjudicates the specs and any case where `develop` and v1.0.0 disagree in the compatibility configuration.

### Step 8 — Pilot: bring the Rust implementation under the verifier

- **Goal.** Run the full loop once on a bounded, high-value target and learn what the spec and harness are missing.
- **Work.** Treat the `rust_implementation` branch as the first candidate against the v1.0.0 model. It must be shown to solve v1's model and return its stable root, not to reproduce v1's solutions. The pilot exercises the verifier and gives a fast v1-compatible solver; whether the port is extended to `develop`'s model is decided in Steps 1 and 6 (see Step 6 for what that takes). Write its feature spec (`specs/features/NNN-rust-solver.md`): scope, acceptance = passes the verifier on the full case set, stable-root rate ≥ v1, performance target. In scope, from `plans/solver_improvements.md`:
  - a stability label on every returned root, and no positional `simulate()[0]` picks in the scripts (item 1);
  - the α stopping rule (item 2), with well 977 as the regression test: the current Rust code puts its stable root at `p_0` = 169.13 bar, v1 at 169.00 bar;
  - a tolerance relative to drawdown for trickle roots, and a check of R before a root is accepted (items 3–4), without which many trickle roots fail `tol_r`;
  - the energy balance: Rust uses the exact solution of the energy ODE instead of v1's implicit-Euler step (19), so it misses v1's energy rows by up to 0.43 K (`plans/solver_description.md` §8). Switch to v1's recursion, which costs the same; the exact solution can be proposed later as a spec change;
  - regenerating the datasets on the branch that were built with `simulate()[0]`.

  Agent adapts the Rust code to emit full state vectors in the case format, wires the C residual function into Rust tests, fixes what fails, and reports. Bjarne reviews the diff against the spec, not for style.
- **Output.** Rust implementation passing CI; a list of spec/harness gaps found during the pilot, fixed before Step 9.
- **Done when.** The verifier, not a person, is what says the Rust port is correct.
- **Who.** Agent implements; Bjarne reviews, and adjudicates only where the Rust port and the verifier disagree on a root or a label.

### Step 9 — Scale out: one spec per feature, then v2 datasets

- **Goal.** Deliver v2 features through the same loop, in parallel where modules are independent.
- **Work.** For each remaining feature from Step 1 (the changes already on `develop` are covered in Step 7): `specs/features/NNN-<name>.md` (motivation, spec delta with new equation IDs, acceptance tests, out of scope) → agent plan → implementation → verifier → private accuracy check (if the model changed) → human review → merge, with spec and code in the same PR. When a feature changes the model, the residual function changes with it, so the verifier is re-derived from the new spec, versioned, and the affected cases are re-labelled. Finally, regenerate the datasets as `manywells-*-2`, storing every root with a stability column next to `solution_number` (default loaders return the stable root) unless Step 1 decides on the operating point only, and every input needed to re-solve each row. Run the distribution check against the stable-only reference and the private accuracy check, publish with a changelog that references spec versions and the aggregate accuracy results and explains why the ~280 K TWH spike is gone, and tag `v2.0.0`.
- **Done when.** `v2.0.0` released; all features have a spec, a verifier version, and a passing CI run; the private accuracy check has been run on the release candidate and its aggregate results recorded.
- **Who.** Agents in parallel; Bjarne owns specs, tolerances, and releases.

## Sequencing

```
Step 1 ─┬─▶ Step 2 ─▶ Step 3 ───────────┐   ┌─▶ Step 7 ─┐
        │                               ├───┤           ├─▶ Step 9
        └─▶ Step 4 ─▶ Step 5 ─▶ Step 6 ─┘   └─▶ Step 8 ─┘
```

Steps 2–3 (verifier + data) and Steps 4–6 (specs) can run in parallel after Step 1. Steps 7 (`develop`) and 8 (Rust pilot) can run in parallel once both branches are done. Step 9 waits for Step 8 only if Step 1 puts the Rust port in the v2 core. Nothing in Steps 7–9 starts until Step 2 is green on v1.0.0.

## Proposed repository layout

```
manywells/
  AGENTS.md                      # agent instructions (exists on develop; CLAUDE.md symlinked)
  .github/workflows/tests.yml    # existing CI; Step 2 adds the verifier job
  plans/                         # drafts such as this one, not specs
  specs/
    goals.md                     # Step 1
    constitution.md              # Step 5
    model.md                     # Step 4 — equations with IDs
    discrepancies.md             # Step 4 — resolved paper/code differences
    architecture.md              # Step 6
    verification.md              # Step 2 — checks, tolerances, case-set definition
    features/
      NNN-<name>.md              # Steps 7–9, one per feature (after the fact for changes since v1.0.0)
  verification/                  # manywells-verify package
    residual_graph/              # Step 2 — serialized CasADi function + C source
  src/manywells/                 # v2 source: the existing package on develop (Python and/or Rust bindings)
  rust/                          # Rust implementation
  tests/
```

## Open decisions (to settle in Step 1)

- Rust core with Python bindings, or two implementations kept equivalent by the verifier? Either way: is the Rust port extended to `develop`'s model, or kept as a fast v1.0.0 solver?
- Which of the five v1 limitations are in scope for v2.0, and which wait for v2.x?
- Name of the public verification dataset (working name `manywells-verification`), and whether the residual graph stays in the code repo (proposed) or is also mirrored there.
- Acceptance bound for the private accuracy check relative to v1.0.0's errors on the real well.
- Initial tolerances: `tol_r` (scaled residual), `tol_x` (operating-point and root-set checks), and the convergence-order bound.
- Whether the CC BY-NC 4.0 license stays for v2 code (a major version is the natural moment to revisit).
- How the operating-point rule handles edge cases: two stable roots, or only unstable roots. Neither was seen on `sol-1`.
- Whether to publish the v1 sample labels as an erratum or companion column for the `-1` datasets, with a corrigendum note on the TWH spike.
- v2 dataset schema: every root with a stability label (proposed), or the operating point only.
- Energy balance in the Rust port: v1's implicit-Euler step (19), or the exact solution of the ODE as a spec change. The exact solution applies only to v1's energy balance, not to `develop`'s.
