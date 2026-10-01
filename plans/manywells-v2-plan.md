# ManyWells v2 foundation — Spec-Driven Development Plan

*Draft 2026-09-29, revised 2026-09-30 (multiple roots and stability; work continues on `develop`; scope decisions; foundation step towards v2; Step 1 decisions; lean verification) and 2026-10-01 (the port of `develop`'s model to the Rust core becomes Step 9, before the new equation, now Step 10). Owner: Bjarne Grimstad. Status: proposal.*

This plan is a step towards v2, not the release plan. v2 will be a major release with new datasets and perhaps a new paper. This plan brings the repo to a state where new equations are easy to add and test: `develop`'s model is specified in `specs/model/`, the verifier covers it, v1's sampling procedure runs on the new code, and both backends implement the model (Step 9). It ends when one new equation has gone through the whole loop (Step 10). What follows is listed under After this plan.

## Starting point

- **v1.0.0 is released** and is the reference implementation of the model equations: Python, CasADi expression graph, Ipopt feasibility NLP. It was validated against a real well and generated the three published datasets (`manywells-sol-1`, `manywells-nsol-1`, `manywells-nscl-1`). The paper (Geoenergy Science and Engineering 257, 2026) is the de-facto model specification.
- **`develop` is ahead of v1.0.0, and v2 work continues there.** As of 2026-09-30 it is 73 commits past the tag (`git log v1.0.0..develop`). It already has work on four of the five v1 limitations below, without specs or verifier coverage: deviated and L-shaped wells (`geometry.py`); friction from pipe roughness with the Chen and Haaland correlations, with a fixed `f_D` as an option (`friction.py`); black-oil PVT with gas dissolving into the oil, and a real-gas z-factor (`pvt/`); and an energy balance with frictional heating, a gravity term and lift-gas temperature. It also has a fixed-rate inflow, an initial-guess march using `ca.rootfinder`, the `src/` layout, a test suite, a CI workflow (`.github/workflows/tests.yml`) and an `AGENTS.md`. Its defaults differ from v1.0.0 (black oil, real gas, roughness-based friction), and no configuration reproduces v1.0.0's model today: dead oil, ideal gas, a fixed `f_D` and a vertical well come close, but the extra energy terms are always on (`src/manywells/simulator.py:315-326`). Known code issues on `develop` are listed in `plans/improvements.md`, each tagged with the step that handles it.
- **A fast Rust implementation exists** on the `rust_implementation` branch and is the first candidate against v1.0.0. It targets v1.0.0's model and API, not `develop`'s: its Python wrapper takes v1's `WellProperties` fields (`L`, `D`, `rho_l`, …), which `develop` has replaced with `geometry` and `fluid`. It must be shown to return v1's reference roots (Step 2), not v1's default choice of root. It finds every root of the steady-state equations and can label each one stable or unstable; v1 returns whichever root Ipopt reaches from its initial guess. On 151 sampled two-root wells, v1 landed on the unstable trickle root in 12 (`plans/solver_improvements.md`). Where the two disagree on the operating point, the stable root is correct.
- **Known v1 defect: no root selection.** The paper says the system has "multiple solutions, some which may be unphysical" and relies on a good initial guess. The generators solve each well at u = 0.5 first and reuse that solution as the initial guess for all 500 samples of the well, which likely explains why trickle samples cluster in a few wells. In `manywells-sol-1`, 3.5% of samples (from 131 of 2,000 wells) have PWH within 10 mbar of PDC, with a median TWH of 279 K against 349 K for samples with PWH more than 1 bar above PDC. They make up 33.5k of the 44k samples between 275 and 285 K, which is the low-temperature spike in the TWH histogram. Step 3 counts them by this signature, which is a lower bound (`docs/corrigendum.md`).
- **Validation data.** Some real-well data for validation resides in `solution-seeker-as/manywells-validation-data` on HuggingFace, which is private and stays private, because real-well data is confidential ("Real data is confidential and cannot be shared", paper, Data availability). Everything the verifier needs is synthetic (cases, reference roots, v1 solutions, dataset summaries) and lives in this repo, so CI, agents and third parties can run the verifier without credentials.
- **Known v1 limitations** (paper §8) are the natural v2 feature candidates: incompressible liquid and no gas–liquid mass transfer; vertical wells only; a single friction factor for the whole well; a simplified thermal model; independent parameter sampling that can produce unrealistic wells. `develop` already has work on the first four; Step 7 brings it under the verifier. The fifth is left for after this plan, which only ports the sampling procedure (see Scope decisions).

## Scope decisions

- **Calibration is deferred until after this plan** (but before `v2.0.0`). `calibration/` gets no spec here, and the private real-well accuracy check, which depends on calibration, is not run.
- **Closed loop is out of scope for v2** and can be added in a later version. There is no `manywells-nscl-2` dataset and no spec or verifier coverage for `closed_loop/`, which stayed on `develop` untouched until Bjarne retired it on 2026-10-01, with its generators; it remains in v1.0.0. Before Step 7 restructures the simulator, it copies today's `SSDFSimulator` verbatim into `closed_loop/` as its private base, so closed loop's behaviour does not change (decided 2026-09-30, `specs/architecture.md`). Step 3's corrigendum note still covers the published `nscl-1` data, but its final states are not in the verifier's case set (decided 2026-09-30).
- **Sampling is ported now and redesigned later.** This plan makes v1's sampling procedure work with `develop`'s models, with the same approach (independent draws) extended to the new inputs: trajectory, black-oil parameters and pipe roughness. It gets its own spec, `specs/sampling.md` (Step 4), and is implemented in Step 7. In the v1-compatibility configuration it draws v1's inputs as before, so the distribution check can compare regenerated samples with the published datasets. After this plan, the procedure will likely change substantially, to generate v2 datasets that differ significantly from v1's.

## Principles

1. **The spec is the source of truth; code is derived.** Every physics expression in the code traces to a numbered equation in `specs/model/`, and a test enforces it (Step 4). No physics change without a spec change in the same PR.
2. **Anchor to v1.0.0's roots, not to one solver's choice of root.** The v1-compatibility configuration and the Rust port are checked against reference root sets computed from v1.0.0: every root two independent searches find, each a v1 solution with its stability label. That survives solver changes and language ports, does not flag an alternative valid root as a regression, and does not take v1's trickle-root picks as the answer. The verifier does not re-implement the model (see New model versions).
3. **Harness before features.** Nothing agent-produced merges until the verifier (Step 2) is green on v1.0.0 itself, meaning it passes everywhere except the expected-failure list of Step 2. The model changes already on `develop` predate the verifier; Step 7 brings them under it after the fact.
4. **Small specs, versioned with the code, reviewed like code.** One spec per feature; the executable part (tests, tolerances, fixtures) is primary, prose is secondary.
5. **Humans decide physics, tolerances, and scope. Agents draft, implement, and run the loop.**
6. **The model's answer is a root set, not a root.** For each case the spec defines the steady-state roots, a stability label for each, and the operating point: the stable root. Which root a solver happens to converge to is not part of the model, so v1's choice of root is not a reference.
7. **Transparent before marginally faster.** The code must be fast and robust, and a person must be able to read a solver routine next to its spec and follow it. A change that adds solver machinery (a new subroutine, a special-case path, a fallback, a tuning constant) has to pay for itself with a large, measured gain in speed or robustness on the verifier's case set; a marginal speed-up does not justify it. Prefer the simplest method that passes the verifier. Correctness and robustness are requirements, checked by the verifier and the stable-root rate; speed and readability are traded against each other. A PR that adds machinery states the measured gain, and Bjarne decides whether it is large enough.

## The verification core

The verifier checks candidates against **reference root sets computed from v1.0.0**. It does not re-implement the model, and it does not take whichever root v1 happened to reach as the answer. For each case in the case set (Step 2), the reference is every steady-state root that two independent searches find: v1.0.0 from its default guess and from cellwise guesses spread over `(p_s, p_r)`, and the Rust implementation, each of whose roots is used as the initial guess for a v1 solve. Every reference root is therefore a v1.0.0 solution, and its stability label comes from v1.0.0's own Jacobian (below). The Rust implementation is a root finder here, not a reference: its roots differ from v1's by up to 0.32 bar and 0.43 K (`plans/solver_improvements.md`), and the v2 Rust core grows out of it, so bugs they share would go unseen.

The verifier holds no model and needs neither `manywells` nor CasADi. It checks `develop` in its v1-compatibility configuration (Step 7) and the Rust port (Step 8). `develop`'s full model has no reference root sets; it is checked as described under "New model versions" below.

A candidate's result for a case (one or more roots, each a full state on the case's grid, from a refactored Python solver, the Rust port or anything else) **passes** if all of the following hold:

| Check | Criterion | Applies to |
|---|---|---|
| **Invariants** | `0 ≤ α ≤ 1`; densities and velocities strictly positive (rules out the zero-flow root, but not the trickle root, whose velocities are small and positive); `α_g ρ_g v_g` and `α_l ρ_l v_l` constant along the well with dead oil (with black oil only the total mass rate is constant); `p` never rises from bottomhole to wellhead; `p_N > p_s` and `p_0 < p_r`; temperature not below the ambient profile; the `CHOKED` flag, if reported, consistent with `p_s ≤ r_c p_N` outside a small dead band | every root the candidate reports |
| **Operating point** | The candidate's operating point passes Invariants and is within `tol_x` of the stable reference root. Where the reference has several stable roots, it is the one with the lowest `p_0` (SOL-6); where it has none, the candidate reports no operating point | every case |
| **Root set** | Every reference root is matched within `tol_x` by a candidate root | candidates that return root sets |
| **Stability** | Each label the candidate reports matches the label of the reference root it matches | candidates that label their roots |
| **Convergence** | Outputs (PBH, PWH, TWH, rates) change at first order as `Δz → Δz/2 → Δz/4` (implicit Euler) | convergence groups, on the stable root |
| **Distributions** | Samples regenerated in the v1-compatibility configuration, at the stable root, match the marginals and rank correlations of the stable-root reference for `sol-1` and `nsol-1` (below): empirical CDF gap ≤ 0.02 per feature, rank-correlation gap ≤ 0.05 | dataset-level, whenever the sampler or the generation pipeline changes |

`tol_x` is a scaled ∞-norm over the whole state, so matching a reference root pins every value on the grid, not only the outputs. A valid candidate root that matches no reference root is a finding for Bjarne, not a failure: either both searches missed it or the candidate is wrong, and the verifier cannot tell which.

**Stability label** (computed when the reference is built, in the v1.0.0 environment): drop the choke row from v1's residual, treat `p_0` as a parameter, and one linear solve with the reduced Jacobian gives d(choke row)/d`p_0` at the root, where the choke row is the rate from the tubing minus the rate the choke passes. Positive is unstable, negative is stable. A positive rescaling of the row does not change the sign, so a squared choke equation gives the same label.

The distribution check follows the stable root, not v1's choice of root. Its reference is the published data with the rows on the trickle-root signature removed (Step 3), so it has at most a trace of the ~280 K TWH spike, which matches what the Rust implementation's stable-root detection produces. Removing rather than replacing those rows slightly underweights the 131 `sol-1` wells whose samples v1 put on the trickle root; the check's bounds are set so that this and sampling noise pass. The regenerated samples are drawn for the published well configs, not for newly sampled wells, because v1's `nsol` generator also filtered wells by their root: it discarded a well when its solve at u = 1 gave `w_tot < 7` kg/s, using whichever root it found. Datasets regenerated with the Rust implementation after its `simulate()[0]` fix (Step 8) are a cross-check of the reference's shape.

Because the problem is a nonconvex feasibility NLP with multiple roots and a 95.6% success rate in v1, the case set also contains **cases where v1 failed or returned the unstable root** from its default guess. Their reference roots come from other starts and from Rust. The headline metric is the **stable-root rate**: of the cases that have a stable root, the share where the candidate returns it as the operating point. No change may lower it. v1's baseline is its success rate minus the cases where it returned the trickle root.

### New model versions

A change on `develop`, and any new equation (Step 10), comes in as an option that is off in the v1-compatibility configuration. That configuration must keep passing against the v1.0.0 reference; a change that alters it is a regression and fails by design. With an option on, there is no reference root set and no second implementation of the model. The option is checked by:

- test vectors for each component function (friction, PVT, slip, …) in its spec file;
- Invariants, dropping those that assume dead oil where gas dissolves into the oil;
- spot checks of relations that need no closure: the inflow equation at the bottom and the choke equation at the top, conservation of total mass, non-negative friction in every cell, and heat flowing outwards;
- Convergence;
- stability property checks with the implementation's own Jacobian: a two-root well has one stable and one unstable root, the unstable one at higher `p_0`;
- once the Rust core covers the option, agreement between the two backends on the rows at the same states and on the operating point for the same cases. Both backends are kept (`specs/goals.md`).

What this does not catch is a self-consistent assembly or discretization error on a new code path, of the kind of the Rust port's exact energy solution: component test vectors do not see how components are assembled, and Convergence passes a consistent scheme. Spec review and the two backends are the defence.

### Multiple roots and stability

- **Why two roots.** At zero rate the tubing is full of liquid. If `p_r − p_s < ρ_l g L / 10⁵` (with `L` the true vertical depth for a deviated well), the static column cannot reach the separator. Gas lightens the column at moderate rates and friction dominates at high rates, so the tubing curve crosses the choke curve twice. This criterion split all 2,000 `sol-1` configs into two-root and one-root wells (`plans/solver_description.md` §7). Every `sol-1` config has `w_lg = 0`, so the criterion is untested with gas lift, which can remove the trickle root (well 977 at 0.3 kg/s). In Step 2's case set, gas lift removed it in 9 of 10 wells that meet the criterion without it.
- **Which root is stable.** This is the standard nodal-analysis argument. Raise the rate slightly: at the low-`p_0` root the tubing then delivers less wellhead pressure than the choke needs and the flow falls back; at the high-`p_0` (trickle) root it delivers more and the flow runs away. This is static stability only; heading and other dynamic instabilities are outside a steady-state model.
- **Evidence (v1, 2026-09-30).** The graph-based label was checked on well 977 and 50 sampled `sol-1` configs, 40 of them two-root wells. Every two-root well had exactly one stable and one unstable root, the unstable one at higher `p_0`, as the Rust bracket rule says; every one-root well's root was stable. Running v1 from its default guess plus five cellwise guesses spread over `(p_s, p_r)` found every root in all 50 wells, at a median of about 3 s per well. v1's default solve returned the trickle root in 1 of the 40 two-root wells, and in well 977. Trickle roots are genuine roots: at well 977, v1's residual is `9·10⁻¹⁴` at the trickle root and the smallest velocity is 0.097 m/s. The scripts and results are in `plans/evidence/`.
- **On `develop`.** `develop` still returns whichever root Ipopt reaches. Its initial guess now comes from a `ca.rootfinder` march instead of a per-cell Ipopt solve, so which root it lands on has to be measured, not assumed to match v1. The graph-based label and the multi-start search work on `develop`'s CasADi graph as they do on v1's, so `develop` can return labelled root sets without the Rust port.
- **Rust solver details** are in `plans/solver_description.md` (§7–8) and `plans/solver_improvements.md`.

### Real-well accuracy (private)

The checks above establish that a candidate solves the model and returns its stable root. They say nothing about whether the physics is accurate. That is what the §6.1 comparison against a real well measures: calibrate the uncertain parameters (friction factor, heat capacity, choke parameters) on part of the data, then compare simulated PBH, PWH and TWH against the 100 measured points, using v1.0.0's errors on the same points as the baseline, after confirming that v1 was on the stable root at each point. Because the data is confidential, this check:

- runs manually or on a private runner, never in public CI and never in an agent session;
- is not run in this plan, because it depends on calibration, which is deferred (see Scope decisions); for `v2.0.0` it runs once, on the release candidate, not per feature or per PR;
- reports aggregates only (error metrics per output), so nothing confidential reaches the public repo, PRs, reports or changelogs.

## Steps

### Step 1 — Write the v2 goals and non-goals (1 page)

- **Goal.** Give the process the input it cannot produce itself.
- **Work.** Decide which v1 limitations v2 relaxes; whether v2 is Rust core + Python bindings, Rust and Python side by side, or Python only; which API and dataset-schema breaks are acceptable, including reporting every root with a stability label; what stays out of scope (transient flow, competing with OLGA/LedaFlow).
- **Output.** `specs/goals.md`, covering the goals of this plan and the direction for v2 that it prepares for; the v2 goals can be revised after this plan.
- **Done when.** You would hand it to a new contributor and expect no clarifying questions about scope.
- **Who.** Bjarne. An agent can interview and draft; the decisions are yours.
- **Status.** Done 2026-09-30. `specs/goals.md` records the decisions; the steps below are updated to match.

### Step 2 — Build the verifier and its v1.0.0 reference

- **Goal.** A frozen, implementation-independent reference with a hard automatic pass/fail.
- **Work.**
  1. Set up two build environments: a worktree of the `v1.0.0` tag with its pinned CasADi version, and a worktree of `rust_implementation`. They produce the reference data only. `develop`'s package is also called `manywells`, so neither is imported next to it.
  2. Write the `manywells-verify` package: `verify(case, candidate_roots, reference) → report` implementing the checks above, with a clear verdict. It depends on neither `manywells` nor CasADi.
  3. Define the case set, about 150 cases committed to the repo: sampled wells and operating points covering all regimes, choked/unchoked, gas lift on/off, and the open-loop stationary and nonstationary modes; there are no closed-loop cases (decided 2026-09-30). Include cases where v1 failed, two-root wells, wells near the fold where the two roots merge, cases past the fold with no root, cases for the Bernoulli choke and productivity-index inflow that no dataset uses, gas-lift cases (the two-root criterion is untested with gas lift), wells whose first solve at u = 0.5 lands on the trickle root, and convergence groups at N, 2N and 4N.
  4. Label each case by its reference root set, found two independent ways: v1 from its default guess plus cellwise guesses spread over `(p_s, p_r)`, and `manywells_rs`, whose roots are each re-solved by v1. Accept a root only if Ipopt reports success and it passes Invariants, and take its stability label from v1's Jacobian. A case is settled when both searches find the same roots. Disagreements, and labels whose slope is too small to trust, go to Bjarne, and those cases stay out of the case set until he rules. Also record v1's outcome from its default guess: stable root, unstable root, or failed. A v1-only sweep of the choke row over `p_0` does not work as a completeness check: the row is NaN when `p_L < p_s`, which is exactly the region next to the trickle root, and the cellwise march fails at high rates.
  5. Propose initial tolerances (`tol_x`, invariant slack, convergence-order bounds) and confirm them with Bjarne.
  6. Run the verifier on v1.0.0's own default-guess solutions. Cases where v1 returns the unstable root or fails go on an expected-failure list as known v1 defects. Any other failure is a finding about v1 or the reference.
- **Output.** The `verification/` package, the committed case set and reference root sets (`verification/data/`), `specs/verification.md`, and a job in the existing `.github/workflows/tests.yml` that runs the verifier on every PR. The job needs no secrets, so it also runs on PRs from forks (the Rust port arrived as one).
- **Done when.** v1.0.0 passes on the case set apart from the expected-failure list; the report is readable by a human in under a minute.
- **Who.** Agent builds; Bjarne sets tolerances and adjudicates disagreements and any v1 findings.
- **Status.** Done 2026-09-30 (`specs/verification.md`). 141 cases with 200 reference roots are committed. Cases where the two searches disagree were left out rather than adjudicated one by one (Bjarne): 17 of the 158 drawn. v1.0.0 passes with 35 expected failures (20 trickle roots, 15 failed solves) and a stable-root rate of 74.3% on this deliberately hard set. Findings for later steps are recorded in Steps 3, 4 and 8.

### Step 3 — Distribution reference and the v1 erratum note

- **Goal.** The reference for the Distributions check, and a note for v1 users on the trickle root.
- **Work.**
  - Build the stable-root reference from the published `sol-1` and `nsol-1` rows, with the rows on the trickle-root signature (PWH within 10 mbar of PDC) removed. Labelling every row exactly would mean rebuilding each row's inputs and re-solving it, so the signature is used instead (decided 2026-09-30, "keep it simple"). On Step 2's case set it catches 47 of the 64 unstable roots and none of the 136 stable ones, so some trickle rows stay in the reference (in `sol-1`, roughly 1% of rows are ambiguous).
  - Store the reference as summary statistics, committed under `verification/`: per feature, its values at the percentiles 1..99 and the empirical CDF there, and the Spearman correlation matrix. The check compares a candidate's empirical CDFs and rank correlations with them, within bounds that pass sampling noise and fail the full trickle spike.
  - Add a note to `docs/corrigendum.md` with the trickle-signature counts per dataset (lower bounds) and the second erratum found in Step 2: the `nsol-1` and `nscl-1` configs store `f_D = 0.05` for every well, although the generator draws it from U(0.01, 0.08). For `nsol-1`, the friction factor each well was simulated with can be recovered from its stored final state (`verification/build/make_cases.py`, `recover_f_D`). No per-sample labels are published and no public verification dataset is created (decided 2026-09-30).
- **Private data.** `manywells-validation-data` stays private and holds only the confidential real-well data, which feeds only the private accuracy check. Bjarne confirmed on 2026-09-30 that it holds nothing else.
- **Done when.** The Distributions check runs from a clean clone, and the corrigendum note is in `docs/`.
- **Who.** Agent prepares; Bjarne reviews.
- **Status.** Done 2026-09-30; Bjarne confirmed the removal of signature rows and the bounds. `verification/data/distribution_reference.json` (171 kB) holds the reference, `manywells-verify-distributions` runs the check (bounds: CDF gap 0.02, rank-correlation gap 0.05), and `docs/corrigendum.md` has both errata. Calibration: a 50k-row subsample of the reference gives gaps of 0.005 and 0.009; the published `sol-1` rows with the trickle rows left in give 0.034 (TWH) and 0.092, so they fail; `nsol-1`'s 0.15% of trickle rows are too few to matter (0.001, 0.004).

### Step 4 — Extract the spec from paper + v1.0.0, and reconcile

- **Goal.** A structured, numbered model spec, and a settled list of where code and paper disagree.
- **Work.** An agent drafts `specs/model/` from paper §2–§4 and the appendix (equations with IDs, closures per regime, fluid property and thermal models, inflow and choke boundary conditions), then cross-checks against v1.0.0 and produces a **discrepancy list**: clipping, smoothing at regime transitions, safeguards, unit conversions, or constants not in the paper. Bjarne adjudicates each item: paper is right, code is right, or both change. Each equation ID gets a pointer to the test vectors or spot check that exercise it. This step writes the `v1.0.0` configuration. The model changes on `develop` since v1.0.0 go on a separate list, as input to Step 7, which adds them to the component files; `docs/thermal_energy_modeling.md` and `docs/corrigendum.md` on `develop` are inputs. The spec is organised as follows:
  - **One file per model part, mirroring `src/manywells/`** (tree under Proposed repository layout). Where an equation belongs, and which spec file a PR touches, then follows from the code. `README.md` holds the scope, the ID scheme and how the parts compose; `nomenclature.md` defines symbols, units and the state vector once for every file.
  - **Same template in every component file:** purpose; interface (inputs, outputs, units, and whether it must work on CasADi symbols); equations with IDs; options, and which configuration uses each; safeguards (clipping, smoothing, bounds); sources; test vectors (inputs with expected outputs, checked by the tests in `tests/`); pointers to code and verifier rows.
  - **The spec says what; `docs/` says why.** Spec files state equations, units, validity ranges and safeguards, and cite derivations, which stay in `docs/` (for example `docs/thermal_energy_modeling.md`). That keeps each file to a few pages.
  - **Stable, namespaced equation IDs**, such as `FRIC-2`, `PVT-OIL-5` or `CHK-3`. IDs are never renumbered, and removing an equation retires its ID. Paper numbers such as (16) are recorded as aliases.
  - **Options in the component files, configurations in `README.md`.** Each component file lists its alternatives, for example fixed `f_D` or Chen in `friction.md`. `README.md` defines named configurations, one choice per component: `v1.0.0` and `develop`'s default to begin with. The v1.0.0 model and `develop`'s model are then two configurations of one spec, not two documents.
  - **A traceability test.** Code carries tags such as `# spec: FRIC-2`. A pytest checks that every tag names an existing ID, and that every ID is tagged in code or marked spec-only. This makes principle 1 a CI check; it must pass from Step 7 on.

  The spec also needs two items the paper does not define, which Bjarne decides:
  - **Solution set and operating point** (`solution.md`). The roots on `(p_s, p_r)`; the static stability criterion (the nodal-analysis argument above, dR/d`p_0` < 0 in the shooting form); the operating point as the stable root; "no stable root" meaning the well cannot flow at those conditions; and the operating point when there are two stable roots. Step 2 found that case near the fold, where the model has more than two roots: `fold-1505` and `fold-0485` each have two stable v1 roots, and v1's residual crosses zero more often than either root search resolves. The two-root criterion is recorded as a tested property, not a definition.
  - **The choke residual for `p_L ≤ p_s`** (`choke.md`). Equation (11) has no real value there: v1 returns NaN and the Rust port squares the equation. The definition of the root set depends on it.

  Sampling gets its own spec, `specs/sampling.md`, outside `specs/model/` because it is not physics. It records v1's procedure (`scripts/data_generation/well.py`: per-well draws in `sample_well`, per-sample redraws in `sample_new_conditions`) and extends it to `develop`'s inputs: trajectory, black-oil parameters and pipe roughness. The approach stays the same (independent draws), and the v1-compatibility configuration draws v1's inputs as before.
- **Output.** `specs/model/` with the `v1.0.0` configuration, `specs/sampling.md`, `specs/discrepancies.md` (resolved), the list of model changes on `develop` since v1.0.0, and the traceability test.
- **Done when.** Every equation in v1.0.0's discretized system (the rows its simulator assembles) maps to an equation ID, and every equation ID has test vectors, is exercised by a spot check, or is marked spec-only with a reason.
- **Who.** Agent drafts and cross-checks; Bjarne adjudicates.
- **Status.** Done 2026-09-30; Bjarne ruled on every item, accepting the proposed rulings; D-9's surface-tension coefficients match the source (Abdul-Majeed and Abu Al-Soof, 2000) exactly. `specs/model/` has the `v1.0.0` configuration: 71 equation IDs, each covered by test vectors from v1.0.0, a verifier check or a spec-only reason, and every row of v1.0.0's system maps to an ID. The two open definitions are settled: where `p_N ≤ p_c` the choke passes no flow, and any row with the same sign is allowed (CHK-11); with several stable roots the operating point is the one with the lowest `p_0` (SOL-6), which the verifier now checks. `specs/discrepancies.md` records the rulings on 24 model and 12 sampling items; the paper errors among them, the closed-loop error in (52) and the `nsol-1`/`nscl-1` config erratum are in `docs/corrigendum.md`. `specs/sampling.md` records v1's procedure; its seeding (SMP-31) and the extension to `develop`'s inputs (SMP-40 to SMP-44) are placeholders that Step 7 settles. `plans/develop_model_changes.md` is the list for Step 7, with the row-level gaps measured. `tests/test_spec_traceability.py` passes except its tagging check, a strict expected failure until Step 7; `tests/test_spec_vectors.py` checks `develop` against the vectors, with each known gap a strict expected failure.

### Step 5 — Write the constitution and the agent instructions

- **Goal.** Short, stable rules that every agent session starts from.
- **Work.** `specs/constitution.md` (1–2 pages): purpose, non-goals, the principles above, SI units internally, code names follow the paper's nomenclature (`alpha_g`, `rho_l`, `v_m`, `w_g`, `f_D`), determinism given a seed, dataset reproducibility (every dataset row carries the inputs needed to re-solve it, and generators record their seeds), no confidential real-well data in the public repo or datasets. `AGENTS.md` already exists on `develop` (environment, layout, contracts, "done means"); extend it, and symlink `CLAUDE.md` to it, with: how to run the verifier, how to add a case, how to regenerate reference data, a "Done means" line for principle 7 ("if the change adds a solver routine or path, state the measured gain on the case set"), a slow test that runs `scripts/sim_examples/` headless so the existing "examples still run" rule is checked (`plans/improvements.md` §3), the correct command for examples that import `scripts.*` (§1.6), what requires human sign-off (any change to `specs/model/`, any tolerance, any schema change to the published datasets), and that agents never access the private real-well data (the accuracy check is run by Bjarne).
- **Done when.** A fresh agent session, given only the repo, runs the verifier correctly on the first try.
- **Who.** Agent drafts; Bjarne edits.
- **Status.** Done 2026-09-30; Bjarne confirmed the sign-off list, the rule that generators record their code version as well as their seeds, and a `CLAUDE.md` that imports `AGENTS.md` (`@AGENTS.md`) instead of a symlink, which Windows clones without `core.symlinks` would check out as a text file. He rewrote the confidential-data rule: some real-well data for validation resides in `manywells-validation-data`. `specs/constitution.md` holds the rules, with the principles above numbered as here. `AGENTS.md` now covers the verifier, adding a case, regenerating reference data, the principle-7 "Done means" line, the `-m` command for scripts, what needs Bjarne's sign-off, and the private real-well data. `tests/test_examples.py` (slow) runs every example headless. The plan's wording changed in two places: "SI units internally" is stated as in `specs/model/nomenclature.md` (SI in equations, bar and kelvin at interfaces), and the void fraction's code name is `alpha`, as in the state vector, not `alpha_g`.

### Step 6 — Write the v2 architecture spec (short)

- **Goal.** Module boundaries and extension points, designed around the limitations v2 relaxes.
- **Work.** Start from the modules that already exist on `develop` (`geometry.py`, `pvt/`, `slip.py`, `inflow.py`, `choke.py`, `friction.py`, `ca_functions.py`, `units.py`, `calibration/`) and add what is missing: discretization/integrator, solver adapter (Ipopt today; others possible), sampling, dataset schema and writers. `calibration/` gets its contract later, before `v2.0.0`; `closed_loop/` is out of scope. Extension points for pluggable friction models, well trajectory, and richer thermal models. Interface contracts with types and units; a solver returns a root set with stability labels, not a single solution. Inputs from `plans/improvements.md`: the `isinstance` choke dispatch (§2.3), the hidden `_w_l_inflow` state (§2.4), the simulator mutating the caller's objects (§2.5), and building the NLP once with the boundary conditions as parameters (§4.1), weighed under principle 7. Design the Rust core for `develop`'s model, with Python bindings as the public API (Step 1 made Rust the v2 core; see `specs/goals.md`). The port's two shortcuts do not carry over to `develop`'s physics: it computes temperature in closed form, but the frictional-heating term depends on pressure; and it holds the phase mass rates fixed along the well, but they vary once gas dissolves into the oil. The core therefore needs a temperature solve per cell and phase rates that vary along the well.
- **Output.** `specs/architecture.md`.
- **Done when.** Each planned v2 feature can be located in exactly one module.
- **Who.** Agent drafts; Bjarne reviews.
- **Status.** Done 2026-09-30. Bjarne ruled on the five open decisions, taking each recommendation: extension by contributors rather than runtime plug-ins; a frozen copy of today's `SSDFSimulator` as `closed_loop/`'s private base; all four API breaks (`simulate(bc)` returning a `Root`, `friction` and `thermal` objects, frozen inputs, `p_sep` and `p_bubble` in bar); the flux-difference mass-row form, whose spec text comes to him in Step 7; and the verifier's `tol_x` and `label_min` in the root search. `specs/architecture.md` splits `simulator.py` into `discretization`, `thermal`, `solvers/` and `solution`, adds `configurations`, `sampling/` and `datasets/`, and gives the interface contracts, the extension points, the Rust core's design for `develop`'s model and a feature map with one module per feature. It settles `plans/improvements.md` §2.3, §2.4, §2.5 and §4.1 for Step 7. Measured on two wells, building each well's NLP once bounds the speed-up of a warm re-solve at about 40x (the build is 97% of its time); Step 7 repeats this on the case set.

### Step 7 — Bring `develop` under the verifier

- **Goal.** Put the model changes already on `develop` under the same spec-and-verifier discipline as new features, anchored to v1.0.0.
- **Work.**
  1. Take the list of model changes since v1.0.0 from Step 4: geometry and inclination corrections in the slip model, friction, PVT, the energy balance, lift-gas temperature, fixed-rate inflow, and the initial-guess march.
  2. Add a v1-compatibility configuration to the code (the `v1.0.0` configuration in `specs/model/README.md`), in which every change is switched off: vertical well, fixed `f_D`, dead oil, ideal gas, v1 energy balance. The extra energy terms have no switch today. Keep the configuration as a supported mode; it is the regression anchor to v1.0.0.
  3. Require `develop` in that configuration to pass the verifier on the case set, including Operating point.
  4. Write a feature spec after the fact for each change (`specs/features/NNN-<name>.md`: motivation, delta, acceptance), add the change to the component files in `specs/model/` as a new option with new equation IDs, and tag the code. The feature specs are the change record; `specs/model/` always shows the current model. Fix known errors before specifying them: `water_fvf` has the wrong sign (`plans/improvements.md` §1.2). Pin the energy balance with test vectors in `thermal.md` (§3), and make the slip parameters dataclass fields while specifying `slip.md` (§2.7).
  5. Check `develop`'s full model as described under New model versions, on cases that exercise the new features (deviated wells, black oil, real gas, lift-gas temperature): component test vectors, Invariants (without the dead-oil ones for black-oil cases), the spot checks, Convergence and the stability property checks.
  6. Implement the root reporting decided in Step 1 in `develop`'s simulator: `simulate()` returns the operating point (the stable root) and raises if there is none, and the root set with a stability label per root is available on request. Use the multi-start search and the graph-based label, and measure the stable-root rate.
  7. Implement the ported sampler from `specs/sampling.md`, in the module Step 6 assigns to sampling. `scripts/data_generation/` is broken against `develop`'s API, and some wrong assignments fail silently rather than raising (`plans/improvements.md` §1.3). Check that in the v1-compatibility configuration it reproduces v1's input distributions, and that samples regenerated at the stable root (item 6) pass the distribution check.
- **Output.** The v1-compatibility configuration and its CI job; a feature spec per change since v1.0.0; the spot and property checks for `develop`'s model; the ported sampler.
- **Done when.** `develop` passes the verifier in the v1-compatibility configuration and the spot and property checks on the full model, and the traceability test passes.
- **Who.** Agent implements and drafts the specs; Bjarne adjudicates the specs and any case where `develop` and v1.0.0 disagree in the compatibility configuration.
- **Status.** Implemented 2026-10-01 (feature specs `specs/features/001` to `013`). Bjarne's rulings, 2026-10-01: he signed off the flux-difference mass rows (decision 4's text), dropping the $10^{-9}$ guard of SLIP-10, the water FVF as $1 - c_w\Delta p$, and the Step 7 changes to `specs/architecture.md`; he ruled that PVT-OIL-5 follows the source ($\log_{10}$, now in the code); and he accepted all four root-search additions under principle 7. He kept the sampler's starting points until the redesign and decided that the Distributions check weights a candidate's wells as the reference's. Open: the margin of the `sol-1` regeneration (item 7).
  - **Restructure** (`specs/architecture.md`): `closed_loop/` runs on a frozen copy of the old simulator, pinned by `tests/test_closed_loop.py`; frozen, validated inputs with `friction` and `thermal` components; `discretization.build_system`, `solvers/`, `solution.py`; `SSDFSimulator(wp).simulate(bc)` returns the operating point, `root_set(bc)` every root; `configurations.v1_well` and `check`. Every API break is in `CHANGELOG.md`.
  - **Items 2–3.** `develop` in the `v1.0.0` configuration reproduces every v1.0.0 row vector to $10^{-10}$, and passes the verifier with no expected failures: 141 cases, stable-root rate 100% (v1.0.0: 74.3%), every reference root found with its label. The CI job `verify` runs it (`scripts/verification/develop_candidate.py`); the job could not fail before, because its pipe into `tee` lacked `pipefail`, which is fixed.
  - **Item 4.** `specs/model/` has `develop`'s options as new IDs (GEO-3, GEO-4, DISC-7 to DISC-11, BAL-10 to BAL-12, THM-4 to THM-7, SLIP-10, SLIP-11, FRIC-3 to FRIC-6, PVT-GAS-3 to PVT-GAS-8, PVT-OIL-4 to PVT-OIL-14, PVT-WAT-2, PVT-WAT-3, PVT-MIX-6 to PVT-MIX-10, INF-8, SMO-4), each traced to code, with develop vectors (`specs/tools/make_develop_vectors.py`) that pin them for the Rust core. The traceability test passes in full. `water_fvf`'s sign is fixed, the slip parameters are fields, the energy terms have vectors. Found: the Vazquez–Beggs separator correction used `ln` where the source has $\log_{10}$ (PVT-OIL-5); fixed on Bjarne's ruling.
  - **Item 5.** `tests/test_model_properties.py`: Invariants without the dead-oil ones, the spot checks, convergence at first order, and the two-root stability property, on deviated and L-shaped black-oil wells and with cold lift gas. All pass.
  - **Item 6.** The root search: seven starts (method A's and 0.975, 0.999), a bounded-Ipopt fallback in the march, acceptance on `Solve_Succeeded`, and the gas-law row in its canonical form. Measured gains in `specs/features/012-root-search.md`.
  - **Item 7.** `manywells.sampling` and `manywells.datasets` port SMP-1 to SMP-30, with SMP-31 (seeding) and SMP-40 written; Bjarne kept the starting points of SMP-41 to SMP-43 until the sampling redesign. The generators are thin callers. Distributions: samples regenerated at the stable root for the 2,000 published `sol-1` wells, 5 per well (9,879 rows, 17 min on 24 cores; `scripts/verification/regenerate_distributions.py`), failed the check as written (CDF gap 0.026 for PWH, bound 0.02). Five rows per well of the reference's own rows failed it too (0.035), because the reference mixes the wells by their published rows, which the trickle signature reduced in 131 wells, while a regeneration draws the same number per well. Bjarne decided that the check weights a candidate's rows so that its wells mix as the reference's do; the reference now stores each well's rows (a rebuild left every other value unchanged). Weighted, the reference's own 5-per-well rows give 0.008 and 0.011, and `develop`'s regeneration passes at 0.0196 and 0.031, close to the CDF bound. Open: whether to tighten that margin (Bjarne to say how). Step 8's regeneration with the Rust core belongs to the same question: 9,920 rows, CDF gap 0.021, failing narrowly, mostly from samples the CasADi search does not solve. `nsol-1` is not regenerated. 2 published wells have no operating point at u = 0.5 under `develop` (v1.0.0 returned a root for them), and 121 of the 10,000 samples have none.
  - **Build once** (§4.1, `plans/evidence/build_once.py`): on the 136 cases with a reference root, a warm solve from the case's root took 0.507 s before Step 7 (develop at `c5fedf2`, rebuilding on every call) and 3.5 ms after (medians; 110x median, 122x in total); a cold solve, march and Ipopt, 0.719 s and 24.7 ms (31x median, 9x in total, where a few long Ipopt runs dominate). The build, once per well, takes a median of 0.30 s. The sampler does not get this gain: each sample has its own fractions and so its own fluid (SMP-22, SMP-24), and its well is rebuilt. Making the fluid's fractions parameters of the system would give it to dataset generation (`plans/improvements.md` §4.5, with measurements against `main` and `rust_implementation`; for Bjarne, after the plan).

### Step 8 — Pilot: bring the Rust implementation under the verifier

- **Goal.** Run the full loop once on a bounded, high-value target and learn what the spec and harness are missing.
- **Work.** Treat the `rust_implementation` branch as the first candidate against the v1.0.0 model. It must be shown to return v1's reference roots and pick the stable one, not to reproduce v1's default choice of root. The pilot exercises the verifier and gives a fast v1-compatible solver. Step 1 made the port the v2 core; extending it to `develop`'s model is designed in Step 6 and done in Step 9. Write its feature spec (`specs/features/NNN-rust-solver.md`): scope, acceptance = passes the verifier on the full case set, stable-root rate ≥ v1, performance target. Principle 7 decides between the options in `plans/solver_improvements.md`: for example, the bracketed α solve alone is the default over Aitken acceleration with a bracketed fallback (item 2), unless its 3.1x slowdown in the JavaScript port matters; and the scan refinement (item 5) is added only if the case set has wells near the fold. In scope, from `plans/solver_improvements.md`:
  - a stability label on every returned root, and no positional `simulate()[0]` picks in the scripts (item 1);
  - the α stopping rule (item 2), with well 977 as the regression test: the current Rust code puts its stable root at `p_0` = 169.13 bar, v1 at 169.00 bar;
  - a tolerance relative to drawdown for trickle roots, and a check of R before a root is accepted (items 3–4), without which many trickle roots miss the reference root by more than `tol_x`;
  - the energy balance: Rust uses the exact solution of the energy ODE instead of v1's implicit-Euler step (19), so it misses v1's energy rows by up to 0.43 K (`plans/solver_description.md` §8). Switch to v1's recursion, which costs the same. This was settled in Step 1: the exact solution applies only to v1's energy balance, not to `develop`'s, which the core must implement;
  - regenerating the datasets on the branch that were built with `simulate()[0]`.

  Step 2 found, on its case set: Rust found no root at `nofold-0567`, where v1 finds a stable and an unstable root; near the fold its residual jitters (12 sign changes within 1.5 bar at `fold-0485`), as item 2 predicts, and there it missed one of the two stable v1 roots. The other way round, Rust's roots found 13 roots that v1's own starts missed, mostly trickle roots, so it stays useful as a root finder.

  Agent adapts the Rust code to emit full state vectors in the case format, runs the verifier on them, fixes what fails, and reports. Bjarne reviews the diff against the spec, not for style.
- **Output.** Rust implementation passing CI; a list of spec/harness gaps found during the pilot, fixed before Step 9.
- **Done when.** The verifier, not a person, is what says the Rust port is correct.
- **Who.** Agent implements; Bjarne reviews, and adjudicates only where the Rust port and the verifier disagree on a root or a label.
- **Status.** Done 2026-10-01 (`specs/features/014-rust-solver.md`). Bjarne's rulings, 2026-10-01:
  - he signed off the full scan, the bracketed void fraction, the drawdown ladder, the jump guard, and the tolerances and slope;
  - he set no performance target yet;
  - he accepted the Step 8 edits to `specs/architecture.md` and `specs/verification.md`;
  - he ruled on each gap below.

  The port is merged into `rust/`: the crate only, keeping Kajrakso's commits. It is built as `manywells._core` by maturin and restructured into the architecture's modules, with each point's rows defined once. It runs as `SSDFSimulator(wp, backend='rust')` for wells in the `v1.0.0` configuration.
  - **Verifier.** The core passes with no expected failures. Every one of the 200 reference roots is matched with its label (median distance 7e-14), and the stable-root rate is 136/136. The search takes 65 ms per case (median, one core), 23× faster than `develop`'s CasADi backend. The core's rows match v1.0.0's row vectors to 1e-10. The CI job `verify` runs it.
  - **Changes from the port**, each with its measured gain in the feature spec, signed off under principle 7:
    - v1.0.0's energy rows, which the port's exact ODE solution missed by up to 0.43 K;
    - Brent on the slip row for the void fraction, instead of a stop on the step size;
    - the canonical choke row, with an exact stop below $p_s$;
    - a full scan with every bracket, refinement near folds, and a drawdown ladder near $p_r$;
    - Brent in the drawdown, with an $\lvert R\rvert$ acceptance;
    - a check that each cell's row is zero, so that a jump is not taken for a root;
    - a central-difference slope, labelled as the CasADi backend labels.

    The port's early-stop scan, with the refinement and the ladder, also passes, at 18 ms per case, but assumes at most one negative region of $R$. Bjarne kept the full scan.
  - **Datasets.** The branch's Rust datasets were local and never published. Regenerated with the core instead: 5 samples per published `sol-1` well, 9,920 rows in 49 s on 24 cores. The Distributions check fails narrowly: CDF gap 0.021 for PWH against a bound of 0.02 (`develop`: 0.0196). The extra rows are samples the CasADi search does not solve.
  - **Spec and harness gaps** (feature spec, Findings), each with Bjarne's ruling:
    1. `fold-1503`'s reference misses a v1.0.0 root (unstable, 128.0882 bar). Ruling: a known finding; the reference stays unchanged.
    2. The slip law can have three void fractions at sampled states, so the root set contains roots that differ in the branch. Neither backend finds them all, and the spec says nothing about which branch is physical. Ruling: an open model question, noted in `slip.md` and `solution.md`, to be ruled after the plan with the new flow-regime model.
    3. `develop`'s CasADi search misses roots that the core finds. Not changed in Step 8. Both backends are kept, so it is a backlog item (`plans/improvements.md` §2.9).
    4. The case set has no low-$u$ wells with both roots within a scan step of $p_r$, no states with several void fractions, and no case with more than two roots. Its near-fold cases were placed with the old port's grid. Ruling: after the plan.
    5. The Distributions check counts against a solver every row that v1.0.0 failed to solve. Ruling: decided with Step 7's open item on the check's margin.

    `specs/verification.md`'s "Rust returns at most two roots" and `specs/architecture.md`'s "Step 7 adds the `backend` argument" are corrected.

### Step 9 — Port `develop`'s model to the Rust core

- **Goal.** Both backends implement `develop`'s whole model before any new equation goes in.
- **Why before a new equation** (Bjarne, 2026-10-01):
  - `develop`'s model has no second implementation. Its changes since v1.0.0 came in without specs and were specified after the fact in Step 7. They are checked by component vectors, invariants, spot checks, convergence and the property checks, which do not catch a self-consistent assembly error (New model versions). Comparing the two backends' rows at the same states does, and the core is the only second implementation the plan has.
  - Each equation goes into both backends (`specs/goals.md`). With the core at the `v1.0.0` configuration only, Step 10's equation would go into the core as the one option on top of v1.0.0's model, a configuration nobody uses, and the backends could be compared on it only there. The loop Step 10 proves would not be the one v2 work uses.
  - The port is needed for v2.0.0 anyway. It lies on the path to the v2 datasets, the sampling redesign and calibration, which need the full model at the core's speed; a new equation lies on none of them. Moving the port earlier changes the order of the work, not its amount.
- **Work.**
  1. Port every option in `specs/model/` that the core lacks (about 45 equation IDs: geometry, inclination in the slip law, friction from roughness, real gas, black oil and dissolved gas, water, mixing, the energy terms, lift-gas temperature, fixed-rate inflow), each as an enum variant (`specs/architecture.md`, Rust core, design point 1) with `// spec:` tags, checked against `develop`'s vectors (`specs/tools/make_develop_vectors.py`). One PR per feature spec, 001 to 011, in dependency order. Each PR compares the two backends in the `v1.0.0` configuration with its option on, before options are combined.
  2. Make the two changes to the core's method that `specs/architecture.md` gives (Rust core, design points 4 and 5): a temperature solve per cell, because frictional heating (THM-6) and the gravity term (THM-7) make the energy row depend on $p_i$; and the phase rates at each point, because dissolved gas (PVT-OIL-13) makes them vary along the well. Methods and constants are decided under principle 7, with measured gains, as in `specs/features/014-rust-solver.md`. Measure the temperature solve first. It nests a scalar solve inside the pressure solve, and the void fraction is solved inside it at each trial, so it decides whether the core keeps its speed advantage over the CasADi backend (23× at the median in the `v1.0.0` configuration). That advantage is the main reason to have the core.
  3. Check the method's assumptions on the full model by a test or a measured property, not by argument: that [0, 1] brackets the void fraction (it holds while $C_0 \ge 1$ and $v_\infty \ge 0$, which $\cos\theta \in [0, 1]$ in `WellGeometry` keeps); that the momentum row is U-shaped in $p_i$, which the cell's golden-section search assumes; that the energy row has one root in $T_i$ in its bracket; and that the shooting on $p_0$ and its label still hold with fixed-rate inflow (INF-8), where $w_\text{res}$ does not depend on $p_0$.
  4. Check the core against the CasADi backend, since `develop`'s full model has no reference root sets. The checks, strongest first:
     - the rows at the same states agree in every configuration the core covers (design point 2);
     - on a comparison set of wells drawn with the ported sampler, SMP-40 to SMP-44 included (deviated and L-shaped wells, black oil, real gas, roughness, gas lift with lift-gas temperature, fixed-rate inflow), the core finds every root the CasADi backend finds, with the same label, and every root only the core finds zeroes every CasADi row. Step 8 used the same rule. Roots only the core finds are recorded (`plans/improvements.md` §2.9), not fixed. The set lives in `tests/`, not in `verification/`, whose references all come from v1.0.0;
     - the property checks of `tests/test_model_properties.py` pass on the core's roots.

     At states where the slip law has several void fractions, the backends are compared on rows only. Which branch is physical is an open model question (Step 8, gap 2).
  5. Keep the `v1.0.0` configuration exactly as it is: the core's rows match v1.0.0's row vectors, and its verifier report stays at PASS, 100%, with no expected failures.
  6. Bjarne sets the performance target that `specs/features/014-rust-solver.md` deferred to this step.
- **Out of scope.** The batch API (`plans/improvements.md` §4.3), prebuilt wheels, the CasADi search's missed roots (§2.9), a ruling on the void-fraction branch, the four-regime flow-regime model, and making the fluid's fractions parameters (§4.5), which the port makes unnecessary unless datasets are generated on the CasADi backend.
- **Output.** The core implements every option in `specs/model/`, and `SSDFSimulator(wp, backend='rust')` accepts every well. A feature spec, `specs/features/015-rust-develop-model.md`, gives the methods and their measured gains. The comparison set and its tests are committed, and `AGENTS.md` describes the core as covering the whole model.
- **Done when.** The two backends agree on the rows in every configuration and on the comparison set as above, the `v1.0.0` configuration's verifier reports are unchanged, and the traceability test passes.
- **Who.** Agent implements and drafts the feature spec; Bjarne signs off the method changes under principle 7, sets the performance target, and adjudicates where the backends disagree.
- **Status.** Not started. Bjarne signed off the step on 2026-10-01:
  - the reordering, with its change to `specs/goals.md`;
  - the checks against the CasADi backend, with the comparison set in `tests/`; its tolerances come to him during the step;
  - one PR per feature spec, the temperature solve measured first, and the scope above;
  - Finding 4's ruling in `specs/features/014-rust-solver.md`, renumbered so that the case set stays as it is through Step 10.

### Step 10 — Prove the loop: add one new equation

- **Goal.** Show that the repo has reached this plan's end state, in which a new equation is easy to add and test.
- **Work.** Pick one small, self-contained addition, for example another friction-factor correlation as an option in `friction.md`. Take it the whole way through the loop that v2 work will use: feature spec (`specs/features/NNN-<name>.md`: motivation, delta to the component files in `specs/model/` with new equation IDs, acceptance tests, out of scope) → agent plan → implementation in both backends, with `spec:` tags and test vectors in both → spot and property checks for the new option → the two backends compared with the option on, as in Step 9 → verifier (the v1-compatibility configuration unchanged) → human review → merge, with spec and code in the same PR. Record every step that needed more than `AGENTS.md` and the specs describe.
- **Output.** The new equation, merged; a list of the gaps found, each fixed; a "how to add an equation" section in `AGENTS.md`, checked against what was actually done.
- **Done when.** The addition went through the loop with no undocumented steps left, and CI is green, including the traceability test.
- **Who.** Agent implements, working only from `AGENTS.md` and the specs; Bjarne reviews the spec delta.

## After this plan: the path to v2.0.0

v2 is a major release with new datasets and perhaps a new paper. After this plan, the remaining work is:

- **New equations and models** through the loop proven in Step 10, in parallel where modules are independent. Each is an option that is off in the v1-compatibility configuration, so the reference root sets stay valid.
- **Rust core.** The core covers `develop`'s model from Step 9 on. The Python/CasADi simulator stays: the two backends are developed together, each equation in both, and checked against each other (`specs/goals.md`, decided 2026-10-01). Publish prebuilt wheels on PyPI, so installing needs no Rust toolchain.
- **Verifier coverage** (Step 8, gaps 2 and 4): cases for low-$u$ wells whose roots lie within one scan step of $p_r$, states with several void fractions, and wells with more than two roots. The old port, method B, misses the first kind, so the build needs a search that finds them, such as v1.0.0 solved from the Rust core's roots. Then a ruling on the void-fraction branch (`specs/model/slip.md`, Open question), with the new flow-regime model.
- **Sampling redesign.** A new procedure for v2 datasets that differ significantly from v1's; it can address the unrealistic-wells limitation. `specs/sampling.md` gets a new version, and the distribution check keeps covering the v1-compatibility configuration.
- **Calibration.** A spec for `calibration/` (with `plans/improvements.md` §2.8), which must work with the Rust core, then the private real-well accuracy check on the release candidate, with its aggregate results recorded.
- **v2 datasets.** Generated with the redesigned sampler, possibly through a library-level batch API (`plans/improvements.md` §4.3), storing every root with a stability column next to `solution_number` (default loaders return the stable root) unless decided otherwise, and every input needed to re-solve each row. No `nscl-2`, since closed loop is out of scope for v2. The changelog references spec versions and the aggregate accuracy results, and notes that v1's ~280 K TWH spike came from trickle roots.
- **Paper**, if there is one, and the `v2.0.0` tag.

Decisions for v2.0.0, not needed in this plan:

- Which of the model features in `specs/goals.md` (the four v1 limitations with work on `develop`, and the new flow-regime model) are finished in v2.0, and which wait for v2.x.
- Acceptance bound for the private accuracy check relative to v1.0.0's errors on the real well (to settle with calibration).
- v2 dataset schema: every root with a stability label (proposed), or the operating point only.
- Whether the CC BY-NC 4.0 license stays for v2 code (a major version is the natural moment to revisit).

## Sequencing

```
Step 1 ─┬─▶ Step 2 ─▶ Step 3 ───────────┐   ┌─▶ Step 7 ─┐
        │                               ├───┤           ├─▶ Step 9 ─▶ Step 10
        └─▶ Step 4 ─▶ Step 5 ─▶ Step 6 ─┘   └─▶ Step 8 ─┘
```

Steps 2–3 (verifier + data) and Steps 4–6 (specs) can run in parallel after Step 1. Steps 7 (`develop`) and 8 (Rust pilot) can run in parallel once both branches are done. Step 9 waits for both: it ports the model that Step 7 specified to the core that Step 8 built. Step 10 waits for Step 9, so that its equation goes into both backends. Nothing in Steps 7–10 starts until Step 2 is green on v1.0.0.

## Proposed repository layout

```
manywells/
  AGENTS.md                      # agent instructions (CLAUDE.md imports it)
  .github/workflows/tests.yml    # existing CI; Step 2 adds the verifier job
  plans/                         # drafts such as this one, not specs
  specs/
    goals.md                     # Step 1
    constitution.md              # Step 5
    model/                       # Step 4 — one file per model part, mirroring src/manywells/
      README.md                  # scope, ID scheme, how the parts compose, named configurations
      nomenclature.md            # symbols, units, state vector
      balances.md                # mass, momentum, energy balances (continuous form)
      discretization.md          # grid, implicit Euler, residual rows and their order
      solution.md                # root set, stability, operating point
      geometry.md                # trajectory, MD/TVD, inclination
      pvt/
        gas.md                   # ideal and real gas, z-factor
        oil.md                   # dead oil, black oil, dissolved gas
        water.md
        mixture.md               # liquid mixing, viscosity mixing, surface tension
      slip.md                    # drift flux, flow-regime classifier, rise velocities
      friction.md                # fixed f_D, Haaland, Chen
      thermal.md                 # heat loss, frictional heating, gravity term, lift-gas mixing
      inflow.md                  # Vogel, productivity index, fixed rate
      choke.md                   # Simpson, Bernoulli, critical pressure ratio, residual for p_L ≤ p_s
      smoothing.md               # smooth max/min, softmax (part of the model, not just numerics)
    sampling.md                  # Step 4 — v1's sampling procedure, ported to develop's inputs
    discrepancies.md             # Step 4 — resolved paper/code differences
    architecture.md              # Step 6
    verification.md              # Step 2 — checks, tolerances, case set
    features/
      NNN-<name>.md              # Steps 7–10, one per feature (after the fact for changes since v1.0.0)
  verification/                  # manywells-verify package (Step 2)
    data/                        # case set and reference root sets from v1.0.0
  src/manywells/                 # Python package: develop's simulator now; Python bindings over the Rust core in v2
  rust/                          # Rust core
  tests/
```

## Open decisions

Step 1 settled the decisions that were listed here: `specs/goals.md` records them, and the steps above are updated. The technical ones moved to the steps that have the evidence: the initial tolerances to Step 2 item 5, the operating point with two stable roots to `solution.md` in Step 4 (with only unstable roots the well cannot flow, as `solution.md` already states), and the Rust energy balance to Step 8 (v1's implicit-Euler step (19)).
