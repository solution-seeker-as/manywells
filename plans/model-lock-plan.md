# Lock the model for the new sampler — plan

*Draft 2026-10-29. Owner: Bjarne Grimstad. Status: in progress; L4 done, L2 decided but for decision 5, L2 and L3 implemented on branches.*

The next step towards v2 is a new sampling procedure: synthetic wells and data that are more realistic than v1's (`specs/goals.md`, limitation 5). The sampler draws the inputs of a model, so the model should not move under it. This plan brings `develop` to the point where the sampler can be implemented:

- a named, pinned v2 configuration;
- the inputs it takes and the ranges in which each correlation is valid;
- correlations checked against independent sources;
- a settled regression check for the `v1.0.0` configuration;
- no known bugs in the sampler's code path.

It ends when Step L5 is done and Steps L6 and L7 are merged. The redesign itself, with its own spec (a new version of `specs/sampling.md`), comes after it.

## Starting point

- **The v2 foundation plan is done** (`manywells-v2-plan.md`, Steps 1 to 10).
  - `develop`'s model is specified in `specs/model/`, with `v1.0.0` as a named configuration.
  - Both backends implement every option and are compared with each other.
  - The verifier passes both candidates at 100%.
- **Since then:**
  - PVT tables (018), five flow regimes after Hasan et al. (019) and a profile scale for calibration (020) are merged.
  - Bjarne signed off the feature specs 001 to 011, 013, 019 and 020 on 2026-10-29.
  - PR #29 fixed the two open model bugs: the live-oil density (#16, PVT-OIL-15) and the CasADi backend's temperature bounds (#13). Bjarne signed off what it left open the same day (`823de32`).
  - `plans/improvements.md` holds the backlog, merged with the review of 2026-10-16.
- **`develop`'s model is the defaults of the dataclasses** (`specs/model/README.md`). Nothing names it, so a changed default changes the model a dataset is drawn from without any check noticing.
- **The sampler is v1's procedure ported to `develop`** (`specs/sampling.md`): independent draws, extended to the trajectory, roughness and black-oil inputs, with Step 4's starting points (SMP-41 to SMP-43).
  - Its wells are often outside the correlations' ranges (`specs/model/validity.md`, Where the sampler leaves the ranges). 69% of the sampled cases have a gas–oil ratio above 534 Sm³/Sm³, the lower end of gas condensates. 42% lie outside Hasan et al.'s gas–liquid ratios, and 24% have horizontal cells, which no option models with stratified flow.
- **The validity ranges are drafted.** `specs/model/validity.md` (`5294d52`) has a table of every correlation's published range, the range the sampler reaches, and a proposed rule for each. It is a draft for Bjarne's sign-off, with five decisions. `plans/evidence/validity_ranges.py` measures the sampler's side.

## Steps

### Step L1 — Decide which wells and inputs v2 covers

- **Goal.** Fix the scope that the sampler draws from: the kinds of well, and the inputs that describe them.
- **Work.** An agent drafts each question with evidence: how common the case is in realistic wells, and how far today's sampler is from it. Bjarne decides.
  - **Well types.** Black-oil wells, volatile-oil wells, gas-condensate wells, dry-gas wells.
    - The Vazquez–Beggs black oil does not model the oil carried in a rich gas, which drops out as the pressure falls.
    - `validity.md`'s decision 4 recommends letting the black oil take any gas–oil ratio, as today. A gas condensate whose dropout matters then enters as a PVT table from a flash (018). The alternative is to refuse gas–oil ratios above a bound such as 3000 scf/STB. The sampler would then have to draw tables to have gas wells at all.
  - **Horizontal sections** (`validity.md`'s decision 5). The redesign keeps drawing L-shaped wells, with the known limit that no option has stratified flow, or it stops until a model with stratified flow exists, which would need its own feature spec.
  - **Decided 2026-10-29.** The black oil takes any gas–oil ratio (`validity.md`, decision 4), so the model needs no change for gas-rich wells. Which well types the datasets draw, and whether they draw horizontal sections (decision 5), Bjarne decides with the sampling distribution in the redesign. What remains of this step is the inputs below.
  - **Inputs the model lacks:**
    - *Lift-gas injection depth.* Lift gas enters at the bottomhole (`inflow.md`); real gas-lift wells inject through a valve above the perforations.
    - *A gas-rate inflow for gas wells.* The reservoir gas is a fixed fraction of the liquid inflow (INF-4), and every sampled well uses Vogel's inflow, which Vogel derived for solution-gas drive below the bubble point (`validity.md`, note 10).
    - *A tubing diameter that changes along the well.*
    - *Heat capacities that depend on $(p, T)$.*
    - *A heat-transfer coefficient that depends on depth or time.*

  Each input taken into v2 becomes a feature spec and goes through `AGENTS.md`, Adding an equation, before Step L5. One left out is recorded as a non-goal or as v2.x work.
- **Output.**
  - The v2.0 feature split that `specs/goals.md` defers, recorded there (needs sign-off).
  - A feature spec per new input.
- **Done when.** Every question above has a recorded answer, and each new input's feature spec is approved.
- **Who.** An agent drafts; Bjarne decides.

### Step L2 — Validity ranges, with a rule for each

- **Goal.** One table of every correlation's valid range, the range the sampler reaches, and what the model does outside it: the input from which the redesign chooses its ranges.
- **Status.** Drafted and committed on 2026-10-29 (`5294d52`), for Bjarne's sign-off. Bjarne took decisions 1 to 4 as recommended on 2026-10-29 and deferred decision 5 to the sampling redesign; the record, the follow-ups and the refusals are on the branch `validity-refusals`. `specs/model/validity.md` proposes three rules.
  - *Refuse* inputs fixed before the solve where the correlation fails.
  - *Extrapolate* every state-dependent quantity, and any input whose correlation degrades gradually.
  - *Clamp* only a PVT table.

  The `v1.0.0` configuration refuses nothing that v1.0.0 accepts. The measurements cover 1453 operating points of 500 sampled wells.
- **Work.**
  - **Bjarne's five decisions** (`validity.md`, Decisions for Bjarne):
    1. the three rules, with no fourth rule that flags a root outside a range;
    2. refuse API gravities outside 16 to 58, instead of 10 to 40;
    3. refuse a relative roughness above 0.05, where the friction correlations have a pole and a spurious valley;
    4. the black oil's gas–oil ratio, with Step L1;
    5. horizontal sections, with Step L1.
  - **The follow-ups the draft lists:**
    - `pvt/oil.md` and the code's docstring on Beggs–Robinson's range (16 to 58 °API, 70 to 295 °F);
    - Papay's range in `pvt/gas.md` (note 7), and a source for the water viscosity in `pvt/water.md` (note 5);
    - the Safeguards sections of `pvt/oil.md` and `friction.md`;
    - Vazquez's thesis (1976), for the Vazquez–Beggs range, which the draft takes from Beggs and Robinson's until it is checked (note 1).
  - **The refusals of decisions 2 and 3**, in `BlackOilPVT` and `WellProperties`, and in the Rust adapter (`solvers/rust.py`, `Well::new`), with tests in both backends.
  - **One addition:** a note on the CasADi backend's limit from feature 012. Below a geothermal gradient of about 5 K/km, the gravity term can cool a root below the lower temperature bound, so that only the Rust core finds it. The sampler draws 32 to 37 K/km; the redesign should stay well above 5 K/km.
- **Output.** `specs/model/validity.md` and the component files, signed off; the refusals, in both backends.
- **Done when.** Every row has a signed-off rule, the follow-ups are done or recorded as open, and the refusals are tested.
- **Who.** An agent implements; Bjarne decides.

### Step L3 — Check the correlations against independent sources

- **Status.** In progress on the branch `independent-checks` (2026-10-29).
- **Goal.** Independent evidence that each correlation in the v2 configuration is implemented as published. Today the `develop` vectors come from `develop` itself, and the two backends implement the same spec, so agreement between them does not check the values. This step takes over Step V1 of the validation plan's draft (`plans/validation-plan.md`).
- **Work.**
  - **Against `fluids`** (Caleb Bell, MIT licence), run by hand with `uv run --with fluids==1.3.1`, so it is not a dependency. A script in `plans/evidence/` compares:
    - Chen and Haaland (FRIC-4, FRIC-5);
    - the critical pressure ratio (CHK-4);
    - the mixture viscosity (PVT-MIX-9).

    A first run on 2026-10-02 found them identical or within 6e-16. Record in `friction.md` how far Chen and Haaland are from the exact Colebrook solution: 0.32% and 1.4% over Re 4000 to 10⁸ and ε/D 0 to 0.05.
  - **Against published worked examples**, for the correlations `fluids` lacks:
    - Vazquez–Beggs $R_{so}$ and $B_o$;
    - Dranchuk–Abou-Kassem's $Z$;
    - Lee–Gonzalez–Eakin;
    - Beggs–Robinson;
    - the water correlations;
    - Abdul-Majeed's surface tension.

    Sources are the papers (`validity.md`, Sources, lists those in hand), and textbooks such as Ahmed's *Reservoir Engineering Handbook*, ch. 2. The water viscosity has no source in hand yet (`validity.md`, note 5). For the five flow regimes, use Hasan and Kabir (2018) where their worked examples give the void fraction or the pressure gradient. Each vector names its source and is compared at the precision the example prints. Bjarne supplies the sources an agent cannot reach.
  - **The record.**
    - A note next to each checked table, with the commit.
    - A list in `specs/model/README.md` (Test vectors) of the IDs with an independent check, and of those with no independent source.
    - A mismatch is a finding. It is ruled on before Step L5.
- **Output.** The script and its row in `plans/evidence/README.md`; the vectors from worked examples; the notes and the list.
- **Done when.** Every ID the v2 configuration uses is on the list, checked or marked as having no independent source.
- **Who.** An agent runs the checks and drafts the notes; Bjarne signs off the notes and rules on any finding.

### Step L4 — Rule on the several-void-fractions question

- **Status.** Done 2026-10-29. Bjarne ruled that the new sampler uses `develop`'s default only, the five regimes, so the v2 configuration leaves the fitted classifier out, and the question stays open for the fitted classifier alone (`slip.md`, `solution.md`).
- **Goal.** Close the last open model question in `specs/model/`.
- **Status.** `slip.md`, Open question: settled for the five regimes, `develop`'s default, where the slip law has one root in α. It is open for the fitted classifier, which the `v1.0.0` configuration keeps and `develop` has as an option (014, Finding 2; 015, Findings 2 and 3).
- **Work.** Bjarne rules. The options:
  - keep the question open, scoped to the fitted classifier, which the v2 configuration would then exclude;
  - specify which branch is physical, if the fitted classifier stays an option in v2.

  Update `slip.md` and `solution.md`.
- **Done when.** `specs/model/` has no open question that applies to the v2 configuration.
- **Who.** Bjarne.

### Step L5 — Name and pin the v2 configuration

- **Goal.** The lock: a named configuration that the sampler draws in and that a test pins, so that later work on `develop` cannot change it unnoticed.
- **Work** (after Steps L1 to L4, and the feature specs that L1 adds):
  1. In `specs/model/README.md`, a column for the configuration next to `v1.0.0`, one option per part, with the five regimes and not the fitted classifier (L4). The name is Bjarne's to choose; `v2.0` is proposed. `develop`'s defaults can then move on without moving it.
  2. In `configurations.py`:
     - add the name to `CONFIGURATIONS`;
     - have `check(wp, name)` test every option, with exact types (`plans/improvements.md` §2.11, Step L7);
     - add a constructor for its wells if the sampler needs one.
  3. Pin it with row vectors from `develop` in that configuration, in both backends: rows at perturbed states of a few wells, as `specs/model/vectors/v1_rows.json` pins `v1.0.0`.
  4. Write the rule for changing it: a change to a pinned configuration is a new configuration (`v2.1`), never an edit. Datasets already record their configuration and code version (SMP-31).
- **Output.** The configuration in the spec and the code, its vectors and the test, and the rule in `specs/constitution.md` or `specs/model/README.md`.
- **Done when.** `check` passes on wells built in the configuration, both backends' rows match its vectors, and CI runs the test.
- **Who.** An agent implements; Bjarne signs off the configuration and the rule.

### Step L6 — Settle the distribution-check margin

- **Goal.** A regression check for the `v1.0.0` configuration that both backends pass. The redesign must keep that configuration's samples matching the published datasets (`specs/goals.md`).
- **Status.** Open since Step 7 of the v2 plan. Samples regenerated at the stable root for the published `sol-1` wells, 5 per well, give a PWH CDF gap of:
  - 0.0196 with `develop`'s CasADi backend;
  - 0.021 with the Rust core;
  - 0.008 for the reference's own rows.

  The bound is 0.02. Step 8's gap 5 adds that the check counts against a solver every row v1.0.0 failed to solve.
- **Work.**
  - Rerun both regenerations on today's `develop` (`scripts/verification/regenerate_distributions.py`, about 17 minutes on 24 cores).
  - Measure the gap's spread over repeated draws.
  - Bjarne then chooses:
    - a bound set from the measured noise;
    - more samples per well;
    - or a narrow failure accepted and recorded.
- **Output.** The ruling in `specs/verification.md` (needs sign-off); both backends' reports.
- **Done when.** Both backends pass the check as ruled, or the ruling records why one does not.
- **Who.** An agent measures; Bjarne decides.

### Step L7 — Fix the sampler's and the inputs' known bugs

- **Goal.** No known bug on the sampler's code path or in the validation of its inputs.
- **Work.** The backlog items, one PR, each with a test, each moved to the Done table of `plans/improvements.md`:
  - §1.10, `write_dataset` truncates a dotted name;
  - §1.11, the QTOT filter's standard deviation (needs sign-off; or left to the redesign, if it replaces the filter);
  - §1.4, `cpr` silently overwritten;
  - §2.11, the configuration check's `isinstance`;
  - §2.14, integer thermal switches crash the Rust backend;
  - §2.15, NaN slip constants, NaN diameter and unknown friction correlations.
- **Done when.** The PR is merged with the full suite and both verifier candidates passing.
- **Who.** An agent.
- **Status (2026-10-29).** Done but for §1.11, which waits for Bjarne's ruling: §1.4, §1.10, §2.11, §2.14 and §2.15 are fixed, with the backlog's other any-time items (`a1e8da4`).

### Step L8 — Housekeeping

- `plans/validation-plan.md` (untracked): Step V1 moves into Step L3. Commit it with Step V2 alone, the fixed wellhead pressure, which the sampler does not need. Or drop it.
- Issue #2, "Add dataset generation test and bump numpy version to >=2.0", is from 2025. Check it against `tests/test_sampling.py` and `pyproject.toml`, then close or update it.

## Sequencing

```
L1 ─┐
L2 ─┼─▶ L5
L3 ─┤
L4 ─┘
L6, L7, L8: independent, any time
```

Steps L1 to L4 can run in parallel: L1, L2 and L4 are mostly Bjarne's decisions, and an agent can start L3 now. A new input from L1 goes through the feature loop before L5. L6, L7 and L8 can be done at any time. L7 is best done before L5, because L5's check relies on §2.11.

## What the sampler redesign starts from

- the v2 configuration's name, and `check` to confirm that a sampled well is in it;
- the inputs the configuration takes, with their valid ranges and rules (`specs/model/validity.md`): the bounds within which the new sampler draws;
- correlations with an independent check, or a recorded reason why none exists;
- the `v1.0.0` configuration's regression checks: the verifier, and a distribution check that both backends pass;
- the sampler's code without known bugs.

## Out of scope, after this plan

- **Roots each backend's search misses** (`plans/improvements.md` §2.9).
- **Verifier cases** for wells whose roots lie near $p_r$, and for wells with more than two roots.
- **The Rust core's performance target,** and the batch API (§4.3), which belongs with the v2 dataset generation.
- **The fixed-wellhead-pressure boundary** (validation plan, Step V2).
- **Calibration's open items:** the false minimum (§2.10), whether $p_r$ is free, and the default noise.
- **v2.0.0 release work:** prebuilt wheels, the license, the v2 dataset schema.

## Needs Bjarne's sign-off

- the v2.0 feature split and any new input (L1), in `specs/goals.md` and the feature specs;
- `specs/model/validity.md`: its rules and five decisions, and the follow-ups in the component files (L1, L2);
- the independent-check notes, and a ruling on any finding (L3);
- the several-void-fractions ruling (L4): done 2026-10-29;
- the v2 configuration, its name, and the rule for changing it (L5);
- the distribution check's margin (L6);
- the QTOT filter's standard deviation (L7, §1.11).

## Open decisions

- **The inputs v2 adds** (L1): lift-gas injection depth, a gas-rate inflow, a tubing diameter that changes along the well, heat capacities that depend on $(p, T)$, a heat-transfer coefficient that varies.
- **Which well types the datasets draw, and horizontal sections** (`validity.md`'s decision 5): with the sampling distribution, in the redesign.
- **The configuration's name** (L5).
