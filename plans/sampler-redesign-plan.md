# Sampler redesign — plan

*Draft 2026-10-30. Owner: Bjarne Grimstad. Status: draft for Bjarne's review; no step started.*

## Where we are (2026-10-30)

- **Done before this plan:** the model-lock plan (`plans/model-lock-plan.md`). `v2.0.0` is named and pinned, the correlations' validity ranges are recorded, the correlations are checked against independent sources, and both backends pass the `v1.0.0` configuration's regression checks.
- **Next:** Step R1, the interview.

This plan replaces v1's sampling procedure for the v2 datasets (`specs/goals.md`, limitation 5: unrealistic sampled wells). It ends with:

- a new version of `specs/sampling.md`, signed off, that draws realistic wells and operating points in the `v2.0.0` configuration;
- the sampler implemented and tested, with the `v1.0.0` configuration's draws unchanged;
- the v2 datasets generated and checked, ready for the release.

## How the work runs

Bjarne reviews the work at two gates: the interview at the start and the checkpoint before implementation. Between them the agents work on their own.

```
R1 interview ─▶ R2–R7 research, design, prototype ─▶ R8 checkpoint ─▶ R9–R10 features, sampler ─▶ R11 datasets
  (Bjarne)          (agents, on their own)              (Bjarne)            (agents)                  (pilot: Bjarne)
                          ▲                                │
                          └──────── rework ────────────────┘
```

1. **Interview (R1).** Bjarne sets the goals before any research starts. An agent asks, Bjarne answers, and the agent writes the goals into this plan for Bjarne to confirm.
2. **Autonomous work (R2 to R7).** Agents research public sources, design the procedure, draft the feature specs the design needs, and prototype and measure the sampler. They do not stop for decisions (Working assumptions).
3. **Checkpoint (R8).** Bjarne reviews one package: the draft procedure, its evidence and measurements, every working assumption, and a list of the papers and books the agents could not reach. He rules, supplies resources and sends parts back. The reworked parts return to a smaller checkpoint, until he signs off.
4. **Implementation and generation (R9 to R11)**, after the sign-off, through the usual loop (`AGENTS.md`). Bjarne reviews a pilot dataset before the full generation.

### Working assumptions

Between the interview and the checkpoint, an agent does not wait for a ruling. Where a choice is open, it:

- takes the option it recommends;
- records the choice in the decision log (`plans/sampler-redesign/decisions.md`). An entry gives the alternatives and the evidence for each. It says whether the case occurs in realistic wells and in the draws, with numbers, and whether the choice needs sign-off;
- carries on.

Bjarne rules on the whole log at the checkpoint. A ruling that reverses an assumption sends back the work that depends on it.

An agent stops and asks before:

- anything that would touch real-well data (Data and sources);
- work towards a non-goal (`specs/goals.md`);
- changing a decided spec, the verifier or reference data, as opposed to drafting a change;
- implementing a model feature, whose spec Bjarne approves first;
- a run that would load the machine for more than about an hour.

### Where the work lives

- **A branch, `sampler-redesign`**, in a worktree (`.worktrees/sampler-redesign`), holds the drafts:
  - `plans/sampler-redesign/sources.md`: the public sources, each number with its page, table or figure, and the population it describes;
  - `plans/sampler-redesign/sampling-draft.md`: the draft procedure, in the form of `specs/sampling.md`, without SMP IDs;
  - `plans/sampler-redesign/datasets-draft.md`: the dataset design;
  - `plans/sampler-redesign/decisions.md`: the decision log;
  - feature-spec drafts in `specs/features/`, numbered after checking every branch and worktree (`AGENTS.md`);
  - scripts in `plans/evidence/`, each with its row in the README there.
- **The local papers library** holds the PDFs and longer research notes. It is outside the repo, and the repo names no path in it.
- **Handoff.** Each step ends by updating Where we are and the decision log, so that a new session can start from them.
- **The machine is shared.** At most two agents or heavy runs at once, and every Python run under a memory cap and a timeout.

## Data and sources

The rules are in `AGENTS.md` (Data and sources) and the constitution (Conventions, Confidential data and Sources). For this plan they mean:

- **The draws rest on public sources.** Each distribution cites published field statistics, correlations or textbooks (R2). No agent fits a distribution to real-well data, or to results derived from it.
- **Bjarne's private checks enter as his decisions.** He may set goals in R1, and rule in R8, from his private comparisons with field data. Each is recorded as his decision. The research then looks for public support for it; if it finds none, the spec records it as a design choice.
- **Samples for his checks.** R6 and R11 write their samples in the datasets' row format, so that Bjarne can run his private comparisons on them. What comes back into the repo is his ruling.
- **Missing sources.** A source that the agents could not reach and that matters to a decision goes on the checkpoint's resource list (R8), with the decision it bears on and the assumption made in its place.

## Starting point

- **The model is locked.** `v2.0.0` (`specs/model/README.md`, `configurations.py`) fixes every model option and leaves the well's inputs free, a PVT table among them. `specs/model/validity.md` gives each correlation's valid range and rule. The black oil refuses API gravities outside 16 to 58, `WellProperties` refuses a relative roughness above 0.05, and everything else extrapolates.
- **The sampler is v1's procedure** (`specs/sampling.md`, `manywells.sampling`): independent draws, extended to the trajectory and the roughness (SMP-41 to SMP-43). Its wells often leave the correlations' ranges (`validity.md`, Where the sampler leaves the ranges):
  - 69% of the cases have a gas–oil ratio above 534 Sm³/Sm³, the lower end of gas condensates. Their API gravities are 21 to 40, where a condensate's are 40 to 60.
  - 42% lie outside Hasan et al.'s gas–liquid ratios.
  - 24% have horizontal cells, which no option models with stratified flow (decision 5, open).
- **The sampler's structure, apart from its ranges:**
  - $p_r$ and $T_r$ are fixed functions of depth (SMP-13, SMP-14). The gradients have no spread, and there is no overpressure or depletion; $p_r$ varies by ±2% between samples (SMP-21).
  - The fluid is drawn as mass fractions (SMP-7), not as a gas–oil ratio and a water cut, so wells with little oil get very high gas–oil ratios.
  - Every well uses Vogel's inflow (SMP-8), gas wells included (`validity.md`, note 10). Its $w_{l,\max}$ does not depend on the tubing, the depth or the trajectory.
  - Only wells with $f_g \le 0.2$ get gas lift (SMP-17). The lift-gas rate does not scale with the well (SMP-19), and the gas enters at the bottomhole: the injection depth of feature 021 is not drawn.
  - The choke position is uniform on (0.05, 1) in every sample (SMP-18).
  - These inputs are fixed: the surface temperature at 4 °C, the heat capacities, the water density, standard separator conditions, and the lift gas's temperature at $T_r$ (SMP-44).
  - SMP-28 has three filters. Backlog §1.11 left the standard deviation in its QTOT filter to this redesign.
- **The model's gaps that limit the population** (`specs/goals.md`):
  - there is no gas-rate inflow;
  - gas condensates and volatile oils enter only as PVT tables (feature 018), made by a flash (`plans/evidence/pvt_table_eos.py`, NeqSim);
  - there is no stratified flow.

  A tubing diameter that changes along the well, $c_p(p, T)$ and a varying $h$ wait for v2.x.
- **Evidence in hand.**
  - The local papers library holds three papers of field statistics:
    - Nadeau et al. (2023), on reservoir temperatures;
    - Ehrenberg and Nadeau (2005), on depth, porosity and permeability;
    - Valkó and McCain (2003), on the ranges of gas–oil ratio, bubble point, API gravity and gas gravity.
  - It also holds Whitson and Brulé (2000), for the classification of reservoir fluids.
  - A research report in the library proposes correlated, hierarchical draws. It is a list of leads, mostly from web sources, and its numbers need checking against primary sources before use.
- **Tools.**
  - `plans/evidence/validity_ranges.py` measures how far a set of draws leaves the ranges.
  - The Rust core solves a case of `develop`'s model in about 0.15 s (median), and the generator's process pool runs it (`Settings(backend='rust')`).
  - The core's batch API (backlog §4.3) is not built.
- **Constraints carried over.**
  - The `v1.0.0` configuration keeps v1's draws, so that the Distributions check still compares regenerated samples with `sol-1` and `nsol-1` (`specs/verification.md`).
  - Every draw comes from a generator seeded by its place in the dataset (SMP-31).
  - The non-goals limit what may be drawn. A well has one flow path, with no annulus flow or multilaterals. Oil and water are one liquid. Nothing downstream of the choke is modelled. A compositional fluid enters only as a table.
  - The CasADi backend misses roots below a geothermal gradient of about 5 K/km (`validity.md`), so the draws should stay well above it.

## Steps

### Step R1 — Interview: the goals

- **Goal.** Bjarne's goals for the v2 datasets, recorded before any research starts.
- **Work.** An agent asks the questions below in rounds, a few at a time, with what each answer decides. Bjarne answers what he wants to settle now. The agent then writes a Goals section into this plan, and Bjarne confirms it. A question he leaves open becomes a working assumption for R2 to R7.
  - **Round 1: purpose.**
    1. Which machine-learning tasks should the v2 datasets serve first? For example virtual flow metering, surrogate models of the simulator, learning across wells, hybrid physics and machine-learning models, or gas-lift optimisation. The answer decides which inputs must vary, which outputs are the targets, and what must be covered.
    2. Which datasets: stationary (`sol-2`), non-stationary (`nsol-2`), or both? How many wells, and how many samples per well?
    3. Realism or coverage? The population of wells can follow realistic frequencies, or spread evenly over the realistic range with a realistic joint structure. Is a tail of less likely wells wanted for coverage, and should the data mark it?
    4. What would make the v2 datasets a failure? The answer gives the checkpoint's acceptance criteria.
  - **Round 2: the yardstick.**
    5. Which population counts as realistic: global, offshore, onshore or a mix? Recent or older wells? Counted by wells, fields or volumes?
    6. Which sources count: peer-reviewed statistics and textbooks only, or also aggregates from public well databases kept by regulators? May a distribution be fitted to a published per-well table, with only the fitted parameters committed?
    7. How will Bjarne judge realism at the checkpoint: which comparisons and plots? Will he run his private comparisons with field data on the prototype's samples, and in which format does he want them?
    8. Are there goals from Bjarne's private checks that he wants to set? They are recorded as his goals, without their source (Data and sources).
  - **Round 3: the wells.**
    9. Which well types, and in what mix: black oil, volatile oil, gas condensate, wet gas, dry gas? Naturally flowing and gas-lifted wells? Wells with water?
    10. Horizontal sections (`validity.md`, decision 5). Keep drawing L-shaped wells with the known limit, or draw only vertical and deviated wells until a model with stratified flow exists?
    11. Gas wells. A gas-rate inflow as a new well option needs a feature spec for v2.0, and a ruling on whether `v2.0.0`'s check widens. The alternatives are gas wells on today's inflows, or no gas wells.
    12. Fluids as tables. How many distinct fluids should a flash make, from which public compositions? Is NeqSim (Java) acceptable as a build-time tool for the tables?
    13. Which inputs that are fixed today become draws: the surface temperature (onshore, offshore, subsea), the heat capacities, the heat-transfer coefficient, the separator conditions, and the lift gas's injection depth and temperature?
  - **Round 4: variation within a well.**
    14. `sol-2`. Which inputs change between a well's samples (choke, lift-gas rate, separator pressure, reservoir pressure, fractions), and over what ranges? Should the choke be drawn uniformly, or realistically, for example with weight near fully open?
    15. Should one dataset hold wells at different stages of their life, with depleted reservoirs, high water cuts and gas–oil ratios that rise below the bubble point?
    16. `nsol-2`, if wanted. What evolves (pressure decline, water breakthrough, gas–oil ratio), with what time step and lifetime? How do the controls change over time? They stay open loop.
  - **Round 5: the datasets.**
    17. The schema. Every root with its stability label, or the operating point only (`specs/goals.md`, Decided elsewhere)? Columns for the new inputs (trajectory, injection depth, fluid type, $p_r$)? A table of wells?
    18. The filters. Keep SMP-28's filters (the choke's spread, the share of choked samples and the variation of QTOT, §1.11), or replace them? Should failed solves and wells without an operating point be recorded rather than dropped?
    19. Splits and documentation: a train–test split by well, and a datasheet (Gebru et al. 2021)?
  - **Round 6: the process.**
    20. May agents commit and push to `sampler-redesign` during the autonomous work, and open a draft PR for the checkpoint?
    21. How much machine time may the prototypes and the full generation take? Is there a target date?
    22. What form should the checkpoint take: a report page, the drafts on the branch, or both? At what level of detail?
    23. Resources. Should sources already known to be missing be ordered now, such as textbooks on production engineering and on fluid properties?
- **Output.** The Goals section of this plan: the purpose, the yardstick, the population, the variation, the dataset choices made so far, the checkpoint's acceptance criteria, and what the autonomous work may do. A goal that changes `specs/goals.md`, such as gas wells or horizontal sections, is drafted there for sign-off.
- **Done when.** Bjarne confirms the Goals section.
- **Who.** An agent asks and writes; Bjarne answers and confirms.

### Step R2 — The public sources

- **Goal.** For each input the sampler will draw: the public evidence on its distribution and on how it depends on the other inputs, with the population each source describes.
- **Work.**
  - Search the local papers library first, then open-access sources, then the textbooks Bjarne holds.
  - The topics:
    - *depth and trajectory:* the depth of producing reservoirs; the shares of vertical, deviated and horizontal wells; kickoff depth, inclination and horizontal length;
    - *pressure:* the hydrostatic gradient and its spread, overpressure, depletion;
    - *temperature:* the geothermal gradient, surface and seabed temperatures;
    - *fluids:* classification by gas–oil ratio, API gravity and gas gravity; the joint distribution of API gravity, gas–oil ratio, gas gravity and bubble point; water cut; formation-water density; heat capacities;
    - *productivity:* oil wells' productivity index by well type, gas wells' deliverability;
    - *completion:* tubing size against rate (erosional velocity, liquid loading), roughness;
    - *surface:* choke sizes, wellhead and separator pressures, heat-transfer coefficients;
    - *gas lift:* which wells are lifted, lift-gas rates, injection depth;
    - *a well's life:* the decline of rate and pressure, the trends of water cut and gas–oil ratio.
  - For each number, record the source and its page, table or figure. Record the population too: region, onshore or offshore, era, fluid type, and whether it counts wells, fields, reservoirs or volumes. A share by volume needs reweighting before it can become a distribution per well.
  - Flag weak evidence: a single study, a web page, a narrow sample.
  - Keep the resource list: sources that matter but could not be reached.
- **Output.** `sources.md` on the branch, the longer notes in the library, and the resource list.
- **Who.** One research agent. A second pass after R3 fills the gaps the design finds.

### Step R3 — The well draws

- **Goal.** An ordered sequence of conditional draws that gives consistent wells.
- **Work.**
  - Choose the order. A likely one is: well type; depth and trajectory; pressure and temperature; fluid; completion; inflow; choke; surface; gas lift. Each draw is conditioned only on earlier ones, so that the scheme is a sequence of seeded draws.
  - For each draw, state:
    - its distribution, parameters and bounds, in the model's names and units;
    - what it depends on, and its sources;
    - its validity: the rule of each correlation it feeds (`validity.md`), and whether a draw outside a range is truncated, rejected or accepted;
    - how it maps to the inputs of a well in `v2.0.0`, checked by `configurations.check`.
  - Say where a realistic distribution leaves part of the needed coverage thin: flow regimes, choked and unchoked flow, gas lift, the range of rates. The choice goes in the log.
- **Output.** The well sections of `sampling-draft.md`, and entries in the log.
- **Who.** An agent.

### Step R4 — The samples within a well

- **Goal.** How a well's operating points vary: the `sol-2` draws, the `nsol-2` evolution if R1 wants it, and the generation rules.
- **Work.**
  - The draws per sample, and the evolution over time, with sources as in R3.
  - The generation rules: the starts, the filters (with §1.11), and what happens to a failed solve or a well without an operating point.
  - How the filters shift the accepted distributions away from the drawn ones, which R6 measures.
- **Output.** The sample sections of `sampling-draft.md`, and entries in the log.
- **Who.** An agent.

### Step R5 — The model's gaps

- **Goal.** A decision-ready answer for each gap that the population runs into.
- **Work.**
  - *Gas wells:* a feature-spec draft for a gas-rate inflow, following `AGENTS.md`, Adding an equation, step 1. It covers the motivation, the delta to `specs/model/` with new IDs, how the option is off in `v1.0.0`, the acceptance checks and the validity rows. It includes whether `v2.0.0`'s check widens.
  - *Volatile oils and gas condensates:* how to make their tables from public compositions with the flash of feature 018, how many fluids, and the cost of a table per well.
  - *Horizontal sections* (decision 5): their share in realistic wells against the known limit, with measurements from R6.
  - No feature is implemented before Bjarne approves its spec at the checkpoint. Where the model allows it, a prototype measures the effect, for example a fixed-rate inflow in place of a gas-rate one.
- **Output.** Feature-spec drafts, and entries in the log.
- **Who.** An agent.

### Step R6 — Prototype and measure

- **Goal.** Evidence that the draft procedure gives realistic, solvable and varied wells.
- **Work.**
  - Build a prototype of the draft, as a script in `plans/evidence/` or a module on the branch. It draws wells in the `v2.0.0` configuration and solves them with the Rust core, and a subset with the CasADi backend for comparison.
  - Measure:
    - the drawn inputs against the sources: marginals, rank correlations and bounds;
    - the share of wells and samples with an operating point, and why the rest fail. A fall against today's sampler is a solver finding;
    - the exits from the validity ranges (`validity_ranges.py`);
    - the outputs' coverage: flow regimes at the bottom and the top, the share of choked samples, gas lift, and the ranges of rates, pressures and temperatures, before and after the filters;
    - the cost per sample, and for the full datasets;
    - the distributions next to `sol-1`'s and today's sampler's.
  - Write the samples in the datasets' row format, so that Bjarne can run his private comparisons on them.
  - Iterate with R3 to R5 until the measurements and the draft agree.
- **Output.** The scripts and their results in `plans/evidence/README.md`, and figures for the checkpoint.
- **Who.** An agent.

### Step R7 — The dataset design

- **Goal.** What the v2 datasets hold and how they are generated.
- **Work.**
  - the datasets, their names (`manywells-sol-2`, and so on) and sizes;
  - the schema: columns for the new inputs, every root or the operating point only, and a table of wells;
  - the filters, splits and documentation (a datasheet);
  - what the generation needs, measured in R6: the process pool, or the batch API (§4.3).
- **Output.** `datasets-draft.md`, and entries in the log.
- **Who.** An agent.

### Step R8 — The checkpoint

- **Goal.** Bjarne's rulings on the whole design, before anything is implemented.
- **Work.**
  - The package:
    - the goals (R1), and how the draft meets each;
    - the draft procedure (R3, R4) and the dataset design (R7);
    - the sources, and what each number rests on;
    - the measurements (R6), with figures;
    - the feature-spec drafts (R5);
    - the decision log, each entry with its alternatives and a recommendation;
    - the resource list: papers and books the agents could not reach, each with the decision it bears on and the assumption made in its place;
    - what the agents are least sure of.
  - The agent publishes the package as a private report page that links the branch, or in the form R1 chose.
  - Bjarne reads it, runs his private comparisons if he wishes, rules on each entry of the log, supplies resources and sends parts back. Rework goes through the same steps, and the next checkpoint covers only what changed.
- **Done when.** Bjarne signs off the draft procedure and the dataset design, and approves or rejects each feature spec.
- **Who.** An agent prepares; Bjarne decides.

### Step R9 — The model features

- **Goal.** The approved features in the model, in both backends.
- **Work.** Each feature goes through `AGENTS.md`, Adding an equation, in its own worktree. `v2.0.0`'s check widens if Bjarne ruled so.
- **Done when.** Each feature is merged, with the full suite and both verifier candidates passing.
- **Who.** An agent implements; Bjarne signs off each feature's measurements.

### Step R10 — The sampler

- **Goal.** The signed-off procedure in the spec and the code.
- **Work.**
  - A new version of `specs/sampling.md`: the v2 procedure with SMP IDs and a Coverage table, and v1's procedure kept for the `v1.0.0` configuration.
  - The code in `manywells.sampling`: v2's draws next to v1's, seeded as in SMP-31, with each sampled well checked against `v2.0.0`.
  - Tests of the draws' marginals, rank correlations and bounds, and that a rerun gives the same draws.
  - The batch API, if R7 calls for it.
- **Done when.** The full suite, both verifier candidates and the Distributions check pass, with the `v1.0.0` configuration's draws unchanged, and ruff is clean.
- **Who.** An agent implements; Bjarne signs off the spec.

### Step R11 — The datasets

- **Goal.** The v2 datasets, generated and checked.
- **Work.**
  - *Pilot.* Generate a small dataset and check it against the spec's checks. Bjarne reviews it, and may run his private comparisons on it.
  - *Full generation*, with the same checks.
  - The datasheet, the v2 schema in `docs/datasets.md`, and the CHANGELOG entry.
- **Done when.** The datasets pass their checks and Bjarne accepts the pilot. Publishing them is release work and needs his sign-off.
- **Who.** An agent generates; Bjarne reviews.

## Sequencing

```
R1 ─▶ R2 ─▶ R3 ─┬─▶ R6 ─▶ R8 ─▶ R9 ─▶ R10 ─▶ R11
            R4 ─┤    ▲     │
            R5 ─┤    │     └─ rework ─▶ R2–R7
            R7 ─┘    └─ iterate with R3–R5
```

R2 feeds the design steps, and its second pass follows R3. R3 to R5 and R7 can run in parallel, two agents at a time. R6 starts as soon as R3 has a first draft, and iterates with R3 to R5. R9 and R10 can overlap where a feature does not change the draws.

## Needs Bjarne's sign-off

- the goals (R1), and any change they make to `specs/goals.md`;
- the new version of `specs/sampling.md` (R8, R10);
- each feature spec (R5, R8), and whether `v2.0.0`'s check widens;
- `validity.md`'s decision 5, on horizontal sections;
- the dataset design: schema, sizes and filters, §1.11 among them (R7, R8);
- any solver machinery the redesign needs, with its measured gain;
- publishing the datasets (R11).

## Open decisions

These are the interview's questions (R1). Those already open elsewhere:

- the well types and their mix, and horizontal sections (`validity.md`, decision 5);
- a gas-rate inflow, and whether `v2.0.0`'s check widens (`specs/goals.md`);
- the QTOT filter's standard deviation (§1.11, SMP-28);
- the v2 dataset schema: every root or the operating point only (`specs/goals.md`, Decided elsewhere).

## Out of scope

- The `v1.0.0` configuration's sampling, and the published v1 datasets.
- Closed loop: there is no `nscl-2`.
- The model changes that wait for v2.x: a changing tubing diameter, $c_p(p, T)$, and a varying $h$.
- Release work beyond the datasets: prebuilt wheels, the license, and the rule for changing a configuration.
- Roots the searches miss (§2.9). If the prototype shows that they bias a dataset, that is a finding for the checkpoint.
