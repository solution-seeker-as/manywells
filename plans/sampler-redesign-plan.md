# Sampler redesign — plan

*Draft 2026-10-30. Owner: Bjarne Grimstad. Status: R1 to R8 done; the rework and R9's integration done; the second checkpoint ruled on 2026-11-09 and 2026-11-10; R10 done on 2026-11-10 and merged on 2026-11-11; R11's pilot done on 2026-11-11 and handed over for review.*

## Where we are (2026-11-11)

- **Done before this plan:** the model-lock plan (`plans/model-lock-plan.md`). `v2.0.0` is named and pinned, the correlations' validity ranges are recorded, the correlations are checked against independent sources, and both backends pass the `v1.0.0` configuration's regression checks.
- **R1 done:** the interview was held on 2026-10-30, and Bjarne confirmed the Goals below the same day, with the changes they make to `specs/goals.md` and `specs/constitution.md` (Where the goals change the specs).
- **On the branch `sampler-redesign`** (`.worktrees/sampler-redesign`, pushed):
  - **R2 done:** `plans/sampler-redesign/sources-reservoir-fluids.md`, `sources-wells-production.md` and, after Bjarne asked for more real geometries, `sources-geometry.md`.
  - **R3 done:** the well draws in `plans/sampler-redesign/sampling-draft.md`, D-1 to D-36, and the draw-only script `plans/evidence/sampler_draft_draws.py`.
  - **R4 done:** the samples and generation rules in the same draft, and `plans/evidence/sampler_draft_samples.py`.
  - **R5 done:** the draft feature spec `specs/features/022-gas-rate-inflow.md` (Forchheimer in $p^2$, not implemented), the table fluids in `fluids-draft.md` (NeqSim proof tables made), and the proposed edits to decided specs in `spec-changes.md`.
  - **R7 done:** the dataset design in `datasets-draft.md`, with R4's rules revised for redraws.
  - **R6 done:** the prototype `plans/evidence/sampler_prototype.py`, pilots of 400 `sol-2` and 100 `nsol-2` wells of 200 operating points in the interim mapping, and the measurements in `r6-measurements.md`; it changed the draft in D-100 to D-103.
  - **The decision log** `decisions.md` holds D-1 to D-103, the entries of every step.
- **Bjarne's rulings since the Goals** (each in the decision log and the Goals):
  - 2026-10-30: wells with a horizontal drain are modelled from the heel up, with a limit of about 75° (D-10).
  - 2026-11-02: the geometry is an even mix of four classes, and the highly deviated class is half drains to the heel and half high-angle tangents (D-9, D-57). The public evidence shows offshore wells as mostly directional; the even mix is a design choice for coverage.
  - 2026-11-02: "200 samples per well" means 200 operating points; failed draws are redrawn, and every redraw is logged (D-52).
  - 2026-11-05: feature 022 approved as drafted, Forchheimer's law in the pressure squared with constant $a$ and $b$ (D-60, checkpoint J1); implemented in R9.
- **The condensate research is done:** `sources-condensate-tables.md`. Bjarne asked on 2026-11-02 how other simulators represent condensates before ruling on D-64; its options and recommendation are in the checkpoint's K1.
- **R8, the checkpoint, is ready** (2026-11-02): `plans/sampler-redesign/checkpoint.md`, a private report page built from it with R6's figures and the decision log, and a draft PR of the branch into `develop`. It waits for Bjarne's rulings; nothing is implemented before them.
- **The checkpoint's evidence from per-well data and physics** (2026-11-05, at Bjarne's ask: "let data and physics decide"): BOEM's production by completion and ANP's and NSTA's production by well, summarised by setting, years on production and stage of life [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well], and the tubing's erosional capacity (`plans/evidence/erosional_capacity.py`), revise C1, D1, D2, D3, E1, F1 and D-41; the checkpoint's Section 6 now sorts the resources by who can get them.
- **The checkpoint's evidence from open data** (2026-11-06, at Bjarne's go-ahead): BSEE's well tests (choke sizes, tubing and casing pressures), Petrobras' 3W (40 subsea satellite wells' operation: pressures, lift gas, choke histories) and NOAA's ocean atlas (seabed temperatures at wells) [own analysis of BSEE's well tests, Petrobras' 3W Dataset and NOAA's World Ocean Atlas 2023]; they revise E1, F2, D-14, D-15, D-29, D-37 and D-47; USGS's produced-water database, which Bjarne downloaded the same day [own analysis of USGS's produced-water database], revises D-11. The analyses are kept outside the repo (the maintainer's decision, 2026-11-06).
- **Bjarne's rulings on the checkpoint** (2026-11-06), after physics from four books he obtained (Craft and Hawkins's material balance, Joshi's and Brekke and Lien's drains, Elsharkawy's dew points) settled C1, D-21, B1, D-65 and J2: every card and bulk list is ruled and section 4 is signed off in substance, except D2 and D-62, which wait for feature 022. Notable rulings: 0.10 onshore inside the datasets (A1); the wells' shapes uniform over the ranges pooled from all the surveys, drains to 2500 m with friction in place of the cap of 6 (B1, D-104, D-21); the GOR from the material balance (C1); permeability by well type with the erosional capacity as a check (D1); the lift-gas capacity 0.1 to 3.5 kg/s for every well type (F2); the logs in a companion file (I1); condensate tables by option A′ (K1). The rework and R9 can start.
- **The rework and R9, 2026-11-06.**
  - The rework implements the rulings in the draft's scripts and `sampling-draft.md`. Its pilots led to further rulings the same day:
    - one life for both datasets, a `sol-2` well being its `nsol-2` life at one date, with lives running to an exhausted reservoir and stopping at the economic floor (D-105);
    - D2 settled with D-62's non-Darcy coefficient and the gas wells' permeability at 1/14 of the oil wells' (D-62);
    - the lift-gas capacity truncated, not clipped (D-31);
    - a floor on the lift-gas set point (D-47);
    - R4's control change kept for redraws (D-83).
  - Feature 022 (#38) and K1's retrograde tables (#39) are merged into `develop`, and J2's per-fluid offsets are in `table_fluids.py`.
  - R9's integration: the generator's Forchheimer inflow fitted as D-61 says, the per-well NeqSim tables with D-106's repair of their $p/Z$ (needs sign-off), and the rerun of the pilots with them (`r9-measurements.md`). Outcomes, decline and stages hold; the condensates' gas rates fall by a quarter. The cost rises by 70% (`sol-2`) and 110% (`nsol-2`), mostly from a few volatile oils whose temperature solve fails with Joule–Thomson cooling on their tables: a Rust core finding, open.
- **The second checkpoint, 2026-11-09** (`plans/sampler-redesign/checkpoint-2.md`): R9's choices and the open findings.
  - Ruled by Bjarne the same day: D-106's repair of the tables approved (L1); a table fluid's saturation pressure is its flash's, and R3's undersaturation draw applies to black oils only (L2); $b = 0$ in D-61's fit kept as a known limit of the $p^2$ form (L3); the platform oil wells' upper tail kept and measured by well type on R11's pilot (L5).
  - Ruled on 2026-11-10, after a day as working assumptions: a gas well with pressure support waters out at a WGR of 10,000 (L4, D-107); the Rust core's guard against DAK's Joule–Thomson pole on tables is approved (L6; branch `fix/table-jt-temperature-solve`, its PR to `develop`), which makes the volatile oils' solves five times cheaper with the same roots; and the points of R10's spec draft follow the recommendations (`checkpoint-2.md`, section 3).
  - The rerun of both pilots with the tables, D-107 and the guard (`r9-measurements.md`): the supported gas wells decline as the regions' do, and the cost falls back near the rework's apart from building each attempt's table.
- **R10, 2026-11-10** (branch `r10-sampling-spec`, one PR for sign-off):
  - The new `specs/sampling.md`: the v2 procedure as SMP-50 to SMP-117, each rule with its decision entries, its sources and a **Basis** line (data, physics, a design choice or a guess), and a section that tabulates the guesses and the design choices. v1's procedure is unchanged.
  - The sampler in the package, `manywells.sampling.v2` and `manywells.datasets.v2`, and the generator `scripts/data_generation/v2/generate_dataset.py`: each part proved bit for bit against the prototype before the rules changed; an attempt's PVT table read about a hundred times faster.
  - The triage of the guesses (`guesses.md`): data, published sources or physics replaced most of them. R10's working assumptions are D-108 to D-133: a fixed count of numbers per block, the guesses replaced, the schema's choices, and an independent review's fixes. Two of the triage's proposals wait for Bjarne (G7, G30).
  - The approved changes to `specs/goals.md`, `specs/constitution.md`, `specs/model/validity.md` and feature 022.
  - The pilots from the package (`r10-measurements.md`): 400 `sol-2` and 100 `nsol-2` wells; a dataset of 10,000 wells costs about 120 to 150 CPU-hours.
  - Merged into `develop` by Bjarne on 2026-11-11 (#43).
- **R11, 2026-11-11** (branch `r11-pilot`, one PR for sign-off):
  - Working assumptions D-134 to D-140: G7 and G30 taken (the offshore gas wells' line, D-134; the oils' later breakthrough, D-135, and wet starts, D-140), and R10's drops against SMP-106's targets fixed (D-136 to D-139: the lift gas kept on in an `nsol-2` redraw, the limits raised, a sign change next to a failed march counted as no operating point, a small table allowed to lose one pressure).
  - The datasheets of `sol-2` and `nsol-2`, the v2 schema in `docs/datasets.md` and the CHANGELOG entry, drafted.
  - The pilot, 400 `sol-2` and 100 `nsol-2` wells, run twice (`r11-measurements.md`): `sol-2` has no solver failure and drops 4 wells at the limits; the oils' decline lies within the regions' span in the second and third fifths of the life; a dataset of 10,000 wells costs about 110 to 130 CPU-hours. Open: the oils' late life, three tail volatile-oil tables, a near-critical condensate, and the reservoirs' heat and overpressure against Nadeau et al. (2023).
  - Next: Bjarne's review of the pilot and the working assumptions; then the full generation, at his go-ahead.

This plan replaces v1's sampling procedure for the v2 datasets (`specs/goals.md`, limitation 5: unrealistic sampled wells). It ends with:

- a new version of `specs/sampling.md`, signed off, that draws realistic wells and operating points in the `v2.0.0` configuration;
- the sampler implemented and tested, with the `v1.0.0` configuration's draws unchanged;
- the v2 datasets generated and checked, and handed to Bjarne, who validates and checks them privately and publishes them himself.

## Goals

*Set by Bjarne in the interview of 2026-10-30 (R1). A goal he based on his private checks is recorded as his decision (`AGENTS.md`, Data and sources).*

### Purpose

- **Tasks.** The datasets serve four machine-learning tasks: virtual flow metering, surrogate models of the simulator, learning across wells, and production optimisation (choke and gas lift).
- **Datasets.** Two: `manywells-sol-2` (stationary) and `manywells-nsol-2` (non-stationary), each of 10,000 wells with 200 samples per well.
- **Realism and coverage.** A realistic core of wells, and a labelled tail that stretches over the model's valid range for coverage. The research proposes the tail's share; 10% to 20% is the starting point.
- **Coverage first, shape second** (Bjarne, 2026-11-05). First, the distribution covers almost all real wells and holds no unrealistic ones. Second, it puts more weight on typical wells than on rare ones. A sample of real wells, public or private, represents some fields and areas only: the Gulf of Mexico has some very deep wells that are not typical, and no Brazilian field is in hand. So a sample's range is a lower bound on what the draws must cover, and its shape is weak evidence of what is typical.

### Acceptance criteria

The checkpoint judges the draft by these. The datasets fail if they have:

1. **implausible wells:** combinations of depth, pressure, temperature, fluid, completion and rates that a production engineer would reject;
2. **wells that would not be produced in practice,** with a small margin so that marginally economic wells stay in;
3. **missing real conditions:** conditions common in real wells that the draws do not reach, including those of Bjarne's goals below;
4. **solver-shaped distributions:** distributions shaped by solver failures or filters rather than by the physics and the draws.

Criteria 1 and 3 are the coverage of the support, and come first. Within it, the draws should weigh typical wells more than rare ones (Coverage first, shape second, above).

### The yardstick

- **Population.** Realistic means global statistics weighted towards offshore production wells, on platforms and subsea, that flow naturally or on gas lift, counted by wells.
- **Sources.** Published statistics first. Where they are thin, distributions are fitted to public per-well data (paper appendices, regulators' open databases) outside the repo, and only the fitted parameters, the method and the citation are committed.
- **Judging.** Bjarne judges realism at the checkpoint, from the report and the prototype's samples in the datasets' row format, on which he may run his private comparisons. There is no earlier hand-over.

### Bjarne's goals for the population

The research looks for public support for each (Data and sources):

1. depleted reservoirs, with $p_r$ below hydrostatic;
2. separator pressures below 10 bar;
3. overpressured, high-rate wells;
4. some weight at a fully open choke, kept modest: the choke's distribution stays close to uniform, because a wide span is more informative to machine-learning models;
5. a lift-gas rate scaled to the well, and gas lift in gassier wells too.

### The wells

- **Fluid types:** black oil, volatile oil, gas condensate, and wet or dry gas. The research sets the mix from public statistics, with a floor of about 10% of the wells per type.
- **Labels.** Each well carries a label for its fluid type, for its well type (for example offshore with a dry tree, offshore subsea, onshore), for its geometry, and for every other categorical choice, such as its life stage, inflow, gas lift and choke profile.
- **Geometry.** Mostly vertical wells. Some are slightly deviated. Some have a highly deviated bottom section, like an L-shaped well whose last section is not horizontal. Inclinations stay where stratified flow is not expected, since no option models it. The shares and inclinations are checked against public statistics of real wells' geometries, and an agent asks Bjarne where unsure. This answers `validity.md`'s decision 5: no horizontal sections, with the inclination limit set in R3.
  - *Ruled 2026-10-30, after R2:* a well with a horizontal drain is modelled from the heel up, its bottom section at most about 75° from vertical (R3 sets the limit between 70° and 80°), and its inflow stands for the drain. The shares of the geometries wait for more examples of real geometries (Where we are).
  - *Ruled 2026-11-02, after the geometry evidence:* an even mix of four classes, a quarter each: vertical, slightly deviated (5° to 20°), deviated (20° to 60°) and highly deviated bottom (60° to 75°), the last half drains to the heel and half high-angle tangents. This replaces "mostly vertical" (D-9).
- **Gas wells** flow in through a new gas-rate inflow: a feature spec in R5, whose option `v2.0.0`'s check accepts before the release.
- **PVT tables.** Volatile oils and gas condensates enter as tables from a NeqSim flash. The compositions are synthetic, drawn per fluid class from published ranges, plus published example fluids, each cited.
- **New draws:**
  - the surface temperature, by well type;
  - the lift gas's injection depth (feature 021) and temperature;
  - the fluid constants: the water density from salinity, heat capacities by fluid type, and separator conditions.

### Variation within a well

- **`sol-2`.** The controls vary between a well's samples: the choke position (close to uniform, with modest weight at fully open), a lift-gas rate scaled to the well, and the separator pressure. The reservoir pressure and the fractions drift slightly, as in v1 (±2% and ±5%).
- **Life stage.** Each `sol-2` well is drawn at a stage of its life, such as initial, depleted or watered out. Its $p_r$, water cut and gas–oil ratio are consistent with that stage, and the stage is a label.
- **`nsol-2`.** The reservoir evolves with linked trends: pressure decline, water breakthrough, and a gas–oil ratio that rises below the bubble point. The controls follow a mixed history: piecewise constant with occasional changes, as in operations, plus regular test periods that sweep the choke and the lift gas. The controls stay open loop. A well's 200 samples span its life.

### The datasets

- **Roots.** The operating point only; unstable roots are not published.
- **Layout.** Two tables, joined by a well ID. The wells table holds every well input and the labels. The samples table holds each sample's inputs and its operating point.
- **Filters.** No unrealistic wells, and no wells that would not be produced in practice, with a small margin. Failed solves are logged, not silently redrawn. *Ruled 2026-11-02:* each well has 200 operating points; a failed draw is redrawn, and every redraw is logged with its inputs and reason (D-52).
- **Working assumptions** (not asked): a train–test split by well, and a datasheet for each dataset (Gebru et al. 2021).
- **Publishing** (Bjarne, 2026-11-05). Not a step of this plan. The plan hands Bjarne the generated and checked datasets; he validates and checks them privately and publishes them himself.

### The process

- **Git.** Agents commit to `sampler-redesign` as they go, push it, and open a draft PR at the checkpoint. Nothing merges into `develop` without Bjarne's sign-off.
- **Machine.** At most half the machine: 12 cores and 32 GB at a time, two agents at once, and every run capped in memory and time. The full generation (R11) waits for Bjarne's go-ahead.
- **Checkpoint.** A private report page with the plots and the decision log, linking the drafts on the branch and the draft PR.
- **Sources in hand.** Besides the field statistics under Starting point, the local papers library now holds McCain (1990), *The properties of petroleum fluids*; Pedersen, Christensen and Shaikh (2015), *Phase behavior of petroleum reservoir fluids*; Tewari, Dandekar and Moreno Ortiz (2019), *Petroleum fluid phase behavior*; Economides, Hill and Ehlig-Economides (1994), *Petroleum production systems*; and Golan and Whitson (1991), *Well performance*.

### Where the goals change the specs

Drafted with these goals, for sign-off:

- `specs/goals.md`:
  - v2.0's model: the gas-rate inflow, the well types, and no horizontal sections;
  - Compatibility: the two datasets, their size, and the operating point only;
  - Decided elsewhere: the schema's choice of roots.
- `specs/constitution.md`, Dataset reproducibility: a row and its well's row together carry the inputs to re-solve it.
- `specs/model/validity.md`, decision 5: ruled in R3, with the evidence on geometries.

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
  - `plans/sampler-redesign/sources-reservoir-fluids.md` and `sources-wells-production.md`: the public sources, each number with its page, table or figure, and the population it describes;
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

- **The draws rest on public sources.** Each distribution cites published field statistics, correlations or textbooks (R2). No agent fits a distribution to confidential real-well data, or to results derived from it.
- **Open real-well data.** Statistics computed from publishers' open per-well data (BOEM, ANP, NSTA, Sodir, NLOG, BSEE, Petrobras' 3W, NOAA's ocean atlas at well locations, USGS's produced waters) are the maintainer's analyses, kept outside the repo since 2026-11-06. The plans state what they show in words or as comparisons and cite them as [own analysis of …] (`AGENTS.md`, Data and sources).
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
  - Catalogue the books added to the library on 2026-10-30 (Goals, The process).
  - The topics:
    - *well types:* the shares of offshore wells with dry trees, subsea wells and onshore wells that flow naturally or on gas lift, and how their surface conditions differ;
    - *depth and trajectory:* the depth of producing reservoirs; the shares of vertical, deviated and highly deviated wells; kickoff depth, build-up and inclination profiles; the inclination at which stratified flow sets in for the rates in question;
    - *pressure:* the hydrostatic gradient and its spread, overpressure, depletion;
    - *temperature:* the geothermal gradient, surface and seabed temperatures;
    - *fluids:* classification by gas–oil ratio, API gravity and gas gravity; the joint distribution of API gravity, gas–oil ratio, gas gravity and bubble point; compositions by fluid class, for the synthetic compositions; water cut; formation-water salinity and density; heat capacities;
    - *productivity:* oil wells' productivity index by well type, gas wells' deliverability;
    - *completion:* tubing size against rate (erosional velocity, liquid loading), roughness;
    - *surface:* choke sizes, wellhead and separator pressures, heat-transfer coefficients;
    - *gas lift:* which wells are lifted, lift-gas rates, injection depth;
    - *a well's life:* the decline of rate and pressure, the trends of water cut and gas–oil ratio;
    - *economic limits:* the lowest rates at which wells are still produced, by well type, for the filter on wells that would not be produced.
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
- **Done when.** The datasets pass their checks, Bjarne accepts the pilot, and the full datasets are handed to him. Publishing them is not a step: Bjarne validates and checks them privately and publishes them himself.
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
- any solver machinery the redesign needs, with its measured gain.

## Open decisions

The interview settled the questions that were open elsewhere: the well types, horizontal sections in principle, the gas-rate inflow, and the schema's choice of roots (Goals). Still open, for the autonomous work:

- the mix of fluid types and well types (R2, R3);
- the inclination limit that keeps stratified flow out (R3; `validity.md`, decision 5);
- the gas-rate inflow's form, backpressure or Forchheimer (R5);
- the filter on wells that would not be produced, and with it the QTOT filter's standard deviation (§1.11, SMP-28), which lapses if that filter goes (R4);
- the share of the coverage tail (R3).

## Out of scope

- The `v1.0.0` configuration's sampling, and the published v1 datasets.
- Closed loop: there is no `nscl-2`.
- The model changes that wait for v2.x: a changing tubing diameter, $c_p(p, T)$, and a varying $h$.
- Publishing the v2 datasets: Bjarne publishes them himself, after validating and checking them privately (Goals, The datasets).
- Other release work: prebuilt wheels, the license, and the rule for changing a configuration.
- Roots the searches miss (§2.9). If the prototype shows that they bias a dataset, that is a finding for the checkpoint.
