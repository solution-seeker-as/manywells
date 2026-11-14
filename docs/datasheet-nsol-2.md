# Datasheet for `manywells-nsol-2`

*Draft, 2026-11-11 (Step R11 of `plans/sampler-redesign-plan.md`; D-94). Release documentation, for Bjarne's sign-off. `[full generation: …]` marks a number that exists only after the full generation, and `[maintainer: …]` a fact for the maintainer to give.*

*A full generation before the final one is a beta (SMP-50): `manywells-nsol-2-beta1` was the first, and the code now generates `manywells-nsol-2-beta2`, with a choke sized to the well's rate (SMP-77) and a start hurdle (SMP-102), both working assumptions of which Bjarne has taken two values (D-150 to D-153), and with his rulings on the first beta's analysis (D-158 to D-162). This datasheet is filled once, from the final run, for the final dataset, which takes the plain name (his ruling of 2026-11-14); the first beta's numbers are in `plans/sampler-redesign/beta1-analysis.md`, section 11.*

This datasheet follows Gebru et al. (2021), one section per section of theirs (§3.1 to §3.7), each question in bold and its answer after it. The schema, every column with its unit and type, is in [`datasets.md`](datasets.md), v2 datasets; the procedure that draws the wells and samples is `specs/sampling.md`, SMP-50 to SMP-117. The stationary dataset has its own datasheet, [`datasheet-sol-2.md`](datasheet-sol-2.md).

## 1. Motivation

**For what purpose was the dataset created?** For machine-learning research on oil and gas wells. Four tasks were in mind: virtual flow metering, surrogate models of the simulator, learning across wells, and production optimisation with the choke and the lift gas (`plans/sampler-redesign-plan.md`, Goals). `manywells-nsol-2` is non-stationary: each well's 200 samples span its life. The reservoir's pressure, GOR and water follow linked trends, the well's heat transfer, roughness and inflow change with age, the separator steps down, gas lift may start mid-life, and the controls follow an open-loop history with regular test periods. It serves tasks where a well changes over time, such as a flow meter or a model that must follow a well through its life.

**Was there a specific gap that needed to be filled?** The v1 datasets drew their wells from simple independent ranges, and many of the wells were unrealistic (`specs/goals.md`, limitation 5). They were vertical wells with an incompressible liquid, and their samples were whichever root the solver reached, some on the unstable trickle root ([`corrigendum.md`](corrigendum.md)). `manywells-nsol-1` evolved its wells over 500 weekly samples, with random walks of the fractions and a decay of the reservoir pressure (SMP-23 to SMP-27). `manywells-nsol-2` draws realistic wells, with a labelled tail for coverage, gives each a life from first production to the date it stops paying, solves it in the `v2.0.0` configuration, and keeps the operating point only.

**Who created the dataset, and on behalf of which entity?** Solution Seeker AS, with ManyWells' generator. [maintainer: the team or people to name]

**Who funded the creation of the dataset?** [maintainer: funding]

## 2. Composition

**What do the instances represent?** An instance is an operating point of a simulated well at a date of its life: the steady state of the flow from the bottomhole to the choke at that date's controls and reservoir state. The instances are grouped by well and ordered by date. A well is synthetic, drawn by the procedure, and stands for no real well. Each sample is a steady state of its own: nothing is simulated between dates.

**How many instances are there in total?** 2,000,000 samples: 10,000 wells with 200 samples each, of which 160 are operating samples at even dates and 40 are 8 test periods of 5 samples (SMP-97). The wells' surveys and PVT tables are tables of their own: [full generation: rows of `geometry` and of `pvt`, and each file's size]. The companion logs hold [full generation: wells drawn, rows of `redraws`].

**Does the dataset contain all possible instances, or is it a sample from a larger set?** A sample from the synthetic population that the v2 procedure defines. The generator draws wells $k$ = 0, 1, 2, … and keeps, in each of four strata, the wells it accepts until the stratum's quota is reached: 1500 tail wells, and 4675 platform, 2975 subsea and 850 onshore core wells (SMP-110). The population is a design, not a census:

- **A realistic core**, weighted towards offshore production wells on platforms and subsea, that flow naturally or on gas lift, counted by wells (SMP-50). The core wells are platform 0.55, subsea 0.35 and onshore 0.10 by their quotas (SMP-54, SMP-110); the drawn core wells are black oil 0.55, volatile oil 0.10, gas condensate 0.15 and wet or dry gas 0.20 (SMP-55).
- **Choices for coverage** that are not statistics: an even mix of four geometry classes, where real offshore wells are mostly deviated (SMP-59); and volatile oils and condensates over-weighted to about a tenth of the wells each (SMP-55).
- **A coverage tail**: 15% of the wells, by its quota, whose draws are flat over each rule's range so that the tail stretches over the model's valid range (SMP-53). A tail well's life starts from a core state and ends at its tail state (SMP-83).
- **Lives** of 3 to 30 years, log-uniform (SMP-84), each run towards an exhausted reservoir or a watered-out well (SMP-83) and ended where the well stops paying (SMP-101). The trends' forms and ranges are in SMP-84 to SMP-90.

The accepted wells are the drawn ones that are produced from their start and reach 200 operating points within the generation's limits (section 4). [full generation: share of the drawn wells kept; shares of the accepted wells by fluid type and geometry class; the distribution of `T_end`]. `specs/sampling.md`, v2: guesses and design choices, lists every design choice and guess of the procedure.

**What data does each instance consist of?** A row of `samples` holds the well's `ID`, the sample's `slot` and `attempt`, its date (`WEEKS` from first production, the life stage `STAGE`, and in a test its period `TEST` and what it sweeps, `SWEEP`), its inputs (the choke position `CHK`, the pressure downstream of the choke `PDC`, the lift-gas rate `WGL`, the reservoir pressure `PRES`, `GOR`, `WLR`, whether lift is installed `LIFT`, the heat-transfer coefficient `HTC`, the roughness `ROUGH` and the inflow's coefficients) and its operating point: the pressures and temperatures at the bottomhole and the wellhead, the mass and standard volume rates of gas, oil, water and lift gas, the reservoir stream's mass fractions, whether the choke is choked, and the flow regimes at both ends. The well's row of `wells` holds every input that does not vary, the life's parameters, its labels and its split.

**Is there a label or target associated with each instance?** No single target. The operating point's columns are the natural targets: the rates for virtual flow metering, every column for a surrogate model. Each sample carries its life stage, and each well a label for every categorical choice (SMP-52): coverage tail, well type, fluid type and class, drive, pressure class, saturation at the start, geometry class and family, PVT model and fluid source, inflow, tubing, choke, lift and the rule that started it. Users can group, reweight or drop wells by them.

**Is any information missing from individual instances?** Nothing that applies. `TEST` and `SWEEP` are null for an operating sample, and an inflow coefficient that the well's inflow model does not use is null. In `wells` a column that does not apply to a well is null, never a sentinel: the injection depth of a well without a lift design, the table columns of a black oil, a gas well's productivity index.

**Are relationships between individual instances made explicit?** Yes. `ID` joins the samples to their well's row, its survey in `geometry` and, through `pvt_table_id`, its tables in `pvt`. A well's slots are in date order, with `WEEKS` as the date, and a test's five samples share their date and reservoir state (SMP-97). The controls of the operating samples follow set points that hold between changes (SMP-98). `draw_index` joins the wells to the logs.

**Are there recommended data splits?** Yes, by well (SMP-110, D-90). `wells.split` puts 20% of the wells in `test` and 80% in `train`, stratified by fluid type, well type, geometry class and coverage tail: within each stratum every fifth well, in an order drawn from the dataset's seed, is `test`. A well's samples, survey and tables follow its split. A split by row would put a well's samples in both parts and measure interpolation within wells, not the step to new wells that learning across wells needs. Tail wells are in both parts; a user who wants the core only filters on `coverage_tail`. A task that follows one well, or forecasts, splits within the well by date, for example the first 80% of its life against the rest; the dataset does not fix that split. [full generation: wells in train and in test]

**Are there any errors, sources of noise, or redundancies in the dataset?**

- **No measurement noise.** The values are the model's operating point, solved by the Rust core to its tolerance. The GOR and the water drift by ±5% between dates around their trends, and the reservoir pressure follows its trend without noise (SMP-100); this is variation of the inputs, not noise on the outputs. A task that stands in for field data adds its own noise.
- **The model's limits** (`specs/goals.md`, v2.0's model and Non-goals): steady state at every date; oil and water as one mixed liquid; no stratified flow, so no cell is more than 75° from vertical, and a horizontal drain is modelled from its heel up, its length entering the inflow only (SMP-61); one tubing diameter along the well; heat capacities that do not depend on pressure and temperature; a heat-transfer coefficient constant along the well at each date; nothing downstream of the choke.
- **Correlations outside their ranges.** Where an input lies outside a correlation's published range, the model extrapolates (`specs/model/validity.md`). [full generation: share of the samples outside each correlation's range]. The volatile oils' and condensates' tables come from an untuned equation of state (SMP-112), interpolated bilinearly (`specs/features/018-pvt-tables.md`). [full generation: share of the samples with a point off its table's grid]
- **The procedure's guesses.** Where a value rests on neither data nor physics, `specs/sampling.md` says so in its Basis lines and lists it under v2: guesses and design choices; for the life, among others, the water's breakthrough and the set points' steps (SMP-86, SMP-98).
- **The redraws' conditioning.** A date's draw without an operating point is redrawn with the date kept and the separator pressure and controls changed for that slot only (SMP-105). The samples are the draws at which the well flows, and a redrawn slot's controls depart from the set-point history for that slot. The redraw log holds the draws that were redrawn. [full generation: redraws per sample, by outcome and reason; the shift between the drawn and the accepted inputs]
- **Several stable roots.** Where a root set has more than one stable root, the operating point is the one with the lowest bottomhole pressure (SOL-6); `drawn_wells.n_several_stable` counts such samples. [full generation: their share]
- **No redundancies by design**, though a test's five samples share their date and reservoir state.

**Is the dataset self-contained?** Yes. It carries each well's survey and each table fluid's PVT tables, so a sample re-solves from its rows with ManyWells at the code version in `metadata.json`, without NeqSim (SMP-110). `manywells.datasets.v2.sample_case` builds a sample's well and boundary conditions from its rows, and `resolve_sample` solves it again: on the generator's platform the features are the sample's to the bit, elsewhere within the verifier's `tol_x` applied to them (D-92). [full generation: result of the re-solve check]

**Does the dataset contain data that might be considered confidential?** No. Every well and sample is drawn and simulated. The procedure's distributions were set with published statistics and with the maintainer's analyses of open per-well data from public databases, which are kept outside the repo and cited in `specs/sampling.md`, v2: references. The dataset holds no real-well data.

**Does the dataset contain data that might be offensive, or does it relate to people?** No. It does not relate to people, so Gebru et al.'s questions on people do not apply.

## 3. Collection process

**How was the data associated with each instance acquired?** By simulation. Each well is drawn by the v2 procedure (SMP-50 to SMP-117), built in the `v2.0.0` configuration with 100 cells (SMP-111), and each sample is solved by the Rust core, `SSDFSimulator(wp, backend='rust')`, whose operating point is the root set's stable root (SOL-4). A well's choke is sized to its rate at its life's start, so that the fully open choke takes 2% to 45% of the pressure upstream of it there, and is the well's for its life (SMP-77). The life (SMP-83 to SMP-91) is scanned at 21 dates for its best rate, which ends the life where the well stops paying and may start gas lift where natural flow ends (SMP-101). Its 200 slots span the life to that end (SMP-97); the choke and the lift gas follow an open-loop history of set points (SMP-98), with 8 test periods that sweep the choke or the lift gas (SMP-99), and the GOR and the water drift between dates (SMP-100).

**What mechanisms or procedures were used, and how were they validated?** ManyWells' generator, `scripts/data_generation/v2/generate_dataset.py` with `manywells.sampling.v2`, and NeqSim 3.4.0 for the volatile oils' and condensates' tables (SMP-112 to SMP-116). The model is specified in `specs/model/`. The Rust core is checked against the CasADi backend, and in the `v1.0.0` configuration by the verifier against v1.0.0's roots (`specs/verification.md`). The draws and the life are tested for their bounds, shares, marginals and reproducibility (`specs/sampling.md`, v2: checks; `tests/test_sampling_v2_*.py`). The generation is checked for the counts (10,000 wells, 200 samples each, every slot once), every well against `configurations.check(wp, 'v2.0.0')`, the re-solve of a seeded sample of rows read back from the files, and the split's shares per stratum (`plans/sampler-redesign/datasets-draft.md`, Checks at generation). [full generation: the results of these checks]

**What was the sampling strategy?** The designed population of section 2, drawn block by block in a fixed order (SMP-51), each block from a generator seeded by the dataset's seed and the draw's place (SMP-109); the life and each well's plan of controls have generators of their own. A run gives the same wells whatever the order of the wells or the number of processes. The dataset's seed is its own, so its wells are independent of `manywells-sol-2`'s. [full generation: the seed]

**Who was involved in the data collection process?** No people were measured or surveyed. The maintainer designed the procedure and ruled on its rules, with coding agents drafting and implementing it (`plans/sampler-redesign-plan.md`). The generation runs unattended.

**Over what timeframe was the data collected?** The procedure was written from 2026-10-30 to 2026-11-10 (Steps R1 to R10 of the plan). [full generation: the run's dates, machine, processes and CPU hours]. The instances have no calendar time: `WEEKS` counts from each well's first production.

**Were any ethical review processes conducted?** No: the data involves no people.

## 4. Preprocessing, cleaning and labelling

**Was any preprocessing, cleaning or labelling of the data done?** No value is changed after its solve. Wells, lives and draws are selected:

- **The start hurdle** (SMP-102). A platform or subsea well whose design rate in oil equivalent lies below its type's start hurdle, 20 and 500 Sm³/d, would not be drilled, and is dropped before any solve.
- **The producibility filter and the life scan** (SMP-101 to SMP-103). A well whose best rate at its start, at a fully open choke and at three lift-gas rates for a lifted well, is below half its economic floor, or an oil whose WLR is above 0.98, would not be produced and is dropped. A natural oil well that cannot flow at its start first takes its life's lift design from the start (`lift_start_rule` `no_natural_flow`). Along the life, the scan ends the life at its last point that passes before the first that fails, and may start lift at the point before a natural well's best rate fails (SMP-90; `lift_start_rule` `end_of_natural_flow`). A life that ends at its start is dropped (`short_life`).
- **The fluid's class** (SMP-115). A volatile oil or condensate whose built fluid is not of its class, by the GOR that the field would report, is drawn again, at most 9 times (`fluid_redraws`); a well whose reservoir holds no fluid of its class is dropped (`not_producible`, with the reason `no_fluid_of_class`).
- **Redraws** (SMP-104, SMP-105). A slot's draw without an operating point is redrawn, at the same date, until the slot has one.
- **Limits** (SMP-106). A well is dropped, and the next drawn well takes its place, when it needs more than 1000 redraws or has more than 10 solver failures, its inputs fail `configurations.check`, or its table cannot be built. The drops are reported by group, and the final dataset is held to thresholds on the wells past the filters: at most 2% dropped at the limits, and at most 10% in any tenth of a drawn input and in any group of 100 drawn wells or more (SMP-106): [full generation: wells dropped at the limits by fluid class, well type and lift rule; the shift of each drawn input between the drawn and the kept wells]
- **Dead dates** (SMP-106). A date at which the well has no operating point in 100 draws of its controls does not drop it: a natural well with a lift design is lifted from that date (`lift_start_rule` `no_natural_flow`), and any other well's life ends at the date before (`T_end` earlier than `T_scan`). The slots are then planned and solved again; a well with a dead date after five such changes is dropped as `dead_date`. [full generation: wells whose lift a dead date started; wells whose life one ended, and the share of the life cut]

Nothing else drops a well or a sample: v1's filters (SMP-28) do not apply, and a small rate stays a sample. The labels are drawn or set before the solves, except the lift's labels that the life scan sets (SMP-52, SMP-101). [full generation: wells dropped by outcome, `dead_date` and `short_life` among them; share of the attempts redrawn]

**Was the "raw" data saved in addition to the preprocessed data?** Yes, as two logs in a companion folder next to the dataset (SMP-108). `drawn_wells` holds every drawn well, accepted or not, with its inputs, labels, outcome, best rate, life scan and counts of attempts; `redraws` every attempt without an operating point and every attempt of a dropped well, with its inputs, outcome, reason and root counts. With `samples` they hold every draw of the accepted wells, so the shift that the redraws cause can be measured input by input. The search counts and times of the attempts stay with the run.

**Is the software that was used available?** Yes: ManyWells at the commit in `metadata.json` (`code.commit`), under CC BY-NC 4.0, and NeqSim 3.4.0, an open-source library, for the tables. A dataset to publish is written only from a working tree without changes, so that its commit reproduces it (SMP-110).

## 5. Uses

**Has the dataset been used for any tasks already?** Not at the time of writing. The ManyWells paper's examples (`scripts/ml_examples/`) use the v1 datasets.

**Is there a repository that links to papers or systems that use the dataset?** [maintainer: where uses will be listed]

**What other tasks could the dataset be used for?** Besides the four tasks: virtual flow metering over a well's life, as its reservoir and completion change; surrogate models of the simulator; learning across wells, which the split by well tests; and production optimisation, where the test periods give each well its choke performance curve, and a lifted well its lift performance curve, at several dates of its life (SMP-99). Also detecting a change of state, such as water breakthrough or the start of lift, and following a model's drift and recalibration over a life.

**Is there anything about the composition or collection that might impact future uses?**

- The mix is a design (section 2), and so are the lives: their trends are rules with drawn parameters, not fits to wells (SMP-84 to SMP-90). Statistics over the dataset are statistics of the design, not of real wells: reweight or filter by the labels.
- The redraws condition the samples on the well flowing, and a redrawn slot's controls depart from the history; the logs let a user measure the shift.
- The controls are open loop: they never follow a solve (SMP-98).
- The data has no noise, no instrument errors, no missing measurements and no transients, unlike field data.

**Are there tasks for which the dataset should not be used?**

- Estimating statistics of real wells or fields, such as decline rates or lives: the shares and the trends are a design.
- Transient behaviour, such as heading, slug dynamics, start-up or the response between dates: each sample is a steady state.
- Closed-loop control: the set points do not respond to the well.
- Anything downstream of the choke, such as flowlines, risers, separators and networks: `PDC` is a boundary condition.
- Stratified or horizontal flow, and oil–water slip.
- Predicting a real well without calibrating the model to it ([`calibration.md`](calibration.md)).

## 6. Distribution

**Will the dataset be distributed to third parties?** Yes, publicly.

**How will it be distributed?** On Hugging Face, in the repository `solution-seeker-as/manywells`, as parquet files under `v2/` with a config for each table and the unchanged v1 datasets next to them (D-86). The samples will load with `datasets.load_dataset('solution-seeker-as/manywells', name='manywells-nsol-2')`. This datasheet goes on the dataset card. [maintainer: a DOI, if one is minted]

**When will it be distributed?** After the maintainer has validated and checked it (`plans/sampler-redesign-plan.md`, Step R11). [maintainer: the release date]

**Under what license?** CC BY-NC 4.0, as the code and the v1 datasets (`README.md`, License). [maintainer: how to cite the v2 datasets]

**Have any third parties imposed restrictions?** None known. The published example fluids' compositions are cited with their sources (SMP-114).

**Do any export controls or other regulatory restrictions apply?** None known.

## 7. Maintenance

**Who will support, host and maintain the dataset?** Solution Seeker AS. Hugging Face hosts it.

**How can the owner be contacted?** Through the maintainer named in ManyWells' `pyproject.toml`, or an issue in the GitHub repository `solution-seeker-as/manywells`.

**Is there an erratum?** Errata go in [`corrigendum.md`](corrigendum.md), as the v1 datasets' do.

**Will the dataset be updated?** No. A published dataset is never modified: a fix is a new dataset or a note in the corrigendum (`specs/constitution.md`, Dataset reproducibility).

**Will older versions continue to be hosted?** The v1 datasets stay published, unchanged, next to the v2 ones (D-86). A later version of this dataset is a new dataset.

**Is there a mechanism for others to extend or contribute?** Through the GitHub repository. The generator, the seeds and the code version are public, so anyone can regenerate the dataset, or draw another with a seed of their own. A change of the procedure is a change of `specs/sampling.md`, which needs the maintainer's sign-off.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H. and Crawford, K. (2021), "Datasheets for datasets", *Communications of the ACM* 64(12), 86–92; arXiv preprint 1803.09010v8, §3.1 to §3.7, pp. 4–10.
