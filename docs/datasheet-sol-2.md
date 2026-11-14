# Datasheet for `manywells-sol-2`

*Draft, 2026-11-11 (Step R11 of `plans/sampler-redesign-plan.md`; D-94). Release documentation, for Bjarne's sign-off. `[full generation: …]` marks a number that exists only after the full generation, and `[maintainer: …]` a fact for the maintainer to give.*

*A full generation before the final one is a beta (SMP-50): `manywells-sol-2-beta1` was the first, and the code now generates `manywells-sol-2-beta2`, with a choke sized to the well's rate (SMP-77) and a start hurdle (SMP-102), both awaiting Bjarne's ruling (D-150 to D-153). This datasheet is filled for a beta first, and again for the final dataset, which takes the plain name.*

This datasheet follows Gebru et al. (2021), one section per section of theirs (§3.1 to §3.7), each question in bold and its answer after it. The schema, every column with its unit and type, is in [`datasets.md`](datasets.md), v2 datasets; the procedure that draws the wells and samples is `specs/sampling.md`, SMP-50 to SMP-117. The non-stationary dataset has its own datasheet, [`datasheet-nsol-2.md`](datasheet-nsol-2.md).

## 1. Motivation

**For what purpose was the dataset created?** For machine-learning research on oil and gas wells. Four tasks were in mind: virtual flow metering, surrogate models of the simulator, learning across wells, and production optimisation with the choke and the lift gas (`plans/sampler-redesign-plan.md`, Goals). `manywells-sol-2` is stationary: each well's 200 samples are operating points around one date of its life, at varied choke openings, lift-gas rates and separator pressures, with a small drift of the reservoir.

**Was there a specific gap that needed to be filled?** The v1 datasets drew their wells from simple independent ranges, and many of the wells were unrealistic (`specs/goals.md`, limitation 5). They were vertical wells with an incompressible liquid, and their samples were whichever root the solver reached, some on the unstable trickle root ([`corrigendum.md`](corrigendum.md)). `manywells-sol-2` draws realistic wells, with a labelled tail for coverage, solves them in the `v2.0.0` configuration, and keeps the operating point only.

**Who created the dataset, and on behalf of which entity?** Solution Seeker AS, with ManyWells' generator. [maintainer: the team or people to name]

**Who funded the creation of the dataset?** [maintainer: funding]

## 2. Composition

**What do the instances represent?** An instance is an operating point of a simulated well: the steady state of the flow from the bottomhole to the choke at one choke opening, lift-gas rate, separator pressure and reservoir state. The instances are grouped by well. A well is synthetic, drawn by the procedure, and stands for no real well.

**How many instances are there in total?** 2,000,000 samples: 10,000 wells with 200 samples each. The wells' surveys and PVT tables are tables of their own: [full generation: rows of `geometry` and of `pvt`, and each file's size]. The companion logs hold [full generation: wells drawn, rows of `redraws`].

**Does the dataset contain all possible instances, or is it a sample from a larger set?** A sample from the synthetic population that the v2 procedure defines. The generator draws wells $k$ = 0, 1, 2, … and keeps the first 10,000 it accepts (SMP-110). The population is a design, not a census:

- **A realistic core**, weighted towards offshore production wells on platforms and subsea, that flow naturally or on gas lift, counted by wells (SMP-50). The drawn core wells are platform 0.55, subsea 0.35 and onshore 0.10 (SMP-54), and black oil 0.55, volatile oil 0.10, gas condensate 0.15 and wet or dry gas 0.20 (SMP-55).
- **Choices for coverage** that are not statistics: an even mix of four geometry classes, where real offshore wells are mostly deviated (SMP-59); volatile oils and condensates over-weighted to about a tenth of the wells each (SMP-55); and the life stages initial, mid-life and late at 0.25, 0.40 and 0.35 (SMP-56).
- **A coverage tail**: 20% of the drawn wells, whose draws are flat over each rule's range so that the tail stretches over the model's valid range (SMP-53). The spec expects about 15% of the accepted wells, since more tail wells are dropped. [full generation: the tail's share of the accepted wells]

The accepted wells are the drawn ones that are produced and reach 200 operating points within the generation's limits (section 4). [full generation: share of the drawn wells kept; shares of the accepted wells by fluid type, well type, geometry class and tail]. `specs/sampling.md`, v2: guesses and design choices, lists every design choice and guess of the procedure.

**What data does each instance consist of?** A row of `samples` holds the well's `ID`, the sample's `slot` and `attempt`, its inputs that vary within the well (the choke position `CHK`, the pressure downstream of the choke `PDC`, the lift-gas rate `WGL`, the reservoir pressure `PRES`, `GOR` and `WLR`) and its operating point: the pressures and temperatures at the bottomhole and the wellhead, the mass and standard volume rates of gas, oil, water and lift gas, the reservoir stream's mass fractions, whether the choke is choked, and the flow regimes at both ends. The well's row of `wells` holds every input that does not vary, its labels and its split.

**Is there a label or target associated with each instance?** No single target. The operating point's columns are the natural targets: the rates for virtual flow metering, every column for a surrogate model. Each well carries a label for every categorical choice (SMP-52): coverage tail, well type, fluid type and class, drive, life stage and state, pressure class, saturation, geometry class and family, PVT model and fluid source, inflow, tubing, choke and lift. Users can group, reweight or drop wells by them.

**Is any information missing from individual instances?** Nothing in `samples`. In `wells` a column that does not apply to a well is null, never a sentinel: the injection depth of a natural well, the table columns of a black oil, a gas well's productivity index.

**Are relationships between individual instances made explicit?** Yes. `ID` joins the samples to their well's row, its survey in `geometry` and, through `pvt_table_id`, its tables in `pvt`. A well's samples share every input in `wells`, and are independent draws around its nominal state: `slot` has no order in time. `draw_index` joins the wells to the logs.

**Are there recommended data splits?** Yes, by well (SMP-110, D-90). `wells.split` puts 20% of the wells in `test` and 80% in `train`, stratified by fluid type, well type, geometry class and coverage tail: within each stratum every fifth well, in an order drawn from the dataset's seed, is `test`. A well's samples, survey and tables follow its split. A split by row would put a well's samples in both parts and measure interpolation within wells, not the step to new wells that learning across wells needs. Tail wells are in both parts; a user who wants the core only filters on `coverage_tail`. A task on one well splits its slots as it chooses. [full generation: wells in train and in test]

**Are there any errors, sources of noise, or redundancies in the dataset?**

- **No measurement noise.** The values are the model's operating point, solved by the Rust core to its tolerance. A task that stands in for field data adds its own noise.
- **The model's limits** (`specs/goals.md`, v2.0's model and Non-goals): steady state; oil and water as one mixed liquid; no stratified flow, so no cell is more than 75° from vertical, and a horizontal drain is modelled from its heel up, its length entering the inflow only (SMP-61); one tubing diameter along the well; heat capacities that do not depend on pressure and temperature; a heat-transfer coefficient constant along the well; nothing downstream of the choke.
- **Correlations outside their ranges.** Where an input lies outside a correlation's published range, the model extrapolates (`specs/model/validity.md`). [full generation: share of the samples outside each correlation's range]. The volatile oils' and condensates' tables come from an untuned equation of state (SMP-112), interpolated bilinearly (`specs/features/018-pvt-tables.md`). [full generation: share of the samples with a point off its table's grid]
- **The procedure's guesses.** Where a value rests on neither data nor physics, `specs/sampling.md` says so in its Basis lines and lists it under v2: guesses and design choices.
- **The redraws' conditioning.** A draw without an operating point is redrawn from scratch (SMP-105), so the samples follow the drawn distribution conditioned on the well flowing. The redraw log holds the draws that were redrawn. [full generation: redraws per sample, by outcome and reason; the shift between the drawn and the accepted inputs]
- **Several stable roots.** Where a root set has more than one stable root, the operating point is the one with the lowest bottomhole pressure (SOL-6); `drawn_wells.n_several_stable` counts such samples. [full generation: their share]
- **No redundancies by design**: each sample is a draw of its own.

**Is the dataset self-contained?** Yes. It carries each well's survey and each table fluid's PVT tables, so a sample re-solves from its rows with ManyWells at the code version in `metadata.json`, without NeqSim (SMP-110). `manywells.datasets.v2.sample_case` builds a sample's well and boundary conditions from its rows, and `resolve_sample` solves it again: on the generator's platform the features are the sample's to the bit, elsewhere within the verifier's `tol_x` applied to them (D-92). [full generation: result of the re-solve check]

**Does the dataset contain data that might be considered confidential?** No. Every well and sample is drawn and simulated. The procedure's distributions were set with published statistics and with the maintainer's analyses of open per-well data from public databases, which are kept outside the repo and cited in `specs/sampling.md`, v2: references. The dataset holds no real-well data.

**Does the dataset contain data that might be offensive, or does it relate to people?** No. It does not relate to people, so Gebru et al.'s questions on people do not apply.

## 3. Collection process

**How was the data associated with each instance acquired?** By simulation. Each well is drawn by the v2 procedure (SMP-50 to SMP-117), built in the `v2.0.0` configuration with 100 cells (SMP-111), and each sample is solved by the Rust core, `SSDFSimulator(wp, backend='rust')`, whose operating point is the root set's stable root (SOL-4). A `sol-2` well is its life (SMP-83) at one date, drawn within its life stage (SMP-92). Its choke's bean is sized to its rate at that date, so that the fully open choke takes 2% to 45% of the pressure upstream of it (SMP-77). Its samples vary the choke (SMP-93), the lift gas (SMP-94) and the separator pressure (SMP-95), and drift the reservoir pressure by ±2% and the GOR and the water by ±5% (SMP-96).

**What mechanisms or procedures were used, and how were they validated?** ManyWells' generator, `scripts/data_generation/v2/generate_dataset.py` with `manywells.sampling.v2`, and NeqSim 3.4.0 for the volatile oils' and condensates' tables (SMP-112 to SMP-116). The model is specified in `specs/model/`. The Rust core is checked against the CasADi backend, and in the `v1.0.0` configuration by the verifier against v1.0.0's roots (`specs/verification.md`). The draws are tested for their bounds, shares, marginals and reproducibility (`specs/sampling.md`, v2: checks; `tests/test_sampling_v2_*.py`). The generation is checked for the counts (10,000 wells, 200 samples each, every slot once), every well against `configurations.check(wp, 'v2.0.0')`, the re-solve of a seeded sample of rows read back from the files, and the split's shares per stratum (`plans/sampler-redesign/datasets-draft.md`, Checks at generation). [full generation: the results of these checks]

**What was the sampling strategy?** The designed population of section 2, drawn block by block in a fixed order (SMP-51), each block from a generator seeded by the dataset's seed and the draw's place (SMP-109). A run gives the same wells whatever the order of the wells or the number of processes. The dataset's seed is its own, so its wells are independent of `manywells-nsol-2`'s. [full generation: the seed]

**Who was involved in the data collection process?** No people were measured or surveyed. The maintainer designed the procedure and ruled on its rules, with coding agents drafting and implementing it (`plans/sampler-redesign-plan.md`). The generation runs unattended.

**Over what timeframe was the data collected?** The procedure was written from 2026-10-30 to 2026-11-10 (Steps R1 to R10 of the plan). [full generation: the run's dates, machine, processes and CPU hours]. The instances have no calendar time: a well's date, `life_date`, is in years from its first production.

**Were any ethical review processes conducted?** No: the data involves no people.

## 4. Preprocessing, cleaning and labelling

**Was any preprocessing, cleaning or labelling of the data done?** No value is changed after its solve. Wells and draws are selected:

- **The start hurdle** (SMP-102). A platform or subsea well whose design rate in oil equivalent lies below its type's start hurdle, 50 and 500 Sm³/d, would not be drilled, and is dropped before any solve.
- **The producibility filter** (SMP-102, SMP-103). A well whose best rate, at a fully open choke and at three lift-gas rates for a lifted well, is below half its economic floor, or an oil whose WLR is above 0.98, would not be produced and is dropped. A natural oil well that cannot flow first takes its life's lift design (`lift_start_rule` `no_natural_flow`).
- **Redraws** (SMP-104, SMP-105). A slot's draw without an operating point is drawn again from the slot's own generator until the slot has one.
- **Limits** (SMP-106). A well is dropped, and the next drawn well takes its place, when a slot needs more than 100 attempts, the well more than 1000 redraws or 10 solver failures, when its inputs fail `configurations.check`, or when its table cannot be built. The drops are reported by group, with no target (SMP-106): [full generation: wells dropped at the limits by fluid class, well type and lift rule; the shift of each drawn input between the drawn and the kept wells]

Nothing else drops a well or a sample: v1's filters (SMP-28) do not apply, and a small rate stays a sample. The labels are drawn or set before the solve, except `lift_start_rule` `no_natural_flow`, which the filter sets (SMP-52). [full generation: wells dropped by outcome; share of the attempts redrawn]

**Was the "raw" data saved in addition to the preprocessed data?** Yes, as two logs in a companion folder next to the dataset (SMP-108). `drawn_wells` holds every drawn well, accepted or not, with its inputs, labels, outcome, best rate and counts of attempts; `redraws` every attempt without an operating point and every attempt of a dropped well, with its inputs, outcome, reason and root counts. With `samples` they hold every draw of the accepted wells, so the shift that the redraws cause can be measured input by input. The search counts and times of the attempts stay with the run.

**Is the software that was used available?** Yes: ManyWells at the commit in `metadata.json` (`code.commit`), under CC BY-NC 4.0, and NeqSim 3.4.0, an open-source library, for the tables. A dataset to publish is written only from a working tree without changes, so that its commit reproduces it (SMP-110).

## 5. Uses

**Has the dataset been used for any tasks already?** Not at the time of writing. The ManyWells paper's examples (`scripts/ml_examples/`) use the v1 datasets.

**Is there a repository that links to papers or systems that use the dataset?** [maintainer: where uses will be listed]

**What other tasks could the dataset be used for?** Besides the four tasks: virtual flow metering, rates from pressures, temperatures and the choke; surrogate models of the simulator; learning across wells, which the split by well tests; and production optimisation, where the lift-gas rate runs past the optimum (SMP-94). Also classifying flow regimes or choked flow, the sensitivity of the operating point to each input, and inferring a well's inputs from its samples.

**Is there anything about the composition or collection that might impact future uses?**

- The mix is a design (section 2). Statistics over the dataset are statistics of the design, not of real wells: reweight or filter by the labels.
- The redraws condition the samples on the well flowing; the logs let a user measure the shift and undo it.
- The data has no noise, no instrument errors, no missing measurements and no transients, unlike field data.
- A well's samples drift little around one state. For a well's change over its life, use `manywells-nsol-2`.

**Are there tasks for which the dataset should not be used?**

- Estimating statistics of real wells or fields: the shares are a design.
- Transient behaviour, such as heading, slug dynamics or start-up: the model is steady state.
- Anything downstream of the choke, such as flowlines, risers, separators and networks: `PDC` is a boundary condition.
- Stratified or horizontal flow, and oil–water slip.
- Predicting a real well without calibrating the model to it ([`calibration.md`](calibration.md)).

## 6. Distribution

**Will the dataset be distributed to third parties?** Yes, publicly.

**How will it be distributed?** On Hugging Face, in the repository `solution-seeker-as/manywells`, as parquet files under `v2/` with a config for each table and the unchanged v1 datasets next to them (D-86). The samples will load with `datasets.load_dataset('solution-seeker-as/manywells', name='manywells-sol-2')`. This datasheet goes on the dataset card. [maintainer: a DOI, if one is minted]

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
