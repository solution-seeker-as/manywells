# Sampler redesign — the dataset design

*Step R7 of `plans/sampler-redesign-plan.md`, 2026-11-02. A draft for Bjarne's checkpoint (R8). Plans are drafts, not specs. The well draws (R3) and the samples and generation rules (R4, revised by R7 for the ruling on D-52) are in `sampling-draft.md`; the working assumptions of this file are D-80 to D-94 in `decisions.md`.*

This file proposes what the v2 datasets hold and how they are generated: their files, their tables and columns, how a row is re-solved, the labels, the split, the datasheet, the generation, and what changes against v1. It keeps three things apart:

- **Decided:**
  - `specs/goals.md`, Compatibility → Datasets (Bjarne, 2026-10-30): two datasets, `manywells-sol-2` and `manywells-nsol-2`, each of 10,000 wells with 200 samples per well, in two tables, one of wells with their inputs and labels and one of samples holding each sample's operating point only. The published v1 datasets are never modified.
  - The Goals of the plan: unstable roots are not published; the wells table holds every well input and the labels, the samples table each sample's inputs and its operating point; a train–test split by well and a datasheet for each dataset are working assumptions.
  - Bjarne's ruling on D-52 (2026-11-02): "200 samples per well" means 200 operating points; a failed draw is redrawn until the well has 200, and every redraw is logged with its inputs and its reason.
  - The constitution, Dataset reproducibility: a row and its well's row together carry the inputs needed to re-solve it, and the generator records its seeds and the code version it ran.
- **Proposed:** everything below. Each choice rests on an entry in `decisions.md`.
- **Needs sign-off:** the schema as a whole (`AGENTS.md`: any change to a dataset schema), the limits and tolerances in it (D-81, D-92), the layout on Hugging Face and the split (release work), and the spec text it needs (`spec-changes.md`).

## 1. The datasets

| | `manywells-sol-2` | `manywells-nsol-2` |
|---|---|---|
| Kind | stationary: each sample is a draw around the well's state at one stage of its life | non-stationary: a well's 200 samples span its life, in time order |
| Wells | 10,000 | 10,000 |
| Samples per well | 200 operating points | 200 operating points: 160 operating samples and 8 test periods of 5 |
| Model | the `v2.0.0` configuration, $N$ = 100 cells | the same |
| Solver | the Rust core, with the CasADi backend as the retry of a solver failure (D-54, D-82; R6 measured no gain and proposes dropping the retry, `checkpoint.md` H1) | the same |
| Seed | its own (D-91) | its own |

Each dataset has four tables and two logs (D-87):

| Table | One row per | Holds |
|---|---|---|
| `samples` | accepted sample (operating point) | the sample's inputs that vary within the well, and its operating point |
| `wells` | accepted well | every input that is constant over the well's samples, every label, and the split |
| `geometry` | survey station | each well's survey of MD and TVD |
| `pvt` | node of a PVT table | the tables of the volatile oils and condensates, referenced by an ID from `wells` |
| `redraws` | attempt that was redrawn, recovered by the CasADi retry, or made by a dropped well | the redraw log: each such attempt's inputs, outcome and reason |
| `drawn_wells` | drawn well, accepted or not | the well log: every drawn well's inputs and labels, with its outcome |

The two tables of the decided layout are `wells` and `samples`. `geometry` and `pvt` are parts of the wells that do not fit one row (a survey has tens to hundreds of stations, and a well's PVT table set hundreds of nodes per GOR). The logs are the redraws the ruling asks to log. `spec-changes.md` proposes the matching text for `specs/goals.md`.

### On Hugging Face

The v2 datasets join the existing repository `solution-seeker-as/manywells` as new configs, with their files under `v2/` (D-86). The v1 files stay as they are: adding files to the repository does not modify them. The dataset card gains a v2 section and the datasheets (Section 5).

```
README.md                               the dataset card: v1's section, unchanged in substance, and v2's
data/manywells-*-1*.zip                 v1, unchanged
v2/manywells-sol-2/
    samples/train-00000-of-00004.parquet ...    sorted by ID and slot
    samples/test-00000-of-00001.parquet
    wells/train.parquet
    wells/test.parquet
    geometry.parquet
    pvt.parquet
    redraws.parquet
    drawn_wells.parquet
    metadata.json                       seed, configuration, code version, limits, counts, checksums
v2/manywells-nsol-2/                    the same files
```

The card's configs, for `sol-2` (and the same for `nsol-2`):

```yaml
configs:
- config_name: manywells-sol-2
  data_files:
  - split: train
    path: v2/manywells-sol-2/samples/train-*.parquet
  - split: test
    path: v2/manywells-sol-2/samples/test-*.parquet
- config_name: manywells-sol-2-wells
  data_files:
  - split: train
    path: v2/manywells-sol-2/wells/train.parquet
  - split: test
    path: v2/manywells-sol-2/wells/test.parquet
- config_name: manywells-sol-2-geometry
  data_files: v2/manywells-sol-2/geometry.parquet
- config_name: manywells-sol-2-pvt
  data_files: v2/manywells-sol-2/pvt.parquet
- config_name: manywells-sol-2-redraws
  data_files: v2/manywells-sol-2/redraws.parquet
- config_name: manywells-sol-2-drawn-wells
  data_files: v2/manywells-sol-2/drawn_wells.parquet
```

`load_dataset('solution-seeker-as/manywells', name='manywells-sol-2')` then gives the samples with a train and a test split, as `name='manywells-sol-1'` gives v1's. The samples are sharded by well, in parquet row groups of about 50 wells, so that a reader can select wells without reading the whole file.

**Sizes** (R7's estimates; R6's measurements at the end): the samples are 2 million rows of about 35 (`sol-2`) and 45 (`nsol-2`) columns, about 0.5 GB and 0.7 GB of float64 before compression, and a few hundred MB as parquet. The wells table is about 10,000 rows of 110 to 130 columns, a few MB. The geometry is about 0.5 million rows, the redraw log $2r$ million rows at a redraw share $r$ (redraws per accepted sample), and the PVT tables, one table set per table well (D-67), up to about 1 GB as CSV for `sol-2` at R6's 3200 table wells (`fluids-draft.md`, One table per well, or a library). R6 measured about 420 MB and 440 MB of samples before compression, and about 130,000 rows of redraws (`r6-measurements.md`, Cost).

## 2. The schema

### Conventions

- **Units** are the model's (`specs/model/nomenclature.md`): m, bar (absolute), K, kg/s, kg/m³, J/(kg K), W/(m² K), and Sm³/Sm³ at standard conditions (PVT-GAS-2). The samples keep v1's Sm³/h for standard volumetric rates (`docs/datasets.md`).
- **Names** (D-88). The samples keep v1's features by name (`CHK`, `PBH`, …) and add new ones in the same style: short, upper case. The wells table names a model input by its code name in `nomenclature.md` (`rho_w`, `K_c`, `injection_md`), and a drawn descriptor that is not a model input by a short lower-case name (`z`, `water_depth`). A label is named as in R3's Labels table.
- **One place for each input** (D-87). An input that is constant over a well's samples is in `wells`; one that varies is in `samples`. In `sol-2` the reservoir pressure, the GOR and the water vary (the drift); in `nsol-2` the heat-transfer coefficient, the roughness, the inflow's coefficients and whether lift is installed vary too, so they move to `samples` there.
- **Types** (D-88). Every continuous quantity is float64, so that a re-solve uses the generator's inputs to the bit. IDs are int32, slots and attempts int16, flags boolean, labels strings stored dictionary-encoded (category in pandas). A value that does not apply, such as the injection depth of a natural well, is null, never a sentinel; pandas reads a null float as NaN.
- **Marks.** *Kept*: a v1 feature with the same name and meaning (`docs/datasets.md`), or for the wells a column of v1's config table (`manywells.sampling.wells.WellDraw`, as `datasets.io.write_dataset` writes it) with the same name; *renamed*: a v1 quantity under a new name or with a changed meaning, given after the arrow; *new*: not in v1.
- **Source draws** refer to the sections of `sampling-draft.md`: R3's well draws by their section names, R4's sample draws by theirs.

### `wells`

One row per accepted well, sorted by `ID`. The columns marked (s) are in `sol-2`'s table only, those marked (n) in `nsol-2`'s only; in `nsol-2` the columns marked (s) that vary with time are in `samples` (below). A row, its `geometry` rows and its `pvt` rows, with a row of `samples`, carry every input of a `WellProperties` and a `BoundaryConditions` (Section 3).

**Identity and split**

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `ID` | – | int32 | the well's ID in the dataset, 0 to 9,999, in draw order; unique within a dataset only | generator | kept |
| `draw_index` | – | int64 | $k$, the well's place in the sequence of drawn wells, which seeds its draws (SMP-31) | generator (D-91) | new |
| `split` | – | category | `train` or `test` | Section 5 (D-90) | new |
| `configuration` | – | category | `v2.0.0` | Mapping | new |
| `n_cells` | – | int16 | the grid's cells, 100 | Mapping | new |

**Labels** (codes in Section 4)

| Column | Type | Source | v1 |
|---|---|---|---|
| `coverage_tail` | category | Coverage tail | new |
| `well_type` | category | Well type | new |
| `fluid_type`, `fluid_class`, `class_rule` | category | Fluid class | new |
| `drive` | category | Drive and life stage | new |
| `life_stage` (s) | category | Drive and life stage | new |
| `life_state` (s) | category | derived from the stage and the drive | new |
| `pressure_class` | category | Initial reservoir pressure | new |
| `saturation` | category | Reservoir state at the life stage (`nsol-2`: at the start) | new |
| `geometry_class` | category | Geometry: the ruled class | renamed: `trajectory` → `geometry_class` and `geometry_family` |
| `geometry_family` | category | Geometry: the family that drew the survey | (as above) |
| `pvt_model` | category | Mapping | new |
| `fluid_source` | category | `fluids-draft.md`, Which wells take a table: `synthetic`, or the published example's key (D-66); null for a black oil | new |
| `saturation_kind` | category | `fluids-draft.md`, The saturation pressure: the table's saturation point at $T_r$, bubble or dew (D-65); null for a black oil | new |
| `saturation_rule` | category | `fluids-draft.md`, The saturation pressure: whether the drawn GOR was kept, or the fluid changed so that $p_\text{sat}(T_r) \le p_i$ (D-65); null for a black oil | new |
| `inflow` | category | Inflow | new |
| `tubing` | category | Tubing | new |
| `choke_model` | category | Choke | new |
| `chk_profile` | category | Choke (R3's label `choke_profile`) | kept |
| `lift` | category | Gas lift: assignment (`nsol-2`: lifted at some time in the life, D-49) | renamed: `has_gas_lift` → `lift` |
| `lift_reason` (s) | category | Gas lift: assignment; `no_natural_flow` after the best-rate solve (D-101) | new |
| `lift_start_rule` (n) | category | Gas lift that starts mid-life | new |

**Depth and geometry** (the survey itself is in `geometry`)

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `z` | m | float64 | reservoir depth below the mudline | Depth below the mudline | new |
| `water_depth` | m | float64 | $d_w$; 0 onshore | Water depth and the well's length | new |
| `L` | m | float64 | true vertical depth of the bottom of the flow path below the wellhead (GEO-3) | Water depth and the well's length | kept |
| `md_bottom` | m | float64 | measured depth of the bottom, the survey's last station | Geometry | new |
| `max_inclination` | ° | float64 | the largest cell inclination from vertical on the grid | Geometry | new |
| `kickoff_tvd` | m | float64 | $K$; null for `vertical` | Geometry | renamed: `kickoff_fraction` → `kickoff_tvd` |
| `build_rate`, `build_rate_2` | °/30 m | float64 | $b$, and $b_2$ of `high_angle` | Geometry | new |
| `hold_inclination` | ° | float64 | $\theta_h$ | Geometry | renamed: `inclination` → `hold_inclination` |
| `drop_rate`, `end_inclination`, `end_section_tvd` | °/30 m, °, m | float64 | $d$, $\theta_e$ and the TVD of the last section of `s_shaped` | Geometry | new |
| `heel_inclination`, `tangent_md` | °, m | float64 | $\theta_\text{heel}$ and the tangent's MD of `high_angle` | Geometry | new |
| `drain_length` | m | float64 | $L_d$, the drain the heel stands for (`high_angle`) | Productivity | renamed: `horizontal_length` → `drain_length` (not modelled) |

**Water, pressure and temperature**

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `salinity` | – | float64 | $w_s$, mass fraction of dissolved solids | Formation water | new |
| `rho_w` | kg/m³ | float64 | water density at standard conditions (`FluidModel.rho_w`) | Formation water | kept |
| `p_h` | bar | float64 | hydrostatic pressure at the reservoir | Initial reservoir pressure | new |
| `overpressure_ratio` | – | float64 | $\lambda = (p_i - p_h)/(p_\text{lith} - p_h)$, negative for an underpressured reservoir | Initial reservoir pressure | new |
| `p_i` | bar | float64 | initial reservoir pressure | Initial reservoir pressure | new |
| `T_s` | K | float64 | surface temperature (`BoundaryConditions.T_s`) | Surface temperature | kept |
| `geothermal_gradient` | K/km | float64 | $G$ below the mudline | Geothermal gradient | new |
| `T_r` | K | float64 | reservoir temperature (`BoundaryConditions.T_r`) | Geothermal gradient and reservoir temperature | kept |

**Separator and fluid**

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `p_sep`, `T_sep` | bar, K | float64 | the host's first-stage separator (`BlackOil.p_sep`, `T_sep`) | Separator conditions | new |
| `p_s_nom` (s) | bar | float64 | nominal pressure downstream of the choke at the stage | Separator conditions | renamed: `p_s` → `p_s_nom` |
| `r_s` (s) | – | float64 | the stage's factor on $p_\text{sep}$ | Separator conditions | new |
| `dp_flowline` | bar | float64 | $\Delta p_\text{fl}$ of a subsea well's flowline and riser; 0 otherwise | Separator conditions | new |
| `api`, `sg_gas` | °API, – | float64 | stock-tank oil gravity and gas gravity | Oil and gas at standard conditions | new |
| `rho_o` | kg/m³ | float64 | stock-tank oil density (`BlackOil.rho_o`, or the table's) | Oil and gas at standard conditions | kept |
| `rho_g` | kg/m³ | float64 | gas density at standard conditions (`BlackOil.rho_g`, or the table's) | Oil and gas at standard conditions | renamed: `R_s` → `rho_g` ($R_s = p_\text{ref}/(\rho_g T_\text{ref})$) |
| `p_sat` | bar | float64 | saturation pressure (bubble or dew point) at $T_r$; for a table fluid the table's (D-65) | Saturation pressure and the initial GOR | new |
| `R_si` | Sm³/Sm³ | float64 | initial producing GOR; for a table fluid after D-65's rule | Saturation pressure and the initial GOR | new |
| `pvt_table_id` | – | string | the well's table set in `pvt`, one per table well (D-67); null for a black oil | Mapping; `fluids-draft.md` | new |

**Table fluids** (volatile oils and condensates, `pvt_model = table`; every column null for a black oil; `fluids-draft.md`, The composition, D-63). The labels `fluid_source`, `saturation_kind` and `saturation_rule` are under Labels. The composition's generator, `rng_for(seed, k, 'v2', 'composition')`, follows from the dataset's seed and `draw_index`, so with these columns a user can make the table again.

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `comp_w1`, `comp_w2`, `comp_w3` | – | float64 | the Dirichlet weights $w$ of the class's three published examples in the light ends, in the order of `fluids-draft.md`; null for an example fluid | The composition | new |
| `z_c7p`, `M_c7p` | –, g/mol | float64 | the heptanes-plus's mole fraction $z_{7+}$ before the GOR is set, and its molar mass $M_{7+}$: drawn, or the example's | The composition | new |
| `alpha_c7p` | – | float64 | the gamma distribution's shape $\alpha$ that splits the heptanes-plus: drawn, 1 for a lumped example, null for an example with its fractions given | The composition | new |
| `C_f`, `f_int` | – | float64 | the levers matched to the drawn API and gas gravity: Søreide's $C_f$ (for a lumped example, matched to its heptanes-plus gravity) and the factor $f_\text{int}$ on C₂ to C₆; null where the fluid is an example that does not use them | The composition | new |

**Reservoir state** (`sol-2`: the stage's nominal state, around which the samples drift; `nsol-2`: the start and end states and the trends' parameters)

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `p_r_nom` (s) | bar | float64 | nominal reservoir pressure at the stage | Reservoir state at the life stage | renamed: `p_r` → `p_r_nom` |
| `gor_nom`, `wlr_nom` (s) | Sm³/Sm³, – | float64 | nominal producing GOR and water–liquid ratio | Reservoir state at the life stage | renamed: `fractions` → `gor_nom`, `wlr_nom` |
| `wgr_nom` (s) | Sm³/10⁶ Sm³ | float64 | a gas well's nominal water–gas ratio; null for oils | Reservoir state at the life stage | new |
| `k_g`, `gor_factor_cond` | – | float64 | the oils' GOR factor and the condensates' factor; null where not applicable | Reservoir state at the life stage | new |
| `T_life`, `T_end` (n) | years | float64 | the life span and the end of the life after the scan | Life span; The end of a life | new |
| `t_plateau`, `decline_D`, `decline_b` (n) | years, 1/year, – | float64 | the clock's plateau and Arps's parameters | Life span and the depletion clock | new |
| `r_p_start`, `r_p_end`, `gamma_p` (n) | – | float64 | $p_r/p_i$ at the start and the end, and $\gamma$ | Reservoir pressure | new |
| `F_bt`, `wlr_bt`, `water_start`, `water_end` (n) | – | float64 | breakthrough, the jump, and the start and end water (WLR for oils, WGR for gases) | Water | new |
| `t_sep_1`, `t_sep_2`, `r_s_mid`, `r_s_late` (n) | years, – | float64 | the separator's step-downs | Separator over the life | new |
| `p_change` (n) | – | float64 | $p_c$, the probability of a set-point change | Control history | new |
| `lift_start` (n) | years | float64 | the start of gas lift; null for a well that stays natural | Gas lift that starts mid-life | new |

**Productivity and inflow** (in `nsol-2` the inflow's coefficients vary and are in `samples`)

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `permeability`, `net_pay`, `skin`, `ln_re_rw` | mD, m, –, – | float64 | $k$, $h_n$, $s$, $\ln(r_e/r_w)$ | Productivity | new |
| `drawdown_design` | – | float64 | $\delta_d$ | Productivity | new |
| `anisotropy`, `productivity_factor` | – | float64 | $I_\text{ani}$ (`high_angle`), and the geometry's factor $f$ | Productivity | new |
| `J` | Sm³/(d bar) | float64 | an oil well's productivity index at the stage; null for gas wells | Productivity | new |
| `non_darcy_D` | d/Sm³ | float64 | a gas or condensate well's non-Darcy coefficient $D$ (D-23; D-62 proposes its draw), a descriptor behind `forchheimer_b`; null for oils | Productivity | new |
| `k_l` (s) | kg/(s bar) | float64 | `ProductivityIndex.k_l`; null unless `inflow` is `productivity_index` | Inflow | new |
| `w_l_max` (s) | kg/s | float64 | `Vogel.w_l_max`; null unless `inflow` is `vogel` | Inflow | kept |
| `forchheimer_a`, `forchheimer_b` (s) | bar² s/kg, bar² s²/kg² | float64 | `Forchheimer.a` and `Forchheimer.b` (INF-10, feature 022) in the model's units, fitted to the pseudo-pressure (D-61); null unless `inflow` is `forchheimer` | Inflow (D-60, D-61) | new |
| `floor`, `best_rate`, `best_rate_w_lg` | Sm³/d, Sm³/d, kg/s | float64 | the economic floor, the best rate in oil equivalent and its lift-gas rate (`nsol-2`: at the start) | The economic floor; Producibility filter | new |

**Completion, choke, lift and heat**

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `D` | m | float64 | tubing inner diameter | Tubing | kept |
| `roughness` (s) | m | float64 | `RoughnessFriction.roughness` (Chen) | Roughness | kept |
| `roughness_start`, `roughness_end` (n) | m | float64 | the initial and late draws between which it ages | Well properties that age | new |
| `bean_ratio`, `C_D` | – | float64 | $\beta$ and the discharge coefficient | Choke | new |
| `K_c` | m² | float64 | `SimpsonChokeModel.K_c` | Choke | kept |
| `injection_md` | m | float64 | `WellProperties.injection_md`; null for a natural well | Gas lift: injection depth | new |
| `injection_tvd` | m | float64 | $H$ | Gas lift: injection depth | new |
| `T_lg` | K | float64 | `BoundaryConditions.T_lg`; null for a natural well | Gas lift: lift-gas temperature | new |
| `w_lg_max` | kg/s | float64 | the lift-gas capacity; null for a natural well | Gas lift: capacity | new |
| `glr_cap`, `p_inj`, `dp_valve`, `packer_offset`, `phi_lg` | Sm³/Sm³, bar, bar, m, – | float64 | the lift design's draws | Gas lift | new |
| `h` (s) | W/(m² K) | float64 | `ThermalModel.h` | Heat-transfer coefficient | kept |
| `U_t`, `k_e`, `well_age` | W/(m² K), W/(m K), d | float64 | the draws behind $h$ (`nsol-2`: `well_age` at the start) | Heat-transfer coefficient | new |
| `cp_o`, `cp_w`, `cp_g` | J/(kg K) | float64 | heat capacities (`FluidModel`) | Heat capacities | kept |

**Generation summary**

| Column | Type | Definition | v1 |
|---|---|---|---|
| `n_redraws` | int32 | redraws over the well's slots: attempts beyond the first | new |
| `n_redraws_no_op`, `n_redraws_solver` | int32 | the redraws by outcome: `no_operating_point` and `solver_failure` | new |
| `n_recovered` | int32 | attempts whose Rust failure the CasADi retry recovered | new |

v1's `f_D` (a fixed friction factor) has no counterpart: `v2.0.0` takes friction from the roughness.

### `samples`

One row per accepted sample, sorted by `ID` and `slot`. In `nsol-2` the slots are in time order.

**Keys and time**

| Column | Unit | Type | Definition | Source | v1 |
|---|---|---|---|---|---|
| `ID` | – | int32 | the well | – | kept |
| `slot` | – | int16 | $j$, 0 to 199 | Slots and outcomes | new |
| `attempt` | – | int16 | the accepted attempt: 0 for the slot's first draw, $a$ for its $a$-th redraw | Slots and outcomes (D-80) | new |
| `WEEKS` (n) | weeks | float64 | time since the well's first production, $t_j$ | Test periods; The end of a life | kept, now a float from first production |
| `STAGE` (n) | – | category | life stage at the date | The well and its states | new |
| `TEST` (n) | – | int8 | the test period, 1 to 8; null for an operating sample | Test periods | new |
| `SWEEP` (n) | – | category | `choke` or `lift` in a test; null otherwise | Test periods | new |

**The sample's inputs**

| Column | Unit | Type | Definition (model input) | Source | v1 |
|---|---|---|---|---|---|
| `CHK` | – | float64 | choke position $u$ (`BoundaryConditions.u`) | Choke position; Control history | kept |
| `PDC` | bar | float64 | pressure downstream of the choke (`BoundaryConditions.p_s`) | Separator pressure | kept |
| `WGL` | kg/s | float64 | lift-gas rate (`BoundaryConditions.w_lg`) | Lift-gas rate; Control history | kept |
| `PRES` | bar | float64 | reservoir pressure (`BoundaryConditions.p_r`) | Drift of the reservoir; Reservoir pressure | new |
| `GOR` | Sm³/Sm³ | float64 | producing gas–oil ratio (`FluidModel.gor`) | Drift; Gas–oil ratio | new |
| `WLR` | – | float64 | water–liquid ratio at standard conditions (`FluidModel.wlr`) | Drift; Water | new |
| `LIFT` (n) | – | bool | lift installed at the date: the well has its injection point | Gas lift that starts mid-life | new |
| `HTC` (n) | W/(m² K) | float64 | `ThermalModel.h` at the date | Well properties that age | new |
| `ROUGH` (n) | m | float64 | `RoughnessFriction.roughness` at the date | Well properties that age | new |
| `KL`, `WLMAX`, `FORCHA`, `FORCHB` (n) | as in `wells` | float64 | the inflow's coefficients at the date (`k_l`, `w_l_max`, `forchheimer_a`, `forchheimer_b`); null where the well's `inflow` does not use them | Well properties that age | new |

**The operating point** (computed from the stable root, as SMP-30)

| Column | Unit | Type | Definition | v1 |
|---|---|---|---|---|
| `PBH`, `PWH` | bar | float64 | pressure at the bottomhole and the wellhead | kept |
| `TBH`, `TWH` | K | float64 | temperature at the bottomhole and the wellhead | kept |
| `WGAS`, `WLIQ`, `WOIL`, `WWAT` | kg/s | float64 | the reservoir stream's gas (without lift gas), liquid, oil and water | kept |
| `WTOT` | kg/s | float64 | WLIQ + WGAS + WGL | kept |
| `QGL`, `QGAS`, `QLIQ`, `QOIL`, `QWAT`, `QTOT` | Sm³/h | float64 | the same at standard conditions, with QTOT = QLIQ + QGAS + QGL | kept |
| `FGAS`, `FOIL`, `FWAT` | – | float64 | the reservoir stream's mass fractions; now derived from GOR, WLR and the densities, not drawn | kept, now derived |
| `CHOKED` | – | bool | critical flow at the choke (CHK-12) | kept |
| `FRBH`, `FRWH` | – | category | the flow regime at the bottomhole and the wellhead: the five regimes of SLIP-16 | kept, with the five regimes |

The rates are those of `datasets/rows.py`: the reservoir's at standard conditions, with the stock-tank oil and the gas that leaves it counted in `WOIL` and `WGAS`, so v1's relations between them hold. The samples hold no solver diagnostics: those of the attempts that needed more than one Rust solve are in `redraws`, and the counts per well in `wells` and `drawn_wells`.

### `geometry`

One row per survey station, sorted by `ID` and `station`.

| Column | Unit | Type | Definition | v1 |
|---|---|---|---|---|
| `ID` | – | int32 | the well | new |
| `station` | – | int16 | 0 at the wellhead | new |
| `md`, `tvd` | m | float64 | measured and true vertical depth below the wellhead | new (v1: `trajectory` and its parameters) |

The stations are the drawn survey's: two for a vertical well, one per degree along build and drop sections (Geometry). `WellGeometry.from_survey(md, tvd, n_cells, D)` builds the grid from them, as the generator did (GEO-4).

### `pvt`

One row per node of a table, in the columns of feature 018's CSV files (`PVTTable.read_csv`), so that a table is rebuilt from its rows to the bit.

| Column | Unit | Type | Definition | v1 |
|---|---|---|---|---|
| `pvt_table_id` | – | string | the table, referenced by `wells.pvt_table_id` | new |
| `gor` | Sm³/Sm³ | float64 | the GOR of a table on a GOR axis (`PVTTableSet`); null for a single table | new |
| `p`, `T` | bar, K | float64 | the node | new |
| `R_so`, `B_o`, `Z` | Sm³/Sm³, –, – | float64 | PVT-TAB-1's columns | new |
| `mu_o`, `mu_g`, `sigma` | Pa s, Pa s, N/m | float64 | the optional columns; null where the table has none | new |
| `rho_o`, `rho_g` | kg/m³ | float64 | the table's densities at standard conditions, the same in every row of a table | new |
| `name`, `source`, `separation` | – | string | the table's free text; `source` names the fluid (`synthetic` with the well's `draw_index`, or the example's key and reference), NeqSim's version and the equation of state (`fluids-draft.md`, The table: Format; D-68) | new |

D-67 proposes one table set per table-fluid well, on its own grid and GOR axis (`fluids-draft.md`, One table per well, or a library), so each `pvt_table_id` belongs to one well. The schema would also hold a library of fluids, since several wells can share an ID. Every dataset carries its own tables, so that it is self-contained.

### `redraws`: the redraw log

One row per attempt that did not end as a sample from a Rust solve alone (D-85): every redrawn attempt of an accepted well, every attempt of a well dropped at a limit or a dead date, and every attempt that the CasADi retry recovered, whose operating point is the sample. Together with `samples` it holds every draw of the accepted wells, so the inputs over all draws and over the accepted ones can be compared row for row.

| Column | Unit | Type | Definition |
|---|---|---|---|
| `draw_index` | – | int64 | the well's $k$ |
| `ID` | – | int32 | the well's ID; null for a dropped well |
| `slot`, `attempt` | – | int16 | the attempt |
| the inputs | as `samples` | as `samples` | every input column of `samples`: `CHK`, `PDC`, `WGL`, `PRES`, `GOR`, `WLR`, and in `nsol-2` the time, stage, test, sweep and the inputs that age |
| `outcome` | – | category | `no_operating_point`, `solver_failure`, `recovered` (the retry found the operating point, which is the sample), or `operating_point` (an attempt of a dropped well) |
| `reason` | – | category | why there was no operating point (Section 4); for `recovered`, the Rust core's reason |
| `retried` | – | bool | whether the CasADi retry ran |
| `start` | – | string | for a retry, the extra start: `slot:<j>` or `best_rate:<w_lg/w_lg,max>`; null otherwise |
| `n_roots`, `n_stable` | – | int8 | the size of the attempt's root set and its stable roots, from the backend that decided the outcome |

### `drawn_wells`: the well log

One row per drawn well $k$, from 0 to the last one the generator needed, accepted or not. It has every column of `wells` (with `ID` and `split` null for a well that was not accepted), and:

| Column | Type | Definition |
|---|---|---|
| `outcome` | category | `accepted`, or why the well was dropped (Section 4) |
| `n_attempts` | int32 | attempts made over its slots, 0 for a well dropped before its slots; with `wells`' counts of redraws by outcome and of recoveries |
| `n_several_stable` | int32 | samples whose root set had more than one stable root (SOL-6's case) |
| `best_rate_outcomes` | list of category | the outcomes of the best-rate solves, one per lift-gas rate |
| `scan_best_rate` (n) | list of float64 | the life scan's best rates, 21 points, null where a point failed |
| `scan_outcome` (n) | list of category | the life scan's outcomes |

The rows of the accepted wells repeat `wells`; the log is the table to use for the effect of the well filters, since it holds the dropped wells' inputs.

### `metadata.json`

The dataset's name and version; the date; the seed; the configuration, $N$, the backend and the retry's backend; the code version from `datasets.io.code_version` with the full commit; the SHA-256 of `uv.lock`; Python's version and the platform; the redraw limits and the producibility filter's parameters; the counts of wells drawn, accepted and dropped by reason, and of attempts by outcome; the CPU time; and each file's name, row count and SHA-256.

## 3. Re-solving a sample

A row of `samples`, its well's row of `wells`, the well's `geometry` rows and, for a table fluid, its `pvt` rows give a `WellProperties` and a `BoundaryConditions` in `v2.0.0`. For `sol-2`:

```python
from manywells.choke import SimpsonChokeModel
from manywells.configurations import V2, V2_SLIP, check
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Forchheimer, ProductivityIndex, Vogel  # Forchheimer: feature 022, not implemented yet
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.thermal import ThermalModel


def case(w, s, survey, pvt_rows):
    """The well and boundary conditions of sample row s of well row w (sol-2)."""
    if w.pvt_model == 'black_oil':
        hc = BlackOil(rho_o=w.rho_o, rho_g=w.rho_g, gas_law='dak', p_sep=w.p_sep, T_sep=w.T_sep)
    else:
        hc = pvt_from_rows(pvt_rows)  # R10: a PVTTable or PVTTableSet from the rows of w.pvt_table_id
    if w.inflow == 'vogel':
        inflow = Vogel(w_l_max=w.w_l_max)
    elif w.inflow == 'productivity_index':
        inflow = ProductivityIndex(k_l=w.k_l)
    else:  # 'forchheimer'
        inflow = Forchheimer(a=w.forchheimer_a, b=w.forchheimer_b)
    lifted = w.lift == 'gas_lift'
    wp = WellProperties(
        geometry=WellGeometry.from_survey(survey.md, survey.tvd, n_cells=w.n_cells, D=w.D),
        fluid=FluidModel(pvt=hc, rho_w=w.rho_w, gor=s.GOR, wlr=s.WLR, cp_g=w.cp_g, cp_o=w.cp_o, cp_w=w.cp_w),
        friction=RoughnessFriction(roughness=w.roughness, correlation='chen'),
        thermal=ThermalModel(h=w.h),  # every v2.0.0 term on: the defaults
        slip=V2_SLIP,
        inflow=inflow,
        choke=SimpsonChokeModel(K_c=w.K_c, chk_profile=w.chk_profile),
        injection_md=w.injection_md if lifted else None,
    )
    bc = BoundaryConditions(
        p_r=s.PRES, p_s=s.PDC, T_r=w.T_r, T_s=w.T_s, T_lg=w.T_lg if lifted else None, u=s.CHK, w_lg=s.WGL
    )
    check(wp, V2)
    return wp, bc


op = SSDFSimulator(wp, backend='rust').simulate(bc)
```

In `nsol-2` the heat-transfer coefficient, the roughness and the inflow's coefficients come from the sample (`HTC`, `ROUGH`, and `KL`, `WLMAX` or `FORCHA` and `FORCHB`), and the injection point is set where `LIFT` is true. A sample that the CasADi retry recovered (its attempt is in `redraws` with outcome `recovered`) is re-solved with `backend='casadi'`, with the operating point of its logged start, re-solved first, as `x_guess`.

**The library helper** (R10). `manywells.datasets` gets a function that does the above from the rows, and one that computes a sample's features from an operating point (SMP-30's row with the v2 columns). The datasheet shows them in place of this listing.

**The check** (D-92). Re-solve the sample and compute its features from the operating point. With the code version and backend in `metadata.json`, on the generator's platform, every feature is expected to be identical: the core is deterministic, and float64 round-trips through parquet. Elsewhere, the features agree within the verifier's `tol_x` (`specs/verification.md`, Tolerances), applied to them: `PBH` and `PWH` within $10^{-4}\thinspace(p_r - p_s)$, `TBH` and `TWH` within $10^{-4}\thinspace(T_r - T_s)$, every mass rate within $10^{-4}$ of `WTOT`, `CHOKED` the same unless the wellhead lies within the verifier's `choke_band` of the switch, and the regimes the same unless the point lies within the slip model's transition width. These tolerances need sign-off. R11 runs the check on a seeded sample of 1000 rows of each dataset, read back from the published files, and on 100 of them with the CasADi backend; R10 makes it a test on a pilot dataset.

## 4. Labels and codes

Labels are stored as strings (D-89). The codes below are fixed: a user who needs integers takes the position in the list, and a later version adds categories at the end. R3's Labels table and R4's rules define each value.

| Label | Table | Codes |
|---|---|---|
| `coverage_tail` | wells | 0 `core`, 1 `tail` |
| `well_type` | wells | 0 `platform`, 1 `subsea`, 2 `onshore` |
| `fluid_type` | wells | 0 `black_oil`, 1 `volatile_oil`, 2 `gas_condensate`, 3 `gas` |
| `fluid_class` | wells | 0 `black_oil`, 1 `volatile_oil`, 2 `gas_condensate`, 3 `wet_gas`, 4 `dry_gas` |
| `class_rule` | wells | 0 `mccain_1990` |
| `drive` | wells | 0 `depletion`, 1 `pressure_support` |
| `life_stage`, `STAGE` | wells (s), samples (n) | 0 `initial`, 1 `mid_life`, 2 `late` |
| `life_state` | wells (s) | 0 `initial`, 1 `mid_life`, 2 `depleted`, 3 `watered_out` |
| `pressure_class` | wells | 0 `under`, 1 `normal`, 2 `over` |
| `saturation` | wells | 0 `undersaturated`, 1 `saturated`, 2 `single_phase_gas` |
| `geometry_class` | wells | 0 `vertical`, 1 `slightly_deviated`, 2 `deviated`, 3 `highly_deviated` |
| `geometry_family` | wells | 0 `vertical`, 1 `build_and_hold`, 2 `s_shaped`, 3 `high_angle` |
| `pvt_model` | wells | 0 `black_oil`, 1 `table` |
| `fluid_source` | wells | 0 `synthetic`, 1 `wb2000_gas_condensate`, 2 `pedersen2015_t1_3`, 3 `pedersen2015_t3_8`, 4 `wb2000_near_critical_oil`, 5 `wb2000_volatile_oil`, 6 `pedersen2015_t1_4` (the examples' keys of `fluids-draft.md`); null for a black oil |
| `saturation_kind` | wells | 0 `bubble`, 1 `dew`; null for a black oil |
| `saturation_rule` | wells | 0 `matched`, 1 `gor_moved`, 2 `equilibrium_phase`; null for a black oil |
| `inflow` | wells | 0 `vogel`, 1 `productivity_index`, 2 `forchheimer` (R3 drafted `gas_rate`; feature 022, D-60) |
| `tubing` | wells | 0 `2-7/8`, 1 `3-1/2`, 2 `4-1/2`, 3 `5-1/2`, 4 `7`, 5 `9-5/8` |
| `choke_model` | wells | 0 `simpson` |
| `chk_profile` | wells | 0 `linear`, 1 `sigmoid`, 2 `convex`, 3 `concave` (`CHOKE_PROFILES`) |
| `lift` | wells | 0 `natural`, 1 `gas_lift` |
| `lift_reason` | wells (s) | 0 `drawn`, 1 `needs_lift`, 2 `no_natural_flow` (D-101) |
| `lift_start_rule` | wells (n) | 0 `from_start`, 1 `static_check`, 2 `end_of_natural_flow`, 3 `drawn_time`; null for a natural well |
| `split` | wells | 0 `train`, 1 `test` |
| `SWEEP` | samples (n) | 0 `choke`, 1 `lift` |
| `FRBH`, `FRWH` | samples | 0 `bubbly`, 1 `dispersed-bubbly`, 2 `slug`, 3 `churn`, 4 `annular` (SLIP-16) |
| `outcome` | redraws | 0 `operating_point`, 1 `no_operating_point`, 2 `solver_failure`, 3 `recovered` |
| `reason` | redraws | 0 `p_r_le_p_s` (no solve), 1 `no_root` (no root and no rejected sign change, whatever the non-finite scan points, D-102), 2 `exception`, 3 `invalid_row`, 4 `incomplete_scan` (a rejected sign change, D-102), 5 `unstable_only` (D-53) |
| `outcome` | drawn_wells | 0 `accepted`, 1 `not_producible`, 2 `best_rate_solver_failure`, 3 `invalid_input`, 4 `slot_limit`, 5 `redraw_limit`, 6 `solver_failure_limit`, 7 `dead_date` (n), 8 `short_life` (n) |

Reasons 0 and 1 make an attempt `no_operating_point`, reasons 2 to 5 a `solver_failure` (D-53, revised by D-102).

The geometry labels follow Bjarne's ruling on D-9: `geometry_class` holds the ruled classes and `geometry_family` R3's families, as `sampling-draft.md`'s Labels table now has them. The choke's profile keeps v1's name, `chk_profile`, where R3 wrote `choke_profile`.

## 5. The split and the documentation

### A train–test split by well

- **The split** (D-90). 80% of each dataset's wells are `train` and 20% `test`, by well: all of a well's samples, geometry and redraws follow its `split`. The split is stratified by `fluid_type` × `well_type` × `geometry_class` × `coverage_tail` (96 strata): within each stratum the wells are put in an order drawn from `rng_for(seed, 'v2', 'split')`, and every fifth is `test`, so each class has its share in both parts to within one well.
- **Why by well.** A well's samples share every well input, so a split by row puts the same wells in both parts and measures interpolation within wells, not the generalisation to new wells that learning across wells and virtual flow metering on a new well need. v1 published no split; users split as they chose.
- **Within a well.** A task that learns one well, or forecasts `nsol-2` in time, splits within the well's samples: by slot in `sol-2`, by date in `nsol-2` (for example the first 80% of the life). The datasheet says so; the dataset does not fix it.
- **The tail.** Tail wells are in both parts. A user who wants the realistic core only filters on `coverage_tail`.

### A datasheet for each dataset

After Gebru et al. (2021), whose questions are grouped in seven sections (§3.1 to §3.7, pp. 4–10 of the preprint). Each dataset gets one (D-94), written in R11 as `docs/datasheet-sol-2.md` and `docs/datasheet-nsol-2.md` in the repo, and copied to the dataset card. The outline, with what each section answers:

1. **Motivation** (§3.1). The four tasks (virtual flow metering, surrogate models, learning across wells, production optimisation); the gap (v1's sampled wells were unrealistic, `specs/goals.md`, limitation 5); who made it, for whom, and its funding (for Bjarne to fill).
2. **Composition** (§3.2). What an instance is (an operating point of a simulated well, grouped by well; `nsol-2` in time order); the counts; the population it samples, a synthetic one: offshore-weighted, counted by wells, with design choices that are not statistics (the even geometry mix, the fluid floor of 10%, the coverage tail, 20% of the drawn wells and about 15% of the accepted); what each row holds (Section 2); the labels; what is missing (nothing in the samples; nulls where a column does not apply); the relations (`ID`, `slot`, time); the split; the sources of error: no measurement noise, the model's limits (steady state, one liquid, no stratified flow), the correlations used outside their ranges (R6's shares from `validity_ranges.py`), and the redraws' conditioning (the samples are the draws at which the well flows); self-contained (the PVT tables are included); no confidential data and nothing about people.
3. **Collection process** (§3.3). The simulator and its configuration (`v2.0.0`), the backends, the sampling procedure (`specs/sampling.md`'s v2 version) and the public sources it rests on; the seeds and the code version; the generation's dates and machine; the checks run (Section 6). A private comparison with field data, if Bjarne makes one, is named in general terms only (constitution, Confidential data).
4. **Preprocessing, cleaning, labelling** (§3.4). The producibility filter; the redraws and the redraw limits; the dropped wells; that nothing else is removed; the logs as the "raw" data; the software at the recorded commit.
5. **Uses** (§3.5). The four tasks, and what not to use the data for: estimating real field statistics (the mix is designed), transient behaviour (no heading or slugging), anything downstream of the choke, stratified flow, oil–water slip; the redraws' effect on where wells die, and how to use the logs for it.
6. **Distribution** (§3.6). Hugging Face; CC BY-NC 4.0; whether to mint a DOI (for Bjarne); no export controls known.
7. **Maintenance** (§3.7). Solution Seeker AS and its contact; errata in `docs/corrigendum.md`; a published dataset is never modified, so a fix is a new version (constitution); contributions through the GitHub repository.

### Against 3W

Vargas et al. (2019) publish the 3W dataset of real, simulated and hand-drawn instances of undesirable events in offshore wells. Three of its practices carry over: a fixed set of units for every file (§3, p. 6); each instance's source and label stated with it, the labels as numbered codes with names (§3, p. 6; Table 2, p. 7); and the dataset's own difficulties counted and stated, with benchmarks that fix how to train and test (§3, p. 6; §4, pp. 6–8). The v2 datasets do the same with fixed units (Section 2), coded labels (Section 4), the redraw and well logs as the counted difficulties, and the split. 3W stores one CSV file per instance in folders by label; v2 stores tables in parquet, because its instances are rows of features, not time series of varying length.

## 6. The generation

### Cost

R4's estimate, at the Rust core's median of 0.15 s per solve: `sol-2` needs 2.03 million solves (2 million first attempts and the best-rate solves), about 85 CPU hours; `nsol-2` about 2.4 million with the life scans, about 100 CPU hours. The redraws and the retries add to it:

| | `sol-2` | `nsol-2` |
|---|---|---|
| Without redraws or retries | 85 CPU h | 100 CPU h |
| Redraws at $r$ = 0.05 / 0.15 / 0.30 per sample | +4 / +13 / +25 CPU h | the same |
| CasADi retries on 0.5% / 2% of attempts, at 20 Rust solves each | +9 / +35 CPU h | the same |
| At $r$ = 0.15 and 0.5% retries | about 105 CPU h, 9 h on 12 processes | about 120 CPU h, 10 h on 12 processes |

So both datasets take about a day on half the machine (the Goals: 12 cores), less if the redraws and retries are rare. A well dropped by the producibility filter costs one to three solves; a well dropped at a redraw limit costs up to 600. The 0.15 s is a median: R6 measures the mean, which sets the total, and the cost of a table fluid's solve.

R6 measured 113 CPU hours for `sol-2` and 128 for `nsol-2`, about 22 hours on 12 processes for both (`r6-measurements.md`, Cost). Neither that projection nor the table above includes making the PVT tables or solving the table fluids on them: R6 solved them as black oil. The tables add about 2 to 15 CPU hours for `sol-2` and 5 to 49 for `nsol-2` at R6's 3200 table wells per dataset (`fluids-draft.md`, One table per well, or a library). The solves on tables are not measured; feature 018 found a table no dearer than its black oil on the Rust core, for oils only.

### The process pool or the batch API

- **The proposal** (D-93). The library's process pool (`manywells.sampling.generate.generate`) runs the wells in parallel, one well per task, with the Rust core solving one case per call. It is enough if three things hold, which R6 measures on the prototype: the time outside the core is small next to the solve (Python builds the well and the row per sample; the core well costs little to build); 12 processes fit in 32 GB with the CasADi backend loaded for the retries; and the pool's speed-up from 1 to 12 processes stays near 12, since a well's slots take about 30 s and 10,000 wells balance well.
- **The batch API** (backlog §4.3; `specs/architecture.md`, Rust core, design point 7) solves many operating points of one well in threads with the GIL released. The redraw passes (D-80) are natural batches: pass 0 is the well's 200 first attempts. It would save the time outside the core and the memory of 12 processes, but it is solver machinery and has to pay for itself (constitution, principle 7). Build it in R10 only if R6 measures a gain of more than about 20% in the generation's wall time; Bjarne decides.

### Seeds, versions and resumability

- **Seeds** (D-91). Each dataset has its own seed, recorded in `metadata.json`. Every draw comes from `rng_for(seed, …)` by its place (SMP-31 and R3's blocks; R4's slots; D-80's redraws), so a run is the same whatever the order of the wells or the number of processes. The generator draws $k$ = 0, 1, 2, … and keeps the first 10,000 wells it accepts, numbered by `ID` in that order.
- **Code version and configuration.** The generator records the configuration, $N$, the backends, the limits, `datasets.io.code_version` with the full commit, the SHA-256 of `uv.lock`, Python's version and the platform. It refuses to write a published dataset from a working tree with changes (`-dirty`), so that the commit reproduces it.
- **Resumability.** The generator works in chunks of 100 draw indices. Each chunk's samples, redraws and well-log rows go to one file, written to a temporary name and renamed when complete, with the run's settings and code version in its metadata. A restart checks that metadata and skips the chunks done; a mismatch stops the run. Since a well's result depends only on the seed, its index and the code, the assembled dataset is the same however often the run stops. An assembly step reads the chunks in order, keeps the first 10,000 accepted wells, assigns `ID` and `split`, and writes the published files with their checksums.
- **The machine.** The full generation runs within half the machine (12 processes, 32 GB, each process capped) and waits for Bjarne's go-ahead (the Goals). The pilot (R11) is a few hundred wells per dataset.

### Checks at generation (R11)

- the counts: 10,000 wells, 200 samples each, slots 0 to 199 once each, `ID` dense;
- every row passes SMP-28's row check (non-negative rates, fractions in [0, 1]) and v1's relations between the rates;
- every well passes `configurations.check(wp, 'v2.0.0')`;
- the re-solve check (Section 3);
- the redraw and drop shares against R6's targets (`sampling-draft.md`, What R6 must measure), and the shift between all draws and the accepted ones, reported in the datasheet;
- the split's shares per stratum.

## 7. What changes against v1

For the CHANGELOG and the v2 section of `docs/datasets.md` (R11). The v1 datasets and their files are never modified; their section of `docs/datasets.md` stays as it is.

- **Datasets.** Two, `manywells-sol-2` and `manywells-nsol-2`; there is no closed-loop dataset (`specs/goals.md`). Each has 10,000 wells of 200 samples, where v1 had 2,000 wells of 500.
- **Model.** The `v2.0.0` configuration in place of v1.0.0's model: deviated wells and drains to the heel, black oil with dissolved gas and PVT tables, real gas, friction from roughness, the energy terms, five flow regimes, the lift gas's injection depth, and Forchheimer's gas inflow if Bjarne approves feature 022.
- **Wells.** Drawn by the redesigned procedure (`specs/sampling.md`, v2), with every input and label in a wells table that replaces v1's config files.
- **Roots.** Every sample is the operating point, the stable root. v1's samples were whichever root the solver reached, some on the unstable trickle root (`docs/corrigendum.md`).
- **Files.** Parquet in place of zipped CSV; four tables and two logs per dataset, each a config on Hugging Face; a train–test split by well.
- **Samples.** New: `slot`, `attempt`, `PRES`, `GOR`, `WLR`; in `nsol-2` also `STAGE`, `TEST`, `SWEEP`, `LIFT`, `HTC`, `ROUGH` and the inflow's coefficient. Changed: `FGAS`, `FOIL` and `FWAT` are derived from the GOR and the water, not drawn; `FRBH` and `FRWH` take the five regimes; `WEEKS` is a float from the well's first production; the rates count the gas that leaves the oil as gas, as `datasets/rows.py` does.
- **Filters.** v1's three filters (SMP-28) and its sample floors are gone; a producibility filter drops wells that would not be produced. A failed draw is redrawn, as in v1, but every redraw is now logged and published.
- **Reproducibility.** Seeded draws, the code version and the configuration in each dataset's metadata, and a check that a row re-solves to its operating point. v1's datasets cannot be regenerated (SMP-31, S-9).

## Spec changes proposed

The design needs two edits to decided specs, drafted in `spec-changes.md` for sign-off: `specs/goals.md`, Compatibility → Datasets (the tables beyond the two, and the ruling on the redraws), and `specs/constitution.md`, Dataset reproducibility (the tables a row references). The schema itself goes into `docs/datasets.md` and the redraw rules into `specs/sampling.md` in R10 and R11.

## Open issues

- **For R6:** the redraw shares and the drops at each limit (`sampling-draft.md`, What R6 must measure); the time outside the core per solve and the pool's speed-up, for the batch API; the mean solve time with table fluids; the sizes of the files from the prototype's rows; a pilot's rows written in this schema for Bjarne's private comparisons.
- **Answered by R5:** the gas inflow's parameters and code names, `Forchheimer(a, b)` (feature 022), which name `forchheimer_a` and `forchheimer_b`; the table fluids' labels and parameters, and one table set per well (D-63 to D-67), which name the Table fluids columns.
- **For R10:** the helpers that build a case from the rows and a row from an operating point; the v2 row function with the new columns; the generator's passes, logs, chunks and assembly.
- **For Bjarne now:** nothing blocks R6. At the checkpoint: the schema as a whole; the same repository or a new one on Hugging Face (D-86); the redraw limits (D-81) and the re-solve tolerance (D-92); a DOI; and whether the logs are published (D-85, proposed yes).

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H. and Crawford, K. (2021), "Datasheets for datasets", *Communications of the ACM* 64(12), 86–92; arXiv preprint 1803.09010v8: §3.1 to §3.7, pp. 4–10.
- Vargas, R.E.V., Munaro, C.J., Ciarelli, P.M., Medeiros, A.G., do Amaral, B.G., Barrionuevo, D.C., de Araújo, J.C.D., Ribeiro, J.L. and Magalhães, L.P. (2019), "A realistic and public dataset with rare undesirable real events in oil wells", *Journal of Petroleum Science and Engineering* 181, 106223: §3 (p. 6), Table 2 (p. 7), §4 (pp. 6–8).
