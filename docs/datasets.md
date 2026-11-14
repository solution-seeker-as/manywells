# Dataset description

The published ManyWells datasets are on [Hugging Face](https://huggingface.co/datasets/solution-seeker-as/manywells):
`manywells-sol-1` (stationary, open loop), `manywells-nsol-1` (non-stationary, open loop) and `manywells-nscl-1`
(non-stationary, closed loop). Each has one million samples, 500 from each of 2000 wells, generated with ManyWells
v1.0.0, and a config file, `<dataset>_config.zip`, with the parameters each well was simulated with. The
[README](../README.md#datasets) shows how to load them, and [`corrigendum.md`](corrigendum.md) lists their errata.
The published datasets are never modified.

The generators in `scripts/data_generation/` write new datasets with the same features
([`simulate.md`](simulate.md), Generating datasets). Their config file holds each well's draws
(`manywells.sampling.wells.WellDraw`), whose columns differ from those of the published config files. Its trajectory
(SMP-41) takes four columns: `trajectory`, the kind (`vertical`, `deviated` or `l_shaped`); `kickoff_fraction` and
`inclination` (degrees) of a deviated well; and `horizontal_length` (m) of an L-shaped one, empty where they do not
apply.
v1.0.0's generators wrote pickle files, which its scripts `read_dump.py` and `read_dump_nonstationary.py`
compiled into the published datasets; they are at the
[`v1.0.0` tag](https://github.com/solution-seeker-as/manywells/tree/v1.0.0/scripts/data_generation).

### Dataset features

| Feature | Description                                                       |  Unit |
|:--------|:------------------------------------------------------------------|------:|
| ID      | Unique well ID                                                    |     - |
| CHK     | Choke position in \[0, 1\]                                        |     - |
| PBH     | Pressure bottomhole                                               |   bar |
| PWH     | Pressure wellhead                                                 |   bar |
| PDC     | Pressure downstream choke                                         |   bar |
| TBH     | Temperature bottomhole                                            |     K |
| TWH     | Temperature wellhead                                              |     K |
| WGL     | Mass flow rate, lift gas                                          |  kg/s |
| WGAS    | Mass flow rate, gas excl. lift gas                                |  kg/s |
| WLIQ    | Mass flow rate, liquid                                            |  kg/s |
| WOIL    | Mass flow rate, oil                                               |  kg/s |
| WWAT    | Mass flow rate, water                                             |  kg/s |
| WTOT    | Mass flow rate, total incl. lift gas                              |  kg/s |
| QGL     | Volumetric flow rate at standard conditions, lift gas             | Sm³/h |
| QGAS    | Volumetric flow rate at standard conditions, gas excl. lift gas   | Sm³/h |
| QLIQ    | Volumetric flow rate at standard conditions, liquid               | Sm³/h |
| QOIL    | Volumetric flow rate at standard conditions, oil                  | Sm³/h |
| QWAT    | Volumetric flow rate at standard conditions, water                | Sm³/h |
| QTOT    | Volumetric flow rate at standard conditions, total incl. lift gas | Sm³/h |
| FGAS    | Inflow gas mass fraction                                          |     - |
| FOIL    | Inflow oil mass fraction                                          |     - |
| FWAT    | Inflow water mass fraction                                        |     - |
| WEEKS   | Weeks since first data point                                      |    7d |
| CHOKED  | Boolean indicating if flow is choked                              |     - |
| FRBH    | Flow regime at bottomhole (represented by string)                 |     - |
| FRWH    | Flow regime at wellhead (represented by string)                   |     - |

The following relations apply:
- WLIQ = WOIL + WWAT
- WTOT = WLIQ + WGAS + WGL
- QLIQ = QOIL + QWAT
- QTOT = QLIQ + QGAS + QGL

The flow regime features, FRBH and FRWH, may take on one of the following string values: 
'bubbly', 'slug-churn', 'annular'. These are the labels of v1.0.0's fitted classifier, which the v1 datasets and the `v1.0.0` configuration use. Datasets made with `develop`'s default slip model, the five regimes (`specs/features/019-five-flow-regimes.md`), take 'bubbly', 'dispersed-bubbly', 'slug', 'churn' or 'annular'. 

The WEEKS feature is only present in non-stationary datasets.

## v2 datasets: `manywells-sol-2` and `manywells-nsol-2`

*Draft, 2026-11-11 (Step R11 of `plans/sampler-redesign-plan.md`). The schema needs Bjarne's sign-off: D-85 to D-94 are ruled, and R10's D-122 to D-126, which this section follows, await their rulings (`plans/sampler-redesign/decisions.md`). The v2 datasets are not published yet. The v1 datasets and the text above are unchanged.*

*A full generation before the final one is a beta (SMP-50, D-149). The first beta's datasets are `manywells-sol-2-beta1` and `manywells-nsol-2-beta1`, with `release` `beta1` in their `metadata.json`. This section describes the datasets that the code generates as it stands, which carry `beta2` (D-153): against the first beta's, their `wells` add the columns `design_rate_oe` and `choke_dp_share`, their beans are sized to the wells' rates (SMP-77), and their logs add the reason `below_start_hurdle` (SMP-102); these are working assumptions, of which Bjarne has taken two values (D-150 to D-153). The final datasets take the plain names, after the betas' analysis.*

The v2 datasets are new datasets with a schema of their own (`specs/goals.md`, Compatibility):

| | `manywells-sol-2` | `manywells-nsol-2` |
|---|---|---|
| Kind | stationary: a well's samples vary around one date of its life | non-stationary: a well's samples span its life, in date order |
| Wells | 10,000 | 10,000 |
| Samples per well | 200 operating points | 200 operating points: 160 operating samples and 8 test periods of 5 |
| Model | the `v2.0.0` configuration with 100 cells, solved by the Rust core | the same |
| Procedure | `specs/sampling.md`, SMP-50 to SMP-117 | the same |
| Datasheet | [`datasheet-sol-2.md`](datasheet-sol-2.md) | [`datasheet-nsol-2.md`](datasheet-nsol-2.md) |

Every sample is an operating point, the stable root (SOL-4, SOL-6); unstable roots are not published. A draw without an operating point is redrawn, and the redraw is logged (SMP-104, SMP-105). `scripts/data_generation/v2/generate_dataset.py` makes a dataset end to end with `manywells.sampling.v2`: the table fluids' build, the run over the wells in resumable chunks, and the assembly of the files.

### Files

A dataset is a folder of four tables, each a parquet file, with a companion folder of its two logs next to it (SMP-108, SMP-110; D-126):

```
manywells-sol-2/
    wells.parquet         one row per accepted well: its inputs that do not vary, its labels, its split
    samples.parquet       one row per sample: its inputs that vary, and its operating point
    geometry.parquet      one row per survey station
    pvt.parquet           one row per node of a table fluid's PVT tables
    metadata.json         the generation's records, and each file's rows and SHA-256
manywells-sol-2-logs/
    drawn_wells.parquet   the well log: one row per drawn well, accepted or not
    redraws.parquet       the redraw log: the attempts without an operating point, and those of dropped wells
    metadata.json         the same records, with `companion_of`
```

`manywells-nsol-2/` and `manywells-nsol-2-logs/` hold the same files. Each input is in one place (D-87): an input that is constant over a well's samples is in `wells`, one that varies is in `samples`. So in `nsol-2`, where the heat transfer, the roughness, the inflow's coefficients and the lift's installation change over the life, they are in `samples`. A row of `samples`, its well's row of `wells`, the well's `geometry` rows and, for a table fluid, its `pvt` rows carry every input of the sample's `WellProperties` and `BoundaryConditions` (`specs/constitution.md`, Dataset reproducibility). The train–test split is the `wells` column `split`. On Hugging Face each table is to be a config of `solution-seeker-as/manywells`, under `v2/`, next to the unchanged v1 files (D-86); that layout is release work.

### Reading a dataset and re-solving a sample

```python
from manywells.datasets.v2 import read_dataset, resolve_sample, sample_case, sample_rows

data = read_dataset('manywells-sol-2')  # 'wells', 'samples', 'geometry' and 'pvt' as DataFrames, and 'metadata'
logs = read_dataset('manywells-sol-2-logs')  # 'drawn_wells' and 'redraws', and 'metadata'
df = data['samples'].merge(data['wells'], on='ID')  # each sample with its well's inputs and labels

well, sample, survey, pvt = sample_rows(data, ID=0, slot=0)
wp, bc = sample_case(well, sample, survey, pvt)  # WellProperties and BoundaryConditions in v2.0.0
features = resolve_sample(well, sample, survey, pvt)  # the operating point's features, solved again
```

A sample re-solved from its rows with the code version and backend in `metadata.json`, on the generator's platform, gives its features to the bit; elsewhere they agree within the verifier's `tol_x` applied to them (SMP-110, D-92). NeqSim is not needed: the dataset carries its PVT tables.

### Conventions

- **Units** are the model's (`specs/model/nomenclature.md`): m, bar (absolute), K, kg/s, kg/m³, J/(kg K), W/(m² K), and Sm³/Sm³ at standard conditions (PVT-GAS-2). The samples' standard volumetric rates stay in Sm³/h, as v1's; the wells' rates (`design_rate`, `floor`, `best_rate`) are in Sm³/d, the best rate and the floor as oil equivalent, oil plus gas/1000. Dates and durations are in years in `wells`, except `well_age` in days, and in weeks in `samples`.
- **Names** (D-88). The samples keep v1's features by name and add new ones in their style, short and upper case. The wells table names a model input by its code name (`rho_w`, `K_c`, `injection_md`) and a descriptor by a short lower-case name (`z`, `water_depth`).
- **Types** (D-88). Every continuous quantity is float64, so that a re-solve uses the generator's inputs to the bit. `ID` is int32 (nullable, Int32, in the logs), `draw_index` int64, `slot`, `attempt`, `station` and `n_cells` int16, the counts int32, and the flags `CHOKED` and `LIFT` boolean. A label is a pandas category whose categories are its codes in their order (Labels and codes, below). A value that does not apply, such as the injection depth of a natural well, is null, never a sentinel; pandas reads a null float as NaN.
- **(s)** marks a column of `sol-2` only, **(n)** one of `nsol-2` only.
- **v1** says how a column relates to v1: *kept*, a v1 feature (above), or a field of v1's well draws (`manywells.sampling.wells.WellDraw`), with the same name and meaning; *changed*, kept with a changed meaning; *renamed*, a v1 quantity under a new name, the old name first; *new*, not in v1.

### `samples`

One row per sample, sorted by `ID` and `slot`. In `nsol-2` the slots are in date order.

**Keys and date**

| Column | Unit | Type | Meaning | v1 |
|---|---|---|---|---|
| `ID` | – | int32 | the well, as in `wells` | kept |
| `slot` | – | int16 | the slot, 0 to 199 (SMP-105) | new |
| `attempt` | – | int16 | the attempt that gave the operating point: 0 for the slot's first draw, $a$ for its $a$-th redraw (SMP-105) | new |
| `WEEKS` (n) | weeks | float64 | the sample's date: time since the well's first production (SMP-97), at 52.1775 weeks per year | changed: v1's weeks since the first sample, one per week |
| `STAGE` (n) | – | category | the life stage of the date (SMP-91) | new |
| `TEST` (n) | – | Int8 | the test period, 1 to 8; null for an operating sample (SMP-99) | new |
| `SWEEP` (n) | – | category | what a test sweeps, `choke` or `lift`; null for an operating sample (SMP-99) | new |

**The sample's inputs**

| Column | Unit | Type | Meaning (model input) | v1 |
|---|---|---|---|---|
| `CHK` | – | float64 | choke position $u$ (`BoundaryConditions.u`; SMP-93, SMP-98) | kept |
| `PDC` | bar | float64 | pressure downstream of the choke (`BoundaryConditions.p_s`; SMP-95) | kept |
| `WGL` | kg/s | float64 | lift-gas rate (`BoundaryConditions.w_lg`; SMP-94, SMP-98); 0 for a natural well | kept |
| `PRES` | bar | float64 | reservoir pressure (`BoundaryConditions.p_r`; SMP-96, SMP-85) | new |
| `GOR` | Sm³/Sm³ | float64 | producing gas–oil ratio (`FluidModel.gor`; SMP-96, SMP-100) | new |
| `WLR` | – | float64 | water–liquid ratio at standard conditions (`FluidModel.wlr`; SMP-96, SMP-100) | new |
| `LIFT` (n) | – | bool | lift installed at the date: the well has its injection point (SMP-90) | new |
| `HTC` (n) | W/(m² K) | float64 | `ThermalModel.h` at the date (SMP-88) | new |
| `ROUGH` (n) | m | float64 | `RoughnessFriction.roughness` at the date (SMP-88) | new |
| `KL`, `WLMAX`, `FORCHA`, `FORCHB` (n) | as `k_l`, `w_l_max`, `forchheimer_a`, `forchheimer_b` in `wells` | float64 | the inflow's coefficients at the date (SMP-88); null where the well's `inflow` does not use them | new |

**The operating point** (SMP-30's features, from the stable root)

| Column | Unit | Type | Meaning | v1 |
|---|---|---|---|---|
| `PBH`, `PWH` | bar | float64 | pressure at the bottomhole and at the wellhead | kept |
| `TBH`, `TWH` | K | float64 | temperature at the bottomhole and at the wellhead | kept |
| `WGAS`, `WLIQ`, `WOIL`, `WWAT` | kg/s | float64 | the reservoir stream's gas (without the lift gas), liquid, oil and water | changed: stock-tank oil and the gas that leaves it |
| `WTOT` | kg/s | float64 | WLIQ + WGAS + WGL | kept |
| `QGL`, `QGAS`, `QLIQ`, `QOIL`, `QWAT`, `QTOT` | Sm³/h | float64 | the same at standard conditions, with QTOT = QLIQ + QGAS + QGL | changed, as the mass rates |
| `FGAS`, `FOIL`, `FWAT` | – | float64 | the reservoir stream's mass fractions | changed: derived from GOR, WLR and the densities, not drawn |
| `CHOKED` | – | bool | critical flow at the choke (CHK-12) | kept |
| `FRBH`, `FRWH` | – | category | the flow regime at the bottomhole and at the wellhead: SLIP-16's five regimes | changed: five regimes in place of v1's three |

The rates are the reservoir's at standard conditions (`datasets/rows.py`, SMP-30): with gas dissolved in the oil, `WOIL` is the stock-tank oil and `WGAS` holds the gas that leaves it, so v1's relations between the rates hold. The samples hold no solver diagnostics.

### `wells`

One row per accepted well, sorted by `ID`.

**Identity and split**

| Column | Unit | Type | Meaning | v1 |
|---|---|---|---|---|
| `ID` | – | int32 | the well's ID, 0 to 9,999 in draw order; unique within a dataset only. The wells are quotas by stratum, which fill at different draw indices, so the last IDs are mostly subsea wells (SMP-110) | kept |
| `draw_index` | – | int64 | the well's place $k$ in the sequence of drawn wells, which seeds its draws (SMP-109) | new |
| `split` | – | category | `train` or `test` (SMP-110) | new |
| `configuration` | – | string | `v2.0.0` | new |
| `n_cells` | – | int16 | the grid's cells, 100 (SMP-111) | new |

**Labels** (their codes are under Labels and codes)

| Column | Type | Meaning | v1 |
|---|---|---|---|
| `coverage_tail` | category | the realistic `core` or the coverage `tail` (SMP-53) | new |
| `well_type` | category | `platform`, `subsea` or `onshore` (SMP-54) | new |
| `fluid_type`, `fluid_class` | category | the fluid type, and its class with gas split into wet and dry gas (SMP-55) | new |
| `class_rule` | category | the rule that classes the fluids, McCain's (1990) limits (SMP-55) | new |
| `drive` | category | `depletion` or `pressure_support` (SMP-56) | new |
| `life_stage` (s) | category | the well's drawn stage, within which its date lies; a tail well's date is its life's end (SMP-56, SMP-92) | new |
| `life_state` (s) | category | the stage, with `late` split by the drive into `depleted` and `watered_out` (SMP-52) | new |
| `pressure_class` | category | `under`, `normal` or `over` (SMP-63) | new |
| `saturation` | category | the reservoir against its saturation pressure (SMP-68): `sol-2` at its date, `nsol-2` at its start | new |
| `geometry_class` | category | the inclination class (SMP-59) | renamed: `trajectory` → `geometry_class` and `geometry_family` |
| `geometry_family` | category | the family that drew the survey (SMP-59, SMP-60) | (as above) |
| `pvt_model` | category | `black_oil`, or `table` for volatile oils and condensates (SMP-112) | new |
| `fluid_source` | category | a table fluid's source: `synthetic`, or a published example's key (SMP-112, SMP-114); null for a black oil | new |
| `saturation_kind` | category | a table fluid's saturation point at $T_r$: `bubble`, `dew`, or `none` without one (SMP-115); null for a black oil | new |
| `saturation_rule` | category | how a table fluid was made one phase at $p_i$ (SMP-115); null for a black oil | new |
| `inflow` | category | the inflow model (SMP-74) | new |
| `tubing` | category | the tubing's nominal size in inches (SMP-75) | new |
| `choke_model` | category | the choke model (SMP-77) | new |
| `chk_profile` | category | the choke's profile (SMP-77) | kept |
| `lift` | category | `natural` or `gas_lift`: in `sol-2` at its date, in `nsol-2` lifted at some date of its life (SMP-52, SMP-90) | renamed: `has_gas_lift` → `lift` |
| `lift_start_rule` | category | the rule that started the well's lift (SMP-90, SMP-101, SMP-103, SMP-106); null for a natural well | new |

The other groups' columns are float64 unless their row says otherwise.

**Depth and geometry** (the survey itself is in `geometry`)

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `z` | m | the reservoir's true vertical depth below the mudline (SMP-57) | new |
| `water_depth` | m | $d_w$; 0 onshore (SMP-58) | new |
| `L` | m | the true vertical depth of the bottom of the flow path below the wellhead (GEO-3, SMP-58) | kept |
| `md_bottom` | m | the bottom's measured depth, the survey's last station | new |
| `max_inclination` | ° | the largest cell inclination from vertical on the model's grid | new |
| `kickoff_tvd` | m | the kickoff's true vertical depth $K$; null for a vertical well (SMP-60) | renamed: `kickoff_fraction` → `kickoff_tvd` |
| `build_rate`, `build_rate_2` | °/30 m | the build rate, and a `high_angle` well's landing build (SMP-60) | new |
| `hold_inclination` | ° | the hold's inclination; a `high_angle` well's first tangent's (SMP-59, SMP-60) | renamed: `inclination` → `hold_inclination` |
| `drop_rate`, `end_inclination`, `end_section_tvd` | °/30 m, °, m | an `s_shaped` well's drop rate, last inclination and last section's true vertical depth (SMP-60) | new |
| `heel_inclination`, `tangent_md` | °, m | a `high_angle` well's inclination at the heel, and the measured length of its tangent there (SMP-59, SMP-60) | new |
| `drain_length` | m | the horizontal drain that a `high_angle` well's inflow stands for; the drain has no cells (SMP-61, SMP-71) | renamed: `horizontal_length` → `drain_length`, no longer modelled |

**Water, pressure and temperature**

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `salinity` | – | the formation water's mass fraction of dissolved solids (SMP-62) | new |
| `rho_w` | kg/m³ | the water's density at standard conditions (`FluidModel.rho_w`; SMP-62) | kept |
| `p_h` | bar | the hydrostatic pressure at the reservoir (SMP-63) | new |
| `overpressure_ratio` | – | $\lambda = (p_i - p_h)/(p_\text{lith} - p_h)$; negative for an underpressured reservoir (SMP-63) | new |
| `p_i` | bar | the initial reservoir pressure (SMP-63) | new |
| `T_s` | K | the surface temperature (`BoundaryConditions.T_s`; SMP-64) | kept |
| `geothermal_gradient` | K/km | the gradient below the mudline (SMP-65) | new |
| `T_r` | K | the reservoir temperature (`BoundaryConditions.T_r`; SMP-65) | kept |

**Separator and fluid**

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `p_sep`, `T_sep` | bar, K | the host's first-stage separator (`BlackOil.p_sep`, `T_sep`; SMP-66) | new |
| `p_s_nom` (s) | bar | the nominal pressure downstream of the choke at the date, around which `PDC` is drawn (SMP-66, SMP-89); a tail well's is SMP-66's tail draw as drawn (SMP-92) | renamed: `p_s` → `p_s_nom` |
| `r_s` (s) | – | the separator's factor at the date (SMP-66, SMP-89); 1 for a tail well (SMP-92) | new |
| `dp_flowline` | bar | a subsea well's flowline and riser loss; 0 otherwise (SMP-66) | new |
| `api`, `sg_gas` | °API, – | the stock-tank oil's API gravity and the gas gravity (SMP-67); for a published example, or a fluid taken as its equilibrium phase, the fluid's own as reported at the separator (SMP-114, SMP-115) | new |
| `rho_o` | kg/m³ | the stock-tank oil's density (`BlackOil.rho_o`, or the table's) | kept |
| `rho_g` | kg/m³ | the gas's density at standard conditions (`BlackOil.rho_g`, or the table's) | renamed: `R_s`, the gas constant that follows from it → `rho_g` |
| `p_sat` | bar | the saturation pressure at $T_r$ (SMP-68; a table fluid's flash, SMP-115); null for wet and dry gas | new |
| `R_si` | Sm³/Sm³ | the initial producing GOR (SMP-68; a table fluid's after SMP-115) | new |
| `pvt_table_id` | – | string: the well's tables in `pvt`, `k` and the draw index in six digits; null for a black oil | new |

**Table fluids** (`pvt_model` `table`; every column null for a black oil, and the composition's for a published example)

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `comp_w1`, `comp_w2`, `comp_w3` | – | the weights of the class's three published examples in the light ends (SMP-113) | new |
| `z_c7p`, `M_c7p` | –, g/mol | the heptanes-plus's mole fraction before the GOR is set, and its molar mass (SMP-113) | new |
| `alpha_c7p` | – | the gamma distribution's shape that splits the heptanes-plus (SMP-113) | new |
| `C_f`, `f_int` | – | Søreide's factor and the factor on C₂ to C₆, matched to the gravities (SMP-113) | new |
| `c7_mol` | – | the reservoir fluid's heptanes-plus mole fraction, after its GOR is set | new |
| `gor_moved` | – | the factor by which SMP-115's rule moved the GOR; null where `matched` | new |
| `fluid_redraws` | – | the draws of the well's fluid rejected before the kept one, because their built fluid was not of the well's class (SMP-115); 0 for a well that keeps its own draw | new |
| `api_drawn`, `sg_gas_drawn`, `R_si_drawn`, `p_sat_drawn` | °API, –, Sm³/Sm³, bar | the `fluid` block's draws, those of the fluid's draw that was kept (SMP-115), kept as records where the table replaces them (SMP-68, SMP-114, SMP-115) | new |
| `offset_api`, `offset_sg_gas`, `offset_gor` | °API, –, – | the fluid's offsets of the separator's report from one stage at standard conditions: the gravities' differences and the GORs' ratio (SMP-113) | new |
| `field_api`, `field_sg_gas`, `field_gor` | °API, –, Sm³/Sm³ | the fluid as reported through the well's separator and a stock tank (SMP-113) | new |
| `n_gors`, `table_nodes` | – | the GORs on the table set's axis, and the nodes of its tables (SMP-116) | new |

**Reservoir state** (`sol-2`: the nominal state at the well's date, around which the samples drift; `nsol-2`: the life's parameters)

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `life_date` (s) | years | the well's date in its life (SMP-92) | new |
| `p_r_nom` (s) | bar | the nominal reservoir pressure at the date (SMP-92) | renamed: `p_r` → `p_r_nom` |
| `gor_nom`, `wlr_nom` (s) | Sm³/Sm³, – | the nominal producing GOR and WLR at the date | renamed: `fractions` → `gor_nom`, `wlr_nom` |
| `wgr_nom` (s) | Sm³ per 10⁶ Sm³ | a gas well's nominal water–gas ratio; null for oils | new |
| `kgko_weight` | – | an oil's blend of the $k_g/k_o$ curves, 0 in the core (SMP-70); null for gas classes | new |
| `gor_factor_cond` | – | a condensate's GOR factor $c_k$ (SMP-70); null otherwise | new |
| `T_life`, `T_end`, `T_scan` (n) | years | the life span (SMP-84), the life's end, and the end that the life scan gave it (SMP-101); `T_end` is earlier than `T_scan` where a dead date ended the life (SMP-106) | new |
| `t_plateau`, `decline_D`, `decline_b` (n) | years, 1/year, – | the depletion clock's plateau and Arps's $D_\text{A}$ and $b_\text{A}$ (SMP-84) | new |
| `gamma_p`, `r_p_start`, `r_p_end` (n) | – | the pressure's exponent, and $p_r/p_i$ at the life's start and end (SMP-83, SMP-85) | new |
| `F_bt`, `wlr_bt` (n) | – | the water's breakthrough recovery, and an oil's WLR after it (SMP-86) | new |
| `water_start`, `water_end` (n) | – or Sm³ per 10⁶ Sm³ | the water at the life's start and end: an oil's WLR, a gas well's WGR (SMP-83, SMP-86) | new |
| `t_sep_1`, `t_sep_2`, `r_s_mid`, `r_s_late` (n) | years, – | the separator's step-down dates and factors (SMP-89) | new |
| `p_change` (n) | – | the probability of a set-point change (SMP-98) | new |
| `lift_start` (n) | years | the date lift starts; null for a natural well (SMP-90, SMP-101) | new |

**Productivity and inflow**

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `permeability` | mD | the permeability, for gas classes an effective one (SMP-71) | new |
| `net_pay`, `skin`, `ln_re_rw` | m, –, – | the net pay, the skin and $\ln(r_e/r_w)$ (SMP-71) | new |
| `drawdown_design` | – | the design drawdown, a fraction of $p_r$ (SMP-71) | new |
| `anisotropy` | – | the anisotropy of a deviated well; null for a vertical one (SMP-71) | new |
| `productivity_factor` | – | the geometry's factor on the radial productivity (SMP-72) | new |
| `J` | Sm³/(d bar) | an oil well's productivity index at its initial-stage draw (SMP-72); null for gas classes | new |
| `non_darcy_D` | d/Sm³ | a gas or condensate well's non-Darcy coefficient (SMP-71); null for oils | new |
| `k_l` (s) | kg/(s bar) | `ProductivityIndex.k_l` at the date (SMP-111); null unless `inflow` is `productivity_index` | new |
| `w_l_max` (s) | kg/s | `Vogel.w_l_max` at the date (SMP-111); null unless `inflow` is `vogel` | kept |
| `forchheimer_a`, `forchheimer_b` (s) | bar² s/kg, bar² s²/kg² | `Forchheimer.a` and `Forchheimer.b` at the date (SMP-117); null unless `inflow` is `forchheimer` | new |
| `design_rate` | Sm³/d | the liquid rate the tubing was sized for (SMP-75) | new |
| `design_rate_oe` | Sm³/d | the design rate in oil equivalent, which a platform or subsea well's start hurdle is set against (SMP-102) | new |
| `floor` | Sm³/d | the economic floor in oil equivalent (SMP-102) | new |
| `best_rate`, `best_rate_w_lg` | Sm³/d, kg/s | the best rate in oil equivalent and its lift-gas rate (SMP-103): `sol-2` at its date, `nsol-2` at its start | new |

**Completion, choke, lift and heat**

| Column | Unit | Meaning | v1 |
|---|---|---|---|
| `D` | m | the tubing's inner diameter (SMP-75) | kept |
| `roughness` (s) | m | `RoughnessFriction.roughness` at the date (SMP-76, SMP-88) | kept |
| `roughness_start`, `roughness_end` (n) | m | the roughness at the life's start and end, between which it ages (SMP-88) | new |
| `choke_dp_share` | – | the share $\xi$ of the pressure upstream of the choke that the fully open choke takes at its sizing (SMP-77) | new |
| `bean_ratio`, `C_D` | – | the fully open bean's share of the tubing's diameter, the bean being sized to the well's rate, and the discharge coefficient (SMP-77) | new |
| `K_c` | m² | `SimpsonChokeModel.K_c`, the sized choke's, held below the tubing's erosional velocity (SMP-77) | kept |
| `injection_md`, `injection_tvd` | m | the operating valve's measured and true vertical depth (`WellProperties.injection_md`; SMP-79) | new |
| `T_lg` | K | the lift gas's temperature (`BoundaryConditions.T_lg`; SMP-79) | new |
| `w_lg_max` | kg/s | the lift-gas capacity (SMP-80) | new |
| `glr_cap`, `p_inj`, `dp_valve`, `packer_offset`, `psi_lg` | Sm³/Sm³, bar, bar, m, – | the lift design's draws: the injected GLR at capacity, the injection pressure, the valve's drop, the packer's height above the bottom, and the lift gas's share of the way to $T_r$ (SMP-79, SMP-80) | new |
| `h` (s) | W/(m² K) | `ThermalModel.h` at the date (SMP-81, SMP-88) | kept |
| `U_t`, `k_e`, `well_age` | W/(m² K), W/(m K), d | the draws behind the heat transfer, at the initial-stage draw (SMP-81) | new |
| `cp_o`, `cp_w`, `cp_g` | J/(kg K) | the heat capacities (`FluidModel`; SMP-82) | kept |
| `flags` | – | string: the draws' truncations, redraws and fallbacks that applied, joined by `;`, such as SMP-60's `geometry_redrawn:n`, SMP-75's `design_rate_above_tubing_capacity`, SMP-77's `choke_at_erosional_rate`, a choke made smaller to hold the well below its tubing's erosional velocity, or `erosional_cap_unmet` where its search did not reach it, and SMP-94's `lift_gas_always_on`, a lifted `sol-2` well that cannot flow without lift gas and so has no sample without it. The lift block's flags, `injection_at_packer`, `injection_raised_to_0.3L` and `lift_capacity_at_bound`, are those of the lift design that the well uses (SMP-90); a well that is not lifted has none | new |

The lift design's columns, `injection_md` to `psi_lg`, are null for a well without lift: in `sol-2` a well natural at its date, in `nsol-2` a well with no lifted date in its life (`lift` `natural`), whose draws may have held a design that its life never used.

**Generation summary** (int32)

| Column | Meaning | v1 |
|---|---|---|
| `n_redraws` | the redraws of the well's slots: its attempts beyond each slot's first | new |
| `n_redraws_no_op`, `n_redraws_solver` | its attempts without an operating point, by outcome: `no_operating_point` and `solver_failure` (SMP-104) | new |
| `n_redraws_<reason>` | the same by reason, one column for each of SMP-104's six: `p_r_le_p_s`, `no_root`, `exception`, `invalid_row`, `incomplete_scan` and `unstable_only` | new |
| `n_dead_dates` (n) | the dead dates that changed the well's life, each by starting its lift or ending the life (SMP-106); the counts above are those of the slots' last pass | new |

v1's `f_D`, a fixed friction factor, has no counterpart: `v2.0.0` takes friction from the roughness.

### `geometry`

One row per survey station, sorted by `ID` and `station`. The stations are the drawn survey's: two for a vertical well, one per degree along a build or a drop (SMP-60). `WellGeometry.from_survey(md, tvd, n_cells, D)` builds the grid from them (GEO-4).

| Column | Unit | Type | Meaning | v1 |
|---|---|---|---|---|
| `ID` | – | int32 | the well | new |
| `station` | – | int16 | the station, 0 at the wellhead | new |
| `md`, `tvd` | m | float64 | measured and true vertical depth below the wellhead | new, in place of `trajectory` |

### `pvt`

One row per node of a table, in the columns of a PVT table file (`PVTTable.read_csv`, PVT-TAB-1), so that `manywells.datasets.v2.pvt_from_rows` rebuilds a well's table set to the bit. Each table well has its own set, on its own grid and GOR axis (SMP-116), so a `pvt_table_id` belongs to one well. Every column is new.

| Column | Unit | Type | Meaning |
|---|---|---|---|
| `pvt_table_id` | – | string | the table set, as in `wells` |
| `gor` | Sm³/Sm³ | float64 | the GOR of the node's table on the set's axis (PVT-TAB-7) |
| `p`, `T` | bar, K | float64 | the node |
| `R_so`, `B_o`, `Z` | Sm³/Sm³, –, – | float64 | PVT-TAB-1's columns |
| `mu_o`, `mu_g`, `sigma` | Pa s, Pa s, N/m | float64 | the optional columns; null where a table has none |
| `rho_o`, `rho_g` | kg/m³ | float64 | the table's densities at standard conditions, the same in every row of a set |
| `name`, `source`, `separation` | – | string | the table's free text; `source` names the build, NeqSim's version and its equation of state, and the fluid: `synthetic` with its draw index, or the example's key |

### `redraws`: the redraw log

One row per slot attempt without an operating point, of every drawn well, and one per slot attempt of a dropped well, its operating points included (SMP-108). With `samples` it holds every draw of the accepted wells' slots, so the inputs of all draws and of the accepted ones can be compared column by column. The best-rate solves and the life scan's are not in it.

| Column | Unit | Type | Meaning |
|---|---|---|---|
| `draw_index` | – | int64 | the well's draw index |
| `ID` | – | Int32 | the well's ID; null for a well that was not accepted |
| `slot`, `attempt` | – | int16 | the attempt |
| `CHK` to `WLR`, and `WEEKS` to `SWEEP` and `LIFT` to `FORCHB` (n) | as `samples` | as `samples` | the attempt's date and inputs, as in `samples` |
| `outcome` | – | category | `operating_point` (an attempt of a dropped well), `no_operating_point` or `solver_failure` (SMP-104) |
| `reason` | – | category | why the attempt has no operating point (SMP-104); null for `operating_point` |
| `n_roots`, `n_stable` | – | int8 | the size of the attempt's root set, and its stable roots |

The search counts and times of the attempts stay with the run (SMP-108).

### `drawn_wells`: the well log

One row per drawn well, accepted or not, in draw order: each stratum's wells (the tail, and the core's platform, subsea and onshore wells) from its first to the last of its quota's accepted wells (SMP-110). A draw index of a stratum whose quota was reached earlier is not in the log. It has every column of `wells`, with `ID` and `split` null for a well that was not accepted, and:

| Column | Type | Meaning |
|---|---|---|
| `outcome` | category | `accepted`, why the well was dropped (SMP-106), or `crash` where its generation raised |
| `not_producible_reason` | category | the best rate's status of a well dropped for it (SMP-103), its choke's sizing's (SMP-77), `below_start_hurdle` for a platform or subsea well whose design rate lies below its start hurdle (SMP-102), or `no_fluid_of_class` for a table well none of whose fluid draws gives a fluid of its class (SMP-115); null otherwise |
| `error` | string | the message of an `invalid_input` or `table_failed` well, or of a well without a fluid of its class, or a crash's traceback, the home directory written as `~`; null otherwise |
| `n_attempts` | int32 | the attempts of its slots; 0 for a well dropped before its slots |
| `n_several_stable` | int32 | its samples whose root set had more than one stable root (SOL-6) |
| `best_rate_outcomes` | list of strings | the outcomes of its best-rate solves, one per lift-gas rate (SMP-103; `nsol-2`: at its start), or of its choke's sizing's solves for a well dropped there (SMP-77) |
| `scan_best_rate` (n) | list of float64 | the life scan's best rates in Sm³/d, up to the point that ends the life (SMP-101); NaN where a point has no operating point |
| `scan_outcome` (n) | list of strings | the statuses of those points (SMP-101, SMP-103) |
| `seconds`, `cpu_seconds`, `table_seconds` | float64 | the well's wall and CPU time, and its table build's, in seconds; they vary with the machine |

A crashed well's row holds its draw index, outcome, error and times only. The well log is the table to use for the effect of the well filters, since it holds the dropped wells' inputs.

### `metadata.json`

The records of SMP-110: `dataset`, `release` (`beta1`, `beta2` and so on for a beta, null for the final datasets), `kind`, `seed`, `n_wells`, `quotas` (the wells by stratum: `tail`, `platform`, `subsea` and `onshore`), `n_slots`, `configuration`, `n_cells`, `backend` (`rust`) and `retry_backend` (null, D-122); `limits` and `producibility` (the floors, the margin, the start hurdles, the best rate's lift-gas fractions, the life scan's points and the water cut); `split` (the test share and the strata); `chunk_size`; `tables`, the folder of tables relative to the run's, and `table_build` (the SHA-256 of its `wells.json` and `axes.json`, the draw indices and table wells it covers, and NeqSim's version); `code`, the code that ran the wells, and `assembled_by`, the code that wrote the files, each with its `version` (`datasets.io.code_version`), full `commit` and `dirty`; `lock_sha256`, the SHA-256 of `uv.lock`; `environment`, the versions of Python, numpy and scipy and the machine's architecture that the run had, which the assembly must have too; `created`; `python`, `platform` and `machine`, the assembly's; `counts`, the wells drawn and accepted, the wells by outcome, the slots' attempts by outcome, `samples_off_table` (the samples with a point off their PVT table's grid, PVT-TAB-2), `dead_dates` (n: those that started a lift and those that ended a life) and the CPU seconds; and `files`, each file's rows and SHA-256. The logs' copy adds `companion_of`.

### Labels and codes

A label is stored as its string, and its code is its position in the list (D-89); a later version adds categories at the end (`manywells.datasets.v2`, `_CODES`).

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
| `fluid_source` | wells | 0 `synthetic`, 1 `wb2000_gas_condensate`, 2 `pedersen2015_t1_3`, 3 `pedersen2015_t3_8`, 4 `wb2000_near_critical_oil`, 5 `wb2000_volatile_oil`, 6 `pedersen2015_t1_4` (SMP-114) |
| `saturation_kind` | wells | 0 `bubble`, 1 `dew`, 2 `none` |
| `saturation_rule` | wells | 0 `matched`, 1 `gor_moved`, 2 `equilibrium_phase` |
| `inflow` | wells | 0 `vogel`, 1 `productivity_index`, 2 `forchheimer` |
| `tubing` | wells | 0 `2-7/8`, 1 `3-1/2`, 2 `4-1/2`, 3 `5-1/2`, 4 `7`, 5 `9-5/8` |
| `choke_model` | wells | 0 `simpson` |
| `chk_profile` | wells | 0 `linear`, 1 `sigmoid`, 2 `convex`, 3 `concave` |
| `lift` | wells | 0 `natural`, 1 `gas_lift` |
| `lift_start_rule` | wells | 0 `from_start`, 1 `static_check`, 2 `end_of_natural_flow` (n), 3 `drawn_time`, 4 `no_natural_flow` (the producibility filter's lift, SMP-103, in `nsol-2` from the life's start; n: also a dead date's, SMP-106) |
| `split` | wells | 0 `train`, 1 `test` |
| `SWEEP` | samples (n) | 0 `choke`, 1 `lift` |
| `FRBH`, `FRWH` | samples | 0 `bubbly`, 1 `dispersed-bubbly`, 2 `slug`, 3 `churn`, 4 `annular` (SLIP-16) |
| `outcome` | redraws | 0 `operating_point`, 1 `no_operating_point`, 2 `solver_failure` |
| `reason` | redraws | 0 `p_r_le_p_s`, 1 `no_root`, 2 `exception`, 3 `invalid_row`, 4 `incomplete_scan`, 5 `unstable_only` |
| `outcome` | drawn_wells | 0 `accepted`, 1 `not_producible`, 2 `best_rate_solver_failure`, 3 `invalid_input`, 4 `slot_limit` (s), 5 `redraw_limit`, 6 `solver_failure_limit`, 7 `dead_date` (n), 8 `short_life` (n), 9 `table_failed`, 10 `crash` |
| `not_producible_reason` | drawn_wells | 0 `water_cut`, 1 `below_floor`, 2 `no_operating_point`, 3 `solver_failure`, 4 `below_start_hurdle`, 5 `no_fluid_of_class` |

Reasons 0 and 1 make an attempt `no_operating_point`, reasons 2 to 5 a `solver_failure` (SMP-104).

### What changes against v1

- **Datasets.** Two, `manywells-sol-2` and `manywells-nsol-2`, each of 10,000 wells with 200 samples, where v1 had 2,000 wells with 500. There is no closed-loop dataset (`specs/goals.md`).
- **Model.** The `v2.0.0` configuration in place of v1.0.0's model: deviated wells and drains to the heel, black oil with dissolved gas and PVT tables, real gas, friction from roughness, the energy terms, five flow regimes, the lift gas's injection depth, and Forchheimer's inflow for gas wells.
- **Wells.** Drawn by the v2 procedure (`specs/sampling.md`, SMP-50 to SMP-117), with every input and label in a `wells` table in place of v1's config files.
- **Roots.** Every sample is the operating point, the stable root. v1's samples were whichever root the solver reached, some on the unstable trickle root (`corrigendum.md`).
- **Files.** Parquet in place of zipped CSV: four tables and two logs per dataset, and a train–test split by well.
- **Samples.** New: `slot`, `attempt`, `PRES`, `GOR`, `WLR`; in `nsol-2` also `STAGE`, `TEST`, `SWEEP`, `LIFT`, `HTC`, `ROUGH` and the inflow's coefficients. Changed: `FGAS`, `FOIL` and `FWAT` follow from the GOR and the water, not drawn; `FRBH` and `FRWH` take the five regimes; `WEEKS` is a float from the well's first production; the rates count the gas that leaves the oil as gas.
- **Filters.** v1's three filters (SMP-28) and its floors on the samples are gone. A producibility filter drops wells that would not be produced (SMP-103). A failed draw is redrawn, as in v1, but every redraw is now logged and published (SMP-108).
- **Reproducibility.** Seeded draws (SMP-109), the code version and the configuration in each dataset's records, and a re-solve of a row from the dataset alone (SMP-110). v1's datasets cannot be regenerated (SMP-31, S-9).
