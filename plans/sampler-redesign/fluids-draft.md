# Sampler redesign — draft procedure: the table fluids

*Step R5 of `plans/sampler-redesign-plan.md`, 2026-11-02. A draft for Bjarne's checkpoint (R8), written in the form of `sampling-draft.md`'s sections so that it can be merged into that file after its Mapping section; without SMP IDs. Plans are drafts, not specs.*

This part proposes how a volatile oil or a gas condensate becomes a PVT table: a composition drawn for the well's class and matched to its drawn descriptors, or a published example fluid, flashed by NeqSim and turned into a table by PVT-TAB-6. It replaces R3's interim mapping, in which these fluids stand in as black oil. It keeps the same three kinds apart:

- **Decided** (the Goals, 2026-10-30): volatile oils and gas condensates enter as tables from a NeqSim flash; the compositions are synthetic, drawn per fluid class from published ranges, plus published example fluids, each cited. Wet and dry gases stay black oil (D-36, `validity.md` decision 4).
- **Proposed:** every rule below, each resting on an entry of `decisions.md` (D-63 to D-68).
- **Needs a ruling before any condensate's table is built** (R6 ran the interim mapping, D-69): PVT-TAB-1 refuses a condensate's table today, because its $R_{so}$ falls as the pressure rises where liquid condenses (The table; D-64; `spec-changes.md`).

The numbers marked *proof* come from `plans/evidence/table_fluids.py` on the first two core volatile oils and the first two core gas condensates of R3's draws (wells 3, 8, 9 and 15, seed 2026), and on Pedersen et al.'s Table 3.8 condensate; NeqSim 3.4.0, Peng–Robinson with the Peneloux volume correction. Four fluids are a proof of the method, not a measurement of its distribution, which is R6's.

## Which wells take a table

- **Rule.** `fluid_class` volatile oil or gas condensate (`sampling-draft.md`, Fluid class): a table on the well's own grid, at the GORs its samples reach (`pvt_model = table`). Black oils, wet gases and dry gases: the black oil (`pvt_model = black_oil`). In the core 25% of the wells take a table (volatile oil 0.10, condensate 0.15, D-4), in the tail half.
- **Source of the fluid** (`fluid_source`): with probability 0.2 a published example fluid of the class (Published example fluids), else a synthetic composition (The composition). D-66.
- **Order.** The composition and its saturation pressure come after R3's fluid block, because they match its descriptors; then R3's stage block, whose GOR below the saturation pressure reads the table's $p_\text{sat}$; then the tables at the stage's GORs, before the inflow block, whose gas inflow reads the table's $\mu_g Z$. The composition's draws have a generator of their own, so the order moves no other block's draws.

## The composition

- **Components.** N₂, CO₂, C₁, C₂, C₃, i-C₄, n-C₄, i-C₅, n-C₅, C₆ (as n-hexane), and the heptanes-plus as five fractions (D-63).
- **Light ends.** A mixture of the class's published examples, normalised over N₂ to C₆, with weights $w \sim \text{Dirichlet}(1, \dots, 1)$:
  - gas condensate: Whitson and Brulé's (2000, Table 2.1) gas condensate, and Pedersen et al.'s (2015) Tables 1.3 and 3.8;
  - volatile oil: Whitson and Brulé's near-critical oil and volatile oil (Table 2.1), and Pedersen et al.'s Table 1.4 near-critical mixture.

  Then C₂ to C₆ are scaled by one factor $f_\text{int}$, the lever for the gas gravity (below), and the light ends renormalised.
- **Heptanes-plus.** Mole fraction before the GOR is set $z_{7+} \sim U(0.04, 0.10)$ for condensates and $U(0.10, 0.22)$ for volatile oils; molar mass $M_{7+} \sim U(140, 200)$ and $U(190, 240)$ g/mol; split into five fractions with boundaries at 135, 205, 275 and 415 g/mol (about C₉/C₁₀, C₁₄/C₁₅, C₁₉/C₂₀ and C₂₉/C₃₀) by Whitson's gamma distribution with shape $\alpha \sim U(0.5, 2.5)$ and $\eta$ from $\alpha$ (Whitson and Brulé 2000, Eqs. 5.13 to 5.15; "$\alpha$ usually ranges from 0.5 to 2.5 for reservoir fluids", §5.3.2). Each fraction's specific gravity by Søreide's $\gamma_i = 0.2855 + C_f(M_i - 66)^{0.13}$ (Eq. 5.44), with $C_f$ the lever for the API gravity.
- **Separation.** One stage at standard conditions (1.01325 bar, 288.15 K), as `pvt_table_eos.py` and PVT-TAB-1's GOR axis have it: every GOR of a well's table set then lies on one tie line, with the same $\rho_o$ and $\rho_{g,\text{sc}}$ (D-63).
- **What is matched, and how:**

  | Descriptor (R3's draw) | Lever | Match | Proof |
  |---|---|---|---|
  | GOR $R_{si}$ | recombination of the separator gas and the stock-tank oil at $R_{si}$ | exact, unless the fluid would lie above its saturation pressure at $p_i$ (next section) | 2 of 4 exact; 2 moved along the tie line, by factors 0.73 and 1.73 |
  | API gravity | Søreide's $C_f$ | Newton's method on both levers, to 0.5 °API and 0.005 | within 0.01 °API, in 3 iterations, for all 4; well 9's equilibrium phase (next section) then has 55.3 against 52.3 |
  | gas gravity $\gamma_g$ | $f_\text{int}$ | as above | within 0.0003 |
  | saturation pressure $p_\text{sat}$ at $T_r$ | none | follows from the flash; R3's $p_\text{sat}$ is replaced (D-65) | the table's against R3's draw: 386 against 513, 359 against 339, 255 against 250, 289 against 351 bar |
  | heptanes-plus content (McCain's class limits, a check) | – | follows | volatile oils 15.1 and 17.9 mol% (McCain: 12.5 to 30); condensates 1.4 and 6.0 mol% (below 12.5) |

  Matching $p_\text{sat}$ as well would need a third lever, such as $M_{7+}$, at the cost of the API gravity; with the separator's gas and oil fixed, the GOR and the saturation pressure are tied to each other, and R3 draws them independently.
- **The basis of the descriptors.** They are matched as single-stage values at standard conditions. The published examples show what that does to field-reported values, which come from separators at higher pressure: Whitson and Brulé's three examples, separated in one stage by the same flash, have GORs 12% to 19% higher, API gravities 3 to 6 °API lower and gas gravities 0.05 to 0.07 heavier than their Table 2.1 reports (Published example fluids). R3's descriptors of table fluids are then single-stage values; whether to shift R3's targets by these amounts is D-63.
- **Sources.** Whitson and Brulé (2000), Table 2.1, §5.3.2 and Eqs. 5.13 to 5.15 and 5.44; Pedersen et al. (2015), Tables 1.3, 1.4 and 3.8; McCain (1990), pp. 151–156, for the heptanes-plus limits of the classes [RF §8].
- **Seeding.** A block of its own, `rng_for(seed, k, 'v2', 'composition')` (`sampling-draft.md`, Seeding); the dataset's seed and the well's `draw_index` give it, and the composition's parameters are columns (`datasets-draft.md`, Table fluids), so that a user can make the table again.
- **Labels.** `fluid_source` (`synthetic` or the example's key), and the composition's parameters ($w$, $z_{7+}$, $M_{7+}$, $\alpha$, $C_f$, $f_\text{int}$) with the well.

## Published example fluids

- **Rule.** With probability 0.2 a table well of a class takes one of the class's examples, with equal shares, recombined along its own single-stage tie line to the well's drawn GOR. The example's stock-tank oil and separator gas are kept, so its API gravity and gas gravity replace the drawn ones; the GOR is matched as for a synthetic fluid (D-66).
- **The examples,** each with the reference that a dataset's `pvt` rows cite:

  | Key | Class | Reference | Heptanes-plus | Single-stage GOR, API, $\gamma_g$ (proof) | Reported |
  |---|---|---|---|---|---|
  | `wb2000_gas_condensate` | condensate | Whitson and Brulé (2000), Table 2.1 | 8.21 mol%, 184 g/mol, 0.816, lumped | 1153 Sm³/Sm³, 42.9, 0.749 | 971 Sm³/Sm³, 49, 0.70; $p_\text{sat}$ 452 bar |
  | `pedersen2015_t1_3` | condensate | Pedersen et al. (2015), Table 1.3 | C₇ to C₁₉ and C₂₀₊ | 1564, 42.1, 0.774 | – |
  | `pedersen2015_t3_8` | condensate | Pedersen et al. (2015), Table 3.8; constant-mass expansion at 155 °C, Table 3.9 | C₇ to C₁₉ and C₂₀₊ | 2338, 43.6, 0.777 | dew point 388 bar at 155 °C |
  | `wb2000_near_critical_oil` | volatile oil | Whitson and Brulé (2000), Table 2.1 | 10.04 mol%, 219, 0.839 | 771, 38.7, 0.775 | 650 Sm³/Sm³, 45, 0.71; $p_\text{sat}$ 484 bar |
  | `wb2000_volatile_oil` | volatile oil | Whitson and Brulé (2000), Table 2.1 | 21.76 mol%, 228, 0.858 | 297, 34.9, 0.774 | 265 Sm³/Sm³, 38, 0.70; $p_\text{sat}$ 374 bar |
  | `pedersen2015_t1_4` | volatile oil | Pedersen et al. (2015), Table 1.4 | C₇ to C₁₉ and C₂₀₊ | 599, 38.7, 0.815 | – |

  A lumped heptanes-plus is split as a synthetic one's, with $\alpha = 1$ and $C_f$ matched to its gravity. By McCain's GOR limits, Whitson and Brulé's volatile oil is a black oil (265 Sm³/Sm³) and their near-critical oil a condensate (650); the recombination to the drawn GOR puts each into the class it serves.
- **Candidates not taken:** Pedersen et al.'s Tables 3.6 (an oil at 198 Sm³/Sm³, a black oil by McCain) and 3.15 (a condensate with constant-volume depletion, not transcribed yet); Pedersen et al.'s Table 1.6 near-critical fluid (Yang et al. 1997), whose phase envelope was measured, is the next to add for volatile oils [RF §8a].
- **Check.** The one example with a measured saturation pressure at a stated temperature, Pedersen et al.'s Table 3.8 condensate, has a dew point of 356 bar at 155 °C in the flash, against 388 bar measured (−8%), with no tuning of the equation of state.

## The saturation pressure and the reservoir's state

- **Rule** (D-65). The table's saturation pressure at $T_r$ replaces R3's $p_\text{sat}$ (`sampling-draft.md`, Saturation pressure and the initial gas–oil ratio):
  - found by TP flashes on 10 bar steps up to $1.3\thinspace p_i$ and bisection to $10^{-3}$ bar on the edge of the two-phase region, with NeqSim's check for more phases on, without which its TP flash missed a condensate's 1 to 2 mol% of liquid up to 26 bar below the dew point (proof);
  - its kind, bubble or dew, from the incipient phase, is a label (`saturation_kind`), beside McCain's class (`class_rule`); a fluid that is an oil by GOR and a condensate at its $T_r$, or the reverse, keeps its class and shows it here;
  - **saturated at discovery:** where the fluid's $p_\text{sat}(T_r)$ exceeds $p_i$, it could not be one phase at $p_i$. Its GOR moves along its tie line until $p_\text{sat}(T_r) = p_i$, by bisection on $\ln R$ within a factor 4; where no GOR on the line does it, the reservoir fluid is the phase of its class (the vapour for a condensate, the liquid for an oil) in equilibrium at $(p_i, T_r)$, re-separated. Both are labelled (`saturation_rule`: `matched`, `gor_moved`, `equilibrium_phase`). Proof: well 8's volatile oil moved to 0.73 of its drawn GOR (555 to 403 Sm³/Sm³); well 9's condensate had no GOR on its line with a dew point below its $p_i$ of 255 bar, and its equilibrium vapour has 8864 Sm³/Sm³ (1.73 times the drawn), API 55.3 against 52.3, and its dew point at $p_i$.
- **The stage's GOR** (`sampling-draft.md`, Reservoir state at the life stage) is R3's formula with the table's $p_\text{sat}$ and the table's initial GOR in place of $R_{si}$.
- **Sources.** McCain (1990), pp. 151–156; Pedersen et al. (2015), §1.4: the class follows the mixture's critical temperature against $T_r$, which the flash knows.

## The table

- **Grid** (the well's own; D-67):
  - temperatures every 10 K from $10\thinspace\lfloor(T_s - 15\ \text{K})/10\rfloor$ to $10\thinspace\lceil(T_r + 1\ \text{K})/10\rceil$: the tubing's fluid stays warmer than $T_s$ (018, design choice 3), and the margin of 15 K is for Joule–Thomson cooling near a gas well's wellhead;
  - pressures 1, 2.5, 5 and 7.5 bar, then every 10 bar to $10\thinspace\lceil 1.05\thinspace p_i/10\rceil + 10$ bar, which covers `sol-2`'s drift of $p_r$ to $1.02\thinspace p_i$; every temperature's saturation pressure is a node, so that the bilinear interpolation keeps its kink (PVT-TAB-1, as `pvt_table_eos.py`);
  - proof: 9 to 18 temperatures and 41 to 81 pressures, 369 to 1116 nodes.
- **GOR axis** (`PVTTableSet`, PVT-TAB-7): one table per GOR that the well's samples reach, at most 25% apart, so that the axis errs by 0.01% to 0.3% at the median in no-slip density (018, design choice 5):
  - `sol-2`: from the smallest to the largest GOR the well's samples can reach, R4's drift formulas at the ends of their ranges (D-40: ±5% on the GOR, and for a saturated fluid the GOR's response to the ±2% on $p_r$, up to about ±30% where $k_g$ is 15 just below the saturation pressure); 3 GORs for most wells, R4's draws giving 0.92 to 1.09 of the nominal GOR for saturated oils at the 5th and 95th percentiles;
  - `nsol-2`: from the initial GOR to the largest of the life's $R_{go}$ (R4, Gas–oil ratio), with ratios of at most 1.25; a condensate whose GOR triples needs 6, an oil whose GOR rises tenfold 11.
- **Columns:** $R_{so}$, $B_o$, $Z$, $\mu_o$, $\mu_g$ and $\sigma_{go}$, by PVT-TAB-6 from the flash's vapour mass fraction $X$ and phase densities, with the separation's $\rho_o$, $\rho_{g,\text{sc}}$ and GOR. Where the stream is all liquid, above a bubble point, the gas columns are the incipient vapour's, flashed at $(p, T)$, as `pvt_table_eos.py` has them.
- **Where the stream is all vapour** (above a dew point, or below a lower dew point): $X = 1$ gives $B_o = 0$ and the oil phase no mass, which PVT-TAB-1 refuses ($B_o > 0$). Proposed (D-64): the table keeps a liquid mass fraction $\delta = 10^{-4}$, $X = 1 - \delta$, with the density, viscosity and surface tension of the liquid at the nearest two-phase node of the same temperature; also in the two-phase region where the flash's liquid is less than $\delta$. The model's equations need no change: $R_{so} = R_{go} - (1 - \delta)\thinspace m/\rho_{g,\text{sc}}$ is near $-\rho_o/\rho_{g,\text{sc}}$, PVT-OIL-13 moves all but $\delta$ of the oil into the gas, and the liquid is the water and a trace of oil. Proof: $B_o$ down to $5\cdot10^{-4}$, $R_{so}$ down to −848 Sm³/Sm³.
- **Retrograde condensation.** Below a dew point, liquid condenses as the pressure falls, so $R_{so}$ falls as the pressure rises: on the condensates' tables at 176, 248 and 316 pairs of neighbouring nodes. PVT-TAB-1 refuses that today ("R_so must not fall as the pressure rises"); the volatile oils' tables pass. The proposed change to PVT-TAB-1, which needs Bjarne's ruling before condensate tables are built, is in `spec-changes.md` (`specs/model/pvt/table.md`). The Rust core does not check the monotonicity. With the check relaxed, the evidence script builds `FluidModel` on every table and evaluates its phase rates and densities on floats at $(p_r, T_r)$ and at $(p_s, T_s + 20\ \text{K})$; no well was solved.
- **Format:** PVT-TAB-1's CSV, or the dataset's `pvt` rows in its columns (`datasets-draft.md`), with the header's `rho_o`, `rho_g` and `gor`, and the source: the well's `draw_index` (its composition's parameters are columns of `wells`) or the example's key, NeqSim's version and the equation of state.
- **Sources.** `specs/features/018-pvt-tables.md` (design choices 1 to 5) and `specs/model/pvt/table.md`.

## One table per well, or a library

- **Proposed** (D-67): one table set per table-fluid well, on its own grid and GOR axis, made when the well is drawn.
- **Alternative:** a library of fluids per class, each a table set with a GOR axis over the class's whole GOR range and a grid over every well's pressures and temperatures, assigned to a well by its nearest API and gas gravity.

| | Per well (proposed) | Library |
|---|---|---|
| Matched | GOR exactly; API and $\gamma_g$ within the tolerance; grid and GORs fitted to the well | GOR exactly by the axis; API and $\gamma_g$ to the nearest fluid |
| Fluids in a dataset | about 2500, one per table well (about 3200 at R6's mix; below) | a few hundred |
| Cost (NeqSim, one process) | 0.1 to 1 s to match, 0.05 to 2.6 s for the saturation pressure, 0.5 to 4.6 s per GOR (proof); a `sol-2` well with 3 GORs about 2 to 17 s, an `nsol-2` well with up to 11 GORs about 6 to 55 s | one class's library of 100 fluids with 12 GORs on a grid of about 2000 nodes, a few hours |
| Size | 370 to 1100 nodes of 9 values per GOR, up to 100 kB of CSV each: up to 0.75 GB of CSV for 2500 wells with 3 GORs, before parquet's compression | fewer fluids, but each table larger (a wider grid, more GORs) |
| Re-solving | the dataset's own rows | the same |
| Diversity | every well a fluid of its own | fluids repeat across wells |

The cost does not decide it: 2500 wells take about 1 to 12 hours of one process for `sol-2` and 4 to 38 hours for `nsol-2`, at most about 3 hours on 12 processes; each process's JVM holds about 0.6 GB (proof: 0.59 GB of resident memory for the whole run). Per well keeps the drawn descriptors, which a library would round to its nearest fluid, and gives every table well a fluid of its own. The size is R7's to weigh; a 20 bar step above 100 bar would halve it, at the medium grid's errors of 018.

**The count of table wells.** The numbers above, and D-66 and D-67, assume 2500 table wells per dataset: a quarter of 10,000 wells, the core's share. The tail draws half its wells as volatile oils and condensates, and R6's accepted `sol-2` wells are 32% table fluids (128 of 400; `r6-measurements.md`, Outcomes), so a dataset has about 3200. Scaled by 3200/2500, about 1.28:

- making the tables takes about 2 to 15 hours of one process for `sol-2` and 5 to 49 hours for `nsol-2`, at most about 1.3 and 4 hours on 12 processes;
- `sol-2`'s tables take up to about 1 GB of CSV before compression;
- about 640 wells take a published example, about 100 per example.

These are scaled from the proof's four fluids, not measured on a dataset. Neither R6's projection of the generation (`r6-measurements.md`, Cost) nor R7's (`datasets-draft.md`, Cost) includes them.

## How a table replaces the interim mapping

- **Maps to:** `FluidModel(pvt=PVTTableSet(tables), gor=R_go, wlr, rho_w, cp_g, cp_o, cp_w)`, the surface tension model `'oil'` (the table's $\sigma_{go}$ is used where it has the column), in place of the interim `BlackOil`. $\rho_o$ and $\rho_{g,\text{sc}}$ are the table's, which the match makes those of the drawn API and gas gravity within its tolerance. `configurations.check(wp, 'v2.0.0')` accepts it as it is (`PVTTable`, `PVTTableSet`).
- **What changes,** on the proof's wells, the table against the interim black oil at the drawn API, gas gravity and GOR (`table_fluids.py check`):
  - the volatile oils' bubble points at $T_r$: Vazquez–Beggs gives 556 and 426 bar where the tables give 386 and 359;
  - the free gas's mass share of the stream at the separator's pressure and 20 K above $T_s$: 0.30 and 0.22 with the tables, 0.34 and 0.30 with the black oil;
  - the condensate above its dew point at $(p_r, T_r)$ (well 15): the table has all the hydrocarbon in the gas, a free-gas share of 0.84 with the water as the liquid; the black oil keeps the condensate liquid, a share of 0.43, so the bottomhole's mixture is heavier and its liquid rate higher;
  - the condensate below its dew point (well 9): 0.89 against 0.83.

  The rerun of R6's pilots with the tables (R9) measures what these do to the operating points.
- **The gas-rate inflow** of a condensate takes $\mu_g Z$ of the table's gas at $(p, T_r)$ for its pseudo-pressure (`specs/features/022-gas-rate-inflow.md`, Units).

## Validity

- **PVT-TAB-2's clamp.** The grid covers $p_r$ and $T_r$ by construction, so the simulator's refusal of $p_r$ or $T_r$ off the grid does not arise; `nsol-2`'s $p_r$ only falls. A root's points can leave the grid only below $T_s - 15$ K or below 1 bar: the root then carries `off_table` (`pvt/table.md`). The rerun of R6's pilots with the tables counts the share of operating points with it, by fluid class; above a share of 1% the margin of 15 K widens.
- **The GOR axis.** A GOR outside the axis raises `ValueError` (PVT-TAB-7). The axis spans every GOR the well's samples reach, by the rules above; a sample outside it is an `invalid_input` of the generation log (`sampling-draft.md`, The generation log), which should not occur.
- **PVT-TAB-1.** Condensates need the proposed change (The table). $R_{so} < 0$ needs $\mu_o$ and $\sigma_{go}$, which every table here has.
- **The equation of state.** Peng–Robinson with Peneloux and NeqSim's default interaction parameters, untuned: Pedersen et al.'s Table 3.8 condensate's dew point 8% below the measured (above). The surface tension near the critical point falls to $5\cdot10^{-9}$ N/m in a table (well 15); it enters the slip through fourth roots (`validity.md`, note 4).
- **Ranges of the draws.** A table holds whatever the flash gives; no correlation's range applies to it. The classes' heptanes-plus contents stay within McCain's limits (The composition).

## Checks a test can run

- every table passes PVT-TAB-1 as changed, and builds a `FluidModel`;
- the table's GOR equals the drawn $R_{si}$ for `saturation_rule = matched`, to rounding; its API and gas gravity within 0.5 °API and 0.005 of the drawn;
- $p_\text{sat}(T_r) \le p_i$ for every table, to the bisection's $10^{-3}$ bar;
- the heptanes-plus content within McCain's limits for the class;
- the grid covers $p_r$ and $T_r$, and the GOR axis every sample's GOR;
- a rerun gives the same compositions and tables to the bit, on the same NeqSim version.

## Open issues

- **For Bjarne now:** the change to PVT-TAB-1 that lets a condensate's table in (D-64; `spec-changes.md`). No condensate table can be built without it, or a temporary subclass that only the CasADi backend runs.
- **For the rerun of R6's pilots with the tables (R9; R6 ran the interim mapping, D-69):**
  - the tables of R3's 2000 wells: the shares of each `saturation_rule`, the distributions of the table's $p_\text{sat}$ against R3's, of the GOR moves and of the API and gas-gravity errors;
  - the solved wells with the tables against the interim black oil, on both backends, and the share of `off_table` roots;
  - whether the Rust core's march and temperature solve hold with $R_{so}$ falling in $p$ (its assumption tests on a test well with a condensate table);
  - the cost and size per dataset, for R7.
- **For R2's second pass:** a public set of condensate and volatile-oil compositions with their measured saturation pressures, for a check of the flash beyond one example (Nemeth and Kennedy 1967 and Elsharkawy 2002, on R2's resource list); Pedersen et al.'s Tables 1.6 and 3.15 transcribed.

## References

- McCain, W.D. (1990), *The properties of petroleum fluids*, 2nd ed., PennWell: pp. 151–156.
- Pedersen, K.S., Christensen, P.L. and Shaikh, J.A. (2015), *Phase behavior of petroleum reservoir fluids*, 2nd ed., CRC Press: §1.4, Tables 1.3, 1.4, 3.8 and 3.9.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20: Table 2.1, §5.3.2, Eqs. 5.13 to 5.15 and 5.44.
- NeqSim 3.4.0, an open-source Java library: Peng–Robinson with the Peneloux volume correction, its TBP characterisation and default interaction parameters.
- The proof: `plans/evidence/table_fluids.py` (`plans/evidence/README.md`, Sampler redesign: the model's gaps).
