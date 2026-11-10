# Sampler redesign — the second checkpoint (after R9)

*2026-11-09. The smaller checkpoint that `checkpoint.md` (section 8) and the plan's step R8 call for: it covers only what changed since Bjarne's rulings of 2026-11-06, namely R9's integration of feature 022's inflow and the per-well tables, and the findings that the rework's and R9's pilots left open. Plans are drafts, not specs. Nothing proposed here is decided until he rules.*

**Ruled:** L1, L2, L3 and L5 on 2026-11-09; L4, L6 and the points of R10's spec draft (section 3) on 2026-11-10. Each ruling is under its item.

**Citations.** D-n is an entry of `decisions.md`. *Rework* and *R9* are `rework-measurements.md` and `r9-measurements.md`, the pilots of 400 `sol-2` and 100 `nsol-2` wells on the same seeds and draw indices (seeds 2026 and 2027). [own analysis of …] is the maintainer's analysis of a publisher's open data, kept outside the repo (the maintainer's decision, 2026-11-06): the repo states what it shows, not its values.

## 1. What changed since the first checkpoint

- **The rework** implemented the rulings of 2026-11-06 in the draft's scripts and `sampling-draft.md`, and its pilots led to five more rulings that day: one life for both datasets (D-105), D2 with D-62's non-Darcy coefficient and the gas wells' permeability at 1/14 of the oil wells' law, the lift-gas capacity truncated (D-31), a floor on the lift-gas set point (D-47), and R4's control change kept for redraws (D-83). Each is under its decision entry.
- **Feature 022** (#38, Forchheimer's law in $p^2$) and **K1's retrograde tables** (#39, option A′) are merged into `develop`.
- **R9's integration** (`plans/evidence/README.md`, R9's integration): the prototype fits each gas and condensate well's Forchheimer coefficients to the pseudo-pressure at every reservoir state (D-61), and builds a NeqSim table set for every volatile oil and condensate (D-63 to D-68) with J2's per-fluid offsets and D-106's repair of the tables' $p/Z$. Its pilots (R9) rerun the rework's on the same draws:
  - the black oils solve as in the rework, bit for bit;
  - wells kept: 82% of drawn `sol-2` wells (core 85%, tail 70%) and 87% of `nsol-2`, as in the rework;
  - the condensates' gas rates fall by a quarter and their oil rates by half; the wet gas wells' by a tenth (Forchheimer's inflow alone);
  - the decline of a life and the agreement of the two datasets by stage (D-105) hold;
  - the cost rises by 70% (`sol-2`) and 110% (`nsol-2`), mostly from 8 slow volatile oils (L6).

## 2. Rulings needed

For each option: what it means physically, what it changes in the model and for the machine-learning tasks, and how common the case is in real wells.

| Item | Settled by | Left to choose |
|---|---|---|
| L1 · D-106, the tables' $p/Z$ repair | NeqSim's flash artefacts near critical conditions (R9's check run) | Approve, and the limit of 5% |
| L2 · The tables' saturation pressures | The fluid's own flash (D-63, D-65 as ruled) | Confirm that R3's undersaturation draw no longer applies to table fluids |
| L3 · $b = 0$ in D-61's fit | The pseudo-pressure's physics at high pressure | Approve as a known limit of the $p^2$ form |
| L4 · Gas wells with pressure support | The regions' stopped gas wells; the Gulf's gas sands by drive | Ruled 2026-11-10: they water out (D-107) |
| L5 · The platform oil wells' upper tail | Coverage first (the Goals) | Keep, or narrow it |
| L6 · The slow table wells | A Rust core finding: DAK's Joule–Thomson pole | Ruled 2026-11-10: the guard approved |
| Noted, no ruling | H3's targets as ruled, H4's well 62, D-105's stages | – |

### L1 · Drop a table's pressure nodes where $p/Z$ falls (D-106)

**Question.** NeqSim's TP flash, near a hot fluid's near-critical saturation point, splits the stream into two near-identical liquids or misses the split at a single node. The table's $p/Z$ then falls with pressure at one or two nodes, and PVT-TAB-1 refuses the table. Approve D-106's repair?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **D-106:** drop the nodes off the longest run along which $p/Z$ rises, at every temperature of that GOR's table; fail a table that would lose more than 5% of its pressures | The dropped nodes are flash errors, not physics; the bilinear interpolation bridges them | R9 dropped 33 pressures from 19 of 176 `sol-2` table sets and 61 from 17 of 48 `nsol-2` sets, 1 or 2 of 75 to 119 pressures each; one `nsol-2` volatile oil lost 3 of 50 at one GOR (6%) and is `table_failed` | Hot condensates and volatile oils near critical conditions are the fluids this hits |
| Drop the wells whose tables fail | – | R9's check run lost 12 of 86 table sets this way, 5 of `nsol-2`'s 9 condensates, all in hot reservoirs: a bias against hot table fluids | – |
| Relax PVT-TAB-1's $p/Z$ check | – | Lets real table errors through; K1 added the check to catch them | – |

**Recommendation.** Approve D-106 with its 5% limit. Count `table_failed` wells with the drops that H3's targets judge on R11's pilot: one of the 224 table sets in R9's pilots failed.

**Ruling:** Bjarne, 2026-11-09: approved as recommended.

### L2 · The tables' saturation pressures replace R3's (D-65, D-17)

**Question.** Should the table wells keep the saturation pressures their flashes give, now that these differ from R3's draw?

**What R9 found.** The tables move $p_\text{sat}$ from R3's by a factor of 0.46 to 2.2 (P10 to P90; median 0.99) and put it lower against the initial pressure: $p_\text{sat}/p_i$ has a median of 0.80 on the tables, against R3's 0.93 (condensates) and 0.98 (volatile oils). 53 of the 154 `sol-2` table wells change their saturation label, in both directions. The share saturated at discovery stays about the same: 26 of 86 against R3's 21 in the check run.

**Physically.** R3 drew $p_\text{sat} = p_i(1 - x)$ and, for these classes, $R_{si}$ separately: a stand-in, because a fluid's saturation pressure follows from its composition. The table's composition is matched to the drawn API gravity, gas gravity and GOR (D-63), so the flash fixes $p_\text{sat}$, and $x$ is no longer free. D-65, as ruled, already makes the table's $p_\text{sat}$ replace R3's. Where the flash puts $p_\text{sat}$ above $p_i$, D-65 moves the GOR so that the fluid is saturated at discovery.

**Model and ML tasks.** The saturation pressure decides Vogel's or the productivity index's inflow for a volatile oil (D-22) and where the GOR starts to rise in a life (D-44). The condensates' lower liquid rates (oil at 0.53 of the interim black oil's) come with the tables' yields and saturation pressures. R9 did not separate the two causes.

**In real wells.** Volatile oils and condensates are found both saturated, under a gas cap or as an oil rim, and undersaturated. No public count of these classes by saturation is in hand. R3's shares saturated at discovery (0.40 and 0.30) and its median undersaturation of 5% were design values.

**Recommendation.** Confirm: R3's undersaturation draw $x$ (D-17) applies to black oils only, and a table well's saturation pressure and label are its flash's. No change to the scripts.

**Ruling:** Bjarne, 2026-11-09: confirmed as recommended.

### L3 · $b = 0$ in D-61's fit for most gas wells

**Question.** D-61 fits INF-10's $a$ and $b$ to the pseudo-pressure IPR by least squares of $(p_r^2 - p_{wf}^2)/q$ against $q$. Where the free slope comes out negative, $b = 0$ and $a$ is the mean, as INF-10 needs $b \ge 0$. In R9, $b = 0$ in every attempt of at least half the gas and condensate wells (104 of 174 `sol-2` wells have it in some attempt), and in 60% of the states in the check run. Accept this?

**Physically.** In the pseudo-pressure, $p_r^2 - p_{wf}^2 = \int \mu_g Z\thinspace dm$ over the drawdown. At high pressure $\mu_g Z$ grows with $p$. So as the rate rises and $p_{wf}$ falls, each added unit of $m$ adds less $p^2$, and $(p_r^2 - p_{wf}^2)/q$ falls with $q$, a negative slope. Non-Darcy flow gives a positive one. With D-62's smaller non-Darcy coefficient (R3's was 8.7 times Eq. 4-34's at the median), the gas's own curvature wins in most high-pressure wells. The $p^2$ form fits the pseudo-pressure only where $\mu_g Z$ changes little over the drawdown, which is at low pressure.

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Keep D-61 as ruled** ($b = 0$ where the slope is negative) | Darcy flow in $p^2$; the non-Darcy term is lost where it is small anyway | Where $b = 0$, the fit's largest error from 20% drawdown to the AOF is 5.4% at the median and 22% at the 95th percentile (0.7% and 5.6% where $b > 0$). The datasets stay consistent with the model, which is what the ML tasks learn | High-pressure gas wells with weak non-Darcy flow |
| Weight the fit towards the samples' drawdowns | – | Smaller error where the samples lie; more machinery for a shape the model cannot hold | – |
| An inflow in $p$, not $p^2$, at high pressure | The textbook form at high pressure | A new model option: outside feature 022 as approved | – |

**Recommendation.** Keep D-61. Record the $p^2$ form's limit at high pressure, and the fit's error, as a known limit in `specs/features/022-gas-rate-inflow.md` and `specs/model/validity.md` (a spec change, for sign-off with R10's specs).

**Ruling:** Bjarne, 2026-11-09: approved as recommended.

### L4 · Gas wells with pressure support decline little

**Question.** Over a life, a core gas well with pressure support keeps about three quarters of its first year's rate in the last fifth of its life (0.67 in the rework, 0.76 in R9). Depletion gas wells keep 0.18, and the regions' stopped gas wells keep about a tenth in every setting of the Gulf [own analysis of BOEM's OGOR-A production]. How should a supported gas well's life end?

**Why they do not decline.** D-105 ends a supported life with the reservoir pressure at R3's late draw (0.89 of $p_i$ at the median) and the water at a WGR of 1000 Sm³ per 10⁶ Sm³, the coverage tail's top. INF-11 adds the water to the gas inflow at the WGR, so the water takes no inflow capacity from the gas. At 1000 the water is too little to load the tubing, and at nearly the initial pressure the well flows almost as it did at the start.

**In real wells.** In the Gulf's gas sands, water drive and partial water drive hold most of the completed wells, and depletion drive far fewer. The water-drive sands recover the least gas, partial water drive more, depletion the most [own analysis of BOEM's sands atlas]. So water-drive gas wells stop while their reservoir pressure is still high: they water out. The Gulf's gas wells' WGR in their later years has an upper quartile near the draft's end of 1000 and an upper decile several times above it [own analysis of BOEM's OGOR-A production].

**Measured** (`nsol-2`, 100 wells, seed 2027, the interim mapping without tables; only the 21 core gas wells with pressure support change, and every well's outcome is the same in all runs):

| Run | Last fifth over first year, median (P75) | Last rate over first year | $p_r/p_i$ at the end | Lives cut by the floor | Redraws per sample | CPU per well |
|---|---|---|---|---|---|---|
| As drafted | 0.78 (0.86) | 0.78 | 0.89 | 0 of 21 | 0.029 | 53 s |
| A · water out: end WGR 10,000 | 0.27 (0.39) | 0.21 | 0.89 | 1 of 21 | 0.130 | 63 s |
| B · partial water drive: end $p_r/p_i \sim U(0.3,\ \text{late draw})$ | 0.53 (0.72) | 0.54 | 0.61 | 3 of 21 | 0.110 | 55 s |
| C · both | 0.27 (0.41) | 0.20 | 0.62 | 9 of 19 | 0.083 | 62 s |
| Depletion gas wells, for comparison | 0.16 (0.22) | 0.12 | 0.19 | 8 of 14 | – | – |

Under A, half the supported gas wells' samples have a WGR above 1000 (P90 7800), and no attempt fails in the solver. C's median is A's, and two of its supported gas wells are dropped (one at the redraw limit, one with a dead date), so C keeps 19.

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **A · Water out** (recommended): the end WGR 10,000, near the upper decile of the Gulf's late gas well-years; the tail's top raised with it, so that the tail still spans the core | The water drive keeps the pressure up and the wells die of water, by liquid loading in the tubing | More samples near liquid loading and in churn flow, which G1 found thin. Further outside INF-10's validity (`validity.md`: single-phase gas near the well, the draft extrapolating to a WGR of 500): real gas inflow would also fall as water reaches the well, which INF-11 ignores, so the model's watered-out wells decline less than real ones, not more | The Gulf's commonest drive for gas sands, and its lowest recovery |
| B · Partial water drive | The aquifer slows the pressure's fall but does not stop it | Lower pressures in late gas wells, more of them below hydrostatic; the water stays within the draft's range | Partial water drive is common in the Gulf's gas sands too |
| C · Both | Both mechanisms | Two changes for one finding, with no further decline than A's and two wells dropped | – |
| Keep | Supported gas wells stay strong to the end | Few dying gas wells under support in `nsol-2` | Unlike the regions' stopped wells |

**Recommendation.** A: one number, the mechanism the Gulf's recoveries point to, and a decline within the regions' spread. A raises the sampler's WGR bound from 1000 to 10,000, so the tail's draw becomes $\text{LogU}(1, 10\thinspace 000)$ and `validity.md`'s row moves with it; both are measured on the rerun.

**Ruling:** Bjarne, 2026-11-10: A, as recommended (D-107). The rerun of 2026-11-09 already took it as a working assumption.

**The rerun with A** (`r9-measurements.md`, The rerun after the second checkpoint): with the tables, the supported gas wells' last fifth is 0.25 (R9: 0.76), the core gas wells' median-sample P50 128,000 Sm³/d (R9: 154,000), and `nsol-2`'s outcomes unchanged; `sol-2`'s tail keeps 66% of its drawn wells against R9's 70%, as its gas wells now reach a WGR of 10,000.

### L5 · The platform oil wells' upper tail

**Question.** The rework's first run found the platform oil wells' upper tail of median-sample rates above every region's [own analysis of BSEE's well tests, ANP's production by well and NSTA's production by well]. R9's core offshore oil wells (platform and subsea together) make 6 / 91 / 1985 Sm³/d of oil equivalent (P10 / P50 / P90), 19% of them above 1018 Sm³/d. Narrow the tail?

**Where the tail comes from** (draws only, 3000 wells, seed 2026; the design reference rate $q_{l,\text{ref}}$, not the solved rate). Core platform oils reach above 1018 Sm³/d in 18% of the wells. By geometry class: vertical 14%, slightly deviated 16%, deviated 12% and highly deviated 28%, where half are drains (D-21's Joshi index with friction, drains of 300 to 2500 m). The narrower permeability spread of the rework (1.2 in place of 1.5) did not move the tail. So the tail is spread over every class and every draw, with the drains adding most, and no single draw sets it.

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Keep, and measure it by well type on R11's pilot** | High-rate platform wells, horizontal producers among them | More samples at high rates, where friction takes a larger share of the pressure drop | The per-well data in hand come from mature basins (the Gulf, Brazil's post-salt, a few UK wells). Norway publishes production by field only; its drill-stem tests of black oils, exploration wells on test chokes, already reach into the pilot's upper tail [own analysis of Sodir's FactPages: drill-stem tests] |
| Lower the platform oils' permeability median | Weaker platform wells overall | Moves the whole platform distribution, not only its tail; the subsea wells, already below the regions at the median, are untouched | – |
| Shorten the drains | Fewer long horizontal producers | Undoes part of B1/D-21 as ruled | Drains over 1 km are on public record (B1) |

**Recommendation.** Keep, under coverage first, shape second: the regions in hand are lower bounds on the support, and the platforms' tail is not out of range. Report it by well type on R11's pilot. Revisit only if the pilot shows platform rates above the tubing's erosional capacity (D-100's check).

**Ruling:** Bjarne, 2026-11-09: kept; measured by well type on R11's pilot.

### L6 · The slow table wells: a Rust core finding

**Question.** In R9, 8 of the 76 volatile oils on their tables took 540 to 2930 s each, 37% of `sol-2`'s CPU. Approve the fix on the branch `fix/table-jt-temperature-solve` (from `develop` at `d6c0b7d`, not pushed)?

**Root cause.** A NeqSim table with viscosity and surface-tension columns gives finite properties at every temperature (PVT-TAB-2 clamps them). The one term that can blow up is DAK's Joule–Thomson factor (PVT-GAS-10), which the core evaluates at the table's gas density. At large drawdowns the first cell chokes, and its energy row has no root above DAK's pole, where $Z + \rho_r Z_\rho = 0$, here 122 K below the table's lowest temperature. The row changes sign only across the pole. Brent's method converges onto it, and the march continues with $J$ near $\pm 10^{15}$ through every cell, at about 80 ms a march against 1 ms. Whether a high-rate march survives changes from one $p_0$ to the next, so the scan refines many brackets: 494 marches against 158 on draw 162. On the black oil, DAK's density solve fails at those temperatures and the march stops at once; that is why only table wells were slow. A DAK black oil tabulated with every column is slow in the same way.

**The fix.** `dak_jt_factor` returns NaN where $Z + \rho_r Z_\rho \le 0$, where $J$ is undefined (only below $T_{pr}$ = 1.05). A state there fails, as it does where DAK's density solve fails. There is no new constant and no new path. The spec text changes PVT-GAS-10's Safeguards (`specs/model/pvt/gas.md`), the table's Safeguards, the architecture's design point 4, and feature 018's Findings. The CasADi backend is unchanged: its signed-off lower temperature bound already keeps it off the pole. Evidence: `plans/evidence/table_jt_pole.py`.

**Measured** (one process; R9's tables rebuilt for `sol` draws 0 to 559; the first 4 attempts of every table well and the first 2 of the 360 others):

| Attempts | Roots | Total time, before → after |
|---|---|---|
| 336, volatile oils | identical to the bit | 332 → 62 s; P95 6.0 → 0.75 s; max 23.5 → 1.3 s |
| 319, condensates | identical to the bit | 50.8 → 50.0 s |
| 749, black oils, wet and dry gas | identical, search for search | unchanged |

- R9's 8 slow wells now run end to end in 24 to 211 s each, and all 8 are accepted.
- Two condensates became slower: 1.5 → 5.6 s and 1.8 → 2.5 s over 4 attempts. Their high-rate marches now fail beyond the pole, which adds bracket refinements.
- Five of the 8 slow wells still cost 4 to 8 times the median volatile oil per solve, from marches that run between the pole and the table's lowest temperature.

Tests: `cargo test` (58), the full suite (not slow 1198, slow 177), and the Rust verifier candidate (PASS, 100%, no expected failures). On the verifier's case set the guard is never reached (ideal gas, cooling off), so its results and cost are unchanged (principle 7).

| Option | Model and ML tasks | Cost |
|---|---|---|
| **The guard as on the branch** (recommended) | Same roots everywhere measured | Volatile oils' solves 5 times cheaper in total |
| The guard, and the core's temperature solve bounded as CasADi's, $\min(T_s, T_{lg}, 1.05\thinspace T_{pc})$ | The same roots | Removes the remaining slow marches; new solver machinery in the core |
| The guard with $J$ also NaN below $T_{pr}$ = 1.05 | Refuses states the CasADi backend admits: one table gas has $1.05\thinspace T_{pc}$ above its well's $T_s$ | Volatile oils 61 s, condensates 47 s |

**Recommendation.** Approve the guard as on the branch, open its PR to `develop`, and measure the cost again on the rerun of the pilots. Consider the temperature bound only if the remaining slow wells still dominate there.

**Ruling:** Bjarne, 2026-11-10: the guard approved as on the branch; its PR goes to `develop`. The temperature bound is considered only if the remaining slow wells dominate a later pilot.

**The rerun with the guard** (`r9-measurements.md`): `sol-2`'s CPU per accepted well 53.6 s (R9: 70.1; the rework 41.3), the pool's efficiency back to 0.94 from 0.46, the volatile oils' mean solve 0.22 s (R9: 0.81). What remains above the rework's cost lies outside the core, in building each attempt's table (`PVTTableSet.at_gor`), for R10.

### Noted, no ruling needed

- **H3's targets**, ruled on 2026-11-06 to be judged on R11's pilot: in R9, redraw-limit drops are 2.1% of the core's drawn `sol-2` wells (target 1% per class), dead dates 2 of 115 `nsol-2` wells (target 0.5%), and best-rate solver failures 2 of 487 `sol-2` wells (target 0.5%). R9 neither adds to them nor removes them. D-83 names the fix for dead dates, a finer life scan, which R10 implements.
- **H4's well 62** (an unstable root without a stable one, `plans/backlog.md` §2.17): not yet looked at with feature 022's inflow, as ruled. The well was a black oil of the draft's run, so the inflow does not bear on it; it stays in the backlog for R10.
- **D-105's two datasets by stage**: $p_r/p_i$ and the oils' water cut of `sol-2`'s wells and `nsol-2`'s samples agree by stage and drive to 0.01, as in the rework.
- **Building an attempt**: `PVTTableSet.at_gor` interpolates a table at each attempt's GOR in Python, about 30 to 40 ms per attempt, 11% (`sol-2`) and 20% (`nsol-2`) of the CPU. R10 moves the sampler into `manywells.sampling` and makes this cheap.

## 3. R10's spec draft: points for Bjarne

A first draft of the new `specs/sampling.md` is on the local branch `r10-sampling-spec` (`.worktrees/r10-sampler`, not pushed), for Bjarne's sign-off: v1's procedure unchanged, the v2 procedure as SMP-50 to SMP-117, D-107 marked pending. Drafting it found these:

- **One lift label for both datasets** (needed a ruling): `lift_start_rule` (`from_start`, `static_check`, `drawn_time`, `end_of_natural_flow`, `no_natural_flow`), since D-105 gives a `sol-2` well its life's lift; `datasets-draft.md`'s `lift_reason` predates D-105 (R11's schema).
- **Filled from the prototype:** a geometry redrawn at most 20 times, then vertical (no well reached it); the black oil at the table's stock-tank densities as the stand-in for the design steps of a table fluid (static check, tubing, lift design and capacity, the volatile oils' GOR balance, the condensates' $p/Z$ trend), the table only in the inflow; R3's $x$ still drawn for table fluids, as a record; the tail's depth and subsea water-depth ranges reaching past the core's bounds, as drafted; no tolerance on the field-reported API and gas gravity after J2's two passes (a lean condensate missed by 2.8 °API), only the single-stage match's; D-50's 21 scan points kept, the finer life scan named as D-83's fix; the depth and overpressure fits to BOEM's data as design values.
- **Drafts behind the rulings** (the spec follows the rulings; the drafts are plans and stay as they are): the lift-gas capacity's point masses (D-31), the lift-off share 0.20 (D-38), the logs in a companion file (I1), no CasADi retry (H1), the outcome `table_failed` (L1), `saturation_kind` `none`, a `sol-2` well's lift and inflow rule from its life (D-105, D-45).
- **Other specs to carry with R10:** `specs/goals.md` (the datasets' files with I1's companion, v2.0's model), `specs/constitution.md` (dataset reproducibility), `specs/model/validity.md` (decision 5, the inclination row, L3's limit, D-107's WGR if ruled, and the Sampler columns after R10's code), feature 022 (L3), and L6's spec text with its branch.

**Ruling:** Bjarne, 2026-11-10: follow the recommendations. The lift label as drafted, the points filled from the prototype confirmed, and the other specs carried with R10. The spec itself is signed off with R10's PR.

## 4. After the checkpoint

1. Done on 2026-11-09: the rulings in the draft's scripts and `sampling-draft.md`, L6's fix in its own branch, the rerun of both pilots with the tables, and R10's first spec draft.
2. Done on 2026-11-10, on Bjarne's rulings of L4 and L6: D-107 kept, and the PR of L6's branch to `develop`.
3. R10: the new `specs/sampling.md` (for sign-off) and the sampler in `manywells.sampling`, with tests.
4. R11: the pilot datasets for Bjarne's review, then the full generation.
