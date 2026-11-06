# Sampler redesign — the checkpoint (R8)

*Step R8 of `plans/sampler-redesign-plan.md`, 2026-11-02. The file Bjarne reads first: what the draft proposes, how it meets the Goals, and what needs his ruling. Plans are drafts, not specs. Nothing proposed here is decided until he rules.*

**Decided, not for ruling:** the Goals of the plan (2026-10-30) and five rulings: drains modelled to the heel with a limit of about 75° (D-10, 2026-10-30); the geometry as an even mix of four classes (D-9, 2026-11-02); 200 operating points per well, with every redraw logged (D-52, 2026-11-02); publishing the datasets is not a step of the plan, since Bjarne publishes them himself after validating and checking them privately (2026-11-05); and feature 022 approved as drafted (D-60, J1, 2026-11-05). **Proposed:** everything else, each item resting on an entry of `decisions.md`.

**Citations.** D-n is an entry of `decisions.md`. *R6, Section* is a section of `r6-measurements.md`, the prototype's pilots in the interim mapping (volatile oils and condensates as black oil, gas wells on a productivity index). [RF §n], [WP §n] and [GEO §n] are sections of `sources-reservoir-fluids.md`, `sources-wells-production.md` and `sources-geometry.md`. [own analysis of …] is the maintainer's analysis of a publisher's open data, kept outside the repo (the maintainer's decision, 2026-11-06): the repo states what it shows, not its values. *Draws* are the draw-only scripts' numbers, quoted in `sampling-draft.md`.

## 1. The proposal in one page

**A well** is a sequence of seeded draws in blocks, each conditioned only on earlier blocks and each with its own generator, so that a change to one block moves no other block's draws (D-7):

1. **Labels.** Coverage tail 0.20 of the drawn wells (D-103). Well type: platform 0.55, subsea 0.35, onshore 0.10 (D-5). Fluid class: black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20, by McCain's (1990) limits (D-3, D-4). Drive: pressure support 0.7 for oils, 0.4 for gas. Life stage: initial 0.25, mid-life 0.40, late 0.35 (D-18).
2. **Depth** below the mudline by class, fitted to BOEM by wells (D-8); water depth by well type; a platform's riser is part of the flow path.
3. **Geometry** (ruled): vertical, slightly deviated (5° to 20°), deviated (20° to 60°, a third S-shaped), highly deviated bottom (60° to 75°, half drains to the heel), a quarter each. No cell above 75°.
4. **Water, pressure, temperature.** Brine salinity sets $\rho_w$ and the hydrostatic gradient (D-11). Overpressure rises with depth; 4% start underpressured; $p_i \le 950$ bar (D-12, D-35). $T_s$ from the seabed or the land (D-14); the gradient from 30 K/km on shelves to 20 K/km in deep water; $T_r \le 450$ K (D-13).
5. **Separator and fluid.** The host's first stage, about 35 bar offshore and 15 bar onshore; the pressure downstream of the choke falls with the stage (D-15). API and gas gravity by class (D-16). The saturation pressure as a fraction of $p_i$, and a black oil's GOR from the model's own Vazquez–Beggs (D-17).
6. **Reservoir state** at the stage: $p_r/p_i$, water and producing GOR by drive and stage (D-18, D-19).
7. **Inflow.** Permeability, pay, skin and drainage give an oil well's productivity index, on Vogel or a straight line by rule (D-20, D-22, D-100), or a gas well's Forchheimer coefficients (D-61, feature 022).
8. **Completion and choke.** The smallest tubing that keeps the initial design rate below the erosional velocity (D-25); roughness by stage (D-26); a Simpson choke with a bean of 0.25 to 0.6 of the tubing (D-27).
9. **Gas lift.** Wells that cannot lift their static column, and others by stage (D-28); the valve at the pressure-balance point (D-29); a capacity of 50 to 400 Sm³/Sm³ times the design rate (D-31).
10. **Heat.** $h$ from the wellbore and the formation at the well's age (D-32); heat capacities from correlations (D-33).

Volatile oils and condensates, a quarter of the core wells, then get a PVT table of their own from a NeqSim flash: a synthetic composition matched to the drawn API, gas gravity and GOR, or, in a fifth of them, a published example fluid (D-63, D-66, D-67).

**Labels.** About twenty categorical labels per well (tail, well type, fluid type and class, drive, stage, state, pressure class, saturation, geometry class and family, PVT model, inflow, tubing, choke, profile, lift and its reason). All are drawn before any solve, except the lift of a `sol-2` oil well that cannot flow naturally (D-101) and the start of lift in `nsol-2` (D-49).

**Within a `sol-2` well.** 200 operating points. Each slot draws the choke (fully open with probability 0.05, else uniform from 0.05), the lift gas (off in 10%, else uniform to the capacity), the separator pressure (±10%) and the drift of $p_r$ (±2%) and of the GOR and water (±5%), the GOR tied to the drifted $p_r$ (D-37 to D-40).

**An `nsol-2` well's life.** The same well drawn at its initial stage, ending at its late-stage state over 8 to 30 years. One clock, the cumulative production with a plateau and Arps's decline, drives the pressure and the water; the pressure drives the GOR (D-41 to D-44). $h$, roughness and productivity age (D-45); the separator steps down twice (D-46). The controls are piecewise constant, open loop (D-47), with 8 test periods of 5 sweeps, 20% of the slots (D-48). Lift starts mid-life where needed (D-49); the life ends at the last scan point whose best rate passes the floor (D-84).

**Filters, redraws and logs.** A well whose best rate, fully open at its nominal state, is below half its economic floor (1.6 Sm³/d of oil equivalent onshore and on platforms, 8 Sm³/d subsea) is dropped (D-34, D-50), after a natural oil well has been tried with lift (D-101). SMP-28's three filters go (D-51). A draw without an operating point is redrawn from its own generator (D-80), within limits of 50 attempts per slot, 400 redraws and 10 solver failures per well (D-81). Every redraw and every drawn well is logged, and the logs are part of each dataset (D-85).

**The datasets.** `manywells-sol-2` and `manywells-nsol-2`, each 10,000 wells of 200 operating points, as parquet (D-86): `wells`, `samples`, `geometry`, `pvt`, and the logs `redraws` and `drawn_wells` (D-87). An 80/20 train–test split by well, stratified (D-90); a datasheet for each (D-94). The plan ends with the datasets generated and checked; Bjarne validates and checks them privately and publishes them himself.

**Key numbers** (R6, What was run, Outcomes, Redraws, Cost):

| | `sol-2` pilot | `nsol-2` pilot |
|---|---|---|
| Wells drawn for 400 / 100 kept | 450: 88.9% kept (core 94.2%, tail 66.7%) | 109: 91.7% kept |
| Dropped as not producible | 10.2% of the drawn wells (core 5.5%) | 6 at the start |
| Redraws per sample | 0.036 | 0.040 |
| Solver failures left in kept wells | 0 | 1 of 20,808 attempts |
| Rust solve, median | 0.159 s | 0.162 s |
| Projected, 10,000 wells | 113 CPU-h, 9.7 h on 12 processes | 128 CPU-h, 12.0 h on 12 processes |
| Samples table, float64 before compression | about 420 MB | about 440 MB |

## 2. The goals and acceptance criteria, met or not

The R6 numbers are in the interim mapping unless said otherwise.

| Goal or criterion | How the draft meets it | R6's number | Status |
|---|---|---|---|
| **Purpose:** the four tasks | Uniform controls, lift curves past the optimum, the redraw log where wells stop, every label stored | choke 0.10 / 0.55 / 1 (P5 / P50 / P95); 34.8% choked; injected GLR 2.6 / 78 / 988 Sm³/Sm³ (R6, Coverage) | **Partly:** the choke moves the rate little in many wells (E1), which weakens virtual flow metering and choke optimisation |
| **Purpose:** two datasets of 10,000 × 200 | As decided | about 22 h on 12 processes for both (R6, Cost) | **Met** in design; the tables add up to about 5 h for both, and their solves are not measured (Section 8) |
| **Purpose:** realistic core, labelled tail of 10% to 20% | Tail drawn at 0.20, flattened within the core's bounds | 14.5% of kept `sol-2` wells, 19% of `nsol-2` (R6, Summary by goal) | **Met** after D-103 (9.25% before) |
| **Yardstick:** offshore-weighted, by wells | Shares by wells (D-1); 90% offshore (D-5) | – | **Met;** the onshore share is a design choice (A1) |
| **Yardstick:** published statistics first | BOEM, Sodir, EIA, textbooks, each cited | initial pressure gradients and $T_r$ by class within 0.012 bar/m and 11 K of BOEM's medians (R6, Realism; [own analysis of BOEM's sands atlas]) | **Partly:** stage shares, control histories, tests and beans are own choices without a public count; lift shares, water and GOR by age, decline, lives and final rates have per-well public statistics since 2026-11-05 [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well] |
| **Yardstick:** Bjarne judges from samples in row format | R6 wrote the pilots in the datasets' schema, kept outside the repo | 400 + 100 wells | **Met** |
| **Goal 1:** depleted, $p_r$ below hydrostatic | Life stages with depletion | 44.5% of `sol-2` wells, 28.0% below 0.9 of it; 56.4% of `nsol-2` samples | **Met** |
| **Goal 2:** separators below 10 bar | Lower stages late in life; onshore gathering | 16.5% of `sol-2` samples, 19.5% of wells; 16.2% of `nsol-2` samples | **Met** |
| **Goal 3:** overpressured, high-rate wells | Overpressure with depth; productivity | 26.0% of the kept wells, core and tail, above 0.6 psi/ft at their initial pressure (22.5% of the kept core wells, near BOEM's share by wells; R6, Realism; [own analysis of BOEM's sands atlas]); 11.5% both overpressured and above 1000 Sm³/d | **Met** |
| **Goal 4:** modest weight at fully open | 5% point mass, else uniform | 5.1% fully open in both datasets | **Met** |
| **Goal 5:** lift scaled to the well, gassier wells too | Capacity scales with the design rate; no GOR limit | Spearman 0.81 between capacity and liquid rate; oil wells lifted by GOR quartile 48 / 30 / 25 / 21%; gas classes 7% to 13% | **Partly:** gassy wells are lifted less often (F1) |
| **Wells:** four fluid types, floor about 10% | McCain's classes, floor in the mix | kept `sol-2` wells about 47 / 15 / 17 / 21% black oil / volatile oil / condensate / gas (R6, Outcomes) | **Met** in the draw; volatile oils and condensates are black oil in the pilots |
| **Wells:** labels | Drawn before the solves | – | **Met;** two labels follow a solve (D-49, D-101) |
| **Wells:** geometry as ruled | Four classes, 75° | 27.0 / 24.3 / 23.3 / 25.5% kept; no cell above 75° | **Met;** the shapes' parameters are not yet the surveys' (B1) |
| **Wells:** gas-rate inflow | Feature 022 approved (J1, 2026-11-05) | interim index: 0.57 to 2.05 times the reference rate (022, Measurements) | **Not met:** not implemented |
| **Wells:** PVT tables | Per-well tables from NeqSim (D-63 to D-68) | proof on 4 wells and 1 example; no well solved | **Not met:** condensates wait for D-64 (K1) |
| **Wells:** new draws ($T_s$, injection depth, $T_{lg}$, fluid constants) | D-11, D-14, D-15, D-29, D-30, D-33 | – | **Met** |
| **Variation:** `sol-2` controls and drift | D-37 to D-40 | $p_s/p_{s,\text{nom}}$ 0.91 / 1.00 / 1.09 (draws) | **Met** |
| **Variation:** a life stage per well | Stage, drive and a consistent state | – | **Met** |
| **Variation:** `nsol-2` linked trends, mixed open-loop history, a life per well | D-41 to D-49, D-84 | oil equivalent 0.93 to 0.35 of the first sample over the fifths of the life; 38% lifted at some time; 8% of lives ended early by the floor | **Met** |
| **Datasets:** operating point only | Stable roots only | every sample the stable root; 1500 rows re-solve identically (R6, Backends) | **Met** |
| **Datasets:** two tables joined by an ID | Two main tables, plus `geometry`, `pvt` and two logs | – | **Partly:** needs the edit to `specs/goals.md` (Section 4) |
| **Datasets:** filters, failed draws redrawn and logged | Producibility filter, redraws, logs | the logs written in the pilots | **Met** |
| **Datasets:** split by well, datasheet | D-90, D-94 | – | **Met** in design |
| **Criterion 1:** no implausible wells | Bounds and links; `configurations.check` | no well dropped as invalid; offshore oil wells above 1018 Sm³/d: 23% against EIA's 5.5%; gas wells 0.9 million Sm³/d at the median against a GoM mean of 32,700 | **Partly:** each well is plausible, the rate distribution is not (D1, D2) |
| **Criterion 2:** no wells that would not be produced | Half the floor; lift before dropping | 10.2% of drawn `sol-2` wells dropped (core 5.5%) | **Met** |
| **Criterion 3:** no missing real conditions | Life stages, low separators, overpressure, wide rates | churn 1.5% (bottom) and 2.0% (top) of samples; core offshore oil wells below 16 Sm³/d: 16% against EIA's 52% | **Partly:** churn flow thin (G1); EIA's low brackets include wells shut in part of the year |
| **Criterion 4:** no solver-shaped distributions | Redraws from the slot's own generator, retry, logs, targets | solver-failure redraws shift no input by more than a KS $D$ of 0.0001; no-operating-point redraws shift the lift gas by 0.054 (physics); limit drops by at most 0.010 (R6, Criterion 4) | **Met** in the pilots; drop targets exceeded on small counts (H3), two solver findings (H4) |

## 3. Rulings needed

In each group the items that need judgement come first, then the entries that can be approved in bulk. For each option: what it means physically, what it changes in the model and for the machine-learning tasks, and how common the case is in real wells.

**Coverage first, shape second** (Bjarne, 2026-11-05; the plan's Goals). First, the draws cover almost all real wells and hold no unrealistic ones; second, they weigh typical wells more than rare ones. Every sample of real wells in hand, the Gulf of Mexico's statistics included, represents some fields and areas only, so its range is a lower bound on the support and its shape weak evidence of what is typical. D1, D2 and E1 were revised on 2026-11-05 to follow this.

**Ruled on 2026-11-06:** every card and bulk list, and section 4 in substance, except D2 and its D-62, which wait for feature 022 (R9). Each ruling is under its card and in its decision entry; the rework they ask for is in section 8.

**What the evidence settles, and what is left to choose** (2026-11-06, at Bjarne's ask to let physics and data decide as much as possible). A card's recommendation rests on data, physics or the project's rules where this table says so; approve it unless the evidence looks wrong to you. The last column holds what no evidence can settle.

| Card | Settled by | Left to choose |
|---|---|---|
| A1, A2 | – | Ruled 2026-11-06 |
| B1 | The surveys' distributions; the physics of long drains (`drain_friction.py`); the kickoff below the mudline (a fix) | The drainage geometry of long drains (Joshi's growing area is the recommendation) |
| C1 | Water by age (per-well data); the GOR's rise (material balance, its $k_g/k_o$ chosen by the per-well data); the initial depletion range (`nsol-2`'s first year) | The stage shares 0.25 / 0.40 / 0.35: no public count of wells by stage |
| D1, D2 | Rates by well type (per-well data); the cap at the tubing's erosional capacity (physics); D-62's non-Darcy coefficient (Economides et al.'s Eq. 4-34) | The medians are set on the rework's pilot against the data |
| D3 | The stopped wells' final rates (per-well data) | – |
| E1 | The tested choke sizes and the 4 in cap (BSEE's well tests) | The lower end of $\beta$ at 0.15, for coverage of the textbook beans |
| F1, F2 | Lift by GOR quartile (BOEM); the lift-gas rates (3W); D-101 as operators do | – |
| G1, H3, H4 | Re-measure on the rework's pilot, or log to the backlog | Agree to defer |
| H1 | The constitution's principle 7 (no measured gain, so drop) | – |
| H2, K1 | Engineering choices with precedent (`sources-condensate-tables.md` for K1) | Approve or pick another option |
| I1, I2 | – | What the datasets carry, and the re-solve tolerance (a tolerance needs sign-off) |
| J2 | Each fluid's offsets by its own separator flash (physics) | – |
| Bulk lists | Each entry's evidence; D-11, D-14, D-15, D-29, D-37, D-41, D-43, D-44, D-47, D-65 revised by data or physics since 2026-11-05 | D-15, D-21, D-29, D-41, D-47 are not bulk-approvable as drafted: approve their revisions |
| Section 4 | – | Every spec change needs sign-off |

### A. Population and fluids

#### A1 · The onshore share (D-5)

**Question.** Keep 10% onshore wells in an offshore-weighted population?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **0.10** (draft) | Land wells: surface 268 to 303 K, separators around 15 bar, permeability median 3 mD | About 1000 wells per dataset with land surface temperatures and low-pressure gathering; 57% of them below 10 bar (draws) | By wells the world is onshore: 918,481 producing US wells against 2,216 in the federal GoM [WP §1]; any share is a design choice |
| 0 | Offshore only | Matches the yardstick; goal 2 still met by late platform wells, 45% of them below 10 bar (draws) | As above |
| 0.25 | More land wells | More low-rate, low-pressure wells; the population drifts from the yardstick | As above |

**Recommendation:** 0.10.

**Ruling:** Bjarne, 2026-11-06: 0.10 onshore, inside `sol-2` and `nsol-2` and labelled by `well_type`. Separate onshore datasets were considered and not chosen: ManyWells models natural flow and gas lift only, while most onshore wells are rod-pumped or on ESPs; no onshore yardstick is in hand; and they would change the Goals and add cost. Users split by the label (D-5).

#### A2 · The coverage tail: its share and form (D-6, D-103)

**Question.** Draw 20% tail wells so that about 15% of the kept wells are tail? And may tail wells take lift and geometry regardless of stage and type, as drafted?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **0.20 drawn, flat independent draws** (R6) | Less likely combinations within the core's bounds, such as $p_r$ at 0.15 $p_i$ with $p_s$ near 120 bar | 14.5% of kept `sol-2` wells; widens gas and water rates and low PBH and PDC; churn at the bottom 3.4% against 1.1% in the core (R6, Coverage); 30% of tail wells are dropped as not producible | A coverage device, not a population; labelled, so a user can drop it |
| 0.15 drawn (D-6) | As above | 9.25% of kept wells, below the Goals' 10% | As above |
| 0.20 with $p_{s,\text{nom}}$ tied to $p_r$ | Fewer tail wells that cannot flow | Needs the separator drawn after the reservoir state, a change of the block order (D-7) | As above |

**Recommendation:** 0.20 drawn, independent draws; a dropped well costs one to three solves and is logged. Tail wells may take lift and geometry regardless of stage and type.

**Ruling:** Bjarne, 2026-11-06: 0.20 drawn, independent draws; tail wells may take lift and geometry regardless of stage and type (D-6, D-103).

**Bulk-approvable** (Bjarne, 2026-11-06: approved, D-11 with its two changes; D-16 ruled separately: independent draws, black oils' gas heavier):

- D-1 · Wells as the unit of weighting.
- D-2 · Nadeau et al. (2023) by reservoir counts, as checks.
- D-3 · Fluid classes by McCain (1990).
- D-4 · The fluid mix (0.55 / 0.10 / 0.15 / 0.20) and 7% dry gas.
- D-7 · The order of the draws and one generator per block (R10: draw every number of a block whatever its branch; R6, Task 1).
- D-8 · Depth by fluid class, water depth by well type.
- D-11 · Brine salinity and density. *Bulk-approvable with two changes* (2026-11-06, from USGS's produced-water database).
  - **Offshore.** The US offshore wells' produced water, mostly the GoM shelf's, lies within the draft's range but for a few percent below its floor, with nothing near its cap, and their median salinity lies below the draft's median of 80,000 ppm. Their brine's specific gravity gives a hydrostatic gradient near the draft's median of 0.104 bar/m [own analysis of USGS's produced-water database: US offshore].
  - **Onshore and gas wells.** These spread wider. About a fifth of the US onshore conventional wells, and most tight gas wells, lie below the draft's floor of 1%. A few percent lie above its cap of 25%, near or beyond halite's saturation near 26%: concentrated brines, most of them in the database's Eastern, Rocky Mountains and Northern Great Plains, and Canadian regions [own analysis of USGS's produced-water database].
  - **The changes.** Lower the median to about 60,000 ppm and the floor to about 0.2% (coverage first), and keep the cap at 25%, near halite's saturation.
- D-16 · Oil and gas gravities by class. Norwegian drill-stem tests (2026-11-05): black oils' gases run heavier than the draws', volatile oils' API reaches below the draws' lower bound, and API and gas gravity correlate in oils, where the draft draws them independently [own analysis of Sodir's FactPages: NCS drill-stem tests by class].
- D-17 · Undersaturation, and the black oil's GOR from the model's own correlation.
- D-36 · Volatile oils and condensates as tables; wet and dry gas as black oil.

### B. Geometry (mostly ruled)

*Ruled:* D-9 (the four classes, a quarter each; drains and tangents in the highly deviated class) and D-10 (75°, drains to the heel).

#### B1 · The shapes' parameters against the surveys (no entry yet)

**Question.** The draft's ranges are R3's, set before the surveys were read; `sampling-draft.md` (Geometry) cites the surveys' distributions, in [own analysis of BOEM's directional surveys] (GoM) and [own analysis of NLOG's deviation surveys] (the Netherlands). Adopt them in R10?

| Parameter | Draft | Surveys |
|---|---|---|
| Build rate | $U(1.5, 3)$ °/30 m | gentler at the median, and wider [own analysis of BOEM's directional surveys] |
| Kickoff, fraction of TVD | $U(0.1, 0.7)$ for build and hold, up to 0.6 for S, $U(0.05, 0.4)$ for high angle | shallower at the median, with a long deep tail; deeper below the mudline in subsea wells [own analysis of BOEM's directional surveys] |
| S wells' drop rate | $U(1, 2.5)$ °/30 m | slower at the median [own analysis of BOEM's directional surveys] |
| S wells' bottom inclination | $U(0°, 10°)$ | near vertical at the median, with a tail to steeper bottoms [own analysis of BOEM's directional surveys] |
| Drains' landing build | $U(2, 4)$ °/30 m | steeper at the median [own analysis of BOEM's directional surveys] |
| Kickoff of a platform well (a fix) | $K$ from $0.1L$ with $L = z + d_w$, so in deep water the build can start inside the riser | every surveyed well kicks off below the mudline; draw $K$ as a fraction of $z$ and add $d_w$ |
| High-angle tangent $\theta_h$ | $U(35°, 65°)$ | the GoM's drains hold lower tangents at their lower quartile, then land late [own analysis of BOEM's directional surveys]: let $\theta_h$ start near 10° |
| Drain length (D-21) | $U(300, 1000)$ m | a fifth or more of the Dutch drains and a few percent of the GoM's are longer than 1 km [own analysis of NLOG's deviation surveys and BOEM's directional surveys]; physics (below) lets drains to about 2500 m add productivity |

- **Physically:** gentler main builds and slower drops, so longer curved sections; S wells cross the reservoir less vertical.
- **Model and ML tasks:** small changes to MD/TVD and to each cell's inclination, which the slip's inclination terms see; little effect on rates.
- **In real wells:** the GoM's and the Netherlands' surveyed producing boreholes [own analysis of BOEM's directional surveys and NLOG's deviation surveys]. Norway's production wellbores, which have no published surveys, have grown longer and more deviated since the 1970s: the platform wellbores' median MD/TVD rises decade by decade, and the subsea wellbores' upper tail most of all [own analysis of Sodir's FactPages: NCS platform and subsea production wellbores by completion era].

**Physics of long drains** (2026-11-06, `drain_friction.py`, from Joshi 1991, Ch. 3 and Ch. 10, and Brekke and Lien 1994). Dikken's coupled model, Joshi's index along the drain with the liner's turbulent friction, checked against Joshi's Table 3-2 and Example 10-5 and against Brekke and Lien's Troll drain:

- Joshi's own geometry ends the gain once a drain reaches about 1.6 times the drainage radius; with the drainage area growing with the drain, a median well's $J_H/J_V$ is about 4.5 at 500 m, 7 at 1 km and 10 from 2 km.
- Friction takes little below 100 mD (median loss 1%, 90% of the drawn drains), but above about 250 mD it stops the gain at 450 to 1200 m; in gas drains it takes more than 10% in most of them.
- So the cap of 6 is not physics: it binds on about a third of the draft's drains already, and on half or more of drains to 2500 m. Friction, not a cap, is the limit, and a closed form, $\eta = (1 + \Pi/1.5)^{-1/3}$ with $\Pi$ the friction-to-drawdown number of Dikken's model, gives it within 2% of the march.

**Recommendation** (revised 2026-11-06): adopt the surveys' distributions, truncated to the classes' ranges, and recheck MD within 9342 m and the share of redrawn fits. Draw the kickoff below the mudline, let the high-angle tangent start near 10°, and draw drains $U(300, 2500)$ m, with D-21 revised as in group D. The drainage geometry of long drains (Joshi's area growing with the drain, against a fixed spacing between drains, which raises the upper tail) remains a choice; the first is Joshi's and the recommendation.

**Ruling:** Bjarne, 2026-11-06: use physics where possible, but keep it simple, and base the shapes on all the surveys in hand, not the GoM's and the Netherlands' alone. Each shape parameter is uniform over its pooled range (D-104): the kickoff below the mudline $U(0.05, 0.75)$ of TVD, the build $U(0.3, 3.6)$ °/30 m, the S wells' drop $U(0.2, 2.2)$ °/30 m and bottom $U(0°, 35°)$, the drains' tangent $U(10°, 65°)$, landing $U(1.5, 7)$ °/30 m and length $U(300, 2500)$ m, and high-angle tangent wells holding $U(60°, 75°)$ to the bottom; D-21 in its simple physical form.

**Bulk-approvable** (Bjarne, 2026-11-06: approved):

- D-57 · S-shaped wells as a third of the deviated class.

### C. Pressure and temperature

#### C1 · Life stages and how depleted the data are (D-18, D-19)

**Question.** Accept the stage shares (0.25 / 0.40 / 0.35), the state tables and the GOR's rise below the saturation pressure? Widen the initial stage's pressure range under depletion?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **As drafted** | A third of the wells late: depleted, or watered out with a median water cut of 0.77; GOR up to ten times $R_{si}$ | 44.5% of `sol-2` wells below hydrostatic (R6); a wide span of $p_r/p_i$ (0.15 to 1.02) and water cut | No public count of wells by stage [RF §3, §11]; the $p/Z$ at which GoM depletion gas sands end [own analysis of BOEM's sands atlas]; Norwegian oil fields' water cut by years on production, in 2025 [own analysis of Sodir's FactMaps] and over every field-year since 1971 [own analysis of Sodir's FactPages: NCS oil fields by years on production]; per-well water cut and producing GOR by years on production in the GoM, Brazil and the UK [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: well-year by years on production] |
| A younger mix, such as 0.40 / 0.40 / 0.20 | Fewer depleted and watered-out wells | Less coverage of goal 1 | As above |
| Widen the depletion initial range from $U(0.97, 1.0)$ to about $U(0.85, 1.0)$ | A depletion well loses pressure in its first year | `sol-2`'s initial stage then matches `nsol-2`'s first year, whose P5 is 0.85 of $p_i$ (`sampling-draft.md`, What R6 must measure) | No statistic; Golan and Whitson's Example 4.11 lost 12% of its pressure over its first step of cumulative oil |

**Recommendation** (revised 2026-11-05, from the per-well data). Accept the shares and the water tables, and widen the initial range under depletion.

- **Water.** The draft's water cut by years on production (D-43) lies inside the span of the regions' per-well medians, which runs from the GoM shelf, wetter at every age, to Brazil's pre-salt and the GoM's subsea wells, far drier [own analysis of BOEM's OGOR-A production and ANP's production by well: water_cut by years on production]. Real wells reach the cap of 0.98 late in life, so thicken the upper tail towards it.
- **GOR** (D-19, D-44; revised 2026-11-06: physics and data decide). The material balance of solution-gas drive (`solution_gas_drive.py`: Muskat's method, Craft and Hawkins 1991, Ch. 9, Eqs. 9.33 to 9.35, checked on the book's Example 9.1 and Ch. 10 history match) on the model's own black oil gives the GOR's shape: it dips to the dissolved gas just below the saturation pressure until the gas can flow, then rises steeply and would fall only below about 0.15 to 0.25 of $p_\text{sat}$, where no core state lies. The draft's linear rise from $p_\text{sat}$ is not that shape. The one free input is the rock's $k_g/k_o$, which moves the rise about fivefold across the book's laboratory curves. The data choose it: with the book's least mobile gas (Fig. 9.16), the late-stage wells' GOR over its initial value has a median of 1 and an upper tail within the span of the real wells' GOR over their first year at 10 to 30 years on production, while the book's more mobile curves put the 90th and 95th percentiles above every region's [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: gor_over_first_year]. The draft has a median too high and a tail too thin. Recommendation: the factor from the material balance per well at $p_r/p_\text{sat}$, with Fig. 9.16's $k_g/k_o$ in the core and the book's range of curves in the tail; $R_{si}$ wherever $p_r \ge p_\text{sat}$, so pressure support needs no rule of its own; no cap, since the curve bounds itself.
- **Gas depletion.** The UK gas wells' bottomhole and wellhead pressures in their last year lie within R6's ranges [own analysis of NSTA's production by well: well, last year], so the public data ask for no deeper gas depletion.

**Ruling:** Bjarne, 2026-11-06: stage shares 0.25 / 0.40 / 0.35, with every evidence-based change above: the water cut's tail towards 0.98, the initial depletion range $U(0.85, 1.0)$, and the GOR from the material balance with Fig. 9.16's $k_g/k_o$ in the core and the book's range in the tail (D-18, D-19, D-44).

**Bulk-approvable** (Bjarne, 2026-11-06: approved, D-15 with its subsea revision, D-35 with its MD bound kept at 9342 m):

- D-12 · Overpressure rising with depth; 4% initially underpressured. Norwegian drill-stem tests' bottomhole pressures have an upper tail close to the draws' (2026-11-05) [own analysis of Sodir's FactPages: NCS drill-stem tests, bottomhole_pressure_gradient].
- D-13 · The geothermal gradient, one log-normal falling with water depth. Norwegian drill-stem tests put the shelf's median gradient near the draws' upper percentile, as its logging temperatures did (2026-11-05) [own analysis of Sodir's FactPages: NCS drill-stem tests, geothermal_gradient].
- D-14 · Surface temperature by well type, a seabed profile offshore. The seabed temperatures at the producing wells of the GoM, Norway and the UK, and on the sea floor of Brazil's Campos and Santos basins, lie within the draft's shelf and deep-water ranges: the GoM's shallow shelf and Brazil's at the shelf range's top, the GoM's deep water at the deep range's top and Norway's deep subsea wells at its bottom (2026-11-06) [own analysis of NOAA's World Ocean Atlas 2023].
- D-15 · Separator conditions, and the nominal downstream pressure falling with the stage. *Not bulk-approvable as drafted for subsea wells* (2026-11-06).
  - 3W's subsea satellite oil wells hold their trees above the draft's truncation of $p_{s,\text{nom}}$ at 120 bar in more than half the wells. Most of their flowline-and-riser pressure differences exceed the largest of the draft's $\Delta p_\text{fl} = U(5, 30) + 0.02\thinspace d_w$ bar at any water depth shallower than about 2 km [own analysis of Petrobras' 3W Dataset: subsea_tree_pressure, flowline_and_riser_pressure_difference].
  - The GoM's subsea wells test at flowing tubing pressures far above its platform wells' [own analysis of BSEE's well tests: flowing_tubing_pressure].
  - Raise the subsea flowline loss (an oil-filled riser is heavier than the draft's light column) and the truncation for subsea wells, set on the rework's pilot against these rows. The platform and onshore draws stand.
- D-35 · Truncations at 950 bar, 450 K, 9342 m of MD and 16 °API.

### D. Productivity and rates

#### D1 · How close the oil rates should come to EIA's brackets (D-100, D-20, D-24)

**Question.** With D-100's permeability (15 mD offshore, 3 mD onshore), 23% of the core offshore oil wells make more than 1018 Sm³/d of oil equivalent, against 5.5% of the Gulf of Mexico's oil wells in 2024 (R6, Realism; EIA 2025, Table B32). What tolerance does the check get, and what permeability follows?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **15 / 3 mD** (R6) | A well's effective permeability at the low end of reservoir medians (10 to 570 mD by porosity), standing for what the model lacks: relative permeability, partial completion, operating limits | Median 197 Sm³/d, mean 822; 23% above 1018 and 11.6% above 2035 Sm³/d (EIA 1.3%); high-rate, tubing-limited wells well covered | EIA's brackets are calendar-day averages of a mature shelf, mean 167 Sm³/d; an offshore Norwegian well makes at least 10,000 bbl/d (Golan and Whitson 1991, Preface) [WP §5]; Norwegian oil fields' averages per active well put a share of field-years above 1018 Sm³/d between the GoM's and the pilot's [own analysis of Sodir's FactPages: NCS oil fields]; per well, the share above 1018 Sm³/d runs from almost none of the GoM shelf's wells to more than the pilot's 23% in the GoM's deep-water subsea wells and Brazil's pre-salt, and the share below 16 Sm³/d the other way [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well] |
| 10 / 2 mD | As above, lower | About 110 Sm³/d at the median, 12% above 1018 and 2.4% above 2035 (D-100's estimate from the pilot's slope, not run); more weak wells dropped | Twice EIA's share above 1018 |
| 15 mD with a log spread of 1.0 in place of 1.5 | Fewer very permeable and very tight wells | Trims both ends without moving the median | No source on the spread of wells' permeability |

**Recommendation** (revised 2026-11-05, from the per-well data and the physics). The rates depend on the well type in every region with per-well data, and R6's do not. R6's core platform and subsea oil wells have the same median, about 200 Sm³/d, and the same 23% above 1018 Sm³/d, and 24% of its subsea oil wells lie below 16 Sm³/d (`pilot_by_well_type.py`), where almost no real subsea well does [own analysis of BOEM's OGOR-A production and ANP's production by well]. Real subsea wells are the high-rate wells, since a subsea well pays only at a high rate; platform wells span the shelf's low rates and deep water's higher ones [own analysis of BOEM's OGOR-A production: well median, by setting].

- **Shape.** Draw the permeability by well type, and set each type's median and spread on the rework's pilot against these rows, keeping the support wide (coverage first): subsea wells against the GoM's deep-water subsea wells and Brazil's pre-salt, platform wells against the GoM shelf, its deep-water platforms and Brazil's post-salt [own analysis of BOEM's OGOR-A production and ANP's production by well].
- **Top end, by physics.** Redraw a well whose design rate exceeds the 7 in tubing's erosional capacity, as for gas (D2): 7,700 to 17,700 m³/d of wellhead mixture for $C$ of 100 to 150 (`erosional_capacity.py`), of the order of the best months on public record [own analysis of BOEM's OGOR-A production and ANP's production by well: well peak].

**Ruling:** Bjarne, 2026-11-06: permeability by well type, set on the rework's pilot against the per-well rates; the erosional capacity as a check only, a redraw added only if the flagged share stays above a few percent (D-100). Keep the method simple.

#### D2 · Gas wells' rates, and the non-Darcy coefficient (D-62, D-23, D-61)

**Question.** The core gas and condensate wells make 0.14 / 0.90 / 3.1 million Sm³/d (P10 / P50 / P90), 83% above 270,000 Sm³/d, against the GoM gas wells' mean of 32,700 Sm³/d and 2.3% above 270,000 (R6, Realism; [WP §5]). D-62 would raise them further. What should set the gas wells' rates?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **D-62:** $D$ from Economides et al.'s Eq. 4-34, then lower the gas classes' $kh$ to meet a rate check | Less non-Darcy flow in thick, permeable reservoirs ($D \propto 1/h$); tighter gas reservoirs than oil ones | The IPR's curvature follows its sources; lower rates put more gas wells near liquid loading, where churn flow lives (G1) | R3's $D$ is 12 times Eq. 4-34 at the median; the non-Darcy term is over half the drop in 66% of gas wells (D-62) |
| R3's $D$, $\text{LogU}(2\cdot10^{-4}, 2\cdot10^{-3})$ d/Mscf, and the rates as they come | Strong non-Darcy flow in most wells | Rates stay far above the GoM; few gas wells load | "Frequently of the order of $10^{-3}$" d/Mscf (Economides et al. 1994, p. 71) |
| Wait for feature 022 before any change | – | The interim index is 0.57 to 2.05 times the reference across drawdowns, so today's numbers may move by a factor 2 either way | – |

**In real wells:** the GoM's gas wells are mostly old shelf wells, 45% at or below 15 BOE/d; "a strong gas well will never produce to capacity because of equipment limitations" (Golan and Whitson 1991, Preface); 7 in tubing carries about 2.8 million Sm³/d [WP §5, §6]. Norway publishes production by field only: its gas fields' averages per active well by field-year lie below the core's P50 and P90; in each field's best year their median is near the core's, and the best field's best year stays below the pilot's 99.5% point (2026-11-05) [own analysis of Sodir's FactPages: NCS gas fields; the 10 largest by cumulative gas]. These are field averages, so a field's best wells produce more.

Per well, the share of gas wells above 270,000 Sm³/d at their median runs from a small share of the GoM shelf's and deep-water platforms' wells to most of its deep-water subsea wells; the pilot's 87%, on platforms and subsea alike, lies above every region's [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well; `pilot_by_well_type.py`].

**Recommendation** (revised 2026-11-05, from the per-well data and the physics). Adopt D-62.

- **The cap, by physics.** D-25's own rule already finds the design rate above the largest tubing's erosional capacity in 6% of the draft's gas wells and 12% of its condensates (21% of its volatile oils, 7% of its black oils), and keeps them in the largest tubing (`sampler_draft_draws.py`). 9-5/8 in tubing carries 2.4 to 11 million Sm³/d at the erosional velocity, over wellhead pressures of 30 to 300 bar and $C$ of 100 to 150 (`erosional_capacity.py`). Redraw the permeability of a well above its capacity, in place of a cap at a record. The best months on public record and Norway's best field-year per well lie inside that capacity [own analysis of BOEM's OGOR-A production, ANP's production by well (well peak) and Sodir's FactPages (NCS gas fields)].
- **Shape.** As for oil (D1), draw the gas classes' permeability by well type, and set the medians on the rework's pilot with feature 022 against these rows, keeping the support down to the weak wells near liquid loading.

**Deferred:** Bjarne, 2026-11-06: wait for feature 022 (R9), then decide D2 on gas rates measured with the real inflow.

#### D3 · The economic floor and its margin (D-34, D-50)

**Question.** Drop a well whose best rate is below half its floor (0.8 Sm³/d of oil equivalent onshore and on platforms, 4 Sm³/d subsea)?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Half the floor** (draft) | Keeps marginal wells | 10.2% of drawn `sol-2` wells dropped, 5.5% of the core (38 without an operating point at their best rate, 8 below half the floor); core oil rates from 2.1 Sm³/d (P5) | 9.8% of GoM oil wells average 10 BOE/d or less, partial years included [WP §11]; wells that stopped: platform wells' final rates reach down to the draft's margin at their low percentiles, while subsea wells stop above the draft's subsea floor, most at several times it, in the GoM and Brazil's pre-salt [own analysis of BOEM's OGOR-A production and ANP's production by well: final_oil_equivalent_rate] |
| The full floor | Drops a few more weak wells | Fewer low-rate samples | Stripper limits: 10 bbl/d or 60 Mcf/d (IOGCC) |
| A floor per structure, shared by its wells | Offshore economics are per structure | Needs a structure draw the model does not have | Kaiser (2011), abstract only |

**Recommendation** (revised 2026-11-05, from the per-well data). Half the floor for platform and onshore wells, whose stopped wells' low final rates sit near it. Raise the subsea floor towards the stopped subsea wells' low final rates, a design value of about 20 Sm³/d of oil equivalent with half of it as the margin [own analysis of BOEM's OGOR-A production (GoM deep water subsea, final_oil_equivalent_rate) and ANP's production by well (offshore pre-salt)]. Wells also stop for water, sand or a failed completion, so a final rate is an upper bound on the economic floor; the low percentiles are the evidence used.

**Ruling:** Bjarne, 2026-11-06: half the floor for platform and onshore wells; the subsea floor raised to about 20 Sm³/d of oil equivalent, half of it the margin (D-34, D-50).

**Bulk-approvable** (Bjarne, 2026-11-06: approved; D-21 ruled with B1):

- D-21 · Drains' and slant wells' productivity factors. *Not bulk-approvable as drafted* (2026-11-06, `drain_friction.py`, B1). Drop the cap of 6: $J = J_\text{Joshi}\thinspace\eta$, with Joshi's index on a drainage area that grows with the drain and the friction efficiency $\eta$ of Dikken's model in closed form, the liner taken as the tubing, at the design rate. For gas drains, replace the placeholder factor of 1.5 to 4 on $kh$, which physics does not support, by the drain's Forchheimer coefficients: $a$ over Joshi's ratio, and $b$ scaled by $(I_\text{ani} h/L)^2$ plus the friction term. The slant factor stays a placeholder (Economides et al.'s §4-6 and §5-4, Section 6).
- D-22 · Vogel where the saturation pressure is at least 0.8 of $p_r$ and the water cut below 0.6, else a productivity index.
- D-23 · Forchheimer's parameters for gas wells (its $D$ and mapping are revised by D-61 and D-62).
- D-24 · The rate target is a check, not a draw.

### E. Completion and choke

#### E1 · The bean size (D-27)

**Question.** Replace the fully open bean $\beta = d/D \sim U(0.25, 0.6)$ with R6's suggestion, $U(0.15, 0.45)$?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **$U(0.25, 0.6)$** (draft) | Large beans: the choke restricts little until it is nearly closed | The median sample makes 95% of its well's best rate; samples below $u$ = 0.3 make 70% of the rate of those above 0.9; 5.0% of wells vary QTOT by less than 5% (R6, Coverage); 34.8% of samples choked | v1's $K_c$ is $\beta$ = 0.27 to 0.53 at $C_D$ = 0.85; `sol-1` is 34.0% choked; UK gas wells' rate at a choke of 40% or less is close to that at 90% or more within a well-year at the median, with a long tail of wells that respond strongly [own analysis of NSTA's production by well: rate_ratio_choke_40_to_90] |
| $U(0.15, 0.45)$ (R6's suggestion) | Smaller beans: the choke holds the rate back over more of its travel | The rate responds to $u$ over more of the range, which virtual flow metering and choke optimisation need; more choked samples; lower rates (D1); more redraws at small openings, already 10% at $u$ from 0.05 to 0.15 | Textbook beans of 24/64 and 32/64 in are $\beta$ = 0.13 to 0.20 in 2-7/8 to 3-1/2 in tubing [WP §8]; no distribution of bean sizes over wells is in hand |
| $U(0.15, 0.6)$ (revised recommendation) | Small and large beans, a well each | Both behaviours in the data: wells whose choke holds the rate back over its travel, and wells whose rate barely moves until the choke is nearly closed | Covers the textbook beans (0.13 to 0.20) and v1's (0.27 to 0.53) |

**In real wells, by size** (2026-11-06). BSEE's well tests give the choke of every test of the GoM's wells since 1975 [own analysis of BSEE's well tests: choke_diameter]. A well's largest test choke is a lower bound on its fully open bean.

- **Oil wells.** R6's fully open beans in mm ($\beta$ times the tubing's diameter, `draws_r6.csv`) lie within the GoM's deep-water oil wells' largest test chokes, from the lower quartile to the upper tail. They lie above the shelf's, whose wells are mostly tested at or below 1 in.
- **Gas wells.** Their beans are larger: R6's median gas well has a bean near the real gas wells' upper quartile, and its upper tail passes the largest test choke of almost every real gas well. Their large tubing causes this, through D2's high rates, not the range of $\beta$.
- **Largest choke.** In nearly every group the largest test choke is 4 in (256/64 in).
- **Rates.** Tests at high rates use larger chokes than tests at middling rates, and the low-rate tests on the shelf run wide open on 1 in bodies.

**Recommendation** (revised 2026-11-06, from the well tests): $U(0.15, 0.6)$, which covers both the textbook beans and v1's range, with the bean capped at 4 in (101.6 mm), the largest choke tested in the GoM's groups [own analysis of BSEE's well tests: well, largest test choke]. Measure it in the rework after D1 and D2 have lowered the gas wells' rates and tubing, and check the beans in mm against these rows by setting and well kind. The rates' fix belongs to D1 and D2, not to the beans. The UK's per-well data show both behaviours of the rate against the opening [own analysis of NSTA's production by well].

**Ruling:** Bjarne, 2026-11-06: $U(0.15, 0.6)$ with the bean capped at 4 in (D-27).

**Bulk-approvable** (Bjarne, 2026-11-06: approved):

- D-25 · Tubing from the initial design rate and the erosional velocity. The draws find the design rate above the largest tubing's capacity in 7%, 21%, 12% and 6% of the black oils, volatile oils, condensates and gases, which keep the largest tubing; D1 and D2 propose to redraw them (2026-11-05, `erosional_capacity.py`).
- D-26 · Roughness by life stage.
- D-32 · The heat-transfer coefficient from Hasan and Kabir (2012) at the well's age.
- D-33 · Heat capacities from correlations (the gas's scaling with gravity is an assumption).

### F. Gas lift

#### F1 · Gas lift in gassier wells (goal 5; D-28)

**Question.** The top GOR quartile of oil wells is lifted 21% of the time, the bottom quartile 48%; gas classes 7% to 13% (R6, Summary by goal and Coverage). Does that meet "gas lift in gassier wells too"?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **As drafted** | Heavy, low-GOR columns fail the static check and need lift; gassy wells flow longer on their own | Lift curves concentrate in low-GOR oil wells; about a fifth of the gassiest oil wells still lifted | Review literature names high-GOR wells as lift candidates (von Flatern 2015); in the GoM, the gas-lifted share of oil well-years falls with the GOR quartile on the shelf and on deep-water platforms, less steeply than R6's, and rises with years on production and water cut; its subsea wells are rarely on gas lift [own analysis of BOEM's OGOR-A production] |
| Raise the drawn lift share for the upper GOR quartiles until each quartile is lifted about equally | Lifts some gassy wells that would flow | More lift curves in gassy wells, where added gas does less | As above |
| Raise the gas classes' late-stage share (0.15 now) | Lift for loaded gas wells | More lifted gas wells late in life | Gas lift for loaded gas wells is "applicable" (von Flatern 2015) |

**Recommendation** (revised 2026-11-05, from the per-well data). Keep the static check: the GoM's oil wells show the same fall with GOR. The fall is milder there, so raise the drawn lift shares of the upper GOR quartiles until R6's ratio of the top quartile's lifted share to the bottom's matches the GoM's platform wells' [own analysis of BOEM's OGOR-A production: gas_lifted by producing GOR quartile]. Keep D-28's stage shares, which lie within the GoM's lifted shares by years on production for platform wells, and the gas classes' shares, which are near the GoM's. The GoM lifts few subsea wells, but that is one region's practice, and ANP's data carry no lift method, so keep the subsea wells' shares.

**Ruling:** Bjarne, 2026-11-06: as drafted; keep it simple and do not lean on the GoM's data (D-28).

#### F2 · A natural oil well that cannot flow is lifted (D-101), and the lift capacity's floor (D-31)

**Question.** Accept D-101, and raise the capacity's lower bound from 0.02 kg/s?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **D-101** (R6) | An operator lifts a well that stops flowing rather than abandon it | 22 wells lifted, 8 kept; the label `lift_reason = no_natural_flow` follows a solve | Golan and Whitson's Example 4.11; `nsol-2` does the same (D-49) |
| The draft: drop them | – | 2.4% of the core wells lost (criterion 2) | – |
| Also raise the capacity floor to 0.1 kg/s (about 10,800 Sm³/d; WP-5's bound) | A lift system delivers a minimum rate, below which lifted wells head (Asheim 1988) | 12 of the 13 lifted wells that still failed had 0.02 to 0.04 kg/s (about 2,200 to 4,300 Sm³/d at 0.8 kg/Sm³), so more weak lifted wells would be kept | Examples inject 14,000 to 184,000 Sm³/d (von Flatern 2015) and 28,000 Sm³/d for a well of 100 Sm³/d (Guo et al. 2007) [WP §9] |

**In real wells** (2026-11-06). Petrobras' 3W satellite oil wells, subsea and gas-lifted, inject far more lift gas than R6's lifted wells can: R6's subsea capacity at the median is about a tenth of 3W's median injection rate, and R6's cap of 2.0 kg/s lies below 3W's highest well medians [own analysis of Petrobras' 3W Dataset: lift_gas_rate; `draws_r6.csv`]. The floor of 0.1 kg/s lies below nearly all of 3W's lifted wells. 3W publishes no production rates, so its injected GLR is unknown.

**Recommendation** (revised 2026-11-06): accept D-101 and raise the floor to 0.1 kg/s, measured in the rework. Raise the capacity's cap for subsea wells to about 3.5 kg/s, and check the lifted subsea wells' capacities against 3W's injection rates once D1 has raised the subsea wells' design rates, on which the capacity scales.

**Ruling:** Bjarne, 2026-11-06, decided from data at his ask: D-101 approved; the lift-gas capacity from 0.1 to 3.5 kg/s for every well type, with the injected GLR at capacity widened to 30 to 1000 Sm³/Sm³ times the design liquid rate (D-31).

**Bulk-approvable** (Bjarne, 2026-11-06: approved; D-29 by well type, D-38 with the lift gas off in 20% of the slots):

- D-29 · The injection depth at the pressure-balance point. *Not bulk-approvable as drafted* (2026-11-06). The draft draws every lifted well's surface injection pressure from onshore practice, $U(48, 76)$ bar. The GoM's gas-lifted wells' casing pressures reach that range's top and a little beyond [own analysis of BSEE's well tests: gas-lifted tests, casing_or_line_pressure]. 3W's subsea satellite wells take their lift gas at more than twice the range's top, both topside and in the annulus at the tree [own analysis of Petrobras' 3W Dataset: gas_lift_choke_downstream_pressure, annulus_pressure]. Draw the injection pressure by well type, in design values: platform and onshore wells about $U(48, 85)$ bar, subsea wells about $U(130, 210)$ bar at the tree. Set them on the rework's pilot, since the pressure-balance point, and so the injection depth, moves with them.
- D-30 · The lift gas's temperature, up to 30% towards $T_s$.
- D-38 · `sol-2`'s lift-gas rate, uniform to the capacity, off in 10% of the slots.
- D-49 · Gas lift that starts mid-life in `nsol-2`.

### G. Samples, coverage and `nsol-2`

#### G1 · Thin churn flow

**Question.** Churn flow is 1.5% of the `sol-2` samples at the bottom and 2.0% at the top, dispersed bubbly flow 0.5% at the top; the tail does not add them (R6, Coverage). Act now?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Accept, and re-measure after D2** | Churn needs gas rates near the churn–annular transition: gas wells near liquid loading, lifted wells at high lift-gas rates | About 30,000 churn samples per dataset; lower gas rates (D2) put more gas wells near loading | Loading sets in near 2.2 to 2.6 m/s of gas, for water at 60 bar [WP §6]; no public share of samples by regime |
| A coverage draw for regimes | Wells placed near the transitions on purpose | New draw machinery; labelled | – |

**Recommendation:** accept; re-measure after D2 and E1, and add a draw only if churn stays this thin.

**Ruling:** Bjarne, 2026-11-06: accepted; churn re-measured after D2 and E1.

**Bulk-approvable** (Bjarne, 2026-11-06: approved; D-41 and D-47 ruled with their revisions, D-44 with C1):

- D-37 · `sol-2`'s choke: 5% fully open, else uniform from 0.05. In 3W's normal operation, few wells ever open their production choke fully, and those that do stay there; the pooled time fully open lies above 5% and the typical well's below it (2026-11-06) [own analysis of Petrobras' 3W Dataset: share_of_time_production_choke_fully_open]. The UK's gas wells spend a share of their days fully open [own analysis of NSTA's production by well: share_of_days_fully_open]. A modest mass at fully open stands.
- D-39 · The separator pressure per sample, ±10%.
- D-40 · The drift, with the GOR tied to the drifted pressure.
- D-41 · `nsol-2`'s life span and the depletion clock. *Not bulk-approvable as drafted* (2026-11-05): real wells reach their best year within the first few years at the median, and in the last fifth of a stopped well's life its rate is a small fraction of its first year's in the GoM, Brazil and the UK, far below `nsol-2`'s 0.35 (R6); oil wells' Kaplan–Meier lives centre near the draft's median, with a lower tail well below its 8 years [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: rate_over_first_year by fifth of the life, producing_life]. The data argue for a steeper decline (a larger $D$, or lives that run to the floor), set on the rework's pilot against these rows.
- D-42 · `nsol-2`'s reservoir pressure.
- D-43 · `nsol-2`'s water: a jump at breakthrough, then the semilog WOR. Over every field-year since 1971, Norwegian oil fields' median water cut lies below the draws' at every age, most in the first ten years (fields, not wells; 2026-11-05) [own analysis of Sodir's FactPages: NCS oil fields by years on production]. Per well, the medians by years on production in the GoM, Brazil and the UK bracket the draws' [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: water_cut by years on production].
- D-44 · `nsol-2`'s GOR, without the late fall. Per well, the median real GOR stays near its first year's (C1) [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: gor_over_first_year]. Revised with C1 (2026-11-06): the material balance's factor at $p_r(t)/p_\text{sat}$; leaving out the late fall agrees with the physics, which puts it below every core state.
- D-45 · Well properties that age; the inflow model fixed.
- D-46 · The separator's two step-downs.
- D-47 · Piecewise-constant set points. *Not bulk-approvable as drafted* (2026-11-06). In 3W's normal operation, the typical well's production choke is moved by a few points every day or so [own analysis of Petrobras' 3W Dataset: production_choke_changes_per_day]. D-47 changes it about once in 11 months, so between two of `nsol-2`'s samples a real well's set point has almost always moved. Change the choke's set point at most samples, mostly by small steps, keeping the occasional redraw, and set the rates on the rework's pilot. The 3W windows are hours to weeks around events, so they show how often operators adjust, not how far a set point travels over a life.
- D-48 · Eight test periods of five sweeps.

### H. Generation, redraws and filters

#### H1 · Drop the CasADi retry (D-54, D-82)

**Question.** Keep re-solving the Rust core's solver failures with the CasADi backend?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Drop it** (R6) | – | Solver failures are redrawn and logged with their reason; the samples are conditioned on the core's failures, which shift no input by more than a KS $D$ of 0.0001 | – |
| Keep it | – | 116 retries over four runs recovered 2 attempts, both of a kind D-102 now calls no operating point; none of 23 under R6's rules; 0.2% to 4.8% of the CPU, 15 to 28 Rust solves each (R6, The CasADi retry) | – |

**Recommendation:** drop it (constitution, principle 7: no measured gain). D-56's starts then serve only comparison runs; R11 keeps D-92's CasADi re-solve of 100 rows.

**Ruling:** Bjarne, 2026-11-06: dropped (D-54, D-82).

#### H2 · What counts as no operating point (D-102, D-53)

**Question.** Count a scan with no root and no rejected sign change as no operating point, whatever its non-finite scan points?

- **Physically:** the core leaves out scan points where a march leaves the closures' range, near choked cells or where Joule–Thomson cooling leaves a cell's energy row without a root.
- **Model and ML tasks:** decides which attempts the data call physical. 14% of the operating points' searches have such points; of 72 such attempts retried, CasADi found no root in 70. About 3% of them may be misclassified as physical.
- **Under D-53 as drafted:** 4 tail wells were dropped at the solver-failure limit, 3 for such attempts only.

**Recommendation:** accept D-102.

**Ruling:** Bjarne, 2026-11-06: D-102 accepted.

#### H3 · The drop targets and the redraw limits (D-81, D-82, D-83)

**Question.** The targets are at most 1% of a class's wells dropped at the limits (D-81), 0.5% of `nsol-2` wells as `dead_date` (D-83), and 0.5% of attempts and wells as solver failures (D-82). The pilots exceed the first two: black and volatile oils 1.4% each in `sol-2` (3 of 218, 1 of 71), one `dead_date` of 109 `nsol-2` wells (0.9%), and one redraw-limit well of 46 `nsol-2` black oils (2.2%) (R6, Outcomes).

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Keep targets and limits; judge on R11's pilot** | The wells at the limit are lifted wells that flow only at high lift-gas rates, and late natural oil wells at their floor | Their drop shifts no well input by more than a KS $D$ of 0.010 | Such wells exist: lifted wells head below a minimum injection (Asheim 1988) |
| Loosen to 600 redraws per well | Keeps wells that flow in a quarter of their draws | More weak wells; up to 800 solves for one well | As above |
| Tighten to 200 redraws per well | Keeps only wells that flow in half their draws | Drops more weak wells | As above |

**Recommendation:** keep the targets and limits, and judge them on R11's pilot: 1 to 3 wells per class is too few to tell 1.4% from 1%. F2's capacity floor may remove some of these drops.

**Ruling:** Bjarne, 2026-11-06: kept, judged on R11's pilot (D-81, D-82, D-83).

#### H4 · Two solver findings

**Question.** Investigate before R11, or log to the backlog?

- **`nsol-2` well 62** (the draft's run, a late tail black oil): root sets with an unstable root and no stable one at 15 dates, on both backends, with rejected sign changes in 64 scans. A continuous $R(p_0)$ cannot do that (D-53), so a root is missed or $R$ is discontinuous. 1 of 100 `nsol-2` wells.
- **CasADi misses a stable root** in 4 of 200 compared `sol-2` samples, highly deviated wet-gas wells at PBH of 14 to 41 bar, where the core finds it (R6, Backends). It does not touch the generation once H1 drops the retry.

**Recommendation:** log both to the backlog, and look at well 62 in R9 next to feature 022's measurements of $R(p_0)$ in gas wells, since R11 counts an unstable-only root set as a solver failure.

**Ruling:** Bjarne, 2026-11-06: logged to the backlog; well 62 looked at in R9 with feature 022.

**Bulk-approvable** (Bjarne, 2026-11-06: approved):

- D-51 · SMP-28's three filters dropped; backlog §1.11 lapses.
- D-55 · Wells whose best-rate solve fails are dropped and counted.
- D-56 · The starts (only for comparison runs once H1 drops the retry).
- D-80 · Redraws from the slot's own generator, solved in passes.
- D-84 · `nsol-2`'s life ends at the last scan point that passes; lift starts before natural flow fails.
- D-93 · The process pool, resumable chunks, a clean commit; no batch API (R6 measures at most a 0.1% gain, against D-93's 20%).

### I. Datasets

Publishing is not asked here: Bjarne publishes the datasets himself, after validating and checking them privately (2026-11-05). Where they go (D-86 proposes the same repository, `solution-seeker-as/manywells`, under `v2/`) and whether they get a DOI are his choices then.

#### I1 · The logs in the datasets (D-85)

**Question.** Make `redraws` (about 130,000 rows per dataset at R6's rate) and `drawn_wells` part of each dataset?

- **For the tasks:** the attempts without an operating point mark where each well cannot flow, which surrogates and production optimisation can use, and the logs let a user measure and undo the redraws' conditioning (criterion 4).
- **Against:** larger files, and more to document.

**Recommendation:** include both; they are synthetic and hold nothing confidential.

**Ruling:** Bjarne, 2026-11-06: the logs go in a separate companion file next to each dataset, not in it (D-85).

#### I2 · The re-solve tolerance (D-92)

**Question.** A re-solved row gives identical features on the generator's platform and code version, and elsewhere agrees within the verifier's `tol_x` applied to the features: pressures within $10^{-4}\thinspace(p_r - p_s)$, temperatures within $10^{-4}\thinspace(T_r - T_s)$, mass rates within $10^{-4}$ of WTOT. Accept? The alternative stores each operating point's full state, about 11 GB per dataset.

- **Measured:** 1500 rows re-solved identically; CasADi agrees within a relative $8\cdot10^{-8}$ (R6, Backends).

**Recommendation:** accept.

**Ruling:** Bjarne, 2026-11-06: accepted (D-92).

**Bulk-approvable** (Bjarne, 2026-11-06: approved):

- D-86 · Parquet, one file per table; where the files are published is Bjarne's choice when he publishes them.
- D-87 · Four tables and two logs, each input in one place (with the edit to `specs/goals.md`, Section 4).
- D-88 · Column names and types: v1's names kept, float64, nulls.
- D-89 · Labels as strings with fixed codes.
- D-90 · An 80/20 train–test split by well, stratified.
- D-91 · A seed per dataset; `ID` and `draw_index`.
- D-94 · A datasheet per dataset, in the repo.

### J. The model's gaps: feature 022 and the table fluids

#### J1 · Feature 022's form (D-60)

**Question.** Approve feature 022 as drafted: Forchheimer's law in the pressure squared on the reservoir gas's mass rate (INF-10), the liquid from the gas by the fluid's ratios (INF-11)? Section 5 summarises it.

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Forchheimer in $p^2$, constant $a$, $b$** | Darcy flow with an inertial term; $a$ and $b$ from the rock and gas | Finite slope at $p_r$, defined above it; fitted coefficients within 2% of the pseudo-pressure at the median and 6% at P5 and P95 from 20% drawdown to the AOF (D-61) | Gas and condensate wells are 35% of the core; 79% of R3's gas wells lie above 138 bar, where the $p^2$ form needs the fit |
| The backpressure law | Empirical, from field tests | Fits a test as well (−14% to +11% at the AOF against −22% to 0%); infinite slope at $p_r$, not real above it, no expression in the reservoir's properties | $n$ from 0.5 to 1 in hundreds of tests |
| A tabulated pseudo-pressure | Exact at any pressure | A table per well on the inflow | – |

**Recommendation:** approve feature 022 as drafted. `v2.0.0`'s check accepts the option (decided in the Goals).

**Ruling:** Bjarne, 2026-11-05: approved as drafted, Forchheimer's law in $p^2$ with constant $a$ and $b$ (D-60). D-61 and D-62 remain open.

#### J2 · The table fluids' descriptors: single-stage or field-reported (D-63)

**Question.** The compositions match the drawn API and gas gravity as single-stage values at standard conditions. Separated in one stage, the published examples have GORs 12% to 19% higher, API 3 to 6 °API lower and gas gravities 0.05 to 0.07 heavier than reported (`fluids-draft.md`, The composition). Shift R3's targets by those offsets?

| Option | Physically | Model and ML tasks | In real wells |
|---|---|---|---|
| **Single-stage, as drafted** | A table fluid is 3 to 6 °API lighter in field terms than the drawn statistic | Volatile oils and condensates sit slightly lighter than the black oils' statistics imply | The drawn statistics are field-reported (BOEM, McCain's limits) |
| **Shift the targets by the examples' offsets** | Table fluids match field-reported values | Consistent with the black oils | Offsets from three examples only |
| Match on two stages at the well's separator | Field-reported exactly | Breaks the single tie line that PVT-TAB-1's GOR axis needs | – |

**Recommendation** (revised 2026-11-06: physics decides). Compute each fluid's own offsets, not the three examples': in R9, flash the matched composition through the well's own separator (its stages at $p_\text{sep}$ and $T_\text{sep}$, then stock tank) and through one stage at standard conditions, and shift the targets by the difference (one or two passes, since the offsets depend on the composition), so that the table fluid's field-reported GOR, API and gas gravity are the drawn ones. The single tie line that PVT-TAB-1's GOR axis needs is kept, since the matching stays single-stage; only the targets move. Record the offsets with the fluid.

**Ruling:** Bjarne, 2026-11-06: per-fluid offsets by each fluid's own separator flash, in R9 (D-63).

**Bulk-approvable** (Bjarne, 2026-11-06: approved):

- D-61 · Forchheimer's coefficients fitted to the pseudo-pressure at the wellstream's gravity.
- D-65 · The table's saturation pressure replaces R3's; a fluid is saturated at discovery where it must be. Published evidence supports using the untuned flash's dew point (2026-11-06): over 340 gas condensates, Peng–Robinson without tuning predicts the dew point with an average absolute deviation of 8.6% to 9.3% and a bias of −1% to +3%, by the plus fraction's characterization, against measurement errors of 5% to 10% in the data themselves (Elsharkawy 2002, Table 3 and §8.1). R5's one check, 8% below the measured dew point, lies within that spread, and the table and its saturation pressure come from the same flash, so the fluid stays consistent. The paper's samples span measured dew points of 108 to 816 bar at 4 to 171 °C (Table 1). R3's interim draw, $p_\text{sat} = p_i(1 - x)$, gives R6's condensates 76 to 780 bar at P5 to P95, its low end below any measured sample, a further reason to let the flash set the dew point from the composition.
- D-66 · Six published example fluids, a fifth of the table wells.
- D-67 · One table set per well, on its own grid and GOR axis.
- D-68 · NeqSim as a build-time tool of the generator, outside ManyWells' dependencies.
- D-69 · R6's gas wells on the interim index (no sign-off needed; R6 did not run the CasADi prototype it proposed).

### K. Condensate tables (D-64)

Bjarne asked on 2026-11-02 how other simulators represent gas condensates before he rules. The research pass is `sources-condensate-tables.md`, with the evidence script `plans/evidence/condensate_rv_form.py`.

**How other tools carry a condensate's liquid:**

| Tool | How it carries the liquid | Above the dew point | Checks on order |
|---|---|---|---|
| Modified black oil (Whitson and Brulé 2000, §7.3) | $R_s$, $B_o$ for the liquid; $R_v$, the oil vaporised in the gas | undersaturated gas | none stated |
| OPM Flow and the Eclipse format (PVTO, PVTG) | $R_v$, tables against pressure | a switch of primary variable; no trace needed | $R_s$ strictly increasing; $B_o$ and $R_v$ decreasing within each undersaturated record; PVDG's $B_g$ decreasing; the saturated $R_v$ unchecked |
| MRST | as Eclipse | as Eclipse | none in the code |
| NeqSim's black-oil converter | each phase of the flash separated alone, one temperature | flags the fluid as a retrograde condensate | none in the converter |
| OLGA's `.tab` tables (from PVTsim, Multiflash, NeqSim) | the gas mass fraction and each phase's density on $(p, T)$ | gas mass fraction 1; the absent liquid filled from the nearest node | only nonzero densities |
| `fluids`, `thermo` | no black-oil tables | – | – |

PIPESIM, Prosper and LedaFlow publish too little to say.

**What it found for ManyWells:**

- **No tool checks the order of a net transfer** such as ManyWells' $R_{so}$, which is gas dissolved in the oil less oil vaporised in the gas. The check PVT-TAB-1 has today comes from oils, where it holds.
- **The trace is common practice.** OLGA's table generators fill an absent phase from the nearest node, as R5's trace of $10^{-4}$ does; it moves the no-slip density by less than $10^{-4}$ of itself.
- **$R_v$ is not the same as $R_{so} < 0$.** A modified black-oil state maps exactly onto ManyWells' columns at one GOR, but not back. On Pedersen's condensate, $R_{so} > 0$ at 143 of 385 two-phase nodes although the gas there carries up to 47% of the stock-tank oil.
- **PVT-TAB-6 is the more accurate form at the nodes.** A modified black-oil table made the usual way errs in the gas mass fraction by up to 0.12 (median 0.0064). At 0.8 and 1.25 times the GOR, the black-oil rule on $R_{so}$ errs by 3.0% and 2.5% in no-slip density at the median, the modified black oil by 0.44% and 0.45%; the GOR axis that D-67 proposes does better than both, 0.01% to 0.3%.
- **$R_v$ is not monotone either**: it has a minimum at 5 to 70 bar, as in Whitson and Brulé's Fig. 7.10, so a "monotone $R_v$" check would refuse their own example.
- **Two checks hold for every fluid in hand:** the oil phase's mass $\rho_o + R_{so}\rho_{g,\text{sc}} > 0$, and the free gas's $p/Z$ strictly increasing in $p$, at every node pair of all eight tables (the five proof tables and three in `tests/data/pvt/`), as Eclipse's PVDG requires of $B_g$.

#### K1 · How tables take condensates (D-64)

**Question.** PVT-TAB-1 refuses every condensate table: below the dew point $R_{so}$ falls as the pressure rises (retrograde condensation), and above it the stream is all vapour, so $B_o = 0$. Condensates are 15% of the core wells. How should a table take them?

| Option | What changes | Model and datasets | Cost and precedent |
|---|---|---|---|
| **A′: trace, drop the order check, add two checks** (recommended) | A liquid trace of $10^{-4}$ where the flash has none; no check on the order of $R_{so}$; checks that the oil phase's mass is positive and $p/Z$ rises with $p$ | Condensate tables build; the rows and both backends unchanged; the new checks still catch swapped columns and wrong units | About a day with tests; the trace as OLGA's generators fill an absent phase; the $p/Z$ check as Eclipse's on $B_g$ |
| A: trace, drop the order check (R5's draft) | As A′ without the two checks | As A′, with weaker input checks | About half a day |
| B: A′ plus a declared kind | A field marks a table as oil or dew-point fluid; oil tables keep the order check | As A′, with the old check on oils | About a day more, and new API; can be added later on top of A′ |
| C: an $R_v$ form in the model | A new PVT form with vaporised oil | Less accurate at the nodes than PVT-TAB-6; both backends change | A feature spec and weeks of work |
| D: OLGA-style columns, $\rho_L$ in place of $B_o$ | A new table schema | Without water, the liquid mass rate is still zero above the dew point, so the trace moves rather than disappears | Several days to a week, both backends, and the CSV schema breaks; a reserve if a tiny $B_o$ causes trouble in R9 |
| E: no condensate tables | Condensates stay black oil | Free-gas share 0.43 against the table's 0.84 above the dew point (proof); against the Goals | None |

**When.** In R9, with feature 022, before the rerun of R6's pilots: the Rust core's march on a retrograde table is then measured on solved wells, together with $B_o$ down to $5\cdot10^{-4}$ (R5's proof).

**Recommendation:** A′, implemented in R9. The proposed text is in `spec-changes.md` (`specs/model/pvt/table.md`).

**Ruling:** Bjarne, 2026-11-06: A′, implemented in R9 (D-64).

## 4. Proposed changes to decided specs

From `spec-changes.md`. **Every row needs Bjarne's sign-off** (`AGENTS.md`, Needs Bjarne's sign-off). *Signed off in substance by Bjarne on 2026-11-06; each edit's final text is reviewed in the PR that makes it (R9, R10).*

| Spec file | What the change says | From | Sign-off |
|---|---|---|---|
| `specs/features/022-gas-rate-inflow.md` | Approve the feature spec; approval comes before its implementation (R9) | D-60 (J1) | approved 2026-11-05 |
| `specs/goals.md`, Compatibility → Datasets | Each sample is an operating point; a draw without one is redrawn and logged; two main tables, plus the surveys and the PVT tables, with the two logs in a companion file next to each dataset (I1, 2026-11-06) | D-52 (ruled), D-85 (ruled), D-87 | approved in substance 2026-11-06 |
| `specs/goals.md`, Direction for v2 → v2.0's model | Names Forchheimer's law (022); volatile oils and condensates as tables from a flash, wet and dry gas as black oil; every cell at most 75°, drains modelled from the heel | D-9, D-10, D-36, D-60 | approved in substance 2026-11-06 |
| `specs/goals.md`, header and Decided elsewhere | Adds the ruling of 2026-11-02; points the columns, filters and redraws to `docs/datasets.md` and `specs/sampling.md` once written | D-52 | approved in substance 2026-11-06 |
| `specs/constitution.md`, Dataset reproducibility | A row and the rows it references (surveys, PVT tables) carry the inputs to re-solve it | D-87, D-92 | approved in substance 2026-11-06 |
| `specs/model/validity.md` | Status line; decision 5 closed (no horizontal cell, at most 75°, drains to the heel); the inclination row's rule; rows for INF-10 and INF-11; notes 9 and 10 updated; a new note 12 on the $p^2$ form's error; items 1 and 3 of Where the sampler leaves the ranges | D-9, D-10, D-22, D-57, D-60, D-61 | approved in substance 2026-11-06 |
| `specs/model/inflow.md` | INF-10 and INF-11; `liquid_mass_flow_rate(p_0, p_r, fluid=None)`; the options row; safeguards (odd extension above $p_r$, refusals); sources, vectors, coverage | D-60 | approved in substance 2026-11-06 |
| `specs/model/solution.md` | SOL-2 lists INF-10 among the inflows that need $p_0 < p_r$ | D-60 | approved in substance 2026-11-06 |
| `specs/model/README.md` | The Inflow row; independent checks for INF-10 and INF-11; a fifth well in `vectors/v2_rows.json`, a change to `v2.0.0`'s row vectors | D-60 | approved in substance 2026-11-06 |
| `specs/model/nomenclature.md` | $a$, $b$, $\Delta$; a gas well's CGR and WGR held as `gor` and `wlr` | D-60 | approved in substance 2026-11-06 |
| `specs/model/pvt/table.md` | PVT-TAB-1 drops the check that $R_{so}$ does not fall with pressure, and adds two: the oil phase's mass positive, and $p/Z$ strictly increasing in $p$; PVT-TAB-6 keeps a liquid trace $\delta$ where the stream is all vapour; feature 018's ruling 10 lapses | D-64 (K1) | approved in substance 2026-11-06 (K1) |
| `specs/architecture.md` | The `InflowModel` row with the fluid argument; the discretization's call; the Rust core's refusal of a gas-rate inflow without gas; the feature map | D-60 | approved in substance 2026-11-06 |
| `specs/features/README.md` | A row for 022 once approved | D-60 | approved in substance 2026-11-06 |
| Not drafted yet (R10, R11) | The v2 version of `specs/sampling.md` with SMP IDs; the v2 schema in `docs/datasets.md` | all entries | approved in substance 2026-11-06 |

## 5. Feature 022 in brief

`specs/features/022-gas-rate-inflow.md`, a draft for approval; nothing is implemented.

- **Why.** Vogel's inflow was derived for solution-gas drive oil reservoirs (`validity.md`, note 10), and every inflow today returns the liquid rate. The interim index gives 0.57 times the reference rate at 5% drawdown and 2.05 times the AOF, at the median of R3's 703 gas and condensate wells.
- **INF-10.** $p_r^2 - p_0^2 = a\thinspace w_{g,\text{res}} + b\thinspace w_{g,\text{res}}^2$ with constants $a > 0$ and $b \ge 0$ in bar², s and kg, solved in closed form, with an odd extension above $p_r$ so that the inflow is finite and strictly decreasing everywhere.
- **INF-11.** The liquid comes with the gas in the fluid's ratios, INF-4 read backwards, so the rows downstream are unchanged.
- **Interface.** `Forchheimer(a, b)` in `inflow.py` and `Inflow::Forchheimer` in Rust; `liquid_mass_flow_rate(p_0, p_r, fluid=None)` gains the fluid, which the liquid inflows ignore.
- **Against the backpressure law.** Forchheimer's law has a physical basis, a finite slope at $p_r$ and a value above it; the backpressure law's slope is infinite at $p_r$ and it has no expression in the reservoir's properties.
- **Error and how the sampler removes it.** Above 138 bar the $p^2$ form with $\mu_g Z$ at $p_r$ gives 14% less than the pseudo-pressure at the AOF (median). The sampler fits $a$ and $b$ to the pseudo-pressure (D-61): 2% at the median, 6% at P5 and P95.
- **Configurations.** Off in `v1.0.0`, whose rows and verifier reports are unchanged; accepted by `v2.0.0`'s check (decided in the Goals).
- **Acceptance.** Vectors for INF-10 and INF-11 and a fifth `v2.0.0` row well; unit tests with two worked examples (Golan and Whitson's Example 2.4, Economides et al.'s Example 4-7); property tests on solved gas wells, liquid loading included; overlays in the backend comparison; the Rust core's tests; measurements on sampled gas wells on both backends.
- **Out of scope.** The backpressure law, a tabulated pseudo-pressure, a condensate bank near the well, a horizontal gas well's own law, and calibrating $a$ and $b$.
- **Open.** A condensate on its table needs D-64 (K1). D-69's CasADi prototype was not run, so no solved well has the inflow yet.

## 6. Resources wanted

*Revised 2026-11-05: data and physics decide where they can (Bjarne, 2026-11-05).* The sources are sorted by who can get them and by the rulings they would settle. The per-well data fetched since the checkpoint (ANP, NSTA, Sodir's bulk tables) and BOEM's production by completion, in hand since R2 and summarised on 2026-11-05, now give C1, D1, D2, D3, F1 and D-41 their evidence [own analysis of ANP's production by well, NSTA's production by well, Sodir's FactPages and BOEM's OGOR-A production]. The open data fetched at Bjarne's go-ahead on 2026-11-06 (BSEE's well tests, Petrobras' 3W, NOAA's ocean atlas, and USGS's produced waters, which he downloaded) give E1, F2, D-11, D-14, D-15, D-29, D-37 and D-47 theirs [own analysis of BSEE's well tests, Petrobras' 3W Dataset, NOAA's World Ocean Atlas 2023 and USGS's produced-water database]. The analyses, their scripts and the publishers' licences are kept outside the repo (the maintainer's decision, 2026-11-06).

### Only Bjarne can get

*Revised 2026-11-06:* three of the four requests arrived and are used (No longer needed, below). The one left is optional: D1's top end already has the tubing's erosional capacity.

| Reference | Settles | How | Assumed without it |
|---|---|---|---|
| ANP's production by well from 2024 (its query tool, behind a captcha): two or three recent months of the offshore file | D1's top end | Data: whether the newest pre-salt wells pass the best months of 2005 to 2023 | ANP to 2023 [own analysis of ANP's production by well: well peak] |

### In hand, for the agents to read

- Economides et al. (1994), §4-6 and §5-4: D-21's slant factor and the gas drains.
- Guo et al. (2007), Appendix B: D-25's tubing inner diameters.

### No longer needed

| Reference | Was for | Now |
|---|---|---|
| Per-well production offshore outside the GoM; ANP's production by well; BOEM's production by completion; Sodir's bulk tables | D1, D2, D-4, D-16, D-41, D-43 | Fetched and summarised [own analysis of ANP's production by well, NSTA's production by well, BOEM's OGOR-A production and Sodir's FactPages] |
| Joshi (1991), *Horizontal well technology*; Brekke and Lien (1994), *JPT*, in place of Dikken (1990) | D-21, B1 | Obtained on 2026-11-06; Dikken's coupled model as Joshi gives it (Ch. 10) sets the drains' productivity by physics (`drain_friction.py`, B1, D-21) |
| Craft and Hawkins (1991), *Applied petroleum reservoir engineering* | C1, D-19, D-44 | Obtained on 2026-11-06; Muskat's material balance sets the GOR's rise by physics, and the per-well data choose its $k_g/k_o$ (`solution_gas_drive.py`, C1) |
| Elsharkawy (2002), *Fluid Phase Equilibria*; Nemeth and Kennedy (1967) not needed | J2, D-65 | Obtained on 2026-11-06; the untuned equation of state's dew-point errors over 340 condensates (Table 3) support D-65; J2's offsets are computed per fluid (J2) |
| Petrobras 3W 2.0.0 | D-15, D-29, D-31, D-37, D-47, F2 | Fetched and summarised on 2026-11-06 [own analysis of Petrobras' 3W Dataset]; it has no production rates, so D-48's test schedule and the injected GLR stay without data |
| Choke vendors' catalogues; BSEE's scanned well potential test reports; choke-correlation papers | E1 (D-27) | BSEE's tabulated well tests, 1975 to 2026, give the tested choke sizes [own analysis of BSEE's well tests] |
| NOAA, World Ocean Atlas 2023 | D-14 | Fetched and summarised on 2026-11-06 [own analysis of NOAA's World Ocean Atlas 2023] |
| USGS, National Produced Waters Geochemical Database (ver. 3.0) | D-11 | Downloaded by Bjarne and summarised on 2026-11-06 [own analysis of USGS's produced-water database] |
| The highest per-well oil and gas rates on public record | D1, D2: the top end | Physics: D-25's erosional capacity (`erosional_capacity.py`), with the per-well best months as a check (D1, D2) |
| Kaiser (2011); Kaiser and Yu (2010) | D3 | The final rates of wells that stopped (D3) |
| Fetkovich (1973) | D-45, D-100 | The permeability set by well type against per-well rates (D1) |
| Brown (1980); Arnold and Stewart, *Surface production operations* | D-15, D-29, D-39 | 3W's and BSEE's pressures [own analysis of Petrobras' 3W Dataset and BSEE's well tests] |
| A global count of offshore producing wells by fluid type | D-4 | Commercial databases only; the GoM's, Norway's, the UK's and Brazil's counts serve |
| Höök et al. (2009, 2014); IEA (2008); Arps (1945) | D-41 | Per-well declines and lives (D-41) |

### Low priority, kept for the record

| Reference | Bears on | Assumed without it |
|---|---|---|
| API RP 14E; Turner, Hubbard and Dukler (1969); Coleman et al. (1991) | D-25; gas wells' loading (G1) | The rules as quoted [WP §6] |
| Swarbrick and Osborne (1998), or another review of overpressure | D-12 (lithostatic gradient) | 0.226 bar/m; Nadeau et al. (2023), Fig. 20 |
| The paper by Bertuzzi et al. that Asheim (1988) cites | D-38, F2 | Asheim (1988) |
| Fevang and Whitson (1996) | J1 (condensate bank, out of scope) | The skin stands for it |
| McCain (1994) | D-3 | McCain (1990)'s limits |
| McCain (1991); Spivey, McCain and North (2004) | D-11, D-33 | A quadratic through McCain (1990); Sharqawy et al. (2010) |
| Farshad and Rieke (2006) | D-26 | The stage split |
| Alves, Alhanati and Shoham (1992); Willhite (1967) | D-32 | Hasan and Kabir's examples |
| Ehrenberg, Nadeau and Steen (2008b) | D-12, D-13 | BOEM sands |
| Taitel and Dukler (1976); Barnea et al. (1985); Barnea (1987); Shoham (2006) | D-10 (ruled) | Our scan of Taitel–Dukler |
| More surveys: NSTA's per-row files and deviation surveys, ANP's well table, AOGCC, NOPIMS; Bourgoyne et al. (1986); Mitchell and Miska (2011) | D-9 (ruled), B1 | GoM and Dutch surveys |
| Forchheimer (1901); Rawlins and Schellhardt (1936); Al-Hussainy et al. (1966) | J1 | As Golan and Whitson (1991) and Ahmed (2006) give them |
| PIPESIM, Prosper and LedaFlow technical manuals | K1 (D-64): how commercial pipe simulators carry a condensate's liquid | The open tools and OLGA's table format (`sources-condensate-tables.md`) |

## 7. What we are least sure of

- **The interim mapping.** Every number on gas wells, condensates and volatile oils, and every number over all wells, is from the interim mapping: a third of the wells on a linear index that errs by up to a factor 2, and a quarter as black oil whose bubble points are Vazquez–Beggs's (556 and 426 bar against the tables' 386 and 359 in the proof). Feature 022 and the tables may move the rates, regimes, choked shares and drops.
- **The rates.** The permeability was tuned to a check (D-100), not drawn from a per-well statistic. Per-well offshore rates are now in hand for the GoM, Brazil and the UK, and they show rates that depend on the well type, which the draft does not draw (D1, D2).
- **Condensate tables in the solver.** Whether the Rust core's march and temperature solve hold with $R_{so}$ falling in $p$ is untested; the equation of state is untuned (8% low on the one measured dew point).
- **Own choices without counts:** the stage shares, 10% lift-off, the test schedule and the tail's form. The lift shares, the GOR factors and the decline now have per-well data to be checked against (F1, C1, D-41); the beans have the GoM's tested choke sizes (E1); the choke's history, the lift-gas rates and the injection and subsea pressures have 3W's 40 wells, observed over short windows (D-47, F2, D-29, D-15).
- **Small pilots.** 400 and 100 wells; the drop targets were judged on one to three wells per class.
- **Regional samples.** Every sample of real wells in hand covers some fields and areas only, the Gulf of Mexico above all; Brazil and the UK add per-well data since 2026-11-05, and Norway its production by field; since 2026-11-06 the GoM adds its well tests, Petrobras 40 subsea satellite wells' operation, and the US its produced waters; other regions, such as West Africa, are missing. The support's ends rest on few sources.
- **Two solver findings** are unexplained (H4).
- **Placeholder factor** for slant wells (D-21) affects the highly deviated class, a quarter of the wells; the drains' and gas drains' productivity now follow physics (B1, D-21).
- **The cost of table fluids.** Making them is scaled from R5's proof on four fluids, and solving them is not measured; R6's projection leaves both out (Section 8).

## 8. After the checkpoint

Rulings that reverse an entry send its work back through R2 to R7, and a smaller checkpoint covers only what changed. The likely rework from the recommendations above: the beans and their cap at 4 in (E1); the permeability by well type against the per-well rates, with a redraw above the tubing's erosional capacity (D1, D2); the subsea floor (D3); the lift shares of the upper GOR quartiles (F1), the capacity's floor and its subsea cap (F2), and the injection pressure by well type (D-29); the subsea flowline loss and its truncation (D-15); the choke's set points changed at most samples (D-47); the salinity's median and floor (D-11); the GOR's rise from the material balance (C1, D-19, D-44); the drains to 2500 m with friction in place of the cap of 6, the kickoff below the mudline and the lower high-angle tangent (B1, D-21); the GOR tied to depletion and the water cut's upper tail (C1); a steeper decline in `nsol-2` (D-41); the shapes' parameters (B1) and the stage range (C1), measured on a rerun of R6's pilots against the per-well statistics.

| Step | Work | Cost |
|---|---|---|
| **R9 · The features** | Feature 022 through `AGENTS.md`, Adding an equation, in its own worktree: spec files, Python and Rust, vectors, tests, backend comparison, and measurements on sampled gas wells on both backends. If D-64 is ruled, the change to PVT-TAB-1 and PVT-TAB-6, with a Rust assumption test on a condensate table. Then R6's pilots rerun with the features and the tables. | Estimates. The measurements: under an hour on 12 processes at CasADi's 3.7 s per case. The pilots' rerun: about 26 and 8 minutes on 11 processes (R6, What was run), plus their tables, up to about 40 minutes of one process for each pilot (D-67's 2 to 55 s per well). |
| **R10 · The sampler** | The v2 version of `specs/sampling.md` with SMP IDs and a Coverage table, v1's procedure kept; v2's draws next to v1's in `manywells.sampling`; the composition and table step in NeqSim's own environment; the generator's passes, logs, chunks and assembly; helpers that build a case from the rows and a row from an operating point; tests of marginals, rank correlations, bounds and reruns. No batch API. | No heavy runs: the full suite, both verifier candidates, and the Distributions check (about 17 minutes on 24 cores at 5 samples per well). |
| **R11 · The datasets** | A pilot of a few hundred wells per dataset, checked and reviewed by Bjarne (his private comparisons if he wishes); then the full generation after his go-ahead; the datasheets, the v2 schema in `docs/datasets.md` and the CHANGELOG. Bjarne then validates and checks the datasets privately and publishes them himself; publishing is not a step of the plan. | Estimates from R6. The pilot: about half an hour per dataset on 12 processes. The full generation: about 240 CPU-hours, 22 hours on 12 processes for both (R6, Cost), plus making the tables, up to about 1.3 and 4 hours on 12 processes for `sol-2` and `nsol-2`, and the table fluids' solves, not measured yet (below). About 0.4 GB of samples per dataset before compression (R6, Cost). |

**The table fluids' cost.** Neither R6's projection of 22 hours nor R7's estimate (`datasets-draft.md`, Cost) includes making the PVT tables or solving the table fluids: R6 solved them as black oil. D-66 and D-67 also assume 2500 table wells per dataset, but R6's kept mix has 32% volatile oils and condensates (128 of 400), about 3200 per dataset, which raises the tables' cost and size by 3200/2500, about 1.28 (`fluids-draft.md`, One table per well, or a library). An estimate from R5's proof:

- **Making the tables.** 0.5 to 4.6 s per table, one table per GOR, plus 0.15 to 3.6 s to match the composition and find its saturation pressure. A `sol-2` well with about 3 GORs takes 2 to 17 s, an `nsol-2` well with up to 11 GORs 6 to 55 s. For 3200 wells, about 2 to 15 CPU-hours for `sol-2` and 5 to 49 for `nsol-2`: at most about 1.3 and 4 hours on 12 processes, and at most about a quarter on top of R6's 240 CPU-hours. `sol-2`'s tables take up to about 1 GB of CSV before compression.
- **Solving on the tables.** Not measured. Feature 018 found a table no dearer than its black oil on the Rust core, for oils (`specs/features/018-pvt-tables.md`, Cost); if that holds here, R6's projection covers the solves. A condensate's table, with $R_{so}$ falling in $p$, is untested (D-64).

## 9. Where everything is

**On the branch `sampler-redesign`:**

- `plans/sampler-redesign-plan.md`: the plan, with the Goals and the rulings.
- `plans/sampler-redesign/`:
  - `checkpoint.md`: this file;
  - `sampling-draft.md`: the draft procedure, the well draws (R3) and the samples and generation rules (R4, revised by R7);
  - `fluids-draft.md`: the table fluids (R5);
  - `datasets-draft.md`: the dataset design (R7);
  - `spec-changes.md`: the proposed edits to decided specs;
  - `decisions.md`: the decision log, D-1 to D-103;
  - `r6-measurements.md`: the prototype's pilots and measurements;
  - `sources-reservoir-fluids.md`, `sources-wells-production.md`, `sources-geometry.md`: the public sources;
  - `sources-condensate-tables.md`: how other tools represent condensates (K).
- `specs/features/022-gas-rate-inflow.md`: the feature spec.
- `plans/evidence/`, each with its row in the README there: `erosional_capacity.py` and `pilot_by_well_type.py` (the evidence of 2026-11-05 for D1, D2 and F1), `solution_gas_drive.py` and `drain_friction.py` (the physics of 2026-11-06 for C1, B1 and D-21), `stratified_flow_limit.py` (D-10), `condensate_rv_form.py` (the condensate research), `sampler_draft_draws.py` and `sampler_draft_samples.py` (the draws, without solving), `sampler_prototype.py` and `sampler_prototype_report.py` (the prototype and its measurements), `gas_inflow_shapes.py` (feature 022's measurements), `table_fluids.py` and `pvt_table_eos.py` (the table fluids), `validity_ranges.py`.

**The figures** (R6, Figures; kept outside the repo with the pilots' tables in the datasets' schema, and shown on the report page):

| File | Shows |
|---|---|
| `sol2_bias_cdf.png`, `nsol2_bias_cdf.png` | The inputs' CDFs over all draws and over the kept samples, with the KS $D$ (criterion 4) |
| `sol2_redraws_by_bin.png`, `nsol2_redraws_by_bin.png` | Attempts without an operating point by choke, lift gas and $p_r/p_i$ |
| `sol2_regimes.png`, `nsol2_regimes.png` | The five regimes at the bottom and the top |
| `sol2_features_vs_sol1.png`, `nsol2_features_vs_sol1.png` | Rates, pressures and temperatures next to `sol-1`'s |
| `sol2_eia_brackets.png`, `nsol2_eia_brackets.png` | Offshore oil wells by rate bracket next to EIA's (D1) |
| `nsol2_histories.png` | Six `nsol-2` wells' oil equivalent over their life, with their tests |

The draft's runs, before D-100 to D-103, have figures of the same names in their own folder.

**Stale text in the drafts.** The list that stood here was fixed in a consistency pass on 2026-11-02, changing no decision and no measured number:

- *Fixed:* `fluids-draft.md`'s entry numbers from before the log was merged (now D-63, D-65, D-66 and D-67, each checked against the entry's title).
- *Fixed:* `datasets-draft.md`: the `inflow` code `forchheimer` with the columns `forchheimer_a` and `forchheimer_b` in the model's units and `non_darcy_D` as a descriptor; `lift_reason`'s `no_natural_flow`; the table fluids' labels (`fluid_source`, `saturation_kind`, `saturation_rule`), codes and parameter columns; the `pvt` section on D-67's one table set per well; the Solver row now notes H1.
- *Fixed:* `sampling-draft.md`: the Labels table after the geometry ruling and D-101; the tail at 0.20 drawn (D-103); the lifted shares by GOR quartile; Geometry now says its ranges are R3's, with the surveys' medians next to them (B1).
- *Fixed:* `r6-measurements.md` says which wells each overpressure share counts: 26.0% of the kept wells, core and tail, and 22.5% of the kept core wells, both at the initial pressure.
- *Noted, not changed:* D-66 and D-67 count 2500 table wells per dataset, against about 3200 at R6's mix; the scaled cost is in Section 8 and `fluids-draft.md`.
