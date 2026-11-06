# Sampler redesign — draft procedure: the well draws

*Step R3 of `plans/sampler-redesign-plan.md`, 2026-10-30. A draft for Bjarne's checkpoint (R8), in the form of `specs/sampling.md`, without SMP IDs. Plans are drafts, not specs. The samples within a well are R4's, in the second part of this file (The samples within a well); the dataset design is R7's.*

This file proposes how the v2 sampler draws one well in the `v2.0.0` configuration: an ordered sequence of seeded draws, each conditioned only on earlier ones. It keeps three things apart:

- **Decided:** the Goals of the plan (Bjarne, 2026-10-30) and his ruling of the same day on horizontal drains: a well with a horizontal drain is modelled from the heel up, its bottom section at most about 75° from vertical (R3 took 75°), and its inflow stands for the drain (D-10).
- **Proposed:** every distribution, parameter and rule below. Each rests on a working assumption in `decisions.md`: R3's D-1 to D-36, R4's D-37 to D-56, D-57 (which applies the ruling on D-9), R5's D-60 to D-69 and R7's D-80 to D-85, which Bjarne rules on at the checkpoint, or R6's revisions D-100 to D-103, made after the prototype's measurements (`r6-measurements.md`).
- **Ruled since:** the geometry as an even mix of four classes, a quarter each (Bjarne, 2026-11-02, D-9; Geometry); and 200 operating points per well, a failed draw redrawn and logged (Bjarne, 2026-11-02, D-52; The samples within a well).

Sources are cited as [RF §n] for `sources-reservoir-fluids.md`, [WP §n] for `sources-wells-production.md` and [GEO §n] for `sources-geometry.md`, with the primary reference and its page, table or figure. The draws were checked without solving by `plans/evidence/sampler_draft_draws.py` on 2000 wells, seed 2026; its numbers are quoted as *draws*, for the 1690 core wells unless said otherwise (Checks). They are R3's, at a tail share of 0.15 and the permeability before D-100, except where a section says *R6's draws*: the script now draws the revised draft (1591 core wells at a tail share of 0.20). R6 solved the draft in a prototype; its measurements are in `r6-measurements.md`.

## Conventions

- **Units** are the model's: m, bar (absolute), K, kg/s, kg/m³, Sm³/Sm³ at standard conditions (PVT-GAS-2), J/(kg K), W/(m² K). Rates of liquid are Sm³/d of stock-tank liquid, of gas Sm³/d.
- **Distributions.** $U(a, b)$ uniform; $\text{LogU}(a, b)$ log-uniform; $N_{[a,b]}(\mu, \sigma)$ normal truncated to $[a, b]$; $\text{LogN}_ {[a,b]}(m, s)$ log-normal with median $m$ and log standard deviation $s$, truncated to $[a, b]$; $\text{Cat}$ categorical. A truncated draw is made by the inverse CDF of one uniform draw, so it never loops.
- **Outside a range**, each draw says which of three it does:
  - *truncated:* the distribution is restricted to the bound;
  - *redrawn:* the block is drawn again from its own generator, and the redraw is counted and logged;
  - *accepted:* the value lies outside a correlation's published range and the model extrapolates (`specs/model/validity.md`, Rules).
- **Core and tail.** Each draw gives its core form and, where it differs, its tail form (Coverage tail). The tail flattens a draw's shape to a (log-)uniform over its bounds; it keeps the bounds and the links to earlier draws.
- **Depths.** $z$ is the true vertical depth of the reservoir below the mudline: the seabed offshore, the ground onshore. $L$ is the model's length, the true vertical depth of the bottom of the modelled flow path below the wellhead (GEO-3): the mid-perforation, or the heel of a well with a horizontal drain. A platform's wellhead is taken at sea level, a subsea well's at the seabed.
- **Seeding** (D-7). Each block of draws below has its own generator, `rng_for(seed, k, 'v2', block)` for well $k$ (SMP-31's function), so that a change to one block's distribution moves no other block's draws. The `v1.0.0` configuration keeps v1's draws (SMP-1 to SMP-17) unchanged.
- **The model's own correlations** are used inside the draws where they link two inputs: Vazquez–Beggs's $R_{so}$ (PVT-OIL-6, with PVT-OIL-5's separator correction), $B_o$ (PVT-OIL-8), Beggs–Robinson's viscosity (PVT-OIL-10, PVT-OIL-11), Dranchuk–Abou-Kassem's $Z$ (PVT-GAS-11) and Lee–Gonzalez–Eakin's gas viscosity (PVT-GAS-7), all evaluated on floats before the well is built. The well's bubble point then agrees with the model's.

## The order of the draws

```
labels ─┬─ coverage tail
        ├─ well type ───────────────┬──────────────────────────────┐
        ├─ fluid class ─────────────┤                              │
        └─ drive, life stage ───────┤                              │
depth ─── z (fluid class); water depth d_w (well type); L          │
geometry ── class and family (D-9), survey to TVD L                │
water ───── salinity, rho_w                                        │
pressure ── p_i (z, d_w, rho_w)                                    │
temperature ─ T_s (well type, d_w); G (d_w, L/z); T_r              │
separator ── p_sep, T_sep (well type); p_s,nom (stage, d_w) ───────┘
fluid ───── API, gas gravity (class); saturation pressure (p_i); initial GOR (VB at p_b, T_r, separator)
stage ───── p_r (drive, stage, p_i); water; producing GOR (p_r against p_b)
inflow ──── k, h, skin, drainage (well type, geometry); J or Forchheimer a, b (fluid at p_r, T_r)
completion ─ tubing D (design rate at the initial state); roughness (stage)
choke ───── bean, discharge coefficient (D); profile
lift ────── assignment (class, stage, static check); valve depth, T_lg, capacity (pressures, rates)
heat ────── h (D, stage); c_po, c_pw, c_pg (API, salinity, gas gravity, p, T)
```

| Block | Draws | Depends on |
|---|---|---|
| labels | coverage tail, well type, fluid class, drive, life stage | – |
| depth | $z$, $d_w$, $L$ | fluid class, well type |
| geometry | class, family, survey | $L$ (the classes' shares are the same for every well type, D-9) |
| water | salinity, $\rho_w$ | – |
| pressure | $p_i$ | $z$, $d_w$, $\rho_w$ |
| temperature | $T_s$, $G$, $T_r$ | well type, $d_w$, $z$, $L$ |
| separator | $p_\text{sep}$, $T_\text{sep}$, $p_{s,\text{nom}}$ | well type, life stage, $d_w$ |
| fluid | API, $\gamma_g$, $p_\text{sat}$, $R_{si}$ | fluid class, $p_i$, $T_r$, $z$, separator |
| stage | $p_r$, water–liquid ratio, $R_{go}$ | drive, life stage, $p_i$, $p_\text{sat}$, fluid |
| inflow | $k$, $h_n$, $s$, drainage, $J$ or $(a, b)$ | well type, geometry, fluid at $(p_r, T_r)$ |
| completion | $D$, $\varepsilon$ | design rate, fluid, separator, life stage |
| choke | $K_c$, profile | $D$ |
| lift | lift label, injection depth, $T_{lg}$, $w_{lg,\max}$ | class, stage, pressures, rates, survey |
| heat | $h$, $c_{po}$, $c_{pw}$, $c_{pg}$ | $D$, stage, API, salinity, $\gamma_g$, $p$, $T$ |

The order differs from the plan's likely order in two places, both for consistency: the formation water's salinity comes before the pressure, because the same brine sets the hydrostatic gradient [RF §9]; and the separator comes before the fluid, because Vazquez–Beggs's $R_{so}$ depends on the separator conditions (PVT-OIL-5).

## Labels

Every categorical choice is stored with the well. All are drawn or set before the well is built, with two exceptions that follow a solve: the lift of a `sol-2` oil well that cannot flow naturally (`lift_reason = no_natural_flow`, D-101), and the start of lift in `nsol-2` (D-49, rule 2). The table fluids' labels are in `fluids-draft.md`, and the codes of every label in `datasets-draft.md` (Section 4).

| Label | Values | Set by |
|---|---|---|
| `coverage_tail` | `core`, `tail` | Coverage tail |
| `well_type` | `platform` (offshore, dry tree), `subsea`, `onshore` | Well type |
| `fluid_type` | `black_oil`, `volatile_oil`, `gas_condensate`, `gas` | Fluid class |
| `fluid_class` | the four, with `gas` split into `wet_gas` and `dry_gas` | Fluid class |
| `class_rule` | `mccain_1990` | Fluid class: the rule that set the class, so that a user can reclassify |
| `drive` | `depletion`, `pressure_support` | Drive and life stage |
| `life_stage` | `initial`, `mid_life`, `late` | Drive and life stage |
| `life_state` | `initial`, `mid_life`, `depleted` (late, depletion), `watered_out` (late, pressure support) | derived |
| `pressure_class` | `under`, `normal`, `over` ($\lambda \ge 0.25$, about 0.6 psi/ft) | Initial reservoir pressure |
| `saturation` | `undersaturated`, `saturated`, `single_phase_gas` | Reservoir state at the life stage |
| `geometry_class` | `vertical`, `slightly_deviated`, `deviated`, `highly_deviated` | Geometry: the ruled class (D-9) |
| `geometry_family` | `vertical`, `build_and_hold`, `s_shaped`, `high_angle` | Geometry: the family that drew the survey |
| `pvt_model` | `black_oil`, `table` | Fluid class (Mapping) |
| `inflow` | `vogel`, `productivity_index`, `forchheimer` (R3 drafted `gas_rate`; feature 022, D-60) | Inflow |
| `tubing` | the nominal size, `2-7/8` to `9-5/8` | Tubing |
| `choke_model` | `simpson` | Choke |
| `choke_profile` | `linear`, `sigmoid`, `convex`, `concave` | Choke |
| `lift` | `natural`, `gas_lift` | Gas lift: assignment |
| `lift_reason` | `drawn`, `needs_lift`, `no_natural_flow` (`sol-2`, D-101) | Gas lift: assignment |

## Well draws

### Coverage tail

- **Draw.** `coverage_tail` is `tail` with probability 0.20, else `core` (D-6, revised by D-103: R3 drew 0.15, and R6's pilot of the draft kept 60% of the tail wells against 96% of the core wells, so 0.15 drawn gave 9.25% of the accepted wells; 0.20 gives about 15%).
- **Effect.** In a tail well every draw below takes its tail form: categorical shares become equal (well type, fluid type, drive, life stage, geometry), and continuous draws become (log-)uniform over the draw's bounds. The links stay: pressure still follows depth and brine, the bubble point still follows the fluid, tubing still follows the rate. So a tail well is less likely, not implausible.
- **Sources.** The Goals set 10% to 20% as the starting point; 15% is its middle, which D-103 aims at among the accepted wells by drawing 0.20. No public source sets a share; R6 measured what the tail adds to the coverage (Where realism and coverage conflict; `r6-measurements.md`, Coverage).
- **Validity.** The tail keeps every bound of the core, so it leaves no range the core does not.
- **Draws.** 15.5% of the 2000 wells (R6's draws at 0.20: 20.4%). R6's `sol-2` pilot: 19.3% of the drawn wells and 14.5% of the accepted.

### Well type

- **Draw.** $\text{Cat}$(platform 0.55, subsea 0.35, onshore 0.10); tail: equal shares (D-5).
- **Depends on.** Nothing.
- **Sources.** [WP §1]: in deep water in the Gulf of Mexico, 84% of producing wells have dry trees (997 against 191; Liu, Wang and Kaiser 2014); on the Norwegian shelf a large minority of the development production wellbores are subsea [own analysis of Sodir's FactMaps]. The shelf of the Gulf of Mexico is almost all platform wells. Counted by wells the world is onshore (EIA 2025, p. 3: 918,481 producing wells in the US alone), so the onshore share is the Goals' offshore weighting, not a statistic.
- **Validity.** None; the well type sets surface conditions only.
- **Maps to.** No model input; it conditions the water depth, the surface temperature, the separator, the permeability and the economic floor.

### Fluid class

- **Draw.** $\text{Cat}$(black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20); tail: 0.25 each (D-4). A gas well is `dry_gas` with probability 0.07, else `wet_gas` (tail: 0.25).
- **Classes** by McCain (1990), pp. 151–158 (D-3): black oil with an initial producing gas–oil ratio up to 356 Sm³/Sm³ (2000 scf/STB) and below 45 °API; volatile oil from 356 to 588 Sm³/Sm³ (3300 scf/STB) and at least 40 °API; gas condensate from 588 to 8900 Sm³/Sm³ (50,000 scf/STB); wet gas above 8900 Sm³/Sm³; dry gas with practically no surface liquid, here a condensate yield below 2.8 Sm³ per 10⁶ Sm³ (0.5 STB/MMscf).
- **Sources.** [RF §5, §6]: the shares by wells of the Gulf of Mexico [own analysis of BOEM's sands atlas] and of Norwegian production wellbores [own analysis of Sodir's FactMaps]: the Gulf mostly black oil and wet gas with few volatile oils, its deep water mostly black oil, the Norwegian shelf mostly oil. The Goals' floor of about 10% per type sets the volatile-oil share and, with the Norwegian shelf, the condensate share. The dry-gas probability 0.07 lies between the Gulf's dry-gas shares by wells and by sands.
- **Validity.** The classes are chosen so that the black oil (PVT-OIL-4) carries only McCain's black oils and the wet and dry gases, whose liquid is a light stock-tank condensate; volatile oils and condensates go to tables (Mapping).
- **Draws.** Black oil 57.0%, wet gas 17.5%, condensate 14.4%, volatile oil 10.4%, dry gas 0.8% of core wells.

### Drive and life stage

- **Draw.** `drive` is `pressure_support` (natural water drive, or injection) with probability 0.7 for oil classes and 0.4 for gas classes, else `depletion`; tail 0.5. `life_stage` is $\text{Cat}$(initial 0.25, mid-life 0.40, late 0.35); tail equal (D-18).
- **Sources.** [RF §3]: most of the Gulf's undersaturated oil sands, by wells, have full or partial water drive, and few depletion or solution-gas drive; most of its gas sands have water drive [own analysis of BOEM's sands atlas]. The Norwegian shelf supports most fields' pressure with water or gas injectors [own analysis of Sodir's FactMaps]. The stage shares follow fields' lives of 10 to 30 years with a plateau in the first third [RF §11; UKERC 2009, Table 4.4; own analysis of Sodir's FactMaps]; no public count of wells by stage is in hand.
- **Maps to.** No input directly; it sets the reservoir's state (Reservoir state at the life stage), the separator's stage, the roughness and the heat-transfer coefficient.

### Depth below the mudline

- **Draw.** $z \sim \text{LogN}_ {[600, 6500]}(m, s)$ m by fluid class (D-8):

  | Class | $m$ (m) | $s$ |
  |---|---|---|
  | black oil | 2674 | 0.317 |
  | volatile oil | 3159 | 0.315 |
  | gas condensate | 3424 | 0.278 |
  | wet gas | 2419 | 0.422 |
  | dry gas | 1141 | 0.586 |

  Tail: $\text{LogU}(600, 7000)$.
- **Sources.** [RF §1]: $m$ and $s$ are an own fit to the P5, P50 and P95 of the depth below sea level of BOEM's sands, by wells and fluid class [own analysis of BOEM's sands atlas]. Most of the Gulf's wells lie in shallow water [own analysis of BOEM's sands atlas], so its depths below sea level stand for depths below the mudline. The final TVDs of Norwegian production wellbores agree within 10% for oil [own analysis of Sodir's FactMaps].
- **Validity.** Hasan et al.'s measured depths run from 498 to 9342 m (`validity.md`); 600 m is above the lower end. Truncated.
- **Draws.** P5 / P50 / P95 = 1530 / 2768 / 4831 m.

### Water depth and the well's length

- **Draw** (D-8):
  - platform: with probability 0.85 a shelf platform, $d_w \sim \text{LogN}_ {[15, 400]}(90, 0.6)$ m, else a deep-water dry tree, $\text{LogU}(400, 2000)$ m; tail $\text{LogU}(15, 2000)$;
  - subsea: $d_w \sim \text{LogN}_ {[80, 3000]}(350, 0.8)$ m; tail $\text{LogU}(50, 3000)$;
  - onshore: $d_w = 0$.
- **Length.** $L = z + d_w$ for a platform (the riser through the water is part of the flow path), $L = z$ for subsea and onshore wells.
- **Sources.** [WP §2]: the water depths of Norwegian platform and subsea wellbores by fluid [own analysis of Sodir's FactMaps] and of the Gulf's completed development boreholes [own analysis of BOEM's borehole file]; the distributions above are own choices that span them. Deep-water dry trees (TLPs, spars) reach about 2000 m in the Gulf (Liu, Wang and Kaiser 2014).
- **Validity.** No correlation; THM-4's linear ambient profile cannot follow a deep water column (Surface temperature).
- **Draws.** Platform P5 / P50 / P95 = 34 / 99 / 1174 m; subsea 120 / 373 / 1409 m; $L$ 1582 / 2885 / 5083 m.

### Geometry

*Bjarne ruled the classes' shares and the shapes of the highly deviated class on 2026-11-02, after the geometry evidence (`sources-geometry.md`): an even mix (D-9). He ruled the limit of about 75° and the heel approach on 2026-10-30 (D-10). The families' parameters are proposed here: R3's ranges, set before the surveys were read (checkpoint item B1).*

- **Families** (D-9). Each is a survey of stations $(\text{MD}, \text{TVD})$ from the wellhead to the bottom at TVD $L$, with circular build and drop sections (one station per degree):

  | Family | Shape | Parameters (core) |
  |---|---|---|
  | `vertical` | straight down | – |
  | `build_and_hold` | vertical to the kickoff $K$, build at $b$ to $\theta_h$, hold to $L$ | $\theta_h \sim N_{[5,65]}(35°, 13°)$; $b \sim U(1.5, 3)$ °/30 m; $K \sim U(0.1L, K_\text{max})$, $K_\text{max} = \min(0.7L,\ L - 50\ \text{m} - \Delta\text{TVD}_ \text{build})$ |
  | `s_shaped` | build to $\theta_h$, hold, drop at $d$ to $\theta_e$, then straight to $L$ | $\theta_h \sim U(15°, 45°)$; $b$ as above; $d \sim U(1, 2.5)$ °/30 m; $\theta_e \sim U(0°, 10°)$ over the last $U(100, 400)$ m of TVD; $K$ up to $0.6L$ |
  | `high_angle` | build to $\theta_h$, hold, build at $b_2$ to $\theta_\text{heel}$, a short tangent to the heel at $L$ | $\theta_h \sim U(35°, 65°)$; $\theta_\text{heel} \sim U(\max(60°, \theta_h), 75°)$; $b_2 \sim U(2, 4)$ °/30 m; tangent $U(50, 300)$ m of MD; $K \sim U(0.05L, 0.4L)$ within the fit |

  The classes below set $\theta_h$ for `build_and_hold` and $\theta_\text{heel}$ for `high_angle`, in place of the distributions in the table.
- **Fit.** If a shape does not fit $L$, the build rate is raised to at most 5 °/30 m, then $\theta_h$ is lowered in steps of 10%; an S or high-angle shape that still does not fit, or a well whose measured depth exceeds 9342 m, is redrawn (R3's draws: 0.8% of the wells redrawn at least once, 3 of 2000 fall back to vertical after 20 tries; R6's draws with the ruled classes: 66 redraws in 2000 wells, none falls back).
- **Classes and shares** (Bjarne, 2026-11-02; D-9). An even mix for every well type and fluid class, in core and tail alike:

  | Class (label) | Share | Family and parameters |
  |---|---|---|
  | `vertical` | 0.25 | `vertical` |
  | `slightly_deviated` | 0.25 | `build_and_hold`, $\theta_h \sim U(5°, 20°)$ |
  | `deviated` | 0.25 | two thirds `build_and_hold` with $\theta_h \sim U(20°, 60°)$; one third `s_shaped` as in the table (D-57) |
  | `highly_deviated` | 0.25 | half `high_angle`, a drain modelled to its heel with $\theta_\text{heel} \sim U(60°, 75°)$; half a high-angle tangent, `build_and_hold` with $\theta_h \sim U(60°, 75°)$, capped at the limit |

  The class and the family are labels. The evidence the ruling weighed [`sources-geometry.md`; own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps]: public offshore surveys have few vertical wells; most of the Gulf of Mexico's producing boreholes are deviated, with high-angle wells and drains a small share; drains are a minority of the highly deviated wells there, and a majority in the North Sea and in wells drilled since 2000. The even mix is a design choice for coverage, not a statistic.
- **The families' parameters against the surveys.** The build, kickoff, drop and landing ranges in the table above are R3's, drawn up before the surveys were read; they do not follow them. The surveys' distributions of the kickoff, the build, a J well's hold, an S well's drop and bottom inclination, and the drains' tangent and landing build are in [GEO §2; own analysis of BOEM's directional surveys and NLOG's deviation surveys]: their main builds are gentler and their drops slower than R3's ranges. Adopting the surveys' values is checkpoint item B1; the parameters are unchanged until Bjarne rules.
- **Inclination limit** (D-10). Every cell is at most $\theta_\text{max} = 75°$ from vertical. Taitel and Dukler's stratified–non-stratified transition, scanned over oil and water at 10 to 200 bar in 62 to 157 mm tubing [WP §3], leaves stratified upward flow only below $v_{SL}$ = 0.006 m/s at 10° above horizontal (80° from vertical), below 0.004 m/s in one case at 15° (75°), and nowhere above 0.003 m/s at 20° (70°). Experiments agree: in a 9.72 cm pipe at 2.7 to 14.8 bar, stratified flow had disappeared at +2° (Wilkens, Gopal and Jepson), and Mukherjee and Brill's map has no stratified region in upward flow (Brill and Mukherjee 1999, p. 44). 75° keeps the margin of 70° for liquid rates above about 7 m³/d in 157 mm tubing, and admits the heel sections of the Gulf's horizontal wells, whose equivalent tangent lies below 75° at the median [WP §3; own analysis of BOEM's borehole file].
- **Heel approach** (Bjarne's ruling). A `high_angle` well stands for a well with a horizontal drain: the model's bottom is the heel, at the reservoir's TVD $L$, and the drain's length enters the inflow (Productivity). The drain itself has no cells, so no cell is horizontal.
- **Validity.** Hasan et al.'s five regimes are fitted to vertical and deviated wells with no range of inclination (`validity.md`, note 9); none of the draws reaches stratified flow's inclinations. Measured depth stays within Hasan et al.'s 9342 m (redrawn).
- **Maps to.** `WellGeometry.from_survey(md, tvd, n_cells=100, D)` (GEO-3, GEO-4), which `configurations.check` accepts in `v2.0.0` for any survey.
- **Draws** (R6's, with the ruled classes; P5 / P50 / P95). Vertical 23.1%, slightly deviated 26.1%, deviated 24.9%, highly deviated 26.0% of core wells. MD/TVD 1.00 / 1.01 / 1.04 (slightly deviated), 1.02 / 1.09 / 1.39 (deviated), 1.20 / 1.47 / 1.99 (highly deviated); the largest cell inclination 5.7° / 12° / 19°, 20° / 36° / 58° and 61° / 67° / 74°; drains' heels 61° / 67° / 74°; no cell above 75°, and no well falls back to vertical. R3's placeholder shares gave MD/TVD 1.02 / 1.15 / 1.61 over the deviated families.

### Formation water

- **Draw.** Salinity, the mass fraction of dissolved solids, $w_s \sim \text{LogN}_ {[0.01, 0.25]}(0.08, 0.6)$; tail $\text{LogU}(0.005, 0.25)$ (D-11). The water's density at standard conditions is

  $$\rho_w = 999.1 + 692\thinspace w_s + 311\thinspace w_s^2\ \text{kg/m}^3,$$

  a quadratic through McCain's (1990) 999 kg/m³ at no solids, 1047.6 kg/m³ at 6.8 wt% (Example 16-6, p. 449) and about 1200 kg/m³ at 26 wt% (Fig. 16-8), from [RF §9].
- **Sources.** [RF §9]: formation waters run from 10,000 to about 300,000 ppm, usually below 80% of the NaCl limit (Whitson and Brulé 2000, §9.2, Eq. 9.1); seawater is about 35,000 ppm (McCain 1990, p. 438). No public distribution over offshore reservoirs is in hand. The median, 80,000 ppm, gives 1056 kg/m³ and 0.104 bar/m, near the Gulf's normal gradient [own analysis of BOEM's sands atlas].
- **Validity.** PVT-WAT-1 takes any constant density. 0.25 stays below 80% of NaCl's limit at reservoir temperatures (Whitson and Brulé 2000, Eq. 9.1). Truncated.
- **Maps to.** `FluidModel(rho_w=ρ_w)`; the salinity also sets the hydrostatic pressure and $c_{pw}$.
- **Draws.** 0.029 / 0.079 / 0.19; $\rho_w$ 1020 / 1056 / 1141 kg/m³.

### Initial reservoir pressure

- **Draw** (D-12). With the hydrostatic and lithostatic pressures

  $$p_h = p_\text{atm} + \frac{g\thinspace(\rho_\text{sea} d_w + \rho_w z)}{10^5}, \qquad p_\text{lith} = p_\text{atm} + \frac{g\rho_\text{sea} d_w}{10^5} + 0.226\thinspace z,$$

  in bar with $\rho_\text{sea}$ = 1025 kg/m³ and 0.226 bar/m (1 psi/ft) below the mudline, the initial pressure is $p_i = p_h + \lambda\thinspace(p_\text{lith} - p_h)$ with the overpressure ratio

  $$\text{logit}\thinspace\lambda \sim N_{[-12,\ \text{logit}\thinspace\lambda_\text{max}]}\big(\text{logit}\thinspace\lambda_{50}(z),\ 0.9\big), \qquad \lambda_{50}(z) = 0.02 + \frac{0.45}{1 + e^{-(z/1000\ \text{m} - 3.8)/0.9}}.$$

  With probability 0.04 (tail 0.10) the reservoir is initially underpressured instead: $p_i = p_h\thinspace U(0.8, 1.0)$. Tail: $\lambda \sim U(0, \min(0.8, \lambda_\text{max}))$.
- **Bounds.** $\lambda_\text{max} = \min\big(0.85,\ (950 - p_h)/(p_\text{lith} - p_h)\big)$: below a fracture gradient near lithostatic, and $p_i \le 950$ bar (D-35). Truncated.
- **Sources.** [RF §2]: BOEM's sands by wells give the median excess over the Gulf's normal gradient, and the share above 0.6 psi/ft, in depth bins from 0–1 to 5–7 km [own analysis of BOEM's sands atlas]; both rise with depth. Converted to $\lambda$ with the lithostatic gradient above, $\lambda_{50}$ and the spread 0.9 are an own fit to these bins. Globally 16.2% of the reservoirs of large fields lie above SG 1.3 and 6.0% below SG 0.8 (Nadeau et al. 2023, Fig. 20, counted by reservoirs, D-2); few of the Gulf's wells lie below its normal gradient [own analysis of BOEM's sands atlas]. The lithostatic gradient of 1 psi/ft is a textbook rule with no source in hand (Resource list).
- **Validity.** Hasan et al.'s bottomhole pressures run to 949 bar; 950 bar is their end. Beggs–Robinson's viscosity is fitted to 363 bar and Lee–Gonzalez–Eakin to 552 bar: accepted, the model extrapolates (`validity.md`).
- **Maps to.** No input directly; $p_r$ follows from it (Reservoir state).
- **Draws.** $p_i$ 191 / 363 / 756 bar; gradient below sea level 0.103 / 0.118 / 0.159 bar/m; 20.4% above 0.6 psi/ft; medians by class 0.117 (black oil), 0.122 (volatile oil), 0.125 (condensate), 0.117 (wet gas) and 0.109 bar/m (dry gas), each within 0.008 bar/m of BOEM's by wells. Compare the percentiles and the share above 0.6 psi/ft with [own analysis of BOEM's sands atlas]. The bound on 950 bar cuts into the central 90% of the draw in 6.8% of the wells.

### Surface temperature

- **Draw** (D-14). $T_s$ is the top of THM-4's ambient profile.
  - offshore (platform and subsea): the seabed temperature at the water depth, $T_s = T_\text{deep} + (T_\text{shelf} - T_\text{deep})\thinspace e^{-d_w/400\ \text{m}}$, with $T_\text{deep} \sim U(272, 278)$ K and $T_\text{shelf} \sim U(279, 297)$ K;
  - onshore: the mean annual surface temperature, $U(268, 303)$ K; tail $U(258, 308)$ K.
- **Sources.** [WP §2]: 277.6 K at 1219 m with 297 K at the surface (Hasan and Kabir 2018, Fig. 7.17); 274 to 277 K below 1500 m (JPT, weak); 272 K at 850 to 1100 m in the Norwegian Sea (trade press, weak); 280 to 283 K at 540 m (MacDonald et al. 2005). The profile gives 275.6 K at 1219 m at the means of $T_\text{deep}$ and $T_\text{shelf}$. Onshore, R2's lead of 263 to 303 K for producing regions, narrowed at the cold end; no distribution is in hand (Resource list: NOAA World Ocean Atlas, climate normals).
- **Validity.** No correlation. For a platform the linear profile from the seabed temperature at the wellhead misses the warm surface water; in deep water it puts the riser's ambient too warm, since the water column is near $T_\text{deep}$ throughout. This is THM-4's limit, which a profile with a water column (v2.x) would remove.
- **Maps to.** `BoundaryConditions(T_s=…)`, constant for the well.
- **Draws.** Offshore 274.9 / 281.8 / 291.5 K.

### Geothermal gradient and reservoir temperature

- **Draw** (D-13). $G \sim \text{LogN}_ {[G_\text{lo}, G_\text{hi}]}(G_{50}, 0.22)$ K/km below the mudline, with $G_{50} = 20 + 10\thinspace e^{-\max(d_w - 100\ \text{m},\ 0)/400\ \text{m}}$, which is 30 K/km on shelves and onshore and falls to 20 K/km in deep water; tail $U(G_\text{lo}, G_\text{hi})$. Then $T_r = T_s + G z$.
- **Bounds.** $G_\text{lo} = \max(12,\ 10\thinspace L/z)$ keeps the model's gradient $(T_r - T_s)/L$ at 10 K/km or more, well above the CasADi backend's limit near 5 K/km (`validity.md`); $G_\text{hi} = \min\big(55,\ (450 - T_s)/z\big)$ keeps $T_r \le 450$ K (D-35). Truncated.
- **Sources.** [RF §4]: the gradient $(T_i - 20\ °\text{C})/z$ of BOEM's sands on the Gulf's shelf and, below the mudline, in its deep water [own analysis of BOEM's sands atlas], and of the Norwegian shelf from uncorrected logging temperatures [own analysis of Sodir's FactMaps]: lowest in deep water, highest on the Norwegian shelf; 14 to 36 K/km in the Gulf's Cenozoic reservoirs (Nadeau et al. 2023, Fig. 7A). One log-normal spans the shelf and the North Sea; R2's mixture of three basins (D-13) would add modes that no per-well statistic supports.
- **Validity.** Valkó and McCain's oils run from 289 to 445 K (Table 2.1); Beggs–Robinson to 419 K and Lee–Gonzalez–Eakin to 444 K: accepted above.
- **Maps to.** `BoundaryConditions(T_r=…)`.
- **Draws.** $G$ 17.8 / 27.2 / 39.4 K/km; over the modelled well 15.9 / 26.2 / 38.6 K/km; $T_r$ 316 / 358 / 424 K. Medians by class: 354 K (black oil), 367 K (volatile oil), 378 K (condensate), 354 K (wet gas), 324 K (dry gas), each within 6 K of BOEM's by wells [own analysis of BOEM's sands atlas]. 17.1% lie above 395 K (122 °C), against 12.9% of the reservoirs of large fields (Nadeau et al. 2023, Table 1). $G_\text{lo}$ binds in 3.1% of the wells, $G_\text{hi}$ in 13.2% (deep, hot wells).

### Separator conditions and the nominal downstream pressure

- **Draw** (D-15). The host's first-stage separator at the start of the well's life:
  - platform and subsea: $p_\text{sep} \sim \text{LogN}_ {[5, 85]}(35, 0.45)$ bar; $T_\text{sep} \sim U(300, 340)$ K on a platform, $U(290, 320)$ K for subsea (the flow cools in the flowline);
  - onshore: $p_\text{sep} \sim \text{LogN}_ {[2, 70]}(15, 0.6)$ bar, $T_\text{sep} \sim U(285, 320)$ K.

  The nominal pressure downstream of the choke, $p_{s,\text{nom}} = r_s\thinspace p_\text{sep} + \Delta p_\text{fl}$, truncated to [1.5, 120] bar, with $r_s$ = 1 at the initial stage, $U(0.5, 1)$ in mid-life and $U(0.1, 0.6)$ late (the well moved to a lower stage), and for subsea wells the flowline and riser, $\Delta p_\text{fl} = U(5, 30) + 0.02\thinspace d_w$ bar (a light multiphase riser column, about 200 kg/m³). Tail: $p_{s,\text{nom}} \sim \text{LogU}(1.5, 120) + \Delta p_\text{fl}$.
- **Sources.** [WP §7]: first stages at 41 to 83 bar (Guo, Lyons and Ghalambor 2007, §10.2.5, p. 10/124), below about 52 bar in the North Sea (weak); a North Sea train at 65, 20 and 2 bara (weak); low-pressure vessels rated at 8.6 bar (Guo et al. 2007, Table 10.4); Oseberg B moved oil from 67 to 20 bar after 18 years (trade, weak). R2 proposed 2 to 80 bar for platform and onshore wells and 10 to 100 bar for subsea wells, which these draws reproduce.
- **Validity.** PVT-OIL-5's correction is defined for any separator. No correlation limits $p_s$.
- **Maps to.** `BlackOil(p_sep=…, T_sep=…)`, the fluid's separator for the gas gravity's correction (PVT-OIL-5): the GOR and gas gravity are taken as reported at the host's first stage. $p_{s,\text{nom}}$ is the well's nominal separator pressure, around which R4 draws each sample's `BoundaryConditions(p_s=…)`.
- **Draws.** $p_\text{sep}$ 12.6 / 32.7 / 66.0 bar; $p_{s,\text{nom}}$ 4.9 / 29.5 / 75.4 bar. Below 10 bar: 15.6% of core wells; 45% of late-life platform wells, 57% of onshore wells and none of the subsea wells, whose $p_s$ includes the flowline.

### Oil and gas at standard conditions

- **Draw** (D-16):

  | Class | API gravity | Gas gravity $\gamma_g$ (air = 1) | Tail |
  |---|---|---|---|
  | black oil | $N_{[16,45]}(33, 4.5)$ | $N_{[0.58,0.95]}(0.66, 0.06)$ | $U$ over the bounds |
  | volatile oil | $U(40, 50)$ | $N_{[0.62,1.0]}(0.75, 0.06)$ | $\gamma_g$: $U$ |
  | gas condensate | $U(42, 58)$ | $N_{[0.60,0.90]}(0.72, 0.05)$ | $\gamma_g$: $U$ |
  | wet and dry gas | $U(45, 58)$ | $N_{[0.571,0.75]}(0.62, 0.035)$ | $\gamma_g$: $U$ |

- **Sources.** [RF §7]: the API and gas gravities of the Gulf's oils by wells, few below 22 °API, and its gas sands' gas gravities [own analysis of BOEM's sands atlas]; the API and gas gravity of Norwegian drill-stem tests, all fluids [own analysis of Sodir's FactMaps]; Valkó and McCain's 1745 oils, 6.0 to 63.7 °API and separator gases of 0.555 to 1.685 (Table 2.1). Condensates' stock-tank liquids are 40 to 60 °API (McCain 1990, pp. 155–156; Whitson and Brulé 2000, §2.5). The example fluids of Whitson and Brulé (2000, Table 2.1) give 0.63 (black oil), 0.70 (volatile oil and condensate) and 0.61 (wet gas).
- **Validity.** The black oil refuses API gravities outside 16 to 58 (decision 2): all draws lie inside, so heavy oils below 16 °API are not drawn (D-35). Sutton's gas gravities start at 0.571 (`validity.md`, note 6): the gas classes are truncated there; no draw lies below. Truncated.
- **Maps to.** $\rho_o$ from the API gravity (PVT-OIL-2) and $\rho_{g,\text{sc}}$ from $\gamma_g$ (PVT-GAS-6), as `BlackOil(rho_o=…, rho_g=…)` or a table's stock-tank densities.
- **Draws.** Black oil 25.4 / 32.7 / 40.3 °API; $\gamma_g$ 0.59 / 0.67 / 0.76.

### Saturation pressure and the initial gas–oil ratio

- **Draw** (D-17). For oils and condensates, the saturation pressure (bubble or dew point) at $T_r$ is $p_\text{sat} = p_i\thinspace(1 - x)$: $x = 0$ (saturated at discovery) with probability 0.20 for black oils, 0.40 for volatile oils and 0.30 for condensates, else $x \sim \text{LogN}_ {[0.005, 0.85]}(0.10, 1.2)$ for black oils and $(0.05, 1.2)$ for the others; tail $x \sim U(0, 0.85)$. Then the initial gas–oil ratio:
  - **black oil:** $R_{si} = R_{so}(p_\text{sat}, T_r)$ by the model's Vazquez–Beggs (PVT-OIL-6) with the well's API, $\gamma_g$ and separator. If it exceeds McCain's 356 Sm³/Sm³, $p_\text{sat}$ is lowered to where $R_{so} = 356$ (draws: 3.9% of wells), so a deep black oil is more undersaturated;
  - **volatile oil:** $R_{si} \sim \text{LogU}(356, 588)$ Sm³/Sm³;
  - **gas condensate:** $R_{si} \sim \text{LogN}_ {[588, 8900]}\big(2000\thinspace e^{-0.3\thinspace(z/1000\ \text{m} - 3.4)},\ 0.6\big)$ Sm³/Sm³, richer with depth; tail $\text{LogU}(588, 8900)$;
  - **wet gas:** the condensate yield $\text{LogN}_ {[2.8, 112]}(30, 1.2)$ Sm³ per 10⁶ Sm³, $R_{si} = 10^6$/yield; tail $\text{LogU}(2.8, 112)$; **dry gas:** yield $\text{LogU}(0.5, 2.8)$.
- **Sources.** [RF §7]: Standing's bubble point from the Gulf's sands gives $p_i - p_b$ by wells, and the share of its oil wells in saturated sands; its $R_{si}$ by wells and their rank correlations with $p_i$, $T_i$ and depth; condensate yields rise with depth, and the yields of its gas sands, condensates included [own analysis of BOEM's sands atlas]. Norwegian condensate fields' producing GOR late in life [own analysis of Sodir's FactMaps]. The example condensate of Whitson and Brulé (2000, Table 2.1) has 971 Sm³/Sm³.
- **Validity.** Vazquez–Beggs's $R_{so}$ stays inside Beggs and Robinson's 3.6 to 369 Sm³/Sm³ for black oils. The gases' ratios lie far above the black oil's range: accepted (decision 4), since the liquid's share of the flow is small.
- **Maps to.** `FluidModel(gor=R_go)` at the life stage (next draw). The black oil's bubble point is then $p_\text{sat}$ at $T_r$, because $R_{si}$ is its own $R_{so}$ there (PVT-OIL-15); no `p_bubble` is passed. For a volatile oil or condensate, the table's saturation pressure at $T_r$ replaces this $p_\text{sat}$ (D-65; `fluids-draft.md`, The saturation pressure and the reservoir's state).
- **Draws.** Black oil $R_{si}$ 71 / 168 / 356 Sm³/Sm³; $p_i - p_b$ 0 / 26 / 209 bar. Rank correlations in Checks.

### Reservoir state at the life stage

- **Draw** (D-18, D-19). The pressure ratio $r_p = p_r/p_i$:

  | Stage | Depletion | Pressure support |
  |---|---|---|
  | initial | $U(0.97, 1.0)$ | $U(0.97, 1.0)$ |
  | mid-life | $U(0.55, 0.92)$ | $U(0.85, 1.0)$ |
  | late | oil $U(0.3, 0.6)$, gas $U(0.2, 0.55)$ (`depleted`) | $U(0.70, 1.0)$ (`watered_out`) |

  Tail: $U(0.15, 1.0)$. The water, as the water–liquid ratio at standard conditions for oils and the water–gas ratio WGR (Sm³ per 10⁶ Sm³ of gas) for gas classes:

  | Stage | Oils, depletion | Oils, support | Gas, depletion | Gas, support |
  |---|---|---|---|---|
  | initial | 0 with probability 0.5, else $U(0, 0.1)$ | same | $\text{LogU}(2, 20)$ | same |
  | mid-life | $U(0, 0.3)$ | $U(0.1, 0.6)$ | $\text{LogU}(5, 50)$ | $\text{LogU}(10, 150)$ |
  | late | $U(0.05, 0.4)$ | $U(0.6, 0.95)$ | $\text{LogU}(5, 80)$ | $\text{LogU}(50, 500)$ |

  Tail: oils $U(0, 0.98)$, gas $\text{LogU}(1, 1000)$. A gas well's water–liquid ratio is WGR/(WGR + yield). The producing GOR: oils $R_{go} = R_{si}\thinspace\min\big(10,\ 1 + k_g\max(0, 1 - p_r/p_\text{sat})\big)$ with $k_g \sim \text{LogU}(2, 15)$; condensates $R_{go} = R_{si}\big(1 + U(0.5, 3)\max(0, 1 - p_r/p_\text{sat})\big)$, as liquid drops out in the reservoir; wet and dry gas $R_{go} = R_{si}$.
- **Sources.** [RF §3, §9, §11]: the $p/Z$ of initial at which the Gulf's depletion-drive gas sands end, from their recovery factors [own analysis of BOEM's sands atlas]; a subsea oil well's average pressure fell from 276 to 159 bar under solution-gas drive, its producing GOR rose from 107 to a peak of 890 Sm³/Sm³ (8.3 times) and then fell (Golan and Whitson 1991, Example 4.11, Table E4.11a, p. 454); Norwegian oil fields' water cut by years on production and gas fields' WGR, by fields [own analysis of Sodir's FactMaps]. The linear rise of the GOR and its factors $k_g$ and the condensate's are own choices that bracket the example (D-19); no statistic is in hand.
- **Validity.** Hasan et al.'s water cuts run to 98% (`validity.md`): the core's oils stop at 0.95, the tail's at 0.98. Vogel's inflow is for solution-gas drive (note 10; Inflow). `FluidModel` takes a water–liquid ratio below 1, which every draw is.
- **Maps to.** `BoundaryConditions(p_r=p_r)` as the well's nominal reservoir pressure, around which R4 drifts each sample; `FluidModel(gor=R_go, wlr=…)`.
- **Draws.** $r_p$ 0.36 / 0.90 / 0.995; $p_r$ 115 / 297 / 658 bar; oils' water cut 0 / 0.25 / 0.87 (late, pressure support: median 0.77); gas WGR 3.0 / 19 / 222. Below hydrostatic: 46.7% of core wells, 30.1% by more than 10% of it; by state, 5% of initial wells, 77% of mid-life and 99% of late depletion wells, and 36% and 56% of mid-life and late wells with pressure support.

### Productivity

- **Draw** (D-20, D-21). Per well, independent of the fluid:
  - permeability $k$: offshore $\text{LogN}_ {[0.5, 3000]}(15, 1.5)$ mD, onshore $\text{LogN}_ {[0.05, 1000]}(3, 1.5)$ mD; tail $\text{LogU}$ over the bounds (D-100: R3 drew medians of 50 and 10 mD, with which R6's pilot put 35% of the core offshore oil wells above 1018 Sm³/d of oil equivalent, against 5.5% of the Gulf of Mexico's oil wells; now 23%);
  - net pay $h_n \sim \text{LogN}_ {[3, 100]}(15, 0.8)$ m; tail $\text{LogU}(3, 100)$;
  - skin $s \sim N_{[-3, 15]}(2, 3)$; tail $U(-3, 15)$;
  - drainage: $\ln(r_e/r_w) \sim U(7, 9)$, and $U(8.5, 9.5)$ for a high-angle well, with $r_w$ = 0.108 m (8.5 in hole);
  - the design drawdown $\delta_d \sim U(0.1, 0.4)$ of $p_r$, for the tubing and the lift (below).
- **Oil wells.** The radial productivity index at the life stage's fluid, $J_V = 2\pi k h_n/\big(\mu_o B_o(\ln(r_e/r_w) - 0.75 + s)\big)$ in Sm³/(d bar) of stock-tank liquid, with $\mu_o$ and $B_o$ from the model's black oil at $(p_r, T_r)$ and $R_{go}$. Then $J = f\thinspace J_V$ with the geometry's factor $f$:
  - `high_angle`: Joshi's horizontal-well index (Economides, Hill and Ehlig-Economides 1994, Eqs. 2-46 to 2-48) for a drain of $L_d \sim U(300, \min(1000, 1.6\thinspace r_e))$ m with anisotropy $I_\text{ani} \sim \text{LogU}(1, 3)$, over $J_V$, truncated at 6;
  - `build_and_hold`, `s_shaped`: $f = (\cos\theta_\text{end})^{-1/2}$ for the slant of the last section, a placeholder until Cinco-Ley's slant skin (Economides et al. 1994, §5-4) is read;
  - `vertical`: $f = 1$.
- **Gas and condensate wells.** Forchheimer's $p_r^2 - p_{wf}^2 = a\thinspace q + b\thinspace q^2$ (Economides et al. 1994, Eqs. 4-32 and 4-33), with $a = p_\text{sc}T_r\mu_g Z(\ln(r_e/r_w) - 0.75 + s)/(\pi k h_n T_\text{sc})$ from the model's $Z$ and $\mu_g$ at $(p_r, T_r)$, and $b$ from the non-Darcy coefficient $D \sim \text{LogU}(2\cdot10^{-4}, 2\cdot10^{-3})$ d/Mscf (D-23). R5 proposes to fit $a$ and $b$ to the pseudo-pressure at the wellstream's gravity (D-61) and to draw $D$ from permeability, pay and the completion (D-62); neither is in R6's pilots. A high-angle gas well multiplies $k h_n$ by $U(1.5, 4)$, a placeholder until Economides et al.'s horizontal gas well (§4-6) is read; a slanted one by $(\cos\theta_\text{end})^{-1/2}$.
- **Sources.** [WP §4]: permeability medians of 10 to 1000 mD by porosity, 47% of siliciclastic samples above 100 mD (Ehrenberg and Nadeau 2005, Table 1, p. 437; reservoirs, not wells; R3's median of 50 mD was lower because a well's average includes the poorer layers, an assumption, and D-100's 15 mD stands for a well's effective permeability, which also carries what the model lacks); continuous gas lift needs a PI of about 0.5 bbl/(d psi), 1.15 Sm³/(d bar), or more (Guo et al. 2007, §13.1). Golan and Whitson (1991): $\ln(r_e/r_w) - 0.75 = 7$ in their examples (Examples 2.4 and 2.6, pp. 137–145); a Kansas gas well with $kh$ = 800 md ft, $s$ = 1.89 and $D = 2.9\cdot10^{-4}$ d/Mscf, AOF 15.8 MMscf/d, $n$ = 0.766 (Example 2.4); a Nebraska well with $k$ = 20 md, 22 ft and $D = 1.5\cdot10^{-3}$ d/Mscf (Example 2.6). Economides et al. (1994): $D$ "frequently of the order of $10^{-3}$" d/Mscf and $\ln(r_d/r_w)$ usually 7 to 9 (§4-4, p. 71); $n$ between 0.5 and 1 (Eq. 4-26, p. 69); horizontal-to-vertical PI ratios of 2 to 5 for a 2000 ft drain in their Example 2-10 (Fig. 2-10, p. 35). Golan and Whitson (1991, Preface, pp. vii–viii): a good Oklahoma oil well makes 10 to 100 bbl/d, a Middle East one several hundred to several thousand, an offshore Norwegian one at least 10,000, and "a strong gas well will never produce to capacity because of equipment limitations".
- **Validity.** No correlation range: the PI and Forchheimer's law are definitions with parameters. Vogel's range is the inflow's (next draw).
- **Draws** (R6's, after D-100). $k$ 1.1 / 14 / 149 mD; oils' $J$ 0.10 / 2.0 / 42 Sm³/(d bar) (P5 / P50 / P95); high-angle oils' $J_H/J_V$ 1.9 / 4.0 / 6.0; the design liquid rate $J\delta_d p_r$ 11 / 37 / 139 / 531 / 1790 Sm³/d (P10 / P25 / P50 / P75 / P90); gas wells' AOF 0.05 / 0.74 / 4.3 million Sm³/d (P5 / P50 / P95) and design rate 0.05 / 0.41 / 1.7 million Sm³/d (P10 / P50 / P90). R3's draws, with medians of 50 and 10 mD: $k$ 3.0 / 43 / 472 mD, $J$ 0.32 / 6.6 / 134, the design liquid rate 33 / 113 / 451 / 1854 / 6411 Sm³/d, AOF 0.12 / 1.5 / 8.1 million Sm³/d. The solved rates against EIA's brackets are in `r6-measurements.md` (Realism).

### Inflow

- **Draw** (D-22, D-23). For oils, `vogel` (INF-1) if $p_\text{sat} \ge 0.8\thinspace p_r$ and the water cut is below 0.6, else `productivity_index` (INF-2): Vogel where most of the drawdown lies below the bubble point, the straight line where the oil is strongly undersaturated or water dominates. Gas and condensate wells take `forchheimer`, Forchheimer's gas inflow of feature 022 (D-60).
- **Sources.** Vogel (1968) for solution-gas drive below the bubble point, not where oil, gas and water all flow (`validity.md`, note 10). The 0.8 is an own choice.
- **Validity.** Vogel's rule as above; the well carries the label, so that a user can drop or reweight.
- **Maps to.** `ProductivityIndex(k_l=J ρ_l,sc / 86400)` in kg/(s bar), with $\rho_{l,\text{sc}}$ the stock-tank liquid's density at the water–liquid ratio; `Vogel(w_l_max=k_l p_r / 1.8)`, so that its slope at $p_0 = p_r$ is the same $J$ [WP §4]. `Forchheimer(a, b)` takes $a$ in bar² s/kg and $b$ in bar² s²/kg², converted from bar² per (Sm³/d) and per (Sm³/d)² of separator gas (feature 022, Units; D-61). Until it exists, the interim mapping is a `ProductivityIndex` whose $k_l$ gives the liquid that the gas carries at the design drawdown: the gas rate becomes linear in the drawdown (Mapping).
- **Draws.** Core oils: 64% Vogel; core wells: Vogel 43%, productivity index 24%, Forchheimer 33%.

### Tubing

- **Draw** (D-25). The smallest inner diameter from {62.0, 76.0, 100.5, 124.3, 157.1} mm (2-7/8, 3-1/2, 4-1/2, 5-1/2 and 7 in tubing), and 220.5 mm (9-5/8 in) for gas and condensate wells, whose no-slip mixture velocity at the design rate stays below the erosional velocity $v_e = 1.22\thinspace C/\sqrt{\rho_m}$ m/s, $C \sim U(100, 150)$; with probability 0.3 (tail 0.5) one size larger. The design point is the initial state, when the tubing is run: $p_i$, $R_{si}$, no water for oils and a WGR of 10 for gases, the drawdown $\delta_d$, and the wellhead at $1.5\thinspace p_\text{sep} + 5$ bar (plus the riser for subsea wells) and $T_s + 0.6\thinspace(T_r - T_s)$. If even the largest size is too small, the well gets it and is rate-limited by its tubing (draws: 17% of black oils, 31% of volatile oils, 23% of condensates, 11% of gases).
- **Sources.** [WP §6]: API tubing to 4.5 in OD and casing-sized strings of 5-1/2, 7 and 9-5/8 in in high-rate wells (Guo et al. 2007, Appendix B); API RP 14E's $C$ of 100 for continuous service and up to 150 to 200 where corrosion is controlled (as quoted in an open-access paper; the standard is not in hand); 7 in tubing for about 100 MMscf/d. The inner diameters are the common weights' and need checking against Guo et al.'s Appendix B (R2's second pass).
- **Validity.** The relative roughness stays below 0.0025, far from the refusal at 0.05 (decision 3).
- **Maps to.** `WellGeometry.from_survey(…, D=D)`.
- **Draws.** 62 mm 28%, 76 mm 16%, 100.5 mm 9%, 124.3 mm 10%, 157.1 mm 24%, 220.5 mm 13%. Late-life wells keep the tubing sized for their start, which is how wells load up with liquid.

### Roughness

- **Draw** (D-26). $\varepsilon \sim \text{LogU}$ by life stage: initial $(1.5\cdot10^{-6}, 3\cdot10^{-5})$ m, mid-life $(10^{-5}, 6\cdot10^{-5})$ m, late $(2\cdot10^{-5}, 1.5\cdot10^{-4})$ m; tail $\text{LogU}(1.5\cdot10^{-6}, 1.5\cdot10^{-4})$, SMP-42's range.
- **Sources.** [WP §6]: 30 µm (Jansen 2015), 46 µm (Hasan and Kabir 2018, Ch. 5), relative 0.0006 (Guo et al. 2007); roughness grows with corrosion and scale. The stage split is an own choice; Farshad and Rieke (2006) are on the resource list.
- **Validity.** Chen's $\varepsilon/D$ from $5\cdot10^{-7}$ to 0.05: the draws give $7.2\cdot10^{-6}$ to $2.4\cdot10^{-3}$.
- **Maps to.** `RoughnessFriction(roughness=ε)` with Chen (FRIC-4), as `v2.0.0` requires.

### Choke

- **Draw** (D-27). A Simpson choke (CHK-5) for every well. The fully open bean is $\beta = d/D \sim U(0.25, 0.6)$ of the tubing's diameter (tail $U(0.2, 0.75)$) with a discharge coefficient $C_D \sim U(0.75, 0.9)$, so $K_c = C_D\thinspace\beta^2\pi D^2/4$. The profile is one of CHK-7 to CHK-10 with equal shares.
- **Sources.** [WP §8]: $C_D$ of 0.62 to 0.90 in Sachdeva's model and 0.75 in an example (Guo et al. 2007, Eq. 5.4, p. 5/60); bean sizes of 24/64 and 32/64 in in examples. v1's $K_c$ of 0.06 to 0.24 times the tubing area is $\beta$ = 0.27 to 0.53 at $C_D$ = 0.85, inside this draw. No public distribution of choke sizes or profiles over wells is in hand.
- **Validity.** Simpson's multiplier has no range in hand (`validity.md`). The Bernoulli choke (CHK-6), a single-phase model, is not drawn; `configurations.check` would accept it.
- **Maps to.** `SimpsonChokeModel(K_c=…, chk_profile=…)`. The choke's position is R4's.
- **Draws.** $K_c/A$ 0.059 / 0.150 / 0.280.

### Gas lift: assignment

- **Draw** (D-28). An oil well needs lift if its static column cannot reach the surface: $p_r < p_{s,\text{nom}} + g\rho_m L$, with $\rho_m$ the no-slip density of its stream at the mean of $(p_r, p_{s,\text{nom}})$ and of $T_r$ and the design wellhead temperature. Such a well gets gas lift (`lift_reason = needs_lift`). Otherwise gas lift is drawn by stage:

  | | initial | mid-life | late |
  |---|---|---|---|
  | oils, offshore | 0.20 | 0.45 | 0.70 |
  | oils, onshore | 0.15 | 0.25 | 0.35 |
  | gas and condensate | 0 | 0.05 | 0.15 |

  Tail: 0.5 for oils and 0.25 for gas classes. In the core, a drawn well whose pressure-balance point (next draw) lies above 0.3 $L$, and which does not need lift, is left on natural flow: an operator does not lift a strong well (draws: 11% of all wells). No gas–oil ratio limits lift (Bjarne's goal 5), but the static check lifts heavy, low-GOR columns, so the bottom GOR quartile is lifted most, and in R6's pilot the share falls with each quartile: by quartile of the oil wells' GOR, 48 / 34 / 30 / 35% are lifted in the draws (all drawn oil wells, Draws below) and 48 / 30 / 25 / 21% in R6's accepted `sol-2` oil wells, core and tail (`r6-measurements.md`, Summary by goal). Whether that meets goal 5 is checkpoint item F1.

  In `sol-2`, an oil well that this assignment leaves natural and that cannot flow at its best rate (Producibility filter) gets the lift design below as if it had been lifted, with `lift_reason = no_natural_flow` (D-101); the static check, a no-slip column at mean conditions, misses the friction and slip that stop such wells. `nsol-2` starts lift by the same evidence (D-49, rule 2). R6's pilot: 22 wells, of which 8 were kept.
- **Sources.** [WP §1, §9]: gas lift is the main lift offshore and often the method of choice for deep-water subsea wells (Hasan and Kabir 2018, §12.2.1; von Flatern 2015); it suits wells with high PI, high GOR and strong water drive or waterflood, and whole platforms run on it at 80% water cut (Goh et al. 2007, weak). Gas lift for loaded gas wells is applicable in high-GOR wells (von Flatern 2015). The stage shares are own choices; no public count of lifted wells is in hand.
- **Validity.** None. A lifted well's lift-gas rate per sample is R4's.
- **Maps to.** The label; a natural well gets $w_{lg} = 0$ in every sample.
- **Draws.** Core oils 32% lifted (13% initial, 21% mid-life, 58% late), core gas classes 2.2% (6.5% with the tail); lifted share of all oil wells by quartile of $R_{go}$: 48%, 34%, 30%, 35%. 8.8% of all oil wells need lift.

### Gas lift: injection depth and lift-gas temperature

- **Draw** (D-29, D-30). The operating valve sits at the deeper of the pressure-balance point and nothing deeper than the packer, in true vertical depth $H$ below the wellhead:
  - the packer at $L - \Delta z_\text{pk}$, $\Delta z_\text{pk} \sim U(30, 150)$ m above the bottom (the top perforation, or the heel);
  - the surface injection pressure $p_\text{inj} \sim U(48, 76)$ bar and the valve's pressure drop $\Delta p_v \sim U(3.4, 13.8)$ bar; the casing pressure at depth $H$ is Gilbert's $p_\text{inj}(1 + H/12\thinspace 192\ \text{m})$;
  - the tubing pressure below the valve at the lift design point, $p_{wf} - G_l(L - H)$, with $p_{wf} = p_r(1 - U(0.3, 0.7))$ and $G_l$ the no-slip gradient of the stream at $(p_{wf}, T_r)$;
  - $H$ is where the casing pressure less $\Delta p_v$ equals the tubing pressure, truncated to $[0.3L,\ L - \Delta z_\text{pk}]$.

  The measured depth follows from the survey: `injection_md` = MD at TVD $H$. The lift gas's temperature is $T_{lg} = T_a(H) - \phi\thinspace\big(T_a(H) - T_s\big)$, $\phi \sim U(0, 0.3)$: from the ambient at the valve, THM-9's default, to 30% of the way towards the surface temperature, for gas that has not fully warmed in the annulus.
- **Sources.** Golan and Whitson (1991), §5.3, pp. 530–536: the valve is placed a short distance above the pressure-balance point, with an orifice for a 50 to 200 psi (3.4 to 13.8 bar) drop; surface injection pressures are usually 700 to 1100 psia (48 to 76 bar), set by the compressor; the casing pressure by Gilbert (1954), Eq. 5.17; Example 5.7 finds the injection depth of a well with formation pressure 2650 psia at 8000 ft from the IPR, a flowing gradient of 0.39 psi/ft and 900 psia at the surface (its printed 6449 ft does not satisfy its own equation, which gives 6177 ft; the method is used, not the number). [WP §9]: gas enters near the top of the perforations, and in an example the packer is 61 m above mid-perforation (Guo et al. 2007, Example 13.1). The annulus gas exchanges heat with the tubing and formation on its way down (Hasan and Kabir 2018, §12.2.2; no numbers extracted); $\phi$ is an own choice. Offshore systems may inject at higher pressures; no source is in hand (Resource list).
- **Validity.** INF-9 puts the lift gas at the grid point nearest the depth, at most half a cell away. THM-9's mixing has no range.
- **Maps to.** `WellProperties(injection_md=…)`; `BoundaryConditions(T_lg=…)`. A natural well has `injection_md=None`.
- **Draws.** $H/L$ 0.33 / 0.65 / 0.97; at the packer in 6.5% of the wells, raised to 0.3 $L$ in 1.9% (tail wells, which keep their lift); $T_{lg}$ 0.3 / 5.3 / 16.6 K below the ambient at the valve.

### Gas lift: capacity

- **Draw** (D-31). The well's lift-gas capacity, the most R4 may inject in a sample, is $w_{lg,\max} = \rho_{g,\text{sc}}\thinspace\text{GLR}_ \text{cap}\thinspace q_l/86\thinspace 400$ kg/s, with the injected gas–liquid ratio at capacity $\text{GLR}_ \text{cap} \sim \text{LogU}(50, 400)$ Sm³/Sm³ (tail $\text{LogU}(25, 1000)$) and $q_l$ the design liquid rate at the life stage, $J\delta_d p_r$, capped by the tubing's erosional rate; truncated to [0.02, 2.0] kg/s.
- **Sources.** [WP §9]: 0.5 to 6.5 MMscf/d (14,000 to 184,000 Sm³/d) across seven optimised wells producing up to 2400 Sm³/d (von Flatern 2015, Fig. 1), an injected GLR of about 77 Sm³/Sm³ if the most gas goes to the largest well; optimum total GLRs of 430 to 800 Sm³/Sm³ in small tubing (Guo et al. 2007, Example 13.1); injected GLRs of 71 Sm³/Sm³ (Golan and Whitson 1991, Example 5.8, p. 536: 80 Mscf/d for 200 STB/d) and 27 Sm³/Sm³ (Economides et al. 1994, Example 19-1, p. 525: 1.2·10⁵ scf/d for 800 STB/d). Beyond a limiting GLR more gas raises the gradient (Golan and Whitson 1991, p. 538; Economides et al. 1994, §19-1). 2.0 kg/s is about 216,000 Sm³/d at 0.8 kg/Sm³, the top of von Flatern's wells; v1's 5 kg/s reaches 19 MMscf/d.
- **Validity.** None.
- **Maps to.** No model input: R4 draws `BoundaryConditions(w_lg=…)` in $[0, w_{lg,\max}]$.
- **Draws.** 0.02 / 0.38 / 2.0 kg/s.

### Heat-transfer coefficient

- **Draw** (D-32). $h = U_t k_e/(k_e + r_{ti}U_t T_D)$, the wellbore's overall coefficient in series with the formation's transient resistance (Hasan and Kabir 2012, Eq. 6, as [WP §10] maps it to THM-1), with $U_t \sim \text{LogU}(11, 85)$ W/(m² K), $k_e \sim U(1.5, 2.6)$ W/(m K), $r_{ti} = D/2$, and $T_D$ from Hasan and Kabir (2012, Eq. 2) at the well's age, $\text{LogU}$ over 30 to 365 days (initial), 1 to 8 years (mid-life) and 5 to 25 years (late), with Ramey's diffusivity $1.03\cdot10^{-6}$ m²/s and a hole of 0.108 m radius (0.156 m for tubing above 101 mm). Tail $U(2, 40)$, which keeps v1's 10 to 40.
- **Sources.** [WP §10]: Ramey (1962, p. 431) for $k_e$ and the diffusivity; $U_{to}$ = 36 to 56 W/(m² K) along one well and 42 to 57 in another (Hasan and Kabir 2018, Table 6.1, Fig. 7.39); the implied $h$ of 3 to 25 W/(m² K).
- **Validity.** Zhang et al. (2006) state no range for $h$.
- **Maps to.** `ThermalModel(h=…)` with every `v2.0.0` term on.
- **Draws.** 3.2 / 6.3 / 11.9 W/(m² K) core; 3.2 / 6.7 / 26.5 all wells.

### Heat capacities

- **Draw** (D-33). Each at the representative temperature $T_\text{rep} = T_s + 0.8\thinspace(T_r - T_s)$:
  - oil: Cragoe (1929, Eq. 5), $c_{po} = (0.388 + 0.00045\thinspace t)/\sqrt{d}$ Btu/(lb °F) with $t$ in °F and $d$ the stock-tank specific gravity, times 4186.8;
  - water: Sharqawy et al. (2010, Eq. 9) at the salinity, by a quadratic through their values at 0, 35, 100 and 150 g/kg and 20 °C, $c_{pw} = 4190 - 5.62\thinspace S + 0.0076\thinspace S^2$ J/(kg K) with $S$ in g/kg (within 5 J/(kg K) of them; the rise of 1 to 2% from 20 to 120 °C is left out);
  - gas: methane's $c_p$ from NIST at $\big((p_r + p_s)/2,\ T_\text{rep}\big)$, bilinear in their table from 1 to 401 bar and 300 to 380 K and clamped to it, times $(0.554/\gamma_g)^{0.4}$ for heavier gases (an assumption).
- **Sources.** [RF §10]. Today's 2000, 4184 and 2225 J/(kg K) (SMP-9 to SMP-11) are the defaults they replace.
- **Validity.** Cragoe's oils span 0.75 to 0.96 specific gravity (57 to 16 °API) at 0 to 204 °C; Sharqawy's 0 to 180 g/kg, so salinities above 0.18 extrapolate; NIST's table is clamped. The model's heat capacities are constant along the well (v2.x has $c_p(p, T)$).
- **Maps to.** `FluidModel(cp_o=…, cp_w=…, cp_g=…)`.
- **Draws.** $c_{po}$ 1955 / 2116 / 2342; $c_{pw}$ 3398 / 3793 / 4032; $c_{pg}$ 2513 / 2875 / 3178 J/(kg K).

## Mapping to a well in `v2.0.0`

| Input | From |
|---|---|
| `geometry` | `WellGeometry.from_survey(md, tvd, 100, D)` |
| `fluid.pvt` | black oil and wet or dry gas: `BlackOil(rho_o, rho_g, gas_law='dak', p_sep, T_sep)`; volatile oil and condensate: a `PVTTableSet` of the well's own, from a NeqSim flash (`fluids-draft.md`; D-63 to D-67) |
| `fluid` | `FluidModel(pvt, rho_w, gor=R_go, wlr, cp_g, cp_o, cp_w)`, `surface_tension_model='oil'` (the default) |
| `friction` | `RoughnessFriction(roughness=ε)`, Chen |
| `thermal` | `ThermalModel(h)`, every `v2.0.0` term on (the defaults) |
| `slip` | `V2_SLIP`; `profile_scale` is not drawn (1, its default) |
| `inflow` | `Vogel(w_l_max)` or `ProductivityIndex(k_l)`; gas and condensate: `Forchheimer(a, b)` (feature 022) |
| `choke` | `SimpsonChokeModel(K_c, chk_profile)` |
| `injection_md` | the valve's MD, or `None` |
| `BoundaryConditions` | nominal $p_r$, $p_{s,\text{nom}}$, $T_r$, $T_s$, $T_{lg}$; $u$ and $w_{lg}$ per sample (R4) |

**Interim mapping**, used by R6's prototype until feature 022 and the table fluids exist: volatile oils and condensates as `BlackOil` with their API gravity, gas gravity, separator and $R_{go}$ (decision 4 accepts any GOR), so that their bubble or dew point is Vazquez–Beggs's rather than $p_\text{sat}$; gas and condensate wells with the `ProductivityIndex` of the Inflow section. Every one of the 2000 drawn wells built in the interim mapping passes `configurations.check(wp, 'v2.0.0')`, and its boundary conditions build. The final mapping passes it once feature 022's inflow is an option that `v2.0.0`'s check accepts (the Goals).

## The economic floor (for R4's filter)

A well that would not be produced in practice is one whose best rate at its nominal conditions lies below the floor of its well type, with a margin (D-34):

- **The floor**, in oil equivalent with 1000 Sm³ of gas = 1 Sm³: onshore and platform wells 1.6 Sm³/d (10 BOE/d, the IOGCC stripper limit; 1700 Sm³/d of gas is the gas stripper limit), subsea wells 8 Sm³/d (50 BOE/d).
- **The margin.** The filter drops a well only if its best rate is below half its floor, so that marginally economic wells stay in.
- **The best rate**, which R4 defines and solves: the choke fully open, $p_s = p_{s,\text{nom}}$, and for a lifted well the lift-gas rate within $[0, w_{lg,\max}]$ that gives the most oil equivalent.
- **Also dropped** as not produced: oil wells above a water cut of 0.98 (none in the draws).
- **Sources.** [WP §11]: stripper wells produce at most 10 bbl/d or 60 Mcf/d (IOGCC) or 15 bbl/d (IRS) over 12 months (EIA 2025, p. 10); 9.8% of the Gulf's oil wells average 10 BOE/d or less, partial years included (EIA 2025, Table B32); offshore structures need revenue of about USD 1.0 to 1.7 million a year (Kaiser 2011, abstract), about 50 BOE/d at USD 60 a barrel per structure, shared by its wells; a subsea tieback well carries its own cost.
- **Draws.** Before any solve, 4.0% of the wells have a design rate below the floor and 2.0% below half of it. The solved best rates, and the share the filter drops, are R6's.

## Ranges R4 draws within

R3 fixes, per well, the ranges that the per-sample draws use; R4 specifies the draws themselves.

| Per sample (R4) | Range fixed per well (R3) |
|---|---|
| choke position $u$ | $[0, 1]$, the profile and $K_c$ fixed |
| lift-gas rate $w_{lg}$ | $[0, w_{lg,\max}]$ for a lifted well; 0 for a natural one |
| separator pressure $p_s$ | around $p_{s,\text{nom}}$, at least 1.5 bar, and for a subsea well at least $\Delta p_\text{fl}$ |
| reservoir pressure $p_r$ (`sol-2` drift) | around the nominal $p_r$ of the life stage |
| water and GOR (`sol-2` drift) | around the stage's water–liquid ratio and $R_{go}$ |
| `nsol-2` evolution | from the stage's state, with the drive's trends: $p_r$ falls under depletion, the water rises under pressure support, and $R_{go}$ rises below $p_\text{sat}$ by the stage's $k_g$ |

## Bjarne's goals: where the draft meets them

| Goal | Where | Draws |
|---|---|---|
| 1. Depleted reservoirs, $p_r$ below hydrostatic | Drive and life stage; Reservoir state | 46.7% of core wells below hydrostatic, 30.1% by more than 10%; 99% of late depletion wells |
| 2. Separator pressures below 10 bar | Separator: late-life stages and onshore gathering | 15.6% of core wells: 45% of late platform wells, 57% of onshore wells |
| 3. Overpressured, high-rate wells | Initial reservoir pressure (overpressure with depth); Productivity | 20.4% of core wells above 0.6 psi/ft, a third of them in the top quarter of design rates |
| 4. Modest weight at a fully open choke | R4: `sol-2`'s choke position, `nsol-2`'s control history and test periods (below) | – |
| 5. Lift gas scaled to the well, and in gassier wells | Gas lift: capacity scales with the design rate; no GOR limit on the assignment, and gas classes are lifted late in life | core oils 32% lifted; 35% of the oil wells in the top quartile of $R_{go}$ (R6's accepted wells: 21%, against 48% in the bottom quartile); gas classes 2.2% (core), 6.5% (all) |
| Surface temperature by well type | Surface temperature | offshore 275 to 292 K (P5 to P95), onshore $U(268, 303)$ K |
| Injection depth | Gas lift: injection depth | $H/L$ 0.33 / 0.65 / 0.97 |
| Lift-gas temperature | Gas lift: lift-gas temperature | 0.3 to 16.6 K below the ambient at the valve |
| Fluid constants | Formation water ($\rho_w$ from salinity); Heat capacities by fluid; Separator conditions | – |
| A life stage per well | Drive and life stage, with consistent $p_r$, water and GOR | `life_stage`, `life_state` |
| A labelled coverage tail of 10% to 20% | Coverage tail: 0.20 of the drawn wells (D-103) | R3's draws at 0.15: 15.5%; R6's `sol-2` pilot: 14.5% of the accepted wells, within the target |
| No implausible wells | the links (depth → pressure, temperature; fluid → GOR, bubble point; rate → tubing, lift) and the bounds (Checks) | no bound violated |
| No wells that would not be produced | The economic floor (R4 applies it) | 2.0% below half the floor before any solve |
| Gas wells through a gas-rate inflow | Inflow; feature 022 (D-60) | interim PI |
| Volatile oils and condensates as tables | Mapping; `fluids-draft.md` (D-63 to D-68) | interim black oil |
| Horizontal drains to the heel, limit about 75° | Geometry | no cell above 75° |

## Where realism and coverage conflict

The datasets serve machine-learning tasks that need the model's range covered, not only the likely wells. A realistic draw leaves these parts thin:

- **Fluid classes.** By wells, volatile oils are 1 to 2% and condensates 3 to 12% of offshore wells [RF §6]. The Goals' floor of about 10% already over-weights them; the tail adds 25% each.
- **Rates.** Productivity spans four decades, but most realistic wells sit in the middle, and the economic floor removes the bottom. Low-rate wells, where the trickle root and liquid loading occur, and tubing-limited high-rate wells are thin. The tail's log-uniform permeability, net pay and skin spread them.
- **Flow regimes.** Oil wells flow mostly in slug and churn flow, gas wells in annular flow, and undersaturated oils in bubbly flow at the bottom. Dispersed bubbly flow needs high liquid rates, and churn flow in gas wells low gas rates; both are thin. R6 measures the regimes at the bottom and the top before and after the filters; the tail's GORs, water cuts and rates widen them.
- **Choked and unchoked flow.** Low separator pressures (goal 2) and high reservoir pressures make most wells choked at small openings; late, depleted wells at high $p_s$ are mostly unchoked. The share of choked samples depends on R4's choke draw and R3's beans; SMP-28's filter on 80% choked is dropped (D-51).
- **Gas lift.** Realistic lift follows life stage and leaves early-life and gas wells mostly unlifted: 2.2% of core gas-class wells and 13% of core initial oil wells. The tail lifts half its oil wells and a quarter of its gas wells, whatever their stage.
- **Inclination.** The limit of 75° removes the near-horizontal cells a heel's last metres would have; the inflow stands for the drain. Nothing in the datasets will show stratified flow, which the model cannot hold.
- **Extremes.** Pressures above 950 bar, temperatures above 450 K, measured depths beyond 9342 m and heavy oils below 16 °API are excluded, in core and tail alike, because the model's correlations or Hasan et al.'s data end there (D-35).

The tail's share (0.20 of the drawn wells, D-103; 14.5% of R6's accepted `sol-2` wells) and its flattened shapes are the one lever for all of these. R6 measured each item's coverage with and without the tail (`r6-measurements.md`, Coverage), and Bjarne decides at the checkpoint whether that is enough (checkpoint item A2).

## Checks a test can run

For R10's tests and R6's measurements, on the core and the tail separately:

- **Labels:** each share within its sampling error of the draft's.
- **Marginals:** the P5 / P50 / P95 above for depth, pressure gradient, temperature by class, API, gas gravity, $R_{si}$, water cut and the gas-lift capacity.
- **Rank correlations** of core black oils, against the Gulf's by wells [RF §1, §7; own analysis of BOEM's sands atlas]:

  | Pair | Draws |
  |---|---|
  | depth – $p_i$ | 0.90 |
  | depth – $T_r$ | 0.75 |
  | $R_{si}$ – $p_i$ | 0.74 |
  | $R_{si}$ – $T_r$ | 0.32 |
  | $R_{si}$ – depth | 0.63 |
  | API – $R_{si}$ | 0.31 |

  Each lies within 0.12 of the Gulf's but $R_{si}$–$T_r$, which the draws make weaker. The condensates' GOR falls with depth (−0.38), as their yield rises with it in the Gulf.
- **Bounds**, each violated by none of the 2000 wells: every cell within 75° of vertical; MD at most 9342 m; $p_i$ at most 950 bar and $\lambda$ at most 0.85; $T_r$ at most 450 K; $(T_r - T_s)/L$ at least 10 K/km; $p_\text{sat} \le p_i$; black-oil $R_{si}$ from 1 to 356 Sm³/Sm³; API from 16 to 58; gas gravity at least 0.571; oils' water cut at most 0.98; `configurations.check` passes in the interim mapping.
- **Ranges left**, accepted: $p_r$ above 363 bar (Beggs–Robinson) in 32% of the wells and above 552 bar (Lee–Gonzalez–Eakin) in 10%; $T_r$ above 419 K in 7.6%; GOR above 1000 Sm³/Sm³ (Hasan et al.) in 40%, the gas classes and condensates.
- **Reproducibility:** a rerun gives the same draws (SMP-31), and a change of one block's distribution leaves the other blocks' draws as they were.

The sanity script prints all of these. R6 adds the rate check: the solved rates of offshore oil wells against the Gulf's 2024 brackets (EIA 2025, Table B32; D-24), and of gas wells against the AOF and tubing limits.

## The model's gaps (R5)

- **Gas-rate inflow.** Gas and condensate wells need it; the draws give Forchheimer's $a$ and $b$ (Inflow). R5 drafted feature 022, Forchheimer's law in the pressure squared (D-60), which `v2.0.0`'s check accepts by the Goals' decision.
- **PVT tables for volatile oils and condensates.** The draws give each well's descriptors: API gravity, gas gravity, $R_{si}$, $p_\text{sat}$ at $T_r$, the separator, and the ranges of $p$ and $T$ the table must cover ($p_s$ to $p_i$, from below $T_s$ to $T_r$). R5 proposes synthetic compositions matched to them, or a published example in a fifth of the wells, and one table set per well with a GOR axis (PVT-TAB-7) (`fluids-draft.md`; D-63 to D-67). Condensates' tables wait for D-64. Until then the interim black oil stands in.
- **Wet gases** stay black oil (decision 4): their condensate is a small part of the flow. R5 may measure what a table changes for them.
- **The productivity that stands for a horizontal drain** needs no model change: it is the draw's $J_H/J_V$ (Productivity). A gas drain's factor is a placeholder until Economides et al.'s §4-6 is read.
- **A composite IPR**, straight above the bubble point and Vogel below, is not an option; the inflow rule picks one or the other. R5 may weigh it.
- **Heavy oils** below 16 °API need a table; none is drawn.
- **Waiting for v2.x:** a water column in the ambient profile (deep-water platforms), $c_p(p, T)$, a varying $h$, and a tubing that changes diameter.

## Open issues

- **Done by R4** (The samples within a well): the choke draw and its weight at a fully open choke; the lift-gas draw within $w_{lg,\max}$; the drift between samples and the `nsol-2` trends from the life stage's state; the best rate for the economic floor; SMP-28's filters dropped (D-51).
- **Done by R5:** the gas-rate inflow (feature 022); the table fluids and their matching to the descriptors (`fluids-draft.md`); R6's pilots in the interim mapping, volatile oils and condensates as black oil (D-69).
- **Done by R6** (`r6-measurements.md`): the solved rates against the Gulf's brackets, which led to D-100; the shares of wells and samples without an operating point, by class and geometry; the coverage of regimes, choked flow and lift with and without the tail; the ruled geometry and its effect.
- **For R2's second pass:** the gaps on the resource list below, the lithostatic gradient, offshore gas-lift injection pressures and choke sizes. The geometry's shares are ruled (D-9).
- **For Bjarne at the checkpoint:** the onshore share (0.10, a design choice since onshore wells dominate by count; checkpoint item A1), and whether the coverage tail may lift wells and assign geometries regardless of stage and type, as drafted (A2).

## Resource list (additions)

| Reference | Decision it bears on | Assumption without it |
|---|---|---|
| A source for the lithostatic gradient offshore (for example Swarbrick and Osborne 1998) | D-12 | 0.226 bar/m below the mudline |
| Economides et al. (1994), §4-6 (horizontal gas wells) and §5-4 (slant skin) | D-21 | $U(1.5, 4)$ on $kh$; $(\cos\theta)^{-1/2}$ |
| Offshore gas-lift injection pressures (Brown 1980; an operator's design guide) | D-29 | 48 to 76 bar (Golan and Whitson 1991, onshore practice) |
| Joshi (1991), *Horizontal well technology* | D-21 | Joshi's equation as Economides et al. (1994) give it |
| NOAA World Ocean Atlas; climate normals of producing regions | D-14 | the seabed profile and $U(268, 303)$ K |
| Guo et al. (2007), Appendix B, read for the tubing's inner diameters | D-25 | the common weights' inner diameters |

## References

- Brill, J.P. and Mukherjee, H. (1999), *Multiphase flow in wells*, SPE Monograph 17: p. 44.
- Cragoe, C.S. (1929), *Thermal properties of petroleum products*, US Bureau of Standards Miscellaneous Publication 97: Eq. 5.
- Economides, M.J., Hill, A.D. and Ehlig-Economides, C. (1994), *Petroleum production systems*, Prentice Hall: §2-7 (Eqs. 2-46 to 2-48, Example 2-10, Fig. 2-10, pp. 31–35), §4-3 (Eq. 4-26, p. 69), §4-4 (Eqs. 4-32 to 4-34, p. 71), §19-1 and Example 19-1 (pp. 523–525).
- EIA (2025), *The distribution of U.S. oil and natural gas wells by production rate*: pp. 3, 10; Table B32.
- Ehrenberg, S.N. and Nadeau, P.H. (2005), *AAPG Bulletin* 89(4), 435–445: Table 1.
- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall: Preface (pp. vii–viii), Examples 2.4 and 2.6 (pp. 137–145), Example 4.11 (p. 454), §5.3 (Eqs. 5.16 and 5.17, Examples 5.7 and 5.8, pp. 528–538).
- Guo, B., Lyons, W.C. and Ghalambor, A. (2007), *Petroleum production engineering*, Gulf Professional Publishing: §5.1 (p. 5/60), §10.2.5 (p. 10/124), Table 10.4, §13.1, Example 13.1, Appendix B.
- Hasan, A.R. and Kabir, C.S. (2012), *J. Petroleum Science and Engineering* 86–87, 127–136: Eqs. 2 and 6. Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., SPE: §12.2, Table 6.1, Figs. 7.17 and 7.39.
- Liu, M., Wang, Z. and Kaiser, M.J. (2014), *Oil & Gas Journal*, 3 March 2014.
- McCain, W.D. (1990), *The properties of petroleum fluids*, 2nd ed., PennWell: pp. 149–158, 438–449.
- Nadeau, P.H., Sun, S. and Ehrenberg, S.N. (2023), *Marine and Petroleum Geology* 158, 106507: Table 1, Figs. 7 and 20.
- Ramey, H.J. (1962), *Journal of Petroleum Technology* 14(4), 427–435: p. 431.
- Sharqawy, M.H., Lienhard, J.H. and Zubair, S.M. (2010), *Desalination and Water Treatment* 16, 354–380: Eq. 9.
- Valkó, P.P. and McCain, W.D. (2003), *J. Petroleum Science and Engineering* 37, 153–169: Table 2.1.
- Vogel, J.V. (1968), *Journal of Petroleum Technology* 20(1), 83–92.
- von Flatern, R. (2015), "The defining series: gas lift", *Oilfield Review*: Fig. 1.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20: Table 2.1, §2.5, §9.2.
- The public databases behind [RF] and [WP]: BOEM's sands atlas and borehole file, Sodir's FactMaps; R2's statistics of them are the maintainer's analyses, kept outside the repo and cited as [own analysis of …].

## The samples within a well (R4)

*Step R4 of `plans/sampler-redesign-plan.md`, 2026-10-30; revised by R7 on 2026-11-02 for Bjarne's ruling on D-52, under which a failed draw is redrawn until the well has 200 operating points. A draft for Bjarne's checkpoint (R8), in the same form as the well draws above, without SMP IDs. Plans are drafts, not specs.*

This part proposes how a well's samples vary: the per-sample draws of `sol-2`, the evolution of `nsol-2` over a well's life, and the rules of generation (starts, filters and failures). It builds on R3's well draws and keeps the same three kinds apart:

- **Decided** (the Goals, 2026-10-30): 10,000 wells of 200 samples in each dataset. In `sol-2` the choke position (close to uniform, with modest weight at fully open), a lift-gas rate scaled to the well and the separator pressure vary between samples, and the reservoir pressure and the fractions drift by ±2% and ±5%; the life stage is fixed per well. In `nsol-2` the reservoir evolves with linked trends, the controls follow a mixed open-loop history of set points and test periods, and a well's 200 samples span its life. No unrealistic wells and none that would not be produced, with a small margin; distributions are not shaped by solver failures or filters (acceptance criterion 4). *Ruled 2026-11-02 (D-52):* "200 samples per well" means 200 operating points; a failed draw is redrawn until the well has 200, and every redraw is logged with its inputs and its reason.
- **Proposed:** every distribution, parameter and rule below, each resting on a working assumption in `decisions.md` (D-37 to D-56) or, for the redraws, in `decisions.md` (D-80 to D-85), which R7 added when it revised these rules for the ruling.
- **Left to R6 and R7:** the measurements that confirm the choices (What R6 must measure), and the dataset design (R7, `datasets-draft.md`), which publishes the redraw log next to each dataset.

Sources are cited as in R3, [RF §n] and [WP §n], with the primary reference. The sample draws were checked without solving by `plans/evidence/sampler_draft_samples.py` on 1000 wells, seed 2026, which takes R3's wells from `sampler_draft_draws.py`; its numbers are quoted as *draws*. The rules that need a solve (the producibility filter, the end of an `nsol-2` life at the economic floor, and the start of gas lift when natural flow ends) are R6's to measure.

### Conventions of the samples

- **Slots and attempts.** A well has 200 sample slots, $j$ = 0 to 199, and each slot ends with one operating point (D-52, ruled). A slot's first draw, attempt $a = 0$, is made before any solve, from the slot's own generator. An attempt's outcome is an operating point, no operating point, or a solver failure; an attempt without an operating point is redrawn as attempt $a + 1$ (Slots and outcomes). In this part a *sample* is a slot's accepted attempt, and a *draw* is any attempt.
- **Seeding** (D-7 extended; D-80). `sol-2` slot $j$ of well $k$ draws its first attempt from `rng_for(seed, k, j, 'v2', 'sample')`. An `nsol-2` well draws its trend parameters from `rng_for(seed, k, 'v2', 'evolution')` and its control history and per-date drift from `rng_for(seed, k, 'v2', 'controls')`. In both datasets, attempt $a \ge 1$ of slot $j$ draws from `rng_for(seed, k, j, a, 'v2', 'redraw')`, so that a redraw moves no other draw, and the first attempts are the draws drafted below. Every draw of an attempt is made whatever branch it takes, so that a change to one rule moves no other draw.
- **Units** as R3's, with time $t$ in years from the well's first production. The water is the water–liquid ratio (WLR) for oils and the water–gas ratio (WGR, Sm³ per 10⁶ Sm³) for gases, as in R3.
- **Oil equivalent.** $q_\text{oe} = q_o + q_g/1000$ in Sm³/d, with $q_g$ the formation gas only, without the lift gas: R3's economic floor.
- **The same well at another stage.** R3's draws with the life stage set to `initial`, `mid_life` or `late` after it is drawn, from the same generators, so that only the stage's own draws change. `nsol-2` takes its start and end states from it.

## `sol-2`: the draws per sample

### Choke position

- **Draw** (D-37). $u = 1$ with probability $\pi_1 = 0.05$, else $u \sim U(0.05, 1)$; independent per slot, the same for core and tail wells.
- **Depends on.** Nothing: the bean $K_c$ and the profile are the well's (R3, Choke).
- **Sources.** Bjarne's goal 4 sets the shape: modest weight at fully open and close to uniform, because a wide span is more informative to machine-learning models. $u_\text{min} = 0.05$ is v1's (SMP-18). No public distribution of choke openings in operation was found [WP §8]. Chokes are used to limit rates for regulations, to protect surface equipment from slugging, and to avoid sand and water or gas coning (Guo et al. 2007, §5.1, p. 5/60), which is why an operating well is seldom fully open until late in its life.
- **Validity.** No range on $u$. At $u = 0.05$ the open area is 1.2% (sigmoid, CHK-8) to 5% (linear, CHK-7) of the bean's. Small openings are where v1's failed draws lay (`plans/model-lock-plan.md`, Step L6); here an attempt there without an operating point is redrawn and logged, so the accepted positions are thinner at small openings than the draws, by a share R6 measures (What R6 must measure).
- **Outside a range.** Nothing is truncated.
- **Draws.** 5.0% at fully open and 5.1% below 0.1; mean 0.547 and standard deviation 0.287, against 0.525 and 0.274 for $U(0.05, 1)$.

### Lift-gas rate

- **Draw** (D-38). For a lifted well, $w_{lg} = 0$ with probability 0.10 (lift gas off), else $w_{lg} = v\thinspace w_{lg,\max}$ with $v \sim U(0, 1)$. A natural well has $w_{lg} = 0$.
- **Depends on.** R3's lift label and capacity $w_{lg,\max}$, which scales with the well's design liquid rate (Bjarne's goal 5).
- **Sources.** [WP §9]: von Flatern (2015, Fig. 1), seven wells optimised between 0.5 and 6.5 MMscf/d; beyond a limiting gas–liquid ratio more gas raises the gradient (Golan and Whitson 1991, §5.3, p. 538), so the top of the range lies past the optimum for some wells, which a lift-optimisation task needs to see. Lift gas is off during compressor trips and some tests; no public share was found, and 0.10 is an own choice. Below a minimum injection rate gas-lifted wells head and can stop producing (Asheim 1988, p. 1452, after Bertuzzi et al.).
- **Validity.** None. A well that needs lift has no operating point at $w_{lg} = 0$; that is its physics. Such an attempt is redrawn and logged, so that well's samples hold almost no lift-off points, and the redraw log holds where it dies.
- **Draws.** 24.1% of the wells are lifted (all classes, core and tail); 10.0% of their slots have no lift gas; $v$ has a median of 0.44, and $w_{lg}$ is 0 / 0.077 / 1.57 kg/s (P5 / P50 / P95).

### Separator pressure

- **Draw** (D-39). $p_s \sim U\big(\max(0.9\thinspace p_{s,\text{nom}},\ 1.5\ \text{bar},\ \Delta p_\text{fl}),\ 1.1\thinspace p_{s,\text{nom}}\big)$, with R3's nominal pressure and a subsea well's flowline loss.
- **Depends on.** R3's separator block.
- **Sources.** v1's ±10% (SMP-20). The pressure downstream of the choke follows the separator's control and, for a subsea well, the commingled flow in its flowline [WP §7]; no public spread was found. A choke in sonic flow isolates the well from downstream changes (Guo et al. 2007, §5.2, p. 5/60), so the spread matters only to unchoked samples.
- **Validity.** None; truncated at R3's floor.
- **Draws.** $p_s/p_{s,\text{nom}}$ 0.91 / 1.00 / 1.09; 16.6% of the slots below 10 bar (Bjarne's goal 2).

### Drift of the reservoir

- **Draw** (D-40). Around the stage's nominal state (R3, Reservoir state at the life stage):
  - $p_r = p_{r,\text{nom}}\thinspace U(0.98, 1.02)$;
  - the producing GOR: R3's formula evaluated at the drifted $p_r$ with the well's own $k_g$ (oils) or condensate factor, times $U(0.95, 1.05)$; wet and dry gas $R_{si}\thinspace U(0.95, 1.05)$;
  - the water: oils $\min\big(0.98,\ \text{WLR}_ \text{nom}\thinspace U(0.95, 1.05)\big)$; gases $\text{WGR}_ \text{nom}\thinspace U(0.95, 1.05)$, with WLR = WGR/(WGR + 10⁶/$R_{go}$).

  $T_r$, $T_s$, $T_{lg}$ and the inflow's coefficients ($k_l$, $w_{l,\max}$, Forchheimer's $a$ and $b$) do not drift: they are the well's.
- **Consistency with the life stage.** ±2% is small next to the stages' ranges of $p_r/p_i$ (R3: 0.55 to 0.92 in mid-life under depletion), so the stage label holds. Tying the GOR to the drifted $p_r$ keeps a saturated oil's relation between pressure and free gas: a lower $p_r$ releases more gas, as the GOR rises below the bubble point (McCain 1990, pp. 151, 153). An initial-stage well's drifted $p_r$ may exceed $p_i$ by up to 2%: accepted, since a cap would put a point mass at $p_i$.
- **Sources.** The Goals (±2% and ±5%, as v1's SMP-21 and SMP-22). v1 drifted the mass fractions; this draft drifts the ratios it draws.
- **Validity.** The water cut is capped at Hasan et al.'s 0.98 (`validity.md`); the rest as R3.
- **Outside a range.** An attempt with $p_r \le p_s$ has no operating point, without a solve: a root needs $p_0 < p_r$ and $p_N > p_s$ (SOL-2). It is redrawn and logged. Draws: 1.11% of the first attempts.
- **Draws.** $p_r/p_{r,\text{nom}}$ 0.982 / 1.000 / 1.018; $R_{go}/R_{go,\text{nom}}$ 0.95 / 1.00 / 1.06, and 0.92 / 1.00 / 1.09 for saturated oils, whose GOR follows $p_r$; the water 0.954 / 1.000 / 1.047 of its nominal.

## `nsol-2`: a well's life

### The well and its states

- **Draw** (D-41). An `nsol-2` well is R3's well at the initial stage: the start of its production. Its state at the end of its life span is the same well's late-stage state, and its separator's middle step the mid-life one (The same well at another stage). The start state and the end state anchor the trends below; the trends' shapes come from sources, their ends from R3.
- **Tail wells.** A tail well's stage draws are its tail draws, the same at every stage. It starts from the core's initial state ($p_r/p_i \sim U(0.97, 1)$; WLR 0 with probability 0.5, else $U(0, 0.1)$; WGR $\text{LogU}(2, 20)$) and ends at its tail state, with the end pressure at most the start's.
- **The stage label** becomes a label per sample: `initial` in the first year, `late` from half the life span, `mid_life` between, after R3's ages (initial 30 to 365 days, mid-life 1 to 8 years, late 5 to 25 years).
- **Validity.** As R3's at every state, since the trends move between R3's states.

### Life span and the depletion clock

- **Draw** (D-41). The life span $T_\text{life} \sim \text{LogU}(8, 30)$ years. The reservoir's cumulative production, normalised to the life span, is the clock of the trends:

  $$F(t) = \frac{N(t)}{N(T_\text{life})}, \qquad N(t) = \begin{cases} t, & t \le t_\text{pl} \\ t_\text{pl} + \dfrac{1 - (1 + bD\tau)^{1 - 1/b}}{(1 - b)\thinspace D}, & \tau = t - t_\text{pl} > 0 \end{cases}$$

  a plateau of unit rate, then Arps's hyperbolic decline (exponential as $b \to 0$), with $t_\text{pl} = U(0, 0.25)\thinspace T_\text{life}$, $D \sim \text{LogU}(0.07, 0.30)$ per year, and $b \sim U(0, 0.4)$ under depletion and $U(0.3, 0.5)$ under pressure support.
- **Why a clock.** The reservoir is drained by the field, whose decline the well shares, not by the well's own solved rate. The trends therefore depend on no solve, and the controls stay open loop. $F$ stands for the recovery that drives the pressure (material balance) and the water (frontal advance).
- **Sources.** Arps's equations (Ahmed 2006, Eqs. 16-1 to 16-4, pp. 1238–1239). $b$ from 0 to 0.5, exponential decline the most common, about 0.3 for solution-gas drive and 0.5 for water drive (Golan and Whitson 1991, p. 385). Offshore fields decline by 7.3 to 9.7% a year production-weighted, 11.6% for large fields, about 10% in Norway and 20 to 23% in the Gulf of Mexico; 1.4% a year from peak to the end of the plateau (Miller et al. 2009, Tables 4.2 to 4.4, §3) [RF §11]. Wells decline faster than fields, so $D$ starts at the fields' rates. Norwegian fields shut down by 2026 produced for 5 / 11.5 / 32 years (P10 / P50 / P90, fields) [RF §11], a sample cut short because producing fields are left out; v1 drew 10 to 20 years (SMP-23).
- **Validity.** None: the clock is not a model input.
- **Draws.** $T_\text{life}$ 8.6 / 15.6 / 27.9 years.

### Reservoir pressure

- **Trend** (D-42). From the start ratio $r_0 = p_r(0)/p_i$ to the end ratio $r_1$:
  - oils, and every well under pressure support: $p_r(t) = p_i\big(r_1 + (r_0 - r_1)(1 - F)^\gamma\big)$ with $\gamma \sim U(1, 2)$;
  - gas and condensate wells under depletion: $p/Z$ linear in $F$, $p_r/Z(p_r) = (1 - F)\thinspace p_0/Z_0 + F\thinspace p_1/Z_1$, with the model's $Z$ (PVT-GAS-11) at $T_r$.
- **Sources.** The gas material balance: $p/z$ falls on a straight line against cumulative gas (Ahmed 2006, Eq. 13-8, p. 861). Golan and Whitson's (1991, Example 4.11, Table E4.11a, p. 454) subsea oil well under solution-gas drive: 4000, 3520, 3100, 2800, 2500 and 2300 psia at equal steps of cumulative oil. The form fits it within 0.008 $p_i$ at $\gamma$ = 1.40 (own least-squares fit), where a straight line misses by up to 0.055 $p_i$. One example, so $\gamma$ spans 1 to 2. The ends are R3's state tables [RF §3].
- **Validity.** As R3's.
- **Draws.** Core oils' $p_r/p_i$ at $t/T_\text{life}$ = 0, 0.5 and 1: medians 0.985, 0.84 and 0.81 (all drives); gases 0.98, 0.59 and 0.45.

### Water

- **Trend** (D-43). Oils keep their start WLR until breakthrough at the recovery $F_\text{bt}$. There the water cut jumps to $\text{WLR}_ \text{bt}$ and then rises with its logit linear in $F$ to the end state's $\text{WLR}_ 1$:

  $$\text{logit}\thinspace\text{WLR}(F) = \text{logit}\thinspace\text{WLR}_\text{bt} + \big(\text{logit}\thinspace\text{WLR}_1 - \text{logit}\thinspace\text{WLR}_\text{bt}\big)\frac{F - F_\text{bt}}{1 - F_\text{bt}}, \qquad F \ge F_\text{bt},$$

  never below the start's WLR, and with $\text{WLR}_ \text{bt}$ at most $\text{WLR}_ 1$. Under pressure support $F_\text{bt} \sim U(0.05, 0.4)$ and $\text{WLR}_ \text{bt} \sim U(0.05, 0.3)$; under depletion $U(0.2, 0.7)$ and $U(0.02, 0.1)$. Gases keep their start WGR until $F_\text{bt}$ and then rise log-linearly in $F$ to the end's. The logit of the water cut is the logarithm of the water–oil ratio, so this is the semilog straight line of the WOR against recovery.
- **Sources.** The fractional flow of water, $f_w = 1/\big(1 + (k_{ro}/k_{rw})(\mu_w/\mu_o)\big)$ without gravity and capillary pressure, is S-shaped in the water saturation (Ahmed 2006, Eq. 14-22 and Fig. 14-12, p. 941). The central part of a relative permeability ratio is a straight line on semilog paper, $a e^{bS}$ (Eq. 5-28, p. 309, which says the same of $k_{ro}/k_{rw}$), so $\text{logit}\thinspace f_w$ is linear in $S_w$. The displacement efficiency grows with the average water saturation (Eq. 14-10, p. 935), so at a fixed sweep $S_w$ rises with recovery: own step to a logit linear in $F$. At breakthrough "the water cut increases suddenly from zero" to the front's (Ahmed 2006, p. 970), 0.78 in his one-dimensional example (p. 983). A layered reservoir spreads that jump over its layers, so the draft jumps to 0.05 to 0.3, an own choice. The end states are R3's late ones, after Norway's fields [RF §9].
- **Validity.** Water cuts to 0.95 (R3's late range), 0.98 in the tail.
- **Draws.** Core oils' WLR at $t/T_\text{life}$ = 0, 0.25, 0.5, 0.75 and 1: medians 0.010, 0.25, 0.47, 0.61 and 0.71. Under pressure support by years on production: 0.19 / 0.55 / 0.68 / 0.78 at 0–5 / 5–10 / 10–20 / 20–30 years, against the medians of Norwegian oil fields [RF §9; own analysis of Sodir's FactMaps], which count fields whose wells started at different times. Gases' WGR medians 6.5, 20 and 47 at 0, 0.5 and 1.

### Gas–oil ratio

- **Trend** (D-44). $R_{go}(t)$ is R3's formula at $p_r(t)$, with the well's own $k_g$ for oils and its own factor for condensates; wet and dry gas keep $R_{si}$. The GOR is flat while the reservoir is above its saturation pressure and rises once $p_r(t)$ falls below it, so the three trends are linked through $p_r$. The fall of the GOR late in a solution-gas drive is not modelled.
- **Sources.** McCain (1990, pp. 151, 153): the producing GOR rises once the pressure falls below the bubble point. Golan and Whitson (1991, Example 4.11, p. 454): 107, 178, 285, 445 and 891 Sm³/Sm³ as $p_r$ falls from 4000 to 2500 psia, then 445 at 2300 psia (600 to 5000 and 2500 scf/STB). Its GOR rises from the first step, so its oil starts near saturation. R3's linear form with $k_g$ up to 15 reaches 6.6 times $R_{si}$ at 0.625 of the starting pressure, where the example reaches 8.3; it does not fall afterwards.
- **Draws.** Core oils' $R_{go}/R_{si}$ medians 1.01, 1.40 and 1.54 at $t/T_\text{life}$ = 0, 0.5 and 1, P95 5.6 at the end.

### Drift between dates

- **Draw** (D-40). The GOR and the water of each sample date are their trends' values times $U(0.95, 1.05)$ each, as `sol-2`'s drift; the samples of a test period share their date's. $p_r$ follows its trend without noise, since the reservoir pressure changes smoothly.

### Well properties that age

- **Trend** (D-45).
  - **Heat transfer:** R3's $h$ with the well's $U_t$ and $k_e$ and the age $a_0 + t$, $a_0$ the initial draw's 30 to 365 days. A tail well keeps its tail $h$.
  - **Roughness:** log-linear in $t$ from the initial draw's $\varepsilon$ to the late draw's.
  - **Productivity:** R3's formulas at the current state: the oils' PI with $\mu_o$ and $B_o$ at $p_r(t)$ and $R_{go}(t)$; Vogel's $w_{l,\max} = k_l p_r/1.8$ at the current $p_r$; Forchheimer's $a$ and $b$ at the current $Z$ and $\mu_g$, or the interim PI.
  - **Fixed:** the inflow model, chosen by R3's rule at the start; the tubing; the choke's bean and profile; $T_r$, $T_s$, the salinity, the API and gas gravities and $R_{si}$.
- **Sources.** $h$ falls with production time through the formation's transient (Hasan and Kabir 2012, Eq. 2) [WP §10]; roughness grows with corrosion and scale [WP §6]; R3's productivity.
- **Why the inflow model stays.** Switching from a productivity index to Vogel's inflow when $p_r$ falls below $p_\text{sat}$ would cut the rate at a large drawdown by up to a factor 1.8 from one sample to the next, a step that no change in the reservoir explains. A composite inflow (R3, The model's gaps) would remove the need.
- **Draws.** $h$ medians 8.0, 6.2 and 6.0 W/(m² K) at $t/T_\text{life}$ = 0, 0.5 and 1.

### Separator over the life

- **Trend** (D-46). $p_{s,\text{nom}}(t)$ is R3's formula, $r_s(t)\thinspace p_\text{sep} + \Delta p_\text{fl}$ truncated to [1.5, 120] bar, with two step-downs: $r_s = 1$ until $t_1$, the mid-life draw's $r_s$ ($U(0.5, 1)$) until $t_2$, and the smaller of it and the late draw's ($U(0.1, 0.6)$) after, with $t_1 < t_2$ the sorted pair of $U(0.25, 0.9)\thinspace T_\text{life}$. A tail well scales $p_{s,\text{nom}} - \Delta p_\text{fl}$ by the same steps. Each sample then draws $p_s$ as D-39.
- **Sources.** [WP §7]: Oseberg B moved its oil from a first stage at 67 bar to the second at 20 bar after 18 years, and Njord lowered its stage pressures in tail production (trade press, weak); a train at 65, 20 and 2 bara.
- **Draws.** $p_{s,\text{nom}}$ at the end over the start: 0.15 / 0.46 / 0.81. 17.4% of the slots below 10 bar.

### Control history

- **Draw** (D-47). The choke and the lift gas follow set points $u_\text{sp}$ and $w_\text{sp}$ that are constant between changes:
  - **start:** $u_\text{sp} = 1$ with probability 0.05, else $U(0.05, 1)$; $w_\text{sp} \sim U(0, w_{lg,\max})$ from the start of lift;
  - **changes:** at each operating sample after the first, each set point changes with probability $p_c$, drawn per well from $\text{LogU}(0.05, 0.25)$;
  - **a change** is a step of $U(-0.2, 0.2)$ (times $w_{lg,\max}$ for the lift gas), reflected into the range, with probability 0.6, else a redraw: the choke as at the start, the lift gas 0 with probability 0.10, else $U(0, w_{lg,\max})$.

  The set points do not respond to any solve: the controls stay open loop (the Goals).
- **Sources.** No public record of how often or by how much operators change chokes and lift-gas rates was found [WP §8, §9]; the numbers are own choices. Two public datasets may hold such histories (Resource list (R4 additions)).
- **Draws.** 17 changes of the choke per well over its life (7 to 39, P5 to P95), about one every 11 months at the median; the operating samples lie 20 / 36 / 64 days apart. 2.2% of the operating samples are fully open; with the tests, 5.2% of the slots.

### Test periods

- **Draw** (D-48). A life of 200 slots holds 160 operating samples at $t_j = j\thinspace T_\text{end}/159$ and 8 test periods of 5 samples at $t_m = (m - 1/2)\thinspace T_\text{end}/8$, $m$ = 1 to 8. A test's samples share the reservoir state of its date. A natural well sweeps its choke in every test; a lifted well alternates a choke sweep and a lift sweep:
  - **choke sweep:** $u = 1$, then one draw in each quarter of [0.05, 1), from the top down, at the operating lift-gas rate;
  - **lift sweep:** $w_{lg} = w_{lg,\max}$, then one draw in each quarter of $[0, w_{lg,\max})$, from the top down, at the operating choke.
- **Sources.** Multi-rate tests and gas-lift performance curves are standard practice (Golan and Whitson 1991, §5.3, the lift performance curve); no public schedule of tests was found. Eight tests in a life of 8 to 30 years is about one every one to four years; their share of the slots (20%) is a design choice for coverage.
- **Redraws.** A sweep point without an operating point is redrawn like any slot (Redraws in `nsol-2`): its controls step away from the sweep's value, so the point may leave its quarter. Its `attempt` marks it.
- **Draws.** 20.0% of the slots are in tests.

### Gas lift that starts mid-life

- **Rule** (D-49). A well lifted in R3's initial-stage draw is lifted from the start. Otherwise lift starts at the earliest of:
  1. the first date at which R3's static check fails at the current state (every oil well);
  2. the last point of the life scan (The end of a life) before the first at which the natural well's best rate falls below half its floor or it has no operating point (every oil well, and a gas well that R3's late-stage draw lifts; D-84), so that the natural well flows at every date before lift starts;
  3. a drawn time $U(0.2, 0.9)\thinspace T_\text{life}$, for a well that R3's late-stage draw lifts.

  The design (injection depth, $T_{lg}$, $w_{lg,\max}$) is R3's lift block of the late-stage draw, with the lift label set where rule 1 or 2 starts a well that the late draw leaves natural. Before the start the well has $w_{lg} = 0$ and no injection point; after it, the set point of D-47.
- **Sources.** Golan and Whitson (1991, Example 4.11): natural flow ends after 1.7 years and gas lift follows [RF §11]; whole platforms run on gas lift at 80% water cut (Goh et al. 2007) [WP §9]; R3's stage shares (D-28).
- **Draws** (rules 1 and 3 only; rule 2 needs solves). 58.4% of all wells stay natural, 12.6% are lifted from the start and 29.0% later (24.0% at the drawn time, 5.0% by the static check), starting at 0.20 / 0.48 / 0.85 of the life span.

### The end of a life

- **Rule** (D-50, amended by D-84). Before the slots are drawn, a life scan solves the well's best rate (Producibility filter) at 21 points, every 5% of $T_\text{life}$, with the lift that the well has at each. A point fails where the best rate falls below half the floor, the well has no operating point, or an oil well's water cut passes 0.98. The life ends at $T_\text{end}$, the last point before the first that fails, or at $T_\text{life}$ if none fails. The 200 slots then span $[0, T_\text{end}]$. A scan point with a solver failure is skipped. A well whose first point after the start fails has no life to span; it is dropped as `short_life` and logged.
- **Why the last point that passes.** Every date needs an operating point (D-52, ruled). At a failed point the well is not produced, by the filter's definition, and may not flow at any control, so no redraw there need succeed. R4's draft ended the life at the first failed point, whose slots then held no operating point.
- **Sources.** R3's economic floor (D-34) [WP §11].
- **Draws.** Needs solves (R6). Without them, every life runs to $T_\text{life}$.

## Generation rules

### Producibility filter

- **Rule** (D-50). A well's **best rate** is its oil equivalent with the choke fully open ($u = 1$), $p_s = p_{s,\text{nom}}$, the nominal $p_r$, GOR and water, and, for a lifted well, the most of three lift-gas rates, $\lbrace 0,\ 1/2,\ 1\rbrace\thinspace w_{lg,\max}$. A well whose best rate is below half its floor (R3: 1.6 Sm³/d for onshore and platform wells, 8 Sm³/d for subsea wells), or that has no operating point at those conditions, is not produced and is dropped. So is an oil well with a water cut above 0.98.
  - **`sol-2`:** at the well's stage, before its slots are solved.
  - **`nsol-2`:** at the start; a well that fails it is dropped. Later, the life scan ends the life at the last scan point before the best rate falls below the same limit (The end of a life).
- **Why three lift rates.** The lift performance curve is flat near its optimum (Golan and Whitson 1991, §5.3), so the most of three points lies close to the optimum, at three solves.
- **Sources.** D-34, and [WP §11].
- **Effect.** The filter removes wells on purpose (acceptance criterion 2). Its effect on the wells' distributions is measured, not hidden: the generation log keeps every dropped well with its inputs and reason, and R6 compares the marginals before and after. Before any solve 2.0% of the wells have a design rate below half the floor (R3). R6's pilots drop 10.2% of the drawn `sol-2` wells (5.5% of core wells, 30% of tail wells) and 5.5% of the drawn `nsol-2` wells at the start.
- **A natural oil well that fails** (D-101, `sol-2`): an oil well left natural by R3's assignment whose best rate fails is lifted with R3's lift design (Gas lift: assignment) and its best rate is solved again; only if that fails too is it dropped.

### SMP-28's filters

- **Rule** (D-51). None of SMP-28's three filters is kept, for either dataset:
  - **The choke's spread** (a standard deviation below a fifth of the uniform's). The drawn positions do not fail it: the standard deviation of 200 slots drawn as D-37 is about 0.29 ± 0.015, against the threshold of 0.055. It can fail only on the accepted positions, which the redraws thin where a weak well cannot flow, so it would drop wells for where they cannot flow.
  - **More than 80% of the samples choked** (CHK-12). Sonic flow at the choke is common and wanted in operations, since it "stabilizes well production rate and separation operation conditions" (Guo et al. 2007, §5.2, p. 5/60). The filter would drop high-pressure wells and wells at low separator pressures, Bjarne's goals 2 and 3.
  - **The QTOT variation** (a coefficient of variation below 0.05). It drops wells whose rate the choke barely changes. That is physics; if it is common, the bean sizes (R3, D-27) are wrong, which R6 checks.

  With the QTOT filter gone, backlog §1.11 (its standard deviation) lapses. v1's other rules go too: dropping samples below 0.1 kg/s (`sol-1`) or 1 kg/s (`nsol-1`), `nsol-1`'s well filter at 7 kg/s, and the discards after 100 or 200 failures, $5 \cdot 500$ attempts or 50 failed solves in a row (SMP-28, SMP-29), which the redraw limits replace (Slots and outcomes). The producibility filter takes their place at the well level, and a small rate stays a sample.

### Slots and outcomes

- **Rule** (D-52, ruled 2026-11-02; D-80 to D-82). Each well has 200 slots, and each slot ends with an operating point. Each attempt's outcome is one of:
  - `operating_point`: the stable root (SOL-4, SOL-6); the slot takes it, with a row;
  - `no_operating_point`: the well cannot flow at the attempt's conditions (SOL-5), a physical outcome (next section);
  - `solver_failure`: an exception, an invalid row (negative rates, fractions outside [0, 1]), or a root set the search may have missed (next section), left after the CasADi retry (A second backend for solver failures).

  An attempt without an operating point is redrawn: attempt $a + 1$ of the same slot draws from `rng_for(seed, k, j, a + 1, 'v2', 'redraw')`. Every redrawn attempt goes to the redraw log with its inputs, its outcome and its reason (The generation log), and the log is published with the dataset (`datasets-draft.md`).
- **What a redraw draws** (D-80). In `sol-2`, the slot's whole draw again: the choke, the lift gas, the separator pressure and the drift (D-37 to D-40), so the samples follow the drawn distribution conditioned on an operating point. In `nsol-2`, the slot keeps its date and the reservoir's state there, and draws its controls and separator pressure again (Redraws in `nsol-2`).
- **Both outcomes are redrawn** (D-82). A `no_operating_point` is redrawn at once. A `solver_failure` first gets the CasADi retry with the same inputs, and is redrawn only if that fails too. The two are counted apart, since the first conditions the data on the physics and the second on the solver.
- **Passes** (D-80). A well's slots are solved in passes: pass 0 solves attempt 0 of every slot, and pass $a$ attempt $a$ of each slot still without an operating point. A slot's draws therefore do not depend on the other slots' outcomes, and each pass can be solved as one batch. The CasADi retries of a pass follow its Rust solves, in the order of $j$.
- **Limits** (D-81). At most 50 attempts per slot; at most 400 redraws per well, so that at least a third of a kept well's draws flow; and at most 10 solver failures per well after the retry. A well that reaches a limit is dropped: it goes to the well log as `slot_limit`, `redraw_limit` or `solver_failure_limit`, with every attempt it made in the redraw log. The generator draws wells $k$ = 0, 1, 2, … and keeps the first 10,000 it accepts (SMP-31), so the next drawn well takes a dropped well's place. R6 reports the drops by class; above 1% of the wells in any class, the draws or the limits are revisited before R11.
- **Alternative** (R4's draft, reversed by the ruling). 200 fixed slots, nothing redrawn, the slots without an operating point kept in the samples table with a status and empty outputs. The ruling keeps the samples to operating points; the published redraw log keeps what the fixed slots would have shown: where each well cannot flow.

### Redraws in `nsol-2`

- **Rule** (D-83). A slot is a point in the well's life, so a redraw keeps its date $t_j$ and everything that follows from it: the reservoir's pressure, the GOR and the water with the date's drift, $h$, $\varepsilon$, the productivity, $p_{s,\text{nom}}(t_j)$ and whether lift is installed. It draws again:
  - the controls, each by D-47's set-point change applied to the slot's planned value: a step of $U(-0.2, 0.2)$ of its range, reflected into the range, with probability 0.6, else a new draw (the choke 1 with probability 0.05, else $U(0.05, 1)$; the lift gas 0 with probability 0.10, else $U(0, w_{lg,\max})$). Both controls change, the lift gas only where lift is installed. In a test the planned value is the sweep's point;
  - the separator pressure around $p_{s,\text{nom}}(t_j)$, as D-39.

  The redrawn controls hold for that slot only. The later slots keep their planned set points, so the control history does not depend on a solve and stays open loop (the Goals).
- **Why a step from the planned value.** It keeps a redrawn slot near the history around it, so the series stays piecewise constant apart from its redrawn points. The new draws, 40% of the attempts, reach the whole range where the planned value lies in a region in which the well cannot flow.
- **A failed point in time.** A slot that reaches the limit of 50 attempts is a date at which the well flowed at none of its redrawn controls. Since the life ends at the last scan point that passes and lift starts before natural flow fails (D-84), the best rate passes at the scan points on both sides of every date, so a failed date means the well died between two scan points or flows only in a narrow band of controls. The well is dropped as `dead_date` and logged. R6 reports the share; above 0.5% of the `nsol-2` wells, a finer life scan is the fix.

### No operating point or solver failure

- **Rule** (D-53). An attempt has `no_operating_point` when:
  - $p_r \le p_s$ (no solve), or
  - the search found no root at all and rejected no sign change: for the Rust core, its count `rejected` is 0. Its count of non-finite scan points, where a march leaves the range of the closures, may be above 0; the log keeps it (D-102, revising D-53, which asked for both counts at 0).

  Every other attempt without an operating point is a `solver_failure`, with its reason: an exception other than `NoOperatingPoint`; an invalid row; a scan with rejected sign changes; or a root set with unstable or indeterminate roots but no stable one.
- **Why non-finite points do not make a failure** (D-102). They are routine: 14% of the operating points' searches in the draft's pilots had them. Of the attempts with non-finite points, no root and no rejected sign change in the draft's pilots, the CasADi retry ran on 72 and found no root in 70; D-53 counted them all as solver failures and dropped 3 tail wells at the solver-failure limit for them alone.
- **Why a set of unstable roots alone is a failure.** Along the curve of SOL-3, $R(p_0) = w_m - w_c > 0$ at a low $p_0$, where the tubing's pressure falls below $p_s$ (CHK-11). Going up in $p_0$, each crossing from positive to negative is a stable root ($s < 0$) and each crossing back an unstable one. Whatever the sign of $R$ near $p_r$, a continuous $R$ has at least as many stable roots as unstable ones. A root set with an unstable root and no stable one has therefore missed a root, or $R$ is not continuous; either way the outcome is not the well's.
- **Sources.** `specs/model/solution.md` (SOL-2, SOL-3, SOL-5, SOL-7); the Rust core's search counts (`rust/src/march.rs`, `Counts`).

### A second backend for solver failures

- **Rule** (D-54, amended by D-82). Every `solver_failure` of the Rust core is solved again with the CasADi backend, with the same inputs and the nearest operating point of the same well as its extra start (Starts). If it finds an operating point, the attempt takes it, and the redraw log records the attempt as `recovered`, with the backend and the start. Otherwise the attempt is a `solver_failure` and the slot is redrawn. A `no_operating_point` is not retried.
- **Why.** The retry changes no input. Without it every Rust failure would be redrawn, which conditions the samples on where the Rust core fails; the retry shrinks that share, at about 10 to 30 times a Rust solve's cost on a small share of attempts. It is a fallback of the generator and needs its measured gain (constitution, principle 7): R6 counts the attempts it recovers.
- **Measured** (R6). Over four pilot runs (about 222,000 attempts) the retry ran 116 times, at 15 to 28 Rust solves each and 700 s of CPU in all, and recovered 2 attempts, both of a kind that D-102 now counts as no operating point; under R6's rules it recovered none of 23. No measurable gain: R6 recommends dropping it and keeping solver failures as logged redraws (`r6-measurements.md`, The CasADi retry); Bjarne decides.

### Wells whose best-rate solve fails

- **Rule** (D-55). If no solve of the best rate gives an operating point and at least one is a `solver_failure` after the retry, the well is dropped as `best_rate_solver_failure` and logged; a well whose best-rate solves are all `no_operating_point` is not producible (Producibility filter). R6 reports the share by fluid class, well type and geometry; above 0.5% of the wells in any class it is a solver finding to fix before R11, not a filter to keep.
- **Alternative.** Keep the well, solve its slots without an extra start, and judge its producibility from the largest oil equivalent among its slots at $u \ge 0.9$. It keeps the wells, but replaces the filter's definition for some of them.

### Starts

- **Rule** (D-56). The Rust core takes no start: it scans $R(p_0)$ over $(p_s, p_r)$. The starts matter to the CasADi retries and comparison runs, where the start is an extra start of the search (`x_guess`), never the only one:
  - **`sol-2`:** the operating point nearest in $(u,\ w_{lg}/w_{lg,\max},\ p_s/p_{s,\text{nom}})$ among the best-rate solves, the Rust core's operating points of this pass and the earlier ones, and the retries of this pass at smaller $j$; the retries run in the order of $j$ (Slots and outcomes, Passes), so that a rerun gives the same result;
  - **`nsol-2`:** among the same operating points, the one nearest in time to the slot's date (in a sweep, the previous sweep point where it has one).
- **Sources.** SMP-28 and SMP-29's starts (v1's nominal root at $u = 0.5$, and `nsol-1`'s pool of roots); `specs/model/solution.md`, Informative: `develop`'s root search. A start can only add roots to the set; it can change the operating point only where SOL-6's case of several stable roots arises, which the log flags.

### The generation log

Written for every well drawn, accepted or not, and for every attempt, so that R6 and a user can see what the filters, the failures and the redraws removed (D-85):

- **per well** (the well log): the seed and index; every drawn input and label; the outcome (`accepted`, `not_producible`, `best_rate_solver_failure`, `invalid_input` for a well that fails `configurations.check` or a table that does not cover its $p_r$ and $T_r$, `slot_limit`, `redraw_limit` or `solver_failure_limit` for a well dropped at a redraw limit, and for `nsol-2` `dead_date` and `short_life`); the best rate, its lift-gas rate and its solves' outcomes; for `nsol-2`, $T_\text{life}$, the life scan's best rates, $T_\text{end}$, the lift's start and reason; the counts of attempts by outcome and reason, of redraws, and of CasADi retries and what they recovered; the wall time;
- **per attempt:** the slot and the attempt; every input; the outcome and reason; the backend, and a retry's start; the search's counts (marches, non-finite scan points, rejected sign changes); the number of roots and of stable roots (SOL-6's flag); the wall time.

The published part is the well log and the redraw log: every attempt that was redrawn, recovered by the retry, or made by a dropped well (`datasets-draft.md`). The search counts and wall times of the attempts that gave a sample at once stay with the run, for R6 and R11.

### How the rules meet acceptance criterion 4

The ruling on D-52 makes the samples the draws at which a well flows. That conditioning shifts the accepted distributions away from the drawn ones; the rules below keep the shift physical, bounded and measured.

- **The draws are made before any solve, and kept.** A slot's first attempt comes from its own generator, and each redraw from the slot's and the attempt's (D-80); no outcome changes a draw that was made. The inputs over all attempts, the samples together with the redraw log, have exactly the drafted distributions.
- **The samples are the draws conditioned on an operating point.** A redraw draws the same distribution again (`sol-2`), or the same controls' distribution at the same date (`nsol-2`). The accepted inputs therefore differ from the drawn ones only where wells cannot flow, which is physics, and where the solver failed, which is not. The redraw log splits the difference into the two, by input.
- **The shift is measured and published.** R6 measures it (What R6 must measure). The redraw log is published with each dataset, so a user can measure it too, reweight the samples, or use the attempts without an operating point, which mark where each well cannot flow.
- **Solver failures are bounded.** Every Rust failure gets the CasADi retry before it is redrawn (D-82). R6 reports the share of attempts redrawn for a solver failure by class and by decile of choke, lift gas and $p_r/p_i$, and the draft sets a target: below 0.5% of the attempts and of the wells in every class. A share above it is a solver finding (`AGENTS.md`: "a fall in the number of operating points found is a solver finding"), fixed before R11.
- **The well filters are deliberate and logged.** The producibility filter is acceptance criterion 2. The redraw limits (D-81) drop wells that flow in less than a third of their draws or that fail the solver often; they cost criterion 4 a little, and the target of 1% of the wells in every class bounds that cost. Every dropped well is in the well log with its inputs, and R6 compares the wells' marginals before and after.
- **Two `nsol-2` inputs follow a solve, both through the life scan.** The end of a life $T_\text{end}$, which spaces the dates, and the start of lift by rule 2 of D-49 come from the life scan's best rates. Every draw (the trends' parameters, the set points, the sweeps, the drift) is made before any solve; only the dates and the lift's start move with the scan, and the log records both. A redraw changes one slot's controls and not the planned history, so the set points never follow a solve.
- **Against v1:** v1 redrew its failed draws too, 3.5% of them at small openings (L6), but published no record of them, so the shift cannot be measured from its data. Here every redraw is logged and published.

## The samples and the four tasks

| Task | What it needs from the samples | Where the draft gives it | What stays thin |
|---|---|---|---|
| Virtual flow metering | Measured inputs (pressures, temperatures, choke) spread widely within each well, the rates as targets, and drifting wells for `nsol-2` | A choke close to uniform; the separator's ±10%; the drift; `nsol-2`'s trends, which change the rate for the same controls | Measurement noise is not in the samples (a task adds it) |
| Surrogate models of the simulator | The whole input space, the boundary where wells stop flowing, and both sides of choked flow | Uniform controls; the redraw log's attempts without an operating point, published with their inputs; the tail's flat well draws | The extremes R3 excludes (D-35); the samples near the boundary, which the redraws thin |
| Learning across wells | Many wells that differ, with enough samples each to tell them apart | 10,000 wells, 200 operating points each; every label stored, so that a model can group or weight the wells | Weak wells' small openings, which the redraws thin, and the wells dropped at the redraw limits |
| Production optimisation (choke and gas lift) | Each lifted well's lift performance curve to past its optimum, its rate at fully open, and where it dies without lift | $w_{lg}$ uniform to the capacity, 10% of the draws without lift gas, 5% at fully open; `nsol-2`'s lift sweeps; the economic floor; the redraw log, where a well dies without lift | Wells sharing lift gas or a flowline: nothing downstream of the choke is modelled |

- **Choke and lift gas.** `sol-2` covers $u$ from 0.05 to 1 and $w_{lg}$ from 0 to the capacity in every lifted well. `nsol-2` covers them by its set points and, systematically, by its sweeps: 8 tests per life, each spanning its control's whole range.
- **Flow regimes and choked flow.** The slots reach choked flow at small openings and low separator pressures (CHK-12: $p_s \le 0.5445\thinspace p_N$), and unchoked flow at large openings and high separator pressures, so most wells have both. The regimes follow the wells: bubbly flow at the bottom of undersaturated oils, slug and churn flow in oil wells, annular flow in gas wells, and churn or annular flow above a lift valve at high lift-gas rates. R6 measures the shares at the bottom and the top, by class, before and after the producibility filter.
- **Within and across wells.** A model that learns one well needs its controls spread; one that learns across wells needs the wells spread. The slots spread the first, R3's draws the second.

## Where realism and coverage conflict: the samples

- **Choke positions.** An operating well sits at a few set points, often fully open late in life; `sol-2` spreads $u$ uniformly, as Bjarne asked. `nsol-2` keeps realistic set points (2.2% of its operating samples fully open) but adds sweeps.
- **Small openings.** $u$ starts at v1's 0.05 for coverage. No public source says how often wells produce below a tenth of their bean, and some wells have no operating point there; those attempts are redrawn, so the samples are thinner there than the draws, and the redraw log keeps them (D-52, ruled).
- **Lift gas.** Operators run near an allocated rate; `sol-2` spreads $w_{lg}$ uniformly to the capacity, so some samples lie past the optimum and some at rates where real wells head (Asheim 1988, p. 1452), which a steady-state model does not show.
- **Lift off.** 10% of a lifted well's `sol-2` draws have no lift gas; a well that needs lift dies there. That is realistic. Those attempts are redrawn, so such a well's samples hold no lift-off points, and the redraw log holds them.
- **Drift.** ±5% on the GOR and the water, drawn afresh for each sample, may exceed what a well's ratios do between neighbouring dates; no public source is in hand, and the Goals keep it for variation.
- **Test periods.** Eight sweeps in a life, 20% of the slots, are a design choice for coverage: no public schedule says how often wells get multi-rate tests. They give the excitation that makes a well's response identifiable.
- **Late life.** The economic floor ends the lives of weak wells, so late, low-rate conditions are thinner than in a field's own history, as the Goals want (criterion 2) with the margin of half the floor.

## What R6 must measure

*Measured by R6 on pilots of 400 `sol-2` and 100 `nsol-2` wells: `r6-measurements.md`.*

- **Outcomes:** the shares of attempts with `operating_point`, `no_operating_point` and `solver_failure`, by dataset, class, well type, geometry, lift and tail, and by decile of $u$, $w_{lg}/w_{lg,\max}$ and $p_r/p_i$; the reasons for failure; what the CasADi retries recover and cost (D-54's gain).
- **Redraws** (D-80 to D-83): the redraw share, redraws per sample, by class, well type, geometry, lift and tail, and the share of attempts without an operating point by decile of the attempt's $u$, $w_{lg}/w_{lg,\max}$ and $p_r/p_i$, each split into `no_operating_point` and `solver_failure`; the distributions of redraws per well and per slot; the wells dropped at each limit (D-81) and as `dead_date` or `short_life`, by class, against the target of 1%; in `nsol-2`, the redrawn slots by date and by kind (operating or test), and how far the redrawn controls lie from the planned ones.
- **Filters:** the share of wells dropped as not producible, by class and well type, and the marginals of the wells' inputs before and after; the share dropped for a best-rate solver failure; the `nsol-2` lives cut by the floor, and $T_\text{end}/T_\text{life}$.
- **Criterion 4:** the inputs' marginals over all draws (the samples with the redraw log) against those over the accepted samples, per input and class, with the difference split into `no_operating_point` and `solver_failure`; and the wells' marginals over all drawn wells against the accepted ones, with the difference split by the reason a well was dropped.
- **Coverage:** the share of choked samples by well and its spread over wells; the flow regimes at the bottom and the top; the wells whose QTOT varies by less than 5% (the dropped filter's case, a check on R3's beans); the per-well share of draws with an operating point.
- **`nsol-2`:** the lift's start by rule (static check, end of natural flow, drawn time); the solved rates over the life against Arps's shapes; the consistency of the per-sample stage states with `sol-2`'s wells of that stage, which the sanity script checks before solving (below).
- **Cost:** the solves per well (first attempts, redraws, best-rate solves, life scans, retries), and the time per dataset. At the Rust core's 0.15 s per solve, `sol-2`'s 2.03 million solves without redraws take about 85 CPU hours, and `nsol-2`'s about 2.4 million about 100; a redraw share $r$ adds $2r$ million solves to each, and each CasADi retry costs 10 to 30 Rust solves (`datasets-draft.md`, The generation).

The sanity script's check of the `nsol-2` states against R3's: by per-sample stage, $p_r/p_i$ medians of 0.94 (initial, depletion), 0.65 (mid-life, depletion) and 0.45 (late, depletion) against `sol-2`'s 0.99, 0.71 and 0.38, and oils' water cuts of 0.32 (mid-life, pressure support) and 0.68 (late) against 0.36 and 0.77. A depletion well loses more pressure in its first year (P5 0.85) than R3's initial range of 0.97 to 1.0 allows: a finding for R3's initial range, which the checkpoint may widen.

## Bjarne's goals: the samples

| Goal | Where | Draws |
|---|---|---|
| 4. Modest weight at a fully open choke, close to uniform | Choke position; Control history; Test periods | `sol-2` 5.0% at fully open, standard deviation 0.287; `nsol-2` 5.2% |
| 5. Lift gas scaled to the well | Lift-gas rate (R3's capacity); Lift sweeps | $w_{lg}$ uniform to $w_{lg,\max}$, 10% off |
| 2. Separator pressures below 10 bar | Separator pressure; Separator over the life | 16.6% of `sol-2` slots, 17.4% of `nsol-2` slots |
| `sol-2`: drift of ±2% and ±5% | Drift of the reservoir | as decided, with the GOR tied to $p_r$ |
| `nsol-2`: linked trends | Life span and clock; Reservoir pressure; Water; Gas–oil ratio | one clock drives the pressure and the water, and the pressure drives the GOR |
| `nsol-2`: piecewise-constant controls and test sweeps, open loop | Control history; Test periods | 17 choke changes per well; 20% of slots in tests |
| A well's 200 samples span its life | The end of a life | 160 operating samples evenly over $[0, T_\text{end}]$ |
| No wells that would not be produced, with a margin | Producibility filter | half the floor (D-34) |
| 200 operating points per well, every redraw logged (ruled, D-52); criterion 4 | Slots and outcomes; Redraws in `nsol-2`; The generation log | redraws from the slot's and the attempt's own generator; the redraw log published |

## Open issues (R4)

- **Settled for Bjarne:** "200 samples per well" counts operating points (his ruling on D-52, 2026-11-02); R7 revised these rules to match (D-80 to D-85). At the checkpoint: the redraw limits (D-81) and the targets for solver failures (0.5%) and for dropped wells (1%), which are tolerances.
- **For R5:** a composite inflow, so that an `nsol-2` well could change from a straight line to Vogel's curve as its pressure falls (D-45).
- **Done by R6:** everything under What R6 must measure (`r6-measurements.md`). In R6's pilots no attempt of an accepted `sol-2` well was redrawn for a solver failure, and the redraws for no operating point shift no input by more than a KS $D$ of 0.054 (the lift gas). The limits' targets (D-81: 1% per class; D-83: 0.5% `dead_date`) are exceeded in the oil classes of the small pilots (1.4% and 0.9%).
- **Done by R7:** the per-sample columns that `nsol-2` adds, the redraw log and the well log, and their place next to the datasets (`datasets-draft.md`).
- **For R2's second pass:** histories of choke and lift-gas set points (Resource list (R4 additions)); a public shape of the GOR's late fall; per-well water-cut curves offshore.

## Resource list (R4 additions)

| Reference | Decision it bears on | Assumption without it |
|---|---|---|
| Petrobras 3W (CC BY 4.0; R2 lists it, not fetched) | D-47, D-48: control histories of offshore wells: how often and by how much chokes change, and the share of time fully open | own choices for $p_c$ and the step |
| Dake (1978), *Fundamentals of reservoir engineering*, or Craft and Hawkins (1959), *Applied petroleum reservoir engineering* | D-43, D-44: water-cut curves after breakthrough, and the GOR of solution-gas drive with its late fall | Ahmed (2006) and Golan and Whitson's Example 4.11 |
| Fetkovich (1973), SPE 4529 (on R2's list) | D-45: the fall of an oil well's productivity below the bubble point | R3's PI at the current fluid, without the relative permeability's fall |
| Arps (1945), *Trans. AIME* 160, 228–247 | D-41: the decline's forms | Ahmed (2006), Eqs. 16-1 to 16-4 |
| The paper by Bertuzzi et al. that Asheim (1988, ref. 2) cites, on the minimum lift-gas rate | D-38: where low lift-gas rates head | Asheim (1988) |

## References (R4 additions)

R3's references above apply; these are added.

- Ahmed, T. (2006), *Reservoir engineering handbook*, 3rd ed., Gulf Professional Publishing: Eq. 5-28 (p. 309); Eq. 13-8 (p. 861); Eq. 14-10 (p. 935), Eq. 14-22 and Fig. 14-12 (p. 941), pp. 970 and 983; Eqs. 16-1 to 16-4 (pp. 1238–1239).
- Asheim, H. (1988), "Criteria for gas-lift stability", *Journal of Petroleum Technology* 40(11), 1452–1456: p. 1452.
- Goh, K.-C., Moncur, C.E., Van Overschee, P. and Briers, J. (2007), "Production surveillance and optimization with data driven models", IPTC 11647.
- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall: p. 385; Example 4.11 (pp. 454–456); §5.3 (p. 538).
- Guo, B., Lyons, W.C. and Ghalambor, A. (2007), *Petroleum production engineering*, Gulf Professional Publishing: §5.1 and §5.2 (p. 5/60).
- Miller, R., Sorrell, S. and Speirs, J. (2009), *UKERC review of evidence for global oil depletion, Technical report 4: Decline rates and depletion rates*, UK Energy Research Centre: Tables 4.2 to 4.4, §3.
