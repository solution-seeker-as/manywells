# Sampler redesign — draft procedure: the well draws

*Step R3 of `plans/sampler-redesign-plan.md`, 2026-10-30. A draft for Bjarne's checkpoint (R8), in the form of `specs/sampling.md`, without SMP IDs. Plans are drafts, not specs. The samples within a well are R4's and the dataset design R7's; both will be added to this file or sit beside it.*

This file proposes how the v2 sampler draws one well in the `v2.0.0` configuration: an ordered sequence of seeded draws, each conditioned only on earlier ones. It keeps three things apart:

- **Decided:** the Goals of the plan (Bjarne, 2026-10-30) and his ruling of the same day on horizontal drains: a well with a horizontal drain is modelled from the heel up, its bottom section at most about 75° from vertical (set here between 70° and 80°), its inflow stands for the drain, and the shares of the geometries wait for more examples of real geometries.
- **Proposed:** every distribution, parameter and rule below. Each rests on a working assumption in `decisions.md` (D-1 to D-36), which Bjarne rules on at the checkpoint.
- **Pending:** the shares of the geometry families, which wait for the geometry evidence (`sources-geometry.md`, gathered in parallel).

Sources are cited as [RF §n] for `sources-reservoir-fluids.md` and [WP §n] for `sources-wells-production.md`, with the primary reference and its page, table or figure. The draws were checked without solving by `plans/evidence/sampler_draft_draws.py` on 2000 wells, seed 2026; its numbers are quoted as *draws*, for the 1690 core wells unless said otherwise (Checks).

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
geometry ── family (well type), survey to TVD L                    │
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
| geometry | family, survey | well type, $L$ |
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

Every categorical choice is stored with the well. All are drawn or set before the well is built; none depends on a solve.

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
| `geometry_class` | `vertical`, `build_and_hold`, `s_shaped`, `high_angle` | Geometry |
| `pvt_model` | `black_oil`, `table` | Fluid class (Mapping) |
| `inflow` | `vogel`, `productivity_index`, `gas_rate` | Inflow |
| `tubing` | the nominal size, `2-7/8` to `9-5/8` | Tubing |
| `choke_model` | `simpson` | Choke |
| `choke_profile` | `linear`, `sigmoid`, `convex`, `concave` | Choke |
| `lift` | `natural`, `gas_lift` | Gas lift: assignment |
| `lift_reason` | `drawn`, `needs_lift` | Gas lift: assignment |

## Well draws

### Coverage tail

- **Draw.** `coverage_tail` is `tail` with probability 0.15, else `core` (D-6).
- **Effect.** In a tail well every draw below takes its tail form: categorical shares become equal (well type, fluid type, drive, life stage, geometry), and continuous draws become (log-)uniform over the draw's bounds. The links stay: pressure still follows depth and brine, the bubble point still follows the fluid, tubing still follows the rate. So a tail well is less likely, not implausible.
- **Sources.** The Goals set 10% to 20% as the starting point; 15% is its middle. No public source sets a share; R6 measures what the tail adds to the coverage (Where realism and coverage conflict).
- **Validity.** The tail keeps every bound of the core, so it leaves no range the core does not.
- **Draws.** 15.5% of the 2000 wells.

### Well type

- **Draw.** $\text{Cat}$(platform 0.55, subsea 0.35, onshore 0.10); tail: equal shares (D-5).
- **Depends on.** Nothing.
- **Sources.** [WP §1]: in deep water in the Gulf of Mexico, 84% of producing wells have dry trees (997 against 191; Liu, Wang and Kaiser 2014); on the Norwegian shelf 41% of the development production wellbores are subsea (Sodir FactMaps, 1688 of 4093). The shelf of the Gulf of Mexico is almost all platform wells. Counted by wells the world is onshore (EIA 2025, p. 3: 918,481 producing wells in the US alone), so the onshore share is the Goals' offshore weighting, not a statistic.
- **Validity.** None; the well type sets surface conditions only.
- **Maps to.** No model input; it conditions the water depth, the surface temperature, the separator, the permeability and the economic floor.

### Fluid class

- **Draw.** $\text{Cat}$(black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20); tail: 0.25 each (D-4). A gas well is `dry_gas` with probability 0.07, else `wet_gas` (tail: 0.25).
- **Classes** by McCain (1990), pp. 151–158 (D-3): black oil with an initial producing gas–oil ratio up to 356 Sm³/Sm³ (2000 scf/STB) and below 45 °API; volatile oil from 356 to 588 Sm³/Sm³ (3300 scf/STB) and at least 40 °API; gas condensate from 588 to 8900 Sm³/Sm³ (50,000 scf/STB); wet gas above 8900 Sm³/Sm³; dry gas with practically no surface liquid, here a condensate yield below 2.8 Sm³ per 10⁶ Sm³ (0.5 STB/MMscf).
- **Sources.** [RF §5, §6]: by wells, the Gulf of Mexico has 47.7% black oil, 1.3% volatile oil, 11.6% gas condensate, 37.1% wet gas and 2.3% dry gas (BOEM sands, 68,738 completions); its deep water 80.6% black oil; Norwegian production wellbores 78.7% oil. The Goals' floor of about 10% per type sets the volatile-oil share and, with the Norwegian shelf, the condensate share. Dry gas is 5.8% of the Gulf's gas wells by wells, 12.5% by sands; 0.07 lies between.
- **Validity.** The classes are chosen so that the black oil (PVT-OIL-4) carries only McCain's black oils and the wet and dry gases, whose liquid is a light stock-tank condensate; volatile oils and condensates go to tables (Mapping).
- **Draws.** Black oil 57.0%, wet gas 17.5%, condensate 14.4%, volatile oil 10.4%, dry gas 0.8% of core wells.

### Drive and life stage

- **Draw.** `drive` is `pressure_support` (natural water drive, or injection) with probability 0.7 for oil classes and 0.4 for gas classes, else `depletion`; tail 0.5. `life_stage` is $\text{Cat}$(initial 0.25, mid-life 0.40, late 0.35); tail equal (D-18).
- **Sources.** [RF §3]: of the Gulf's undersaturated oil sands by wells, 48% have water drive and 32% partial water drive, against 8% depletion and 2.5% solution gas; of its gas sands 45% water drive, 33% partial, 18% depletion. The Norwegian shelf has one water or gas injector to about five producing wellbores. The stage shares follow fields' lives of 10 to 30 years with a plateau in the first third [RF §11; UKERC 2009, Table 4.4]; no public count of wells by stage is in hand.
- **Maps to.** No input directly; it sets the reservoir's state (Reservoir state at the life stage), the separator's stage, the roughness and the heat-transfer coefficient.

### Depth below the mudline

- **Draw.** $z \sim \text{LogN}_ {[600, 6500]}(m, s)$ m by fluid class (D-8):

  | Class | $m$ (m) | $s$ | BOEM P5 / P50 / P95 by wells (m below sea level) |
  |---|---|---|---|
  | black oil | 2674 | 0.317 | 1417 / 2674 / 4026 |
  | volatile oil | 3159 | 0.315 | 1785 / 3159 / 5031 |
  | gas condensate | 3424 | 0.278 | 1916 / 3424 / 4779 |
  | wet gas | 2419 | 0.422 | 1042 / 2419 / 4176 |
  | dry gas | 1141 | 0.586 | 471 / 1141 / 3236 |

  Tail: $\text{LogU}(600, 7000)$.
- **Sources.** [RF §1], BOEM sands, own fit of $m$ and $s$ to P5, P50 and P95. The Gulf's shelf has a median water depth of 34 m [RF §1], so its depths below sea level stand for depths below the mudline. Norwegian production wellbores' final TVDs (oil P50 2567 m, gas/condensate 3925 m) agree within 10% for oil.
- **Validity.** Hasan et al.'s measured depths run from 498 to 9342 m (`validity.md`); 600 m is above the lower end. Truncated.
- **Draws.** P5 / P50 / P95 = 1530 / 2768 / 4831 m.

### Water depth and the well's length

- **Draw** (D-8):
  - platform: with probability 0.85 a shelf platform, $d_w \sim \text{LogN}_ {[15, 400]}(90, 0.6)$ m, else a deep-water dry tree, $\text{LogU}(400, 2000)$ m; tail $\text{LogU}(15, 2000)$;
  - subsea: $d_w \sim \text{LogN}_ {[80, 3000]}(350, 0.8)$ m; tail $\text{LogU}(50, 3000)$;
  - onshore: $d_w = 0$.
- **Length.** $L = z + d_w$ for a platform (the riser through the water is part of the flow path), $L = z$ for subsea and onshore wells.
- **Sources.** [WP §2]: Norwegian platform oil wellbores in 70 / 109 / 175 m of water (P10 / P50 / P90), subsea oil 124 / 329 / 357 m, subsea gas 93 / 298 / 854 m; the Gulf's completed development boreholes: 46% below 60 m, 14% deeper than 500 m. Deep-water dry trees (TLPs, spars) reach about 2000 m in the Gulf (Liu, Wang and Kaiser 2014).
- **Validity.** No correlation; THM-4's linear ambient profile cannot follow a deep water column (Surface temperature).
- **Draws.** Platform P5 / P50 / P95 = 34 / 99 / 1174 m; subsea 120 / 373 / 1409 m; $L$ 1582 / 2885 / 5083 m.

### Geometry

*The shares are pending the geometry evidence. The families, their parameters, the inclination limit and the heel approach are proposed here.*

- **Families** (D-9). Each is a survey of stations $(\text{MD}, \text{TVD})$ from the wellhead to the bottom at TVD $L$, with circular build and drop sections (one station per degree):

  | Family | Shape | Parameters (core) |
  |---|---|---|
  | `vertical` | straight down | – |
  | `build_and_hold` | vertical to the kickoff $K$, build at $b$ to $\theta_h$, hold to $L$ | $\theta_h \sim N_{[5,65]}(35°, 13°)$; $b \sim U(1.5, 3)$ °/30 m; $K \sim U(0.1L, K_\text{max})$, $K_\text{max} = \min(0.7L,\ L - 50\ \text{m} - \Delta\text{TVD}_ \text{build})$ |
  | `s_shaped` | build to $\theta_h$, hold, drop at $d$ to $\theta_e$, then straight to $L$ | $\theta_h \sim U(15°, 45°)$; $b$ as above; $d \sim U(1, 2.5)$ °/30 m; $\theta_e \sim U(0°, 10°)$ over the last $U(100, 400)$ m of TVD; $K$ up to $0.6L$ |
  | `high_angle` | build to $\theta_h$, hold, build at $b_2$ to $\theta_\text{heel}$, a short tangent to the heel at $L$ | $\theta_h \sim U(35°, 65°)$; $\theta_\text{heel} \sim U(\max(60°, \theta_h), 75°)$; $b_2 \sim U(2, 4)$ °/30 m; tangent $U(50, 300)$ m of MD; $K \sim U(0.05L, 0.4L)$ within the fit |

  "Slightly deviated" wells are `build_and_hold` wells with $\theta_h$ below about 20°: about 7% of that family's draws. Tail: $\theta_h \sim U(5°, 70°)$ in `build_and_hold`; the other parameters as in the core.
- **Fit.** If a shape does not fit $L$, the build rate is raised to at most 5 °/30 m, then $\theta_h$ is lowered in steps of 10%; an S or high-angle shape that still does not fit, or a well whose measured depth exceeds 9342 m, is redrawn (draws: 0.8% of the wells redrawn at least once, 3 of 2000 fall back to vertical after 20 tries).
- **Shares.** *Pending the geometry evidence* (`sources-geometry.md`). The sanity script uses placeholders, not proposals: offshore 0.40 / 0.30 / 0.10 / 0.20 and onshore 0.70 / 0.20 / 0.05 / 0.05 for vertical / build-and-hold / S-shaped / high-angle. The public evidence so far [WP §3]: 3.1% of the Gulf's completed development boreholes are straight, 91.0% directional and 5.9% horizontal, with an equivalent tangent inclination of 13° / 36° / 66° (P5 / P50 / P95); Norwegian production wellbores have a median MD/TVD of 1.54 (platform) and 2.04 (subsea), and gas wellbores 1.10 to 1.18. The Goals ask for mostly vertical wells; the ruling of 2026-10-30 left the shares to this evidence.
- **Inclination limit** (D-10). Every cell is at most $\theta_\text{max} = 75°$ from vertical. Taitel and Dukler's stratified–non-stratified transition, scanned over oil and water at 10 to 200 bar in 62 to 157 mm tubing [WP §3], leaves stratified upward flow only below $v_{SL}$ = 0.006 m/s at 10° above horizontal (80° from vertical), below 0.004 m/s in one case at 15° (75°), and nowhere above 0.003 m/s at 20° (70°). Experiments agree: in a 9.72 cm pipe at 2.7 to 14.8 bar, stratified flow had disappeared at +2° (Wilkens, Gopal and Jepson), and Mukherjee and Brill's map has no stratified region in upward flow (Brill and Mukherjee 1999, p. 44). 75° keeps the margin of 70° for liquid rates above about 7 m³/d in 157 mm tubing, and admits the heel sections of the Gulf's horizontal wells, whose equivalent tangent has a median of 67° [WP §3].
- **Heel approach** (Bjarne's ruling). A `high_angle` well stands for a well with a horizontal drain: the model's bottom is the heel, at the reservoir's TVD $L$, and the drain's length enters the inflow (Productivity). The drain itself has no cells, so no cell is horizontal.
- **Validity.** Hasan et al.'s five regimes are fitted to vertical and deviated wells with no range of inclination (`validity.md`, note 9); none of the draws reaches stratified flow's inclinations. Measured depth stays within Hasan et al.'s 9342 m (redrawn).
- **Maps to.** `WellGeometry.from_survey(md, tvd, n_cells=100, D)` (GEO-3, GEO-4), which `configurations.check` accepts in `v2.0.0` for any survey.
- **Draws** (placeholder shares). Deviated families: MD/TVD 1.02 / 1.15 / 1.61; the largest cell inclination 19° / 42° / 73° (P5 / P50 / P95); high-angle wells 61° / 68° / 74°; no cell above 75°.

### Formation water

- **Draw.** Salinity, the mass fraction of dissolved solids, $w_s \sim \text{LogN}_ {[0.01, 0.25]}(0.08, 0.6)$; tail $\text{LogU}(0.005, 0.25)$ (D-11). The water's density at standard conditions is

  $$\rho_w = 999.1 + 692\thinspace w_s + 311\thinspace w_s^2\ \text{kg/m}^3,$$

  a quadratic through McCain's (1990) 999 kg/m³ at no solids, 1047.6 kg/m³ at 6.8 wt% (Example 16-6, p. 449) and about 1200 kg/m³ at 26 wt% (Fig. 16-8), from [RF §9].
- **Sources.** [RF §9]: formation waters run from 10,000 to about 300,000 ppm, usually below 80% of the NaCl limit (Whitson and Brulé 2000, §9.2, Eq. 9.1); seawater is about 35,000 ppm (McCain 1990, p. 438). No public distribution over offshore reservoirs is in hand. The median, 80,000 ppm, gives 1056 kg/m³ and 0.104 bar/m, near the Gulf's normal gradient of 0.465 psi/ft (0.1052 bar/m, which is $\rho_w g$ at 1072 kg/m³ and about 10 wt%).
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
- **Sources.** [RF §2], BOEM sands by wells: the median excess over 0.465 psi/ft is 3, 12, 34, 92, 182 and 294 bar at 0–1, 1–2, 2–3, 3–4, 4–5 and 5–7 km, which is $\lambda$ = 0.05, 0.07, 0.11, 0.22, 0.33 and 0.40 with the lithostatic gradient above; the share above 0.6 psi/ft, which is $\lambda \approx 0.25$ at every depth, is 0.6%, 4.6%, 17%, 44%, 67% and 91%. $\lambda_{50}$ and the spread 0.9 are an own fit to these bins. Globally 16.2% of the reservoirs of large fields lie above SG 1.3 and 6.0% below SG 0.8 (Nadeau et al. 2023, Fig. 20, counted by reservoirs, D-2); 3.1% of the Gulf's wells lie below 0.465 psi/ft. The lithostatic gradient of 1 psi/ft is a textbook rule with no source in hand (Resource list).
- **Validity.** Hasan et al.'s bottomhole pressures run to 949 bar; 950 bar is their end. Beggs–Robinson's viscosity is fitted to 363 bar and Lee–Gonzalez–Eakin to 552 bar: accepted, the model extrapolates (`validity.md`).
- **Maps to.** No input directly; $p_r$ follows from it (Reservoir state).
- **Draws.** $p_i$ 191 / 363 / 756 bar; gradient below sea level 0.103 / 0.118 / 0.159 bar/m (BOEM by wells: 0.106 / 0.119 / 0.163); 20.4% above 0.6 psi/ft (BOEM by wells: 23%); medians by class 0.117 (black oil), 0.122 (volatile oil), 0.125 (condensate), 0.117 (wet gas) and 0.109 bar/m (dry gas), against BOEM's 0.116, 0.129, 0.130, 0.119 and 0.111. The bound on 950 bar cuts into the central 90% of the draw in 6.8% of the wells.

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
- **Sources.** [RF §4]: $(T_i - 20\ °\text{C})/z$ = 21.5 / 24.2 / 28.9 K/km on the Gulf's shelf (P5 / P50 / P95) and 13.1 / 18.6 / 28.3 K/km below the mudline in its deep water (BOEM sands); 29.6 / 36.5 / 42.8 K/km in the North Sea (P10 / P50 / P90, uncorrected logging temperatures, Sodir); 14 to 36 K/km in the Gulf's Cenozoic reservoirs (Nadeau et al. 2023, Fig. 7A). One log-normal spans the shelf and the North Sea; R2's mixture of three basins (D-13) would add modes that no per-well statistic supports.
- **Validity.** Valkó and McCain's oils run from 289 to 445 K (Table 2.1); Beggs–Robinson to 419 K and Lee–Gonzalez–Eakin to 444 K: accepted above.
- **Maps to.** `BoundaryConditions(T_r=…)`.
- **Draws.** $G$ 17.8 / 27.2 / 39.4 K/km; over the modelled well 15.9 / 26.2 / 38.6 K/km; $T_r$ 316 / 358 / 424 K. Medians by class: 354 K (black oil), 367 K (volatile oil), 378 K (condensate), 354 K (wet gas), 324 K (dry gas), against BOEM's 354, 361, 372, 354 and 320 K. 17.1% lie above 395 K (122 °C), against 12.9% of the reservoirs of large fields (Nadeau et al. 2023, Table 1). $G_\text{lo}$ binds in 3.1% of the wells, $G_\text{hi}$ in 13.2% (deep, hot wells).

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

- **Sources.** [RF §7]: the Gulf's oils, 25 / 33 / 39 °API by wells (P5 / P50 / P95), 0.7% below 22 °API, and gas gravities 0.60 / 0.62 / 0.67 (64% of oil sands lack one); its gas sands 0.58 / 0.60 / 0.64 (P10 / P50 / P90); Norwegian drill-stem tests 51 / 39 / 29 °API and a gas gravity of 0.68 at the median (all fluids); Valkó and McCain's 1745 oils, 6.0 to 63.7 °API and separator gases of 0.555 to 1.685 (Table 2.1). Condensates' stock-tank liquids are 40 to 60 °API (McCain 1990, pp. 155–156; Whitson and Brulé 2000, §2.5). The example fluids of Whitson and Brulé (2000, Table 2.1) give 0.63 (black oil), 0.70 (volatile oil and condensate) and 0.61 (wet gas).
- **Validity.** The black oil refuses API gravities outside 16 to 58 (decision 2): all draws lie inside, so heavy oils below 16 °API are not drawn (D-35). Sutton's gas gravities start at 0.571 (`validity.md`, note 6): the gas classes are truncated there; no draw lies below. Truncated.
- **Maps to.** $\rho_o$ from the API gravity (PVT-OIL-2) and $\rho_{g,\text{sc}}$ from $\gamma_g$ (PVT-GAS-6), as `BlackOil(rho_o=…, rho_g=…)` or a table's stock-tank densities.
- **Draws.** Black oil 25.4 / 32.7 / 40.3 °API; $\gamma_g$ 0.59 / 0.67 / 0.76.

### Saturation pressure and the initial gas–oil ratio

- **Draw** (D-17). For oils and condensates, the saturation pressure (bubble or dew point) at $T_r$ is $p_\text{sat} = p_i\thinspace(1 - x)$: $x = 0$ (saturated at discovery) with probability 0.20 for black oils, 0.40 for volatile oils and 0.30 for condensates, else $x \sim \text{LogN}_ {[0.005, 0.85]}(0.10, 1.2)$ for black oils and $(0.05, 1.2)$ for the others; tail $x \sim U(0, 0.85)$. Then the initial gas–oil ratio:
  - **black oil:** $R_{si} = R_{so}(p_\text{sat}, T_r)$ by the model's Vazquez–Beggs (PVT-OIL-6) with the well's API, $\gamma_g$ and separator. If it exceeds McCain's 356 Sm³/Sm³, $p_\text{sat}$ is lowered to where $R_{so} = 356$ (draws: 3.9% of wells), so a deep black oil is more undersaturated;
  - **volatile oil:** $R_{si} \sim \text{LogU}(356, 588)$ Sm³/Sm³;
  - **gas condensate:** $R_{si} \sim \text{LogN}_ {[588, 8900]}\big(2000\thinspace e^{-0.3\thinspace(z/1000\ \text{m} - 3.4)},\ 0.6\big)$ Sm³/Sm³, richer with depth; tail $\text{LogU}(588, 8900)$;
  - **wet gas:** the condensate yield $\text{LogN}_ {[2.8, 112]}(30, 1.2)$ Sm³ per 10⁶ Sm³, $R_{si} = 10^6$/yield; tail $\text{LogU}(2.8, 112)$; **dry gas:** yield $\text{LogU}(0.5, 2.8)$.
- **Sources.** [RF §7]: Standing's bubble point from the Gulf's sands gives $p_i - p_b$ = −50 / −3 / 30 / 73 / 161 bar (P10 / P25 / P50 / P75 / P90) by wells, and 21% of its oil wells are in saturated sands; $R_{si}$ = 78 / 151 / 284 Sm³/Sm³ by wells, with rank correlations of 0.62 with $p_i$, 0.56 with $T_i$ and 0.60 with depth; condensate yields rise with depth (0.69) and the Gulf's gas sands have yields of 0.4 / 38 / 215 Sm³ per 10⁶ Sm³ (P10 / P50 / P90, all gas sands, condensates included). Norwegian condensate fields produce at 1543 / 4871 / 9564 Sm³/Sm³ late in life (Sodir). The example condensate of Whitson and Brulé (2000, Table 2.1) has 971 Sm³/Sm³.
- **Validity.** Vazquez–Beggs's $R_{so}$ stays inside Beggs and Robinson's 3.6 to 369 Sm³/Sm³ for black oils. The gases' ratios lie far above the black oil's range: accepted (decision 4), since the liquid's share of the flow is small.
- **Maps to.** `FluidModel(gor=R_go)` at the life stage (next draw). The black oil's bubble point is then $p_\text{sat}$ at $T_r$, because $R_{si}$ is its own $R_{so}$ there (PVT-OIL-15); no `p_bubble` is passed. For a volatile oil or condensate, the table's saturation pressure at $T_r$ is to be $p_\text{sat}$ (R5).
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
- **Sources.** [RF §3, §9, §11]: depletion-drive gas sands in the Gulf end at $p/Z$ of 0.28 / 0.49 / 0.88 of initial (P10 / P50 / P90, from their recovery factors); a subsea oil well's average pressure fell from 276 to 159 bar under solution-gas drive, its producing GOR rose from 107 to a peak of 890 Sm³/Sm³ (8.3 times) and then fell (Golan and Whitson 1991, Example 4.11, Table E4.11a, p. 454); Norwegian oil fields' water cut is 0.22 at 0 to 5 years on production, 0.38 at 5 to 10 and 0.75 at 10 to 20 (medians, by fields), and gas fields' WGR 3.4 / 43 / 196 (P10 / P50 / P90). The linear rise of the GOR and its factors $k_g$ and the condensate's are own choices that bracket the example (D-19); no statistic is in hand.
- **Validity.** Hasan et al.'s water cuts run to 98% (`validity.md`): the core's oils stop at 0.95, the tail's at 0.98. Vogel's inflow is for solution-gas drive (note 10; Inflow). `FluidModel` takes a water–liquid ratio below 1, which every draw is.
- **Maps to.** `BoundaryConditions(p_r=p_r)` as the well's nominal reservoir pressure, around which R4 drifts each sample; `FluidModel(gor=R_go, wlr=…)`.
- **Draws.** $r_p$ 0.36 / 0.90 / 0.995; $p_r$ 115 / 297 / 658 bar; oils' water cut 0 / 0.25 / 0.87 (late, pressure support: median 0.77); gas WGR 3.0 / 19 / 222. Below hydrostatic: 46.7% of core wells, 30.1% by more than 10% of it; by state, 5% of initial wells, 77% of mid-life and 99% of late depletion wells, and 36% and 56% of mid-life and late wells with pressure support.

### Productivity

- **Draw** (D-20, D-21). Per well, independent of the fluid:
  - permeability $k$: offshore $\text{LogN}_ {[0.5, 3000]}(50, 1.5)$ mD, onshore $\text{LogN}_ {[0.05, 1000]}(10, 1.5)$ mD; tail $\text{LogU}$ over the bounds;
  - net pay $h_n \sim \text{LogN}_ {[3, 100]}(15, 0.8)$ m; tail $\text{LogU}(3, 100)$;
  - skin $s \sim N_{[-3, 15]}(2, 3)$; tail $U(-3, 15)$;
  - drainage: $\ln(r_e/r_w) \sim U(7, 9)$, and $U(8.5, 9.5)$ for a high-angle well, with $r_w$ = 0.108 m (8.5 in hole);
  - the design drawdown $\delta_d \sim U(0.1, 0.4)$ of $p_r$, for the tubing and the lift (below).
- **Oil wells.** The radial productivity index at the life stage's fluid, $J_V = 2\pi k h_n/\big(\mu_o B_o(\ln(r_e/r_w) - 0.75 + s)\big)$ in Sm³/(d bar) of stock-tank liquid, with $\mu_o$ and $B_o$ from the model's black oil at $(p_r, T_r)$ and $R_{go}$. Then $J = f\thinspace J_V$ with the geometry's factor $f$:
  - `high_angle`: Joshi's horizontal-well index (Economides, Hill and Ehlig-Economides 1994, Eqs. 2-46 to 2-48) for a drain of $L_d \sim U(300, \min(1000, 1.6\thinspace r_e))$ m with anisotropy $I_\text{ani} \sim \text{LogU}(1, 3)$, over $J_V$, truncated at 6;
  - `build_and_hold`, `s_shaped`: $f = (\cos\theta_\text{end})^{-1/2}$ for the slant of the last section, a placeholder until Cinco-Ley's slant skin (Economides et al. 1994, §5-4) is read;
  - `vertical`: $f = 1$.
- **Gas and condensate wells.** Forchheimer's $p_r^2 - p_{wf}^2 = a\thinspace q + b\thinspace q^2$ (Economides et al. 1994, Eqs. 4-32 and 4-33), with $a = p_\text{sc}T_r\mu_g Z(\ln(r_e/r_w) - 0.75 + s)/(\pi k h_n T_\text{sc})$ from the model's $Z$ and $\mu_g$ at $(p_r, T_r)$, and $b$ from the non-Darcy coefficient $D \sim \text{LogU}(2\cdot10^{-4}, 2\cdot10^{-3})$ d/Mscf. A high-angle gas well multiplies $k h_n$ by $U(1.5, 4)$, a placeholder until Economides et al.'s horizontal gas well (§4-6) is read; a slanted one by $(\cos\theta_\text{end})^{-1/2}$.
- **Sources.** [WP §4]: permeability medians of 10 to 1000 mD by porosity, 47% of siliciclastic samples above 100 mD (Ehrenberg and Nadeau 2005, Table 1, p. 437; reservoirs, not wells; the median of 50 mD is lower because a well's average includes the poorer layers, an assumption); continuous gas lift needs a PI of about 0.5 bbl/(d psi), 1.15 Sm³/(d bar), or more (Guo et al. 2007, §13.1). Golan and Whitson (1991): $\ln(r_e/r_w) - 0.75 = 7$ in their examples (Examples 2.4 and 2.6, pp. 137–145); a Kansas gas well with $kh$ = 800 md ft, $s$ = 1.89 and $D = 2.9\cdot10^{-4}$ d/Mscf, AOF 15.8 MMscf/d, $n$ = 0.766 (Example 2.4); a Nebraska well with $k$ = 20 md, 22 ft and $D = 1.5\cdot10^{-3}$ d/Mscf (Example 2.6). Economides et al. (1994): $D$ "frequently of the order of $10^{-3}$" d/Mscf and $\ln(r_d/r_w)$ usually 7 to 9 (§4-4, p. 71); $n$ between 0.5 and 1 (Eq. 4-26, p. 69); horizontal-to-vertical PI ratios of 2 to 5 for a 2000 ft drain in their Example 2-10 (Fig. 2-10, p. 35). Golan and Whitson (1991, Preface, pp. vii–viii): a good Oklahoma oil well makes 10 to 100 bbl/d, a Middle East one several hundred to several thousand, an offshore Norwegian one at least 10,000, and "a strong gas well will never produce to capacity because of equipment limitations".
- **Validity.** No correlation range: the PI and Forchheimer's law are definitions with parameters. Vogel's range is the inflow's (next draw).
- **Draws.** $k$ 3.0 / 43 / 472 mD; oils' $J$ 0.32 / 6.6 / 134 Sm³/(d bar); high-angle oils' $J_H/J_V$ 1.8 / 3.9 / 6.0; the design liquid rate $J\delta_d p_r$ 33 / 113 / 451 / 1854 / 6411 Sm³/d (P10 / P25 / P50 / P75 / P90); gas wells' AOF 0.12 / 1.5 / 8.1 million Sm³/d (P5 / P50 / P95) and design rate 0.13 / 0.87 / 3.2 million Sm³/d (P10 / P50 / P90).

### Inflow

- **Draw** (D-22, D-23). For oils, `vogel` (INF-1) if $p_\text{sat} \ge 0.8\thinspace p_r$ and the water cut is below 0.6, else `productivity_index` (INF-2): Vogel where most of the drawdown lies below the bubble point, the straight line where the oil is strongly undersaturated or water dominates. Gas and condensate wells take `gas_rate`, the new inflow of R5.
- **Sources.** Vogel (1968) for solution-gas drive below the bubble point, not where oil, gas and water all flow (`validity.md`, note 10). The 0.8 is an own choice.
- **Validity.** Vogel's rule as above; the well carries the label, so that a user can drop or reweight.
- **Maps to.** `ProductivityIndex(k_l=J ρ_l,sc / 86400)` in kg/(s bar), with $\rho_{l,\text{sc}}$ the stock-tank liquid's density at the water–liquid ratio; `Vogel(w_l_max=k_l p_r / 1.8)`, so that its slope at $p_0 = p_r$ is the same $J$ [WP §4]. The gas-rate inflow takes $(a, b)$ in bar² per (Sm³/d) and per (Sm³/d)², or what R5's spec asks for. Until it exists, the interim mapping is a `ProductivityIndex` whose $k_l$ gives the liquid that the gas carries at the design drawdown: the gas rate becomes linear in the drawdown (Mapping).
- **Draws.** Core oils: 64% Vogel; core wells: Vogel 43%, productivity index 24%, gas rate 33%.

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

  Tail: 0.5 for oils and 0.25 for gas classes. In the core, a drawn well whose pressure-balance point (next draw) lies above 0.3 $L$, and which does not need lift, is left on natural flow: an operator does not lift a strong well (draws: 11% of all wells). No gas–oil ratio limits lift: gassy oil wells are lifted as often as others (Bjarne's goal 5).
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
| `fluid.pvt` | black oil and wet or dry gas: `BlackOil(rho_o, rho_g, gas_law='dak', p_sep, T_sep)`; volatile oil and condensate: R5's `PVTTable` or `PVTTableSet` |
| `fluid` | `FluidModel(pvt, rho_w, gor=R_go, wlr, cp_g, cp_o, cp_w)`, `surface_tension_model='oil'` (the default) |
| `friction` | `RoughnessFriction(roughness=ε)`, Chen |
| `thermal` | `ThermalModel(h)`, every `v2.0.0` term on (the defaults) |
| `slip` | `V2_SLIP`; `profile_scale` is not drawn (1, its default) |
| `inflow` | `Vogel(w_l_max)` or `ProductivityIndex(k_l)`; gas and condensate: R5's gas-rate inflow |
| `choke` | `SimpsonChokeModel(K_c, chk_profile)` |
| `injection_md` | the valve's MD, or `None` |
| `BoundaryConditions` | nominal $p_r$, $p_{s,\text{nom}}$, $T_r$, $T_s$, $T_{lg}$; $u$ and $w_{lg}$ per sample (R4) |

**Interim mapping**, for R6's prototype until R5's features exist: volatile oils and condensates as `BlackOil` with their API gravity, gas gravity, separator and $R_{go}$ (decision 4 accepts any GOR), so that their bubble or dew point is Vazquez–Beggs's rather than $p_\text{sat}$; gas and condensate wells with the `ProductivityIndex` of the Inflow section. Every one of the 2000 drawn wells built in the interim mapping passes `configurations.check(wp, 'v2.0.0')`, and its boundary conditions build. The final mapping passes it once R5's gas-rate inflow is an option that `v2.0.0`'s check accepts (the Goals).

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
| 4. Modest weight at a fully open choke | R4 (the choke position) | – |
| 5. Lift gas scaled to the well, and in gassier wells | Gas lift: capacity scales with the design rate; assignment does not depend on GOR, and gas classes are lifted late in life | core oils 32% lifted; 35% of the oil wells in the top quartile of $R_{go}$; gas classes 2.2% (core), 6.5% (all) |
| Surface temperature by well type | Surface temperature | offshore 275 to 292 K (P5 to P95), onshore $U(268, 303)$ K |
| Injection depth | Gas lift: injection depth | $H/L$ 0.33 / 0.65 / 0.97 |
| Lift-gas temperature | Gas lift: lift-gas temperature | 0.3 to 16.6 K below the ambient at the valve |
| Fluid constants | Formation water ($\rho_w$ from salinity); Heat capacities by fluid; Separator conditions | – |
| A life stage per well | Drive and life stage, with consistent $p_r$, water and GOR | `life_stage`, `life_state` |
| A labelled coverage tail of 10% to 20% | Coverage tail | 15% |
| No implausible wells | the links (depth → pressure, temperature; fluid → GOR, bubble point; rate → tubing, lift) and the bounds (Checks) | no bound violated |
| No wells that would not be produced | The economic floor (R4 applies it) | 2.0% below half the floor before any solve |
| Gas wells through a gas-rate inflow | Inflow; R5 | interim PI |
| Volatile oils and condensates as tables | Mapping; R5 | interim black oil |
| Horizontal drains to the heel, limit about 75° | Geometry | no cell above 75° |

## Where realism and coverage conflict

The datasets serve machine-learning tasks that need the model's range covered, not only the likely wells. A realistic draw leaves these parts thin:

- **Fluid classes.** By wells, volatile oils are 1 to 2% and condensates 3 to 12% of offshore wells [RF §6]. The Goals' floor of about 10% already over-weights them; the tail adds 25% each.
- **Rates.** Productivity spans four decades, but most realistic wells sit in the middle, and the economic floor removes the bottom. Low-rate wells, where the trickle root and liquid loading occur, and tubing-limited high-rate wells are thin. The tail's log-uniform permeability, net pay and skin spread them.
- **Flow regimes.** Oil wells flow mostly in slug and churn flow, gas wells in annular flow, and undersaturated oils in bubbly flow at the bottom. Dispersed bubbly flow needs high liquid rates, and churn flow in gas wells low gas rates; both are thin. R6 measures the regimes at the bottom and the top before and after the filters; the tail's GORs, water cuts and rates widen them.
- **Choked and unchoked flow.** Low separator pressures (goal 2) and high reservoir pressures make most wells choked at small openings; late, depleted wells at high $p_s$ are mostly unchoked. The share of choked samples depends on R4's choke draw and SMP-28's filter on 80% choked, which R4 revisits.
- **Gas lift.** Realistic lift follows life stage and leaves early-life and gas wells mostly unlifted: 2.2% of core gas-class wells and 13% of core initial oil wells. The tail lifts half its oil wells and a quarter of its gas wells, whatever their stage.
- **Inclination.** The limit of 75° removes the near-horizontal cells a heel's last metres would have; the inflow stands for the drain. Nothing in the datasets will show stratified flow, which the model cannot hold.
- **Extremes.** Pressures above 950 bar, temperatures above 450 K, measured depths beyond 9342 m and heavy oils below 16 °API are excluded, in core and tail alike, because the model's correlations or Hasan et al.'s data end there (D-35).

The tail's share (15%) and its flattened shapes are the one lever for all of these. R6 measures each item's coverage with and without the tail, and Bjarne decides at the checkpoint whether 15% is enough.

## Checks a test can run

For R10's tests and R6's measurements, on the core and the tail separately:

- **Labels:** each share within its sampling error of the draft's.
- **Marginals:** the P5 / P50 / P95 above for depth, pressure gradient, temperature by class, API, gas gravity, $R_{si}$, water cut and the gas-lift capacity.
- **Rank correlations** of core black oils, against the Gulf's by wells [RF §1, §7]:

  | Pair | Draws | BOEM |
  |---|---|---|
  | depth – $p_i$ | 0.90 | 0.95 |
  | depth – $T_r$ | 0.75 | 0.82 |
  | $R_{si}$ – $p_i$ | 0.74 | 0.62 |
  | $R_{si}$ – $T_r$ | 0.32 | 0.56 |
  | $R_{si}$ – depth | 0.63 | 0.60 |
  | API – $R_{si}$ | 0.31 | 0.24 |

  The condensates' GOR falls with depth (−0.38), as their yield rises with it in the Gulf (+0.69).
- **Bounds**, each violated by none of the 2000 wells: every cell within 75° of vertical; MD at most 9342 m; $p_i$ at most 950 bar and $\lambda$ at most 0.85; $T_r$ at most 450 K; $(T_r - T_s)/L$ at least 10 K/km; $p_\text{sat} \le p_i$; black-oil $R_{si}$ from 1 to 356 Sm³/Sm³; API from 16 to 58; gas gravity at least 0.571; oils' water cut at most 0.98; `configurations.check` passes in the interim mapping.
- **Ranges left**, accepted: $p_r$ above 363 bar (Beggs–Robinson) in 32% of the wells and above 552 bar (Lee–Gonzalez–Eakin) in 10%; $T_r$ above 419 K in 7.6%; GOR above 1000 Sm³/Sm³ (Hasan et al.) in 40%, the gas classes and condensates.
- **Reproducibility:** a rerun gives the same draws (SMP-31), and a change of one block's distribution leaves the other blocks' draws as they were.

The sanity script prints all of these. R6 adds the rate check: the solved rates of offshore oil wells against the Gulf's 2024 brackets (EIA 2025, Table B32; D-24), and of gas wells against the AOF and tubing limits.

## The model's gaps (R5)

- **Gas-rate inflow.** Gas and condensate wells need it; the draws give Forchheimer's $a$ and $b$ (Inflow). R5 drafts the feature spec (backpressure or Forchheimer form) and whether `v2.0.0`'s check widens.
- **PVT tables for volatile oils and condensates.** The draws give each well's descriptors: API gravity, gas gravity, $R_{si}$, $p_\text{sat}$ at $T_r$, the separator, and the ranges of $p$ and $T$ the table must cover ($p_s$ to $p_i$, from below $T_s$ to $T_r$). R5 decides how synthetic compositions per class are matched to them: a table per well, or a library of fluids with a GOR axis (PVT-TAB-7) assigned by nearest descriptors, and its cost. Until then the interim black oil stands in.
- **Wet gases** stay black oil (decision 4): their condensate is a small part of the flow. R5 may measure what a table changes for them.
- **The productivity that stands for a horizontal drain** needs no model change: it is the draw's $J_H/J_V$ (Productivity). A gas drain's factor is a placeholder until Economides et al.'s §4-6 is read.
- **A composite IPR**, straight above the bubble point and Vogel below, is not an option; the inflow rule picks one or the other. R5 may weigh it.
- **Heavy oils** below 16 °API need a table; none is drawn.
- **Waiting for v2.x:** a water column in the ambient profile (deep-water platforms), $c_p(p, T)$, a varying $h$, and a tubing that changes diameter.

## Open issues

- **For R4:** the choke draw and its weight at a fully open choke; the lift-gas draw within $w_{lg,\max}$; the drift between samples and the `nsol-2` trends from the life stage's state; the best rate for the economic floor and the filter's place among SMP-28's; whether the filter on choked samples stays.
- **For R5:** the gas-rate inflow; the table fluids and their matching to the descriptors; whether volatile oils with a black-oil fallback are acceptable for the pilot.
- **For R6:** the solved rates against the Gulf's brackets, where the drawn productivity (design liquid rates at a median of 451 Sm³/d) may be high for a population counted by wells; the share of wells and samples without an operating point, by family and class; the coverage of regimes, choked flow and lift with and without the tail; the geometry shares, once ruled, and their effect.
- **For R2's second pass:** the gaps on the resource list below, the lithostatic gradient, offshore gas-lift injection pressures, choke sizes, and the shares of the geometry families.
- **For Bjarne now:** nothing blocks R4 to R6. Two choices he may want to see before the checkpoint: the onshore share (0.10, a design choice since onshore wells dominate by count), and whether the coverage tail may lift wells and assign geometries regardless of stage and type, as drafted.

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
- The public databases behind [RF] and [WP]: BOEM's sands atlas and borehole file, Sodir's FactMaps, as those files describe them.
