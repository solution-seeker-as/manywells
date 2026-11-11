# Sampler redesign — decision log

*The working assumptions of R2 to R7 (`plans/sampler-redesign-plan.md`, Working assumptions). Bjarne rules on each at the checkpoint (R8). Plans are drafts, not specs.*

Each entry records an open choice that an agent made in order to carry on, so that Bjarne can rule on all of them at once. A ruling that reverses an entry sends back the work that depends on it (listed under *Used by*).

## Entry form

```
### D-<n> · <short title>

- **Step:** R<k>, <date>.
- **Choice made:** the option taken, in one sentence.
- **Alternatives:** each with its evidence.
- **Realism:** whether the case occurs in realistic wells and in the draws, with numbers and sources.
- **Needs sign-off:** yes or no, and why (`AGENTS.md`, Needs Bjarne's sign-off).
- **Used by:** the sections of the drafts and the scripts that depend on it.
- **Ruling:** left empty until R8.
```

## Entries

Entries D-1 to D-36 are R3's (2026-10-30), for `sampling-draft.md`. R2's proposals are cited as RF-n (`sources-reservoir-fluids.md`, Working assumptions proposed) and WP-n (`sources-wells-production.md`). *Draws* are the sanity script's (`plans/evidence/sampler_draft_draws.py`, 2000 wells, seed 2026, core wells unless said otherwise). Every entry needs sign-off, because each sets part of the new `specs/sampling.md` (`AGENTS.md`, Needs Bjarne's sign-off); where an entry also touches another signed-off file, it says so.

### D-1 · Wells as the unit of weighting

- **Step:** R3, 2026-10-30. Takes RF-1.
- **Choice made:** weight public statistics by wells: BOEM's sands by their completions (`BHCOMP`), Norwegian wellbores as counted.
- **Alternatives:** by sands, which raises the Gulf's wet-gas share [own analysis of BOEM's sands atlas]; by volume, which puts oil and gas about even and over-weights large fields.
- **Realism:** the Goals count wells. Sidetracks count as wellbores on the Norwegian shelf, which slightly over-weights oil.
- **Needs sign-off:** yes (sampling).
- **Used by:** every share and marginal of `sampling-draft.md`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-2 · Nadeau et al. (2023) by reservoir counts

- **Step:** R3, 2026-10-30. Takes RF-2.
- **Choice made:** use the counts of reservoirs in Nadeau et al.'s Table 1 and Fig. 20 as checks, not their shares of EUR.
- **Alternatives:** EUR shares (5.5% of volume above 122 °C against 12.9% of reservoirs); counts times a well-count proxy such as EUR to the power 0.5.
- **Realism:** the draws put 17.1% of wells above 395 K and 20.4% above 0.6 psi/ft, against 12.9% of reservoirs above 122 °C and 16.2% above SG 1.3.
- **Needs sign-off:** yes (sampling).
- **Used by:** Initial reservoir pressure, Geothermal gradient (as checks).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-3 · Fluid classes by McCain (1990)

- **Step:** R3, 2026-10-30. Takes RF-3.
- **Choice made:** McCain's (1990, pp. 151–158) limits of 356, 588 and 8900 Sm³/Sm³ on the initial producing GOR, with API gravity as a check, and dry gas below a condensate yield of 2.8 Sm³ per 10⁶ Sm³; the rule is stored with the well (`class_rule`).
- **Alternatives:** Whitson and Brulé (volatile oils from 178 Sm³/Sm³); Tewari et al. (from 90 m³/m³), under which most of the Gulf's oil sands would be volatile [own analysis of BOEM's sands atlas]; McCain (1994), not in hand.
- **Realism:** under McCain's limits volatile oils are few of the Gulf's wells [own analysis of BOEM's sands atlas]; the black oil (PVT-OIL-4) carries McCain's black oils within Beggs and Robinson's $R_{so}$ range.
- **Needs sign-off:** yes (sampling).
- **Used by:** Fluid class, Saturation pressure and the initial GOR, Mapping.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-4 · The fluid mix and the gas subtypes

- **Step:** R3, 2026-10-30. Takes RF-4.
- **Choice made:** black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20; a gas well is dry with probability 0.07.
- **Alternatives:** the Gulf's well shares with the 10% floor (black oil 0.45, gas 0.30); the Norwegian shares (black oil 0.70, gas 0.10); dry gas at its share by sands [own analysis of BOEM's sands atlas].
- **Realism:** the shares by wells of the Gulf, its deep water and the Norwegian shelf are in [own analysis of BOEM's sands atlas] and [own analysis of Sodir's FactMaps]. The floor over-weights volatile oils several times over, which the Goals chose for coverage.
- **Needs sign-off:** yes (sampling; the Goals set the floor).
- **Used by:** Fluid class, and every class-dependent draw.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-5 · The well-type mix

- **Step:** R3, 2026-10-30. Takes R2's lead [WP §1].
- **Choice made:** platform (dry tree) 0.55, subsea 0.35, onshore 0.10.
- **Alternatives:** 0.60 / 0.40 / 0 (offshore only); a larger onshore share (onshore wells dominate counts: 918,481 producing wells in the US against 2,216 in the federal Gulf); the Norwegian split of platform and subsea wellbores [own analysis of Sodir's FactMaps].
- **Realism:** deep water in the Gulf is 84% dry tree (Liu, Wang and Kaiser 2014); a large minority of Norwegian wellbores are subsea [own analysis of Sodir's FactMaps]; the onshore share is a design choice of the Goals' offshore weighting, not a statistic.
- **Needs sign-off:** yes (sampling). Bjarne may want to see the onshore share before the checkpoint.
- **Used by:** Well type, and the water depth, surface temperature, separator, permeability, lift and economic floor.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint A1): platform 0.55, subsea 0.35, onshore 0.10, within `sol-2` and `nsol-2`, labelled by `well_type`. Separate onshore datasets were considered and not chosen: ManyWells models natural flow and gas lift only, most onshore wells are rod-pumped or on ESPs, no onshore yardstick is in hand, and they would change the Goals and add cost.

### D-6 · The coverage tail

- **Step:** R3, 2026-10-30.
- **Choice made:** 15% of wells, labelled; in a tail well every draw flattens to a (log-)uniform over its bounds, categorical shares become equal, and the links and bounds stay.
- **Alternatives:** 10% or 20%; a tail that stretches each draw independently with probability 1/2; a tail beyond the core's bounds, up to the model's limits (which would add implausible wells).
- **Realism:** tail wells are less likely, not implausible: none of the 2000 drawn wells breaks a bound. The tail lifts half its oil wells regardless of stage, which is less realistic than the core.
- **Needs sign-off:** yes (the Goals leave the share to the research).
- **Used by:** every draw's tail form; Where realism and coverage conflict; R6's coverage measurements.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint A2): as revised by D-103.

### D-7 · The order of the draws and seeding by block

- **Step:** R3, 2026-10-30.
- **Choice made:** the order of `sampling-draft.md`, with the salinity before the pressure and the separator before the fluid; one generator per block, `rng_for(seed, k, 'v2', block)`.
- **Alternatives:** the plan's likely order (water after the fluid, separator last), which would leave the hydrostatic gradient and Vazquez–Beggs's separator correction without their inputs; one generator per well, under which any change shifts every later draw.
- **Realism:** not applicable; it makes draws reproducible and local.
- **Needs sign-off:** yes (sampling, SMP-31's scheme extended).
- **Used by:** the whole draft; the sanity script.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-8 · Depth by fluid class, and water depth by well type

- **Step:** R3, 2026-10-30. Takes RF §1's lead.
- **Choice made:** depth below the mudline log-normal by class, fitted to BOEM's P5 / P50 / P95 by wells [own analysis of BOEM's sands atlas] and truncated to 600 to 6500 m; water depth by well type (platform 85% shelf around 90 m and 15% deep-water dry tree to 2000 m; subsea around 350 m to 3000 m); a platform's length includes its water.
- **Alternatives:** one depth distribution for all classes; depth before the class (draw the class given depth); the Norwegian TDs.
- **Realism:** dry gas lies shallowest and condensates deepest in the Gulf; deep water is deeper and oilier. The draws give 1530 / 2768 / 4831 m and reproduce BOEM's temperatures and gradients by class within 10 K and 0.008 bar/m [own analysis of BOEM's sands atlas].
- **Needs sign-off:** yes (sampling).
- **Used by:** Depth below the mudline; Water depth and the well's length; everything below them.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

### D-9 · Geometry families and the heel approach; shares pending

- **Step:** R3, 2026-10-30. Supersedes WP-1 with Bjarne's ruling of 2026-10-30.
- **Choice made:** four families (vertical, build-and-hold, S-shaped, high-angle to the heel) with the parameters of `sampling-draft.md`; a well with a horizontal drain is modelled to its heel; the shares stay open, pending `sources-geometry.md`.
- **Alternatives:** v1's three families (vertical, deviated, L-shaped); the two-segment equivalent tangent of the BOEM fit [own analysis of BOEM's borehole file] in place of build sections; build rates from a source (none in hand).
- **Realism:** few of the Gulf's completed development boreholes are straight or horizontal [own analysis of BOEM's borehole file]; Norwegian MD/TVD medians are higher for subsea wellbores than for platform ones [own analysis of Sodir's FactMaps]. The build rates of 1.5 to 3 °/30 m have no source in hand.
- **Needs sign-off:** yes (sampling; the shares are Bjarne's to rule after the geometry evidence).
- **Used by:** Geometry; Productivity (the drain); Gas lift (MD of the valve).
- **Ruling:** Bjarne, 2026-11-02, after `sources-geometry.md` and a private check: an even mix of four classes, 0.25 each of vertical, slightly deviated (5° to 20°), deviated (20° to 60°, J and S) and highly deviated bottom (60° to 75°), for every well type and fluid; the highly deviated class is half drains to the heel and half high-angle tangents capped at 75°, whatever the fluid. A design choice for coverage (`sampling-draft.md`, Geometry).

### D-10 · The inclination limit: 75°

- **Step:** R3, 2026-10-30. Takes WP-2.
- **Choice made:** no cell more than 75° from vertical.
- **Alternatives:** 80° (Taitel–Dukler leaves stratified flow only below $v_{SL}$ = 0.006 m/s; a few per cent of the Gulf's boreholes beyond [own analysis of BOEM's borehole file]); 70° (none in the scan above 0.003 m/s; more of the Gulf's boreholes beyond).
- **Realism:** at 75° Taitel–Dukler allows stratified flow only below 0.004 m/s, about 7 m³/d of in-situ liquid in 157 mm tubing; gas wells with very little liquid could reach it, and they are the least deviated (the Norwegian gas wellbores' MD/TVD is the lowest [own analysis of Sodir's FactMaps]).
- **Needs sign-off:** yes: it decides `validity.md`'s decision 5 with the geometry.
- **Used by:** Geometry.
- **Ruling:** Bjarne, 2026-10-30: about 75° from vertical, with drains modelled to the heel; R3 took 75°.

### D-11 · Brine salinity and density

- **Step:** R3, 2026-10-30. Takes RF-9.
- **Choice made:** salinity log-normal around 80,000 ppm, 1% to 25%; $\rho_w$ by a quadratic through McCain's (1990) points; the same brine sets the hydrostatic gradient.
- **Alternatives:** seawater (35,000 ppm) for every offshore well; v1's fixed 999.1 kg/m³ with a fixed gradient.
- **Realism:** no public distribution of offshore salinities is in hand; the median gives 0.104 bar/m, close to the Gulf's normal gradient [own analysis of BOEM's sands atlas].
- **Needs sign-off:** yes (sampling; it makes the water density a draw, one of the Goals' fluid constants).
- **Used by:** Formation water; Initial reservoir pressure; Heat capacities.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved with the two changes the produced-water statistics support: the salinity's median lowered to about 60,000 ppm and its floor to about 0.2%, the cap kept at 25% (near halite's saturation).

### D-12 · Overpressure with depth, and initial underpressure

- **Step:** R3, 2026-10-30. Takes RF-5 in a form of its own.
- **Choice made:** an overpressure ratio between hydrostatic and lithostatic (1 psi/ft) whose logit is normal around a median rising with depth, fitted to BOEM's depth bins; 4% of wells initially underpressured at 0.8 to 1.0 of hydrostatic.
- **Alternatives:** a depth-independent share; an excess pressure in bar fitted directly; no initial underpressure.
- **Realism:** 20.4% above 0.6 psi/ft, near the Gulf's share by wells; gradients by class within 0.008 bar/m of BOEM's medians [own analysis of BOEM's sands atlas]. The lithostatic gradient has no source in hand.
- **Needs sign-off:** yes (sampling).
- **Used by:** Initial reservoir pressure; Bjarne's goal 3.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups B and C's bulk lists): approved.

### D-13 · The geothermal gradient

- **Step:** R3, 2026-10-30. Rejects RF-6's mixture of three basins.
- **Choice made:** one log-normal below the mudline, median 30 K/km on shelves and onshore falling to 20 K/km in deep water, log-sd 0.22, truncated to 12 to 55 K/km and to keep $(T_r - T_s)/L \ge 10$ K/km and $T_r \le 450$ K.
- **Alternatives:** RF-6's mixture (24, 18 and 37 K/km with ±15 to 20%); a single global log-normal without the water-depth term.
- **Realism:** medians by class within 6 K of BOEM's [own analysis of BOEM's sands atlas]; 17.1% above 395 K (12.9% of large-field reservoirs). The North Sea's gradients are uncorrected logging values, which run low.
- **Needs sign-off:** yes (sampling).
- **Used by:** Geothermal gradient and reservoir temperature.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups B and C's bulk lists): approved.

### D-14 · Surface temperature by well type

- **Step:** R3, 2026-10-30. Takes WP-11 with a seabed profile.
- **Choice made:** offshore wells take the seabed temperature at their water depth, from a shelf value of 279 to 297 K to 272 to 278 K in deep water over a 400 m scale; onshore wells 268 to 303 K.
- **Alternatives:** one range for all; a platform's sea-surface temperature at its wellhead.
- **Realism:** the profile meets the four seabed values of [WP §2] within about 2 K; deep-water platforms' risers get too warm an ambient under THM-4's linear profile.
- **Needs sign-off:** yes (sampling; a Goals' new draw).
- **Used by:** Surface temperature; the gradient's bounds; the lift-gas temperature.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups B and C's bulk lists): approved.

### D-15 · Separator conditions and the nominal downstream pressure

- **Step:** R3, 2026-10-30. Takes WP-8 in a form of its own.
- **Choice made:** the host's first-stage separator ($p_\text{sep}$ around 35 bar offshore, 15 bar onshore) is the fluid's separator in PVT-OIL-5; $p_{s,\text{nom}}$ falls with the life stage (to 0.1 to 0.6 of $p_\text{sep}$ late), and a subsea well adds its flowline and riser.
- **Alternatives:** standard conditions as the separator (today's default); v1's log-normal truncated to 10 to 120 bar with a separate low-pressure tail; WP-8's 2 to 80 and 10 to 100 bar as uniform ranges.
- **Realism:** first stages at 41 to 83 bar (Guo et al. 2007), lower stages at 20 and 2 bar (weak). The separator's correction raises Vazquez–Beggs's $R_{so}$ by about 30% at 40 bar against standard conditions, so it shifts the black oils' GORs.
- **Needs sign-off:** yes (sampling; separator conditions are a Goals' fluid constant).
- **Used by:** Separator conditions; Saturation pressure and the initial GOR; Tubing; Gas lift; Bjarne's goal 2.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group C's bulk list): approved with the revision for subsea wells: a larger flowline-and-riser loss and a higher truncation of $p_{s,\text{nom}}$ than 120 bar, set on the rework's pilot against 3W's subsea tree pressures.

### D-16 · Oil and gas gravities by class

- **Step:** R3, 2026-10-30.
- **Choice made:** black oil 33 ± 4.5 °API (16 to 45) with gas gravity 0.66 ± 0.06; volatile oil 40 to 50 °API; condensate 42 to 58 °API; wet and dry gas 45 to 58 °API with gas gravity 0.62 ± 0.035 from 0.571.
- **Alternatives:** a Gaussian copula of API, GOR and gas gravity on the Gulf's rank correlations (RF §7's lead); API from v1's oil density (21 to 40).
- **Realism:** the Gulf's oils' API and gas gravities are in [own analysis of BOEM's sands atlas]; Valkó and McCain's separator gases average 0.838. API and $R_{si}$ correlate at 0.31 in the draws, through Vazquez–Beggs, against the Gulf's correlation in the same section.
- **Needs sign-off:** yes (sampling).
- **Used by:** Oil and gas at standard conditions; everything that uses the fluid.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A): independent draws, with black oils' gas gravity $N_{[0.58, 0.95]}(0.70, 0.07)$, heavier as Norwegian drill-stem tests show [own analysis of Sodir's FactPages]. The tests' rank correlation of API and gas gravity in oils is physical (lighter oils carry more intermediates into the separator gas) but changes the black oils' $R_{si}$ little, so it is not drawn.

### D-17 · Undersaturation and the initial GOR from the model's own correlation

- **Step:** R3, 2026-10-30. Takes RF-7.
- **Choice made:** draw the saturation pressure as a fraction of $p_i$ (20% of black oils saturated, the rest with a median undersaturation of 10%), and take the black oil's $R_{si}$ from the model's Vazquez–Beggs at that pressure, capped at 356 Sm³/Sm³ by lowering the bubble point.
- **Alternatives:** draw $R_{si}$ given depth and set $p_b$ from it; Standing's $p_b$ (PVT-OIL-14), which the simulator does not use.
- **Realism:** $R_{si}$ 71 / 168 / 356 Sm³/Sm³ and $p_i - p_b$ 0 / 26 / 209 bar, against the Gulf's [own analysis of BOEM's sands atlas]; rank correlations within 0.12 of the Gulf's but for $R_{si}$–$T_r$ (0.32 in the draws, stronger in the Gulf). The well's bubble point is the model's.
- **Needs sign-off:** yes (sampling).
- **Used by:** Saturation pressure and the initial GOR; Reservoir state; Inflow.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved. Then, 2026-11-09 (second checkpoint, L2): for a volatile oil or condensate the table's flash sets the saturation pressure and its label (D-65), so $x$ applies to black oils only.

### D-18 · Drive, life stage and the reservoir's state

- **Step:** R3, 2026-10-30. Takes RF-8 and RF §3, §9, §11's leads.
- **Choice made:** a drive label (pressure support 0.7 for oils, 0.4 for gases), a life stage (0.25 / 0.40 / 0.35), and the state tables of `sampling-draft.md` for $p_r/p_i$ and the water.
- **Alternatives:** v1's ±2% around a hydrostatic $p_r$ with no stage; the stage drawn from a field's age with decline curves (closer to `nsol-2`'s evolution, which R4 designs).
- **Realism:** 46.7% of core wells lie below hydrostatic and 30.1% by more than 10%; late depletion wells reach the Gulf's abandonment $p/Z$ range [own analysis of BOEM's sands atlas]; late wells with support have a median water cut of 0.77, near the Norwegian oil fields' at 10 to 20 years [own analysis of Sodir's FactMaps]. No public count of wells by stage is in hand.
- **Needs sign-off:** yes (sampling; it meets Bjarne's goal 1 and the Goals' life stage).
- **Used by:** Drive and life stage; Reservoir state; Separator; Roughness; Heat transfer; R4's drift and `nsol-2` trends.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint C1): stage shares 0.25 / 0.40 / 0.35 and the state tables, with the water cut's upper tail thickened towards 0.98 and the initial stage's depletion range widened to $U(0.85, 1.0)$.

### D-19 · Producing GOR below the saturation pressure

- **Step:** R3, 2026-10-30.
- **Choice made:** a linear rise of the producing GOR with $1 - p_r/p_\text{sat}$, at a factor of 2 to 15 for oils (capped at ten times $R_{si}$) and 0.5 to 3 for condensates.
- **Alternatives:** a material-balance curve of solution-gas drive (needs a reservoir-engineering text, on R2's resource list); a constant GOR.
- **Realism:** Golan and Whitson's (1991) Example 4.11 rose to 8.3 times its initial GOR at 0.58 of initial pressure, then fell; Norwegian condensate fields' producing GOR late in life is in [own analysis of Sodir's FactMaps]. One example only.
- **Needs sign-off:** yes (sampling).
- **Used by:** Reservoir state; R4's `nsol-2` GOR trend.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint C1): the producing GOR below the saturation pressure from the material balance of solution-gas drive (Muskat; Craft and Hawkins 1991, Ch. 9) on the well's own black oil, with Fig. 9.16's $k_g/k_o$ in the core and the book's range of curves in the tail, chosen by the per-well GOR data; $R_{si}$ wherever $p_r \ge p_\text{sat}$; no cap (`solution_gas_drive.py`). Replaces the linear factor of 2 to 15.

### D-20 · Productivity from permeability, pay, viscosity and skin

- **Step:** R3, 2026-10-30. Takes WP §4's lead.
- **Choice made:** draw $k$ (offshore around 50 mD, onshore around 10 mD), net pay around 15 m and skin around 2, and compute the radial PI with the model's viscosity and $B_o$ at the stage's state; a design drawdown of 10% to 40% of $p_r$ sizes the tubing and the lift.
- **Alternatives:** draw the PI directly as a log-normal; draw the rate and back out the PI (WP-3's alternative); v1's $w_{l,\max}$ of 20 to 200 kg/s independent of everything.
- **Realism:** design liquid rates of 33 / 451 / 6411 Sm³/d (P10 / P50 / P90) for oils: wide, and high for a population counted by wells next to the Gulf's 2024 brackets (52% of oil wells at or below 16 Sm³/d o.e.), but solved rates are lower; R6 checks (D-24).
- **Needs sign-off:** yes (sampling).
- **Used by:** Productivity; Inflow; Tubing; Gas lift: capacity.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint D1): as drafted, with D-100's permeability by well type.

### D-21 · Horizontal drains and slant: the productivity's factor

- **Step:** R3, 2026-10-30.
- **Choice made:** a high-angle oil well's PI is Joshi's (Economides et al. 1994, Eq. 2-46) for a drain of 300 to 1000 m, capped at 6 times the vertical PI; a high-angle gas well multiplies $k h_n$ by 1.5 to 4; a slanted well by $(\cos\theta_\text{end})^{-1/2}$.
- **Alternatives:** one multiplier for every high-angle well (R2's lead); Cinco-Ley's slant skin and the horizontal gas-well IPR (Economides et al. 1994, §5-4 and §4-6, not read).
- **Realism:** the draws give $J_H/J_V$ of 1.8 / 3.9 / 6.0, against 2 to 5 for a 610 m drain in Economides et al.'s Fig. 2-10. The gas factor and the slant factor are placeholders.
- **Needs sign-off:** yes (sampling; it is how the inflow stands for the drain under Bjarne's ruling).
- **Used by:** Productivity; Inflow.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint B1 and group D): the physical form, kept simple: a drain's PI is Joshi's on a drainage length that grows with the drain, times the friction efficiency of Dikken's model in closed form, $\eta = (1 + \Pi/1.5)^{-1/3}$, with the liner taken as the tubing and no cap; a gas drain's Forchheimer coefficients scaled by the same physics (`drain_friction.py`). Drains $U(300, 2500)$ m (D-104). The slant factor stays a placeholder.

### D-22 · Oil wells' inflow: Vogel or a productivity index

- **Step:** R3, 2026-10-30.
- **Choice made:** Vogel where the saturation pressure is at least 0.8 of $p_r$ and the water cut below 0.6, else a productivity index; Vogel's $w_{l,\max}$ gives the same PI at $p_0 = p_r$.
- **Alternatives:** Vogel for every oil well (v1); a composite IPR (not an option of the model).
- **Realism:** Vogel holds for solution-gas drive below the bubble point and not where water flows too (`validity.md`, note 10); 64% of core oils get it.
- **Needs sign-off:** yes (sampling).
- **Used by:** Inflow; Mapping.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-23 · Gas and condensate wells: Forchheimer's parameters, and the interim inflow

- **Step:** R3, 2026-10-30. Takes WP-7 for the draws; the option is R5's.
- **Choice made:** draw Forchheimer's $a$ from $k h_n$, the model's $Z$ and $\mu_g$ and the skin, and $b$ from a non-Darcy coefficient of $2\cdot10^{-4}$ to $2\cdot10^{-3}$ d/Mscf; until R5's gas-rate inflow exists, map to a productivity index that gives the design rate's liquid.
- **Alternatives:** the backpressure form with $n$ from 0.5 to 1 and $C$ from a target AOF; Vogel for gas wells (v1, outside Vogel's range).
- **Realism:** AOF 0.12 / 1.5 / 8.1 million Sm³/d (P5 / P50 / P95); 11% of gas wells' design rates exceed the largest tubing's erosional rate, as "a strong gas well will never produce to capacity" (Golan and Whitson 1991, Preface). The interim makes the gas rate linear in drawdown.
- **Needs sign-off:** yes (sampling; the feature spec is R5's and needs approval before implementation, and widening `v2.0.0`'s check needs Bjarne's ruling).
- **Used by:** Productivity; Inflow; Mapping; R5; R6's interim prototype.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-24 · The rate target is a check, not a draw

- **Step:** R3, 2026-10-30. Takes WP-3.
- **Choice made:** compare R6's solved rates of offshore wells with the Gulf's 2024 brackets (EIA 2025, Table B32) after the economic filter; do not draw rates.
- **Alternatives:** draw the rate and back out the PI.
- **Realism:** the brackets are annual averages over calendar days of a mature shelf, so they sit low for a global offshore population.
- **Needs sign-off:** yes (sampling; the check's tolerance is R6's).
- **Used by:** Checks; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-25 · Tubing from the design rate

- **Step:** R3, 2026-10-30. Takes WP §6's lead.
- **Choice made:** the smallest of 62 to 157 mm (220.5 mm for gases) whose no-slip mixture velocity at the initial design rate stays below the erosional velocity with $C$ from 100 to 150; one size larger with probability 0.3.
- **Alternatives:** v1's uniform draw from 76 to 165 mm; sizing at the current stage; a critical-velocity check for gas wells.
- **Realism:** no public distribution of tubing sizes is in hand. Late-life wells keep their initial tubing, which is how they load up.
- **Needs sign-off:** yes (sampling).
- **Used by:** Tubing; Choke; Heat transfer; Gas lift: capacity.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-26 · Roughness by life stage

- **Step:** R3, 2026-10-30.
- **Choice made:** SMP-42's log-uniform range split by stage: new tubing early, corroded tubing late.
- **Alternatives:** SMP-42 unchanged for all wells.
- **Realism:** the examples in hand give 30 and 46 µm; Farshad and Rieke (2006) are not in hand.
- **Needs sign-off:** yes (sampling).
- **Used by:** Roughness.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-27 · The choke: Simpson only, sized to the tubing

- **Step:** R3, 2026-10-30. Takes WP-10.
- **Choice made:** a Simpson choke for every well, with a fully open bean of 0.25 to 0.6 of the tubing's diameter and a discharge coefficient of 0.75 to 0.90; profiles with equal shares.
- **Alternatives:** v1's $K_c$ draw (SMP-5); the Bernoulli choke for a share of wells.
- **Realism:** no public data on bean sizes or profiles over wells; the draw contains v1's range.
- **Needs sign-off:** yes (sampling).
- **Used by:** Choke; R4's choke draw.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint E1): $\beta \sim U(0.15, 0.6)$ of the tubing's diameter, with the fully open bean capped at 4 in (101.6 mm), the largest choke tested in the GoM's groups [own analysis of BSEE's well tests]; checked in mm against the tests after D1 and D2.

### D-28 · Who gets gas lift

- **Step:** R3, 2026-10-30. Takes WP §9's lead.
- **Choice made:** oils that cannot lift their static column get gas lift; other oils by stage (offshore 0.20 / 0.45 / 0.70, onshore 0.15 / 0.25 / 0.35), gases 0 / 0.05 / 0.15; a strong well whose balance point lies above 0.3 $L$ is left on natural flow in the core; no GOR limit.
- **Alternatives:** v1's rule (gas lift only for $f_g \le 0.2$, with probability 1/2); a share independent of stage.
- **Realism:** core oils 32% lifted, 58% late in life; gas lift is the main lift offshore, but no count of lifted wells is in hand.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5).
- **Used by:** Gas lift; R4's lift-gas draw.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint F1): as drafted: the static-column check and the stage shares, with no tuning of the shares by GOR quartile to the GoM's. Keep it simple and do not lean on one region.

### D-29 · The injection depth

- **Step:** R3, 2026-10-30. Takes WP-6 with Golan and Whitson's (1991, §5.3) pressure balance.
- **Choice made:** the valve at the pressure-balance point of a surface injection pressure of 48 to 76 bar (Gilbert's gas column, a valve drop of 3.4 to 13.8 bar) against the flowing tubing pressure at a lifted design point, between 0.3 $L$ and a packer 30 to 150 m above the bottom.
- **Alternatives:** WP-6's 30 to 150 m above the perforations for every well; the bottomhole (INF-5, today's sampler).
- **Realism:** $H/L$ of 0.33 / 0.65 / 0.97; the injection pressures are an onshore textbook range, and offshore systems may run higher (no source in hand).
- **Needs sign-off:** yes (sampling; the Goals' new draw, feature 021).
- **Used by:** Gas lift: injection depth; Mapping.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group F): the valve at the pressure-balance point, with the injection pressure by well type: platform and onshore wells $U(48, 85)$ bar at the surface, subsea wells $U(130, 210)$ bar at the tree [own analysis of BSEE's well tests and Petrobras' 3W Dataset].

### D-30 · The lift gas's temperature

- **Step:** R3, 2026-10-30.
- **Choice made:** between the ambient at the valve (THM-9's default) and 30% of the way towards the surface temperature.
- **Alternatives:** THM-9's default for every well (`T_lg=None`); the reservoir temperature (SMP-44).
- **Realism:** the annulus gas exchanges heat with the formation on its way down (Hasan and Kabir 2018, §12.2.2), but no numbers are in hand; the 30% is an own choice.
- **Needs sign-off:** yes (sampling; a Goals' new draw).
- **Used by:** Gas lift: lift-gas temperature; Mapping.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group F's bulk list): approved.

### D-31 · The lift-gas capacity

- **Step:** R3, 2026-10-30. Takes WP-5 with a lower ratio.
- **Choice made:** an injected GLR at capacity of 50 to 400 Sm³/Sm³ times the design liquid rate, within 0.02 to 2.0 kg/s.
- **Alternatives:** WP-5's 100 to 800 Sm³/Sm³ within 0.1 to 2 kg/s; v1's 0 to 5 kg/s.
- **Realism:** the examples in hand inject 27 to 77 Sm³/Sm³, and up to 184,000 Sm³/d per well (von Flatern 2015); total GLRs of 430 to 800 Sm³/Sm³ in small tubing are optima. The draws give 0.02 / 0.38 / 2.0 kg/s.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5).
- **Used by:** Gas lift: capacity; R4's lift-gas draw.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint F2), decided from data at his ask: the capacity's floor raised to 0.1 kg/s and its cap to 3.5 kg/s for every well type (lifted wells' injection reaches past 2.0 kg/s on platforms as well as subsea; 3W's subsea wells reach about 3.4 kg/s [own analysis of Petrobras' 3W Dataset]), and the injected GLR at capacity widened to 30 to 1000 Sm³/Sm³ times the design liquid rate, since lifted wells' capacities spread wider than 50 to 400. Then, 2026-11-06, after the rework's draws piled 32% of the lifted core wells on the floor and 15% on the cap: the GLR at capacity is drawn log-uniformly over the part of 30 to 1000 Sm³/Sm³ that maps into [0.1, 3.5] kg/s at the well's design rate (truncated, not clipped); a well too weak for 0.1 kg/s even at 1000 keeps the floor (about 7% of the lifted core wells).

### D-32 · The heat-transfer coefficient

- **Step:** R3, 2026-10-30. Takes WP-4.
- **Choice made:** $h$ from Hasan and Kabir's (2012) series of the wellbore's $U_t$ and the formation's transient resistance at the well's age.
- **Alternatives:** v1's 10 to 40 W/(m² K) (kept in the tail).
- **Realism:** 3.2 / 6.3 / 11.9 W/(m² K), inside the 3 to 25 implied by Hasan and Kabir's examples; v1's range overstates heat loss for wells older than a month.
- **Needs sign-off:** yes (sampling).
- **Used by:** Heat-transfer coefficient.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-33 · Heat capacities from correlations

- **Step:** R3, 2026-10-30. Takes RF-10.
- **Choice made:** oil by Cragoe (1929) at the API gravity, water by Sharqawy et al. (2010) at the salinity, gas by NIST's methane scaled by $(0.554/\gamma_g)^{0.4}$, each at a representative temperature and pressure.
- **Alternatives:** today's constants (SMP-9 to SMP-11).
- **Realism:** $c_{pg}$ of 2513 to 3178 J/(kg K) against today's 2225; the gas's scaling is an assumption until a correlation for natural gases is in hand.
- **Needs sign-off:** yes (sampling; a Goals' fluid constant).
- **Used by:** Heat capacities.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups D and E's bulk lists): approved.

### D-34 · The economic floor

- **Step:** R3, 2026-10-30. Takes WP-9 with a margin.
- **Choice made:** 1.6 Sm³/d of oil equivalent for onshore and platform wells, 8 Sm³/d for subsea wells; R4 drops a well whose best rate at its nominal conditions is below half its floor, and oil wells above a water cut of 0.98.
- **Alternatives:** one floor for all; no margin; a floor per structure shared by its wells.
- **Realism:** stripper limits of 10 bbl/d and 60 Mcf/d (EIA 2025, p. 10); offshore structures' revenue thresholds (Kaiser 2011, abstract only). Before any solve 2.0% of the drawn wells lie below half the floor.
- **Needs sign-off:** yes: the margin is Bjarne's (the Goals), and the filter is part of the dataset design (R7).
- **Used by:** The economic floor; R4; R7.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint D3): the floor as drafted for platform and onshore wells; the subsea floor raised to about 20 Sm³/d of oil equivalent, from the final rates of stopped subsea wells [own analysis of BOEM's OGOR-A production and ANP's production by well].

### D-35 · Truncations at the model's and the data's limits

- **Step:** R3, 2026-10-30.
- **Choice made:** truncate $p_i$ at 950 bar (Hasan et al.'s data), $T_r$ at 450 K, the model's gradient at no less than 10 K/km, measured depth at 9342 m (redrawn), and API gravity at 16 (no heavy oils), in core and tail alike.
- **Alternatives:** accept and extrapolate; the model refuses only the API gravity, so the others are choices.
- **Realism:** the 950 bar bound cuts into the central 90% of the pressure draw in 6.8% of wells, the temperature bound in 13.2%; HPHT wells beyond exist but are rare; heavy oils (few of the Gulf's lie below 22 °API [own analysis of BOEM's sands atlas]) need a table.
- **Needs sign-off:** yes (sampling; `validity.md`'s ranges are signed off, and these bounds choose where the sampler stops).
- **Used by:** Initial reservoir pressure; Geothermal gradient; Geometry; Oil and gas at standard conditions.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group C's bulk list): approved, the measured depth kept at 9342 m, the edge of Hasan et al.'s data, after 9000 m was considered.

### D-36 · Volatile oils and condensates as tables; wet gases as black oil; the interim mapping

- **Step:** R3, 2026-10-30.
- **Choice made:** draw each volatile oil's and condensate's descriptors (API, gas gravity, $R_{si}$, saturation pressure, separator, the $p$ and $T$ to cover) for R5's tables; keep wet and dry gases as black oil (decision 4); for R6's prototype, map volatile oils and condensates to the black oil until the tables exist.
- **Alternatives:** tables for wet gases too; a fixed library of published example fluids with no drawn descriptors.
- **Realism:** the Goals put volatile oils and condensates in tables from a flash; in the interim their bubble and dew points are Vazquez–Beggs's, not the drawn ones.
- **Needs sign-off:** yes (sampling; R5's tables and their cost are for the checkpoint).
- **Used by:** Fluid class; Saturation pressure; Mapping; R5; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group A's bulk list): approved.

Entries D-37 to D-56 are R4's (2026-10-30), for the sample part of `sampling-draft.md` (The samples within a well). Their *draws* are those of `plans/evidence/sampler_draft_samples.py` (1000 wells, seed 2026), which draws without solving. Like R3's, every entry needs sign-off, because each sets part of the new `specs/sampling.md`; an entry that also bears on the dataset design (R7), a tolerance or solver machinery says so.

### D-37 · `sol-2`'s choke position: 5% at fully open, else uniform from 0.05

- **Step:** R4, 2026-10-30.
- **Choice made:** $u = 1$ with probability 0.05, else $U(0.05, 1)$, independent per slot, for core and tail wells alike.
- **Alternatives:** a mass of 0.03 or 0.08 at fully open (the range the Goals' "modest" allows); a ramp of density rising towards 1 in place of a point mass, which blurs "fully open"; $u_\text{min}$ = 0.1, which avoids the openings where wells stop flowing; a mass that grows with the life stage (late wells run fully open more often), which no public source quantifies; stratified (Latin hypercube) draws within a well, which spread a well's 200 positions more evenly but tie the slots together.
- **Realism:** no public distribution of choke openings in operation [WP §8]. Chokes limit rates for regulations, surface equipment, sand and coning (Guo et al. 2007, §5.1, p. 5/60), so operating wells are mostly choked back and late ones often fully open. The draws give 5.0% at fully open and 5.1% below 0.1; v1's draws failed mostly at small openings (L6).
- **Needs sign-off:** yes (sampling; Bjarne's goal 4).
- **Used by:** `sol-2`, Choke position; D-47's set points; `sampler_draft_samples.py`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-38 · `sol-2`'s lift-gas rate: uniform to the capacity, off in 10% of the slots

- **Step:** R4, 2026-10-30.
- **Choice made:** for a lifted well, $w_{lg} = 0$ with probability 0.10, else $U(0, w_{lg,\max})$; natural wells 0.
- **Alternatives:** always on (no zeros), which loses each lifted well's natural-flow point; log-uniform within the capacity, which crowds the low rates where the lift performance curve bends; a draw around an allocated rate near the optimum, closer to operations but narrower; v1's $U(0, 5)$ kg/s for every lifted well (SMP-19).
- **Realism:** von Flatern (2015, Fig. 1) optimises seven wells between 0.5 and 6.5 MMscf/d; beyond a limiting GLR more gas raises the gradient (Golan and Whitson 1991, p. 538); below a minimum rate lifted wells head (Asheim 1988, p. 1452). The share of time with lift gas off has no public source. Draws: 10.0% of lifted wells' slots without lift gas; a well that needs lift has no operating point there.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5).
- **Used by:** `sol-2`, Lift-gas rate; D-47; D-50's best rate.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group F's bulk list): approved with the lift gas off in 20% of a lifted well's slots, not 10%.

### D-39 · The separator pressure per sample: ±10% around the nominal

- **Step:** R4, 2026-10-30.
- **Choice made:** $p_s \sim U(0.9, 1.1)\thinspace p_{s,\text{nom}}$, truncated below at 1.5 bar and at a subsea well's flowline loss.
- **Alternatives:** ±15%, log-uniform; a share of samples routed to a lower stage (a high- and a low-pressure header), which would make the separator a routing control; ±5%, closer to a controlled separator.
- **Realism:** no public spread of separator or manifold pressures [WP §7]; v1 used ±10% (SMP-20). In sonic flow the downstream pressure does not reach the well (Guo et al. 2007, §5.2, p. 5/60). Draws: 16.6% of `sol-2` slots below 10 bar.
- **Needs sign-off:** yes (sampling).
- **Used by:** `sol-2`, Separator pressure; `nsol-2`, Separator over the life.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-40 · `sol-2`'s drift, with the GOR tied to the drifted pressure

- **Step:** R4, 2026-10-30.
- **Choice made:** $p_r$ times $U(0.98, 1.02)$; the producing GOR from R3's formula at the drifted $p_r$, times $U(0.95, 1.05)$; the WLR or WGR times $U(0.95, 1.05)$; the inflow's coefficients, $T_r$, $T_s$ and $T_{lg}$ fixed. `nsol-2` applies the same ±5% to the GOR and the water per date, and no noise to $p_r$.
- **Alternatives:** independent ±5% on the GOR (v1's independence), which can give a saturated oil less gas at a lower pressure; drift on the mass fractions (v1's SMP-22), which the draft no longer draws; a cap of $p_r$ at $p_i$, which puts a point mass there; no noise in `nsol-2`.
- **Realism:** the sizes are the Goals'. Below the bubble point the GOR rises as the pressure falls (McCain 1990, pp. 151, 153), which the tie keeps. Draws: saturated oils' GOR within 0.92 to 1.09 of nominal (P5 to P95), against 0.95 to 1.06 for all wells; 1.11% of the slots have $p_r \le p_s$ and no operating point.
- **Needs sign-off:** yes (sampling; the Goals set the sizes).
- **Used by:** `sol-2`, Drift of the reservoir; `nsol-2`, Drift between dates.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-41 · `nsol-2`'s life: the span, the clock and the start

- **Step:** R4, 2026-10-30.
- **Choice made:** every `nsol-2` well starts at R3's initial stage and ends its life span at R3's late-stage state of the same well; $T_\text{life} \sim \text{LogU}(8, 30)$ years; the trends follow a clock $F(t)$, the reservoir's cumulative production with a plateau of up to a quarter of the life and Arps's decline ($D$ from 0.07 to 0.30 a year, $b$ from 0 to 0.4 under depletion and 0.3 to 0.5 under support); a per-sample stage label (initial in the first year, late from half the life).
- **Alternatives:** infill wells that start at the stage R3 drew, so that some wells start in a depleted, watered reservoir (realistic, but their 200 samples would not span the reservoir's life); time itself as the clock, which gives a linear decline; v1's exponential approach to a drawn $p_r(\infty)$ (SMP-25), close to this clock with $b = 0$; a life of $U(10, 20)$ years (SMP-23).
- **Realism:** Arps's forms (Ahmed 2006, Eqs. 16-1 to 16-4) with $b$ of 0 to 0.5, 0.3 for solution-gas and 0.5 for water drive (Golan and Whitson 1991, p. 385); offshore field declines of 7 to 23% a year (Miller et al. 2009) [RF §11]; Norwegian shut-down fields' producing lives [own analysis of Sodir's FactMaps], a sample cut short by the fields still producing. Draws: $T_\text{life}$ 8.6 / 15.6 / 27.9 years.
- **Needs sign-off:** yes (sampling; the Goals' "a well's 200 samples span its life").
- **Used by:** `nsol-2`, The well and its states; Life span and the depletion clock; every trend.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G): a steeper decline, so that the last fifth of a life falls to about a tenth of the first year's rate, set on the rework's pilot against the per-well rows [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well: rate_over_first_year, producing_life]; lives may be shorter than 8 years. Then, 2026-11-06, after the rework's pilot showed that the clock alone leaves the last fifth at 0.37 of the first sample (the end state, R3's late draw, still makes about a third of the start's rate): a harsher end state for `nsol-2` only, at the far end of the late range, depletion $p_r/p_i \sim U(0.2, 0.4)$ for oils and $U(0.15, 0.35)$ for gases, and an oil well's water cut $U(0.85, 0.98)$ under pressure support; `sol-2`'s late stage unchanged; measured on the rerun and tuned once. Superseded the same day by D-105: one life for both datasets, ending at an exhausted reservoir and stopping at the floor.

### D-42 · `nsol-2`'s reservoir pressure

- **Step:** R4, 2026-10-30.
- **Choice made:** for oils and every well under pressure support, $p_r = p_i\big(r_1 + (r_0 - r_1)(1 - F)^\gamma\big)$ with $\gamma \sim U(1, 2)$; for gas and condensate wells under depletion, $p/Z$ linear in $F$ with the model's $Z$; the ends from R3's initial and late states; a tail well starts from the core's initial state.
- **Alternatives:** $p_r$ linear in $F$ for every well; a material balance of solution-gas drive (Tarner's or Muskat's method), which needs relative permeabilities that the draws do not have; a pressure that recovers under injection.
- **Realism:** the gas material balance is a straight line of $p/z$ against cumulative production (Ahmed 2006, Eq. 13-8, p. 861). Golan and Whitson's (1991) Example 4.11 is fitted within 0.008 $p_i$ at $\gamma$ = 1.40; a straight line misses it by up to 0.055 $p_i$. One example only. Draws: core oils' $p_r/p_i$ medians 0.985, 0.84 and 0.81 at the start, the middle and the end of the life; gases 0.98, 0.59 and 0.45.
- **Needs sign-off:** yes (sampling; Bjarne's goal 1 in `nsol-2`).
- **Used by:** `nsol-2`, Reservoir pressure; the GOR (D-44); the inflow (D-45); the lift's static check (D-49).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-43 · `nsol-2`'s water: a jump at breakthrough, then the semilog WOR

- **Step:** R4, 2026-10-30.
- **Choice made:** an oil well's water cut stays at its start until breakthrough at the recovery $F_\text{bt}$, jumps to $\text{WLR}_ \text{bt}$, and then rises with its logit linear in $F$ to R3's late state. Under pressure support $F_\text{bt} \sim U(0.05, 0.4)$ and $\text{WLR}_ \text{bt} \sim U(0.05, 0.3)$; under depletion $U(0.2, 0.7)$ and $U(0.02, 0.1)$. A gas well's WGR rises log-linearly in $F$ from $F_\text{bt}$.
- **Alternatives:** no jump (a logit-linear rise from a small water cut), which leaves mid-life wells far drier than R3's mid-life states (the draft's first try gave a median of 0.06 against 0.36); the one-dimensional Buckley–Leverett jump to the front's water cut, 0.78 in Ahmed's example (2006, p. 983), which a layered reservoir smooths; a logistic in time.
- **Realism:** the fractional flow is S-shaped in the water saturation (Ahmed 2006, Eq. 14-22, Fig. 14-12) and the ratio of relative permeabilities a straight line on semilog paper (Eq. 5-28), so the logit of the water cut is linear in the saturation; the water cut "increases suddenly" at breakthrough (p. 970). Under pressure support the draws give 0.19 / 0.55 / 0.68 / 0.78 at 0–5 / 5–10 / 10–20 / 20–30 years on production, against Norwegian oil fields' medians (fields) [RF §9; own analysis of Sodir's FactMaps]; by stage, 0.32 in mid-life and 0.68 late against R3's 0.36 and 0.77.
- **Needs sign-off:** yes (sampling).
- **Used by:** `nsol-2`, Water; the lift's static check (D-49).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-44 · `nsol-2`'s GOR: R3's formula along the pressure, without the late fall

- **Step:** R4, 2026-10-30.
- **Choice made:** the producing GOR is R3's formula at $p_r(t)$, with the well's own factor, so that it rises once the pressure falls below the saturation pressure; it does not fall late in life.
- **Alternatives:** a rise then a fall, as Golan and Whitson's (1991) Example 4.11 (to 8.3 times $R_{si}$, then 4.2), with a shape that one example cannot fix; a GOR from a material balance of solution-gas drive (Resource list (R4 additions)).
- **Realism:** the GOR rises below the bubble point (McCain 1990, pp. 151, 153). R3's form reaches at most 6.6 times $R_{si}$ where the example reached 8.3. Draws: core oils' $R_{go}/R_{si}$ medians 1.01, 1.40 and 1.54 over the life, P95 5.6 at the end.
- **Needs sign-off:** yes (sampling).
- **Used by:** `nsol-2`, Gas–oil ratio.
- **Ruling:** Bjarne, 2026-11-06 (with checkpoint C1): the GOR along the life is D-19's material-balance factor at $p_r(t)/p_\text{sat}$; no late fall, which the physics puts below every core state.

### D-45 · `nsol-2`'s well properties that age, and the fixed inflow model

- **Step:** R4, 2026-10-30.
- **Choice made:** $h$ follows the well's age by R3's formula; the roughness rises log-linearly from the initial draw's to the late draw's; the productivity follows the current fluid by R3's formulas; the inflow model, the tubing, the bean and the fluid's gravities stay as at the start.
- **Alternatives:** every well property fixed at its start (v1's `nsol-1`), simpler but without the drifts a virtual flow meter meets; the inflow model chosen again at every sample by R3's rule, which makes a step of up to a factor 1.8 in the rate at a large drawdown; Fetkovich's (1973) fall of the productivity below the bubble point, which R3's `sol-2` productivity does not have either, so `sol-2`'s and `nsol-2`'s late wells would differ; a workover that renews the tubing.
- **Realism:** $h$ falls with production time (Hasan and Kabir 2012, Eq. 2), roughness grows with corrosion and scale [WP §6, §10]. Draws: $h$ medians 8.0, 6.2 and 6.0 W/(m² K) at the start, the middle and the end.
- **Needs sign-off:** yes (sampling; the per-sample $h$ and $\varepsilon$ are new columns for R7).
- **Used by:** `nsol-2`, Well properties that age; R5 (a composite inflow).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-46 · `nsol-2`'s separator: two step-downs

- **Step:** R4, 2026-10-30.
- **Choice made:** the nominal separator factor $r_s$ is 1, then the mid-life draw's, then the smaller of it and the late draw's, with the steps at two sorted times of $U(0.25, 0.9)\thinspace T_\text{life}$; each sample adds D-39's ±10%.
- **Alternatives:** a gradual fall of the separator's pressure; no change over life (v1's SMP-26); a random number of steps.
- **Realism:** Oseberg B moved to a second stage at 20 bar after 18 years and Njord lowered its stage pressures in tail production (trade press, weak) [WP §7]. Draws: the end's nominal pressure 0.15 / 0.46 / 0.81 of the start's; 17.4% of `nsol-2` slots below 10 bar.
- **Needs sign-off:** yes (sampling; Bjarne's goal 2 in `nsol-2`).
- **Used by:** `nsol-2`, Separator over the life.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-47 · `nsol-2`'s control set points

- **Step:** R4, 2026-10-30.
- **Choice made:** piecewise-constant set points of the choke and the lift gas; each changes at an operating sample with a per-well probability of $\text{LogU}(0.05, 0.25)$, by a step of up to ±0.2 of its range with probability 0.6, else by a redraw (the choke: 0.05 at fully open, else $U(0.05, 1)$; the lift gas: 0 with probability 0.10, else uniform to the capacity); open loop.
- **Alternatives:** a new draw at every sample (v1's SMP-27), which is not an operating history; set points that drift towards fully open over the life, closer to late-life operations but narrower; changes on a fixed calendar.
- **Realism:** no public record of how often or how far operators move chokes and lift-gas rates [WP §8, §9]; Petrobras' 3W data may hold such histories (Resource list (R4 additions)). Draws: 17 choke changes per well (7 to 39), one about every 11 months at the median; 2.2% of the operating samples at fully open; 3.8% of the samples with lift on at zero lift gas.
- **Needs sign-off:** yes (sampling; the Goals' "mixed history").
- **Used by:** `nsol-2`, Control history.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G): the choke's set point changes at most samples, mostly by small steps, with occasional redraws, set on the rework's pilot; operators move chokes far more often than once in 11 months [own analysis of Petrobras' 3W Dataset]. Then, 2026-11-06, after the rework's pilot: a lifted well's lift-gas set point stays in [0.1, 1] of its capacity between changes (the walk reflects at 0.1, a fresh draw is uniform on it), and is shut in only by a redraw's chance of 0.2; the tests' lift sweeps keep covering 0 to the capacity. Without the floor the walk sat near zero for weeks and 4 of 110 drawn wells reached the redraw limit.

### D-48 · `nsol-2`'s test periods

- **Step:** R4, 2026-10-30.
- **Choice made:** 160 operating samples evenly over the life and 8 test periods of 5 samples at evenly spaced dates; a choke sweep (fully open, then one draw in each quarter of the range, downwards) in every test of a natural well, and choke and lift sweeps in turn for a lifted well.
- **Alternatives:** 4 or 12 tests; sweeps of 3 or 8 points; sweeps of both controls in every test (a lifted well's tests would take 10 slots); tests at random dates.
- **Realism:** multi-rate tests and lift performance curves are practice (Golan and Whitson 1991, §5.3), but no public schedule was found; 8 tests in 8 to 30 years is one every one to four years. Draws: 20.0% of the slots in tests; with them, 5.2% of `nsol-2` slots fully open, near `sol-2`'s 5.0%.
- **Needs sign-off:** yes (sampling; the Goals' "regular test periods").
- **Used by:** `nsol-2`, Test periods; The samples and the four tasks.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group G's bulk list): approved.

### D-49 · Gas lift that starts mid-life

- **Step:** R4, 2026-10-30.
- **Choice made:** lifted from the start if R3's initial-stage draw lifts the well; otherwise lift starts at the first of: R3's static check failing (every oil well), the end of natural flow in the life scan (every oil well, and gas wells that R3's late draw lifts), or a drawn time of $U(0.2, 0.9)\thinspace T_\text{life}$ (wells that R3's late draw lifts); designed by R3's lift block at the late stage.
- **Alternatives:** lift fixed for the life by R3's late-stage draw; lift only at the static check, which misses wells whose natural rate falls below the floor first; a design at the state on the start date, which needs R3's lift block run at a state R3 does not draw.
- **Realism:** natural flow ends after 1.7 years and gas lift follows in Golan and Whitson's (1991) Example 4.11; platforms run on gas lift at 80% water cut (Goh et al. 2007) [WP §9]. Draws (without the life scan): 12.6% of the wells lifted from the start, 29.0% later, at 0.20 / 0.48 / 0.85 of the life.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5; rule 2 depends on a solve, so the lift label of an `nsol-2` well does too, unlike R3's labels).
- **Used by:** `nsol-2`, Gas lift that starts mid-life; The end of a life.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group F's bulk list): approved.

### D-50 · The producibility filter, the best rate and the end of a life

- **Step:** R4, 2026-10-30. Applies R3's floor (D-34).
- **Choice made:** the best rate is the oil equivalent at a fully open choke, the nominal separator pressure and reservoir state, and the most of three lift-gas rates, $\lbrace 0,\ 1/2,\ 1\rbrace\thinspace w_{lg,\max}$; a well below half its floor, or without an operating point there, is dropped (`sol-2` at its stage, `nsol-2` at its start); an `nsol-2` life ends at the first of 21 scan points where the best rate falls below the same limit.
- **Alternatives:** the operating rate at nominal controls in place of the best rate, which drops wells for their choke; an optimised lift-gas rate (five or more solves); no end at the floor, with samples below it kept and labelled; the full floor without the margin.
- **Realism:** the floor's sources are R3's [WP §11]. Before any solve 2.0% of R3's wells have a design rate below half the floor; the solved share, and the lives the floor cuts, are R6's.
- **Needs sign-off:** yes (the margin is Bjarne's, and the filter is part of the dataset design, R7).
- **Used by:** Producibility filter; The end of a life; D-49; D-55.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint D3): a well whose best rate is below half its floor is dropped, with D-34's raised subsea floor.

### D-51 · SMP-28's three filters are dropped, and §1.11 lapses

- **Step:** R4, 2026-10-30.
- **Choice made:** no filter on the choke's spread, on the share of choked samples or on the variation of QTOT, in either dataset; v1's per-sample rate floors and failure counts go too; backlog §1.11 lapses.
- **Alternatives:** keep them, on the slots with an operating point, which drops wells for where they cannot flow; keep only the QTOT filter as a check on the bean (with §1.11's standard deviation to settle); keep the choked-share filter at a higher share.
- **Realism:** sonic flow at the choke is common and wanted, since it "stabilizes well production rate and separation operation conditions" (Guo et al. 2007, §5.2, p. 5/60); dropping wells choked in most samples removes high-pressure wells and wells at low separator pressures (goals 2 and 3). The drawn choke positions have a standard deviation of about 0.29 per well, against the old threshold of 0.055.
- **Needs sign-off:** yes (the filters are part of the dataset design, R7; §1.11 was left to this redesign by Bjarne, 2026-10-29).
- **Used by:** SMP-28's filters; What R6 must measure (the QTOT and choked shares as checks).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-52 · Fixed sample slots; outcomes recorded, nothing redrawn

- **Step:** R4, 2026-10-30.
- **Choice made:** each well has 200 slots whose inputs are drawn before any solve; each slot's outcome (`operating_point`, `no_operating_point` or `solver_failure`) is recorded; a slot without an operating point keeps its place, its inputs and a status in the samples table, with empty outputs; a target of below 0.5% of slots and of wells in every class for solver failures.
- **Alternatives:** redraw failed slots until 200 operating points, logging the redraws (v1's method with a log; the data would lack the conditions where wells stop flowing, as v1's lack 3.5% of its draws, L6); keep the slots without an operating point in a separate log, not in the samples table; drop `no_operating_point` slots from the data but keep the solver failures' inputs.
- **Realism:** wells do stop flowing at small openings, without lift gas or late in life; the draws put 1.11% of `sol-2` slots and 0.38% of `nsol-2` slots at $p_r \le p_s$ before any solve; the rest of the shares are R6's.
- **Needs sign-off:** yes (the schema and the meaning of "200 samples per well" are Bjarne's, R7; the 0.5% target is a tolerance).
- **Used by:** Slots and outcomes; How the rules meet acceptance criterion 4; R7.
- **Ruling:** Bjarne, 2026-11-02: reversed. "200 samples per well" means 200 operating points: a failed draw is redrawn until the well has 200, and every redraw is logged with its inputs and its reason. R6 measures how far the redraws shift the distributions (acceptance criterion 4); R7 revises the generation rules and the schema to match.

### D-53 · No operating point against solver failure

- **Step:** R4, 2026-10-30.
- **Choice made:** a slot has no operating point if $p_r \le p_s$, or if the search found no root and covered the whole range of $p_0$ (the Rust core's `non_finite` and `rejected` counts at 0); every other slot without an operating point is a solver failure with its reason, including a root set of unstable roots alone.
- **Alternatives:** every `NoOperatingPoint` counted as physical (simple, but a missed root then counts as the well's); every slot without an operating point counted as a failure (which hides where wells die); a second backend's verdict on every `NoOperatingPoint`.
- **Realism:** along SOL-3's curve $R(p_0)$ is positive at a low $p_0$, so a continuous $R$ has at least as many stable roots as unstable ones; a set of unstable roots alone has missed one. The Rust core misses roots on some void-fraction branches with the fitted classifier only, which `v2.0.0` leaves out (`specs/model/solution.md`).
- **Needs sign-off:** yes (sampling; it decides what the data call physical).
- **Used by:** No operating point or solver failure; D-54; D-55; the generation log.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint H2): as revised by D-102.

### D-54 · A second backend for solver failures

- **Step:** R4, 2026-10-30.
- **Choice made:** every solver failure of the Rust core is solved again by the CasADi backend, with the nearest operating point of the same well as an extra start; the slot takes what it finds, and the log records the backend.
- **Alternatives:** no retry (simpler; the failures stay); a retry by the Rust core with a finer scan; the CasADi backend for every slot (10 to 30 times the cost).
- **Realism:** not a question of realism: the retry changes no input, so it cannot bias the drawn distributions; it shrinks the share of failures. Its gain is unknown until R6 counts the slots it recovers.
- **Needs sign-off:** yes (a fallback of the generator: its measured gain is needed, constitution principle 7).
- **Used by:** A second backend for solver failures; D-55; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint H1): the CasADi retry of the Rust core's solver failures is dropped (the constitution's principle 7: no measured gain); R11 keeps D-92's CasADi re-solve of 100 rows.

### D-55 · Wells whose best-rate solve fails

- **Step:** R4, 2026-10-30.
- **Choice made:** a well with no operating point among its best-rate solves and a solver failure among them after the retry is dropped as `best_rate_solver_failure` and logged; above 0.5% of the wells in any class it is a solver finding to fix before R11.
- **Alternatives:** keep the well and judge its producibility from its slots at $u \ge 0.9$; v1's rule, which dropped a well whose first solve at $u = 0.5$ failed (SMP-28) or any of ten solves failed (SMP-29).
- **Realism:** the share is R6's to measure; dropping wells for solver failures favours wells that are easy to solve, which is why the share is bounded and reported by class.
- **Needs sign-off:** yes (sampling; the 0.5% is a tolerance).
- **Used by:** Wells whose best-rate solve fails; the generation log.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-56 · The starts

- **Step:** R4, 2026-10-30.
- **Choice made:** an extra start for the CasADi search only, since the Rust core scans: in `sol-2` the operating point nearest in the controls among the best-rate solves and the slots solved so far, in the order of $j$; in `nsol-2` the previous slot's.
- **Alternatives:** v1's single start at $u = 0.5$ (SMP-28); `nsol-1`'s pool of roots in the coordinates $(f_g, p_r)$ (SMP-29); no extra start.
- **Realism:** not a question of realism. A start can only add roots, and changes the operating point only in SOL-6's case of several stable roots, which the log flags.
- **Needs sign-off:** yes (sampling; part of the generation rules).
- **Used by:** Starts; D-54.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-57 · S-shaped wells as a third of the deviated class

- **Step:** applying Bjarne's ruling on D-9, 2026-11-02.
- **Choice made:** within the deviated class (20° to 60°), two thirds `build_and_hold` and one third `s_shaped`.
- **Alternatives:** no S-shaped wells (J only); half each.
- **Realism:** S wells are about half as common as J wells among the Gulf of Mexico's producing boreholes, and about as common in Dutch offshore platform gas wells [own analysis of BOEM's directional surveys and NLOG's deviation surveys]; a third of the deviated class sits between.
- **Needs sign-off:** yes (sampling).
- **Used by:** Geometry.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, groups B and C's bulk lists): approved.

Entries of R5, merged from its own file on 2026-11-02. The drafts they belong to are `specs/features/022-gas-rate-inflow.md` (the gas-rate inflow), `fluids-draft.md` (the table fluids) and `spec-changes.md` (the edits to decided specs). *Evidence* is `plans/evidence/gas_inflow_shapes.py` on R3's 703 gas and condensate wells of 2000 (seed 2026), and `plans/evidence/table_fluids.py`, a proof on four of R3's wells and one published fluid; neither solves a well. Every entry needs sign-off: a feature spec is approved before its implementation, and the rest sets part of the new `specs/sampling.md` or changes a decided spec.

### D-60 · The gas-rate inflow: Forchheimer's law in the pressure squared, on the reservoir gas's mass rate

- **Step:** R5, 2026-11-02. Takes D-23's form and WP-7 for the option.
- **Choice made:** feature 022 adds INF-10, $p_r^2 - p_0^2 = a\thinspace w_{g,\text{res}} + b\thinspace w_{g,\text{res}}^2$ with constant $a > 0$ and $b \ge 0$ in bar², s and kg, solved in closed form with an odd extension above $p_r$, and INF-11, the liquid from the gas by the fluid's ratios ($w_\text{res} = w_{g,\text{res}}(1 - f_g)/f_g$, INF-4 read backwards); the class `Forchheimer(a, b)`, taking the fluid through an optional argument of `liquid_mass_flow_rate`; the sampler's label `gas_rate` becomes `forchheimer`.
- **Alternatives:**
  - the backpressure law $q = C(p_r^2 - p_{wf}^2)^n$ (Rawlins and Schellhardt 1936): fits a four-point test as well and extrapolates to the AOF within −14% to +11% of the pseudo-pressure reference (Forchheimer in $p^2$: −22% to 0%); but its slope is infinite at $p_r$ and it is not real above it, it has no expression in the reservoir's properties for $n < 1$ (Golan and Whitson 1991, p. 132), and its $n < 1$ at low rates contradicts laminar flow there;
  - the pseudo-pressure, tabulated on the inflow, or from the fluid inside the rows: exact at any pressure; the first needs a table per well, the second an integral of $\mu_g Z$ in the CasADi graph;
  - the standard volume rate, the wellstream, or the total mass with water as the rate: each needs the fluid inside the inflow or a new fraction of it.
- **Realism:** 79% of R3's gas wells have $p_r$ above 138 bar, where $\mu_g Z$ is not constant and the pressure-squared form with $\mu_g Z$ at $p_r$ gives 14% less than the pseudo-pressure at the AOF (median; 28% at the 5th percentile). Fitted coefficients remove most of it (D-61). Vogel through the design point errs by −35% at 5% drawdown and +36% at the AOF; the interim productivity index by −43% and +105%.
- **Needs sign-off:** yes: a new model feature (its spec is approved before implementation), and the change to `v2.0.0`'s check that Bjarne ruled in the Goals.
- **Used by:** `specs/features/022-gas-rate-inflow.md`; `sampling-draft.md`, Inflow and Mapping; `spec-changes.md` (`validity.md`, `inflow.md`, `solution.md`, `README.md`, `architecture.md`); R6; R7's wells table (the columns `forchheimer_a` and `forchheimer_b`).
- **Ruling:** Bjarne, 2026-11-05 (checkpoint J1): approved as drafted. Feature 022 is Forchheimer's law in the pressure squared with constant $a$ and $b$ (INF-10), the liquid from the gas by the fluid's ratios (INF-11), with design choices 1 to 6 as the spec proposes; the backpressure law and a tabulated pseudo-pressure are not taken. The spec is approved for implementation in R9. The sampler's fit of $a$ and $b$ (D-61) and its non-Darcy coefficient (D-62) are ruled separately.

### D-61 · Forchheimer's coefficients in the sampler: fitted to the pseudo-pressure at the wellstream's gravity

- **Step:** R5, 2026-11-02. Revises D-23's mapping of $a$ and $b$.
- **Choice made:** the sampler computes the well's IPR in the pseudo-pressure, $(\mu_g Z)_ r\thinspace(m(p_r) - m(p_{wf})) = a_\text{ws} q + b_\text{ws} q^2$ with R3's $kh$, skin and $D$, and $\mu_g Z$ of the reservoir gas at $T_r$: the black-oil gas at the wellstream's gravity for a wet or dry gas, the table's gas for a condensate. It fits INF-10's $a$ and $b$ to it by least squares of $(p_r^2 - p_{wf}^2)/q$ against $q$ over $p_{wf}/p_r$ from 0.2 to 0.95 and the AOF, converts them from the wellstream to the separator gas ($a_q = c\thinspace a_\text{ws}$, $b_q = c^2 b_\text{ws}$, $c = 1 + C_{og}/R_{go}$; Whitson and Brulé 2000, Eqs. 3.73b and 3.73c), and to kg/s by $\rho_{g,\text{sc}}$.
- **Alternatives:** R3's $\mu_g Z$ at $p_r$ and the separator gas's gravity (conservative by 14% at the AOF, at the median); the coefficients scaled to the pseudo-pressure at the design drawdown (+4% at 5% drawdown, −8% at the AOF); a fit weighted towards small drawdowns, where the unweighted fit is 4% low at the median and 14% low at the 5th percentile.
- **Realism:** the fit is within 2% of the reference at the median and 6% at the 5th and 95th percentiles from 20% drawdown to the AOF. The wellstream factor $c$ is 1.01 / 1.05 / 1.14 for R3's condensates and at most 1.01 for wet and dry gas. `nsol-2` refits as $p_r$ falls, as R4 already recomputes $a$ and $b$ (D-45).
- **Needs sign-off:** yes (sampling).
- **Used by:** `sampling-draft.md`, Productivity; `specs/features/022-gas-rate-inflow.md`, Units; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved. Then, 2026-11-09 (second checkpoint, L3): $b = 0$ where the free slope is negative is kept; the $p^2$ form's limit at high pressure, where $\mu_g Z$ changes over the drawdown, and the fit's error there go into feature 022's spec and `validity.md` with R10's specs.

### D-62 · The non-Darcy coefficient from permeability, pay and the completion

- **Step:** R5, 2026-11-02. A finding on D-23's draw of $D$.
- **Choice made:** proposed for R3: $D$ from Economides et al.'s estimate (1994, Eq. 4-34), $D = 6\cdot10^{-5}\thinspace\gamma_g k^{-0.1} h/(\mu_g r_w h_\text{perf}^2)$ in field units, with a perforated share $h_\text{perf}/h \sim U(0.3, 1)$ and a log-normal factor of median 1 and log standard deviation 0.7 for the estimate's error ("may be off by 100%", Guo et al. 2008, as [WP §4] quotes), truncated to R3's $2\cdot10^{-4}$ to $2\cdot10^{-3}$ d/Mscf widened down to $10^{-5}$.
- **Alternatives:** R3's $D \sim \text{LogU}(2\cdot10^{-4}, 2\cdot10^{-3})$ d/Mscf, independent of $k$ and $h$, which Economides et al.'s "frequently of the order of $10^{-3}$" (p. 71) and Golan and Whitson's examples ($2.9\cdot10^{-4}$ and $1.5\cdot10^{-3}$, Examples 2.4 and 2.6) support; Eq. 4-34 alone, without the factor.
- **Realism:** R3's $D$ is 12 times Eq. 4-34's for a well perforated over its net pay, at the median (1.7 to 81 times, 5th to 95th percentiles), and the non-Darcy term is more than half of the pressure-squared drop at the design rate in 66% of R3's gas wells. With an assumed wellbore radius of 0.3 ft, Eq. 4-34 gives Golan and Whitson's Kansas well's $D$ within 10% for an assumed pay of 20 ft (their example gives only $kh$), and their Nebraska well's (20 md, 22 ft) seven times lower than its core-based value. With the proposal, thick, permeable reservoirs get less non-Darcy flow, as $D \propto 1/h$ says, and the design rates rise; R6 measures them against the Gulf's gas-well brackets.
- **Needs sign-off:** yes (sampling).
- **Used by:** `sampling-draft.md`, Productivity; R6; `sampler_draft_draws.py` (`NONDARCY`, `nondarcy_eq434`, `GAS_K_FACTOR`); `gas_rates_d62.py`.
- **Measured with feature 022** (2026-11-06, `gas_rates_d62.py`, 700 gas and condensate wells of the rework's draws, Forchheimer's inflow on the Rust core). Eq. 4-34 reproduces Golan and Whitson's Kansas well within 10%; on the rework's draws R3's $D$ is 8.7 times Eq. 4-34's at the median for a well perforated over its pay (the 12 above took the pay as $h_n f$, which the drains no longer use), and the non-Darcy term is over half the drop in 53% of the core gas wells. D-62 changes a well's median-sample rate by a factor of 1.24 at the median, while the core gas wells' rates lie several times above the regions' per-well medians under either $D$ [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]: the permeability is the lever, the rate going as $k^{0.6}$ to $k^{0.9}$. With the gas classes' permeability laws at the oil wells' times about 1/14, the core gas wells' median-sample rates lie within the regions' medians on platforms and near the deep-water subsea wells', 3.4% of them are flagged above the tubing's erosional capacity, and the low tail reaches liquid loading (7% of the core platform gas wells and 18% of the onshore ones are not producible).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint D2): D-62 as drafted, and the gas classes' permeability the well type's law (D-100) times one factor of 1/14, bounds included (medians about 1, 4 and 0.2 mD on platforms, subsea and onshore); the tubing's erosional capacity stays a check (D-100). D2's card is settled with it.

### D-63 · Synthetic compositions: matched to the GOR exactly, to the API and gas gravity by two levers

- **Step:** R5, 2026-11-02. Takes RF §8b's lead.
- **Choice made:** N₂ to C₆ from a Dirichlet mixture of the class's published examples, with C₂ to C₆ scaled by $f_\text{int}$; the heptanes-plus with $z_{7+}$ and $M_{7+}$ drawn per class, split into five fractions by Whitson's gamma distribution with $\alpha \sim U(0.5, 2.5)$, their gravities by Søreide with $C_f$; Newton's method on $(C_f, \ln f_\text{int})$ matches the drawn API and gas gravity to 0.5 °API and 0.005; recombination of the single-stage separator gas and stock-tank oil sets the GOR exactly; the saturation pressure follows from the flash (D-65). The descriptors are matched as single-stage values at standard conditions.
- **Alternatives:** perturbing one published example per class (RF §8b; fewer degrees of freedom, closer to measured fluids); a third lever, $M_{7+}$, for the saturation pressure at the cost of the API gravity; matching on a two-stage separation at the well's separator, which breaks the single tie line that PVT-TAB-1's GOR axis needs; shifting R3's targets to single-stage values by the examples' offsets (GOR +12% to +19%, API −3 to −6 °API, gas gravity +0.05 to +0.07).
- **Realism:** the proof matches within 0.01 °API and 0.0003 in 3 iterations for all four wells; the heptanes-plus contents (15.1 and 17.9 mol% for volatile oils, 1.4 and 6.0 for condensates) lie within McCain's limits for their classes. The single-stage basis makes a fluid matched to a field-reported API about 3 to 6 °API lighter in field terms; no statistic of single-stage descriptors is in hand.
- **Needs sign-off:** yes (sampling).
- **Used by:** `fluids-draft.md`, The composition; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint J2): per-fluid offsets: in R9 each matched composition is flashed through the well's own separator and through one stage at standard conditions, and the matching targets are shifted by the difference (one or two passes), so that the fluid's field-reported GOR, API and gas gravity are the drawn ones; the single tie line is kept and the offsets are recorded with the fluid.

### D-64 · Condensates in tables: a trace of liquid where the stream is all vapour, and $R_{so}$ allowed to fall with pressure

- **Step:** R5, 2026-11-02. A finding on feature 018's tables, which no condensate had tested.
- **Choice made:** PVT-TAB-6's tables keep a liquid mass fraction $\delta = 10^{-4}$ of the stream where the flash has less, with the liquid's properties at the nearest two-phase node, so that $B_o > 0$; PVT-TAB-1's check that $R_{so}$ does not fall as the pressure rises is dropped (proposed text in `spec-changes.md`).
- **Alternatives:**
  - allow $B_o = 0$ where $R_{so} = -\rho_o/\rho_{g,\text{sc}}$, the oil phase empty: exact, but PVT-MIX-6's liquid density is then $0/0$ for a fluid without water, so `FluidModel` would refuse such a table without water;
  - keep the check only for tables that are oils, by a flag on the table (`kind = 'dew'`): keeps the check where it caught errors;
  - no condensate tables: condensates stay black oil, which keeps their condensate liquid in the reservoir and at the bottomhole (proof: a free-gas share of 0.43 against the table's 0.84 above the dew point).
- **Realism:** every condensate is retrograde below its dew point: the proof's three condensate tables have $R_{so}$ falling with pressure at 176 to 316 pairs of neighbouring nodes, and reach $B_o$ down to $5\cdot10^{-4}$ with $\delta$. With the check relaxed, `FluidModel` builds on each and its phase rates and densities are finite on floats; no well was solved. The Rust core does not check monotonicity; whether its march holds is R6's measurement.
- **Research** (Bjarne asked, 2026-11-02, how other simulators do it; `sources-condensate-tables.md`): no tool checks the order of a net transfer such as $R_{so}$; OLGA's table generators fill an absent phase from the nearest node, as the trace does; the modified black oil's $R_v$ maps onto ManyWells' columns at one GOR, not back, and is not monotone itself. It recommends this choice plus two checks that hold for every fluid: $\rho_o + R_{so}\rho_{g,\text{sc}} > 0$, and $p/Z$ strictly increasing in $p$ (option A′).
- **Needs sign-off:** yes, at the checkpoint (`checkpoint.md`, K1): it changes `specs/model/pvt/table.md` (PVT-TAB-1 and PVT-TAB-6), a decided spec.
- **Used by:** `fluids-draft.md`, The table; `spec-changes.md`; feature 022's `develop+gas well+pvt table` comparison; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint K1): option A′, implemented in R9: a liquid trace of $10^{-4}$ where the flash has none, no order check on $R_{so}$, and two new checks (the oil phase's mass positive, $p/Z$ rising with $p$); the text proposed for `specs/model/pvt/table.md` in `spec-changes.md`.

### D-65 · The table's saturation pressure replaces R3's, and a fluid is saturated at discovery where it must be

- **Step:** R5, 2026-11-02. Revises D-17 for table fluids.
- **Choice made:** for volatile oils and condensates, $p_\text{sat}$ at $T_r$ is the table's (TP flashes with NeqSim's check for more phases, bisected to $10^{-3}$ bar), not $p_i(1 - x)$; where it exceeds $p_i$, the GOR moves along the fluid's tie line until $p_\text{sat}(T_r) = p_i$, and where no GOR within a factor 4 does it, the reservoir fluid is the class's phase in equilibrium at $(p_i, T_r)$. The kind (bubble or dew) and the rule are labels.
- **Alternatives:** match $p_\text{sat}$ by a third lever (D-63); redraw the composition until $p_\text{sat} \le p_i$, which shapes the compositions by a filter.
- **Realism:** the proof's saturation pressures against R3's: 386 against 513, 359 against 339, 255 against 250 and 289 against 351 bar. Two of four fluids would have been above their saturation pressure at $p_i$: one moved to 0.73 of its GOR, one took the equilibrium vapour at 1.73 times its GOR and 3 °API lighter. R3 draws 30% to 40% of these classes saturated at discovery; the share the rule adds is R6's to measure.
- **Needs sign-off:** yes (sampling).
- **Used by:** `fluids-draft.md`, The saturation pressure; `sampling-draft.md`, Saturation pressure and Reservoir state; R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved.

### D-66 · Six published example fluids, a fifth of the table wells, recombined to the drawn GOR

- **Step:** R5, 2026-11-02. Takes RF §8a.
- **Choice made:** with probability 0.2 a table well takes one of its class's published examples, with equal shares, recombined along its single-stage tie line to the drawn GOR: condensates from Whitson and Brulé (2000, Table 2.1) and Pedersen et al. (2015, Tables 1.3 and 3.8); volatile oils from Whitson and Brulé's near-critical oil and volatile oil (Table 2.1) and Pedersen et al.'s Table 1.4.
- **Alternatives:** the examples at their own GORs, with their descriptors replacing the drawn ones (their classes by McCain's limits then differ: Whitson and Brulé's volatile oil is a black oil at 265 Sm³/Sm³ and their near-critical oil a condensate at 650); no examples, only synthetic fluids; a larger share.
- **Realism:** about 500 of a dataset's 2500 table wells, each example used by about 80; their stock-tank oil and gas keep their published composition. The one example with a measured dew point at a stated temperature, Pedersen et al.'s Table 3.8, has 356 bar in the flash against 388 measured.
- **Needs sign-off:** yes (sampling).
- **Used by:** `fluids-draft.md`, Published example fluids; R7 (`fluid_source`, the `pvt` rows' source).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved.

### D-67 · One table set per well, on its own grid and GOR axis

- **Step:** R5, 2026-11-02. Answers D-36's open choice and R7's question.
- **Choice made:** each table well has its own `PVTTableSet`: temperatures every 10 K from below $T_s - 15$ K to $T_r$, pressures from 1 bar to $1.05\thinspace p_i$ plus a step with each temperature's saturation pressure as a node, and GORs at most 25% apart over every GOR its samples reach.
- **Alternatives:** a library of a few hundred fluids per class with GOR axes over the class's range, assigned by the nearest API and gas gravity: fewer tables, but rounded descriptors, repeated fluids and larger tables.
- **Realism:** the proof's tables take 0.5 to 4.6 s per GOR for 370 to 1116 nodes; a `sol-2` well with 3 GORs 2 to 17 s, an `nsol-2` well with up to 11 GORs 6 to 55 s; about 0.6 GB per process. A dataset's 2500 table wells take at most about 3 hours on 12 processes, and up to 0.75 GB of CSV for `sol-2` before compression.
- **Needs sign-off:** yes (sampling; the size bears on R7's schema).
- **Used by:** `fluids-draft.md`, The table and One table per well; `datasets-draft.md`'s `pvt` table (R7).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved.

### D-68 · NeqSim as a build-time tool of the generator, outside ManyWells' dependencies

- **Step:** R5, 2026-11-02.
- **Choice made:** the generator makes the tables with NeqSim 3.4.0 (Java) in a process environment of its own, as `pvt_table_eos.py` does, pinned in the generation's record; ManyWells' package does not depend on it. The dataset carries the tables, so that a user re-solves without NeqSim (`datasets-draft.md`, Re-solving a sample).
- **Alternatives:** tables made once into a library committed with the generator (no Java at generation time; D-67's alternative); NeqSim a dependency of an optional extra of `manywells`.
- **Realism:** NeqSim needs a JVM; under a memory cap of 8 GB its heap must be capped (`-Xmx2g`), or it fails to start threads. The non-goal (no equation of state in ManyWells) holds: the flash runs before the model, as feature 018 decided.
- **Needs sign-off:** yes (the generation's environment, R7 and R10).
- **Used by:** `fluids-draft.md`; R6, R10 and R11.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved.

### D-69 · R6's gas wells before the feature: the interim index on the core, a prototype subclass on CasADi

- **Step:** R5, 2026-11-02.
- **Choice made:** R6 solves R3's gas and condensate wells with the interim productivity index on both backends, as drafted, and a subset with a prototype of INF-10 and INF-11 as an `InflowModel` subclass in `plans/evidence/` (it holds $f_g$ of its well), which only the CasADi backend runs; the difference measures what the feature changes before it exists. The condensate tables wait for D-64's ruling, or use a `PVTTable` subclass without the check, on the CasADi backend only.
- **Alternatives:** the interim mapping alone (it errs by a factor 0.57 to 2.05 in rate across the drawdowns, at the median); Vogel for gas wells (v1).
- **Realism:** the plan allows a prototype where the model allows it (Step R5); a user's subclass runs on the CasADi backend (architecture, decision 1).
- **Needs sign-off:** no beyond the plan; listed so that Bjarne sees what R6's gas-well numbers rest on.
- **Used by:** R6.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group J's bulk list): approved.

Entries of R7, merged from its own file on 2026-11-02. Entries D-80 to D-85 apply Bjarne's ruling on D-52 (2026-11-02: "200 samples per well" means 200 operating points; a failed draw is redrawn until the well has 200, and every redraw is logged with its inputs and its reason) to R4's generation rules in `sampling-draft.md` (The samples within a well). Entries D-86 to D-94 are the dataset design in `datasets-draft.md`. No solve was run: the shares that decide these entries are R6's to measure, and each entry says which. Every entry needs sign-off, because each sets part of the new `specs/sampling.md` or the dataset schema (`AGENTS.md`, Needs Bjarne's sign-off).

### D-80 · Redraws from the slot's own generator, solved in passes

- **Step:** R7, 2026-11-02. Applies the ruling on D-52.
- **Choice made:** attempt $a \ge 1$ of slot $j$ of well $k$ draws from `rng_for(seed, k, j, a, 'v2', 'redraw')`, and the first attempt keeps R4's generator; in `sol-2` a redraw draws the slot's whole draw again (choke, lift gas, separator pressure and drift, D-37 to D-40); a well's slots are solved in passes, pass $a$ solving attempt $a$ of every slot still without an operating point.
- **Alternatives:** one sequential generator per well for all redraws, under which a redraw in one slot shifts every later slot's redraws, and the result depends on the order of the solves; redrawing only the control that "caused" the failure, which no rule can identify in general (a well dies from the choke, the lift gas, the separator and the drift together); solving slot by slot, with each slot's redraws before the next slot, which gives the same draws but no batches.
- **Realism:** not a question of realism. The redraw distribution is the slot's own, so the samples are the drawn distribution conditioned on an operating point, and the run is deterministic, independent of the order of the wells and the number of processes (SMP-31). The passes make each pass a batch for the batch API (D-93).
- **Needs sign-off:** yes (sampling; the generation rules).
- **Used by:** `sampling-draft.md`, Conventions of the samples, Slots and outcomes, How the rules meet acceptance criterion 4; `datasets-draft.md`, `samples.attempt`, The generation.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-81 · The redraw limits, and what happens to a well that reaches one

- **Step:** R7, 2026-11-02. Applies the ruling on D-52.
- **Choice made:** at most 50 attempts per slot, at most 400 redraws per well (so that a kept well flows in at least a third of its draws) and at most 10 solver failures per well after the CasADi retry; a well that reaches a limit is dropped, logged as `slot_limit`, `redraw_limit` or `solver_failure_limit` with all its attempts, and replaced by the next drawn well; a target of at most 1% of the wells in any class dropped at the limits, above which the draws or the limits are revisited before R11.
- **Alternatives:** v1's limits, $5 \cdot 500$ attempts and 100 failures per well (SMP-28), which allow wells that flow in a fifth of their draws; no limit per well, only per slot, which lets a weak well take thousands of solves; a lower per-well limit of 200 redraws (half its draws flow), which drops more weak wells; no solver-failure limit, which keeps wells with many failures and redraws them all.
- **Realism:** at a per-attempt failure share $p$ a well needs about $200\thinspace p/(1 - p)$ redraws: 11 at $p$ = 5%, 50 at 20%, 200 at 50%, 400 at 67%. R4's draws give 1.1% of `sol-2` first attempts at $p_r \le p_s$, and 10% lift-off attempts in lifted wells, which a well that needs lift cannot flow at; v1 failed in 3.5% of its draws (L6). Weak wells that pass the producibility filter but flow only near fully open are the wells the limit drops; their share is R6's. With $p \le 2/3$ the chance that a `sol-2` slot needs more than 50 attempts is below $10^{-8}$, so in `sol-2` the per-slot limit only guards against a slot that cannot flow at all; in `nsol-2` it defines a dead date (D-83).
- **Needs sign-off:** yes (the limits and the 1% target are tolerances; dropping wells shapes the wells' distributions, criterion 4).
- **Used by:** `sampling-draft.md`, Slots and outcomes, How the rules meet acceptance criterion 4, What R6 must measure; `datasets-draft.md`, `drawn_wells.outcome`, Cost.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint H3): the targets and limits kept, judged on R11's pilot.

### D-82 · Both outcomes are redrawn; a solver failure after the CasADi retry

- **Step:** R7, 2026-11-02. Applies the ruling on D-52; amends D-54.
- **Choice made:** an attempt with `no_operating_point` is redrawn at once; a Rust `solver_failure` first gets D-54's CasADi retry with the same inputs and is redrawn only if the retry fails; a recovered attempt is the sample and is logged as `recovered`; the two kinds of redraw are counted apart.
- **Alternatives:** redraw both at once, without the retry, which is cheaper but conditions the samples on where the Rust core fails; drop the well at its first unrecovered solver failure, which conditions the wells on the solver instead (D-55 does this only for the best-rate solves); treat a solver failure as fatal for its slot, which leaves fewer than 200 operating points, against the ruling.
- **Realism:** a `no_operating_point` redraw conditions the data on the physics: the samples are where the well flows. A `solver_failure` redraw conditions them on the solver, which criterion 4 rules out; the retry shrinks that share, and the target of D-52's draft, below 0.5% of the attempts and of the wells in every class, bounds the rest. The share is R6's.
- **Needs sign-off:** yes (sampling; the retry is solver machinery whose gain R6 measures, constitution principle 7; the 0.5% target is a tolerance).
- **Used by:** `sampling-draft.md`, Slots and outcomes, A second backend for solver failures, How the rules meet acceptance criterion 4; `datasets-draft.md`, `redraws.outcome`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoints H1 and H3): solver failures redrawn and logged without the CasADi retry; the targets kept and judged on R11's pilot.

### D-83 · `nsol-2`'s redraws: the date kept, the controls changed for one slot; a failed date drops the well

- **Step:** R7, 2026-11-02. Applies the ruling on D-52 to `nsol-2`.
- **Choice made:** a redraw keeps the slot's date and the reservoir's state there (pressure, GOR and water with the date's drift, $h$, $\varepsilon$, productivity, nominal separator pressure, lift installed or not), and draws again the controls, each by D-47's set-point change applied to the slot's planned value (a step of up to ±0.2 of its range with probability 0.6, else a new draw), and the separator pressure as D-39; the redrawn controls hold for that slot only; a slot that reaches 50 attempts is a failed date, and the well is dropped as `dead_date`.
- **Alternatives:** new draws from the set points' start distribution in every redraw, which escape a dead region at once but put isolated jumps in the history; a redrawn set point that persists to the later slots, closer to an operator who finds a working setting and keeps it, but then the history follows the solves (feedback, against the Goals' open loop); a redraw of the date as well, which breaks the even spacing of the life; at a failed date, a rescan of the life with the date as an extra scan point and a rerun of the well, which keeps the well at the cost of new machinery.
- **Realism:** an operator whose well stops at a set point changes the set point; a one-slot change from the planned value is the nearest open-loop analogue. With D-84 every date lies between scan points where the best rate passes, so failed dates should be rare; R6 measures their share, and above 0.5% of the `nsol-2` wells a finer life scan is the fix.
- **Needs sign-off:** yes (sampling; the Goals' open loop and "a well's 200 samples span its life").
- **Used by:** `sampling-draft.md`, Redraws in `nsol-2`, Test periods, How the rules meet acceptance criterion 4; `datasets-draft.md`, `samples.attempt`, `drawn_wells.outcome`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint H3): the targets and limits kept, judged on R11's pilot. Then, 2026-11-06: a redraw changes the controls by R4's change (a step of at most 0.2 with probability 0.6, else a fresh draw), not by D-47's smaller steps as ruled, which left 7 of 113 drawn wells at the redraw limit in the rework's pilot.

### D-84 · `nsol-2`'s life ends at the last scan point that passes, and lift starts before natural flow fails

- **Step:** R7, 2026-11-02. Amends D-50 and rule 2 of D-49, which assumed fixed slots.
- **Choice made:** $T_\text{end}$ is the last of the 21 scan points before the first that fails (best rate below half the floor, no operating point, or a water cut above 0.98), or $T_\text{life}$; lift starts by rule 2 at the last scan point before the natural well's best rate first fails; a well whose first scan point after the start fails is dropped as `short_life`.
- **Alternatives:** R4's first failing point, at whose date no redraw need succeed, so a well would end with a dead date by construction; a bisection between the last passing and the first failing point, which locates the end within a fraction of the 5% step at a few more solves; at a failing date, keep the slots and count them as dead (the reversed D-52).
- **Realism:** the life ends up to 5% of $T_\text{life}$ earlier, and lift starts up to 5% earlier, than R4 drafted: within the resolution of the scan, and on the side of the floor's margin. The share of lives cut and of `short_life` wells is R6's.
- **Needs sign-off:** yes (sampling; the producibility filter's use over the life, D-50).
- **Used by:** `sampling-draft.md`, The end of a life, Gas lift that starts mid-life, Producibility filter, Redraws in `nsol-2`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-85 · The redraw log and the well log, published with each dataset

- **Step:** R7, 2026-11-02. Applies the ruling on D-52 ("every redraw is logged with its inputs and its reason").
- **Choice made:** two logs per dataset, published next to its tables: the redraw log, one row per attempt that was redrawn, recovered by the CasADi retry or made by a dropped well, with its inputs, outcome, reason, retry and root counts; and the well log, one row per drawn well with its inputs, labels and outcome; the attempts that gave a sample at once are not logged beyond the samples, and their search counts and wall times stay with the run.
- **Alternatives:** keep the logs with the run and publish only counts, which hides the shift that the redraws cause from users; log every attempt with its search counts and wall time (about 2 million rows per dataset of solver diagnostics, of little use to users); put the attempts without an operating point in the samples table with a status (R4's reversed draft).
- **Realism:** the logs let a user measure the bias the redraws cause (criterion 4), and the `no_operating_point` attempts mark where each well cannot flow, which surrogate models and production optimisation can use. They are synthetic and hold nothing confidential.
- **Needs sign-off:** yes (part of the dataset schema; whether to publish the logs is Bjarne's).
- **Used by:** `sampling-draft.md`, Slots and outcomes, The generation log; `datasets-draft.md`, `redraws`, `drawn_wells`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint I1): the logs (`redraws`, `drawn_wells`) are published as a separate companion file next to each dataset, not in it.

### D-86 · The datasets on Hugging Face: the same repository, new configs, parquet

- **Step:** R7, 2026-11-02.
- **Choice made:** add `manywells-sol-2` and `manywells-nsol-2` to the repository `solution-seeker-as/manywells`, under `v2/`, as parquet files with one config per table (`manywells-sol-2` for the samples, `-wells`, `-geometry`, `-pvt`, `-redraws`, `-drawn-wells`), the samples and wells with a train and a test split; the v1 files untouched.
- **Alternatives:** a new repository, `solution-seeker-as/manywells-2`, which keeps v1's card as it is but splits the project's datasets in two places and two citations; zipped CSV as v1, which loses the types (categories, nulls, booleans) and is larger; one parquet file per dataset with the well columns repeated on every sample row, which multiplies the wells' 110 columns by 200.
- **Realism:** not a question of realism. Parquet with configs is what the Hugging Face loader and viewer read directly, and `load_dataset(…, name='manywells-sol-2')` gives the samples as `name='manywells-sol-1'` gives v1's.
- **Needs sign-off:** yes (publishing is release work; the repository is Bjarne's choice).
- **Used by:** `datasets-draft.md`, The datasets.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

### D-87 · Four tables and two logs, each input in one place

- **Step:** R7, 2026-11-02.
- **Choice made:** each dataset has the tables `wells` and `samples` of the decided layout, plus `geometry` (one row per survey station) and `pvt` (one row per node of a table, referenced by an ID), and the two logs (D-85); an input constant over a well's samples is in `wells`, one that varies is in `samples`, so in `nsol-2` the heat-transfer coefficient, roughness, inflow coefficient and lift's installation move to `samples`.
- **Alternatives:** the survey as list columns of `wells` (`survey_md`, `survey_tvd`), which keeps the two tables of `specs/goals.md` but is awkward in CSV, SQL and some tools; a PVT table per well as list columns, which repeats a library's tables over its wells; duplicate copies of each input in both tables for convenience, which invites them to disagree.
- **Realism:** not a question of realism. A survey has 2 to about 150 stations and many wells may share a table (R5), so neither fits a row. The design needs `specs/goals.md`'s "two tables" to name the others (`spec-changes.md`).
- **Needs sign-off:** yes (the dataset schema; the edit to `specs/goals.md`).
- **Used by:** `datasets-draft.md`, The datasets, The schema; `spec-changes.md`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved, with the two logs in a companion file next to each dataset (I1).

### D-88 · Column names and types

- **Step:** R7, 2026-11-02.
- **Choice made:** the samples keep v1's features by name and add new per-sample columns in v1's style, short and upper case (`PRES`, `GOR`, `WLR`, `STAGE`, `HTC`, …), with `slot` and `attempt` as keys; the wells table names model inputs by their code names in `nomenclature.md` and descriptors by short lower-case names; `WEEKS` stays as a float from first production; `FGAS`, `FOIL` and `FWAT` stay, derived; every continuous column is float64, labels are dictionary-encoded strings, and a value that does not apply is null.
- **Alternatives:** model code names in the samples too (`p_r`, `gor`, `u`), which matches the API but breaks v1 users' code for the kept features; a time column in years (`YEARS`) in place of `WEEKS`; float32 for the outputs, which halves their size but loses digits a re-solve check compares; dropping `FGAS`, `FOIL` and `FWAT`, which are no longer drawn.
- **Realism:** not a question of realism.
- **Needs sign-off:** yes (the dataset schema).
- **Used by:** `datasets-draft.md`, The schema, What changes against v1.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

### D-89 · Labels as strings with fixed codes

- **Step:** R7, 2026-11-02.
- **Choice made:** every label is stored as its string (dictionary-encoded in parquet), with a fixed integer code per category given by its position in `datasets-draft.md`, Section 4; a later version adds categories at the end.
- **Alternatives:** integer codes in the files with a mapping file, as 3W numbers its event types (Vargas et al. 2019, Table 2), which is compact but not self-describing; Hugging Face `ClassLabel` features declared in the card, which gives both in the `datasets` loader but not in pandas or SQL.
- **Realism:** not a question of realism.
- **Needs sign-off:** yes (the dataset schema).
- **Used by:** `datasets-draft.md`, Labels and codes.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

### D-90 · A train–test split by well, 80/20, stratified

- **Step:** R7, 2026-11-02. A working assumption of the Goals ("a train–test split by well").
- **Choice made:** 20% of each dataset's wells are `test`, chosen as every fifth well within each stratum of `fluid_type` × `well_type` × `geometry_class` × `coverage_tail`, in an order drawn from `rng_for(seed, 'v2', 'split')`; tail wells are in both parts; no validation split and no split within wells.
- **Alternatives:** a random 20% without strata, under which small classes (onshore volatile oils in the tail) can miss the test set; 70/15/15 with a validation split, which users can carve from `train`; a test set of core wells only; a temporal split within `nsol-2` wells, which serves forecasting but not learning across wells.
- **Realism:** not a question of realism. A split by row would put a well's samples in both parts and measure interpolation within wells.
- **Needs sign-off:** yes (the dataset design).
- **Used by:** `datasets-draft.md`, The split and the documentation; `wells.split`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

### D-91 · Seeds, IDs and the draw index

- **Step:** R7, 2026-11-02.
- **Choice made:** each dataset has its own seed, so the wells of `sol-2` and `nsol-2` are independent draws; `ID` numbers the first 10,000 accepted wells from 0 in draw order, unique within a dataset only, as v1's; `draw_index` keeps each well's $k$, which seeds its draws and links the logs to the wells.
- **Alternatives:** one seed for both, so that `nsol-2` well $k$ is `sol-2` well $k$ drawn at its initial stage, which links the datasets but puts the same wells in one dataset's train set and the other's test set; `ID` equal to $k$, which leaves gaps where wells were dropped.
- **Realism:** not a question of realism.
- **Needs sign-off:** yes (the dataset design).
- **Used by:** `datasets-draft.md`, The generation, `wells`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

### D-92 · The re-solve check and its tolerance

- **Step:** R7, 2026-11-02.
- **Choice made:** a sample re-solved from its rows, with the recorded code version and backend on the generator's platform, gives identical features; elsewhere the features agree within the verifier's `tol_x` applied to them: pressures within $10^{-4}\thinspace(p_r - p_s)$, temperatures within $10^{-4}\thinspace(T_r - T_s)$, mass rates within $10^{-4}$ of `WTOT`, and `CHOKED` and the regimes the same except within the choke band and the transition width; R11 checks 1000 rows per dataset, 100 of them with the CasADi backend.
- **Alternatives:** store each operating point's full state (707 values per sample, about 11 GB per dataset), which makes the check exact without a re-solve; check the published rows only for their relations (SMP-28's row check), which does not test reproducibility.
- **Realism:** not a question of realism. The Rust core is deterministic, and float64 round-trips through parquet; `tol_x` is about 0.01 bar and 0.01 K (`specs/verification.md`, Tolerances).
- **Needs sign-off:** yes (a tolerance; the constitution's Dataset reproducibility).
- **Used by:** `datasets-draft.md`, Re-solving a sample, Checks at generation.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint I2): accepted: identical features on the generator's platform and code version, elsewhere within the verifier's `tol_x` applied to the features.

### D-93 · The generator: the process pool, resumable chunks and a clean commit

- **Step:** R7, 2026-11-02.
- **Choice made:** the library's process pool runs one well per task; the batch API (backlog §4.3) is built only if R6 measures a gain of more than about 20% in the generation's wall time; the generator writes chunks of 100 draw indices atomically, with the run's settings in each, skips completed chunks on a restart, and assembles the published files in a last step; it refuses to write a published dataset from a working tree with changes.
- **Alternatives:** the batch API first, since the redraw passes are natural batches (D-80), which saves the time outside the core and the memory of 12 processes, at the cost of solver machinery not yet measured; one output file per well (20,000 files); no resumability, rerunning from the start after a stop of a run of about 10 hours.
- **Realism:** not a question of realism. At 0.15 s per solve each dataset takes about 85 to 120 CPU hours, 8 to 10 hours on 12 processes (`datasets-draft.md`, Cost); R6 measures the mean solve time, the time outside the core and the pool's speed-up.
- **Needs sign-off:** yes (new generator machinery; the batch API is solver machinery under principle 7).
- **Used by:** `datasets-draft.md`, The generation.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group H's bulk list): approved.

### D-94 · A datasheet for each dataset, in the repo and on the card

- **Step:** R7, 2026-11-02. A working assumption of the Goals ("a datasheet for each dataset (Gebru et al. 2021)").
- **Choice made:** one datasheet per dataset, following Gebru et al.'s seven sections (§3.1 to §3.7), in the repo as `docs/datasheet-sol-2.md` and `docs/datasheet-nsol-2.md`, written in R11 and copied to the dataset card; it states the designed population (even geometry mix, fluid floor, coverage tail), the model's limits, the redraws' conditioning, and what not to use the data for.
- **Alternatives:** one datasheet for both datasets, shorter since most answers are shared, against the Goals' "for each"; the datasheet on the card only, which leaves it unversioned with the code that made the data.
- **Realism:** not a question of realism.
- **Needs sign-off:** yes (release documentation).
- **Used by:** `datasets-draft.md`, The split and the documentation.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint, group I's bulk list): approved.

Entries of R6, merged from its own file on 2026-11-02. Each entry changes the draft where the prototype's measurements showed it failing a goal or an acceptance criterion (`r6-measurements.md`). The numbers are the prototype's (`plans/evidence/sampler_prototype.py` and `sampler_prototype_report.py`), in the interim mapping: volatile oils and gas condensates as the black oil, and gas and condensate wells on a productivity index (`sampling-draft.md`, Mapping). Every entry needs sign-off, because each sets part of the new `specs/sampling.md` (`AGENTS.md`, Needs Bjarne's sign-off).

### D-100 · The permeability's median lowered to 15 mD offshore and 3 mD onshore

- **Step:** R6, 2026-11-02. Revises D-20.
- **Choice made:** $k \sim \text{LogN}_ {[0.5, 3000]}(15, 1.5)$ mD offshore and $\text{LogN}_ {[0.05, 1000]}(3, 1.5)$ mD onshore, in place of medians of 50 and 10 mD; the spread, the bounds and the tail's $\text{LogU}$ stay. Every productivity, the design rate, the tubing and the lift capacity follow from it as drafted.
- **Alternatives:**
  - the draft's 50 and 10 mD: the core offshore oil wells' median-sample rate is 457 Sm³/d of oil equivalent at the median and 1538 on average, with 34.6% of the wells above 1018 Sm³/d and 24.5% above 2035 Sm³/d, against 5.5% and 1.3% of the Gulf of Mexico's oil wells in 2024 (EIA 2025, Table B32) [WP §5];
  - 10 and 2 mD: from the pilot's slope of the log rate on the log productivity index (0.89), about 110 Sm³/d at the median, 12% above 1018 Sm³/d and 2.4% above 2035 Sm³/d;
  - a narrower spread ($s$ = 1.0 in place of 1.5), which trims the top without moving the median;
  - smaller beans (D-27): the median sample makes 92% of the well's best rate, so the choke holds back little of the rate; but the choke's span is ruled close to uniform, and the beans' sources are two textbook examples.
- **Realism:** the rate is set by the productivity index (Spearman 0.84 between a core offshore oil well's median-sample rate and its $J$) and by the tubing sized for it (0.82). Reservoir permeabilities have medians of 10 to 570 mD by porosity (Ehrenberg and Nadeau 2005, Table 1), so 15 mD lies at the low end of the reservoirs: it stands for a well's effective permeability, which also carries what the model lacks, such as the oil's relative permeability below the bubble point and with water, a partial completion, and operating limits that hold a well below its potential. EIA's brackets are calendar-day averages of a mature shelf, and an offshore Norwegian well makes at least 10,000 bbl/d (Golan and Whitson 1991, Preface), so the median is not matched to the Gulf's: the target is the share above 1018 Sm³/d. Measured after the change (R6's `sol-2` pilot, seed 2026): 22.7% of the core offshore oil wells above 1018 Sm³/d and 11.6% above 2035 Sm³/d, a median of 197 and a mean of 822 Sm³/d; gas and condensate wells 0.9 million Sm³/d at the median (1.8 before); 5.5% of the core wells dropped as not producible (3.7% before). The share above 1018 Sm³/d is still four times EIA's: how close to come is Bjarne's call (`r6-measurements.md`, For Bjarne's ruling).
- **Needs sign-off:** yes (sampling; it revises D-20 against D-24's check).
- **Used by:** `sampling-draft.md`, Productivity; `sampler_draft_draws.py` (`PERM`); every rate in `r6-measurements.md`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint D1): the permeability drawn by well type, each type's median and spread set on the rework's pilot against the per-well rates, keeping the support wide. The tubing's erosional capacity stays a check: D-25's test flags a well whose design rate exceeds the largest tubing's capacity, the rework's pilot measures the flagged share, and a redraw is added only if it stays above a few percent. Keep the method simple. Then, 2026-11-09 (second checkpoint, L5): the platform oil wells' upper tail is kept, under coverage first, and measured by well type on R11's pilot.

### D-101 · `sol-2`: an oil well that cannot flow at its best rate is lifted, not dropped

- **Step:** R6, 2026-11-02. Amends D-28 and D-50 for `sol-2`.
- **Choice made:** a natural oil well whose best rate fails (no operating point, or below half its floor) gets R3's lift design, drawn from its own lift block as if the assignment had lifted it, with the label `lift_reason = no_natural_flow`; its best rate is solved again with lift, and it is dropped as not producible only if that fails too.
- **Alternatives:** the draft, which drops it; a margin on the static check (D-28), which needs no solve but would lift wells that flow; the composite of both.
- **Realism:** an operator lifts an oil well that stops flowing rather than abandon it (Golan and Whitson 1991, Example 4.11; gas lift is the main lift offshore, [WP §9]), and `nsol-2` already does so by D-49's rule 2 (D-84). The static check compares $p_r$ with a no-slip column at mean conditions and leaves out friction and slip. In the draft's pilot it left 9 of 380 core wells natural (2.4%) that then failed the best rate: 7 oil wells with no operating point at a fully open choke (subsea wells in mid-life at $p_s$ of 37 to 72 bar, and late wells at water cuts of 0.74 to 0.94) and 2 below half the floor. In R6's pilot the rule lifted 22 wells (14 core), of which 8 were kept (6 core); 13 fail with lift too, 12 of them with a lift capacity of 0.02 to 0.04 kg/s, at or near D-31's lower bound, because it scales with a small design rate.
- **Needs sign-off:** yes (sampling). It makes the lift label of these wells depend on a solve, against R3's rule that no label does (Labels); the label's own value marks them.
- **Used by:** `sampling-draft.md`, Gas lift: assignment, Producibility filter; `sampler_prototype.py` (`sol2_well`).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint F2): approved: a natural oil well that cannot flow is lifted, as operators do.

### D-102 · A scan with non-finite points and no root has no operating point

- **Step:** R6, 2026-11-02. Revises D-53, and with it where D-54's retry runs.
- **Choice made:** an attempt whose Rust search found no root and rejected no sign change of $R$ is `no_operating_point` (reason `no_root`), whatever the number of non-finite scan points, which the log keeps; a search with a rejected sign change stays a `solver_failure` (`incomplete_scan`), as do the other reasons of D-53.
- **Alternatives:** D-53 as drafted (non-finite points make it a solver failure); the CasADi backend as the judge of every attempt without a root (10 to 30 Rust solves each).
- **Realism:** the core leaves out scan points where a march leaves the range of the closures, near $p_s$ through choked cells or where Joule–Thomson cooling leaves a cell's energy row without a root (`rust/src/shoot.rs`), and refines the edges of the finite region. Such points are routine: 13.9% and 14.2% of the operating points' searches in the draft's `sol-2` and `nsol-2` pilots had them. The draft's pilots had 211 attempts with non-finite points, no root and no rejected sign change (144 in `sol-2`, 67 in `nsol-2`). The CasADi retry ran on 72 of them (the others belonged to wells already at a limit): it found no root in 70 and recovered 2, both in `nsol-2`. So the change counts about 3% of these attempts as no operating point where a root exists, against 97% that D-53 counts as solver failures where none does. Under D-53, 4 tail wells of the `sol-2` pilot were dropped at the solver-failure limit (D-81), 3 of them for such attempts only, and every such attempt counted against the solver, which misstates criterion 4's split. In R6's pilots the solver failures left in the accepted `sol-2` wells fell from 8 to 0, and the retries from 52 to 11.
- **Needs sign-off:** yes (sampling; it decides what the data call physical).
- **Used by:** `sampling-draft.md`, No operating point or solver failure, A second backend for solver failures; `sampler_prototype.py` (`solve`).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint H2): approved.

### D-103 · The coverage tail's drawn share raised to 0.20

- **Step:** R6, 2026-11-02. Revises D-6.
- **Choice made:** 20% of the drawn wells are tail wells, so that about 15% of the accepted wells are, the middle of the Goals' 10% to 20%; the tail's draws are unchanged.
- **Alternatives:** keep 0.15 drawn: in the draft's `sol-2` pilot the tail is 14.0% of the drawn wells (62 of 442) and 9.25% of the accepted (37 of 400), below the Goals' range, because 40% of the tail wells are dropped against 4.5% of the core wells; tie the tail's $p_{s,\text{nom}}$ to $p_r$, which needs the separator drawn after the reservoir state and so changes the order of the blocks (D-7); let the tail's separator and reservoir pressures stay within the core's links.
- **Realism:** the tail's flat draws are independent, so a tail well can draw $p_r$ at 0.15 $p_i$ and $p_s$ near 120 bar: 5 of the 25 dropped tail wells had $p_r \le p_s$ at their nominal state, and most of the rest had too little drawdown to flow. The tail is a coverage device, not a population, and its wells are labelled. In R6's pilots the tail is 19.3% of the drawn `sol-2` wells (87 of 450) and 14.5% of the accepted, and 19% of the accepted `nsol-2` wells.
- **Needs sign-off:** yes (the Goals leave the share to the research; D-6).
- **Used by:** `sampling-draft.md`, Coverage tail; `sampler_draft_draws.py` (`TAIL_SHARE`).
- **Ruling:** Bjarne, 2026-11-06 (checkpoint A2): approved: 0.20 of the drawn wells are tail wells, with independent flat draws, and tail wells may take lift and geometry regardless of stage and type.

### D-104 · The wells' shapes from the pooled surveys, uniform over their ranges

- **Step:** the checkpoint (R8), 2026-11-06. Revises the shapes' parameters of `sampling-draft.md`, Geometry, within the classes of D-9; with D-21.
- **Choice made:** each shape parameter uniform between the pooled 5th and 95th percentiles of the regions in hand: the kickoff below the mudline $U(0.05, 0.75)$ of the TVD below it (a fix: the draft drew it from $0.1L$ with the riser in $L$, so a deep-water platform well could start its build inside the riser); the build rate $U(0.3, 3.6)$ °/30 m; the S wells' drop $U(0.2, 2.2)$ °/30 m and bottom inclination $U(0°, 35°)$; the drains' tangent before landing $U(10°, 65°)$, landing build $U(1.5, 7)$ °/30 m and length $U(300, 2500)$ m; the high-angle tangent wells holding $U(60°, 75°)$ to the bottom, as extended-reach wells do. The deviated class's hold angle stays as drafted. The drains' productivity is D-21's physical form.
- **Alternatives:** log-normal or Beta draws through the pooled percentiles (typical shapes weighted as the regions in hand); R3's ranges with only the kickoff fix and longer drains.
- **Realism:** the GoM's and the Netherlands' station surveys [own analysis of BOEM's directional surveys and NLOG's deviation surveys], the UK's equivalent kickoffs and tangents from top- and bottom-hole positions [own analysis of NSTA's wellbore layer], and Norway's MD/TVD [own analysis of Sodir's FactMaps] agree on gentler builds and shallower kickoffs than R3's ranges; the UK and Norway add extended-reach wells whose MD/TVD passes the draft's highly deviated class. A private comparison with field well paths informed the decision.
- **Needs sign-off:** yes (sampling).
- **Used by:** `sampling-draft.md`, Geometry; R10's `specs/sampling.md`.
- **Ruling:** Bjarne, 2026-11-06 (checkpoint B1): uniform over the pooled ranges, as above. Keep it simple.

### D-105 · One life for both datasets: a `sol-2` well is its `nsol-2` life at one date

- **Step:** the rework, 2026-11-06. Revises D-18 (the states by stage), D-41 (the end of a life) and D-42 to D-44 (the trends' end); D-19's material balance and D-40's drift stay.
- **Choice made:** every well has one life (R4's trends, `sampler_draft_samples.Life`), from R3's initial draw at its start towards its reservoir exhausted at its end: under depletion $p_r/p_i$ from 0.15, the coverage tail's floor, to the bottom of R3's late range ($U(0.15, 0.3)$ for oils, $U(0.15, 0.2)$ for gases); under pressure support the water at its cap, a water cut of 0.98 for oils and a WGR of 1000 Sm³ per 10⁶ Sm³ for gases (the tail's top). An `nsol-2` well lives it until its best rate falls below half the floor (D-50, D-84). A `sol-2` well is the same life at one date drawn uniformly in its drawn stage: the first year (initial), the rest of the first half of the life span (mid-life) or its second half (late); its producibility filter drops a well whose date falls after it stopped paying. A tail well keeps its tail state, its life's end. R3's per-stage tables remain as the life's anchors only: the start, the separator's steps, the roughness and the lift.
- **Alternatives:** an `nsol-2` life that ends at the same well's `sol-2` late draw (R4's design), consistent but with the last fifth of a life at about 0.38 of the first year's rate; a harsher end for `nsol-2` alone, with the stages of the two datasets meaning different states.
- **Realism:** real wells run down to abandonment, and D-41's target, a last fifth at about a tenth of the first year's rate, is set with the regions' stopped wells [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]. A well stock at a date holds wells at every point of their lives that still pay, as `sol-2` now does. In the draws (300 wells, seed 2026) `nsol-2`'s samples and `sol-2`'s wells agree by stage, by construction: late depletion wells at 0.23 and 0.24 of $p_i$ at the median, late oil wells with pressure support at water cuts of 0.96 and 0.97.
- **Needs sign-off:** yes (sampling).
- **Used by:** `sampling-draft.md`, Reservoir state at the life stage and `nsol-2`: a well's life; `sampler_draft_samples.py` (`sol2_well_at`, `Life`); `sampler_prototype.py` (`sol2_well`).
- **Ruling:** Bjarne, 2026-11-06: "One life for both", after asking for a simple and reasonable method and that `nsol-2`'s and `sol-2`'s late stages be the same.

### D-106 · A table's pressure nodes where p/Z falls are dropped as flash artefacts

- **Step:** R9, 2026-11-06. Amends D-67's table build.
- **Choice made:** after a table is flashed (`table_fluids.py`, `rising_pz_nodes`), the pressures at which p/Z does not rise strictly at some temperature, the nodes outside the longest subsequence along which it rises (of the longest, the one ending highest, so that a spike goes), are dropped at every temperature of that GOR's table, so that the grid stays rectangular; a table that would lose more than 5% of its pressures fails, and its well is `table_failed`. The manifest records the dropped pressures.
- **Alternatives:** drop the wells whose tables fail PVT-TAB-1 (`table_failed`), which in R9's check run took 12 of 86 table sets, 6 of 16 in `nsol-2` and 5 of its 9 condensates, all in hot reservoirs, and so biases the table classes; relax PVT-TAB-1's p/Z check, which K1 added to catch such errors; a better flash (NeqSim's settings changed nothing: the artefact is deterministic).
- **Realism:** near a hot fluid's near-critical bubble point NeqSim's TP flash splits the stream into two near-identical liquids, or misses the split at a single node (R9's check run). The dropped nodes are 1 to 2 of 75 to 119 pressures in the affected tables; the bilinear interpolation bridges them.
- **Needs sign-off:** yes (the table build behind the datasets; new data-side machinery).
- **Used by:** `table_fluids.py` (`build_tables`, `rising_pz_nodes`, `MAX_DROPPED`); R9's pilots.
- **Ruling:** Bjarne, 2026-11-09 (second checkpoint, L1): approved with the 5% limit; `table_failed` wells count with the drops that H3's targets judge on R11's pilot.

### D-107 · A gas well with pressure support waters out: its life ends at a WGR of 10,000

- **Step:** the second checkpoint, 2026-11-09 (`checkpoint-2.md`, L4). Revises D-105's end of a supported gas life and R3's tail draw of the WGR.
- **Choice made:** a core gas or condensate well with pressure support ends its life with the water at a WGR of 10,000 Sm³ per 10⁶ Sm³ (`END_WGR`), in place of 1000; its reservoir pressure ends at R3's late draw, as before. The coverage tail's WGR becomes $\text{LogU}(1, 10\thinspace 000)$ in place of $\text{LogU}(1, 1000)$, so that the tail still spans the core; the draw takes the same random number, so no other draw moves.
- **Alternatives:** partial water drive, the end pressure $U(0.3,\ \text{late draw})$ of $p_i$ (B: last fifth 0.53); both (C: 0.27, as A, with two more wells dropped); keep the end at 1000 (0.78).
- **Realism:** the regions' stopped gas wells keep about a tenth of their first year's rate in their last fifth [own analysis of BOEM's OGOR-A production]; in the Gulf's gas sands water drive and partial water drive hold most of the completed wells, and the water-drive sands recover the least gas, so their wells stop at high pressure as they water out [own analysis of BOEM's sands atlas]; the Gulf's gas wells' WGR in their later years has an upper decile several times 1000 [own analysis of BOEM's OGOR-A production]. Measured on `nsol-2` (100 wells, seed 2027, without tables), the 21 supported core gas wells' last fifth falls from 0.78 to 0.27 of the first year's rate (depletion gas wells: 0.16), half their samples lie above a WGR of 1000, redraws per sample rise from 0.03 to 0.13 and no attempt fails in the solver. INF-11 lets the water take no inflow capacity from the gas, beyond INF-10's validity (single-phase gas near the well), so the model's watered-out wells decline less than real ones.
- **Needs sign-off:** yes (sampling; the WGR bound and `validity.md`'s row).
- **Used by:** `sampler_draft_samples.py` (`END_WGR`), `sampler_draft_draws.py` (the tail's WGR); `sampling-draft.md`, The reservoir state and `nsol-2`: a well's life.
- **Ruling:** Bjarne, 2026-11-10 (second checkpoint, L4): approved as recommended. The rerun of 2026-11-09 took it as a working assumption (`r9-measurements.md`, The rerun after the second checkpoint).

*Entries D-108 to D-121 are R10's (2026-11-10): working assumptions taken while the sampler was moved into the package, under Bjarne's rule of that day that an agent works until the step is finished and hands it over once, and that the design rests on data and physics. D-110 to D-120 replace guesses that the triage of `guesses.md` found data, a published source or physics for. Each is its own commit, so that a ruling can reverse it alone. Two of the triage's proposals are not taken and wait for Bjarne: a later water breakthrough with wet starts (G30), and a higher separator pressure for gas wells (G7).*

### D-108 · Every number of a block is drawn, whatever the branch

- **Step:** R10, 2026-11-10. Implements the checkpoint's note on D-7.
- **Choice made:** where the prototype skipped a number that a well's branch did not use (six places in the well draws: the dry-gas number, the surface temperature's second number, the flowline's two numbers, the undersaturation, the stage block's second pressure number and initial water numbers, and the lift's Bernoulli), the block now draws it and leaves it unused, so that every block takes the same count of numbers in the same order for every well. The same holds for the samples' blocks. A core well does not draw a tail well's form of a number as well: the two forms take the same number.
- **Alternatives:** leave the prototype's order, under which a change of one rule moves later draws of the same block; a fixed vector of uniforms per block with every draw an inverse transform, which is stricter and rewrites every draw.
- **Basis:** a design choice for reproducibility (D-7): a change of one rule moves no other draw.
- **Realism:** not applicable. The draws' values change against the prototype's for most wells, and their distributions do not; the port was proved bit for bit against the prototype before this change, and the pilots were rerun after it.
- **Implemented** (R10, 2026-11-10). Besides the six places, the port found these: a platform well's deep-water number and an onshore well's water depth (`depth`); the survey, whose count of numbers varied by family, by whether the shape fitted and by the count of redraws, now six numbers per survey and all 20 surveys' numbers for every well (`geometry`); a core well's tail pressure downstream of the choke (`separator`); a gas class's saturation numbers and a black oil's initial-GOR number (`fluid`); the tail's $r_p$ and WLR, an oil's $k_g/k_o$ blend and a gas class's WGR and $c_k$ (`stage`); a natural well's lift design (`lift`); a core well's tail $h$ (`heat`); and one draw whose count varied with the seed, numpy's Dirichlet of a table fluid's composition weights, now three exponential draws over their sum, one number each. The life's blocks (`evolution`, `date`) and the samples' (`sample`, `controls`, `redraw`) were clean: their counts are fixed by the slots alone. Over 3000 wells, 1500 lives and 400 plans of seed 2026, every key's distribution before and after, core and tail apart, moves within sampling noise: the largest two-sample Kolmogorov statistic $\sqrt{nm/(n + m)}\thinspace D$ is 1.42 (the condensates' $c_k$, 358 core wells), 2 of 303 comparisons lie above the 5% level of 1.36 and none above the 0.1% level of 1.95, and the largest change of a share is 2.1 standard errors. The table fluid of the NeqSim test's well moves with its weights: its bubble point from 274.0 to 270.6 bar.
- **Needs sign-off:** no; it carries out a ruled note.
- **Used by:** `specs/sampling.md`, SMP-51; `sampling/v2/draws.py`, `survey.py`, `_distributions.py` (`unused`) and `table_build.py` (the composition's weights); `life.py` and `samples.py` needed no change; tests that every block takes the same numbers for every well and that a branch's constant moves no later draw.
- **Ruling:**

### D-109 · The initial pressure stops at 950 bar where the hydrostatic pressure alone passes it

- **Step:** R10, 2026-11-10. A finding of the port on D-12 and D-35.
- **Choice made:** where $p_h > 950$ bar, $p_i = 950$ bar and `pressure_class` is `under`; otherwise as SMP-63.
- **Alternatives:** leave $p_i$ up to about 1080 bar in such wells, outside the bound that D-35 ruled; redraw the depth or the water depth, which moves other draws; a lower cap on the depth in deep water.
- **Basis:** physics for the bound, the largest bottomhole pressure of Hasan et al.'s data (D-35, `validity.md`); the cap's form is a design choice, the simplest that keeps the bound.
- **Realism:** $p_h$ passes 950 bar only in the deepest wells in the deepest water: 38 of 300,000 draws of the prototype (seed 2026), 34 of them tail subsea wells, so one or two wells of a dataset. Such wells exist, at pressures above the model's range; a capped well is under its hydrostatic pressure by at most an eighth, inside the underpressured class's range of 0.8 to 1.0.
- **Implemented** (R10, 2026-11-10): in the package after D-108, 5 of 100,000 wells (seed 2026) have $p_h > 950$ bar, 4 tail subsea and 1 tail platform well; 4 of them had $p_i$ above 950 bar, up to 974.5 bar, and one had been drawn underpressured at 903 bar, which the rule raises to 950 bar. No other draw moves.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-63; `sampling/v2/draws.py` (`_pressure`); the bound's test, now a plain test.
- **Ruling:**

### D-110 · Water depths: subsea wells log-uniform, fewer deep-water dry trees, a lower shelf floor

- **Step:** R10, 2026-11-10. Revises D-13 (`guesses.md`, G2).
- **Choice made:** subsea $d_w \sim \text{LogU}(80, 3000)$ m in place of $\text{LogN}_ {[80, 3000]}(350, 0.8)$; deep-water dry trees 0.10 of platform wells in place of 0.15; the shelf platforms' floor 5 m in place of 15 m; the rest of SMP-58 as it is.
- **Alternatives:** keep the log-normal, whose median is one region's; a fit to one region's subsea wells.
- **Basis:** data. The subsea wells of three regions together span the whole range, one centred where the draft's median is and another mostly in deep water, where the draft put few wells; the shelf wells of one region stand in shallower water than the draft's floor; the deep-water share of platform wells lies between one region's present-day share and the others' none [own analysis of BOEM's directional surveys and borehole file, Sodir's FactMaps, NSTA's wellbore layer and NOAA's World Ocean Atlas 2023]. The log-uniform shape is a design choice, D-104's rule of a uniform draw over the pooled range.
- **Realism:** subsea wells in 1000 m of water or more become three in ten of the subsea wells in place of a tenth. Deeper water gives a colder seabed, a larger flowline loss and a lower effective gradient, so the subsea wells' rates and kept share are measured again on the pilot.
- **Implemented** (R10, 2026-11-10). The tail's platform draw starts at 5 m as well, $\text{LogU}(5, 2000)$ m in place of $\text{LogU}(15, 2000)$ m, so that the tail still spans the core (SMP-53). Over 100,000 wells of seed 2026, before and after (5th / 25th / 50th / 75th / 95th percentiles): the core subsea wells' $d_w$ 114 / 214 / 360 / 609 / 1297 m becomes 96 / 195 / 489 / 1214 / 2512 m, with 30.4% of them in 1000 m or more (9.5% before) and 11.2% in 2000 m or more (1.2%); their $T_s$ 274.6 / 279.6 / 287.6 K becomes 273.3 / 278.4 / 288.3 K (5th / 50th / 95th), $T_r$'s median 349 K becomes 346 K, the flowline's $\Delta p_\text{fl}$ 17 / 36 / 85 bar becomes 16 / 42 / 147 bar, and $p_{s,\text{nom}}$ 32 / 62 / 118 bar becomes 32 / 71 / 176 bar, at its cap of 250 bar in 0.14% of them (none before). The core platforms' deep-water share falls from 0.150 to 0.100, and 0.2% of them stand in less than 15 m; the tail platforms' median $d_w$ falls from 176 to 102 m. Wells whose hydrostatic pressure passes 950 bar (D-109) rise from 5 to 8 in 100,000.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-58; `sampling/v2/draws.py` (`_depth`, `_P_DEEP_TREE`).
- **Ruling:**

### D-111 · S wells hold at 20° to 60° and run on for longer; a drain's tangent may be absent

- **Step:** R10, 2026-11-10. Revises D-104's shapes for the S family and the drains (`guesses.md`, G3).
- **Choice made:** an S well's hold $\theta_h \sim U(20°, 60°)$ in place of $U(15°, 45°)$; its end section $\min(U(100, 1200)\ \text{m}, L/3)$ of TVD in place of $\min(U(100, 400)\ \text{m}, L/4)$; a drain's tangent before the heel $U(0, 300)$ m in place of $U(50, 300)$ m; the holds' least lengths as they are.
- **Alternatives:** keep R3's values, set before the surveys were read; fitted shapes.
- **Basis:** data, by D-104's rule of a uniform draw over the pooled range: the S wells of two regions hold up to well above the draft's 45°, and one region's run on below their drop for longer than the draft allowed; most of one region's drains land without a tangent [own analysis of BOEM's directional surveys and NLOG's deviation surveys]. The lower end of 20° is the deviated class's own (SMP-59).
- **Realism:** S wells are a twelfth of the wells. The end section rests on one region.
- **Implemented** (R10, 2026-11-10). Over 100,000 wells of seed 2026, before and after (5th / 25th / 50th / 75th / 95th percentiles): the S wells' hold 16.5 / 22.2 / 29.5 / 37.3 / 43.5° becomes 21.7 / 28.7 / 38.3 / 49.1 / 57.9°, their end section 115 / 171 / 246 / 320 / 384 m becomes 151 / 328 / 562 / 792 / 1060 m, their MD over TVD 1.016 / 1.060 / 1.165 (5th / 50th / 95th) becomes 1.028 / 1.098 / 1.273, and the drains' tangent 63 / 176 / 288 m becomes 15 / 151 / 285 m. The S wells redrawn at least once rise from 2.4% to 6.6% (redraws per S well from 0.05 to 0.22), and 20 S wells fall back to vertical after 20 redraws, 0.24% of the S wells, where none did before: most are shallow, about 650 m below the mudline, where even a hold of 20° with the least end section no longer fits, and a few very deep ones pass the MD limit. The drains' redraws do not change (0.4%). On 40,000 of the surveys drawn after the change, none passes 75° or 9342 m of MD, and no hold is shorter than its least TVD.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-60; `sampling/v2/survey.py` (`_S_HOLD`, `_S_END`, `_S_END_SHARE`, `_HEEL_TANGENT`).
- **Ruling:**

### D-112 · Subsea separator temperatures up to 340 K

- **Step:** R10, 2026-11-10. Revises D-15's temperatures (`guesses.md`, G8).
- **Choice made:** subsea $T_\text{sep} \sim U(290, 340)$ K in place of $U(290, 320)$ K.
- **Alternatives:** keep 320 K.
- **Basis:** data, thinly: subsea satellite wells of one operator arrive topside at temperatures whose median lies above the draft's top [own analysis of Petrobras' 3W Dataset]; a long tieback arrives colder, which the range's bottom keeps.
- **Realism:** the temperature enters only the black oil's separator correction (PVT-OIL-5) and a table fluid's offsets, by a few per cent.
- **Implemented** (R10, 2026-11-10). Over 30,000 wells of seed 2026, before and after (5th / 25th / 50th / 75th / 95th percentiles): the subsea wells' $T_\text{sep}$ 291 / 298 / 305 / 313 / 318 K becomes 292 / 303 / 315 / 328 / 337 K; their black oils' initial GOR rises by 0.0 / 0.3 / 1.3 / 2.5 / 4.1%, and those whose saturation pressure is lowered to the black oils' GOR limit rise from 17.5% to 18.4% of them. The platform and onshore ranges are classed as data, thinly, as the triage proposes.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-66; `sampling/v2/draws.py` (`_separator`).
- **Ruling:**

### D-113 · A slanted well's productivity from Cinco-Ley's slant skin

- **Step:** R10, 2026-11-10. Replaces D-21's placeholder factor (`guesses.md`, G14 and G15).
- **Choice made:** for build-and-hold and S wells, the skin of a slanted well, $s_\theta = -(\theta'/41)^{2.06} - (\theta'/56)^{1.865}\log_{10}(h_D/100)$ with $\theta' = \arctan(\tan\theta_\text{end}/I_\text{ani})$ in degrees and $h_D = h_n I_\text{ani}/r_w$, is added to the skin in $J$ and in a gas well's $a$, in place of the factor $(\cos\theta_\text{end})^{-1/2}$; a gas well's non-Darcy coefficient takes the perforated length along the well, $h_\text{perf}/\cos\theta_\text{end}$, in Eq. 4-34. The gain $(A + s)/(A + s + s_\theta)$, with $A = \ln(r_e/r_w) - 0.75$, is capped where a stimulated well at a high angle would pass it. The drains' form (D-21) is untouched.
- **Alternatives:** keep the placeholder, which is unbiased at the median but blind to the skin, on which the gain depends; Besson's form.
- **Basis:** physics, a published correlation: Cinco-Ley, Ramey and Miller's (1975) slant skin, as tabulated by Economides, Hill and Ehlig-Economides (1994, Table 5-1, p. 91: −0.128, −0.517, −1.178, −2.149 and −3.577 at 15° to 75° for $h_D$ = 100), which the fit reproduces within 3%; valid to 75°, the sampler's limit. The cap is a design choice of the build.
- **Realism:** for a well of skin 2 and $\ln(r_e/r_w)$ = 8 the table gives gains of 1.06, 1.15, 1.30 and 1.63 at 30°, 45°, 60° and 75°, where the placeholder gave 1.07, 1.19, 1.41 and 1.97. Over the draws the placeholder over the slant skin's gain is 0.85 / 1.01 / 1.11 (5th, 50th and 95th percentiles) from 20° to 60° and 0.58 / 1.09 / 1.36 from 60° to 75°.
- **Implemented** (R10, 2026-11-10). The fit gives −0.126, −0.526, −1.211, −2.191 and −3.470 at 15° to 75° for $h_D$ = 100, within 3.0% of Table 5-1 (a test). **The safeguard:** a floor of 2 on $A + s + s_\theta$, the simpler of the two at the same share: over 62,435 slanted wells of 100,000 (seed 2026) it binds in 201, 0.32%, of which 193 are at 60° to 75° (1.6% of that band) and 8 at 20° to 60°; the largest gain it leaves is 3.2. A cap of 2 on the gain would bind in 0.86%, cutting into the band's 95th percentile. Over 3000 drawn wells, the factor $f$ on a slanted well's index before and after (5th / 50th / 95th percentiles): below 20°, 1.00 / 1.01 / 1.03 becomes 1.00 / 1.00 / 1.02; from 20° to 60°, 1.04 / 1.12 / 1.37 becomes 1.01 / 1.04 / 1.20; from 60° to 75°, 1.43 / 1.60 / 1.91 becomes 1.11 / 1.28 / 1.88. An oil's $J$ after over before is 0.98 / 0.99 / 1.00, 0.82 / 0.94 / 0.99 and 0.68 / 0.79 / 1.07 in the three bands. A gas well's $D$ falls with the perforated length along the well, by 0.96, 0.71 and 0.19 at the median of the bands, so its open flow changes less: 0.98 / 1.00 / 1.01, 0.86 / 0.98 / 1.13 and 0.72 / 0.96 / 1.55. Vertical wells do not change.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-71, SMP-72, SMP-73; `sampling/v2/reservoir.py` (`slant_skin`), `draws.py` (`_inflow`, `_on_geometry`, `_SLANT_FLOOR`).
- **Ruling:**

### D-114 · The lift gas arrives at the valve between the ambient and the reservoir temperature

- **Step:** R10, 2026-11-10. Revises D-30's temperature (`guesses.md`, G23).
- **Choice made:** $T_{lg} = T_a(H) + \psi\thinspace(T_r - T_a(H))$ with $\psi \sim U(0, 0.75)$, in place of $T_a(H) - \phi\thinspace(T_a(H) - T_s)$ with $\phi \sim U(0, 0.3)$; one uniform draw, as before.
- **Alternatives:** the draft's, which cools the gas below the ambient at the valve; THM-9's default, the ambient itself, with no draw.
- **Basis:** physics, from published solutions: in Hasan and Kabir's (2018, §12.2.3, Figs. 12.16 to 12.22) examples the annulus gas warms within the first few hundred feet and then runs between the geotherm and the tubing's fluid, about two thirds to three quarters of the way from the formation's temperature to the reservoir's in a deep well at a high rate, and a little above the geotherm at a low rate; it is below the ambient only by a few kelvin, at a valve near the bottom. The uniform shape and the top of 0.75 are a guess read off three examples; a solution of the annulus' energy balance per well would replace them.
- **Realism:** a few kelvin in a lifted well's stream above the valve. $T_{lg}$ stays at or below $T_r$, inside the solver's temperature bound.
- **Implemented** (R10, 2026-11-10). The draw is stored as `psi_lg` in place of `phi_lg`, and `T_lg_below_ambient`, the amount below the ambient, is removed, since the gas is no longer below it; `datasets-draft.md`'s column follows. Over the 964 lifted wells of 3000 (seed 2026), before and after (5th / 25th / 50th / 75th / 95th percentiles): $T_{lg} - T_a(H)$ −20.2 / −11.3 / −6.7 / −3.0 / −0.6 K becomes 0.1 / 0.6 / 1.8 / 8.2 / 28.0 K, small at the median because most valves lie near the packer, where the ambient is near $T_r$; $T_{lg}$ 296 / 311 / 325 / 340 / 376 K becomes 303 / 323 / 339 / 359 / 399 K; and $T_r - T_{lg}$ is at least 0.19 K, so $T_{lg} \le T_r$ in every well.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-79; `sampling/v2/draws.py` (`_lift`, `_PSI_LG`), `life.py` (`_LIFT_KEYS`); `datasets-draft.md`, the lift design's columns.
- **Ruling:**

### D-115 · The pressure's exponent over a life: 0.7 to 2.3

- **Step:** R10, 2026-11-10. Revises D-42 (`guesses.md`, G29).
- **Choice made:** $\gamma \sim U(0.7, 2.3)$ for oils and wells with pressure support, in place of $U(1, 2)$: the 5th and 95th percentiles, rounded, of the exponent fitted to the drawn oils' material balance, which replaced the range of 0.7 to 1.5 logged first (below).
- **Alternatives:** keep the draft's, fitted to one textbook well; the material balance's own $p_r(F)$ with a term for the oil above the bubble point, which adds a compressibility to the draws and stays the named replacement.
- **Basis:** physics, the model's own result: Muskat's material balance on the model's black oil, from the bubble point down, gives exponents from about 0.7 to 1 (`plans/evidence/solution_gas_drive.py`), and the oil above its bubble point raises the exponent of the whole path, as in Golan and Whitson's (1991) Example 4.11, fitted at 1.4. For wells with pressure support the form stays a guess; their pressure moves little.
- **Realism:** depletion oils hold a higher pressure in mid-life than the draft's.
- **Implemented** (R10, 2026-11-10). `plans/evidence/pressure_exponent.py` fits $\gamma$ to each drawn core black oil under depletion, from its life's start $r_0 p_i$ to its end $r_1 p_i$: Muskat's equation with Fig. 9.16's $k_g/k_o$ below the bubble point, and above it the oil's expansion by Vazquez and Beggs's (1980) compressibility, the model's $B_o$ being constant there. Over 1281 such oils of 10,000 wells (seed 2026), $\gamma$ is 0.69 / 0.95 / 2.29 (5th / 50th / 95th percentiles): 0.64 / 0.83 / 0.98 for oils saturated at discovery, 0.70 / 1.02 / 2.72 for undersaturated ones, and 0.78 / 1.17 / 3.55 where the life starts above the bubble point; 400 oils of another seed give 0.69 / 0.94 / 2.20. The computed range differs from 0.7 to 1.5, so the rule takes 0.7 to 2.3. Two sensitivities: with the pore space's and connate water's expansion at typical values (own assumptions) the 95th percentile is 1.97; with the tail's $k_g/k_o$ blend at $w$ = 0.5 the path is 0.48 / 0.65 / 1.81. **Unsure:** a uniform draw over the 5th to 95th percentiles has its median at 1.5 where the material balance's is 0.95, so the realism above does not follow at the median: over 1500 lives, the core depletion oils' $\gamma$ 1.07 / 1.47 / 1.95 becomes 0.81 / 1.45 / 2.22, their $p_r/p_i$ at half the life 0.208 / 0.298 / 0.404 becomes 0.203 / 0.305 / 0.449, and the share of the pressure's fall still to come at half the life 0.032 / 0.103 / 0.244 becomes 0.020 / 0.111 / 0.306; the supported wells' paths spread the same way. A draw closer to the material balance, or its own $p_r(F)$ per well, would follow the oils better; that is for Bjarne's ruling.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-85; `sampling/v2/life.py` (`_GAMMA`); `plans/evidence/pressure_exponent.py`.
- **Ruling:**

### D-116 · The separator's steps may come from a tenth of the life

- **Step:** R10, 2026-11-10. Revises D-46 (`guesses.md`, G32).
- **Choice made:** the two step dates $U(0.1, 0.9)\thinspace T_\text{life}$ in place of $U(0.25, 0.9)$: the range widened downwards only.
- **Alternatives:** the triage's $U(0.1, 0.8)$, which also follows one region's medians; keep the draft's.
- **Basis:** data, thinly and by a proxy: one region's gas wells, with the choke open, lose wellhead pressure earlier in their lives than the draft's first step in most wells [own analysis of NSTA's production by well]. The upper end is kept, since one region's shape is weak evidence (the Goals, coverage first).
- **Realism:** the separator steps down earlier in some lives.
- **Implemented** (R10, 2026-11-10). Over 1500 lives of seed 2026, before and after (5th / 25th / 50th / 75th / 95th percentiles): the first step's date over the life span 0.27 / 0.33 / 0.44 / 0.56 / 0.74 becomes 0.12 / 0.20 / 0.33 / 0.49 / 0.70, the second's 0.38 / 0.56 / 0.71 / 0.81 / 0.88 becomes 0.27 / 0.49 / 0.66 / 0.79 / 0.88; the core wells' nominal pressure downstream of the choke at a quarter of the life 14 / 28 / 43 / 69 / 151 bar becomes 9 / 24 / 39 / 66 / 145 bar, and at half the life its median 34 bar becomes 31 bar.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-89; `sampling/v2/life.py` (`_T_SEP`).
- **Ruling:**

### D-117 · Lift's drawn start from a twentieth of the life

- **Step:** R10, 2026-11-10. Revises D-49's drawn date (`guesses.md`, G33).
- **Choice made:** $U(0.05, 0.9)\thinspace T_\text{life}$ in place of $U(0.2, 0.9)$.
- **Alternatives:** keep the draft's.
- **Basis:** data: of one region's oil wells that were ever on gas lift and not from their start, the first month on lift spans a range that starts well below the draft's 0.2 and ends near its 0.9, by two sources [own analysis of BOEM's OGOR-A production and BSEE's well tests]. The uniform shape is a design choice for simplicity.
- **Realism:** the drawn date is one of three rules, and the earliest wins; some lifts start earlier.
- **Implemented** (R10, 2026-11-10). Over 1500 lives of seed 2026, before and after: the lives whose lift starts at the drawn date rise from 244 to 275 and those by the static check fall from 305 to 274; the drawn starts over the life span, 0.22 / 0.33 / 0.48 / 0.69 / 0.86 (5th / 25th / 50th / 75th / 95th percentiles), become 0.07 / 0.20 / 0.37 / 0.60 / 0.85, and every later start, by either rule, 0.10 / 0.25 / 0.37 / 0.56 / 0.81 becomes 0.07 / 0.19 / 0.30 / 0.51 / 0.80. The lives lifted by a quarter of their span rise from 27.7% to 31.3%, and by half of it from 43.5% to 44.4%. The life scan's earlier start in `nsol-2` (SMP-101) is not measured here.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-90; `sampling/v2/life.py` (`_T_LIFT`).
- **Ruling:**

### D-118 · Set points that change less often: the change probability from 0.2

- **Step:** R10, 2026-11-10. Revises D-47's probability (`guesses.md`, G36).
- **Choice made:** $p_c \sim \text{LogU}(0.2, 0.95)$ per well in place of $\text{LogU}(0.5, 0.95)$; the step of ±0.1 and the share 0.1 of new draws as they are.
- **Alternatives:** keep the draft's, set by one operator's subsea wells.
- **Basis:** data from two regions: sampled at `nsol-2`'s spacings, one region's gas wells change their choke opening in fewer intervals than the draft's lowest 0.5, and the other's wells move theirs far more often [own analysis of NSTA's production by well and Petrobras' 3W Dataset]. The step and the share stay guesses: the one history in hand over whole lives is of swing producers.
- **Realism:** `nsol-2` gains wells whose set points move less often.
- **Implemented** (R10, 2026-11-10). Over the plans of 400 lives of seed 2026, before and after (5th / 25th / 50th / 75th / 95th percentiles): $p_c$ 0.52 / 0.59 / 0.69 / 0.80 / 0.91 becomes 0.22 / 0.30 / 0.43 / 0.63 / 0.86; the share of operating samples at which the choke's set point changes follows it, 0.50 / 0.69 / 0.93 becoming 0.21 / 0.43 / 0.87 (5th / 50th / 95th); the range its set point spans over the life, 0.76 / 0.93 / 0.95 of the choke's range, becomes 0.54 / 0.91 / 0.95, and its standard deviation's median 0.26 becomes 0.25.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-98; `sampling/v2/samples.py` (`_P_CHANGE`).
- **Ruling:**

### D-119 · The liner's water takes the model's viscosity

- **Step:** R10, 2026-11-10 (`guesses.md`, N5).
- **Choice made:** in a drain's friction, the water's viscosity is PVT-WAT-3's at $T_r$ in place of 0.5 mPa s.
- **Alternatives:** keep the constant.
- **Basis:** physics, the model's own correlation: 0.49 mPa s at 330 K and 0.15 at 450 K.
- **Realism:** the viscosity enters Chen's friction factor weakly; a drain's friction changes slightly.
- **Implemented** (R10, 2026-11-10). Over the 225 oil drains with water among 3000 wells of seed 2026, the model's water at $T_r$ is 0.19 / 0.27 / 0.36 / 0.45 / 0.72 mPa s (5th / 25th / 50th / 75th / 95th percentiles), and the drains' friction efficiency changes by at most 1.2%, at the median by less than 0.1%; no other draw changes. The density's clamp is classed as a design choice of the build, as the triage proposes.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-72; `sampling/v2/draws.py` (`_inflow_at_stage`), not `reservoir.py`.
- **Ruling:**

### D-120 · Lighter heptanes-plus for condensates

- **Step:** R10, 2026-11-10. Revises D-63's range (`guesses.md`, G37).
- **Choice made:** a synthetic condensate's $M_{7+} \sim U(120, 200)$ g/mol in place of $U(140, 200)$, if the composition's match to the drawn API gravity holds on the build; otherwise the range stays and the build's result is recorded here.
- **Alternatives:** keep the draft's, which starts at the published average and so leaves out the lighter half of the published condensates.
- **Basis:** data, published: 340 gas condensates' heptanes-plus have molar masses of 106 to 253 g/mol with an average of 142 (Elsharkawy 2002, Table 1, p. 151).
- **Realism:** lighter condensates, with lower dew points.
- **The build's test** (R10, 2026-11-10): **the change is not taken; the range stays $U(140, 200)$ g/mol.** Method: the synthetic gas condensates among the package's first 1235 drawn wells of seed 2026 (after D-108 to D-119), each with its composition generator's seed, built by NeqSim 3.4.0 from a copy of `table_build.py` with $M_{7+} \sim U(120, 200)$: 48 whose composition draws $M_{7+}$ from 120 to 140 g/mol (set A) and, as the baseline, 24 that draw it from 140 to 200 (set B, the present range's fluids). Results: the single-stage match within 0.5 °API and 0.005 holds as often in both sets, 44 of 48 and 22 of 24; every miss in both is on the gas gravity, with the factor on C₂ to C₆ at its floor of 0.25, and every API is matched. But $C_f$ rises: 0.276 / 0.313 / 0.367 (least / median / largest) in set A against 0.260 / 0.284 / 0.334 in set B, and 26 of set A's 48 lie outside 0.27 to 0.31 against 6 of set B's 24. The rise follows the drawn API, which is drawn independently of $M_{7+}$: in set A the condensates drawn at 42 to 48 °API match with $C_f$ of 0.326 at the median and up to 0.367, 15 of 20 above 0.31, where set B's never pass 0.31 there. So a lighter heptanes-plus reaches a heavy stock-tank oil only with fractions heavier than Søreide's correlation gives typical fluids, and the composition no longer reads as a light condensate. The range of 0.27 to 0.31 is taken as Søreide's typical range as quoted by Whitson and Brulé (2000, §5.6) and was not checked against the book here. A coupling of $M_{7+}$ with the drawn API, or an $M_{7+}$ from 120 g/mol only for the higher API gravities, would let the lighter half of Elsharkawy's span in; either is a new rule, for Bjarne's ruling. Field-reported API errors are alike in both sets (median 0.7 and 0.6 °API).
- **Needs sign-off:** yes (sampling).
- **Used by:** none: `specs/sampling.md`, SMP-113, and `sampling/v2/table_build.py` are unchanged.
- **Ruling:**

### D-121 · A gas drain's liner is at most 7 in

- **Step:** R10, 2026-11-10. Records what the prototype did, where D-21 says "the liner taken as the tubing".
- **Choice made:** a gas drain's liner is the tubing, at most 157.1 mm (7 in): a gas well on 9-5/8 in tubing has a 7 in liner.
- **Alternatives:** the liner as the tubing in every case, a 9-5/8 in liner.
- **Basis:** a design choice of the build: 7 in is the largest of the oil wells' tubing sizes (SMP-75). Nothing was found on liner sizes over wells; completion records would replace it.
- **Realism:** 72 of 4000 drawn wells, gas drains on 9-5/8 in tubing.
- **Implemented** (R10, 2026-11-10): no change to the code; SMP-73's rule names the 7 in liner of a gas well on 9-5/8 in tubing, and its Basis line and the design choices' table class the cap as a design choice of the build.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-73; `sampling/v2/draws.py` (`_LINER_MAX`, `_inflow_at_stage`), not `reservoir.py`.
- **Ruling:**

*Entries D-122 to D-126 are R10's too (2026-11-10): the working assumptions of the run over many wells and of the datasets' assembly, where `datasets-draft.md`'s Section 2, written before the checkpoint, and the port's results disagree. Each follows the spec where the spec decides, and the prototype's columns elsewhere; the assembly was proved against the prototype's on 44 wells of both datasets, column by column.*

### D-122 · No columns of the CasADi retry

- **Step:** R10, 2026-11-10. Follows the checkpoint's H1 (D-54 dropped).
- **Choice made:** `redraws` has no `retried` or `start`, and its outcomes no `recovered`; `wells` and `drawn_wells` have no `n_recovered`; the metadata's `retry_backend` is null.
- **Alternatives:** keep the columns, always false or null, which describe a mechanism the generator no longer has.
- **Basis:** a design choice following H1's ruling: the retry was dropped for want of a measured gain (constitution, principle 7), so no attempt is retried or recovered.
- **Realism:** not applicable.
- **Needs sign-off:** yes (the dataset schema).
- **Used by:** `datasets-draft.md`, Section 2 (`redraws`, the generation summary) and Section 4 (the outcomes' codes); `sampling/v2/assemble.py`, `datasets/v2.py`.
- **Ruling:**

### D-123 · `sol-2`'s lift label is `lift_start_rule`

- **Step:** R10, 2026-11-10. Bjarne ruled on 2026-11-10 to follow the recommendations of the spec draft's points, which named this label (SMP-52).
- **Choice made:** both datasets' `wells` have `lift_start_rule`: `from_start`, `static_check` and `drawn_time` from the life, `end_of_natural_flow` from `nsol-2`'s life scan and `no_natural_flow` from `sol-2`'s producibility filter, null for a natural well. The draft's `sol-2` column `lift_reason` (`drawn`, `needs_lift`, `no_natural_flow`), which the prototype derived from the lift block's draw at the date, is not written.
- **Alternatives:** the draft's `lift_reason`, which names the lift block's draw at one stage, not the rule of the life that `sol-2` now shares with `nsol-2` (D-105).
- **Basis:** a design choice of the labels (SMP-52), one label for both datasets.
- **Realism:** not applicable. In the equivalence check every `sol-2` well that the filter lifted has `no_natural_flow` in both the prototype's column and the port's, and the natural wells have neither.
- **Needs sign-off:** yes, as part of the schema; the label follows the ruling of 2026-11-10.
- **Used by:** `datasets-draft.md`, Section 2 (Labels) and Section 4; `sampling/v2/assemble.py`.
- **Ruling:**

### D-124 · The well log counts attempts by reason, and the labels take the spec's values

- **Step:** R10, 2026-11-10. Where the draft and the spec differ, the spec's.
- **Choice made:** `drawn_wells` and `wells` count a well's slot attempts without an operating point by reason, `n_redraws_<reason>` for SMP-104's six reasons, besides the draft's counts by outcome (SMP-108: "the counts of its attempts by outcome and reason"). The labels' codes are Section 4's with the spec's values added at the end of each list: `saturation_kind` `none` (SMP-115), `lift_start_rule` `no_natural_flow` (SMP-52), `drawn_wells.outcome` `table_failed` (SMP-106) and `crash`, a well whose generation raised (logged and counted, as the prototype's run did); `redraws.outcome` loses `recovered` (D-122); `not_producible_reason` takes `water_cut`, `below_floor`, `no_operating_point` and `solver_failure` (SMP-103). A value without a code stops the assembly.
- **Alternatives:** the draft's counts by outcome only, which leaves a user to count the reasons in the redraw log; free strings for the labels, which D-89 ruled out.
- **Basis:** a design choice following SMP-108 and SMP-52 to SMP-115; the codes' order is D-89's rule that a later version adds categories at the end.
- **Realism:** not applicable.
- **Needs sign-off:** yes (the dataset schema).
- **Used by:** `datasets-draft.md`, Sections 2 and 4; `datasets/v2.py` (`CODES`, `WELL_OUTCOMES`), `sampling/v2/assemble.py`.
- **Ruling:**

### D-125 · The wells table as the port gives it

- **Step:** R10, 2026-11-10. Where the draft's columns and the port's results disagree otherwise.
- **Choice made:**
  - `kgko_weight`, the oils' blend of the $k_g/k_o$ curves (SMP-70), in place of the draft's `k_g`, R3's GOR factor, which the material balance replaced (D-19).
  - `pvt_model` is `table` for every volatile oil and condensate (SMP-112), a `table_failed` one included; the prototype's `interim_pvt` and `interim_inflow` are not written, since no table well is solved on the black oil and every gas well takes Forchheimer's inflow (R9).
  - In `nsol-2` the inflow's coefficients are in `samples` only (`KL`, `WLMAX`, `FORCHA`, `FORCHB`): the prototype also wrote `forchheimer_a` and `forchheimer_b` columns, always null, in `nsol-2`'s `wells`.
  - `wells` holds the draft's generation summary (`n_redraws` and its splits); `n_attempts`, `n_several_stable`, the times and the life scan's lists are in `drawn_wells` only, as the draft has them. `n_several_stable` counts the samples whose root set had several stable roots, as the draft defines it, where the prototype counted every slot attempt; the two agree on every well of the equivalence check.
  - `J` and `well_age` are the well's start's, its initial-stage draw's, in both datasets, as the port and the prototype give them; in `sol-2` the date's inflow is in `k_l`, `w_l_max` or `forchheimer_a` and `forchheimer_b`, and the date's heat transfer in `h`. The draft's "at the stage" is corrected.
  - Beyond the draft, the prototype's records of the draws are kept: `life_date` (`sol-2`'s date, years), `design_rate` (the rate the tubing was sized for), `flags` (the draws' truncations), and a table fluid's build: `c7_mol`, `gor_moved`, the drawn `api_drawn`, `sg_gas_drawn`, `R_si_drawn` and `p_sat_drawn` (the prototype's `p_sat_r3`), the offsets `offset_*` and field values `field_*` (SMP-113), `n_gors` and `table_nodes`; and in `drawn_wells` the build's `table_seconds`.
  - A dropped well's row in `drawn_wells` is the port's: an `nsol-2` well dropped before its scan (`table_failed`, `invalid_input`) has its life's parameters and lift, where the prototype left them null, and a `sol-2` `table_failed` well has its date's inflow coefficients, where the prototype had the initial draw's (the port's per-well generation, proved against the prototype with these two differences); a well without best-rate solves has an empty list of their outcomes, where the prototype had null. In the equivalence check these are the only differences: 31 values of one `nsol-2` `table_failed` well.
- **Alternatives:** the draft's columns as written, which name a factor the sampler no longer draws and leave a user without the draws behind a table fluid; the prototype's columns as written, with two flags that are constant since R9.
- **Basis:** a design choice of the schema, following SMP-70, SMP-112 and SMP-113 to SMP-115 where they decide.
- **Realism:** not applicable.
- **Needs sign-off:** yes (the dataset schema).
- **Used by:** `datasets-draft.md`, Section 2 (`wells`, `drawn_wells`); `sampling/v2/assemble.py` (`well_columns`, `drawn_well_columns`, `well_row`).
- **Ruling:**

### D-126 · The files of a dataset and of its run

- **Step:** R10, 2026-11-10. Implements D-85, D-87 and D-93 as ruled.
- **Choice made:** a dataset is a folder `manywells-<kind>-2/` with `wells`, `samples`, `geometry` and `pvt` as one parquet file each and a `metadata.json` (the records of SMP-110 and each file's rows and SHA-256), and its logs a companion folder `manywells-<kind>-2-logs/` next to it with `drawn_wells`, `redraws` and its own `metadata.json` (I1). The split is the `wells` column `split`; the train and test files of `datasets-draft.md`'s Hugging Face layout are left to the release (D-86). A run keeps each well's result as a pickle until its chunk of 100 draw indices is complete, then one pickle per chunk with the run's settings; a restart with other settings or other code stops. The attempts' search counts and times stay in the chunks, and the assembly writes them as the run's attempt log, one parquet file per hundred wells. A dataset is written only from a clean working tree, unless a pilot asks otherwise.
- **Alternatives:** the logs inside the dataset's folder, against I1; one file per well (D-93's rejected alternative), which the run keeps only while a chunk is open; the chunks as parquet, which would fix the attempts' columns before the assembly.
- **Basis:** a design choice of the generation, following D-85, D-87 and D-93 as ruled.
- **Realism:** not applicable.
- **Needs sign-off:** yes (the dataset layout).
- **Used by:** `sampling/v2/run.py`, `sampling/v2/assemble.py`, `datasets/v2.py`; `scripts/data_generation/v2/generate_dataset.py`.
- **Ruling:**

*Entries from D-127 on are R10's too (2026-11-10), from the review of R10's port: each fixes a rule that the port's code and the spec's text gave differently, or a draw that the physics contradicts.*

### D-127 · A table fluid's GOR moves only in its class's direction

- **Step:** R10, 2026-11-10 (the review of R10). Revises D-65's search.
- **Choice made:** where a table fluid's saturation pressure at $T_r$ lies above $p_i$, its GOR moves along its tie line only in its class's direction: down for a volatile oil, trying the factors 0.8, 0.6, 0.4 and 0.25 in turn, and up for a gas condensate, 1.25, 1.6, 2.5 and 4, each then bisected on $\ln R$ (`gor_moved`); where that direction finds no GOR whose saturation pressure is $p_i$, the fluid is the class's phase in equilibrium at $(p_i, T_r)$ (`equilibrium_phase`), as before.
- **Alternatives:** the port's search, 0.8 down to 0.25 and then 1.25 up to 4 for both classes, which tries a condensate richer first; for a condensate that does not cross, a third lever on the composition, which D-65 set aside.
- **Basis:** physics: an oil's bubble point falls as its GOR falls, and a condensate's dew point falls as its GOR rises and it grows leaner. Made richer, a condensate passes the critical region to the bubble side, where its bubble point falls again with the GOR, so the port's richer steps could find a crossing there: a near-critical oil at a black oil's or volatile oil's GOR, still labelled a gas condensate.
- **Realism:** the fluids of the 195 table wells among `sol-2`'s draw indices 0 to 646 at seed 2026, which hold the test build's wells (D-120) and more, built by NeqSim 3.4.0 with the port's search and with this one: 91 volatile oils (69 synthetic, 22 examples) and 104 condensates (92 synthetic, 12 examples).

  | Class | Search | `matched` | `gor_moved` | `equilibrium_phase` | `bubble` | `dew` |
  |---|---|---|---|---|---|---|
  | volatile oil | port | 62 | 22 | 7 | 89 | 2 |
  | volatile oil | class's direction | 62 | 22 | 7 | 89 | 2 |
  | gas condensate | port | 73 | 9 (3 to the bubble side) | 22 | 12 | 92 |
  | gas condensate | class's direction | 73 | 6 | 25 | 9 | 95 |

  The volatile oils keep their rules and values: their search already went down first, and the fluids that no longer try the upward factors differ by at most $2\cdot10^{-13}$ relative, the flash's warm start. Three condensates of 104 (2.9%) change: the port moved them to 0.58, 0.27 and 0.51 of their GOR, to field GORs of 489, 396 and 426 Sm³/Sm³ on the bubble side; in the class's direction no GOR within a factor of 4 crosses, and they take the equilibrium vapour at 2.0 to 5.1 times their GOR, field GORs of 2144 to 4000 Sm³/Sm³ with a dew point. Two findings remain, which the rule does not touch: 9 condensates (8.7%) are `matched` with a bubble point below $p_i$, near-critical oils as drawn by their composition (D-63), and the equilibrium vapour can be far leaner than the drawn fluid, its field GOR at the median 2.2 times the drawn $R_{si}$ and at most 2179 times (a field GOR of 2.2 million Sm³/Sm³, one well), where the fluid at $(p_i, T_r)$ lies deep in its two-phase region.
- **Implemented** (R10, 2026-11-10): `table_build._saturated_at` takes the class and searches its direction only (`_GOR_FACTORS`); SMP-115's rule and Basis line say so. Tests with a stand-in flash: a near-critical condensate moves to the dew side and an oil to the bubble side, and a class whose direction does not cross is the equilibrium phase, the other direction not tried.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-115; `sampling/v2/table_build.py` (`_GOR_FACTORS`, `_saturated_at`, `build_fluid`); `tests/test_sampling_v2_tables.py`.
- **Ruling:**

### D-128 · A tail well's inflow model, non-Darcy coefficient and label at its life's start

- **Step:** R10, 2026-11-10 (the review of R10). Implements SMP-71, SMP-74, SMP-68 and SMP-83 as written.
- **Choice made:** the inflow model (SMP-74), the gas viscosity of the non-Darcy coefficient (SMP-71) and `nsol-2`'s `saturation` label (SMP-68) are taken at the life's start: for a core well its initial-stage draw's state, as before, and for a tail well the core initial state drawn for it (SMP-83, $r_0$ and its water), where the port took the initial-stage draw's state, which for a tail well is its tail state. The life passes its start to every stage's draw (`draws.draw_well`'s `start`), so that the mid-life and late draws, and the lift design made from the late one, have the well's one inflow model and one non-Darcy coefficient. A `sol-2` and an `nsol-2` tail well share the life (D-105), so they take the same model and coefficient. No number is drawn: the start's numbers are the `evolution` block's, drawn for every well (D-108).
- **Alternatives:** keep the initial-stage draw's state, which SMP-83 does not make a tail well's start; take the non-Darcy coefficient's viscosity at $p_i$, a rule change of SMP-71.
- **Basis:** the rules as written (SMP-71, SMP-74, SMP-68, SMP-83); the choice of the start is D-105's ruling.
- **Realism:** over `sol-2`'s draw indices 0 to 2999 at seed 2026 (without tables): the inflow model changes for 61 of 153 tail black oils (40%) and 49 of 153 tail volatile oils (32%), and `nsol-2`'s saturation label for 62 tail black oils, 39 tail volatile oils and 47 of 150 tail condensates; every tail gas well's non-Darcy coefficient changes (310 of 310), and so the inflow's coefficients of every tail well whose model or coefficient changed (420 of 616). No core well's initial-stage draw changes. The late draws' non-Darcy coefficient now is the start's for every gas well, which moves the late lift design (its capacity) of 71 of 780 core gas wells (9.1%) and of 60 of 310 tail gas wells (19%), from the late draw's gas rate; no oil's design changes. Where a tail well's draws stay at its initial-stage draw by the rules' wording, nothing changes: its `J` record (D-125 calls it the start's), a drain's friction efficiency in the geometry's factor (SMP-72, from the initial-stage draw) and the design of a tail well lifted from its start (SMP-90).
- **Implemented** (R10, 2026-11-10): `draws.draw_well(..., start=...)`, `life.Life`; SMP-83 says where a tail well's start applies. Tests: the start sets the model and the coefficient in every stage's draw and moves no number; a tail well's model, coefficient and label are its start's.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-68, SMP-71, SMP-74 and SMP-83; `sampling/v2/draws.py` (`draw_well`, `_inflow`, `_inflow_at_stage`), `sampling/v2/life.py` (`Life`); `tests/test_sampling_v2_draws.py`, `tests/test_sampling_v2_life.py`.
- **Ruling:**

### D-129 · The pressure-balance point where the casing's gas and the tubing do not cross

- **Step:** R10, 2026-11-10 (the review of R10). Corrects SMP-79's text to the code.
- **Choice made:** SMP-79 states each case of the casing's line less the valve's drop against the tubing's, as the port's code takes them: where the tubing's pressure grows faster with depth, $H_\text{bal}$ is their crossing, and where the tubing lies above the casing's gas at every depth, no gas can enter, $H_\text{bal}$ is 0 and the strong-well rule (SMP-78) applies; where the casing's grows at least as fast, or the casing lies above the tubing at every depth, $H_\text{bal} = L$ and the valve is at the packer. The code clips its $H_\text{bal}$ at 0 to say so, which changes no draw (checked on every stage's draw and forced design of 1500 wells, 6000 draws, all equal), since a negative $H_\text{bal}$ was below $0.3\thinspace L$ for the strong-well rule and raised to it for the valve as 0 is.
- **Alternatives:** the text's "$L$ where the two do not cross", which for a tubing above the casing at every depth would put the valve at the packer, where no gas enters, and lift a strong well that the rule leaves natural.
- **Basis:** physics: lift gas enters the tubing only where the casing's pressure less the valve's drop exceeds the tubing's (Golan and Whitson 1991, §5.3).
- **Realism:** of the 6000 stage draws and forced designs of `sol-2`'s draw indices 0 to 1499 at seed 2026, the tubing lies above the casing's gas at every depth in 334 oil and 826 gas draws, of which 24 and 42 are lifted and 25 are an oil's forced late design: those inject at $0.3\thinspace L$, the rest stay natural. One case remains where the code and the physics differ, left for a ruling because a fix moves draws: where the casing's gradient is the larger, the code takes $L$ even where the tubing lies above the casing at every depth (22 gas draws, 2 of them lifted, at the packer), and where the two cross with the casing below at the top (6 gas draws, 1 lifted) $L$ is right, the deepest point at which gas enters.
- **Implemented** (R10, 2026-11-10): SMP-79's text; `draws._lift` clips $H_\text{bal}$ at 0, with a comment that names the cases.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-79; `sampling/v2/draws.py` (`_lift`).
- **Ruling:**

### D-130 · A gas well's tubing is sized at $p_i$ in every stage's draw

- **Step:** R10, 2026-11-10 (the review of R10). Implements SMP-75 as written: "the tubing is the well's for its life".
- **Choice made:** a gas or condensate well's design rate takes $\mu_g$ and $Z$ at $p_i$ and SMP-71's non-Darcy coefficient with $\mu_g$ at $p_i$, from the same number of its error factor, so that the design point is the initial state in every stage's draw and each sizes the same tubing; the lift design of a well lifted later, from its late-stage draw, sits on the well's own tubing, its capacity capped at that tubing's erosional rate. The error factor's number is now drawn as one uniform number and taken through the truncated inverse CDF at each viscosity (`_distributions.tnorm_at`), which gives the coefficient to the bit and takes the same count of numbers (D-108). SMP-75's note that the port took $\mu_g$ and $D$ at the draw's reservoir state is removed.
- **Alternatives:** the port's, $\mu_g$ and $D$ at each draw's reservoir state, under which a later draw could size another tubing; the well's own coefficient (at the life's start, D-128) in the design rate, which sizes the same tubing within a life but not in a stage's draw made alone.
- **Basis:** the rule as written (SMP-75): the tubing is run at the initial state and kept for the life.
- **Realism:** over `sol-2`'s draw indices 0 to 3999 at seed 2026 (1500 gas and condensate wells), the port's draws, after D-128, put a mid-life or late draw or the lift design on another tubing than the start's in 180 wells (12%), 19 of them the lift design; now none. Every gas well's design rate changes, as $\mu_g$ is now at $p_i$; 75 wells (5.0%) take another tubing, and 5 lift designs change their capacity. No oil changes: an oil's design rate was already the initial state's.
- **Implemented** (R10, 2026-11-10): `draws._inflow` keeps the coefficient as a function of $\mu_g$, `draws._completion` sizes a gas well at $p_i$; SMP-75's text. Tests: every stage's draw and the lift design have the well's design rate, tubing and erosional rate; a stage override keeps them, gas wells included.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-75; `sampling/v2/draws.py` (`_inflow`, `_completion`), `sampling/v2/_distributions.py` (`tnorm_at`); `tests/test_sampling_v2_draws.py`, `tests/test_sampling_v2_life.py`.
- **Ruling:**

### D-131 · An `nsol-2` well whose life ends before its lift starts is natural

- **Step:** R10, 2026-11-10 (the review of R10). Implements SMP-52 as written.
- **Choice made:** where the life scan ends an `nsol-2` well's life before the lift's start by the draws (SMP-90's `drawn_time` or `static_check`, a start that falls between two scan points, with the lifted well failing at the later one), the well is labelled `natural`, with no `lift_start` and no `lift_start_rule`. Its plan and its samples do not change: no slot's date reaches the start, and the plan draws the same numbers either way.
- **Alternatives:** the port's label `gas_lift`, with a start after $T_\text{end}$, for a well none of whose samples is lifted.
- **Basis:** a design choice of the labels: SMP-52's `lift` in `nsol-2` is a well lifted at some date of its life.
- **Realism:** it follows from the solves: the start must lie within the scan's step of 5% of the life before the first point at which the lifted well fails. Not measured here; R11's pilot counts the `nsol-2` wells with a lift design that stay natural.
- **Implemented** (R10, 2026-11-10): `generate.generate_nsol2_well`; SMP-101 says so. Test with stand-in solves: such a well is natural, and its samples are the plan's of a natural well.
- **Needs sign-off:** yes (the labels).
- **Used by:** `specs/sampling.md`, SMP-101; `sampling/v2/generate.py`; `tests/test_sampling_v2_generate.py`.
- **Ruling:**

### D-132 · At most 20 surveys, and one flag that counts the redraws

- **Step:** R10, 2026-11-10 (the review of R10). Corrects SMP-60's text to the code.
- **Choice made:** a well draws at most 20 surveys in all, the first and 19 redraws, and is vertical where none fits, as the port's code does and as SMP-51's "all 20 surveys' numbers" says; SMP-60 said "redrawn … at most 20 times". A well's redraws are one flag with their count, `geometry_redrawn:n`, where the port wrote `geometry_redrawn` once per failed survey, 20 times for a well that falls back to vertical.
- **Alternatives:** 20 redraws, 21 surveys, which changes the block's count of numbers and so every later draw of the block (D-108); the flag repeated per redraw.
- **Basis:** a design choice of the generation (SMP-60's fit rule); the count is the prototype's.
- **Realism:** no draw changes; only the `flags` of a well with two or more redraws read differently. No well of 3000 at seed 2026 falls back to vertical (`tests/test_sampling_v2_draws.py`).
- **Implemented** (R10, 2026-11-10): `draws._geometry`; SMP-60's text and its row in the design choices. Test: a survey that fails 0, 1, 7, 19 or 20 times gives one flag with the count, the fallback after 20, and the block's 122 numbers.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-60; `sampling/v2/draws.py` (`_GEOMETRY_SURVEYS`, `_geometry`); `tests/test_sampling_v2_draws.py`.
- **Ruling:**

### D-133 · The records come from the run's settings, and `design_rate` is the tubing's

- **Step:** R10, 2026-11-10 (the review of R10). Implements D-125 and SMP-110 as written.
- **Choice made:** `wells.design_rate` is the liquid rate at which the tubing was sized (`q_design`, SMP-75), as D-125 defines it, where the port wrote the start's reference rate, $J\delta_d p_r$ for an oil and the gas rate at the design drawdown for a gas well; the schema has no column for the reference rate, so it is not kept. The run records in its settings, when it starts, the best rate's lift-gas rates, its scan points and its water cut, and the lock file's hash, besides the limits and the floor; the dataset's records take them from there, and the assembly refuses code whose settings differ from the run's (all but the code's record, which the metadata keeps for both). A well's own file in the run's folder holds the hash of the settings, so that a run with other settings refuses it. The metadata names the folder of tables relative to the run's folder, and a crash's traceback in the well log writes the home directory as `~`.
- **Alternatives:** the port's: the assembling code's constants and lock in the records, which a change between the run and the assembly makes wrong; absolute local paths in the published metadata and logs.
- **Basis:** a design choice of the generation, for reproducibility (SMP-110) and for the rule that no local path is published.
- **Realism:** `design_rate` changes for every well: for an oil, from $J\delta_d p_r$ at the start to $J\delta_d p_i$ at $p_i$ with no water; for a gas well, from a gas rate to the liquid rate at the design point.
- **Implemented** (R10, 2026-11-10): `sampling/v2/run.py` (`run_settings`, the wells' files, the tracebacks), `sampling/v2/assemble.py` (`records`, `write_dataset`, `well_row`); SMP-110's text. Tests.
- **Needs sign-off:** yes (the dataset schema and its records).
- **Used by:** `specs/sampling.md`, SMP-110; `datasets-draft.md`, Section 2 (`design_rate`, the metadata); `sampling/v2/run.py`, `sampling/v2/assemble.py`; `tests/test_sampling_v2_run.py`.
- **Ruling:**

### D-134 · Offshore gas and condensate wells start against a high-pressure line

- **Step:** R11, 2026-11-11. Revises D-15 (`guesses.md`, G7).
- **Choice made:** for a gas class on a platform or subsea, $p_\text{sep} \sim \text{LogN}_ {[10, 120]}(65, 0.5)$ bar in place of the oils' $\text{LogN}_ {[5, 85]}(35, 0.45)$; the oils and the onshore draw unchanged. The draw takes the same one number, so no other well's draws move.
- **Alternatives:** keep one draw for every offshore well, R10's choice while the proxy was loose; a wider spread for the gas classes, since one region's lower quartile lies below the new draw's.
- **Basis:** data from two regions, by a proxy that physics checks. A gas well's tubing pressure on a wide-open choke in its first year lies upstream of the bean; its median lies well above 35 bar in the Gulf of Mexico and in the UK, and the two agree [own analysis of BSEE's well tests; own analysis of NSTA's production by well]. R10 held the proposal back because a strong well can be in critical flow through the bean, where the upstream pressure says little of the line's. The bean's gas-flow equations (Guo, Lyons and Ghalambor 2007, Ch. 5; ideal gas, $k = 1.3$, $C$ from 0.75 to 0.9, the wellhead at 300 to 350 K), applied to each wide-open test of the Gulf's gas wells, put few tests near critical flow, and the line pressure they give lies close to the tubing pressure; where a test reports a casing-or-line pressure, it agrees with that estimate in most tests [own analysis of BSEE's well tests]. The median of 65 bar is a design value chosen with the two regions; the spread and the bounds are chosen, not fitted, the cap being v1's (SMP-16). A limit: only a minority of the Gulf's platform gas wells test wide open in their first year; the others are choked back, and their line is not known.
- **Realism:** offshore gas wells start against a higher back-pressure. Over the 1501 core offshore gas wells of seed 2026's draws 0 to 5999 (5th / 25th / 50th / 75th / 95th percentiles): $p_\text{sep}$ 16 / 26 / 35 / 46 / 67 bar becomes 27 / 44 / 61 / 80 / 108; the nominal pressure downstream of the choke 7 / 19 / 34 / 62 / 141 bar becomes 11 / 32 / 54 / 87 / 163, and its median over $p_r$ 0.13 becomes 0.21 at the initial stage; the wells drawn with $p_r$ at most that pressure, which cannot flow, rise from none to 0.3% at the initial stage and from 1.7% to 3.9% at the late. D-62's gas permeability factor was set against per-well rates at the old back-pressure, so R11's pilot measures the gas rates again.
- **Implemented** (R11, 2026-11-11): `sampling/v2/draws.py` (`_separator`), SMP-66's rule and Basis line, `tests/test_sampling_v2_draws.py`. The analysis is kept outside the repo with the triage's.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-66; `sampling/v2/draws.py`.
- **Ruling:**

### D-135 · An oil's water breaks through later

- **Step:** R11, 2026-11-11. Revises D-43 (`guesses.md`, G30).
- **Choice made:** an oil's breakthrough recovery $F_\text{bt} \sim U(0.1, 0.9)$ for both drives, in place of $U(0.05, 0.4)$ under pressure support and $U(0.2, 0.7)$ under depletion; the jumps unchanged; the gas classes keep their ranges. One number, as before, so no other draw moves.
- **Alternatives:** (a) wet starts as well: a well under pressure support starts wet with probability 0.3 or 0.5, its WLR at $U(0.05, 0.9)$ and rising from the start, logit-linear in $F$, to the end's (measured below); (b) keep R10's ranges; (c) the new range for the gas classes too.
- **Basis:** data for the oils: the recovery at which water breaks through in the Gulf of Mexico's oil wells that start dry spreads over almost the whole range, and its quartiles lie close to the new draw's [own analysis of BOEM's OGOR-A production]; the uniform draw over the pooled range is D-104's rule; the data do not split the wells by drive. The gas classes keep theirs: gas wells' water is thinly reported, and D-107's end was set on lives with these ranges. Wet starts are not taken: the regions disagree, the Gulf's shelf and Brazil's onshore fields starting mostly wet and the Gulf's deep water and Brazil's pre-salt mostly dry [own analysis of BOEM's OGOR-A production; own analysis of ANP's production by well]; the share of lives that start dry lies within that span; a wet start adds a rule; and the states it would add, high water near $p_i$, are those of mid-life wells under pressure support. A known limit: no core life starts wet.
- **Realism:** the oils' water cut by years on production (`plans/evidence/water_breakthrough.py`: 600 core oil lives of seed 2027, medians of their well-years), before and after: 0 to 2 years 0.07 and 0.04, 2 to 5 years 0.52 and 0.17, 5 to 10 years 0.83 and 0.50, 10 to 20 years 0.94 and 0.88, 20 to 30 years 0.97 both. Up to 20 years both lie within the span of the regions' per-well medians, and the later breakthrough moves the middle of the life from the wet end of the offshore regions towards their middle [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]. This corrects `guesses.md`, which expected a later breakthrough alone to move the mid-life water cut away from the regions. Beyond 20 years the lives under pressure support, which run to an end's water cut of 0.98 (D-105), lie above the one region with such ages, before and after. The dry starts' recovery at breakthrough, quartiles 0.19 / 0.28 / 0.38, becomes 0.31 / 0.50 / 0.70. With wet starts (alternative a) the medians would be 0.06 / 0.32 / 0.77 / 0.93 / 0.97 at a probability of 0.3 and 0.09 / 0.64 / 0.88 / 0.95 / 0.97 at 0.5, and the lives whose first year is dry 0.59 and 0.47 of them, against 0.74.
- **Implemented** (R11, 2026-11-11): `sampling/v2/life.py` (`Life.__init__`), SMP-86's rule and Basis line, `tests/test_sampling_v2_life.py`.
- **Needs sign-off:** yes (sampling).
- **Used by:** `specs/sampling.md`, SMP-86; `sampling/v2/life.py`.
- **Ruling:**
