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
- **Alternatives:** by sands, which raises the Gulf's wet-gas share from 37% to 50%; by volume, which puts oil and gas about even and over-weights large fields.
- **Realism:** the Goals count wells. Sidetracks count as wellbores on the Norwegian shelf, which slightly over-weights oil.
- **Needs sign-off:** yes (sampling).
- **Used by:** every share and marginal of `sampling-draft.md`.
- **Ruling:**

### D-2 · Nadeau et al. (2023) by reservoir counts

- **Step:** R3, 2026-10-30. Takes RF-2.
- **Choice made:** use the counts of reservoirs in Nadeau et al.'s Table 1 and Fig. 20 as checks, not their shares of EUR.
- **Alternatives:** EUR shares (5.5% of volume above 122 °C against 12.9% of reservoirs); counts times a well-count proxy such as EUR to the power 0.5.
- **Realism:** the draws put 17.1% of wells above 395 K and 20.4% above 0.6 psi/ft, against 12.9% of reservoirs above 122 °C and 16.2% above SG 1.3.
- **Needs sign-off:** yes (sampling).
- **Used by:** Initial reservoir pressure, Geothermal gradient (as checks).
- **Ruling:**

### D-3 · Fluid classes by McCain (1990)

- **Step:** R3, 2026-10-30. Takes RF-3.
- **Choice made:** McCain's (1990, pp. 151–158) limits of 356, 588 and 8900 Sm³/Sm³ on the initial producing GOR, with API gravity as a check, and dry gas below a condensate yield of 2.8 Sm³ per 10⁶ Sm³; the rule is stored with the well (`class_rule`).
- **Alternatives:** Whitson and Brulé (volatile oils from 178 Sm³/Sm³); Tewari et al. (from 90 m³/m³), under which 91% of the Gulf's oil sands would be volatile; McCain (1994), not in hand.
- **Realism:** under McCain's limits volatile oils are 1.3% of the Gulf's wells; the black oil (PVT-OIL-4) carries McCain's black oils within Beggs and Robinson's $R_{so}$ range.
- **Needs sign-off:** yes (sampling).
- **Used by:** Fluid class, Saturation pressure and the initial GOR, Mapping.
- **Ruling:**

### D-4 · The fluid mix and the gas subtypes

- **Step:** R3, 2026-10-30. Takes RF-4.
- **Choice made:** black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20; a gas well is dry with probability 0.07.
- **Alternatives:** the Gulf's well shares with the 10% floor (black oil 0.45, gas 0.30); the Norwegian shares (black oil 0.70, gas 0.10); dry gas at 0.125, its share by sands.
- **Realism:** by wells the Gulf has 47.7 / 1.3 / 11.6 / 39.4%, its deep water 80.6 / 2.0 / 7.0 / 10.4%, the Norwegian shelf 79% oil wellbores. The floor over-weights volatile oils about fivefold, which the Goals chose for coverage.
- **Needs sign-off:** yes (sampling; the Goals set the floor).
- **Used by:** Fluid class, and every class-dependent draw.
- **Ruling:**

### D-5 · The well-type mix

- **Step:** R3, 2026-10-30. Takes R2's lead [WP §1].
- **Choice made:** platform (dry tree) 0.55, subsea 0.35, onshore 0.10.
- **Alternatives:** 0.60 / 0.40 / 0 (offshore only); a larger onshore share (onshore wells dominate counts: 918,481 producing wells in the US against 2,216 in the federal Gulf); the Norwegian split, 59% platform and 41% subsea.
- **Realism:** deep water in the Gulf is 84% dry tree; the Norwegian shelf 41% subsea; the onshore share is a design choice of the Goals' offshore weighting, not a statistic.
- **Needs sign-off:** yes (sampling). Bjarne may want to see the onshore share before the checkpoint.
- **Used by:** Well type, and the water depth, surface temperature, separator, permeability, lift and economic floor.
- **Ruling:**

### D-6 · The coverage tail

- **Step:** R3, 2026-10-30.
- **Choice made:** 15% of wells, labelled; in a tail well every draw flattens to a (log-)uniform over its bounds, categorical shares become equal, and the links and bounds stay.
- **Alternatives:** 10% or 20%; a tail that stretches each draw independently with probability 1/2; a tail beyond the core's bounds, up to the model's limits (which would add implausible wells).
- **Realism:** tail wells are less likely, not implausible: none of the 2000 drawn wells breaks a bound. The tail lifts half its oil wells regardless of stage, which is less realistic than the core.
- **Needs sign-off:** yes (the Goals leave the share to the research).
- **Used by:** every draw's tail form; Where realism and coverage conflict; R6's coverage measurements.
- **Ruling:**

### D-7 · The order of the draws and seeding by block

- **Step:** R3, 2026-10-30.
- **Choice made:** the order of `sampling-draft.md`, with the salinity before the pressure and the separator before the fluid; one generator per block, `rng_for(seed, k, 'v2', block)`.
- **Alternatives:** the plan's likely order (water after the fluid, separator last), which would leave the hydrostatic gradient and Vazquez–Beggs's separator correction without their inputs; one generator per well, under which any change shifts every later draw.
- **Realism:** not applicable; it makes draws reproducible and local.
- **Needs sign-off:** yes (sampling, SMP-31's scheme extended).
- **Used by:** the whole draft; the sanity script.
- **Ruling:**

### D-8 · Depth by fluid class, and water depth by well type

- **Step:** R3, 2026-10-30. Takes RF §1's lead.
- **Choice made:** depth below the mudline log-normal by class, fitted to BOEM's P5 / P50 / P95 by wells and truncated to 600 to 6500 m; water depth by well type (platform 85% shelf around 90 m and 15% deep-water dry tree to 2000 m; subsea around 350 m to 3000 m); a platform's length includes its water.
- **Alternatives:** one depth distribution for all classes; depth before the class (draw the class given depth); the Norwegian TDs.
- **Realism:** dry gas lies shallowest and condensates deepest in the Gulf; deep water is deeper and oilier. The draws give 1530 / 2768 / 4831 m and reproduce BOEM's temperatures and gradients by class within 10 K and 0.008 bar/m.
- **Needs sign-off:** yes (sampling).
- **Used by:** Depth below the mudline; Water depth and the well's length; everything below them.
- **Ruling:**

### D-9 · Geometry families and the heel approach; shares pending

- **Step:** R3, 2026-10-30. Supersedes WP-1 with Bjarne's ruling of 2026-10-30.
- **Choice made:** four families (vertical, build-and-hold, S-shaped, high-angle to the heel) with the parameters of `sampling-draft.md`; a well with a horizontal drain is modelled to its heel; the shares stay open, pending `sources-geometry.md`.
- **Alternatives:** v1's three families (vertical, deviated, L-shaped); the two-segment equivalent tangent of the BOEM fit in place of build sections; build rates from a source (none in hand).
- **Realism:** 3.1% of the Gulf's completed development boreholes are straight and 5.9% horizontal; Norwegian MD/TVD medians are 1.54 (platform) and 2.04 (subsea). The build rates of 1.5 to 3 °/30 m have no source in hand.
- **Needs sign-off:** yes (sampling; the shares are Bjarne's to rule after the geometry evidence).
- **Used by:** Geometry; Productivity (the drain); Gas lift (MD of the valve).
- **Ruling:**

### D-10 · The inclination limit: 75°

- **Step:** R3, 2026-10-30. Takes WP-2.
- **Choice made:** no cell more than 75° from vertical.
- **Alternatives:** 80° (Taitel–Dukler leaves stratified flow only below $v_{SL}$ = 0.006 m/s; 1.1% of the Gulf's boreholes beyond); 70° (none in the scan above 0.003 m/s; 3.2% beyond).
- **Realism:** at 75° Taitel–Dukler allows stratified flow only below 0.004 m/s, about 7 m³/d of in-situ liquid in 157 mm tubing; gas wells with very little liquid could reach it, and they are the least deviated (Norwegian gas wellbores' MD/TVD 1.10 to 1.18).
- **Needs sign-off:** yes: it decides `validity.md`'s decision 5 with the geometry.
- **Used by:** Geometry.
- **Ruling:**

### D-11 · Brine salinity and density

- **Step:** R3, 2026-10-30. Takes RF-9.
- **Choice made:** salinity log-normal around 80,000 ppm, 1% to 25%; $\rho_w$ by a quadratic through McCain's (1990) points; the same brine sets the hydrostatic gradient.
- **Alternatives:** seawater (35,000 ppm) for every offshore well; v1's fixed 999.1 kg/m³ with a fixed gradient.
- **Realism:** no public distribution of offshore salinities is in hand; the median gives 0.104 bar/m against the Gulf's normal 0.1052 bar/m.
- **Needs sign-off:** yes (sampling; it makes the water density a draw, one of the Goals' fluid constants).
- **Used by:** Formation water; Initial reservoir pressure; Heat capacities.
- **Ruling:**

### D-12 · Overpressure with depth, and initial underpressure

- **Step:** R3, 2026-10-30. Takes RF-5 in a form of its own.
- **Choice made:** an overpressure ratio between hydrostatic and lithostatic (1 psi/ft) whose logit is normal around a median rising with depth, fitted to BOEM's depth bins; 4% of wells initially underpressured at 0.8 to 1.0 of hydrostatic.
- **Alternatives:** a depth-independent share; an excess pressure in bar fitted directly; no initial underpressure.
- **Realism:** 20.4% above 0.6 psi/ft (Gulf by wells 23%); gradients by class within 0.008 bar/m of BOEM's medians. The lithostatic gradient has no source in hand.
- **Needs sign-off:** yes (sampling).
- **Used by:** Initial reservoir pressure; Bjarne's goal 3.
- **Ruling:**

### D-13 · The geothermal gradient

- **Step:** R3, 2026-10-30. Rejects RF-6's mixture of three basins.
- **Choice made:** one log-normal below the mudline, median 30 K/km on shelves and onshore falling to 20 K/km in deep water, log-sd 0.22, truncated to 12 to 55 K/km and to keep $(T_r - T_s)/L \ge 10$ K/km and $T_r \le 450$ K.
- **Alternatives:** RF-6's mixture (24, 18 and 37 K/km with ±15 to 20%); a single global log-normal without the water-depth term.
- **Realism:** medians by class within 6 K of BOEM's; 17.1% above 395 K (12.9% of large-field reservoirs). The North Sea's gradients are uncorrected logging values, which run low.
- **Needs sign-off:** yes (sampling).
- **Used by:** Geothermal gradient and reservoir temperature.
- **Ruling:**

### D-14 · Surface temperature by well type

- **Step:** R3, 2026-10-30. Takes WP-11 with a seabed profile.
- **Choice made:** offshore wells take the seabed temperature at their water depth, from a shelf value of 279 to 297 K to 272 to 278 K in deep water over a 400 m scale; onshore wells 268 to 303 K.
- **Alternatives:** one range for all; a platform's sea-surface temperature at its wellhead.
- **Realism:** the profile meets the four seabed values of [WP §2] within about 2 K; deep-water platforms' risers get too warm an ambient under THM-4's linear profile.
- **Needs sign-off:** yes (sampling; a Goals' new draw).
- **Used by:** Surface temperature; the gradient's bounds; the lift-gas temperature.
- **Ruling:**

### D-15 · Separator conditions and the nominal downstream pressure

- **Step:** R3, 2026-10-30. Takes WP-8 in a form of its own.
- **Choice made:** the host's first-stage separator ($p_\text{sep}$ around 35 bar offshore, 15 bar onshore) is the fluid's separator in PVT-OIL-5; $p_{s,\text{nom}}$ falls with the life stage (to 0.1 to 0.6 of $p_\text{sep}$ late), and a subsea well adds its flowline and riser.
- **Alternatives:** standard conditions as the separator (today's default); v1's log-normal truncated to 10 to 120 bar with a separate low-pressure tail; WP-8's 2 to 80 and 10 to 100 bar as uniform ranges.
- **Realism:** first stages at 41 to 83 bar (Guo et al. 2007), lower stages at 20 and 2 bar (weak). The separator's correction raises Vazquez–Beggs's $R_{so}$ by about 30% at 40 bar against standard conditions, so it shifts the black oils' GORs.
- **Needs sign-off:** yes (sampling; separator conditions are a Goals' fluid constant).
- **Used by:** Separator conditions; Saturation pressure and the initial GOR; Tubing; Gas lift; Bjarne's goal 2.
- **Ruling:**

### D-16 · Oil and gas gravities by class

- **Step:** R3, 2026-10-30.
- **Choice made:** black oil 33 ± 4.5 °API (16 to 45) with gas gravity 0.66 ± 0.06; volatile oil 40 to 50 °API; condensate 42 to 58 °API; wet and dry gas 45 to 58 °API with gas gravity 0.62 ± 0.035 from 0.571.
- **Alternatives:** a Gaussian copula of API, GOR and gas gravity on the Gulf's rank correlations (RF §7's lead); API from v1's oil density (21 to 40).
- **Realism:** the Gulf's oils are 25 / 33 / 39 °API with gas gravities 0.60 / 0.62 / 0.67; Valkó and McCain's separator gases average 0.838. API and $R_{si}$ correlate at 0.31 in the draws (0.24 in the Gulf) through Vazquez–Beggs.
- **Needs sign-off:** yes (sampling).
- **Used by:** Oil and gas at standard conditions; everything that uses the fluid.
- **Ruling:**

### D-17 · Undersaturation and the initial GOR from the model's own correlation

- **Step:** R3, 2026-10-30. Takes RF-7.
- **Choice made:** draw the saturation pressure as a fraction of $p_i$ (20% of black oils saturated, the rest with a median undersaturation of 10%), and take the black oil's $R_{si}$ from the model's Vazquez–Beggs at that pressure, capped at 356 Sm³/Sm³ by lowering the bubble point.
- **Alternatives:** draw $R_{si}$ given depth and set $p_b$ from it; Standing's $p_b$ (PVT-OIL-14), which the simulator does not use.
- **Realism:** $R_{si}$ 71 / 168 / 356 Sm³/Sm³ against the Gulf's 78 / 151 / 284; $p_i - p_b$ 0 / 26 / 209 bar against 30 bar at the median; rank correlations within 0.12 of the Gulf's but for $R_{si}$–$T_r$ (0.32 against 0.56). The well's bubble point is the model's.
- **Needs sign-off:** yes (sampling).
- **Used by:** Saturation pressure and the initial GOR; Reservoir state; Inflow.
- **Ruling:**

### D-18 · Drive, life stage and the reservoir's state

- **Step:** R3, 2026-10-30. Takes RF-8 and RF §3, §9, §11's leads.
- **Choice made:** a drive label (pressure support 0.7 for oils, 0.4 for gases), a life stage (0.25 / 0.40 / 0.35), and the state tables of `sampling-draft.md` for $p_r/p_i$ and the water.
- **Alternatives:** v1's ±2% around a hydrostatic $p_r$ with no stage; the stage drawn from a field's age with decline curves (closer to `nsol-2`'s evolution, which R4 designs).
- **Realism:** 46.7% of core wells lie below hydrostatic and 30.1% by more than 10%; late depletion wells reach the Gulf's abandonment $p/Z$ range; late wells with support have a median water cut of 0.77, near Norway's 0.75 at 10 to 20 years. No public count of wells by stage is in hand.
- **Needs sign-off:** yes (sampling; it meets Bjarne's goal 1 and the Goals' life stage).
- **Used by:** Drive and life stage; Reservoir state; Separator; Roughness; Heat transfer; R4's drift and `nsol-2` trends.
- **Ruling:**

### D-19 · Producing GOR below the saturation pressure

- **Step:** R3, 2026-10-30.
- **Choice made:** a linear rise of the producing GOR with $1 - p_r/p_\text{sat}$, at a factor of 2 to 15 for oils (capped at ten times $R_{si}$) and 0.5 to 3 for condensates.
- **Alternatives:** a material-balance curve of solution-gas drive (needs a reservoir-engineering text, on R2's resource list); a constant GOR.
- **Realism:** Golan and Whitson's (1991) Example 4.11 rose to 8.3 times its initial GOR at 0.58 of initial pressure, then fell; Norwegian condensate fields produce at a median of 4871 Sm³/Sm³ late in life. One example only.
- **Needs sign-off:** yes (sampling).
- **Used by:** Reservoir state; R4's `nsol-2` GOR trend.
- **Ruling:**

### D-20 · Productivity from permeability, pay, viscosity and skin

- **Step:** R3, 2026-10-30. Takes WP §4's lead.
- **Choice made:** draw $k$ (offshore around 50 mD, onshore around 10 mD), net pay around 15 m and skin around 2, and compute the radial PI with the model's viscosity and $B_o$ at the stage's state; a design drawdown of 10% to 40% of $p_r$ sizes the tubing and the lift.
- **Alternatives:** draw the PI directly as a log-normal; draw the rate and back out the PI (WP-3's alternative); v1's $w_{l,\max}$ of 20 to 200 kg/s independent of everything.
- **Realism:** design liquid rates of 33 / 451 / 6411 Sm³/d (P10 / P50 / P90) for oils: wide, and high for a population counted by wells next to the Gulf's 2024 brackets (52% of oil wells at or below 16 Sm³/d o.e.), but solved rates are lower; R6 checks (D-24).
- **Needs sign-off:** yes (sampling).
- **Used by:** Productivity; Inflow; Tubing; Gas lift: capacity.
- **Ruling:**

### D-21 · Horizontal drains and slant: the productivity's factor

- **Step:** R3, 2026-10-30.
- **Choice made:** a high-angle oil well's PI is Joshi's (Economides et al. 1994, Eq. 2-46) for a drain of 300 to 1000 m, capped at 6 times the vertical PI; a high-angle gas well multiplies $k h_n$ by 1.5 to 4; a slanted well by $(\cos\theta_\text{end})^{-1/2}$.
- **Alternatives:** one multiplier for every high-angle well (R2's lead); Cinco-Ley's slant skin and the horizontal gas-well IPR (Economides et al. 1994, §5-4 and §4-6, not read).
- **Realism:** the draws give $J_H/J_V$ of 1.8 / 3.9 / 6.0, against 2 to 5 for a 610 m drain in Economides et al.'s Fig. 2-10. The gas factor and the slant factor are placeholders.
- **Needs sign-off:** yes (sampling; it is how the inflow stands for the drain under Bjarne's ruling).
- **Used by:** Productivity; Inflow.
- **Ruling:**

### D-22 · Oil wells' inflow: Vogel or a productivity index

- **Step:** R3, 2026-10-30.
- **Choice made:** Vogel where the saturation pressure is at least 0.8 of $p_r$ and the water cut below 0.6, else a productivity index; Vogel's $w_{l,\max}$ gives the same PI at $p_0 = p_r$.
- **Alternatives:** Vogel for every oil well (v1); a composite IPR (not an option of the model).
- **Realism:** Vogel holds for solution-gas drive below the bubble point and not where water flows too (`validity.md`, note 10); 64% of core oils get it.
- **Needs sign-off:** yes (sampling).
- **Used by:** Inflow; Mapping.
- **Ruling:**

### D-23 · Gas and condensate wells: Forchheimer's parameters, and the interim inflow

- **Step:** R3, 2026-10-30. Takes WP-7 for the draws; the option is R5's.
- **Choice made:** draw Forchheimer's $a$ from $k h_n$, the model's $Z$ and $\mu_g$ and the skin, and $b$ from a non-Darcy coefficient of $2\cdot10^{-4}$ to $2\cdot10^{-3}$ d/Mscf; until R5's gas-rate inflow exists, map to a productivity index that gives the design rate's liquid.
- **Alternatives:** the backpressure form with $n$ from 0.5 to 1 and $C$ from a target AOF; Vogel for gas wells (v1, outside Vogel's range).
- **Realism:** AOF 0.12 / 1.5 / 8.1 million Sm³/d (P5 / P50 / P95); 11% of gas wells' design rates exceed the largest tubing's erosional rate, as "a strong gas well will never produce to capacity" (Golan and Whitson 1991, Preface). The interim makes the gas rate linear in drawdown.
- **Needs sign-off:** yes (sampling; the feature spec is R5's and needs approval before implementation, and widening `v2.0.0`'s check needs Bjarne's ruling).
- **Used by:** Productivity; Inflow; Mapping; R5; R6's interim prototype.
- **Ruling:**

### D-24 · The rate target is a check, not a draw

- **Step:** R3, 2026-10-30. Takes WP-3.
- **Choice made:** compare R6's solved rates of offshore wells with the Gulf's 2024 brackets (EIA 2025, Table B32) after the economic filter; do not draw rates.
- **Alternatives:** draw the rate and back out the PI.
- **Realism:** the brackets are annual averages over calendar days of a mature shelf, so they sit low for a global offshore population.
- **Needs sign-off:** yes (sampling; the check's tolerance is R6's).
- **Used by:** Checks; R6.
- **Ruling:**

### D-25 · Tubing from the design rate

- **Step:** R3, 2026-10-30. Takes WP §6's lead.
- **Choice made:** the smallest of 62 to 157 mm (220.5 mm for gases) whose no-slip mixture velocity at the initial design rate stays below the erosional velocity with $C$ from 100 to 150; one size larger with probability 0.3.
- **Alternatives:** v1's uniform draw from 76 to 165 mm; sizing at the current stage; a critical-velocity check for gas wells.
- **Realism:** no public distribution of tubing sizes is in hand. Late-life wells keep their initial tubing, which is how they load up.
- **Needs sign-off:** yes (sampling).
- **Used by:** Tubing; Choke; Heat transfer; Gas lift: capacity.
- **Ruling:**

### D-26 · Roughness by life stage

- **Step:** R3, 2026-10-30.
- **Choice made:** SMP-42's log-uniform range split by stage: new tubing early, corroded tubing late.
- **Alternatives:** SMP-42 unchanged for all wells.
- **Realism:** the examples in hand give 30 and 46 µm; Farshad and Rieke (2006) are not in hand.
- **Needs sign-off:** yes (sampling).
- **Used by:** Roughness.
- **Ruling:**

### D-27 · The choke: Simpson only, sized to the tubing

- **Step:** R3, 2026-10-30. Takes WP-10.
- **Choice made:** a Simpson choke for every well, with a fully open bean of 0.25 to 0.6 of the tubing's diameter and a discharge coefficient of 0.75 to 0.90; profiles with equal shares.
- **Alternatives:** v1's $K_c$ draw (SMP-5); the Bernoulli choke for a share of wells.
- **Realism:** no public data on bean sizes or profiles over wells; the draw contains v1's range.
- **Needs sign-off:** yes (sampling).
- **Used by:** Choke; R4's choke draw.
- **Ruling:**

### D-28 · Who gets gas lift

- **Step:** R3, 2026-10-30. Takes WP §9's lead.
- **Choice made:** oils that cannot lift their static column get gas lift; other oils by stage (offshore 0.20 / 0.45 / 0.70, onshore 0.15 / 0.25 / 0.35), gases 0 / 0.05 / 0.15; a strong well whose balance point lies above 0.3 $L$ is left on natural flow in the core; no GOR limit.
- **Alternatives:** v1's rule (gas lift only for $f_g \le 0.2$, with probability 1/2); a share independent of stage.
- **Realism:** core oils 32% lifted, 58% late in life; gas lift is the main lift offshore, but no count of lifted wells is in hand.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5).
- **Used by:** Gas lift; R4's lift-gas draw.
- **Ruling:**

### D-29 · The injection depth

- **Step:** R3, 2026-10-30. Takes WP-6 with Golan and Whitson's (1991, §5.3) pressure balance.
- **Choice made:** the valve at the pressure-balance point of a surface injection pressure of 48 to 76 bar (Gilbert's gas column, a valve drop of 3.4 to 13.8 bar) against the flowing tubing pressure at a lifted design point, between 0.3 $L$ and a packer 30 to 150 m above the bottom.
- **Alternatives:** WP-6's 30 to 150 m above the perforations for every well; the bottomhole (INF-5, today's sampler).
- **Realism:** $H/L$ of 0.33 / 0.65 / 0.97; the injection pressures are an onshore textbook range, and offshore systems may run higher (no source in hand).
- **Needs sign-off:** yes (sampling; the Goals' new draw, feature 021).
- **Used by:** Gas lift: injection depth; Mapping.
- **Ruling:**

### D-30 · The lift gas's temperature

- **Step:** R3, 2026-10-30.
- **Choice made:** between the ambient at the valve (THM-9's default) and 30% of the way towards the surface temperature.
- **Alternatives:** THM-9's default for every well (`T_lg=None`); the reservoir temperature (SMP-44).
- **Realism:** the annulus gas exchanges heat with the formation on its way down (Hasan and Kabir 2018, §12.2.2), but no numbers are in hand; the 30% is an own choice.
- **Needs sign-off:** yes (sampling; a Goals' new draw).
- **Used by:** Gas lift: lift-gas temperature; Mapping.
- **Ruling:**

### D-31 · The lift-gas capacity

- **Step:** R3, 2026-10-30. Takes WP-5 with a lower ratio.
- **Choice made:** an injected GLR at capacity of 50 to 400 Sm³/Sm³ times the design liquid rate, within 0.02 to 2.0 kg/s.
- **Alternatives:** WP-5's 100 to 800 Sm³/Sm³ within 0.1 to 2 kg/s; v1's 0 to 5 kg/s.
- **Realism:** the examples in hand inject 27 to 77 Sm³/Sm³, and up to 184,000 Sm³/d per well (von Flatern 2015); total GLRs of 430 to 800 Sm³/Sm³ in small tubing are optima. The draws give 0.02 / 0.38 / 2.0 kg/s.
- **Needs sign-off:** yes (sampling; Bjarne's goal 5).
- **Used by:** Gas lift: capacity; R4's lift-gas draw.
- **Ruling:**

### D-32 · The heat-transfer coefficient

- **Step:** R3, 2026-10-30. Takes WP-4.
- **Choice made:** $h$ from Hasan and Kabir's (2012) series of the wellbore's $U_t$ and the formation's transient resistance at the well's age.
- **Alternatives:** v1's 10 to 40 W/(m² K) (kept in the tail).
- **Realism:** 3.2 / 6.3 / 11.9 W/(m² K), inside the 3 to 25 implied by Hasan and Kabir's examples; v1's range overstates heat loss for wells older than a month.
- **Needs sign-off:** yes (sampling).
- **Used by:** Heat-transfer coefficient.
- **Ruling:**

### D-33 · Heat capacities from correlations

- **Step:** R3, 2026-10-30. Takes RF-10.
- **Choice made:** oil by Cragoe (1929) at the API gravity, water by Sharqawy et al. (2010) at the salinity, gas by NIST's methane scaled by $(0.554/\gamma_g)^{0.4}$, each at a representative temperature and pressure.
- **Alternatives:** today's constants (SMP-9 to SMP-11).
- **Realism:** $c_{pg}$ of 2513 to 3178 J/(kg K) against today's 2225; the gas's scaling is an assumption until a correlation for natural gases is in hand.
- **Needs sign-off:** yes (sampling; a Goals' fluid constant).
- **Used by:** Heat capacities.
- **Ruling:**

### D-34 · The economic floor

- **Step:** R3, 2026-10-30. Takes WP-9 with a margin.
- **Choice made:** 1.6 Sm³/d of oil equivalent for onshore and platform wells, 8 Sm³/d for subsea wells; R4 drops a well whose best rate at its nominal conditions is below half its floor, and oil wells above a water cut of 0.98.
- **Alternatives:** one floor for all; no margin; a floor per structure shared by its wells.
- **Realism:** stripper limits of 10 bbl/d and 60 Mcf/d (EIA 2025, p. 10); offshore structures' revenue thresholds (Kaiser 2011, abstract only). Before any solve 2.0% of the drawn wells lie below half the floor.
- **Needs sign-off:** yes: the margin is Bjarne's (the Goals), and the filter is part of the dataset design (R7).
- **Used by:** The economic floor; R4; R7.
- **Ruling:**

### D-35 · Truncations at the model's and the data's limits

- **Step:** R3, 2026-10-30.
- **Choice made:** truncate $p_i$ at 950 bar (Hasan et al.'s data), $T_r$ at 450 K, the model's gradient at no less than 10 K/km, measured depth at 9342 m (redrawn), and API gravity at 16 (no heavy oils), in core and tail alike.
- **Alternatives:** accept and extrapolate; the model refuses only the API gravity, so the others are choices.
- **Realism:** the 950 bar bound cuts into the central 90% of the pressure draw in 6.8% of wells, the temperature bound in 13.2%; HPHT wells beyond exist but are rare; heavy oils (0.7% of the Gulf's below 22 °API) need a table.
- **Needs sign-off:** yes (sampling; `validity.md`'s ranges are signed off, and these bounds choose where the sampler stops).
- **Used by:** Initial reservoir pressure; Geothermal gradient; Geometry; Oil and gas at standard conditions.
- **Ruling:**

### D-36 · Volatile oils and condensates as tables; wet gases as black oil; the interim mapping

- **Step:** R3, 2026-10-30.
- **Choice made:** draw each volatile oil's and condensate's descriptors (API, gas gravity, $R_{si}$, saturation pressure, separator, the $p$ and $T$ to cover) for R5's tables; keep wet and dry gases as black oil (decision 4); for R6's prototype, map volatile oils and condensates to the black oil until the tables exist.
- **Alternatives:** tables for wet gases too; a fixed library of published example fluids with no drawn descriptors.
- **Realism:** the Goals put volatile oils and condensates in tables from a flash; in the interim their bubble and dew points are Vazquez–Beggs's, not the drawn ones.
- **Needs sign-off:** yes (sampling; R5's tables and their cost are for the checkpoint).
- **Used by:** Fluid class; Saturation pressure; Mapping; R5; R6.
- **Ruling:**
