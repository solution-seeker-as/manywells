# Sampler redesign — sources: wells and production

*R2 of `plans/sampler-redesign-plan.md`, 2026-10-30. One of two halves of `sources.md`: this file covers the wells and their production; reservoirs and fluids are in the other half. A draft for Bjarne's checkpoint (R8), not a spec. Plans are drafts.*

Each section gives a table of the evidence, the dependencies, leads for R3 (or R5), and the gaps. Values are in the model's units with the source's in brackets. *Population* says what a number describes and whether it counts wells, fields or volumes. *Strength* is strong (large public dataset or primary source), moderate (textbook rule or a few cases) or weak (trade press, a single example, a secondary mention).

Three computations back this file. All are public-data or model calculations, not real-well data of the maintainer, and only their summary numbers are given here:

- **Sodir wellbores.** Server-side statistics from the Norwegian Offshore Directorate's FactMaps service (layer 5000, `wellbore`; NLOD licence; fetched 2026-10-30 as aggregate query results, no per-wellbore rows): counts, water depths and MD/TVD percentiles of development wellbores with purpose "production", by the subsea flag and the content (oil, gas, ...). Each wellbore counts once, so sidetracks and multilateral branches count separately. The maximum inclination is published only for released wellbores and is empty for all but 3 production wellbores.

- **BOEM boreholes.** The US Bureau of Ocean Energy Management's borehole file (data.boem.gov, Well Information, "Borehole", fetched 2026-10-30, file dated 2026-10-25; public domain) lists 55,575 boreholes on the US outer continental shelf, with total MD, TVD, water depth, surface and bottom coordinates, type, status and well class (straight, directional, horizontal). We took the development boreholes (type D) and computed the MD/TVD ratio, the horizontal displacement $H$ between surface and bottom, and a two-segment fit: vertical to a kickoff depth $K$, then straight at an inclination $\theta$, which reproduces MD, TVD and $H$ exactly, $K = (\text{MD}^2 - \text{TVD}^2 - H^2)/\big(2(\text{MD} - \text{TVD})\big)$ and $\tan\theta = H/(\text{TVD} - K)$. $\theta$ is the inclination of an equivalent straight tangent, not the maximum inclination: a real well has a curved build section, and a horizontal well a near-horizontal end. The fit is valid ($0 \le K < \text{TVD}$) for 98% of the boreholes.
- **Taitel–Dukler scan.** The stratified to non-stratified transition of Taitel and Dukler (1976), as Barnea (1987) and Gomez et al. (1999, Eqs. 1–2) use it, evaluated over in-situ superficial velocities $v_{SL}$ from 0.003 to 5 m/s and $v_{SG}$ from 0.03 to 30 m/s, for oil at 10, 60 and 200 bar ($\rho_G$ = 8, 50 and 170 kg/m³) and water at 60 bar, in tubing of 62, 100 and 157 mm ID, at 0° to 45° above horizontal (Section 3).

## 1. Well types

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Producing wells, US | 918,481 (2024); 22% horizontal; 78% produce 15 BOE/d (2.4 Sm³/d oil equivalent) or less | US, onshore and offshore, wells (completions) | EIA (2025), p. 3 and p. 8 | strong, but onshore stripper wells dominate the count |
| Producing wells, US federal offshore Gulf of Mexico | 2,216 (1,869 oil, 347 gas), 0.24% of US wells | GoM OCS, 2024, wells | EIA (2025), Appendix B, Table B32 | strong |
| Deepwater GoM, dry tree against subsea | 997 dry-tree and 191 subsea wells producing to 95 structures in water deeper than 122 m (400 ft): 84% and 16% | GoM, January 2013, wells, deep water only | Liu, Wang and Kaiser (2014), from BOEM databases | moderate (trade journal, BOEM data) |
| Water depth of completed development boreholes, GoM | 46% below 60 m, 35% at 60–200 m, 5% at 200–500 m, 4% at 500–1000 m, 5% at 1000–1500 m, 5% at 1500–2500 m, 0.2% deeper | GoM OCS, boreholes with status "completed", 3,018 | BOEM borehole file, our computation | strong for the GoM |
| NCS production wellbores, subsea against platform | 1,688 subsea and 2,405 platform wellbores (41% subsea). Platform: 1,776 oil, 257 gas, 181 oil/gas, 51 gas/condensate. Subsea: 1,330 oil, 179 gas, 82 oil/gas, 52 gas/condensate | Norway, all years, development production wellbores with a TVD, counting branches | Sodir FactMaps, our statistics | strong for Norway |
| NCS production from subsea against platform wells | 131.3 against 125.4 million Sm³ o.e. in 2010: 51% subsea | Norway, 2010, by volume | NPD, reported in *Offshore* (2011) | moderate (volume, not wells) |
| NCS fields by development | 70 producing fields: 35 subsea, 31 platform, 4 both; of the subsea, 18 gas and 17 oil | Norway, 2011, by fields | Ildstad (2011), NPD | moderate (fields, not wells) |
| Offshore share of world oil | nearly 30% | world, 2015, by volume | EIA (2016), from Rystad Energy | moderate (volume) |
| Offshore share of wells drilled | about 2,500 of 54,000 wells (5%) | world, 2021, wells drilled | Rystad Energy, via web summary | weak |
| Wells on artificial lift | 90% to 95% of producing wells worldwide; no global register exists | world, wells | JPT (industry estimate) | weak |
| Gas lift among artificial lift | ranks second to rod pumping by number of installations; the most attractive method offshore | world | Hasan and Kabir (2018), §12.2.1 | moderate |
| Gas lift offshore | often the method of choice for deepwater subsea wells | world | von Flatern (2015) | moderate |
| Whole platforms on gas lift | a North Sea platform with 33 platform wells and 2 subsea tiebacks, all gas-lifted, water cut 10% to 95% (mean 80%), GOR 78 Sm³/Sm³; a South China Sea platform with 15 wells, all gas-lifted | two platforms | Goh et al. (2007), IPTC 11647 | weak (two cases) |

**Dependencies.** Well type sets the surface conditions: subsea wells have the choke at the seabed and a flowline and riser downstream (Sections 2, 7); platform wells have the choke on the deck. Subsea wells sit in deeper water than platform wells (Liu et al. 2014; BOEM water depths). Per-well rates are higher offshore (Section 5), and the economic limit is higher (Section 11). Platform wells are drilled directionally from a central slot (Section 3).

**Leads for R3.**

- Counted by wells, the world is onshore: the US alone has about 400 times as many producing wells as the GoM. The goal's offshore weighting is therefore a design choice; the evidence only bounds the split within offshore.
- Within offshore, by wells: the GoM shelf is almost all platform wells; deep water is 84% dry tree (GoM 2013); on the NCS 41% of the production wellbores are subsea, and half the volume comes from subsea wells. A starting split of about 55% platform, 35% subsea and 10% onshore, with each above the goal's floors, fits these numbers; it is a lead, not a fitted share.
- Naturally flowing wells are a small minority of all wells, but common offshore early in life; gas lift is the main lift offshore. A well-type label should be drawn with a lift label (natural or gas lift) that depends on life stage and water cut (Section 9).

**Gaps.** No public count of the world's producing wells by type. The NCS counts include plugged wellbores and branches; counts of wells producing today need the status field. Brazil and West Africa, mostly subsea, have no public well counts in hand.

## 2. Surface temperature

The model's $T_s$ is the top of a linear ambient profile in TVD (THM-4), so it plays the role of the geotherm's surface intercept, not of the air at the wellhead.

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Seabed temperature, deep water | 277.6 K at 1,219 m (40 °F at 4,000 ft), with the sea surface at 297 K (75 °F); linear profile assumed | one deepwater GoM well | Hasan and Kabir (2018), Fig. 7.17 | moderate (one example) |
| Seabed temperature, 1500–3000 m | 274 to 277 K (1 to 4 °C) | deep water, generic | flow-assurance article (JPT) | weak |
| Seabed temperature, Girassol (Angola) | 276 K (3 °C) at 1,300 m | one field | *Offshore* (trade) | weak |
| Seabed temperature, Ormen Lange (Norwegian Sea) | about 272 K (−1 °C) at 850–1,100 m | one field | operator and trade press | weak |
| Seabed temperature, GoM upper slope | 280 to 283 K (6.6 to 9.7 °C) at about 540 m | one site | MacDonald et al. (2005) | weak |
| Water depth of producing wells | GoM completed development boreholes: median 66 m, P90 1,010 m, P99 2,307 m | GoM OCS | BOEM borehole file, our computation | strong for the GoM |
| Water depth of subsea against platform wells, NCS | platform oil wellbores P10 / P50 / P90: 70 / 109 / 175 m; subsea oil wellbores 124 / 329 / 357 m; subsea gas wellbores 93 / 298 / 854 m | Norway, production wellbores | Sodir FactMaps, our statistics | strong for Norway |
| Water depth of subsea wells, GoM | not separable in the BOEM file (subsea wells also carry a structure ID); GoM deepwater subsea wells lie in 122 m and deeper | GoM | BOEM; Liu et al. (2014) | weak |

**Dependencies.** Seabed temperature falls with water depth to 1–4 °C below about 1,000 m, except in the Norwegian Sea, where deep water is below 0 °C. On shelves it follows the region's bottom water and the season. Onshore, the surface intercept is near the mean annual air temperature. For a platform well the profile passes through the water column, which the linear profile of THM-4 cannot represent; in the GoM the median water depth is 2% of the median TVD (66 m of 2,840 m), so the error is small on the shelf and larger for deepwater platforms (1,000–2,000 m of water over 4,000–6,000 m TVD).

**Leads for R3.**

- Subsea: $T_s$ from the seabed temperature as a function of water depth, about 277 K below 1,000 m with a tail to 272 K, warmer on shelves.
- Platform: $T_s$ near the seabed temperature for shelf wells, which makes the geotherm right below the mudline; the water column is a small part of the well.
- Onshore: $T_s$ from mean annual air temperatures of producing regions, roughly 263 to 303 K.
- The geothermal gradient (other half) is then $(T_r - T_s)$ over the TVD below the mudline; keep it well above 5 K/km (`validity.md`).

**Gaps.** No open table of seabed temperature against water depth by basin; the World Ocean Atlas (NOAA, public domain) would give one. No open distribution of mean annual surface temperatures over producing wells.

## 3. Geometry and stratified flow

### Trajectories

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Well class shares | straight 3.1%, directional 91.0%, horizontal 5.9% | GoM OCS development boreholes with status "completed" (3,018) | BOEM borehole file, our computation | strong for the GoM |
| Well class shares, all development boreholes | straight 10.6%, directional 86.4%, horizontal 2.9%; straight 16% of those spudded before 1980, 3% since 2010; horizontal 0.1% before 1980, 9% in the 1990s, 7% since 2010 | GoM OCS development boreholes, all statuses (35,222) | BOEM, our computation | strong for the GoM |
| MD/TVD | median 1.11, P5 1.007, P95 1.58; directional median 1.11; horizontal median 1.37, P95 2.54 | completed development boreholes, GoM | BOEM, our computation | strong |
| Equivalent tangent inclination $\theta$ | median 36°, P5 13°, P25 25°, P75 48°, P95 66°. Shares: 3.5% below 10°, 11% at 10–20°, 22% at 20–30°, 22% at 30–40°, 21% at 40–50°, 11% at 50–60°, 6% at 60–70°, 2.1% at 70–80°, 1.1% at 80–90° | completed development boreholes, GoM | BOEM, our computation (two-segment fit) | strong as a fit; the true maximum inclination is higher |
| $\theta$ by class | directional: median 35°, P95 60°; horizontal: median 67°, P5 49°, P95 82°; straight: median 1.5°, P95 9° | as above | BOEM, our computation | strong |
| $\theta$ by water depth | below 61 m (200 ft): median 32°; 61–305 m: 39°; 305–1,524 m: 41°; deeper: 36° | as above | BOEM, our computation | strong |
| Kickoff depth, as a fraction of TVD | median 0.48, P5 0.18, P95 0.87 (wells with $\theta \ge 5$°); deeper in deep water (median 0.69 below 1,524 m of water) | as above | BOEM, our computation (fitted $K$) | moderate (a fitted equivalent) |
| TVD | median 2,840 m, P5 1,480 m, P95 6,300 m | as above | BOEM, our computation | strong |
| MD/TVD, NCS | platform wellbores: P10 1.07, P25 1.20, P50 1.54, P75 1.96, P90 2.46 (2,405); subsea: P10 1.17, P25 1.46, P50 2.04, P75 3.15, P90 4.09 (1,688) | Norway, development production wellbores, branches included | Sodir FactMaps, our statistics | strong for Norway |
| MD/TVD by content, NCS | oil: platform P10 / P50 / P90 1.09 / 1.62 / 2.56, subsea 1.35 / 2.47 / 4.33; gas: platform 1.02 / 1.10 / 1.58, subsea 1.02 / 1.18 / 1.66; gas/condensate: 1.10 / 1.37 / 2.05 (platform), 1.12 / 1.30 / 1.76 (subsea) | as above | Sodir FactMaps, our statistics | strong for Norway |
| Directional share of wells drilled | 1,116 of 1,718 wells drilled in 1998 (65%), including exploration | GoM, 1998 | *Offshore* (2001), MMS data | moderate |
| Horizontal share, US | 22% of producing wells in 2024 (10% in 2014), mostly onshore shale | US, wells | EIA (2025), p. 3 | strong |

### Where stratified flow sets in (upward flow)

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Upward flow patterns in experiments | in a 9.72 cm pipe at 2.7 to 14.8 bar (CO₂, oil and brine), stratified flow has completely disappeared at +2°, for $v_{SG}$ 1 to 11 m/s and $v_{SL}$ 0.1 to 1.5 m/s | laboratory, 0° to ±5° | Wilkens, Gopal and Jepson (ICMT, n.d.), abstract | moderate (one rig, near field pressures) |
| Empirical flow-pattern map | Mukherjee and Brill's map has a stratified boundary for horizontal and downward flow only; upward flow is bubble, slug or annular-mist at every angle from 0° to 90° | 3.8 cm pipe, air with kerosene or lube oil, 0° to ±90° | Brill and Mukherjee (1999), p. 44, Eqs. 4.128–4.134; p. 45, Fig. 4.19 | moderate (one rig, many points) |
| Inclined wells | flow patterns in upward-inclined flow are those of vertical flow; stratified flow is observed in near-horizontal systems; churn flow rarely beyond 20° from vertical | wells | Hasan and Kabir (2018), §4.2.1 | moderate |
| Taitel–Dukler, horizontal | stratified up to $v_{SL}$ = 0.11 to 0.24 m/s, 9% to 23% of the scanned grid with $v_{SL} \ge 0.05$ m/s | the scan's 12 cases | our computation | moderate (model) |
| Taitel–Dukler, 0.5° and 1° above horizontal | at 0.5°, stratified only for $v_{SL} \le$ 0.052 m/s, at most 0.2% of the grid with $v_{SL} \ge 0.05$ m/s; at 1°, only for $v_{SL} \le$ 0.017 to 0.043 m/s, none of that grid | as above | our computation | moderate |
| Taitel–Dukler, 5° above horizontal | $v_{SL} \le$ 0.011 m/s (none for 62 mm oil cases) | as above | our computation | moderate |
| Taitel–Dukler, 10° above horizontal (80° from vertical) | $v_{SL} \le$ 0.006 m/s; about 4 m³/d of in-situ liquid in 100 mm tubing | as above | our computation | moderate |
| Taitel–Dukler, 15° and 20° above horizontal | 15°: $v_{SL} \le 0.004$ m/s, in one case (water, 157 mm); 20° and steeper: none at $v_{SL} \ge 0.003$ m/s | as above | our computation | moderate |
| Validation of Taitel–Dukler for inclined pipes | good to about ±10° | laboratory | Barnea et al. (1985), as cited in a web source | weak (secondary) |
| Drift-flux closures' angles | Hasan and Kabir's deviated model tested against their own data to 32° from vertical and literature data to 86°; Shi et al. fitted drift-flux parameters from 0° to 88° from vertical in a 15 cm pipe | laboratory | Hasan and Kabir (1988), p. 474–475; Shi et al. (2005) | moderate |

**Dependencies.** Platform wells are directional because many wells share one slot cluster; their inclination grows with the reservoir's offset from the platform and is larger in moderate water depths (GoM medians above). Kickoff is deeper in deep water, where the well first passes the water column. Horizontal wells have higher productivity (Section 4). Stratified flow needs low liquid velocity, so it concerns low-rate and gas wells more than oil wells.

**Leads for R3.**

- **The public evidence does not support "mostly vertical" for offshore production wells.** In the GoM, 3% of the completed development boreholes are straight, and the median equivalent tangent inclination is 36°. On the NCS, the median MD/TVD is 1.54 for platform and 2.04 for subsea production wellbores, which only long high-angle or horizontal sections give; NCS oil wellbores are the most deviated (subsea median 2.47), gas wellbores the least (medians 1.10 to 1.18, close to the GoM's). Onshore, 22% of US producing wells are horizontal (EIA 2025); no public share of vertical wells onshore is in hand. Bjarne's goal (mostly vertical, some slightly deviated, some with a highly deviated bottom section) is therefore a design choice for offshore wells, unless a private comparison with field data says otherwise. This needs his ruling.
- **Horizontal wells to the heel.** Many offshore oil wells end in a horizontal drain, which the model cannot hold. A lead: model such a well from the heel (the end of the build section, at the inclination limit) to the wellhead, and let the inflow stand for the horizontal drain, with a PI multiplier. This keeps the share of horizontal oil wells realistic without horizontal cells.
- **Inclination limit (decision 5).** All three lines of evidence agree that stratified flow in upward flow is confined to a few degrees above horizontal at realistic liquid rates. A limit of 80° from vertical leaves stratified flow possible in Taitel–Dukler only below $v_{SL}$ = 0.006 m/s, below the economic limit of an oil well, but it lies beyond the 32° of Hasan and Kabir's own data. A limit of 70° clears the scan entirely. 3.2% of the completed GoM boreholes have an equivalent tangent above 70°, and 1.1% above 80°. The lead: 70° to 80° from vertical, with 75° as the starting point.
- A trajectory draw by well type: straight; build-and-hold with kickoff at 0.2 to 0.85 of TVD and a tangent of 10° to 60°; or a highly deviated bottom section at 60° to 75°. The BOEM shares and quantiles above calibrate it.

**Gaps.** No inclinations for NCS development wellbores (not published); no trajectory statistics for the UK, Brazil or onshore conventional wells in hand. The BOEM fit gives an equivalent tangent, not the maximum inclination or the build rate; the directional survey points (resource list) would give both. The primary papers on the transition (Taitel and Dukler 1976; Barnea et al. 1985; Barnea 1987) are not in hand.

## 4. Productivity

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| PI for continuous gas lift | continuous gas lift for PI of about 0.5 bbl/(d psi) [1.15 Sm³/(d bar), 0.011 kg/(s bar) at 850 kg/m³] or more; down to 0.2 bbl/(d psi) [0.46 Sm³/(d bar)] with high-pressure gas; wells below 0.5 are "low productivity" | textbook rule | Guo, Lyons and Ghalambor (2007), §13.1 and §13.3, p. 13/182–183 | moderate |
| Permeability of reservoirs | median permeability 10 to 1,000 mD depending on porosity (siliciclastics: 10 mD at 5% porosity, 95 mD at 20%, 570 mD at 30%); 47% of siliciclastic and 28% of carbonate points above 100 mD | world, reservoirs (IHS, NPD, DTI, DOE databases) | Ehrenberg and Nadeau (2005), Table 1, p. 437; p. 439 | strong (reservoirs, not wells) |
| Oil inflow | $J = kh/\big(141.2 B\mu(\ln(r_e/r_w) - 0.75 + s)\big)$ in field units; PI scales with $kh/\mu$ | theory | Guo, Sun and Ghalambor (2008), Ch. 4; Ahmed (2006), Ch. 7 | strong (theory) |
| Vogel's inflow | for solution-gas drive below the bubble point; 21 reservoir conditions | simulation | Vogel (1968); `validity.md`, note 10 | strong |
| Fetkovich exponent | $n$ = 0.57 to 1.00 in a large number of oil-well tests | oil wells | Fetkovich (1973), as cited by Jansen (2015), p. 159 | moderate |
| Gas backpressure exponent | $n$ from 1.0 (laminar) to 0.5 (fully turbulent); curves with $n$ outside 0.5 to 1.0 are taken as erroneous | gas wells, more than 500 tests (Rawlins and Schellhardt 1936) | Ahmed (2006), Eq. 8-20, p. 556; Jansen (2015), Eq. 6.63, p. 158; secondary citations of Rawlins and Schellhardt | strong for the range |
| Backpressure coefficient | $C$ = 0.01 Mscf/(d psi²ⁿ) with $n$ = 0.8 in a worked example | one example | Guo et al. (2007), Example Problem 6.1, Table 6.1, p. 6/71 | weak |
| Forchheimer form | $q_g = kh\thinspace[m(\bar p) - m(p_{wf})]/\big(1424T(\ln(r_e/r_w) - 3/4 + S + Dq_g)\big)$, so $\bar p^2 - p_{wf}^2 = aq + bq^2$ with $a$ from $kh$ and the skin, $b$ from the non-Darcy coefficient $D$ | theory | Guo, Sun and Ghalambor (2008), Eq. 4.10, p. 69 | strong (theory) |
| Non-Darcy coefficient | $D$ = 0.001 d/Mscf in worked examples; best measured by multi-rate tests, theoretical estimates may be off by 100% | examples | Guo et al. (2008), pp. 174, 209, 231; a Stanford report on non-Darcy flow | weak |

**Dependencies.** The PI depends on $kh/\mu$, on the skin and on the completion: a horizontal or highly deviated bottom section multiplies it (Joshi 1991, not in hand). INF-1's slope at $p_0 = p_r$ is $1.8w_{l,\max}/p_r$, so a well whose PI at small drawdown is $J$ has $w_{l,\max} = Jp_r/1.8$. A gas well's $a$ scales with $\mu z T/(kh)$ and its $b$ with $D$, which falls with permeability (β correlations).

**Leads for R3 and R5.**

- R3: draw $kh$ (from permeability and net pay, with the other half's reservoir draws), then the PI from the radial-flow formula with a skin, and a multiplier for deviated and highly deviated sections. This ties the rate to the reservoir and the completion, where v1's $w_{l,\max}$ is uniform and independent of both.
- R5: the Forchheimer form with $a$ from $kh$ and $b$ from $D$ is physically parameterised and reduces to the backpressure law's limits ($n$ = 1 when $bq \ll a$, $n$ = 0.5 when $bq \gg a$). The backpressure form needs $C$ and $n$, for which only the range of $n$ has public support.

**Gaps.** No public distribution of oil-well PI or gas-well $C$, AOF or $D$ by well type. Golan and Whitson (1991) and Economides et al. (1994) are in hand but have no usable text layer; their deliverability chapters were not read (resource list).

## 5. Rates

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Oil wells by rate, GoM | share of oil wells by annual-average rate (BOE/d, Sm³/d o.e. in brackets): ≤ 10 [1.6]: 9.8%; ≤ 15 [2.4]: 14.6%; ≤ 100 [16]: 52.4%; 100–400 [16–64]: 16.4%; 400–1,600 [64–254]: 13.9%; 1,600–6,400 [254–1,018]: 11.8%; 6,400–12,800 [1,018–2,035]: 4.2%; above 12,800: 1.3% | GoM OCS, 2024, 1,869 oil wells (GOR ≤ 6,000 cf/b) | EIA (2025), Appendix B, Table B32 | strong |
| Production share of high-rate oil wells, GoM | the 12% of oil wells above 3,200 BOE/d [509 Sm³/d] produce 76% of the oil | as above | EIA (2025), Table B32 | strong |
| Mean oil-well rate, GoM | 1,049 b/d [167 Sm³/d] | as above | EIA (2025), Table B32 | strong |
| Gas wells by rate, GoM | ≤ 15 BOE/d: 45%; ≤ 100: 72%; 100–400: 15%; 400–1,600: 10%; above 1,600 BOE/d (about 9.6 MMscf/d, 270,000 Sm³/d): 2.3%; mean 1,153 Mcf/d [32,700 Sm³/d] | GoM OCS, 2024, 347 gas wells | EIA (2025), Table B32 | strong |
| Oil and gas wells, US | mean oil well 29 b/d [4.6 Sm³/d]; mean gas well 185 Mcf/d [5,200 Sm³/d]; 78% of wells at or below 15 BOE/d; 8% above 100 BOE/d | US, 2024 | EIA (2025), p. 8 | strong |
| A North Sea platform | 6,000 m³/d of oil from 33 platform wells and 2 subsea tiebacks, about 170 m³/d per well, at a mean water cut of 80% | one platform | Goh et al. (2007) | weak |
| Rate against tubing | 100 MMscf/d [2.8 million Sm³/d] per well needed 7 in. tubing (Bay of Bengal); 7 in. and 9-5/8 in. tubing compared for South Pars gas wells | single fields | SLB case study; *Oil & Gas Journal* | weak |

**Dependencies.** Rates are annual averages over calendar days, so wells shut in for part of the year sit in low brackets. Rate depends on PI and drawdown (Section 4), tubing (Section 6), water cut and lift (Section 9). Offshore wells produce about 36 times the US mean oil-well rate. Weighting by volume would favour the high-rate wells, which make up 12% of the GoM's oil wells and 76% of its oil; the goal counts wells.

**Leads for R3.** The GoM bracket distribution is the strongest public per-well rate distribution for offshore wells. It is broad: a log-rate spanning about four decades, with a mode near 50 to 100 BOE/d and a long upper tail. After the economic filter (Section 11) it can serve as the target that the drawn PI, pressures and tubing must reproduce, rather than as a draw.

**Gaps.** No public per-well rate distribution for the NCS, UK or Brazil in hand (the regulators publish per-field or per-well monthly data; resource list). No rate against tubing size.

## 6. Completion

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| API tubing | outer diameters 1.050 to 4.500 in. (27 to 114 mm); larger strings (5-1/2, 7, 9-5/8 in.) use casing-sized pipe in high-rate wells | standard | web summary of the SPE handbook; Guo et al. (2007), Appendix B | moderate |
| Large-bore wells | 7 in. tubing for about 100 MMscf/d; 9-5/8 in. considered for South Pars | gas wells | SLB; *Oil & Gas Journal* | weak |
| Erosional velocity | $v_e = C/\sqrt{\rho_m}$ with $C$ = 100 for continuous and 125 for intermittent service (ft/s, lb/ft³), up to 150–200 where corrosion is controlled and solids are absent; in SI, $v_e = 1.22\thinspace C/\sqrt{\rho_m}$ m/s with $\rho_m$ in kg/m³, so 8.6 m/s at 200 kg/m³ and 4.3 m/s at 800 kg/m³ for $C$ = 100 | design rule | API RP 14E, as quoted in an open-access paper (Contributions to Oil and Gas, 2026) | moderate (the standard is not in hand) |
| Liquid loading | the churn–annular transition, $v_{gc} = 3.1\thinspace[g\sigma(\rho_L - \rho_G)/\rho_G^2]^{1/4}$ (Taitel et al. 1980; Coleman et al. 1991); Turner et al. (1969) raised the constant by 20%. For water at 60 bar ($\rho_G$ = 50 kg/m³, σ = 0.06 N/m): 2.2 m/s, or 2.6 m/s with Turner's constant (our arithmetic) | gas wells | Hasan and Kabir (2018), Eq. 5.15 | strong (standard rule) |
| Wall roughness in examples | relative roughness 0.0006 (Guo et al. 2007, several examples); absolute 30 µm (Jansen 2015, examples); 46 µm (0.0018 in.) for every well of a field study (Hasan and Kabir 2018, Ch. 5) | examples | as given | weak |

**Dependencies.** Tubing is chosen for the design rate: wide enough that friction and erosion do not limit the rate, narrow enough that gas wells do not load up late in life. Gas lift needs tubing large enough for the lifted rate. The roughness grows with corrosion and scale over the well's life.

**Leads for R3.** Choose the tubing after the rate: the smallest size from a list (2-7/8, 3-1/2, 4-1/2, 5-1/2, 7 in.) that keeps the design mixture velocity at the wellhead below the erosional velocity with $C$ = 100 to 150. For gas wells, check the critical velocity at the late-life rate. This replaces SMP-2's uniform draw and links tubing to rate. SMP-42's roughness range (1.5 to 150 µm) brackets the example values.

**Gaps.** No public distribution of tubing sizes by well type or rate class. API RP 14E, Turner et al. (1969), Coleman et al. (1991) and Farshad and Rieke on roughness are not in hand.

## 7. Surface pressures

The model's $p_s$ is the pressure downstream of the production choke. On a platform it is the inlet manifold's pressure, close to the first-stage separator's; for a subsea well it is the subsea flowline's inlet pressure, which is the separator's plus the flowline and riser losses.

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| First-stage separator pressure | 41 to 83 bar (600 to 1,200 psi), set by the flowline pressure | textbook | Guo et al. (2007), §10.2.5, p. 10/124 | moderate |
| Stage pressures | equal pressure ratios between stages (Campbell 1976); two stages plus the stock tank are practically optimal | textbook | Guo et al. (2007), Eq. 10.5, p. 10/124 | moderate |
| Low-pressure separators | standard vessels rated at 125 psi (8.6 bar) working pressure | textbook | Guo et al. (2007), Tables 10.4, 10.6, 10.8 | moderate |
| A North Sea train | 65, 20 and 2 bara at the three stages | one platform (Gullfaks C) | NTNU course exercise (Stanko 2017), via a web search snippet | weak |
| Late-life low-pressure production | Oseberg B's first-stage separators ran at 67 bar; oil was rerouted to the second stage at 20 bar after 18 years of production; Njord lowered first and second stage pressures in tail production | two fields | *Offshore* (trade) | weak |
| Regional practice | North Sea first stages generally below about 52 bar (750 psi), GoM first stages higher | general | web summary | weak |
| Wellhead pressures in examples | 200 and 600 psia (14 and 41 bar) | two textbook examples | Guo et al. (2007), Example Problem 13.1 and a Ch. 6 problem | weak |

**Dependencies.** $p_s$ falls over a field's life as the operator moves wells to lower stages or lowers the stage pressures (Oseberg, Njord). It is higher for subsea wells by the flowline and riser losses. A gas well's $p_s$ is set by the export or compression suction pressure.

**Leads for R3.** A per-well $p_s$ that depends on well type and life stage: platform and onshore wells from about 2 bar (late life, third stage or low-pressure gathering) to 80 bar (early-life first stage); subsea wells from 10 to 100 bar. Bjarne's goal of separator pressures below 10 bar has public support in third stages (2 bara) and low-pressure separators (8.6 bar rated), and in late-life low-pressure production.

**Gaps.** No public distribution of wellhead, manifold or separator pressures over wells. No subsea flowline inlet pressures (the 3W dataset would give some; resource list). Arnold and Stewart, *Surface production operations*, is not in hand.

## 8. Chokes

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Choke types | positive (fixed bean) and adjustable chokes; sizes in 1/64 in. | textbook | Guo et al. (2007), §5.1, p. 5/60 | strong (description) |
| Bean sizes in examples | 24/64 and 32/64 in. (9.5 and 12.7 mm) | textbook examples | Guo et al. (2007), Ch. 5 | weak |
| Discharge coefficient | 0.62 to 0.90 in Sachdeva's model; 0.75 in an example; nozzle-type chokes from a correlation in the diameter ratio and Reynolds number | textbook | Guo et al. (2007), Eq. 5.4, p. 5/60; Eq. 5.17, p. 5/65 | moderate |
| Fitted discharge coefficients | 1.08 for oil-well liquid and condensate-well gas, 0.78 for oil-well gas, 1.53 for condensate-well liquid, fitted to 474 tests in southwest Louisiana | onshore wells | Guo et al. (2002), reported in Guo et al. (2007), p. 5/65–66 | moderate |
| Critical pressure ratio | about 0.55 for natural gas ($k \approx 1.28$) | theory | Guo et al. (2007), Eq. 5.1, p. 5/60 | strong |
| Distribution of choke openings in operation | none found in public sources | – | – | – |

**Dependencies.** The model's $K_c$ is a discharge coefficient times the fully open area. For a fully open bean of diameter $d$ in tubing of diameter $D$, $K_c/A = C_D(d/D)^2$; v1's 0.06 to 0.24 times the tubing area corresponds to $d$ = 0.27 to 0.53 $D$ at $C_D$ = 0.85, a 1.1 to 2.1 in. bean in 4-1/2 in. tubing (our arithmetic).

**Leads for R3.** Draw the fully open bean as a fraction of the tubing ID (0.25 to 0.6) and $C_D$ from 0.75 to 0.90, which keeps $K_c$ in v1's range. The choke's position stays close to uniform with modest weight at fully open (Bjarne's goal); no public evidence contradicts or supports a distribution.

**Gaps.** No public data on choke openings over wells or time; no catalogue of maximum bean sizes by well type in hand (vendor catalogues only).

## 9. Gas lift

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Which wells | continuous gas lift for PI of about 0.5 bbl/(d psi) or more and reasonable bottomhole pressure (supports a liquid column of 70% of the depth); intermittent for low PI or low pressure | textbook | Guo et al. (2007), §13.1, §13.3 | moderate |
| Which wells, offshore | recommended for high-volume wells with high static bottomhole pressure, offshore wells with strong water drive or under waterflood, with high PI and high GOR; applicable in high-angle, low-productivity and high-GOR wells; intermittent lift not above about 30 m³/d | world | von Flatern (2015) | moderate |
| Lift-gas rate per well | 0.5 to 6.5 MMscf/d [14,000 to 184,000 Sm³/d, about 0.13 to 1.7 kg/s at 0.8 kg/Sm³] across seven optimised wells producing up to 15,000 bbl/d [2,400 Sm³/d] | example field | von Flatern (2015), Fig. 1 | moderate |
| Lift-gas rate in an example | 1 MMscf/d [28,000 Sm³/d, 0.26 kg/s] for a well of 632 stb/d [100 Sm³/d] | textbook example | Guo et al. (2007), Example Problem 13.1, p. 13/184–185 | weak |
| Optimum total GLR | 4,500, 3,200 and 2,400 scf/stb [800, 570 and 430 Sm³/Sm³] at 400, 600 and 800 stb/d in 2.259 in. tubing at 200 psia wellhead | textbook example | Guo et al. (2007), Example Problem 13.1 | weak |
| Injection point | as deep as the injection pressure allows; the operating valve is typically the deepest; gas enters near the top of the perforations; in an example the packer is 61 m (200 ft) above mid-perforation and the gas enters just above it | textbook, review | von Flatern (2015); Hasan and Kabir (2018), §12.2.1; Guo et al. (2007), Example Problem 13.1 | moderate |
| Injection pressure in an example | 707 psig (49 bar) at the casing head | textbook example | Guo et al. (2007), Eq. 13.10 example, p. 13/203 | weak |
| Lift-gas temperature | the annulus gas exchanges heat with the tubing and the formation on its way down; modelled with a coupled tubing–annulus–formation balance | review | Hasan and Kabir (2018), §12.2.2 | moderate (no numbers extracted) |

**Dependencies.** Gas lift follows depletion and water cut: as $p_r$ falls and the water cut rises, a well stops flowing and goes on gas lift (Goh et al. 2007: whole platforms at 80% water cut on gas lift). The optimum injected GLR falls with the liquid rate; the lift-gas rate grows with the liquid rate. The injection depth is set by the packer and by the injection pressure.

**Leads for R3.** 

- Give gas lift to wells by life stage and water cut, not by the gas fraction, and keep it for gassy wells: the review literature names high-GOR wells as candidates, which supports Bjarne's goal 5.
- Scale the lift-gas rate to the well: an injected GLR of about 100 to 800 Sm³/Sm³ times the liquid rate, bounded by 0.1 to 2 kg/s. v1's 0 to 5 kg/s reaches 19 MMscf/d, above the examples found.
- Injection depth (feature 021): 30 to 150 m above the top of the perforations, capped by the depth that the injection pressure reaches.
- Lift-gas temperature: feature 021's default (the ambient at the injection point) is consistent with the annulus heat exchange described by Hasan and Kabir.

**Gaps.** No public distribution of lift-gas rates, injection depths or injection pressures over wells. Gas-lift chapters of Economides et al. (1994) and Golan and Whitson (1991) not read (no text layer).

## 10. Heat transfer

The model's $h$ multiplies the difference between the fluid and the undisturbed geotherm on the tubing's inner area (THM-1, THM-4). In Ramey's and Hasan and Kabir's terms the heat loss per length is $2\pi r_{ti} U_t k_e(T_f - T_e)/(k_e + r_{ti} U_t T_D)$ (Hasan and Kabir 2012, Eq. 6), so the model's $h$ is $U_t k_e/(k_e + r_{ti} U_t T_D)$: the wellbore's overall coefficient in series with the formation's transient resistance.

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Earth conductivity and diffusivity | 2.42 W/(m K) (1.4 Btu/(h ft °F)) and 1.03·10⁻⁶ m²/s (0.04 ft²/h) gave good results in different areas | field wells | Ramey (1962), p. 431 | moderate |
| Film coefficients | liquid water or condensing steam 1,100 to 11,000 W/(m² K) (200 to 2,000 Btu/(h ft² °F)), negligible resistance; turbulent gas 11 to 28 W/(m² K) (2 to 5) | rules of thumb | Ramey (1962), p. 428 | moderate |
| Overall coefficient, liquid-filled annulus | $U_{to}$ = 36 to 56 W/(m² K) (6.3 to 9.8 Btu/(h ft² °F)) along one well, smallest at the bottom; 26 W/(m² K) (4.57) without annulus convection | one field example | Hasan and Kabir (2018), Table 6.1 | moderate |
| Overall coefficient, another well | 42 to 57 W/(m² K) (7.4 to 10), from the bottom to the top | one example | Hasan and Kabir (2018), Ch. 7, Fig. 7.39 | weak |
| Dimensionless temperature | $T_D = \ln\big(e^{-0.2t_D} + (1.5 - 0.3719e^{-t_D})\sqrt{t_D}\big)$, with $t_D = \alpha t/r_w^2$; 3.1 at one month, 4.4 at one year, 5.5 at ten years for an 8-1/2 in. hole and Ramey's diffusivity | theory | Hasan and Kabir (2012), Eq. 2 | strong |
| The model's $h$ implied | 3 to 25 W/(m² K): for 4-1/2 in. tubing 4 to 13 W/(m² K) over one month to ten years with $U_t$ = 11 to 85 W/(m² K) and $k_e$ = 1.5 to 2.4 W/(m K); its upper limit $k_e/(r_{ti}T_D)$ is 5 to 15 W/(m² K) | derived | our computation from the rows above | moderate |
| Model's source | Zhang et al. (2006) state no range for $h$ | – | `validity.md` | – |

**Dependencies.** $h$ falls with production time (through $T_D$), with tubing size (through $r_{ti}$), and with annulus gas or insulation; it rises with formation conductivity. In the water column of a platform well, forced convection of seawater gives a much larger coefficient (Hasan and Kabir 2018, §6.3.4).

**Leads for R3.** Draw $h$ from the formula above with $U_t$, $k_e$ and the well's age, or directly from about 3 to 15 W/(m² K) by tubing size. v1's 10 to 40 W/(m² K) lies at and above the formation-limited ceiling for wells produced for more than a month, so it overstates the heat loss; a few higher values could stand for insulation-free early life. Vacuum-insulated tubing gives much smaller values (Hasan and Kabir 2018, Ch. 7, model it in one deepwater well; no number extracted).

**Gaps.** No distribution of $U$ over completions; Willhite (1967) and Alves et al. (1992) not in hand.

## 11. Economic limits

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Stripper well, IOGCC | 10 b/d or less of oil [1.6 Sm³/d], or 60 Mcf/d or less of gas [1,700 Sm³/d], over 12 months | US | EIA (2025), p. 10 | strong (definition) |
| Stripper well, IRS | 15 b/d or less of oil [2.4 Sm³/d], or 90 Mcf/d [2,500 Sm³/d] | US | EIA (2025), p. 10 | strong (definition) |
| Stripper wells' share | about 78% of US wells at or below 15 BOE/d; about 6% of US oil and gas | US, 2024 | EIA (2025), p. 3, p. 10 | strong |
| Low-rate wells offshore | 9.8% of GoM oil wells and 38% of gas wells at or below 10 BOE/d (annual averages, partial years included) | GoM, 2024 | EIA (2025), Table B32 | strong as data; weak as a limit |
| Economic limit of offshore structures | revenue thresholds of about USD 1.0 to 1.7 million per year per structure in 2005–2009 | GoM OCS structures | Kaiser (2011), abstract | weak (abstract only) |
| Offshore against onshore | revenue thresholds of offshore fields above those of bay and land fields; gas fields turn marginal sooner than oil fields | Texas | Kaiser and Yu (2010), abstract | weak |
| A field at its limit | Yme was closed at an expected 9,000 bbl/d [1,400 Sm³/d] for the field | one FPSO field | operator press release | weak |

**Dependencies.** The limit rises with operating cost: onshore stripper wells produce at 1 to 2 Sm³/d, while an offshore structure needs revenue of about USD 1 million a year, which at USD 60 per barrel is about 50 BOE/d per structure, shared by its wells. It rises for subsea tiebacks to leased host capacity, and falls with oil price.

**Leads for R3/R4.** A filter by well type: onshore oil wells below 1.6 Sm³/d of oil (IOGCC stripper boundary) and gas wells below 1,700 Sm³/d are dropped; offshore, the GoM low brackets show that wells do produce at tens of BOE/d, so a margin of about 10 BOE/d (1.6 Sm³/d) per well for platform wells and higher for subsea wells (say 50 BOE/d) is a lead, with Bjarne's ruling on the margin.

**Gaps.** No open per-well offshore economic limit; Kaiser (2011) and Kaiser and Yu (2010) full texts are not in hand.

## 12. Public per-well databases

| Database | URL | Licence | Variables | Records | Usability |
|---|---|---|---|---|---|
| BOEM borehole file (US OCS) | data.boem.gov, Well Information, "Borehole" (5010) | public domain (US government) | API number, spud and TD dates, MD, TVD, water depth, surface and bottom coordinates, type, status, complex ID, well class (straight, directional, horizontal) | 55,575 boreholes (35,222 development) | used here (Section 3); fetched 2026-10-30 |
| BOEM directional surveys | data.boem.gov, "Directional Survey (Points)" | public domain | survey points (MD, inclination, azimuth, coordinates) | all releasable boreholes; large | not fetched; would give maximum inclination, build rates and kickoff |
| BOEM production | data.boem.gov, Production Data | public domain | monthly oil, gas and water by lease and completion | all OCS completions | not fetched; would give per-well rates and decline |
| EIA well distribution | eia.gov/petroleum/wells | public domain | wells and production by rate bracket, state and federal offshore, 2000–2024 | 22 brackets per region and year | used here (Sections 1, 5, 11) |
| Sodir (Norwegian Offshore Directorate) factpages and FactMaps | factpages.sodir.no; factmaps.sodir.no, DataService layer 5000 | NLOD (Sodir's "Use of content" page) | purpose (production, injection, observation), content (oil, gas), subsea flag, total MD, final TVD, maximum inclination (released wellbores only), water depth, kick-off point (of a sidetrack from its parent), bottomhole temperature, completion date, status, facility | 4,093 development production wellbores with a TVD | aggregate statistics fetched through the FactMaps REST service (Sections 1–3); the bulk CSV export was refused by the command sandbox |
| UK NSTA open data | opendata-nstauthority.hub.arcgis.com | Open Government Licence (to be confirmed) | wells and wellbores, depths, locations | UK continental shelf | not fetched |
| Petrobras 3W | github.com/petrobras/3W | CC BY 4.0 (data), Apache 2.0 (code) | time series of pressures and temperatures at the downhole gauge and subsea tree, upstream and downstream of the production choke, and gas lift, for real offshore wells | 1,984 instances in the first version | not fetched (GitHub refused); would give subsea choke pressures |
| Equinor Volve | Equinor data village | Equinor open data licence | production per well, well paths, completions, for one North Sea field | a few production wells | not fetched; small |

## Resource list

| Reference | The decision it bears on | Assumed without it |
|---|---|---|
| Sodir factpages, development wellbore table (CSV export) | joint distributions offshore Norway (MD/TVD with TVD, water depth, status and completion year per wellbore) | the aggregate statistics of Sections 1–3 |
| BOEM directional survey points and production data | maximum inclination and build rates; per-completion rates and decline | the two-segment fit; EIA's brackets |
| Petrobras 3W dataset | subsea wellhead and downstream choke pressures and temperatures | the textbook ranges of Section 7 |
| Taitel and Dukler (1976), *AIChE Journal* 22, 47–55; Barnea, Shoham and Taitel (1985), *Chemical Engineering Science* 40, 131–136; Barnea (1987), *International Journal of Multiphase Flow* 13, 1–12; Shoham (2006), *Mechanistic modeling of gas–liquid two-phase flow in pipes*, SPE | decision 5's inclination limit | our scan of Taitel–Dukler and the two experiments |
| Kaiser (2011), *Applied Energy* 88, 2490–2508; Kaiser and Yu (2010), *Applied Energy* 87, 3235–3254 | the economic filter offshore | the GoM brackets and a margin |
| API RP 14E; Turner, Hubbard and Dukler (1969), *JPT* 21, 1475–1482; Coleman et al. (1991), *JPT* 43 | tubing selection | the rules as quoted in Section 6 |
| Rawlins and Schellhardt (1936), US Bureau of Mines Monograph 7; Fetkovich (1973), SPE 4529 | the gas-rate inflow's form and parameters (R5) | the range of $n$ from secondary sources |
| Golan and Whitson (1991) and Economides et al. (1994): text layers, or a reading of their gas-deliverability and gas-lift chapters | R5 parameters; gas-lift rates | as above |
| Joshi (1991), *Horizontal well technology* | PI of highly deviated sections | a multiplier left to R3 |
| Arnold and Stewart, *Surface production operations* | separator and manifold pressures | Section 7's textbook and trade values |
| NOAA World Ocean Atlas 2023 (public domain) | seabed temperature against water depth by basin | the single values of Section 2 |
| Farshad and Rieke (2006), on tubing roughness | roughness of new and used tubing | SMP-42's range |
| Alves, Alhanati and Shoham (1992); Willhite (1967) | $U$ for completions | Hasan and Kabir's examples |

## Working assumptions proposed

For the decision log (`decisions.md`); none is entered there by this step.

1. **Geometry offshore.** *Proposed:* keep Bjarne's goal (mostly vertical, some deviated, some with a highly deviated bottom section) as a design choice, and record that the GoM's completed development boreholes are 3% straight with a median equivalent tangent of 36°, and that NCS production wellbores have a median MD/TVD of 1.54 (platform) and 2.04 (subsea). *Alternative:* follow the public shares for offshore wells (few straight, most deviated, with oil wells often ending in a horizontal drain that is modelled to its heel, capped at the limit below) and keep "mostly vertical" for onshore wells. Needs Bjarne's ruling, since it bears on a confirmed goal.
2. **Inclination limit (decision 5).** *Proposed:* 75° from vertical. *Alternatives:* 80° (Taitel–Dukler leaves stratified flow only below $v_{SL}$ = 0.006 m/s; 1.1% of GoM boreholes beyond); 70° (none in the scan; 3.2% of GoM boreholes beyond). Needs sign-off (`validity.md`).
3. **Rate target.** *Proposed:* the EIA GoM 2024 distribution of per-well rates as the check for offshore wells, after the economic filter; not drawn directly. *Alternative:* draw the rate and back out the PI.
4. **Heat-transfer coefficient.** *Proposed:* $h$ from Hasan and Kabir's formula with $U_t$, $k_e$, tubing size and age, about 3 to 15 W/(m² K). *Alternative:* keep v1's 10 to 40 W/(m² K) for coverage, labelled as a tail.
5. **Lift-gas rate.** *Proposed:* an injected GLR of 100 to 800 Sm³/Sm³ times the liquid rate, bounded by 0.1 and 2 kg/s. *Alternative:* v1's 0 to 5 kg/s.
6. **Injection depth.** *Proposed:* 30 to 150 m above the top of the perforations, capped by the injection pressure. *Alternative:* at the bottomhole (INF-5).
7. **Gas-rate inflow (R5).** *Proposed:* the Forchheimer form with $a$ from $kh$ and $b$ from a non-Darcy coefficient. *Alternative:* the backpressure form with $n$ from 0.5 to 1 and $C$ from a target AOF.
8. **Separator pressure.** *Proposed:* $p_s$ by well type and life stage, about 2 to 80 bar for platform and onshore wells and 10 to 100 bar for subsea wells. *Alternative:* v1's log-normal truncated to 10 to 120 bar, with a separate low-pressure tail.
9. **Economic filter.** *Proposed:* below 1.6 Sm³/d of oil or 1,700 Sm³/d of gas for onshore and platform wells, about 8 Sm³/d o.e. (50 BOE/d) for subsea wells. *Alternative:* one floor for all well types. Needs Bjarne's ruling on the margin.
10. **Choke size.** *Proposed:* the fully open bean at 0.25 to 0.6 of the tubing ID with $C_D$ from 0.75 to 0.90. *Alternative:* v1's $K_c$ draw (SMP-5).
11. **Surface temperature.** *Proposed:* subsea from water depth (about 277 K below 1,000 m), platform near the seabed temperature, onshore from 263 to 303 K. *Alternative:* one range for all.

## References

- Ahmed, T. (2006), *Reservoir engineering handbook*, 3rd ed., Gulf Professional Publishing: Ch. 7 and Ch. 8, Eq. 8-20.
- Barnea, D. (1987), "A unified model for predicting flow-pattern transitions for the whole range of pipe inclinations", *International Journal of Multiphase Flow* 13(1), 1–12. Not in hand.
- Brill, J. P. and Mukherjee, H. (1999), *Multiphase flow in wells*, SPE Monograph 17: pp. 44–45, Eqs. 4.128–4.134, Fig. 4.19.
- BOEM (2026), Data Center, Well Information, Borehole file (5010) and its record layout and code definitions, data.boem.gov, file dated 2026-10-25, fetched 2026-10-30.
- Ehrenberg, S. N. and Nadeau, P. H. (2005), "Sandstone vs. carbonate petroleum reservoirs: a global perspective on porosity-depth and porosity-permeability relationships", *AAPG Bulletin* 89(4), 435–445: Table 1, p. 437; p. 439.
- EIA (2016), "Offshore production nearly 30% of global crude oil output in 2015", *Today in Energy*, 25 October 2016.
- EIA (2025), *The distribution of U.S. oil and natural gas wells by production rate, with data through 2024*, December 2025: pp. 3, 8, 10; Appendix B, Table B32.
- Goh, K.-C., Moncur, C. E., Van Overschee, P. and Briers, J. (2007), "Production surveillance and optimization with data driven models", IPTC 11647.
- Gomez, L. E., Shoham, O., Schmidt, Z., Chokshi, R. N., Brown, A. and Northug, T. (1999), "A unified mechanistic model for steady-state two-phase flow in wellbores and pipelines", SPE 56520: Eqs. 1–2.
- Guo, B., Lyons, W. C. and Ghalambor, A. (2007), *Petroleum production engineering: a computer-assisted approach*, Gulf Professional Publishing: §5.1–5.5 (pp. 5/60–5/66), Example Problem 6.1 (p. 6/71), §10.2.5 (p. 10/124), §13.1–13.3 and Example Problem 13.1 (pp. 13/182–13/185).
- Guo, B., Sun, K. and Ghalambor, A. (2008), *Well productivity handbook*, Gulf Publishing: Eq. 4.10 (p. 69); examples on pp. 174, 209, 231.
- Hasan, A. R. and Kabir, C. S. (1988), "Predicting multiphase flow behavior in a deviated well", *SPE Production Engineering* 3(4), 474–482.
- Hasan, A. R. and Kabir, C. S. (2012), "Wellbore heat-transfer modeling and applications", *Journal of Petroleum Science and Engineering* 86–87, 127–136: Eqs. 2 and 6.
- Hasan, A. R. and Kabir, C. S. (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., SPE: §4.2.1, Eq. 5.15, §6.3.4, Table 6.1, Fig. 7.17, §12.2.
- Ildstad, K. (2011), "Overview of subsea fields on NCS", Norwegian Petroleum Directorate, FORCE seminar, 5 July 2011.
- Jansen, J. D. (2015), *Nodal analysis of oil and gas wells: theory and numerical implementation*, version 2e, TU Delft: Eq. 6.63 (p. 158), p. 159, Table 6.2.
- Kaiser, M. J. (2011), "Empirical estimates of the economic limit of offshore structures in the Outer Continental Shelf Gulf of Mexico", *Applied Energy* 88(7), 2490–2508. Abstract only.
- Kaiser, M. J. and Yu, Y. (2010), "Economic limit of field production in Texas", *Applied Energy* 87(10), 3235–3254. Abstract only.
- Liu, M., Wang, Z. and Kaiser, M. J. (2014), "Deepwater Gulf decommissioning—2: structure inventory runs gamut of deepwater technologies", *Oil & Gas Journal*, 3 March 2014.
- *Offshore* (2001), "Directional drilling: horizontal-departure-to-TVD ratio decline continues in US Gulf", 1 February 2001.
- *Offshore* (2011), "Subsea wells dominate Norway production", 9 June 2011.
- Ramey, H. J. (1962), "Wellbore heat transmission", *Journal of Petroleum Technology* 14(4), 427–435: pp. 428, 431.
- Shi, H., Holmes, J. A., Durlofsky, L. J., Aziz, K., Diaz, L. R., Alkaya, B. and Oddie, G. (2005), "Drift-flux modeling of two-phase flow in wellbores", *SPE Journal* 10(1), 24–33.
- Sodir, Norwegian Offshore Directorate (2026), FactMaps DataService, layer 5000 (wellbore), factmaps.sodir.no, NLOD; aggregate queries of 2026-10-30.
- Taitel, Y. and Dukler, A. E. (1976), "A model for predicting flow regime transitions in horizontal and near horizontal gas–liquid flow", *AIChE Journal* 22(1), 47–55. Not in hand.
- Vargas, R. E. V. et al. (2019), "A realistic and public dataset with rare undesirable real events in oil wells", *Journal of Petroleum Science and Engineering* 181, 106223.
- Vogel, J. V. (1968), "Inflow performance relationships for solution-gas drive wells", *Journal of Petroleum Technology* 20(1), 83–92.
- von Flatern, R. (2015), "The defining series: gas lift", *Oilfield Review*, Schlumberger.
- Wilkens, R., Gopal, M. and Jepson, W. P. (n.d.), "Flow regime transitions in high-pressure inclined pipelines", Institute for Corrosion and Multiphase Technology, Ohio University, publication 8075: abstract.
- Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J. P. (2006), "Unified model of heat transfer in gas/liquid pipe flow", *SPE Production & Operations* 21, 114–122.
