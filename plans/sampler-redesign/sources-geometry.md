# Sampler redesign — sources: well geometries

*Second pass of R2 of `plans/sampler-redesign-plan.md`, on the geometry of real production wells, 2026-10-30. It extends Section 3 of `sources-wells-production.md` and answers Bjarne's request for examples of real geometries before he sets the shares (plan, Where we are). A draft for the checkpoint (R8), not a spec. Plans are drafts.*

The first pass had borehole summaries only: well-class shares and an equivalent tangent from a two-segment fit for the completed development boreholes in the US Gulf of Mexico (GoM) [own analysis of BOEM's borehole file], and MD/TVD percentiles of Norwegian production wellbores [own analysis of Sodir's FactMaps]. This pass adds directional surveys: per-well surveys from two regulators, read station by station.

Values are in model units: m, degrees from vertical, and °/30 m for build, drop and dogleg rates. *Population* says what a number describes. *Strength* is strong (a large public dataset or a primary source), moderate (a few hundred wells, a textbook rule or a transfer between datasets) or weak (a single example or a secondary mention).

## Method

Four computations back this file, all on public real-well data. Their statistics, populations and methods are kept outside the repo, by the maintainer's decision of 2026-11-06, and this file cites them as the analysis of their public source, such as [own analysis of BOEM's directional surveys]; the files state what the statistics show, not their values.

- **GoM surveys (BOEM).** BOEM's directional survey points (data.boem.gov, public domain, file dated 2026-10-25, fetched 2026-10-30) for the boreholes with production in BOEM's OGOR-A reports from 1996 to 2024, split by fluid, platform or subsea, and gas lift: [own analysis of BOEM's directional surveys], with the split's platform distances and gas-lift shares in [own analysis of BOEM's borehole file]. For subsea wells, depths are measured below the mudline, because the model's well ends at the subsea tree.
- **Dutch surveys (NLOG).** NLOG's bulk deviation surveys (nlog.nl, dated 2026-09-23) of development hydrocarbon boreholes, joined to its borehole and facility layers (fetched 2026-10-30): [own analysis of NLOG's deviation surveys].
- **UK wellbores (NSTA).** Server-side MD/TVD statistics of offshore development producers from the NSTA layer "UKCS offshore wellbores top holes (ED50)", queried 2026-10-30: [own analysis of NSTA's wellbore layer]. The layer has no surveys.
- **Transfer from MD/TVD.** For the UK and Norway, where only MD/TVD is public, the pooled GoM and Dutch surveys give the share of each inclination class per MD/TVD bin [own analysis of BOEM's directional surveys]. Weighting those shares by a population's MD/TVD percentiles gives a rough estimate of its inclination shares [own analysis of NSTA's wellbore layer and Sodir's FactMaps]. It assumes that wells with the same MD/TVD have the same inclinations in every region, which ignores how far wells step out against their depth.

### What is measured

The survey quantities (maximum and bottom inclination, kickoff, build rate, hold angle, drop rate, maximum dogleg, and the cuts at 70°, 75° and 80°) are defined in [own analysis of BOEM's directional surveys], Method.

Profile classes, from the whole trajectory:

| Class | Rule | Sampler class |
|---|---|---|
| V, vertical | maximum below 5° | vertical |
| J, build and hold | maximum 5° or more, bottom below 60°, not S | slightly deviated if the maximum is below 20°, deviated otherwise |
| S, build, hold and drop | maximum 10° or more, and the bottom at least max(10°, a third of the maximum) below it | as J, by its maximum |
| HD, highly deviated bottom | bottom 60° to 80°, not S | high-angle |
| H, horizontal drain | bottom 80° or more | horizontal (modelled to its heel) |

The sampler classes put J and S wells with a maximum of 60° or more, and a bottom below 80°, into high-angle as well.

## 1. Shares and inclinations by split

### Gulf of Mexico, producing boreholes (BOEM surveys)

Own analyses of BOEM's directional surveys for this subsection: profile-class shares, maximum and bottom inclinations, and the shares whose maximum exceeds 70°, 75° and 80°, for all producing boreholes, first boreholes and sidetracks, and by setting and water depth, fluid and spud era, also in the sampler classes [own analysis of BOEM's directional surveys].

### Netherlands, development boreholes (NLOG surveys)

Own analyses of NLOG's deviation surveys for this subsection: the same quantities for Dutch offshore platform and onshore boreholes, by fluid and era [own analysis of NLOG's deviation surveys]. The onshore oil boreholes are mostly shallow heavy-oil wells, produced on pumps and outside the model; they are the only large set of near-vertical production wells in hand. The Dutch offshore is a gas province of platforms and satellites; its subsea boreholes are too few to summarise.

### UK and Norway, from MD/TVD (rough)

Own analyses for this subsection: MD/TVD percentiles of UK offshore development producers, by water depth and spud era, with the inclination shares estimated from them [own analysis of NSTA's wellbore layer]; the same estimates for Norway's platform and subsea production wellbores, by fluid [own analysis of Sodir's FactMaps].

The Norwegian counts include sidetracks and multilateral branches, each a wellbore; a branch is a horizontal drain more often than a main bore, so these shares overstate wells. The maximum inclination is not published for Norwegian development wellbores.

**Dependencies.**

- *Setting.* Platform wells are directional: a platform has many slots and one location. Their maximum inclination rises with the water depth up to 1,000 m, where a platform drains a larger area [own analysis of BOEM's directional surveys]. Subsea wells are drilled from a template above each reservoir, so in the GoM they are no less deviated than platform wells. In shallow water more of them are vertical; in deep water they kick off deeper below the mudline (Section 2).
- *Fluid.* Gas wells are more often vertical than oil wells, in the GoM as a whole and among its subsea wells [own analysis of BOEM's directional surveys], and in Norway gas wellbores have a much lower median MD/TVD than oil wellbores [own analysis of Sodir's FactMaps]. Horizontal drains are rarer in gas wells [own analysis of BOEM's directional surveys].
- *Era.* High-angle wells and drains spread after 1980 and dominate new Dutch wells: most of the offshore boreholes started since 2000 end in a drain or are high-angle [own analysis of NLOG's deviation surveys]. In the GoM the share above 75° rises from the wells spudded before 1980 to those spudded since 2010 [own analysis of BOEM's directional surveys]; in the UK the median MD/TVD rises from the wells spudded before 1990 to those spudded since 2010 [own analysis of NSTA's wellbore layer].
- *Sidetracks.* Sidetracks are more deviated than first boreholes [own analysis of BOEM's directional surveys]. Counting boreholes rather than wells raises every high-angle share; the first-borehole lines are the nearer proxy for wells.
- *Gas lift.* Profile class barely changes whether a GoM oil borehole was ever on gas lift, except for the few vertical ones, which were less often. The setting does: platform oil boreholes were far more often on gas lift than subsea ones [own analysis of BOEM's borehole file]. This is a lead for the gas-lift draw, not for the geometry.

## 2. Profile parameters

Own analyses for this section: kickoff fraction, build rate, maximum dogleg, hold angle and bottom inclination by class and setting, the S wells' drop, the highly deviated wells' profile, and the horizontal drains' tangent, landing build, heel and drain length, for the GoM [own analysis of BOEM's directional surveys] and the Netherlands [own analysis of NLOG's deviation surveys]; MD/TVD against the maximum inclination, pooled over the GoM and Dutch surveys, with the shares of drains and vertical wells per MD/TVD bin [own analysis of BOEM's directional surveys].

**Dependencies.** Kickoff is deeper, as a fraction of TVD, in subsea wells measured from the mudline, and in deep water. Build rates of the main build are gentle; drains land with a second, steeper build. Drains sit in shallower reservoirs in the GoM. The S profile drops back to a near-vertical bottom, so it passes the reservoir almost vertically; S wells are a large minority of offshore boreholes, and more common among the GoM's wells spudded before 1980 [own analysis of BOEM's directional surveys and NLOG's deviation surveys].

## 3. Wells beyond 70°, 75° and 80°

Own analyses for this section: the shares whose maximum exceeds 70°, 75° and 80°, and how far above TD the first 75° point lies, for the GoM and for horizontal drains in the GoM and the Netherlands [own analysis of BOEM's directional surveys], for the Netherlands [own analysis of NLOG's deviation surveys], and, estimated from MD/TVD, for the UK [own analysis of NSTA's wellbore layer] and Norway [own analysis of Sodir's FactMaps].

Two kinds of well cross the limit. A horizontal drain reaches 75° just above its heel, so ending the modelled well there, as Bjarne ruled, loses a small fraction of the TVD and the drain itself. A high-angle tangent above 75°, as in an extended-reach well, crosses the limit far up the well, in a sizeable minority of the GoM wells that reach 75° [own analysis of BOEM's directional surveys]. Cutting such a well there would lose much of it, so it is better modelled with its tangent capped at the limit.

## 4. Published profile types

| Quantity | Value | Population | Source | Strength |
|---|---|---|---|---|
| Profile types | Type I build and hold (J), Type II build–hold–drop (S), Type III deep kickoff and continuous build | textbook classification | Bourgoyne et al. (1986), Ch. 8 (not in hand; to be checked) | weak until checked |
| Dogleg severity | the rate of change of the trajectory, in °/30 m; hybrid rotary steerable tools build up to 18°/30 m; short-radius wells have a curve of 44 m radius or less | definitions | Mantle (2014) | moderate |
| Drift-flux closures over inclination | Hasan and Kabir's deviated-well model tested to 32° (own data) and 86° (literature); Shi et al. fitted 0° to 88° | laboratory | first pass, Section 3 | moderate |

No open-access paper or textbook in hand gives statistics of real production well profiles. The survey statistics cited above are the evidence; the textbook classification only names the types.

## 5. Example profiles

Each profile is built from the medians of its class: TVD, KOP fraction, build rate, hold angle and, for S wells, the drop rate and bottom inclination. The build is a circular arc. None is a single real well. The local papers library holds a real well near each class median for comparison; they have the same shape and nearly the same MD/TVD, except in the HD class, where the published well's MD/TVD, 1.90, lies above the profile's and the class median [own analysis of BOEM's directional surveys]. Depths are from the rotary table for platform and onshore wells and from the mudline for subsea wells.

The profiles, each with its class size and MD/TVD, are in [own analysis of BOEM's directional surveys] for the GoM (platform oil well, build and hold; platform gas well, S; subsea oil well, J, below the mudline; highly deviated bottom; oil well with a horizontal drain, modelled to its heel at 75°) and in [own analysis of NLOG's deviation surveys] for the Netherlands (onshore gas well, slightly deviated; onshore gas well, near-vertical; offshore platform gas well, S). The real horizontal well goes on to 90° and a drain beyond its heel, which the inflow stands for.

No onshore conventional well in hand flows naturally and has a survey, apart from the Dutch onshore gas wells (Gaps).

## 6. Options for the shares

The model can take any (MD, TVD) survey (GEO-3, GEO-4), so every profile above can be drawn. With the ruling of 2026-10-30 the sampler draws four geometry classes: vertical, slightly deviated (tangent 5° to 20°), deviated (J or S, 20° to 60°), and highly deviated bottom (60° to 75°, which takes the horizontal drains modelled to their heel and caps any tangent above 75°). The shares below are by well type, in that order.

| Option | Platform (dry tree) | Subsea | Onshore conventional | Gas wells against oil wells | Overall, at 55% platform, 35% subsea, 10% onshore, before the gas adjustment |
|---|---|---|---|---|---|
| A. Follow the public shares (recent wells) | 5 / 8 / 62 / 25 | 5 / 10 / 50 / 35 | 25 / 30 / 35 / 10 | gas: twice the vertical share, two thirds of the highly deviated share (from Norway, [own analysis of Sodir's FactMaps]; the GoM shows a higher vertical share and a similar highly deviated one, [own analysis of BOEM's directional surveys]) | about 7 / 11 / 55 / 27 |
| B. Nearer the goal ("mostly vertical") | 40 / 25 / 20 / 15 | 40 / 25 / 20 / 15 | 50 / 25 / 15 / 10 | as A | about 41 / 25 / 20 / 15 |
| C. Even mix | 25 / 25 / 25 / 25 | same | same | none | 25 / 25 / 25 / 25 |
| A′. Recommended: A with a vertical floor | 10 / 10 / 55 / 25 | 10 / 10 / 45 / 35 | 30 / 30 / 30 / 10 | as A | about 12 / 12 / 49 / 27 |

Where A comes from: the four-class shares of GoM platforms, overall and in deep water [own analysis of BOEM's directional surveys], and of Dutch platforms, overall and since 2000 [own analysis of NLOG's deviation surveys]; the estimated high-angle shares of UK and Norwegian platforms [own analysis of NSTA's wellbore layer and Sodir's FactMaps]. Subsea: the GoM's subsea shares [own analysis of BOEM's directional surveys], against Norwegian subsea oil wellbores that are mostly above 75° (estimated, branches included) [own analysis of Sodir's FactMaps]; A takes a middle value. Onshore: Dutch onshore gas, and Dutch onshore oil, older, pumped and often vertical [own analysis of NLOG's deviation surveys]. In each, S wells are a sizeable part of the deviated class [own analysis of BOEM's directional surveys and NLOG's deviation surveys].

**Where they disagree, and why.**

- *A against B.* Offshore production wells are rarely vertical in every public set: the GoM [own analysis of BOEM's directional surveys], the Dutch offshore [own analysis of NLOG's deviation surveys], and the UK and Norway as estimated [own analysis of NSTA's wellbore layer and Sodir's FactMaps]. A platform or template has one surface location for many wells, so most wells step out to their targets with a moderate tangent [own analysis of BOEM's directional surveys]. B puts 40% vertical, many times the public share, and would fail acceptance criterion 3 (missing real conditions) for offshore wells. Vertical wells are common only among old onshore wells (Dutch onshore gas started before 1980, and Dutch onshore oil [own analysis of NLOG's deviation surveys]) and in the GoM's shallow-water subsea gas wells [own analysis of BOEM's directional surveys].
- *A against C.* C over-weights the vertical and slightly deviated classes against the offshore evidence, as A reads it (25% each against A's overall 7% and 11%), and under-weights the deviated class (25% against 55%). It spreads the inclinations evenly, which covers the slip closure's inclination terms evenly.
- *Era and counting.* The highly deviated share depends on the era and on counting boreholes or wells: GoM first boreholes have fewer high-angle wells and drains than all producing boreholes, and Dutch offshore wells since 2000 have the most [own analysis of BOEM's directional surveys and NLOG's deviation surveys]. A takes recent wells, counted by first boreholes where possible.
- *The offshore weighting.* Because the sampler counts offshore wells most, the overall shares follow the platform and subsea columns. Onshore wells change the overall vertical share by 2 to 5 points only.

**Recommendation: A′.** It follows the public shape (few vertical wells, most wells J or S with a moderate tangent, a large highly deviated share; Section 1) with a quarter to a third of offshore wells in the highly deviated class, and keeps a floor of 10% vertical wells per offshore type and 30% onshore. The floor keeps v1's geometry in the data for coverage and for comparison with `sol-1`, above the public share. Bjarne's "some slightly deviated" and "some with a highly deviated bottom section" hold in A′; "mostly vertical" does not, and the evidence does not support it for offshore wells. If he keeps "mostly vertical" as a design choice, B should be labelled as such in the datasheet.

**Leads for R3.**

- Draw the class by well type and fluid as in A′, then the profile:
  - *Vertical:* MD = TVD, or a drift of up to 3° as in real near-vertical wells [own analysis of NLOG's deviation surveys].
  - *Slightly deviated and deviated (J):* the KOP fraction about Beta-shaped, fitted to the platform and subsea (below the mudline) kickoffs; the build rate log-normal, fitted to the surveys' build rates; the hold angle 5° to 20° (slight) or 20° to 60° (deviated), shaped like the surveys' J holds [own analysis of BOEM's directional surveys].
  - *S (a third of the deviated class):* as J, then a drop at the surveys' drop rates to their bottom inclinations near TD; at the median rate the drop is long, over a thousand metres of MD or so [own analysis of BOEM's directional surveys and NLOG's deviation surveys].
  - *Highly deviated bottom:* the kickoff and build of the surveys' highly deviated wells, to 60° to 75°; for a drain to its heel, a build to the surveys' drain tangents, then a landing at their landing builds to 75° at the heel's TVD; a tangent drawn above 75° is capped at 75° [own analysis of BOEM's directional surveys].
- Replace SMP-41's straight tangent (no build arc) with a build arc; replace its L-shaped well (a vertical well with a horizontal leg) with the drain to its heel; its kickoff U(0.1, 0.5) and tangent U(10°, 60°) lie inside the public ranges but miss the kickoffs deeper than half the TVD and shallower than a tenth that many real deviated wells have [own analysis of BOEM's directional surveys], and the S profile.
- Couple the drain class to the inflow: a well modelled to its heel stands for a drain of 300 to 700 m (MD beyond 75°, a design range set with [own analysis of BOEM's directional surveys] and [own analysis of NLOG's deviation surveys]), which multiplies the PI (first pass, Section 4; Joshi 1991, not in hand).
- Gas wells: more vertical and slightly deviated, fewer drains (Dependencies above).

**Gaps.**

- Surveys are in hand for two regions only, the GoM and the Netherlands. Norway publishes no inclinations for development wellbores, and the UK's surveys are not open in bulk; their shares here are a rough transfer from MD/TVD. Brazil, West Africa, the Middle East and South-East Asia have none in hand.
- Platform or subsea in the GoM is inferred from the distance to a structure, a proxy; the Dutch set has almost no subsea wells.
- No onshore conventional set of naturally flowing oil wells with surveys: the Dutch onshore oil wells are pumped heavy-oil wells, and the Alaska North Slope surveys (gas-lifted, pad-drilled) are only on request.
- The GoM population is the boreholes producing in 1996–2024, so its old wells are the survivors.
- Counting: boreholes, not wells. First boreholes approximate wells; a well's current producing borehole would be better.
- Inclination through the reservoir is approximated by the bottom 100 m of the survey; reservoir tops are not joined.
- The textbooks on directional well design are not in hand (resource list).

## Resource list

| Reference | The decision it bears on | Assumed without it |
|---|---|---|
| Sodir FactPages bulk export of development wellbores (`wellbore_development_all`, CSV; the command sandbox refused it in the first pass) | Norway by era and field; kickoff points of sidetracks | the aggregate MD/TVD of the first pass and the transfer |
| NSTA offshore wellbores, per-row CSV (the command sandbox refused the hosts) and the National Data Repository's deviation surveys (registration) | UK surveys, by field and era | server-side MD/TVD percentiles and the transfer |
| ANP, Tabela de Poços (Brazil, through the CDP portal; fields DIRECAO, TERRA_MAR, CATEGORIA, depths) | shares for a subsea-dominated region | GoM subsea and Norwegian subsea |
| AOGCC directional surveys (Alaska, on request) | onshore conventional wells, pad-drilled and gas-lifted | Dutch onshore gas wells |
| Geoscience Australia NOPIMS and New Zealand's petroleum data (per-well reports) | further offshore and onshore examples | none |
| Bourgoyne, Millheim, Chenevert and Young (1986), *Applied drilling engineering*, SPE Textbook Series 2, Ch. 8 | profile types and design rules | the measured distributions |
| Mitchell and Miska (2011), *Fundamentals of drilling engineering*, SPE Textbook Series 12 | as above | as above |
| Joshi (1991), *Horizontal well technology* | the PI of a drain modelled to its heel | a multiplier left to R3 |

## Working assumptions proposed

For the decision log; none is entered there by this step.

1. **Geometry classes.** *Proposed:* four classes (vertical; slightly deviated; deviated, J or S; highly deviated bottom up to 75°), each a labelled draw; a horizontal drain enters the fourth class, modelled to its 75° point; a tangent above 75° is capped. *Alternative:* SMP-41's three (vertical, straight tangent, L-shaped). Needs Bjarne's ruling (it sets the shares he asked for).
2. **Shares.** *Proposed:* A′ (Section 6). *Alternatives:* A, B, C. Needs Bjarne's ruling.
3. **Profile parameters.** *Proposed:* the distributions of Leads for R3, from the GoM surveys, with the Dutch ones as a check. *Alternative:* SMP-41's ranges with a build arc added.
4. **Depth reference.** *Proposed:* for subsea wells the trajectory starts at the mudline, and its KOP fraction is drawn below it. *Alternative:* from the rig floor, as the surveys are.
5. **Counting.** *Proposed:* shares by first boreholes of recent wells (spud 2000 on) where the data allow. *Alternative:* all producing boreholes, which raises the highly deviated share [own analysis of BOEM's directional surveys].

## References

- BOEM (2026), Data Center, Well Information: Directional Survey (Points) and Master API List, Borehole (5010); Platform/Rig: Platform Structures and Platform Locations; Production: OGOR-A, 1996–2024. data.boem.gov, files dated 2026-10-25 (surveys), fetched 2026-10-30. Public domain.
- Bourgoyne, A. T., Millheim, K. K., Chenevert, M. E. and Young, F. S. (1986), *Applied drilling engineering*, SPE Textbook Series 2, Society of Petroleum Engineers: Ch. 8. Not in hand.
- Hasan, A. R. and Kabir, C. S. (1988), "Predicting multiphase flow behavior in a deviated well", *SPE Production Engineering* 3(4), 474–482.
- Joshi, S. D. (1991), *Horizontal well technology*, PennWell. Not in hand.
- Mantle, K. (2014), "The defining series: directional drilling practices", *Oilfield Review* 25(4), Schlumberger.
- Mitchell, R. F. and Miska, S. Z. (eds.) (2011), *Fundamentals of drilling engineering*, SPE Textbook Series 12. Not in hand.
- NLOG (2026), thematic borehole data: deviation surveys (`nlog_dirstelsel`, 2026-09-23); WFS layers of boreholes and facilities, gdngeoservices.nl, fetched 2026-10-30. Reuse with NLOG.NL as the source.
- NSTA (2026), "UKCS offshore wellbores top holes (ED50)", open data, feature service queried 2026-10-30. NSTA Open User Licence (NSTA User Agreement, June 2023). Statistics in [own analysis of NSTA's wellbore layer].
- ONRR (2026), *Minerals production reporter handbook*, Appendix H, Well codes, Office of Natural Resources Revenue, onrr.gov, fetched 2026-10-30.
- Shi, H., Holmes, J. A., Durlofsky, L. J., Aziz, K., Diaz, L. R., Alkaya, B. and Oddie, G. (2005), "Drift-flux modeling of two-phase flow in wellbores", *SPE Journal* 10(1), 24–33.
- Sodir, Norwegian Offshore Directorate (2026), FactMaps DataService, layer 5000 (wellbore), aggregate queries of 2026-10-30 (first pass). NLOD.
