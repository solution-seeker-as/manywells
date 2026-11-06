# Sampler redesign — public sources: reservoirs and fluids

*Step R2 of `plans/sampler-redesign-plan.md`, first pass, 2026-10-30. One of two halves: this file covers reservoirs and fluids; wells and production are in a companion file. A draft for Bjarne's checkpoint (R8), not a decision. Plans are drafts, not specs.*

This file collects the public evidence on the reservoir and fluid inputs the v2 sampler will draw: what each source says about a distribution, how the inputs depend on each other, and which population each source describes. It designs no draws; that is R3. Leads for R3 are marked as leads. Every number carries its source and its population. A longer note with the extracted tables, quotes and reasoning is in the local papers library.

## Conventions

- **Units.** Model units first (m, bar, K, Sm³/Sm³, kg/m³), the source's units in brackets. 1 psi/ft = 0.226206 bar/m; 1 scf/STB = 0.178108 Sm³/Sm³; 1 STB/MMscf = 5.615 Sm³ per 10⁶ Sm³; 1 °F/100 ft = 18.23 K/km.
- **Quantiles.** P10/P50/P90 are the 10th, 50th and 90th percentiles of the quantity (not the reserves convention, in which P90 is the low case).
- **Populations.** *Wells* counts wells or wellbores; *reservoirs* counts reservoirs or sands; *fields* counts fields; *volumes* weights by recoverable volume (EUR). The Goals ask for wells, offshore-weighted. A share by volume over-weights large fields and must be reweighted before it becomes a per-well draw.
- **Strength.** *Strong*: a large per-record public database or a standard reference, for the population it covers. *Moderate*: a single study of many records, a textbook rule, or a database outside the target population. *Weak*: a web page, a secondary source, a figure read by eye, or a narrow sample.
- **Own analyses of public databases.** R2's own statistics of two per-record public databases are kept outside the repo, by the maintainer's decision of 2026-11-06, and this file cites them as [own analysis of BOEM's sands atlas] and [own analysis of Sodir's FactMaps]; the file states what they show, not their values:
  - **BOEM sands**: the US Bureau of Ocean Energy Management's *Atlas of Gulf of America Gas and Oil Sands Data*, 2023 update, fetched on 2026-10-30.
  - **Sodir**: the Norwegian Offshore Directorate's FactMaps data service (wellbores, drill-stem tests, fields and field production with produced water), queried on 2026-10-30.

## 1. Depth of producing reservoirs

The model's $L$ is the bottomhole's true vertical depth below the wellhead. On a platform the wellhead is near sea level; on a subsea well it is at the seabed, so a subsea well's $L$ is the depth below sea level less the water depth.

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Top depth of reservoirs | most between 0.5 and 3.5 km, few below 6 km (read from the scatter) | 41,450 reservoirs, global except Canada; 39% North America, 28% former Soviet Union, 10% Europe; onshore-heavy | Ehrenberg and Nadeau (2005), Data section and Fig. 2 | Weak for depth: no histogram, onshore-heavy |
| Depth of reservoirs in large fields | 0 to 7 km below land surface or seafloor, most 1 to 4 km (read from the scatter) | 1065 reservoirs in 954 of the world's largest fields | Nadeau et al. (2023), Fig. 7 | Weak: figure read by eye, large fields only |

Own analyses of BOEM's sands for this section: [own analysis of BOEM's sands atlas]. Own analyses of Sodir's data for this section: [own analysis of Sodir's FactMaps].

**Dependencies.**
- Depth sets pressure and temperature: in GOM oil sands, weighted by wells, depth is strongly rank-correlated with initial pressure and with initial temperature [own analysis of BOEM's sands atlas].
- Fluid class depends on depth: dry gas lies shallowest and gas condensates deepest in the GOM, and the condensate yield of GOM gas sands rises with depth [own analysis of BOEM's sands atlas].
- Deepwater reservoirs are deeper below sea level and more often oil (Section 6).

**Leads for R3.**
- Draw depth after the fluid class, from a class-dependent distribution such as the GOM's, with the Norwegian wellbores as a check.
- For a subsea well, draw the water depth and subtract it from the depth below sea level (the companion file covers water depth and well types).
- A core of 1.5 to 4.5 km below the wellhead, with tails to about 1 km for dry gas and to 6 or 7 km for deepwater and HPHT wells (design values after [own analysis of BOEM's sands atlas] and the table above).

**Gaps.** No per-well depth statistics for the other offshore provinces (Brazil, West Africa, South-East Asia, the Middle East). The Norwegian depths are wellbore TDs, not reservoir depths.

## 2. Initial reservoir pressure

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Hydrostatic gradient, fresh water | 0.0979 bar/m [0.433 psi/ft] | definition | Nadeau et al. (2023), Methods (SG = gradient / 0.433 psi/ft) | Strong |
| Spread of the hydrostatic gradient | brine densities of about 1000 to 1200 kg/m³ (Section 9) give 0.098 to 0.118 bar/m [0.43 to 0.52 psi/ft] | formation waters, 10,000 to 300,000 ppm | derived from Whitson and Brulé (2000), §9.2, and McCain (1990), Fig. 16-8 | Moderate (derived) |
| Gradient classes, global | by reservoirs: SG below 0.8: 66 (6.0%); 0.8 to 1.3: 858 (77.8%); above 1.3: 179 (16.2%), of 1103. By EUR: 4.8%, 81.9%, 13.4% | reservoirs of the world's largest fields, original pressures | Nadeau et al. (2023), Fig. 20 (inset) | Moderate: large fields only |
| Gradient against temperature | most reservoirs below 100 °C near SG 1.0; the P50 of SG rises above 100 °C | as above | Nadeau et al. (2023), Fig. 19 | Moderate |
| Overpressure in hot reservoirs, North Sea | about half of the 25 reservoirs above 122 °C have SG above 1.3; Central Graben chalks 1.58–1.67 | North Sea and Haltenbanken, large fields | Nadeau et al. (2023), §4.2, §5.1 | Moderate |
| Underpressure | SG 0.72 to 0.88 in uplifted basins; also from earlier depletion through a shared aquifer | onshore uplifted basins | Nadeau et al. (2023), §2 and §4.3 | Moderate |

Own analyses of BOEM's sands for this section, including the GOM's normal gradient: [own analysis of BOEM's sands atlas].

SG is the gradient divided by 0.433 psi/ft: SG 1.3 is 0.127 bar/m [0.563 psi/ft], SG 0.8 is 0.078 bar/m.

**Dependencies.**
- Excess pressure grows with depth [own analysis of BOEM's sands atlas] and with temperature above about 100 °C (Nadeau et al. 2023, Fig. 19), which they attribute to quartz cementation above about 120 °C.
- Gas condensates and volatile oils are more often overpressured than black oils and gases in the GOM [own analysis of BOEM's sands atlas]; they also lie deeper (Section 1).
- Deepwater GOM sands have a higher median gradient by wells than shelf sands [own analysis of BOEM's sands atlas]. A normally pressured deepwater sand would lie below the GOM's normal gradient, because its seawater column is lighter than brine, so the deepwater excess is larger than the gradient suggests.

**Leads for R3.**
- Draw $p_i$ as hydrostatic (brine density from Section 9, seawater over a subsea well's mudline) plus an excess that depends on depth below mudline or on temperature, fitted to the GOM bins [own analysis of BOEM's sands atlas] and checked against the global classes (16% of reservoirs above SG 1.3).
- Bjarne's goal 3 (overpressured, high-rate wells) has public support: a sizeable share of GOM wells lies above 0.6 psi/ft [own analysis of BOEM's sands atlas], and 16% of the reservoirs of large fields above SG 1.3 (0.563 psi/ft).
- An initial gradient below hydrostatic is rare in the GOM [own analysis of BOEM's sands atlas] and among large-field reservoirs (6% at SG below 0.8). Bjarne's goal 1 (depleted wells) comes mostly from depletion (Section 3).

**Gaps.** No well-weighted overpressure statistics outside the GOM. The global classes come from a proprietary database (C&C Reservoirs) through one paper.

## 3. Depletion and pressure maintenance

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Example depletion of a subsea oil well | average reservoir pressure 4000 → 2300 psia [276 → 159 bar] over the field's life, solution-gas drive, natural flow then gas lift | a textbook example field | Golan and Whitson (1991), Example 4.11, Table E4.11a, p. 454 | Weak: one hypothetical example |
| Depletion at peak, giant oil fields | 44% of ultimate recovery offshore, 34% on land (production-weighted) | giant fields | UKERC (Miller et al. 2009), Table 5.1, after Höök et al. (2009b) | Moderate: volume-weighted |

Own analyses of BOEM's sands for this section: [own analysis of BOEM's sands atlas]. Own analyses of Sodir's data for this section: [own analysis of Sodir's FactMaps].

**Dependencies.**
- Depletion depends on the drive: a water-driven reservoir keeps its pressure while its water cut rises (Section 9); a depletion-drive gas reservoir loses much of its $p/Z$ [own analysis of BOEM's sands atlas].
- Offshore oil fields are commonly pressure-maintained by water or gas injection; injectors are a sizeable share of the Norwegian shelf's development wellbores [own analysis of Sodir's FactMaps]. A maintained oil reservoir stays near its initial pressure and waters out.
- Below the bubble point, pressure and GOR are linked (Section 11).

**Leads for R3.**
- A life-stage draw could set $p_r/p_i$ by drive: near 1 for maintained or strong-water-drive wells (watered out), and falling to about 0.3 to 0.9 for depletion-drive gas and solution-gas oil (depleted), a design range after the GOM's depletion-drive gas sands [own analysis of BOEM's sands atlas].
- Bjarne's goal 1 ($p_r$ below hydrostatic): a depletion-drive gas sand at the GOM's median implied abandonment $p/Z$ [own analysis of BOEM's sands atlas] has $p_r$ at or below that fraction of its initial pressure ($Z$ falls with $p$ at these pressures), which is below hydrostatic unless the initial gradient exceeded hydrostatic by the inverse of that fraction. The goal has support for depleted gas wells and solution-gas oil wells; a pressure-maintained oil well does not reach it.

**Gaps.**
- No public statistics of current against initial pressure for producing wells.
- No abandonment pressures for oil reservoirs.
- The GOM atlas records natural drive, not injection.
- The share of offshore oil wells under pressure maintenance, counted by wells, is missing.

## 4. Reservoir temperature

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Range of gradients, GOM offshore | 14 to 36 °C/km | GOM Cenozoic reservoirs | Nadeau et al. (2023), Fig. 7A, after Ehrenberg et al. (2008b) | Moderate |
| Temperature classes, by reservoirs | below 58 °C: 259 (22.0%); 58 to 122 °C: 765 (65.1%); above 122 °C: 151 (12.9%), of 1175 | reservoirs of 954 of the world's largest fields, discovery temperatures | Nadeau et al. (2023), Table 1 | Moderate: large fields; counts reservoirs |
| Same, by volume | 20.4%, 74.0%, 5.5% of EUR; median of EUR at 97 °C; 10% of EUR below 35 °C | as above, volume-weighted | Nadeau et al. (2023), Table 1 and §3 | Moderate; volume-weighted |
| Regional counts | North Sea and Haltenbanken: 2/71/25 reservoirs below/within/above 58–122 °C; GOM Cenozoic: 11/53/9 | large fields | Nadeau et al. (2023), Table 1 | Moderate |
| Reservoir temperatures of oils with PVT data | 289 to 445 K [60 to 342 °F], mean 358 K | 1745 PVT records, worldwide | Valkó and McCain (2003), Table 2.1 | Moderate: lab samples |

Own analyses of BOEM's sands for this section: [own analysis of BOEM's sands atlas]. Own analyses of Sodir's data for this section: [own analysis of Sodir's FactMaps].

**Dependencies.**
- Temperature follows depth below the ground or seabed, with the gradient set by the basin: lower in GOM deepwater (below the mudline) than on the GOM shelf, and higher on the Norwegian shelf than in either [own analysis of BOEM's sands atlas and Sodir's FactMaps]. In some South-East Asian basins 120 °C is reached at a little over 1 km (Nadeau et al. 2023, §2).
- Condensates are hotter: 14% of condensate EUR lies above 122 °C against 6% overall (Nadeau et al. 2023, §3), and GOM condensates are hotter than GOM black oils at the median [own analysis of BOEM's sands atlas].
- Heavy oils are colder: 53% of the EUR of oils below 22 °API lies below 58 °C, against 15% for lighter oils (Nadeau et al. 2023, §3).
- Overpressure rises with temperature above about 100 °C (Section 2).

**Leads for R3.**
- $T_r = T_\text{ref} + G\thinspace z$, with $T_\text{ref}$ the mean ground or seabed temperature and $z$ the depth below it, and $G$ drawn by basin type from about 15 to 45 K/km, median near 25 to 35 K/km (design values after [own analysis of BOEM's sands atlas], [own analysis of Sodir's FactMaps] and Nadeau et al.'s range). All of these stay well above the CasADi backend's limit near 5 K/km (`specs/model/validity.md`).
- Nadeau's Table 1 gives reservoir counts as well as volumes. The counts put 12.9% of reservoirs above 122 °C, against 5.5% of the volume, so a per-reservoir draw needs more weight on hot reservoirs than the volume shares suggest.

**Gaps.** No well-weighted temperature statistics outside the GOM. The Norwegian gradients need the usual correction of logging temperatures, which is not applied here.

## 5. Fluid classification

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Black oil | initial producing GOR ≤ 356 Sm³/Sm³ [2000 scf/STB]; stock-tank gravity below 45 °API; $B_o$ ≤ 2.0; C7+ above 30 mol%; very dark stock-tank oil | definitions | McCain (1990), p. 151 | Strong (standard text) |
| Volatile oil | GOR 356 to 588 Sm³/Sm³ [2000 to 3300 scf/STB]; 40 °API or more; $B_o$ above 2.0; C7+ 12.5 to 30 mol%; coloured (brown, orange, green) | definitions | McCain (1990), p. 153 | Strong |
| Retrograde gas (gas condensate) | GOR above about 588 Sm³/Sm³ [3300 scf/STB], up to and above 26,700 [150,000]; 40 to 60 °API; C7+ below 12.5 mol%; above 8900 Sm³/Sm³ [50,000 scf/STB] treat as wet gas | definitions | McCain (1990), pp. 155–156 | Strong |
| Wet gas, dry gas | wet: no liquid in the reservoir, liquid at the separator, constant GOR, water-white stock-tank liquid of the condensates' gravity; dry: no hydrocarbon liquid at the surface | definitions | McCain (1990), pp. 156–158 | Strong |
| Gas condensate (second source) | GOR 534 to 26,700 Sm³/Sm³ [3000 to 150,000 scf/STB], OGR about 350 to 5 STB/MMscf, 40 to 60 °API; colour not reliable | definitions | Whitson and Brulé (2000), §2.5, p. 9 | Strong |
| Volatile oil (second source) | above 35 °API, GOR 178 to 534 Sm³/Sm³ [1000 to 3000 scf/STB], $B_o$ above about 1.5 | definitions | Whitson and Brulé (2000), §2.5, p. 11 | Strong |
| Near-critical boundary by C7+ | gas condensates below about 12.5 mol% C7+, volatile oils 12.5 to 17.5 mol% | near-critical fluids | Whitson and Brulé (2000), §2.5, p. 9 | Strong |
| Black and volatile oil (third source) | black oil 20 to low 30s °API, $R_{si}$ 10 to 90 m³/m³, $B_o$ 1.1 to 1.5; volatile oil low 30s to low 50s °API, $R_{si}$ 90 to 1000 m³/m³, $B_o$ 1.5 to 3.5; heavy oil 10 to 20 °API, $R_{si}$ below 10 m³/m³ | definitions | Tewari et al. (2019), §1.2.4, pp. 12–13 | Moderate (secondary text) |
| Classification by critical temperature | natural gas, gas condensate, near-critical or volatile oil, black oil, heavy oil, by the mixture's critical temperature against the reservoir temperature; no GOR limits | definitions | Pedersen et al. (2015), §1.4 | Strong |

**Dependencies.** The classes overlap and the boundaries differ by source: McCain's black-oil limit is 356 Sm³/Sm³, Whitson and Brulé's volatile oils start at 178, and Tewari et al.'s at 90 m³/m³. With Tewari et al.'s limit, most GOM oil sands would be volatile; with McCain's, few GOM wells are [own analysis of BOEM's sands atlas]. McCain (1990) warns that a fluid outside his rules must be classified in the laboratory (p. 149).

**Leads for R3.**
- McCain's (1990) GOR limits fit the model best: the black oil (PVT-OIL) handles McCain's black oils, and `specs/model/validity.md` already quotes Whitson and Brulé's condensate range. Volatile oils and condensates go to PVT tables; wet and dry gases can stay black-oil gases.
- Label each well with the class and the rule that set it, so that a user can reclassify.

**Gaps.** McCain's later limits (1750 and 3200 scf/STB, 20 mol% C7+) are known only from secondary copies; the 1994 article is on the resource list.

## 6. Fluid-type mix

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Fluid types of large-field reservoirs | oil 921, gas 344, condensate 131 entries (66%, 25%, 9%; 221 reservoirs hold more than one); by EUR oil 60%, gas 37%, condensate 2.6% | reservoirs of the world's largest fields | Nadeau et al. (2023), §3 and Table 1 | Moderate: large fields |

Own analyses of BOEM's sands for this section: [own analysis of BOEM's sands atlas]. Own analyses of Sodir's data for this section: [own analysis of Sodir's FactMaps].

**Dependencies.** The mix depends on the province (GOM shelf is gas-prone, its deepwater oil-prone, the Norwegian shelf oil-prone) and on the weighting: by sands wet gas is the GOM's largest class, by wells black oil [own analysis of BOEM's sands atlas]. Volatile oils are rare under McCain's limits [own analysis of BOEM's sands atlas].

**Leads for R3.**
- By wells, offshore, the GOM's and the Norwegian shelf's counts give black oil the largest share, volatile oil and gas condensate small ones, and wet or dry gas a share that varies with the province [own analysis of BOEM's sands atlas and Sodir's FactMaps]. With the Goals' floor of about 10% per type, the floor sets the volatile-oil share and probably the condensate share.
- A provisional mix for the decision log: black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20.

**Gaps.** No global count of offshore producing wells by fluid type; commercial databases hold it. Wet and dry gas are not told apart on the Norwegian shelf or in Nadeau et al.

## 7. Joint fluid properties

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Ranges of oil PVT data | solution GOR at $p_b$ 1.8/105/395 Sm³/Sm³ (min/mean/max) [10/588/2216 scf/STB]; $p_b$ 5.7/151/462 bar [82/2193/6700 psia]; $T$ 289/358/445 K; 6.0/35.7/63.7 °API; separator-gas gravity 0.555/0.838/1.685 | 1745 PVT records, worldwide, heavy to volatile oils | Valkó and McCain (2003), Table 2.1 | Moderate: ranges and means only; lab samples |
| Regions in that data set | Middle East 157, North Sea 17, Egypt 125, Malaysia 93, USA 53 records named; the rest unattributed | as above | Valkó and McCain (2003), Table 2.3 | Moderate |
| Example fluids' surface properties | wet gas: GOR 105,000 scf/STB, OGR 10 STB/MMscf, 57 °API, $\gamma_g$ 0.61; gas condensate: 5450, 180, 49, 0.70; volatile oil: 1490 scf/STB, 38 °API, $\gamma_g$ 0.70; black oil: 300 scf/STB, 24 °API, $\gamma_g$ 0.63 | one typical fluid per class | Whitson and Brulé (2000), Table 2.1, p. 2 | Strong as examples; not a distribution |

Own analyses of BOEM's sands for this section: [own analysis of BOEM's sands atlas]. Own analyses of Sodir's data for this section: [own analysis of Sodir's FactMaps].

**Dependencies.**
- $R_{si}$ rises with depth, pressure and temperature; API gravity is nearly independent of depth in the GOM, and weakly tied to $R_{si}$ and $B_o$ [own analysis of BOEM's sands atlas].
- Condensate yield rises with depth and temperature, so condensates are richer in deeper, hotter, more overpressured reservoirs [own analysis of BOEM's sands atlas].
- Oils are mostly undersaturated at discovery, but with a small margin in the GOM [own analysis of BOEM's sands atlas], of the size of the error of the Standing correlation behind that estimate (12.7% average, Valkó and McCain 2003, Table 2.2).

**Leads for R3.**
- For black oils: draw API gravity, then $R_{si}$ given depth (or $p_i$ and $T_i$), then the gas gravity, then $p_b$ from Standing (PVT-OIL-14), and check $p_b \le p_i$ for an undersaturated initial stage. A Gaussian copula with the GOM rank correlations [own analysis of BOEM's sands atlas] is one way.
- For gas condensates and wet gases: draw the condensate–gas ratio given depth or temperature (GOM, [own analysis of BOEM's sands atlas]), with the gas gravity rising with it.
- `validity.md`'s API range (16 to 58) covers nearly all GOM oil sands [own analysis of BOEM's sands atlas].

**Gaps.** No joint per-sample public data with $p_b$ measured. The Norwegian DST records, downloaded in bulk, would give a joint GOR–density–gas-gravity–temperature table per test (resource list).

## 8. Compositions by fluid class

### 8a. Published example fluids

| Fluid | Class | Key properties | Source | Strength |
|---|---|---|---|---|
| Six fluids: dry gas, wet gas, gas condensate, near-critical oil, volatile oil, black oil | one per class | mol% of CO2, N2, C1 to C6 and C7+; $M_{C7+}$ 130 (wet gas) to 274 (black oil), $\gamma_{C7+}$ 0.763 to 0.920; GOR, OGR, API, $\gamma_g$, $p_\text{sat}$, $B_\text{sat}$, density at saturation | Whitson and Brulé (2000), Table 2.1, p. 2 | Strong; C7+ lumped |
| Natural gas | dry gas | mol% N2 to C6, no C7+ | Pedersen et al. (2015), Table 1.2 | Strong |
| Gas condensate | condensate | N2 to C19, C20+ ($M$ = 391), with $M$ and density per carbon number | Pedersen et al. (2015), Table 1.3 | Strong |
| Near-critical mixture | near-critical | N2 to C19, C20+ ($M$ = 406) | Pedersen et al. (2015), Table 1.4 | Strong |
| Black oil | black oil | N2 to C9, C10+ ($M$ = 322, 0.936 g/cm³) | Pedersen et al. (2015), Table 1.5 | Moderate: unusual (39 mol% C1, 55 mol% C10+) |
| Near-critical reservoir fluid with measured phase envelope | near-critical | C7+ 9.87 mol%, $M$ = 192.8; critical point 115 °C and 462 bar | Pedersen et al. (2015), Tables 1.6, 1.7 (after Yang et al. 1997) | Strong |
| Oil with CME and differential liberation | black or volatile (borderline) | N2 to C20+; $p_\text{sat}$ 200 bar at 97.5 °C; $R_s$ 198 Sm³/Sm³; $B_o$ 1.71; stock-tank oil 0.838 g/cm³ (37 °API) | Pedersen et al. (2015), Tables 3.5, 3.6, 3.13 | Strong |
| Gas condensate with CME | condensate | N2 to C20+; dew point 388 bar at 155 °C; maximum liquid dropout 11.9% | Pedersen et al. (2015), Tables 3.8, 3.9 | Strong |
| Gas condensate with constant-volume depletion | condensate | N2 to C20+, mol% and wt%; CVD at 150.5 °C | Pedersen et al. (2015), Tables 3.15 to 3.17 | Strong |
| Oil with four-stage separator test | black oil | N2 to C20+; $p_\text{sat}$ 199.7 bar at 97.8 °C; stage GORs 109, 34, 17 and 12 Sm³/Sm³; $B_o$ 1.605; separator-gas compositions per stage | Pedersen et al. (2015), Tables 3.19 to 3.21 | Strong |

### 8b. Ranges of composition per class

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| C7+ by class | black oil above 30 mol%; volatile oil 12.5 to 30; retrograde gas below 12.5, with negligible retrograde liquid below 1 mol% | definitions | McCain (1990), pp. 151, 153, 155 | Strong for the boundaries; no distribution |
| C7+ near the critical point | condensates below about 12.5 mol%, volatile oils 12.5 to 17.5 | near-critical fluids | Whitson and Brulé (2000), §2.5, p. 9 | Strong |
| C1 by class (examples) | dry gas 86 to 90 mol%, wet gas 92, condensate 70 to 74, near-critical 62 to 70, volatile oil 59, black oil 35 to 40 | the example fluids of 8a | Whitson and Brulé (2000), Table 2.1; Pedersen et al. (2015), Tables 1.2–1.6, 3.6, 3.8, 3.15, 3.20 | Weak as a range: about a dozen fluids |
| $M_{C7+}$ and $\gamma_{C7+}$ (examples) | wet gas 130 and 0.763; condensate 184 and 0.816; near-critical 219 and 0.839; volatile oil 228 and 0.858; black oil 274 and 0.920 | one fluid per class | Whitson and Brulé (2000), Table 2.1 | Weak as a range |
| C7+ lumped from the carbon-number tables (examples) | gas condensates 5.7, 6.3 and 7.2 mol%, $M_{C7+}$ 151 to 167, density 0.809 to 0.816 g/cm³; near-critical 14.0 mol%, 198, 0.836; oils 31.6 and 32.0 mol%, 211 and 213, 0.843 and 0.846; the black oil of Table 1.5 57.6 mol%, 312, 0.933 | Pedersen et al.'s example fluids | Pedersen et al. (2015), Tables 1.3–1.5, 3.6, 3.8, 3.15, 3.20; own lumping (mole-weighted $M$, volume-additive density) | Weak as a range |
| Shape of the C7+ distribution | a gamma distribution with shape $\alpha$ usually 0.5 to 2.5 for reservoir fluids | reservoir fluids | Whitson and Brulé (2000), §5, Eq. 5.14 | Strong (method) |

**Dependencies.** C1 falls and C7+ rises from gas to black oil; $M_{C7+}$ and $\gamma_{C7+}$ rise with them. GOR, API gravity and saturation pressure follow from the composition and the temperature through the flash.

**Leads for R3 and R5.**
- Draw a synthetic composition per class by perturbing the published examples: C7+ within McCain's class limits, C1 to C6 scaled to close the sum, $M_{C7+}$ and $\gamma_{C7+}$ within the examples' span, N2 and CO2 a few mol%. Then flash it (NeqSim) and accept it only if its GOR and API gravity fall in the class's range (Section 5).
- The published examples themselves (8a) can serve as a fixed set of table fluids.

**Gaps.** No published statistics (means and spreads) of compositions per class are in hand. Databases of gas-condensate compositions behind the dew-point correlations exist but were not reached (resource list).

## 9. Formation water

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Salinity range | 200 ppm to saturation (about 300,000 ppm) total dissolved solids, mostly NaCl; seawater about 35,000 ppm | oilfield waters | McCain (1990), p. 438 | Strong for the range; no distribution |
| Salinity range (second source) | 10,000 to about 300,000 ppm; seawater about 30,000 ppm; usually below 80% of the NaCl limit, $C_{sw}^\ast = 262,180 + 72T + 1.06T^2$ (ppm, $T$ in °C) | formation waters | Whitson and Brulé (2000), §9.2, Eq. 9.1 | Strong |
| Density at standard conditions against salinity | from 999 kg/m³ at 0 to about 1200 kg/m³ at 26 wt% solids [62.4 to about 75 lb/ft³]; at 6.8 wt%, 1047.6 kg/m³ [65.40 lb/ft³] | brines | McCain (1990), Fig. 16-8 and Example 16-6, p. 449 | Strong |
| Density, reservoir conditions | standard density divided by $B_w$; $B_w$ rarely above 1.06 | brines | McCain (1990), pp. 445, 449 | Strong |
| Water–oil ratio, global | 2 to 3 (US DOE, quoted), 3, 4 to 5 barrels of water per barrel of oil (various); North Sea 2.3 in 2014 (OSPAR, quoted) | world, volume-weighted | web pages and trade sources (longer note) | Weak: secondary, volume-weighted |

Own analyses of Sodir's data for this section (water cuts and water–gas ratios of Norwegian fields): [own analysis of Sodir's FactMaps].

**Dependencies.**
- Brine density follows salinity; the hydrostatic gradient follows brine density (Section 2).
- Water cut rises with time on production [own analysis of Sodir's FactMaps] and with maturity; strong water drive (Section 3) gives the high water cuts, depletion drive low ones.
- The limiting salinity rises with temperature (Eq. 9.1).

**Leads for R3.**
- Draw salinity (for example 20,000 to 250,000 ppm, with seawater-like values common offshore), then $\rho_w$ from McCain's Fig. 16-8 (a quadratic in wt% that reproduces Example 16-6 to 0.02 lb/ft³), and use the same brine for the hydrostatic gradient.
- Draw the water cut by life stage: initial about 0 to 0.1; mid-life 0.1 to 0.7; watered out 0.7 to 0.95, matching the Norwegian fields' spread by years on production [own analysis of Sodir's FactMaps].

**Gaps.**
- No public salinity distribution for offshore reservoirs was reached; the USGS produced-water database is mostly onshore and was blocked by the sandbox.
- Per-well water cuts are missing (the Norwegian data are per field).
- The shape of the water-cut rise after breakthrough (fractional flow) needs a reservoir-engineering text (resource list).

## 10. Heat capacities

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Liquid oil | $c = (0.388 + 0.00045\thinspace t)/\sqrt{d}$ Btu/(lb °F), $t$ in °F, $d$ the specific gravity at 60 °F; from over 100 measurements on 30 oils with $d$ 0.75 to 0.96 at 32 to 400 °F, average deviation below 2%, at most about 4% | petroleum oils at 1 atm | Cragoe (1929), Eq. 5, p. 26 | Strong (still the standard), dead oil |
| Liquid oil, values | 35 °API: 1901 J/(kg K) at 20 °C, 2195 at 100 °C, 2379 at 150 °C; 20 °API: 1813, 2094, 2270; 45 °API: 1957, 2260, 2450 | from Cragoe's Eq. 5 | own evaluation | Strong |
| Live oil | about 1900 J/(kg K) at 0 °C to 2650 at 200 °C, little dependence on pressure from 200 to 800 bar | the oil of Pedersen et al. Table 3.6 (37 °API, $R_s$ 198 Sm³/Sm³), Peng–Robinson | Pedersen et al. (2015), §8.5, Fig. 8.6 | Moderate: one fluid, calculated |
| Methane (lean gas) | 2236 J/(kg K) at 300 K and 1 bar; 3011 at 101 bar; 3549 at 201 bar; 3336 at 401 bar. At 380 K: 2466, 2801, 3079, 3234 | methane, reference equation of state | NIST Chemistry WebBook, fluid properties of methane | Strong for methane |
| Brine | 4190 J/(kg K) at 0 g/kg and 20 °C; 3999 at 35 g/kg; 3699 at 100 g/kg; 3517 at 150 g/kg; rising by 1 to 2% from 20 to 120 °C | seawater and NaCl-dominated brines, 0 to 180 °C, 0 to 180 g/kg, ±0.28% | Sharqawy et al. (2010), Eq. 9 (after Jamieson et al. 1969) | Strong for seawater; an approximation for brines |

Today's fixed values (SMP-9 to SMP-11) are 2000, 4184 and 2225 J/(kg K) for oil, water and gas.

**Dependencies.** The oil's $c_p$ rises with temperature and with API gravity. The gas's $c_p$ rises with pressure, strongly near ambient temperature (by 60% from 1 to 200 bar at 300 K), and falls with molar mass per unit mass. The brine's $c_p$ falls with salinity, by 12% at 100 g/kg.

**Leads for R3.** The model's $c_p$ is constant along the well (no $c_p(p, T)$ until v2.x). Evaluate each correlation once per well at a representative temperature, such as the mean of $T_r$ and the wellhead's, with the well's API gravity and salinity. For the gas, take the methane values at the well's mean pressure and temperature, scaled for gas gravity, or the flash's value for a table fluid.

**Gaps.** No open correlation in hand for the $c_p$ of a natural gas against gas gravity at pressure; heavier gases have a lower $c_p$ per unit mass.

## 11. A well's life

| Quantity | Value or distribution | Population | Source | Strength |
|---|---|---|---|---|
| Field decline after peak, offshore against onshore | production-weighted: offshore 7.3%/year (IEA) or 9.7%/year (Höök et al.); onshore 4.3% or 3.9% | large post-peak oil fields, worldwide | UKERC (Miller et al. 2009), Table 4.2 | Moderate: fields, production-weighted, secondary |
| Decline by field size, offshore | supergiant 3.4%/year, giant 8.6%, other (large) 11.6% | as above, IEA sample | UKERC (2009), Table 4.3 | Moderate |
| Decline by phase | 1.4%/year from peak to end of plateau, 3.6% to 50% of peak, 6.7% after | as above | UKERC (2009), Table 4.4 | Moderate |
| Regional offshore declines | Norway about 10%/year (giant fields 13.8%, small fields 18%), Australia 15%, Brazil 5–12%, GOM 20–23% | regional field data | UKERC (2009), §3, Fig. 3.5 | Moderate |
| Depletion rate at peak | offshore giant fields 11.0% of remaining reserves per year, land 5.8% | giant fields | UKERC (2009), Table 5.1, after Höök et al. (2009b) | Moderate |
| Build-up and plateau | build-up about 6 years, plateau about 7 years at 93% of peak, decline more than 20 years | large fields | CERA (2008), quoted in web reports | Weak: secondary, not in hand |
| Arps exponent | exponential decline ($b = 0$) the most common; $b$ from 0 to 0.5; about 0.3 for solution-gas drive, 0.5 for water drive or gravity drainage | wells and reservoirs | Golan and Whitson (1991), p. 385 | Moderate: textbook summary |
| GOR below the bubble point | producing GOR rises once $p_r$ falls below $p_b$, for black and volatile oils | definitions | McCain (1990), pp. 151, 153 | Strong (qualitative) |
| Example of GOR against depletion | $p_R$ 276 → 159 bar as cumulative oil goes from 0 to 79,500 Sm³ [500,000 STB]; producing GOR 107, 178, 285, 445, 890, then 445 Sm³/Sm³ [600 to 5000 to 2500 scf/STB] | one subsea oil well, solution-gas drive | Golan and Whitson (1991), Example 4.11, Table E4.11a, p. 454 | Weak: one textbook example |
| Water cut against time | Section 9 and [own analysis of Sodir's FactMaps] (Norwegian fields by years on production) | | | |

Own analyses of Sodir's data for this section (years with production of shut-down Norwegian fields): [own analysis of Sodir's FactMaps].

**Dependencies.**
- Rate decline follows the drive: exponential for depletion, hyperbolic with $b$ near 0.5 for water drive (Golan and Whitson).
- GOR rises below the bubble point to several times $R_{si}$ and falls late in life as the gas is depleted (Example 4.11: a peak 8.3 times the initial GOR).
- Offshore fields peak harder and decline faster than onshore ones, and smaller fields faster than large ones (UKERC Tables 4.2, 4.3).

**Leads for R3 and R4 (nsol-2).**
- Over a well's life of about 10 to 30 years: $p_r$ falls exponentially or hyperbolically for depletion and solution-gas drive, and stays near $p_i$ under water drive or injection.
- Water cut rises from breakthrough towards 0.8 to 0.95 under water drive.
- GOR stays at $R_{si}$ above $p_b$, rises below it to a peak of several times $R_{si}$, then falls.
- The three trends are linked by the drive type, drawn per well.

**Gaps.**
- Well-level (not field-level) decline statistics offshore.
- Plateau lengths per well.
- A sourced shape for the GOR against pressure below $p_b$, beyond one example (a material-balance text).

## 12. Public per-reservoir and per-well databases

| Database | URL | Licence | Variables | Records | Use here |
|---|---|---|---|---|---|
| BOEM, Atlas of Gulf of America Gas and Oil Sands Data (2023 update) | `https://www.data.boem.gov/Main/GandG.aspx` | US federal data; disclaimer only, no restriction stated | depth (TVDSS), water depth, initial pressure and temperature, gradients, API, solution GOR, yield, gas gravity, $B_o$, $B_g$, drive, reservoir type, porosity, $S_w$, permeability, volumes, recovery factors, wells completed | sands; counts in [own analysis of BOEM's sands atlas] | Downloaded; analysed outside the repo, cited in Sections 1 to 7 |
| Sodir FactPages and FactMaps (Norway) | `https://factpages.sodir.no`, `https://factmaps.sodir.no` | NLOD | wellbores (purpose, content, TVD, MD, inclination, bottom-hole temperature, water depth, subsea), drill-stem tests (depth, choke, pressures, temperature, oil and gas rates, oil density, gas gravity, GOR), fields (type), annual and monthly production with water, reserves | DSTs, wellbores and fields; counts in [own analysis of Sodir's FactMaps] | Server-side statistics only, analysed outside the repo; the bulk CSV download was blocked by the sandbox |
| USGS National Produced Waters Geochemical Database, v3.0 (2023) | `https://doi.org/10.5066/P9DSRCZJ` | US federal, public domain | water chemistry, TDS, location, formation | many samples, mostly onshore US (count not checked) | Not reached (sandbox) |
| Brazil ANP per-well production | ANP open-data portal | Brazilian federal open data: free to use with credit to the source (Decreto nº 8.777/2016) | monthly oil, gas and water per well, onshore and offshore | 2005–2023 | Analysed outside the repo (2026-11-05) |
| BSEE production by completion (GOM) | `https://www.data.bsee.gov` | US federal | monthly oil, gas, water per completion | GOM history | Not checked; a lead for well-level decline and water cut |
| UK NSTA open data | NSTA open-data portal | NSTA Open User Licence (NSTA User Agreement, June 2023): non-commercial use, with NSTA's attribution statement | production by well of fields that have ceased production | UK fields | Analysed outside the repo (2026-11-05) |
| Shale fluids of Argentina (Cismondi et al. 2018) | figshare, DOI 10.6084/m9.figshare.7185911 | CC BY-NC 4.0 | compositions and PVT of two condensates and a volatile oil | 3 fluids | Not checked; candidate table fluids |

**Leads for R3.** BOEM gives what the GOM can give; a second region at the same detail needs Sodir's bulk tables (DST and wellbores), which a run outside the sandbox can fetch in minutes.

## Resource list

Sources that matter and could not be reached:

| Reference | Decision it bears on | Assumption without it |
|---|---|---|
| McCain (1994), "Heavy components control reservoir fluid behavior", *Journal of Petroleum Technology* 46(9), 746–750 | The fluid-class limits (Section 5) | McCain (1990): 2000 and 3300 scf/STB, 30 and 12.5 mol% C7+ |
| McCain (1991), "Reservoir-fluid property correlations — state of the art", *SPE Reservoir Engineering* 6(2), 266–272 | The brine density from salinity (Section 9) | A quadratic fit to McCain (1990), Fig. 16-8, checked against Example 16-6 |
| Spivey, McCain and North (2004), brine property correlations, *Journal of Canadian Petroleum Technology* 43(7) | Brine density and $c_p$ at reservoir conditions | McCain (1990) with $B_w$, Sharqawy et al. (2010) for $c_p$ |
| Nemeth and Kennedy (1967), dew-point correlation, *SPE Journal* 7(2), 99–104; Elsharkawy (2002), *Fluid Phase Equilibria* 193, 147–165 | Ranges of gas-condensate compositions for synthetic fluids (Section 8b) | The span of the example fluids in Whitson and Brulé and Pedersen et al. |
| Höök, Hirsch and Aleklett (2009), *Energy Policy* 37, 2262–2272; Höök et al. (2014), *Phil. Trans. R. Soc. A* 372, 20120448 (open access, refused to the fetch tool) | Decline rates for nsol-2 (Section 11) | UKERC (2009)'s summary of both |
| IEA (2008), *World Energy Outlook 2008*, Ch. 10–11 | Decline rates and plateau lengths | UKERC (2009)'s summary |
| Ehrenberg, Nadeau and Steen (2008b), on GOM temperature and pressure gradients, *AAPG Bulletin* | Gradients of 11,833 GOM reservoirs (Sections 2, 4) | BOEM sands, which cover the same region |
| Sodir bulk CSV tables (DST, wellbores, production), and the USGS produced-water database | Joint fluid properties per test; salinity distribution | Server-side percentiles; salinity range from textbooks |
| A reservoir-engineering text on fractional flow and solution-gas drive (for example Dake (1978), *Fundamentals of reservoir engineering*, Elsevier) | The shapes of water-cut and GOR trends in nsol-2 (Section 11) | Golan and Whitson's Example 4.11 and the Norwegian field water cuts |
| A count of offshore producing wells by fluid type, global (commercial databases) | The fluid-type mix (Section 6) | GOM and Norwegian counts with the 10% floor |
| Swarbrick and Osborne (1998), or another global review of overpressure | The overpressure share outside the GOM | Nadeau et al. (2023), Fig. 20 |

## Working assumptions proposed

For the decision log (R3 records them there; none is a decision):

1. **Wells as the unit.** Use BOEM's `BHCOMP` (cumulative wells completed in a sand) as the GOM's per-well weight, and Norwegian development wellbores as the Norwegian one, sidetracks included. *Alternatives:* weight by sands (more weight on small gas sands, which raises the wet-gas share), or by volume (oil and gas about even) [own analysis of BOEM's sands atlas].
2. **Reweighting Nadeau et al. (2023).** Use the reservoir counts of their Table 1 and Fig. 20, not the EUR shares, as the closest to a per-well count. *Alternatives:* EUR shares (over-weight the largest fields), or counts times a well-count proxy such as EUR to the power 0.5 (large fields have more wells). The counts put 12.9% of reservoirs above 122 °C against 5.5% of the EUR, and 16.2% above SG 1.3 against 13.4%.
3. **Fluid classes by McCain (1990).** GOR limits 356, 588 and 8900 Sm³/Sm³ [2000, 3300 and 50,000 scf/STB], with API and C7+ as checks. *Alternatives:* Whitson and Brulé (volatile from 178 Sm³/Sm³), Tewari et al. (from 90 m³/m³), or McCain (1994) (311 and 570 Sm³/Sm³).
4. **Fluid mix.** Black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20. The GOM and Norwegian well counts give the black-oil and gas shares, and the floor gives the volatile-oil and condensate shares. *Alternatives:* the GOM well shares with the floor (black oil 0.45, wet or dry gas 0.30), or the Norwegian shares (black oil 0.70, gas 0.10), as design values after [own analysis of BOEM's sands atlas] and [own analysis of Sodir's FactMaps].
5. **Overpressure.** Excess pressure over hydrostatic grows with depth below the mudline, fitted to the GOM bins [own analysis of BOEM's sands atlas], and is checked against the global share of 16% of reservoirs above SG 1.3. *Alternative:* a depth-independent share.
6. **Geothermal gradient.** A mixture over basin types, with a mode at each basin's median gradient: the GOM shelf, the GOM's deep water (below the mudline) and the Norwegian shelf [own analysis of BOEM's sands atlas and Sodir's FactMaps], with a spread of ±15 to 20%. *Alternative:* a single global lognormal with median about 30 K/km.
7. **Undersaturation at the initial stage.** Most initial-stage oil wells undersaturated, with $p_i - p_b$ around the median of the GOM's oil sands and a long tail [own analysis of BOEM's sands atlas]. *Alternative:* all undersaturated by a fixed fraction of $p_i$. The GOM estimate rests on Standing's correlation, whose error is of the same size.
8. **Gas depletion.** At the depleted stage, $p/Z$ falls over the range of initial that the GOM's depletion-drive gas sands reach, from their recovery factors, centred on their median [own analysis of BOEM's sands atlas].
9. **Brine.** Salinity drawn from 20,000 to 250,000 ppm; $\rho_w$ from McCain (1990); the same brine sets the hydrostatic gradient. *Alternative:* seawater salinity (35,000 ppm) for every offshore well.
10. **Heat capacities.** Oil from Cragoe (1929) at the well's API gravity and mean temperature; brine from Sharqawy et al. (2010) at the well's salinity; gas from NIST methane values at the well's mean pressure and temperature, until a gas-gravity correlation is in hand. *Alternative:* keep today's constants (SMP-9 to SMP-11).

## References

- Cragoe, C.S. (1929), *Thermal properties of petroleum products*, Miscellaneous Publication 97, US Bureau of Standards: Eq. 5, pp. 26–28.
- Ehrenberg, S.N. and Nadeau, P.H. (2005), "Sandstone vs. carbonate petroleum reservoirs: a global perspective on porosity-depth and porosity-permeability relationships", *AAPG Bulletin* 89(4), 435–445.
- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall: Ch. 4, p. 385 and Example 4.11, pp. 454–456.
- McCain, W.D. (1990), *The properties of petroleum fluids*, 2nd ed., PennWell: Ch. 5, pp. 149–158; Ch. 16, pp. 438–449.
- Miller, R., Sorrell, S. and Speirs, J. (2009), *UKERC review of evidence for global oil depletion, Technical report 4: Decline rates and depletion rates*, UK Energy Research Centre, UKERC/WP/TPA/2009/019: Tables 4.2–4.4, 5.1, §3.
- Nadeau, P.H., Sun, S. and Ehrenberg, S.N. (2023), "The 'Golden Zone' temperature distribution of oil and gas occurrence examined using a global empirical database", *Marine and Petroleum Geology* 158, 106507 (CC BY 4.0).
- NIST Chemistry WebBook, SRD 69, thermophysical properties of fluid systems: methane, isotherms at 300 and 380 K (accessed 2026-10-30).
- Pedersen, K.S., Christensen, P.L. and Shaikh, J.A. (2015), *Phase behavior of petroleum reservoir fluids*, 2nd ed., CRC Press: §1.4, Tables 1.2–1.7; Ch. 3, Tables 3.5–3.21; §8.5, Fig. 8.6.
- Sharqawy, M.H., Lienhard, J.H. and Zubair, S.M. (2010), "The thermophysical properties of seawater: a review of existing correlations and data", *Desalination and Water Treatment* 16, 354–380: Table 3, Eq. 9.
- Standing, M.B. (1947), "A pressure-volume-temperature correlation for mixtures of California oils and gases", *Drilling and Production Practice*, API, 275–287 (as PVT-OIL-14).
- Tewari, R.D., Dandekar, A.Y. and Moreno Ortiz, J. (2019), *Petroleum fluid phase behavior: characterization, processes, and applications*, CRC Press: §1.2, pp. 4–13.
- Valkó, P.P. and McCain, W.D. (2003), "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities", *Journal of Petroleum Science and Engineering* 37, 153–169: Tables 2.1–2.3.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20: Table 2.1, §2.5, §5 (Eq. 5.14), §9.2 (Eq. 9.1).
- US Bureau of Ocean Energy Management (2024), *Atlas of Gulf of America Gas and Oil Sands Data*, 2023 update, sands as of 1 January 2024, `https://www.data.boem.gov/Main/GandG.aspx` (fetched 2026-10-30).
- Norwegian Offshore Directorate (Sodir), FactMaps data service, layers 5000, 7100 and tables 5008, 7300, `https://factmaps.sodir.no/api/rest/services/DataService/Data/MapServer` (queried 2026-10-30), NLOD.
