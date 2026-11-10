# Sampler redesign — the guesses, triaged

*2026-11-10. A triage of what the basis pass of the v2 sampling spec left as a guess: the 38 rows of the Guesses table of `specs/sampling.md` (v2: guesses and design choices), the 7 parts whose basis the plans do not record, and SMP-56's pressure-support share for gas wells, whose cited evidence the basis pass found to point the other way. For each item: whether data in hand, a published source or physics can replace the guess, and what to do. It follows Bjarne's rule of 2026-11-10: the sampling design rests on physics and data, and a guess stands in only where both are missing. Plans are drafts, not specs. This file changes no spec and no code, and nothing proposed here is decided until Bjarne rules.*

**Ruled:** nothing yet. Every proposal below needs Bjarne's ruling; a change to a draw is a change to `specs/sampling.md`.

**Taken as working assumptions in R10** (2026-11-10, each an entry of `decisions.md` and a commit of its own, so that a ruling can reverse it alone): G2 (D-110), G3 (D-111), G8 (D-112), G14 and G15 (D-113), G23 (D-114), G29 (D-115), G32 with its upper end kept at 0.9 (D-116), G33 (D-117), G36 (D-118), N5 (D-119), G37 if the build's match holds (D-120), and N6 as a design choice (D-121). **Not taken, for Bjarne's ruling:**

- **G30**, the later breakthrough. The same data show that most wells of the mature regions start wet, which the draws do not have, and the water cut by years on production was checked against the regions as the draws are (C1, D-43). A later breakthrough alone moves the mid-life water cut away from two of those regions; with wet starts it is a new rule. The two belong in one ruling.
- **G7**, the gas wells' separator pressure. Its evidence is the tubing pressure upstream of a test choke of 64/64 in or more, read as the line pressure plus a tenth. A gas well at a high rate can be in critical flow through such a choke, where the tubing pressure says nothing of the line's, so the bound is loose for the strong wells that carry the upper half of the distribution. The proposal adds a branch to the rule; it needs a tighter proxy first, such as the tests' line pressures by fluid or the wellhead pressures of wells on large chokes at low rates.

**Citations.** SMP-n is a rule of `specs/sampling.md`; D-n an entry of `decisions.md`; [RF §n] and [WP §n] are `sources-reservoir-fluids.md` and `sources-wells-production.md`. [own analysis of …] is the maintainer's analysis of a publisher's open data, kept outside the repo (the maintainer's decision, 2026-11-06): this file states what it shows in words or as a comparison with the draft, never its values. The analyses made for this triage, with their scripts and outputs, are kept outside the repo with the earlier ones. A number from a paper or a book is quoted with its page, table or figure. A number from the model or from a generated pilot is ManyWells' own and is given.

**Classes.** Each item has one:

- *data in hand*: a statistic of the open per-well data bears directly on the value;
- *published source*: a number in a paper or book that the plans already cite or hold;
- *physics*: a derivation, a correlation, or a result of the model or of an evidence script;
- *needs a fetch*: open data that exists and is not in hand;
- *stays a guess*: nothing found.

**Coverage of the evidence.** Every sample of real wells is regional, so its range is a lower bound on the support and its shape weak evidence of what is typical (the Goals, coverage first). What this triage leans on:

- **The Gulf of Mexico** (BOEM's sands atlas, surveys and production by completion; BSEE's well tests): one region, and by wells mostly its shelf, a mature basin of small sands and old wells; deep water is a small part. The atlas records natural drive, not injection. A well test's shut-in tubing pressure stands for the reservoir pressure of a gas well; for an oil well the liquid column makes it a rough bound. A flowing tubing pressure is upstream of the choke, so it bounds the pressure downstream from above.
- **The UK** (NSTA's production by well): wells of fields that have ceased production, mostly gas fields of the southern sector; their choke records swing as those of wells that produce to demand.
- **Brazil** (ANP's production by well; Petrobras' 3W Dataset): the whole lives in ANP's data are mostly onshore wells of mature fields; 3W's are 40 subsea satellite oil wells of one operator, observed for hours to weeks.
- **Norway** (Sodir): fields and tests, not wells.
- **The model's own results** are from R6's pilot (400 `sol-2` wells in the interim mapping, before the rework and the tables), the pilot whose samples are kept with the generated data outside the repo.

## 1. Summary

Of the 46 items, 21 have data in hand, 6 a published source, 10 physics, 1 needs a fetch, and 8 stay guesses. For 29 of them the proposal is to keep the value, classed by its evidence where there is some; 14 would change a draw (the 13 rows of Section 2, G14 and G15 being one change); 3 are re-classed only.

| Item | Rule | The value | Class | Recommendation |
|---|---|---|---|---|
| G1 | SMP-56 | Pressure support for oils, 0.7 | data in hand | Keep; class as data |
| X1 | SMP-56 | Pressure support for gas classes, 0.4 | data in hand | Keep; class as data. The drive labels point higher, the pressures agree |
| G2 | SMP-58 | Water depths: the shelf share 0.85, the medians 90 and 350 m, the spreads | data in hand | Change: subsea log-uniform over its range; deep-water dry trees 0.10; a lower shelf floor |
| G3 | SMP-60 | S wells' hold and end section; the drain's tangent; the holds' least TVD | data in hand | Change: the S wells' hold and end section; the tangent from zero; keep the least holds |
| G4 | SMP-63 | Lithostatic gradient, 0.226 bar/m | physics | Keep; class as physics |
| G5 | SMP-63 | The caps on $\lambda$, 0.85 and 0.8 | data in hand | Keep; class as data |
| G6 | SMP-64 | Onshore surface temperature, 268 to 303 K | needs a fetch | Keep until a climatology is fetched |
| G7 | SMP-66 | $p_\text{sep}$: medians 35 and 15 bar, spreads, bounds | data in hand | Change for gas classes offshore: a higher median and bound; keep the oils' |
| G8 | SMP-66 | Separator temperatures | data in hand | Change: the subsea range's top to 340 K |
| G9 | SMP-66 | Stage factors $r_s$ | data in hand | Keep; class as data |
| G10 | SMP-68 | A table fluid's $x$ (a record only) | physics | Re-class: no longer a guess |
| G11 | SMP-69 | The ranges of $r_p$ by stage and drive | data in hand | Keep; class as data for the gases, as Goal 1's choice for depleted oils |
| G12 | SMP-70 | Condensates' $c_k \sim U(0.5, 3)$ | physics | Keep; record the replacement, the fluid's own flash |
| G13 | SMP-71 | Net pay, skin, high-angle drainage, hole, design drawdown, anisotropy, perforated share, lower bound of $D$ | stays a guess | Keep. Net pay and drawdown checked against data, the skin's range against a published table |
| G14 | SMP-72 | Slant factor $(\cos\theta_\text{end})^{-1/2}$ on $J$ | published source | Change: Cinco-Ley's slant skin |
| G15 | SMP-73 | The same factor on $a$ and $b$ | published source | Change, with G14 |
| G16 | SMP-74 | Vogel's thresholds, 0.8 and 0.6 | physics | Keep; 0.8 is physics, 0.6 a design choice |
| G17 | SMP-75 | The design point's constants; one size larger with probability 0.3 | physics | Keep; the wellhead's constants are the model's own; 0.3 stays a guess |
| G18 | SMP-76 | Roughness by stage | stays a guess | Keep; Farshad and Rieke (2006) would settle it |
| G19 | SMP-78 | Lift shares by stage | data in hand | Keep; class the offshore shares as data |
| G20 | SMP-78 | A drawn lift left off above 0.3 of $L$ | stays a guess | Keep |
| G21 | SMP-79 | The packer 30 to 150 m above the bottom | stays a guess | Keep |
| G22 | SMP-79 | Lift's design drawdown 0.3 to 0.7; least valve depth 0.3 of $L$ | stays a guess | Keep |
| G23 | SMP-79 | $\phi \sim U(0, 0.3)$, the lift gas cooled towards $T_s$ | published source | Change: the lift gas between the ambient and the reservoir temperature |
| G24 | SMP-80 | Log-uniform injected GLR at capacity | stays a guess | Keep |
| G25 | SMP-81 | The ends of $U_t$ and $k_e$; two hole sizes | published source | Keep; class as data, thinly |
| G26 | SMP-82 | The factor $(0.554/\gamma_g)^{0.4}$ | physics | Keep; class as physics, within about 5% |
| G27 | SMP-82 | $T_\text{rep}$ at 0.8 of the way | physics | Keep; class as the model's own result |
| G28 | SMP-84 | The plateau, $U(0, 0.25)\thinspace T_\text{life}$ | data in hand | Keep; class as data |
| G29 | SMP-85 | The pressure's form, $\gamma \sim U(1, 2)$ | physics | Change: $\gamma \sim U(0.7, 1.5)$ |
| G30 | SMP-86 | Breakthrough recoveries and jumps; the gases' log-linear WGR | data in hand | Change: a later breakthrough, one range for both drives |
| G31 | SMP-88 | Roughness's log-linear path | stays a guess | Keep, as G18 |
| G32 | SMP-89 | Two separator step-downs at $U(0.25, 0.9)\thinspace T_\text{life}$ | data in hand | Change: $U(0.1, 0.8)\thinspace T_\text{life}$ |
| G33 | SMP-90 | Lift's drawn start at $U(0.2, 0.9)\thinspace T_\text{life}$ | data in hand | Change: $U(0.05, 0.9)\thinspace T_\text{life}$ |
| G34 | SMP-94 | Lift gas off in 0.20 of a lifted well's slots | data in hand | Keep; class as data |
| G35 | SMP-95 | Separator pressure within ±10% | data in hand | Keep; class as data |
| G36 | SMP-98 | $p_c$, the step of ±0.1 and the share 0.1 of new draws | data in hand | Change: $p_c \sim \text{LogU}(0.2, 0.95)$; the step and the share stay |
| G37 | SMP-113 | Dirichlet weights; uniform heptanes-plus ranges | published source | Change, if the build allows: the condensates' $M_{7+}$ from 120 g/mol; the shapes stay guesses |
| G38 | SMP-116 | The table's margin of 15 K below $T_s$ | physics | Keep; confirm on R11's pilot |
| N1 | SMP-55 | Dry gases' yield limit, 2.8 Sm³ per 10⁶ Sm³ | data in hand | Re-class as a design choice; state the dry share's basis |
| N2 | SMP-65 | The gradient's bounds, 12 and 55 K/km | data in hand | Keep; class as data |
| N3 | SMP-66 | The floor of 1.5 bar and the cap of 120 bar | data in hand | Keep; the cap as data, the floor as a design choice |
| N4 | SMP-68 | Volatile oils' log-uniform GOR; dry gases' yield from 0.5; bounds of $x$ | data in hand | Keep; class part by part |
| N5 | SMP-72 | The liner's liquid constants | physics | Change: the model's water viscosity for 0.5 mPa s; the density's clamp a design choice |
| N6 | SMP-73 | The liner's cap at 7 in | stays a guess | Re-class as a design choice of the build |
| N7 | SMP-114 | $\alpha = 1$ for a lumped heptanes-plus | published source | Keep; class as a published recommendation |

## 2. The changes proposed, ranked

Ranked by how much the datasets would change. Each needs a ruling, and the first four a re-measurement on the pilot, since earlier rulings were set against the draws as they are.

| Rank | Item | Now | Proposed design value | Changes |
|---|---|---|---|---|
| 1 | G30 · SMP-86 | $F_\text{bt} \sim U(0.05, 0.4)$ under support, $U(0.2, 0.7)$ under depletion | $F_\text{bt} \sim U(0.1, 0.9)$ for both drives; the jumps as they are | The water cut of every oil well between its start and its end: drier mid-life wells |
| 2 | G2 · SMP-58 | Subsea $d_w \sim \text{LogN}_ {[80, 3000]}(350, 0.8)$ m; deep-water dry trees 0.15 of platform wells; shelf from 15 m | Subsea $\text{LogU}(80, 3000)$ m; deep-water dry trees 0.10; shelf from 5 m | The subsea wells' surface temperature, flowline loss and gradient: more deep-water wells |
| 3 | G7 · SMP-66 | Offshore $p_\text{sep} \sim \text{LogN}_ {[5, 85]}(35, 0.45)$ bar for every fluid | Gas classes offshore $\text{LogN}_ {[10, 120]}(65, 0.5)$ bar; oils as they are | Gas and condensate wells' back-pressure and rates early in life |
| 4 | G29 · SMP-85 | $\gamma \sim U(1, 2)$ | $\gamma \sim U(0.7, 1.5)$ | The pressure of oils under depletion in mid-life: higher |
| 5 | G36 · SMP-98 | $p_c \sim \text{LogU}(0.5, 0.95)$ | $p_c \sim \text{LogU}(0.2, 0.95)$ | `nsol-2`'s control histories: wells whose set points move less often |
| 6 | G32 · SMP-89 | Step dates $U(0.25, 0.9)\thinspace T_\text{life}$ | $U(0.1, 0.8)\thinspace T_\text{life}$ | The separator steps down earlier in a life |
| 7 | G33 · SMP-90 | Drawn start $U(0.2, 0.9)\thinspace T_\text{life}$ | $U(0.05, 0.9)\thinspace T_\text{life}$ | Some lifts start earlier |
| 8 | G14, G15 · SMP-72, SMP-73 | $f = (\cos\theta_\text{end})^{-1/2}$ | The slant skin $s_\theta$ added to the skin | Slanted wells' productivity: by a few per cent below 45°, more at 60° to 75° |
| 9 | G3 · SMP-60 | S hold $U(15°, 45°)$; end section $\min(U(100, 400)\ \text{m}, L/4)$; drain's tangent $U(50, 300)$ m | S hold $U(20°, 60°)$; end section $\min(U(100, 1200)\ \text{m}, L/3)$; tangent $U(0, 300)$ m | The S wells' shape, a twelfth of the wells |
| 10 | G23 · SMP-79 | $T_{lg} = T_a(H) - \phi\thinspace(T_a(H) - T_s)$, $\phi \sim U(0, 0.3)$ | $T_{lg} = T_a(H) + \psi\thinspace(T_r - T_a(H))$, $\psi \sim U(0, 0.75)$ | A lifted well's stream by a few kelvin above the valve |
| 11 | G8 · SMP-66 | Subsea $T_\text{sep} \sim U(290, 320)$ K | $U(290, 340)$ K | The black oils' $R_{so}$ correction, by a few per cent |
| 12 | G37 · SMP-113 | Condensates' $M_{7+} \sim U(140, 200)$ g/mol | $U(120, 200)$ g/mol, if the API match holds | Lighter condensates' dew points |
| 13 | N5 · SMP-72 | The liner's water at 0.5 mPa s | PVT-WAT-3 at $T_r$ | A drain's friction, slightly |

## 3. The items

### G1 and X1 · SMP-56 · Pressure support: 0.7 for oils, 0.4 for gas classes

*Data in hand.*

**Evidence.**

- **Oils.** By wells, the share of the Gulf's oil sands whose largest reservoir has a water drive or a partial water drive is within a few points of the draft's 0.7; with the combination drives it is higher, and in deep water alone it is lower [own analysis of BOEM's sands atlas]. The atlas records natural drive only, so injection adds to it. Late in their test histories, a majority of the Gulf's oil completions hold a shut-in tubing pressure above 0.7 of their first year's, a lower bound on the share that keep their reservoir pressure, since a rising water cut lowers an oil well's shut-in tubing pressure at a constant reservoir pressure [own analysis of BSEE's well tests].
- **Gas classes.** The basis pass is right about the labels: by wells, water and partial water drive hold about twice the draft's 0.4 of the Gulf's gas sands, and water drive alone a little more than 0.4 [own analysis of BOEM's sands atlas]. The pressures say otherwise. Late in their test histories, the share of the Gulf's gas completions whose shut-in tubing pressure is still above 0.7 of their first year's is close to the draft's 0.4, a little below it for the longest histories; the rest have lost more, and the distribution is continuous through the draft's gap between the two drives' late ranges [own analysis of BSEE's well tests]. So many gas sands labelled water drive lose a good part of their pressure. In the sampler, pressure support means a late $r_p$ of 0.70 or more (SMP-69), and by that meaning the Gulf's pressures give the draft's share. The UK's gas wells, of depletion-drive fields, end at a small part of their early pressures [own analysis of NSTA's production by well].

**Proposal.** Keep 0.7 and 0.4. Class both as data, chosen and not fitted: the oils' against the Gulf's natural drives and shut-in pressures, the gases' against its shut-in pressures, with the drive labels pointing higher and the UK's gas wells lower. One region carries the count.

### G2 · SMP-58 · Water depths by well type

*Data in hand.* The spec says that the Gulf's borehole file does not separate subsea wells. The earlier analysis of the Gulf's producers did separate them, by the distance from a borehole's surface location to its complex's structures, so the water depths by type are in hand for the Gulf and for Norway [own analysis of BOEM's directional surveys; own analysis of NOAA's World Ocean Atlas 2023, which lists the wells' water depths].

**Evidence.**

- **Shelf platforms.** The draft's median of 90 m lies between the Gulf's shelf wells, whose median is about half of it, and Norway's and the UK's, a little above it. The Gulf's log spread is wider than the draft's 0.6. A sizeable minority of the Gulf's platform wells stand in water shallower than the draft's floor of 15 m.
- **Deep-water dry trees.** Over all of the Gulf's platform boreholes that produced from 1996 to 2024, the share in 400 m of water or more is about a fifth of the draft's 0.15. Among those producing in 2024, and among those that started since 2010, it is a little below 0.15. Norway's platforms have next to none in such water; the UK's wells are not split by type.
- **Subsea.** The draft's median of 350 m is close to Norway's subsea wells'. The Gulf's subsea boreholes are spread over the whole range: a minority on the shelf, mostly older gas wells, and a large part in 1000 m or more. Of its subsea wells producing in 2024, most are in 1000 m or more and a sizeable minority in 2000 m or more, where the draft's log-normal puts about a tenth and a hundredth of its wells. Brazil's subsea wells' water depths are not in hand; its offshore production is largely in deep water, a general statement.

**Proposal.** Subsea: $d_w \sim \text{LogU}(80, 3000)$ m, one log-uniform over the pooled range in place of a log-normal centred on one region (median near 500 m, three in ten wells in 1000 m or more). Platform: keep $\text{LogN}(90, 0.6)$, with its floor lowered from 15 to 5 m, and lower the deep-water dry trees' share from 0.15 to 0.10, between the Gulf's present-day share and the other regions' none; deep-water platform wells are also where THM-4's linear ambient profile is weakest (D-14). Deeper subsea wells have a colder seabed, a larger flowline loss and a lower gradient, so the subsea wells' rates, already below the regions' at the median (L5), must be measured again on the pilot.

### G3 · SMP-60 · The S wells' hold and end section, the drain's tangent, the holds' least TVD

*Data in hand.* These were R3's values from before the surveys were read; the surveys' stations are in hand.

**Evidence** [own analysis of BOEM's directional surveys, and of NLOG's deviation surveys for the hold angle].

- **Hold angle.** The S wells' hold angles in the Gulf and offshore the Netherlands run from about the draft's lower end to well above its upper end of 45°: a sizeable minority of the Gulf's S wells hold above 45°. The pooled 5th to 95th percentiles lie close to the deviated class's own range of 20° to 60° (SMP-59). An S well drawn below 20° would belong to the slightly deviated class by its maximum inclination.
- **End section.** The Gulf's S wells run on below their drop for longer than the draft allows: the median is near the top of the draft's 100 to 400 m of TVD, the 95th percentile about three times it, and as a part of the well's TVD the upper tail passes the draft's cap of a quarter.
- **The holds' least TVD.** Holds shorter than the draft's least 50 m (build-and-hold) and 100 m (S) are few in the Gulf: the two floors sit at its low percentiles.
- **The drain's tangent.** Most of the Gulf's drains pass from 60° to 75° on their landing build without a tangent; a minority hold one of 50 m or more.

**Proposal.** S wells: $\theta_h \sim U(20°, 60°)$, the class's range, and an end section of $\min(U(100, 1200)\ \text{m}, L/3)$ of TVD, by D-104's rule of a uniform draw over the pooled range. The drain's tangent: $U(0, 300)$ m, so that the common drain without one is drawn. Keep the least holds and class them as data. The Dutch stations would add a second region to the end section and the holds; they are in hand and were not run.

### G4 · SMP-63 · The lithostatic gradient of 0.226 bar/m

*Physics.*

**Evidence.** The gradient is $g\rho_b$. A sediment of grain density 2650 kg/m³ filled with brine has a bulk density of 2010 to 2570 kg/m³ at porosities of 0.40 to 0.05, a gradient of 0.197 to 0.252 bar/m; 1 psi/ft is a bulk density of 2307 kg/m³, a porosity of 0.21. Young sediments near the mudline are lighter. In the rule the gradient is a scale and not a claim about a well: $\lambda_{50}(z)$ and its spread were fitted with it, so another gradient would refit them to the same pressures. Together with the cap of 0.85 it bounds the initial pressure's gradient below the mudline at about 0.92 psi/ft (G5).

**Proposal.** Keep. Class it as physics: the overburden of a sediment of 2300 kg/m³, the scale of the fit. Swarbrick and Osborne's (1998) review is no longer needed for it.

### G5 · SMP-63 · The caps on $\lambda$: 0.85, and 0.8 in the tail

*Data in hand.*

**Evidence.** Against a lithostatic 1 psi/ft below the mudline, the Gulf's sands reach $\lambda$ of 0.85 in very few sands, half of them deeper than 5 km, and nearly all of its sands lie below the tail's 0.8 at every depth, the deepest less so [own analysis of BOEM's sands atlas]. The upper tail of the pressure gradients of Norway's drill-stem tests ends near the gradient that the cap implies [own analysis of Sodir's FactPages]. With a lighter overburden of 0.9 psi/ft a few sands would lie above lithostatic, which also supports the scale of G4.

**Proposal.** Keep both. Class them as data: the cap sits at the top of the two regions' initial pressures.

### G6 · SMP-64 · The onshore surface temperature of 268 to 303 K

*Needs a fetch.*

**Evidence.** Nothing in hand gives ground or air temperatures: NOAA's atlas is of the ocean. The locations of onshore wells are in hand for three regions (USGS's produced-water samples in the US, NLOG's Dutch boreholes, NSTA's English wells), so an open climatology of the mean annual temperature read at those locations would give the range. Two such are ERA5-Land's 2 m temperature (Copernicus Climate Change Service) and WorldClim 2.1's annual mean temperature. Other onshore provinces (West Siberia, the Middle East, North Africa) have no well locations in hand; their range would come from the climatology's values over the provinces.

**Proposal.** Keep the value as a guess until the fetch. It touches the onshore wells only, a tenth of the core.

### G7 · SMP-66 · The medians, spreads and bounds of $p_\text{sep}$

*Data in hand.* The basis line says that no public distribution of separator or manifold pressures was found. Two proxies are in hand, both upper bounds: a wellhead or flowing tubing pressure with the choke wide open is the pressure downstream plus the open choke's small loss (a tenth at the median in R6's pilot at openings of 0.8 or more).

**Evidence.**

- **Oil wells on platforms.** The Gulf's platform oil wells tested on a wide-open choke flow at tubing pressures whose median is about half the draft's initial median of 35 bar in their first year, and lower later; its gas-lifted tests the same; the low cluster of its casing-or-line pressures, the lines, lies there too at every age [own analysis of BSEE's well tests]. These wells flow into lines near the draft's late-stage median all their lives. Published first stages, 41 to 83 bar (Guo, Lyons and Ghalambor 2007, §10.2.5), and the North Sea trains of [WP §7] are higher. The draft's 35 bar lies between.
- **Gas wells on platforms.** In their first year of tests the Gulf's platform gas wells flow on wide-open chokes at tubing pressures whose median is about twice the draft's 35 bar, with an upper tail far above its bound of 85 bar [own analysis of BSEE's well tests]. The UK's gas wells, in the first fifth of their lives with the choke open, have the same median, and their upper tail passes 85 bar [own analysis of NSTA's production by well]. Two regions agree: a gas well starts against a high-pressure line, about twice the draft's median.
- **Onshore.** Nothing in hand.

**Proposal.** For gas and condensate wells offshore, $p_\text{sep} \sim \text{LogN}_ {[10, 120]}(65, 0.5)$ bar; keep the oils' $\text{LogN}_ {[5, 85]}(35, 0.45)$ bar, as a value between the Gulf's low-pressure lines and the published first stages, and the onshore draw as a guess. The stage factors then bring the gas wells down as the two regions show (G9). It is one more branch in the rule, with a measured need. The gas wells' permeability factor (D-62) was set against rates at the draft's back-pressure, so their rates need a new measurement on the pilot.

### G8 · SMP-66 · The separator temperatures

*Data in hand,* thinly.

**Evidence.** 3W's subsea satellite wells arrive topside, downstream of the production choke, at temperatures whose median lies above the top of the draft's subsea range of 290 to 320 K and whose spread reaches the platforms' top of 340 K [own analysis of Petrobras' 3W Dataset]. They are one operator's wells on short, insulated lines; a long tieback arrives colder. The UK's gas wells' wellhead temperatures, upstream of the choke, spread over about the draft's platform range [own analysis of NSTA's production by well]. A textbook's separator examples run at 80 °F and 800 psia, and at 70 °F and 600 psig (Guo, Lyons and Ghalambor 2007, Ch. 10): 300 and 294 K, the bottom of the draft's ranges. The temperature enters only the fluid's separator correction (PVT-OIL-5) and a table fluid's offsets.

**Proposal.** Subsea: $U(290, 340)$ K, the platforms' top. Keep the platform and onshore ranges. Class the three as data, thinly.

### G9 · SMP-66 · The stage factors $r_s$

*Data in hand.*

**Evidence.** The UK's gas wells' wellhead pressure with the choke open, by fifth of the life over the first fifth's: its median in the second fifth lies in the lower part of the draft's mid-life $U(0.5, 1)$, in the third fifth below 0.5, and in the last two fifths in the lower half of the draft's late $U(0.1, 0.6)$, with the wells' spread in the last fifth close to that range [own analysis of NSTA's production by well]. The Gulf's platform gas wells' wide-open flowing pressures fall over their test histories by about the same factors; its platform oil wells' lines change little with age (G7) [own analysis of BSEE's well tests].

**Proposal.** Keep both ranges. Class them as data, for gas wells of two regions; for oil wells the Gulf shows little fall, which the mid-life range's top of 1 covers.

### G10 · SMP-68 · A table fluid's $x$

*Physics.* SMP-115's flash sets a table fluid's saturation pressure (L2, ruled 2026-11-09), and $x$ is kept only as a record so that the block's numbers do not depend on the table.

**Proposal.** Take the row out of the guesses: it is a record of the generation, a design choice for reproducibility (SMP-51), and its shares and median decide nothing.

### G11 · SMP-69 · The ranges of $r_p$ by stage and drive

*Data in hand.* Since D-105 the life uses only some of these ranges as pressures: the initial rows as its start, and the late row under pressure support as a supported well's end. A depletion life ends at SMP-83's own ranges, and the late depletion row serves the late-stage lift design.

**Evidence.** The Gulf's gas completions' shut-in tubing pressure late in the test history, over the first year's [own analysis of BSEE's well tests]:

- the part above 0.7 is spread over 0.7 to 1 about as the draft's late $U(0.70, 1.0)$ under support;
- the low tail lies near SMP-83's depletion end for gases, so a life that runs towards exhaustion is within the data;
- in the middle of the history a sizeable minority lie below the draft's mid-life floor of 0.55 under depletion, which the life's own path covers.

For oil wells the shut-in pressure is a rough bound. With a liquid column of constant weight to the well's depth, the Gulf's oil wells keep most of their pressure late in their histories, and few fall into the draft's late depletion range of 0.3 to 0.6 [own analysis of BSEE's well tests]; the atlas gives depletion and solution-gas drive to a small part of the oil wells [own analysis of BOEM's sands atlas]. The first-year reference is not the initial pressure, and a rising water cut biases the oil wells' estimate.

**Proposal.** Keep the ranges. Class the gases' as data (one region, a proxy). Class the oils' deep depletion as the design choice it is: Bjarne's goal 1 asks for depleted wells, and the data in hand show them to be a minority of one region's oil wells.

### G12 · SMP-70 · The condensates' factor $c_k \sim U(0.5, 3)$

*Physics,* on one fluid.

**Evidence.** If only the vapour flows below the dew point, the producing GOR is the inverse of the vapour's vaporised oil–gas ratio, $R_{go}/R_{si} = R_v(p_\text{dew})/R_v(p)$. On the flash of Pedersen et al.'s (2015) Table 3.8 condensate in the modified black-oil form (`plans/evidence/condensate_rv_form.py`), near its reservoir temperature this gives $c_k$ of 1.6 to 2.8 between 0.9 and 0.5 of the dew-point pressure, and 2.4 to 4.0 at 0.3 of it. In a colder reservoir (340 to 380 K) the same fluid drops more liquid, and $c_k$ runs from about 2.5 near the dew point to between 6 and 16 at half of it: the rise is not linear in the pressure. Liquid that flows lowers the GOR, so these are upper values for this fluid. The draft's range brackets it at its own temperature down to half the dew-point pressure; its lower half has no fluid behind it, and a lean condensate was not run.

**Proposal.** Keep $U(0.5, 3)$, and class it as physics on one published fluid. Record the replacement that the plans lack: each table fluid's own $R_v(p)$ at $T_r$, from the flash that already builds its table, which would tie a condensate's GOR trend to its composition as SMP-70 ties an oil's to its material balance. It is build work for R10 or later, not a draw.

### G13 · SMP-71 · Net pay, skin, high-angle drainage, hole, design drawdown, anisotropy, perforated share, lower bound of $D$

*Stays a guess,* with three parts checked.

**Evidence.**

- **Net pay.** The Gulf's sands' net thickness by wells has a median about half the draft's 15 m on the shelf and equal to it for deep-water oil sands, with a log spread a little below the draft's 0.8; a minority of the completions are in sands thinner than the draft's floor of 3 m [own analysis of BOEM's sands atlas]. One region of thin, stacked sands; a completion is in one sand, and a well may produce several.
- **Design drawdown.** For a gas well, one minus the flowing over the shut-in tubing pressure is the drawdown plus the tubing's losses, an upper bound. Its median over the Gulf's gas wells lies within the draft's 0.1 to 0.4 on the shelf, near its top, and subsea; on deep-water platforms it lies above it [own analysis of BSEE's well tests].
- **Skin.** Partial completion alone gives skins of 0.7 to 0.9 at a completed share of 0.75, 2.4 to 3.1 at 0.5, 6.6 to 8.6 at 0.25 and 15 to 21 at 0.1 (Economides, Hill and Ehlig-Economides 1994, Table 5-1, p. 90–91, after Cinco-Ley et al. 1975, for $h_D$ = 100). The draft's range of −3 to 15 spans them with damage and stimulation; its mean of 2 has no source. The rule draws the perforated share for $D$ and the skin independently.
- **The hole of 8.5 in** is the common production hole (Hasan and Kabir's $T_D$ values are for it, [WP §10]): a standard size, not a guess about wells.
- **The rest** (the high-angle wells' drainage of 8.5 to 9.5, the anisotropy of 1 to 3, the perforated share of 0.3 to 1, the lower bound of $D$): nothing found.

The atlas's own rock permeabilities lie more than an order of magnitude above the sampler's effective ones. The permeability was set against per-well rates with these draws as they are (D-100, D-62), so the parts are not identified one by one: only $k h_n$ over the skin and drainage is.

**Proposal.** Keep all eight. Class the net pay as data (between the Gulf's shelf and deep-water medians), the drawdown as data (an upper-bound check), the skin's range as a published range, and the hole as a standard size. The other four stay guesses, and their joint effect is carried by the effective permeability. Lowering the net pay's floor would cover the Gulf's thinnest sands; it is not proposed, because the product with the permeability is what the rates check.

### G14 and G15 · SMP-72, SMP-73 · The slant factor

*Published source,* in hand and now read.

**Evidence.** Economides, Hill and Ehlig-Economides (1994, §5-4, pp. 88–91) give Cinco-Ley et al.'s (1975) skin of a slanted well, $s_\theta$, in tables. For a fully completed well with $h_D = h/r_w$ = 100 (Table 5-1, p. 91) it is −0.128, −0.517, −1.178, −2.149 and −3.577 at 15°, 30°, 45°, 60° and 75°; Table 5-2 gives $h_D$ = 1000. The usual fit to the tables,

$$s_\theta = -\left(\frac{\theta'}{41}\right)^{2.06} - \left(\frac{\theta'}{56}\right)^{1.865}\log_{10}\frac{h_D}{100}, \qquad \theta' = \arctan\left(\frac{\tan\theta}{I_\text{ani}}\right), \qquad h_D = \frac{h_n I_\text{ani}}{r_w},$$

with the angles in degrees, reproduces those five values within 3%; the fit is written here from memory of the literature and is not printed in the book, so the table is the citation. The slant then multiplies the productivity by $(A + s)/(A + s + s_\theta)$, with $A = \ln(r_e/r_w) - 0.75$.

- For the median well (skin 2, $\ln(r_e/r_w)$ = 8) the table gives factors of 1.06, 1.15, 1.30 and 1.63 at 30°, 45°, 60° and 75°; the placeholder gives 1.07, 1.19, 1.41 and 1.97.
- Over the sampler's draws of pay, skin and drainage, the placeholder over the slant skin's factor is 0.85 / 1.01 / 1.11 (5th, 50th and 95th percentiles) from 20° to 60°, and 0.58 / 1.09 / 1.36 from 60° to 75°: unbiased at the median, but blind to the skin, on which the gain depends.
- Where $A + s$ is small (a stimulated well at a high angle) the factor grows without bound, and in about 1% of the draws at 60° to 75° the denominator is near zero or negative. The correlation holds to 75°.

**Proposal.** Replace the placeholder by the slant skin: $J$ and $a$ computed with $s + s_\theta$ in place of $s$; for $b$, which does not depend on the skin, the ruling chooses between a vertical well's and the same factor as $a$'s; with a floor on $A + s + s_\theta$ (a design value, such as 2) where a stimulated well's draw would pass it. It affects build-and-hold and S wells; at their median the change is a few per cent below 45° and up to a fifth at 75°. The drains' own form (D-21) is untouched.

### G16 · SMP-74 · Vogel's thresholds: 0.8 of $p_r$, and a WLR of 0.6

*Physics* for 0.8.

**Evidence.** Take the composite inflow as the reference: straight above the bubble point and Vogel's below, with the same $J$ at $p_r$. For a saturation pressure at 0.8 of $p_r$ and drawdowns up to half of $p_r$, Vogel's curve from $p_r$ is at most 14% below the composite, and the straight index at most 11% above it. The two errors are equal at $p_\text{sat}/p_r$ of about 0.83. Over the whole range to the open flow they are equal only at 0.5, with 29% each. So 0.8 is where the model's two options err about equally on the drawdowns that the samples mostly have. For the WLR nothing was derived: a composite with water weights the curve by the oil's share, so any threshold near one half separates the mostly curved from the mostly straight.

**Proposal.** Keep both. Class 0.8 as physics: the equal-error threshold against the composite inflow, rounded. Class 0.6 as a design choice within Vogel's stated limit (`validity.md`, note 10). The replacement stays the composite inflow, which is not a model option.

### G17 · SMP-75 · The design point's constants, and one size larger with probability 0.3

*Physics* for the wellhead: the model's own result.

**Evidence** (R6's pilot, 342 core wells, per-well medians):

- **Temperature.** The solved wellhead temperature lies 0.60 of the way from $T_s$ to $T_r$ at the median (0.37 to 0.74 between the quartiles; 0.54 at the initial stage with the choke at 0.8 or more). The rule's 0.6 is the model's median.
- **Pressure.** At the initial stage with the choke at 0.8 or more, the solved wellhead pressure is 0.75 of the rule's $1.5\thinspace p_\text{sep} + 5\ \text{bar} + \Delta p_\text{fl}$ at the median (0.66 to 0.89 between the quartiles; oils 0.70, gas classes 0.86). The rule sizes the tubing at a wellhead pressure a third above an open choke's, that of a partly closed choke; a denser design stream gives a slightly smaller tubing in borderline wells.
- **No water, or a WGR of 10,** is the state when the tubing is run: a design choice, not a guess.
- **One size larger with probability 0.3:** no distribution of tubing sizes is public, and none of the per-well sources in hand carries a tubing size.

**Proposal.** Keep all. Class 0.6 as physics, the model's own median; the pressure as a design choice checked against the model; the initial state as a design choice; and 0.3 stays a guess.

### G18 and G31 · SMP-76, SMP-88 · Roughness by stage, and its path

*Stays a guess.*

**Evidence.** Nothing in hand measures tubing roughness against age. The whole range brackets the published examples of 30 and 46 µm ([WP §6]). Farshad and Rieke (2006), on [WP]'s resource list, would give new tubing by material; no source for used tubing is named.

**Proposal.** Keep both. Ask for Farshad and Rieke (2006) only if R11's pilot shows that the roughness moves the rates: by Chen's factor a tenfold roughness changes the friction factor by a few tens of per cent, and friction is a small part of most wells' pressure drop.

### G19 · SMP-78 · The lift shares by stage

*Data in hand.*

**Evidence.**

- **Oils on platforms.** The draft's 0.20, 0.45 and 0.70 lie within the Gulf's gas-lifted shares of oil well-years by years on production, between its deep-water platforms' and its shelf's [own analysis of BOEM's OGOR-A production] (checkpoint F1).
- **Oils subsea.** The Gulf lifts few subsea wells, a small part of the draft's shares at every age; most of 3W's Brazilian subsea satellite wells are on gas lift [own analysis of BOEM's OGOR-A production; own analysis of Petrobras' 3W Dataset]. The draft lies between two regions that differ widely.
- **Gas classes.** The draft's 0, 0.05 and 0.15 are near the Gulf's gas wells', which rise with age from near zero.
- **Onshore.** Nothing in hand; onshore wells are mostly pumped, which the model does not have.

**Proposal.** Keep, as ruled (F1). Class the offshore oils' and the gases' shares as data, chosen and not fitted; the onshore oils' stay a guess.

### G20, G21 and G22 · SMP-78, SMP-79 · The balance point's 0.3 of $L$, the packer, and the lift's design drawdown

*Stay guesses.*

**Evidence.** No source in hand gives valve or packer depths: the Gulf's tests and production carry no depths of completion equipment, and 3W publishes no depths. The packer's 30 to 150 m brackets the one textbook example of 61 m (Guo, Lyons and Ghalambor 2007, Example 13.1). For the lift's design drawdown, the Gulf's gas-lifted tests give flowing tubing pressures and no bottomhole pressures.

**Proposal.** Keep the four values. Golan and Whitson's (1991) §5.3 and Economides et al.'s (1994) gas-lift chapter are in hand as scans without a text layer and were not read for this triage; a reading of their design examples is the named next step, and no fetch is needed.

### G23 · SMP-79 · The lift gas's temperature, $\phi \sim U(0, 0.3)$

*Published source.* The basis line says that no numbers were extracted from Hasan and Kabir (2018, §12.2.2). Their §12.2.3 has the solved examples.

**Evidence.**

- Hasan and Kabir (2018), Figs. 12.16 to 12.19 (a well of 15,000 ft at 9,286 STB/D, Table 12.2): the annulus gas enters at the surface, warms within the first few hundred feet, and then runs between the geotherm and the tubing's fluid, close to the tubing's: "the annular gas also is generally at a high temperature, except in the area close to the wellhead" (§12.2.3). Read off Fig. 12.16, the annulus gas lies about two thirds to three quarters of the way from the formation's temperature to the reservoir's at 6,000 and 12,000 ft.
- Fig. 12.20, the same well at a low oil rate: the tubing's fluid and the annulus gas run parallel to the geotherm and a little above it.
- Figs. 12.21 and 12.22, a well of 4,500 ft: the annulus gas is far above the geotherm at mid-depth and a few °F below the formation's temperature at the bottom.
- By the formation's heat alone (Ramey 1962, the injection solution), the gas would lag the geotherm by a share $\phi = (A/H)(1 - e^{-H/A})$ of the rise, with $A$ the relaxation distance: 0.03 to 0.9 over lift-gas rates of 0.1 to 3.5 kg/s and valve depths of 500 to 4000 m. The tubing's heat removes most of that lag.

So at the valve the lift gas is at the ambient temperature or above it, towards the tubing's fluid; it is below the ambient only by a few kelvin, at a valve near the bottom. The draft cools it by up to 0.3 of the way to the surface temperature, the wrong side for most valves.

**Proposal.** $T_{lg} = T_a(H) + \psi\thinspace(T_r - T_a(H))$ with $\psi \sim U(0, 0.75)$: from THM-9's default at the valve to three quarters of the way to the reservoir temperature, which bounds the tubing's fluid from above. One uniform draw, as now. The values are read off figures of three examples, two of them synthetic. The effect is a few kelvin in a lifted well's stream above the valve.

### G24 · SMP-80 · The log-uniform injected GLR at capacity

*Stays a guess.*

**Evidence.** The Gulf's well tests looked like a source, since they report a gas rate for gas-lifted tests. They are not: a well's test GOR is the same in its gas-lifted and its flowing tests at the median, so the reported gas is the formation's and no injected GLR follows [own analysis of BSEE's well tests]. The Gulf's gas-lifted wells' test liquid rates lie mostly below the rate at which the smallest GLR of the range reaches the capacity's floor of 0.1 kg/s, so for them only the upper part of the GLR range applies, as SMP-80's truncation has it. 3W has rates of lift gas and none of liquid.

**Proposal.** Keep. Injection and liquid rates of the same wells would replace it, and no open source of them is known.

### G25 · SMP-81 · The ends of $U_t$ and $k_e$, and the two hole sizes

*Published source,* thinly, with a sensitivity by physics.

**Evidence.**

- **$U_t$'s lower end of 11 W/(m² K)** is Ramey's (1962, p. 428) film coefficient of a turbulent gas, 11 to 28 W/(m² K), and Hasan and Kabir's (2018, Fig. 12.17) low annulus coefficient of 2 Btu/(h ft² °F). Their coefficients in the gas-lift examples run from 2 to 4 for the annulus and 5 to 20 for the tubing, 11 to 114 W/(m² K) (Figs. 12.16 to 12.18); a liquid-filled annulus gives 36 to 57 (Table 6.1, Fig. 7.39).
- **$U_t$'s upper end** hardly matters: for 4-1/2 in tubing after a year, $h$ rises by 4% to 6% from $U_t$ = 57 to 85 W/(m² K), because the formation's resistance then limits it; from 36 down to 11 it falls by 27% to 36%.
- **$k_e$:** Ramey's 2.42 W/(m K) and Hasan and Kabir's 1.3 Btu/(h ft °F), 2.25 W/(m K) (Table 12.2), lie in the upper part of the draft's 1.5 to 2.6; the lower end has no source.
- **The holes** of 0.108 and 0.156 m are standard bit sizes; the larger lowers $T_D$ from 4.37 to 4.00 after a year.

**Proposal.** Keep. Class the ends of $U_t$ as data (published examples), the hole sizes as standard sizes, and $k_e$'s lower end as the one guess left. Willhite (1967) and Alves et al. (1992) are no longer needed for the range.

### G26 · SMP-82 · The gas's factor $(0.554/\gamma_g)^{0.4}$

*Physics,* at low pressure.

**Evidence.** The ideal-gas heat capacity per mass of methane with ethane to pentane, from the components' polynomials, over methane's: for gravities of 0.60 to 0.81 it is 0.97 to 0.88 at 300 K, against the factor's 0.97 to 0.86, and 0.98 to 0.91 at 380 K. The factor is within 2% at 300 K and up to 6% low at 380 K; with a few per cent of CO₂ and N₂ it is 2% high at 300 K. An exponent of 0.3 would fit both temperatures within about 2%. At pressure the heavier gas's residual heat capacity is larger than methane's, so the factor errs low there by more; that was not computed. The polynomials were written from memory of a standard table and reproduce NIST's methane values, which the spec quotes, within 1.5%.

**Proposal.** Keep the factor, and class it as physics within about 5% at low pressure for gravities up to 0.8. The replacement is in the build's reach: NeqSim gives a natural gas's $c_p$ at pressure, and a table fluid's flash gives its own.

### G27 · SMP-82 · The representative temperature at 0.8 of the way

*Physics:* the model's own result.

**Evidence.** In R6's pilot the mean of the solved bottomhole and wellhead temperatures lies 0.80 of the way from $T_s$ to $T_r$ at the median of the core wells (0.69 to 0.87 between the quartiles; oils and gas classes alike).

**Proposal.** Keep. Class it as physics: the median place of the model's own flowing temperature.

### G28 · SMP-84 · The plateau of $U(0, 0.25)\thinspace T_\text{life}$

*Data in hand.* The gap named, plateau lengths per well, can be measured on the whole lives in the monthly production.

**Evidence.** For the Gulf's wells with a whole life of five years or more in the data, the part of the life until the 12-month rate last holds four fifths of its best is close to the draft's median for oil wells and gas wells alike; in a sizeable minority it passes the draft's cap of 0.25. A well's best 12 months come at the very start at the median [own analysis of BOEM's OGOR-A production]. Brazil's whole lives, mostly onshore wells of mature fields, hold their rate much longer by this measure and peak later [own analysis of ANP's production by well]. The measure counts a rate restored by a workover as a plateau, so its upper tail is too long.

**Proposal.** Keep, and class it as data: the Gulf's median agrees, and the decline was set against the same wells' rates by fifth of the life (D-41), which a longer plateau would undo.

### G29 · SMP-85 · The pressure's form, $\gamma \sim U(1, 2)$

*Physics.* D-42 set the material balance aside because the draws had no relative permeabilities; since C1 they have (SMP-70).

**Evidence.** Muskat's material balance on the model's black oil (`plans/evidence/solution_gas_drive.py`, 75 oils over the draft's ranges), from the bubble point down to 0.2 to 0.3 of it: the pressure against the cumulative oil is close to a straight line, slightly concave. The exponent that fits the sampler's form is 0.84 at the median (0.67 to 0.98 between the 5th and 95th percentiles) with Fig. 9.16's $k_g/k_o$, and 0.65 with the tail's blend at $w$ = 0.5. Above the bubble point the pressure falls fast for little oil, which raises the exponent of the whole path: by a rough estimate, not computed, to about 1 at the draft's median undersaturation, and above 1.5 for the most undersaturated oils. Golan and Whitson's (1991) Example 4.11, fitted at 1.40, is such an oil.

**Proposal.** $\gamma \sim U(0.7, 1.5)$ for oils and supported wells, the material balance's range from saturated to undersaturated oils. For supported wells there is still no physics; their pressure moves little, so one range for both keeps the rule simple. The material balance itself, $p_r(F)$ from the integration that SMP-70 already makes, with a term for the oil above the bubble point, stays the named replacement; it is not proposed now, as it adds a compressibility to the draws.

### G30 · SMP-86 · Breakthrough recoveries and jumps, and the gases' log-linear WGR

*Data in hand.* Per-well water-cut curves, the replacement named, are in hand for the Gulf's whole lives.

**Evidence** [own analysis of BOEM's OGOR-A production], for oil wells with a whole life of five years or more in the data:

- **When.** Of the wells that start dry, water breaks through in nearly all. The share of the well's final cumulative oil produced by then spreads over almost the whole range, with a median more than twice the draft's under pressure support: the draft's $U(0.05, 0.4)$ covers only the earlier breakthroughs, and its $U(0.2, 0.7)$ under depletion misses the latest. The wells cannot be split by drive.
- **The jump.** The water cut in the year after breakthrough has a median at the top of the draft's $U(0.05, 0.3)$; it holds a year's rise as well as the jump, so it does not show that the jump is too small.
- **The start.** Most of the Gulf's shelf oil wells start wet, with a first-year water cut above the draft's initial $U(0, 0.1)$; most of its deep-water wells start dry. In Brazil's whole lives nearly all wells start wet [own analysis of ANP's production by well]. A well drilled into a swept reservoir has no breakthrough of its own.
- **Gas wells.** Not measured: the gas wells' water is thinly reported.

Norway's fields' water cut lies below the draws' at every age (checkpoint, G), which a later breakthrough would bring closer.

**Proposal.** $F_\text{bt} \sim U(0.1, 0.9)$ for both drives, a uniform draw over the pooled range, in place of two ranges that the data cannot tell apart; keep the jumps and the gases' form as guesses. The water cut by years on production must then be measured again against the regions' per-well medians (C1), since mid-life wells become drier. A linked finding for a ruling of its own: the initial state has no wet starts, which most wells of the mature regions in hand are. SMP-69's water is classed as data on Norway's fields, and it is not among this triage's items.

### G32 · SMP-89 · Two step-downs of the separator, at $U(0.25, 0.9)\thinspace T_\text{life}$

*Data in hand,* for gas wells of one region.

**Evidence.** For the UK's gas wells with the choke open, the point of the life at which the wellhead pressure first falls below 0.6 of its first fifth's median is earlier than the draft's first step in most wells, and the point at which it falls below 0.3 is earlier than the draft's second: both medians lie well before the draft's, which are 0.44 and 0.71 of the life, and the range proposed below spans both spreads [own analysis of NSTA's production by well]. The pressure falls in more than two steps, with compression phases and with the field's depletion. The Gulf's platform oil wells' lines show little change with age (G7).

**Proposal.** Step dates of $U(0.1, 0.8)\thinspace T_\text{life}$. Keep two steps: simple, and the sizes are SMP-66's.

### G33 · SMP-90 · Lift's drawn start at $U(0.2, 0.9)\thinspace T_\text{life}$

*Data in hand.*

**Evidence.** Two sources of one region agree. Of the Gulf's oil wells that were ever on gas lift, a sizeable minority are on lift from their first half year. For the others, the first month on lift, as a part of the well's life, spans a range that starts well below the draft's 0.2 and ends near its 0.9, with a median below the draft's lower quartile [own analysis of BOEM's OGOR-A production]; the first gas-lifted test's place in a well's test history gives the same [own analysis of BSEE's well tests]. Deep-water platform and subsea wells start later than shelf wells. In the sampler the drawn date is one of three rules and the earliest wins.

**Proposal.** $U(0.05, 0.9)\thinspace T_\text{life}$: the range by the data, the uniform shape for simplicity.

### G34 · SMP-94 · The lift gas off in 0.20 of a lifted well's slots

*Data in hand.*

**Evidence.** After a well's first gas-lifted test, the share of its later tests that are flowing tests is close to 0.20, pooled over the tests and as a mean over the wells; after a well's first month on gas lift, the share of its producing months not on lift is a little below 0.20 [own analysis of BSEE's well tests; own analysis of BOEM's OGOR-A production]. The typical well is off much less, and a minority of wells make up most of the share. 3W's windows, hours to weeks around events, show the typical lifted well with its lift gas on throughout [own analysis of Petrobras' 3W Dataset].

**Proposal.** Keep 0.20, as ruled (D-38). Class it as data: one region's share of tests and months.

### G35 · SMP-95 · The separator pressure within ±10% of its nominal

*Data in hand,* by a proxy.

**Evidence.** The UK's gas wells' wellhead pressure with the choke open, over its 91-day running median: most days lie within ±10%, and the typical well's 5th to 95th percentile band is as wide as the draft's uniform ±10% [own analysis of NSTA's production by well]. The proxy holds the open choke's loss and the rate's changes as well as the line's pressure, so the line itself varies no more than this.

**Proposal.** Keep ±10%. Class it as data: an upper bound from one region's gas wells.

### G36 · SMP-98 · The change probability, the step and the share of new draws

*Data in hand.* The gap named was histories of set points over wells' lives; the UK's wells have daily choke openings over whole lives.

**Evidence.**

- **How often.** Sampled at 7, 30 and 70 days, the span of `nsol-2`'s sample spacing, the typical UK gas well's choke opening changes in fewer intervals than the draft's lowest 0.5, and the wells spread from well below the draft's range to within it [own analysis of NSTA's production by well]. The draft's range starts at 0.5. 3W's wells move their chokes every day or so at the median, and a sizeable minority not at all within their windows [own analysis of Petrobras' 3W Dataset]. The two regions together span from about 0.2 to nearly 1.
- **How far.** The UK wells' changes are larger than the draft's: a large minority exceed the draft's largest step of 0.1 of the range, where the draft has under a tenth. These are gas wells whose chokes swing between open and partly closed, as wells that produce to demand do; they are fully open far more often than the 0.05 of Bjarne's goal 4. Over a life the typical well's opening travels over a large part of the range, which the draft's walk covers.

**Proposal.** $p_c \sim \text{LogU}(0.2, 0.95)$: the lower end by the UK's wells, the upper by 3W's. Keep the step of ±0.1 and the share of 0.1 of new draws as guesses: the one history in hand is of swing producers, whose shape is not that of an oil well, and a higher share of new draws would move every well's history.

### G37 · SMP-113 · The Dirichlet weights and the uniform heptanes-plus ranges

*Published source,* for one range.

**Evidence.** Elsharkawy (2002, Table 1, p. 151) gives the span of 340 gas condensates: a heptanes-plus molar mass of 106 to 253 g/mol with an average of 142, its specific gravity 0.72 to 0.85 with an average of 0.77, and a heptanes-plus mole fraction up to 0.170. The paper has been in hand since 2026-11-06 for its Table 3. The draft's $M_{7+} \sim U(140, 200)$ g/mol for condensates starts at that average, so the lighter half of the published condensates is not drawn. No such table for volatile oils is in hand, and no statistics of the light ends: the Dirichlet weights over three examples stay a guess.

**Proposal.** Lower the condensates' range to $U(120, 200)$ g/mol, if the composition's match to the drawn API gravity holds there: a lighter heptanes-plus gives a lighter stock-tank oil, so the build must show that $C_f$ stays within Søreide's range. The shapes stay guesses; class the condensates' ranges as data.

### G38 · SMP-116 · The table's margin of 15 K below $T_s$

*Physics:* the model's own result.

**Evidence.** In R6's pilot (80,000 samples, the interim black oil with the Joule–Thomson term on), 1% of the samples have a wellhead temperature below $T_s$, 0.02% more than 5 K below, and none more than 10 K below. The grid's rounding to 10 K adds to the margin.

**Proposal.** Keep 15 K, and class it as physics on the interim mapping. R11's pilot confirms it with the tables by the share of `off_table` roots, as `fluids-draft.md` names.

### N1 · SMP-55 · The dry gases' yield limit of 2.8 Sm³ per 10⁶ Sm³

*Data in hand.*

**Evidence.** The Gulf's gas sands' yields have no gap or second mode near the limit: besides the sands that report no liquid at all, they fall off smoothly below their mode [own analysis of BOEM's sands atlas]. The draft's dry share of 0.07 matches the Gulf's gas wells with no reported liquid, between its counts by wells and by sands. The Gulf's share below the limit is three to four times that, because many sands report a small yield. In the model a yield below the limit is a liquid share of a few parts in a thousand by mass, so a well just below it and one just above behave alike.

**Proposal.** Class the limit as a design choice: a threshold for the label `dry_gas`, standing for McCain's dry gas, which the model can only hold with a trace of liquid. State in SMP-55's basis that 0.07 counts the wells with no reported liquid. No change to a draw.

### N2 · SMP-65 · The gradient's bounds of 12 and 55 K/km

*Data in hand.*

**Evidence.** The lower bound lies below nearly all of the Gulf's deep-water sands' gradients below the mudline, the coldest population in hand [own analysis of BOEM's sands atlas]. The upper bound lies above nearly all of Norway's drill-stem tests' gradients, the hottest [own analysis of Sodir's FactPages]. Hotter basins are on record (Nadeau, Sun and Ehrenberg 2023, §2), where $T_r \le 450$ K caps the draw.

**Proposal.** Keep. Class the bounds as data: just outside the two regions' extremes.

### N3 · SMP-66 · The floor of 1.5 bar and the cap of 120 bar

*Data in hand,* for the cap.

**Evidence.** The UK's gas wells' open-choke wellhead pressures early in life reach the cap only at the top of their range, and the Gulf's platform gas wells' first-year wide-open flowing pressures pass it in their upper tail [own analysis of NSTA's production by well; own analysis of BSEE's well tests]; both are upper bounds on the pressure downstream. The floor lies below the lowest lines of the Gulf's tests and just above the atmosphere.

**Proposal.** Keep both. Class the cap as data and the floor as a design choice, the lowest stage a separator can run at. With G7's higher bound for gas wells the cap is reached more often.

### N4 · SMP-68 · The volatile oils' log-uniform GOR, the dry gases' yield from 0.5, and the bounds of $x$

*Data in hand.*

**Evidence.**

- **Volatile oils.** Norway's drill-stem tests in McCain's volatile range spread over it about as a log-uniform draw does; the Gulf's few volatile-oil sands crowd its lower end [own analysis of Sodir's FactPages; own analysis of BOEM's sands atlas]. The class is over-weighted on purpose (D-4), and a flat draw covers it.
- **Dry gases' yield from 0.5 Sm³ per 10⁶ Sm³.** A floor is needed because the GOR must be finite; 0.5 is at the low end of the Gulf's gas wells' yields.
- **The bounds of $x$.** By Standing's bubble point, the Gulf's undersaturated oil sands exceed 0.85 in very few wells; 0.005 only separates a drawn undersaturation from the saturated wells.

**Proposal.** Keep all. Class the volatile oils' shape as a design choice supported by one region's tests, the yield's floor and 0.005 as design choices of the generation, and 0.85 as data.

### N5 and N6 · SMP-72, SMP-73 · The drain's liner constants

*Physics* for the liquid; the cap *stays a guess.*

**Evidence.** The model has a water viscosity, PVT-WAT-3: 0.49 mPa s at 330 K, 0.32 at 360 K, 0.22 at 400 K and 0.15 at 450 K. The liner's 0.5 mPa s is the model's water at a cold reservoir, and 1.5 to 3 times it at most. The density's clamp of 400 to 1100 kg/m³ spans live oils and brines and guards the closed form. The cap of 7 in is the largest of the oil wells' tubing sizes (SMP-75), since a production liner is not larger than that; nothing was found on liner sizes over wells. The viscosity enters Chen's friction factor weakly.

**Proposal.** Replace 0.5 mPa s by PVT-WAT-3 at $T_r$, so that the liner's water is the model's. Class the clamp and the cap as design choices of the build.

### N7 · SMP-114 · $\alpha = 1$ for a lumped heptanes-plus

*Published source.*

**Evidence.** Whitson and Brulé (2000): with $\alpha$ = 1 the gamma distribution is the exponential one (§5.3.2, with Eq. 5.14); "most reservoir fluids can be characterized with an exponential molar distribution ($\alpha$ = 1) without adversely affecting the quality of EOS predictions" (§5.3.2); and their recommended characterisation for a fluid without a measured distribution is $\alpha$ = 1 with Søreide's gravities, "choosing $C_f$ to match measured C₇₊ specific gravity" (§5.6). That is SMP-114's rule.

**Proposal.** Keep. Class it as a published recommendation, and cite §5.6 in SMP-114.

## 4. What stays a guess

After this triage, if the proposals are taken:

- SMP-64: the onshore surface temperature (until the fetch, G6).
- SMP-66: the onshore $p_\text{sep}$ (G7).
- SMP-71: the skin's mean, the high-angle wells' drainage, the anisotropy, the perforated share and the lower bound of $D$ (G13).
- SMP-74: nothing; the WLR's threshold becomes a design choice (G16).
- SMP-75: one size larger with probability 0.3 (G17).
- SMP-76 and SMP-88: the roughness by stage and its path (G18, G31).
- SMP-78: the onshore oils' lift shares (G19), and the balance point's 0.3 of $L$ (G20).
- SMP-79: the packer's offset, the lift's design drawdown and the least valve depth (G21, G22).
- SMP-80: the injected GLR's shape (G24).
- SMP-81: $k_e$'s lower end (G25).
- SMP-85: the form for supported wells (G29).
- SMP-86: the jumps and the gases' log-linear WGR (G30).
- SMP-98: the step's size and the share of new draws (G36).
- SMP-113: the Dirichlet weights, the uniform shapes, and the volatile oils' heptanes-plus ranges (G37).

## 5. What would settle more

- **A fetch:** an open climatology of the mean annual temperature at onshore wells' locations (G6).
- **In hand, to read:** Golan and Whitson (1991), §5.3, and Economides et al. (1994), the gas-lift chapter, for valve and packer depths and lift design drawdowns (G20 to G22); both are scans without a text layer. The Dutch survey stations for the S wells' end section and the holds (G3).
- **For Bjarne to obtain, only if the pilot shows the need:** Farshad and Rieke (2006), for G18. It bears on the roughness of new tubing; without it the stage split stands.
- **In the build's reach:** a table fluid's own $R_v(p)$ for the condensates' GOR (G12), NeqSim's $c_p$ of a natural gas at pressure (G26), and the material balance's $p_r(F)$ for oils under depletion (G29).
- **On R11's pilot:** the water cut by years on production after G30; the subsea and the gas wells' rates after G2 and G7; the share of `off_table` roots (G38).
- **No longer needed:** Swarbrick and Osborne (1998) for the lithostatic gradient (G4); Willhite (1967) and Alves et al. (1992) for $U_t$'s range (G25).

## 6. Unsure

- **The proxies.** G7, G9, G32, G35 and N3 rest on pressures upstream of an open choke, and G1, X1 and G11 on shut-in tubing pressures against a first year of tests that is not the initial state. Each is a bound or a proxy, as its entry says.
- **One region.** G1, X1, G3, G5, G11, G28, G30, G33 and G34 rest on the Gulf alone, and G32, G35 and half of G36 on the UK's gas fields. By Bjarne's rule the ranges proposed are pooled ranges with uniform or log-uniform shapes, and no shape is fitted to a region.
- **G7's branch by fluid** adds a rule. The need is measured on two regions' gas wells; whether oil wells elsewhere also start against higher lines than the Gulf's is not known.
- **G23** is read off figures of three examples.
- **G14's fit** is written from memory and checked against the book's table at $h_D$ = 100 only; Table 5-2 would check its second term.
- **G26's polynomials** are written from memory and checked against NIST's methane values only.
- **R6's pilot** (G17, G27, G38) predates the rework and the tables; the three results should be read again on R11's pilot.
