# Sampler redesign — the beta's analysis (2026-11-14)

*The first full generation of the v2 datasets is a beta (D-149): `manywells-sol-2-beta1` and `manywells-nsol-2-beta1`, 10,000 wells of 200 samples each, generated on 2026-11-12 and 13 from `e23705d`, the merge of #46. This file holds what the analysis of the beta measured: the checks at generation (`datasets-draft.md`, Checks at generation), the drops and shifts that D-145 asks the datasheets to report, the four rule points left for after the beta (`preflight-measurements.md`, For after the beta) and a fifth that the analysis found, and options for the thresholds that D-145 left to the beta. Every measurement is `plans/evidence/beta1_checks.py`'s, on the datasets as written, their logs and the run's attempt log. Every number here is ManyWells' own, from synthetic wells. The rulings are Bjarne's; the recommendations are working assumptions until he rules.*

## Summary

- **The datasets are sound.** Both pass every check at generation, and 10,000 rows, 5000 of each dataset, solve again from their rows to the bit. 400 of them agree with the CasADi backend within the verifier's `tol_x`, 13 after a start from the core's root, where CasADi's own search missed it.
- **The limits thin the weak tail, little.** They drop 1.1% (`sol-2`) and 1.4% (`nsol-2`) of the wells that pass the filters, the weak ones, of low productivity and reservoir pressure, which flow at few of their draws. No drawn input's distribution moves by more than a KS distance of 0.010, and no decile of an input loses more than 9.7% and 7.3% of its wells.
- **Some groups lose more.** The redraw limit takes most of the drops, and it falls on wells that flow at few of their draws: `nsol-2`'s oils lifted by the static check (17% of those that pass the filters), `sol-2`'s oils lifted because they cannot flow naturally (5.6%), and the tail, onshore wells and lifted wells generally (2% to 3%).
- **Dead dates are handled.** In `nsol-2`, a dead date started the lift of 2.3% of the kept wells and cut the life of 0.4%; 4 drawn wells were dropped after five.
- **Five rule points for Bjarne** (section 8): the class labels of table fluids whose GOR moved; the `sol-2` tail's downstream pressure, which never reaches SMP-66's tail range; the tubing flag on 10% of the kept wells; the lift capacity's floor, held mostly by gas wells; and a new one, `nsol-2` dropping a natural oil that cannot flow at its start, which `sol-2` lifts.
- **Thresholds** (section 9): the recommendation is at most 2% dropped at the limits overall, and at most 10% in any decile of a drawn input and in any group of 100 wells or more. The beta meets them but for `nsol-2`'s static-check oils (17%), which a redraw limit of 3000 in place of 1000 brings to 3%, for 4% to 7% more CPU.

## 1. The run

| | `sol-2` | `nsol-2` |
|---|---|---|
| Started; finished | 2026-11-12 16:48; 2026-11-13 08:19 | 2026-11-12 16:48; 2026-11-13 10:35 |
| Processes | 10 | 10, then 20 from 08:19 |
| CPU hours; per drawn well | 150; 44 s | 192; 64 s |
| Restarts, errors | none | none |
| Code; environment | `e23705d`, clean; Python 3.14.6, numpy 2.5.3, scipy 1.18.1, Linux x86_64 | the same |
| Seed | 2026 | 2027 |
| Wells drawn; kept | 12,299; 10,000 (81.3%) | 10,816; 10,000 (92.5%) |
| Rows: `samples`, `geometry`, `pvt` | 2,000,000; 348,324; 5,520,986 | 2,000,000; 343,812; 11,521,148 |
| Rows of the logs: `drawn_wells`, `redraws` | 12,299; 390,082 | 10,816; 443,850 |
| Size (MB): `wells`, `samples`, `geometry`, `pvt` | 6.2, 355.6, 6.2, 235.5 | 7.4, 397.3, 6.1, 488.7 |
| Size (MB): `drawn_wells`, `redraws` | 8.0, 19.3 | 10.0, 21.4 |

The cost lies above R11's estimate of 110 to 130 CPU-hours per dataset, and per drawn well above the review's pilot (36 s and 47 s on 12 processes, `preflight-measurements.md`). The run kept 20 processes busy on the machine's 24 cores throughout, while the machine was in use, and `nsol-2`'s dead dates plan and solve a well's slots again (D-142); the cause was not measured further.

## 2. The checks at generation

| Check | `sol-2` | `nsol-2` |
|---|---|---|
| 10,000 wells, `ID` dense from 0; 200 samples each, slots 0 to 199 once each | yes | yes |
| Every well's survey; every table well's table | yes; 3065 tables | yes; 3108 tables |
| SMP-28's row check: no negative rate, fractions in [0, 1], every feature finite | 0 rows fail | 0 rows fail |
| v1's relations between the rates, and $F_\text{GAS} + F_\text{OIL} + F_\text{WAT} = 1$, largest relative difference | $4.8\cdot10^{-16}$ | $4.8\cdot10^{-16}$ |
| `configurations.check(wp, 'v2.0.0')`, through `sample_case`, at each well's first sample (and `nsol-2`'s first lifted one) | 10,000 built, 0 refused | 15,334 built, 0 refused |
| The split: train; test | 8042; 1958 | 8039; 1961 |
| Every stratum with $\lfloor n/5\rfloor$ test wells (96 strata); test share in strata of 50 wells or more | yes; 18.5% to 20% | yes; 18.8% to 20% |

## 3. The re-solve check

`resolve` draws 5000 rows of each dataset with a generator seeded by the dataset's seed, reads each row's well, survey and table back from the published files, and solves it again with `sample_case` and the Rust core, as a user would (D-92; the draft asks for 1000 rows and 100 with CasADi, and the solves are cheap enough for more).

| | `sol-2` | `nsol-2` |
|---|---|---|
| Rows re-solved by the Rust core | 5000 | 5000 |
| Operating point found; every feature the same to the bit | 5000; 5000 | 5000; 5000 |
| Solve (s): median, P95, largest | 0.14, 0.39, 2.5 | 0.15, 0.35, 2.0 |
| Rows solved with the CasADi backend too | 200 | 200 |
| — its own search reaches the core's operating point | 193 | 194 |
| — from the core's root, where its own search found no operating point | 6 | 6 |
| — from the core's root, where its own search took another stable root | 1 | 0 |
| Within `tol_x`, with the same choked state and regimes | 200 | 200 |
| Largest deviation, each over its `tol_x` scale: pressure, temperature, mass rate | $1.4\cdot10^{-8}$, $6.4\cdot10^{-9}$, $3.5\cdot10^{-8}$ | $5.7\cdot10^{-8}$, $1.8\cdot10^{-8}$, $1.1\cdot10^{-7}$ |
| CasADi's solve (s): median, P95, largest | 4.0, 52, 269 | 4.0, 34, 149 |

The row where CasADi's own search took another root is `sol-2`'s well 17 at slot 90, a natural black oil. Its root set has two stable roots, at bottomhole pressures of 57 bar (annular flow) and 115 bar (slug flow), and two unstable ones at 110 and 182 bar. The core finds all four and takes the lower stable root (SOL-6), which is the sample. CasADi's own search misses the root at 57 bar, so it takes the one at 115; from the core's root it converges to it within $10^{-15}$ and labels it stable. Its search's misses are known (`backlog.md`, 2.9); the core's operating point is the model's.

## 4. The drops at the limits (D-145)

The filters drop wells that would not be produced (SMP-101 to SMP-103: `not_producible`, `short_life`); the limits drop wells the generation could not finish (SMP-106: the redraw and solver-failure limits, a failed table, five dead dates, a failed best-rate solve). "Past the filters" counts the drawn wells less those the filters dropped; a failed table, which drops a well before its filter, counts there too.

| Outcome | `sol-2` | `nsol-2` |
|---|---|---|
| Accepted | 10,000 | 10,000 |
| `not_producible`; `short_life` | 2187; – | 559; 119 |
| `redraw_limit` | 84 | 108 |
| `best_rate_solver_failure` | 16 | 0 |
| `table_failed` | 10 | 21 |
| `solver_failure_limit` | 2 | 5 |
| `dead_date` | – | 4 |
| At the limits, of the drawn wells; of those past the filters | 0.91%; 1.11% | 1.28%; 1.36% |

**By group.** The share dropped at the limits of the wells past the filters, with the group's drawn wells:

| Group | `sol-2` | `nsol-2` |
|---|---|---|
| Black oil | 1.29% (5973) | 1.93% (5318) |
| Volatile oil | 1.13% (1633) | 1.71% (1398), 16 of the 23 a failed table |
| Gas condensate | 0.88% (2143) | 0.61% (1874) |
| Wet gas | 0.58% (2231) | 0.39% (1938) |
| Dry gas | 1.89% (319) | 0.45% (288) |
| Platform | 1.00% (6282) | 1.32% (5461) |
| Subsea | 0.77% (4206) | 0.91% (3769) |
| Onshore | 2.32% (1811) | 2.70% (1586) |
| Core | 0.73% (9859) | 1.21% (8624) |
| Tail | 2.88% (2440) | 2.03% (2192) |
| Natural | 0.30% (7606) | 0.17% (4793) |
| Lift from the start | 2.40% (2164) | 2.91% (1797) |
| Lift at a drawn time | 1.29% (1207) | 0.10% (1975) |
| Lift by the static check (oils) | 3.15% (832) | 17.4% (205; 16 of the 92 past the filters) |
| Lift at the end of natural flow (oils) | – | 3.35% (1583) |
| Lift where the well cannot flow naturally (`no_natural_flow`) | 5.59% (490) | 1.69% (237) |

- **The redraw limit takes most drops**: 84 of 112 in `sol-2` and 108 of 138 in `nsol-2`. As the review found (`preflight-measurements.md`), these are wells that flow at few of their draws, and lifted oils are most of them.
- **`nsol-2`'s static-check oils** remain the most selective group, though far less than before the dead-date rule (15 of 27 in the review's 1000 wells): 14 at the redraw limit and 2 tables. The group is small: 76 kept wells, 0.8%.
- **The volatile oils' tables** fail at 1.1% of `nsol-2`'s table volatile oils (16 of 1398), as D-143's rebuild measured.

**Which wells the limits drop.** The weak ones. Against the kept wells, by the KS distance and the place of the dropped wells' median among the kept:

| | `sol-2` (112 dropped) | `nsol-2` (138 dropped) |
|---|---|---|
| First | Vogel's `w_l_max`: 0.56, at the kept wells' 5th percentile | the water at the life's end: 0.56, 12th percentile |
| Second | the nominal reservoir pressure: 0.52, 9th (103 bar against 283) | the reservoir's pressure at the end: 0.51, 28th (0.24 of $p_i$ against 0.74) |
| Then | the lift capacity 0.39, 16th; the separator's factor 0.34, 26th; the productivity index 0.34, 18th | the start's pressure 0.43, 21st; the lift capacity 0.35, 21st; the productivity index 0.36, 21st |

They are wells of low productivity and low reservoir pressure, and in `nsol-2` mostly wells under depletion whose reservoir runs far down: wells that flow at few of their draws, which the redraw limit drops. (`nsol-2`'s list also holds its failed tables, which have no GOR axis.)

**How much that moves the kept wells.** The KS distance between the wells past the filters and the kept can be no larger than about the share dropped among the wells that have the input, so it measures the size of the effect, not its selectivity; the share dropped in each decile of an input measures how hard the tails are thinned:

| | `sol-2` | `nsol-2` |
|---|---|---|
| Largest KS distance, of 99 and 103 inputs | 0.0098 (`w_lg_max`) | 0.0087 (`glr_cap`) |
| Inputs with KS above 0.02 | 0 | 0 |
| Largest change of a label's share | 0.52 points (`lift` natural) | 0.77 points (`lift_start_rule` `drawn_time`) |
| Largest share dropped in a decile of an input (with 1000 wells or more) | 9.7%: the lowest tenth of `w_lg_max` | 7.3%: the lowest tenth of `w_lg_max` |
| — the lowest tenth of the nominal reservoir pressure; of Vogel's `w_l_max` | 5.8%; 7.7% | – |
| Inputs with a decile above 5%; above 10% | 5; 0 | 4; 0 |

The limits move no input's distribution by more than one percentage point, and take less than a tenth of any decile: they thin the weak tail without cutting it off.

**The filters' shift**, which is intended (a well that would not be produced is not a sample of produced wells), for the datasheet: from all drawn wells to the kept, `sol-2`'s largest KS distances are the nominal reservoir pressure 0.100 (kept wells at higher pressure), `w_lg_max` 0.095, `glr_cap` 0.082, the design rate 0.077 and the nominal GOR 0.068, with 14 inputs above 0.05; the labels move by up to 6.5 points (`drive` depletion), the producibility filter dropping depleted wells. `nsol-2`'s filter acts at the start only, and its largest KS distance is 0.038 (`glr_cap`), none above 0.05; its largest change of a label is the static check's share of the lift rules, 2.1 points down (section 8.5).

## 5. Dead dates (`nsol-2`, D-142)

| | Wells |
|---|---|
| Dead dates handled (in all drawn wells): lift starts; life ends | 239; 91 |
| Drawn wells with a dead date | 286 (2.64%) |
| Kept wells with a dead date: one, two, three, five | 275 (2.75%): 259, 10, 3, 3 |
| Kept wells whose lift a dead date started (`no_natural_flow`) | 233 (2.33%) |
| Kept wells whose life a dead date cut ($T_\text{end} < T_\text{scan}$) | 44 (0.44%) |
| The share of the scan's life cut: median, P95, largest | 8.4%, 69%, 86% |
| Dropped as `dead_date` after five | 4 |

The kept lives end at 1.0, 3.9, 7.4, 14.1 and 25.5 years (P5, P25, P50, P75, P95), of drawn lives of 3 to 30 years.

## 6. The redraws and the samples' conditioning

| | `sol-2` | `nsol-2` |
|---|---|---|
| Redraws of the kept wells | 289,049 | 305,471 |
| Per sample; share of the attempts redrawn | 0.145; 12.6% | 0.153; 13.3% |
| By outcome and reason | 289,039 without an operating point (no root), 10 solver failures | 305,447 without an operating point (no root), 24 solver failures |
| Kept wells with no redraw | 64% | 51% |
| Redraws per sample by well: P90, P99, largest | 0.41, 2.2, 5.0 | 0.38, 2.5, 4.9 |

**The shift between the draws and the samples.** The KS distance between every attempt of the kept wells (samples and redraws) and the samples, by input, and where the samples' median lies in the attempts' distribution:

| Input | `sol-2` | `nsol-2` |
|---|---|---|
| `PRES` | 0.043 (0.54) | 0.049 (0.55) |
| `WGL`, where gas is injected | 0.048 (0.54) | 0.057 (0.54) |
| `CHK` | 0.019 (0.52) | 0.031 (0.53) |
| `GOR` | 0.017 (0.51) | 0.017 (0.50) |
| `WLR` | 0.013 (0.49) | 0.014 (0.51) |
| `PDC` | 0.005 (0.50) | 0.008 (0.49) |
| `WLMAX` (Vogel wells) | – | 0.037 (0.53) |
| `KL`, `HTC`, `ROUGH`, `FORCHA` | – | at most 0.011 |

The samples lean to a higher reservoir pressure, more lift gas and a more open choke than the draws: the conditioning on the well flowing that the datasheets describe. In `sol-2` a redraw draws the slot again from scratch, so its attempts follow the drawn distribution; in `nsol-2` it keeps the date and redraws the controls and the separator pressure.

## 7. The model's ranges, the tables' grid and several roots

**Correlations outside their ranges** (`specs/model/validity.md`), on the 5000 re-solved samples of each dataset: the share of the samples whose input, or a grid point of whose operating point, lies outside the published range. The black oil's correlations count the black-oil wells only (black oils, wet and dry gases); Forchheimer's range the wells with that inflow.

| Correlation | Input | Published range | `sol-2` | `nsol-2` |
|---|---|---|---|---|
| Beggs and Robinson (samples of black-oil wells: 3497; 3423) | API gravity | 16 to 58 | 0% | 0% |
| | $p$ | up to 5250 psig | 11.5% | 11.9% |
| | $T$ | 70 to 295 °F | 24.2% | 25.2% |
| | $R_{so}$ | 20 to 2070 scf/STB | 8.3% | 8.3% |
| Abdul-Majeed, dead oil | $T$ | 15.6 to 54.4 °C | 88.6% | 88.2% |
| | API gravity | 15 to 50 | 16.1% | 17.3% |
| Abdul-Majeed, live oil | $R_{so}$ | 3 to 208 Sm³/Sm³ | 24.0% | 23.2% |
| Sutton | $\gamma_g$ | 0.571 to 1.679 | 0% | 0% |
| Dranchuk and Abou-Kassem | $p_{pr}$; $T_{pr}$ | 0.2 to 30; 1.0 to 3.0 | 5.6%; 0% | 5.8%; 0% |
| Lee, Gonzalez and Eakin | $p$; $T$ | 100 to 8000 psia; 100 to 340 °F | 5.1%; 42.4% | 5.4%; 45.2% |
| Hasan et al. (every well) | gas–liquid ratio, lift gas included | 104 to 5615 scf/STB | 36.6% | 43.5% |
| | liquid rate | 100 to 27,760 STB/d | 23.6% | 28.2% |
| | water cut | 0 to 98% | 5.2% | 3.6% |
| | bottomhole pressure; measured depth | 108 to 13,755 psig; 498 to 9342 m | 0.4%; 0% | 0.4%; 0% |
| Chen | Reynolds number | 4000 to $4\cdot10^{8}$ | 11.0% | 10.7% |
| | $\varepsilon/D$ | $5\cdot10^{-7}$ to 0.05 | 0% | 0% |
| Forchheimer, $p^2$ form (samples: 1722; 1913) | $p_r$ | below about 138 bar | 79.6% | 83.3% |

- Every rule here is *extrapolate* (`validity.md`); none refuses or clamps.
- The temperatures leave the surface tension's fitted range in most black-oil samples, as `validity.md` expected (note 4), and the gas viscosity's lower end at cold wellheads.
- Hasan et al.'s gas–liquid ratio and liquid rate are left by the gas wells, whose liquid is small. A water cut above 0.98 occurs in gas and condensate wells only (4.6% and 3.9% of all samples); no oil sample lies above 0.98, the model's cap.
- 80% of the Forchheimer wells' samples lie above 138 bar, where the pressure-squared form is conservative (note 12, accepted by Bjarne on 2026-11-09).

**Points off a table's grid** (PVT-TAB-2, D-147), from the attempt log: in `sol-2`, 1592 samples (0.08% of all, 0.26% of the table wells'), in 20 wells, all volatile oils; in `nsol-2`, 2224 samples (0.11%, 0.36%), in 66 wells, 2205 of them volatile oils' and 19 condensates'. The volatile oils heat above their table's top temperature, as the review expected.

**Several stable roots**: 16,428 samples of `sol-2` (0.82%) and 25,891 of `nsol-2` (1.29%) have more than one stable root; the operating point is the one with the lowest bottomhole pressure (SOL-6).

**Small rates.** No floor applies to a sample's rate (SMP-106): 4.9% of `sol-2`'s samples and 6.7% of `nsol-2`'s take less than 0.1 kg/s from the reservoir, v1's floor, and 0.03% and 0.04% less than 1 g/s.

## 8. The rule points

### 8.1 The class of a table fluid whose GOR moved (SMP-115, D-127)

**What happens.** A volatile oil or condensate is drawn in its class by its GOR (SMP-55, McCain's limits on the initial producing GOR) and built by NeqSim. Where the flash puts it in two phases at $p_i$, SMP-115 moves its GOR in its class's direction until its saturation pressure is $p_i$ (`gor_moved`), or takes the class's phase in equilibrium at $(p_i, T_r)$ (`equilibrium_phase`). The fluid keeps its drawn class. The review compared the table's single-stage GOR, `R_si`, with the class; the class's limits are on the GOR as the field reports it, which is the built fluid's `field_gor` (SMP-113), and the comparison below uses it.

| Kept table wells, by the class of their field GOR | `sol-2` | `nsol-2` |
|---|---|---|
| Volatile oils with a black oil's GOR | 356 of 1483 (24.0%): 286 `gor_moved`, 64 `equilibrium_phase`, 6 `matched` | 315 of 1324 (23.8%): 262, 42, 11 |
| Condensates with a wet gas's GOR | 100 (91 `equilibrium_phase`, 9 `gor_moved`) | 130 (110, 20) |
| Condensates with a dry gas's yield | 12, all `equilibrium_phase`; field GOR up to $5\cdot10^7$ | 6 |
| Condensates with a volatile oil's GOR | 5 `matched`, at the class's edge | 1 |
| Condensates outside their class, in all | 122 of 1582 (7.7%), 5 of them with no field liquid | 141 of 1784 (7.9%), 4 with no field liquid |
| Kept wells whose class would change | 478 (4.8%) | 456 (4.6%) |

A moved volatile oil's field GOR is at the median 0.46 of the drawn one (`sol-2`; 0.50 in `nsol-2`); a moved condensate's 10.5 times (5.3), and up to $1.4\cdot10^4$ times ($1.5\cdot10^7$). Along a well's life its producing GOR leaves the initial class as it would in the field (38% of the volatile oils' samples lie above the class), which is the samples' state, not the label's subject.

- **Physically**, a moved volatile oil is an oil saturated at $p_i$ with a black oil's GOR: it is a black oil, described by an equation of state instead of the black-oil correlations. A condensate moved to the equilibrium vapour is a leaner gas, a wet gas, and at the extreme a dry gas.
- **For the model** nothing changes: the table is the fluid's, and the label enters no equation.
- **For the ML tasks** the label misleads: a quarter of the wells labelled `volatile_oil` carry a black oil's GOR, and `fluid_type`, which the split stratifies by, carries the same label.

**Options.**

- **A. Label the fluid that was built** (recommended): `fluid_class` and `fluid_type` by McCain's limits on the built fluid's field GOR, with `pvt_model` still `table`. The drawn class remains readable from `R_si_drawn`. The classes' shares move: `sol-2`'s volatile oils from 14.8% to 11.3% of the kept wells and its condensates from 15.8% to 14.6%; `nsol-2`'s from 13.2% to 10.1% and from 17.8% to 16.4%. The volatile oils stay at about the Goals' floor of a tenth; raising their drawn share (SMP-55's 0.10) would restore the margin.
- **B. Keep the drawn class**, as now, and say so in the datasheets: the label is the draw's intent.
- **C. Redraw a fluid whose flash moves it out of its class**: the labels and shares hold, but the compositions are then filtered by $p_i$ and $T_r$, which D-65 chose moving to avoid.

### 8.2 The `sol-2` tail's downstream pressure (SMP-66, SMP-89, SMP-92)

**What happens.** SMP-66 gives a tail well $p_{s,\text{nom}} = \text{LogU}(1.5, 120) + \Delta p_\text{fl}$ for coverage. SMP-92 puts a `sol-2` tail well at the end of its life, whatever its stage label, and SMP-89 scales its $p_{s,\text{nom}} - \Delta p_\text{fl}$ by the separator's factor there, $\min(r_\text{mid}, r_\text{late}) \le 0.6$. So SMP-66's tail form never reaches `sol-2`:

| Kept `sol-2` wells' $p_{s,\text{nom}}$ (bar) | At 1.5 bar | P50 | P95 | Largest |
|---|---|---|---|---|
| Core, platform (4785) | 0.04% | 29.9 | 77.9 | 120 |
| Core, onshore (780) | 2.2% | 10.9 | 34.6 | 68.9 |
| Tail, platform (636) | 25.6% | 4.7 | 33.5 | 63.9 |
| Tail, onshore (568) | 29.8% | 3.4 | 31.5 | 52.7 |
| Tail, subsea (516) | 0% | 45.5 | 132 | 224 |
| SMP-66's tail form, without $\Delta p_\text{fl}$ | – | 13.4 | 96.5 | 120 |

The tail, meant to widen the coverage, is narrower than the core at high downstream pressures on platforms. `nsol-2`'s tail wells start at SMP-66's draw and step down along the life, so their samples reach 131 bar (P95 79 to 82 bar on platforms and onshore).

- **Physically**, the downstream pressure sets the choke's pressure ratio, the wellhead pressure and with them the drawdown; a high line pressure on a depleted well is the low-drawdown, near-unloading corner of the operating envelope, the corner a coverage tail exists for.
- **For the ML tasks**, `sol-2` has no tail sample above 71 bar downstream on a platform or onshore, and a quarter of its non-subsea tail wells share one value of $p_{s,\text{nom}}$.

**Options.**

- **A. A `sol-2` tail well takes SMP-66's tail draw unscaled** (recommended): the separator's factor applies to the core only in `sol-2`, and `nsol-2` is unchanged. The tail then covers 1.5 to 120 bar plus the flowline. More tail wells may fail the producibility filter at high line pressures with their depleted end-of-life state; the pilot would measure it.
- **B. Scale by the drawn stage's factor** (1, $r_\text{mid}$ or the late factor): ties $p_{s,\text{nom}}$ to the stage label, though the date stays the life's end.
- **C. Keep**, and say so in the datasheet.

### 8.3 The tubing flag (D-25, D-100)

**What happens.** SMP-75 sizes a well's tubing for its design rate at completion; where even the largest size (7 in for oils, 9-5/8 in for gas wells) carries less at the erosional velocity, the well takes the largest and is flagged `design_rate_above_tubing_capacity`. D-100's ruling kept the flag as a check, a redraw to follow "only if it stays above a few percent".

| Flagged | `sol-2` | `nsol-2` |
|---|---|---|
| Drawn; kept | 8.98%; 10.4% | 9.33%; 9.99% |
| Kept, core: black oil, volatile oil, gas, condensate | 9.2%, 19.2%, 2.1%, 3.1% | 10.0%, 20.7%, 1.4%, 2.0% |
| Kept, tail: black oil, volatile oil, gas, condensate | 26.5%, 30.3%, 13.0%, 10.5% | 24.0%, 30.1%, 9.5%, 8.9% |
| Kept: platform, subsea, onshore | 7.1%, 16.8%, 8.7% | 7.3%, 15.3%, 6.7% |
| Share of the flagged wells kept; of the others | 94.6%; 80.0% | 99.0%; 91.8% |

**The samples' velocities.** On the re-solved samples (section 3), API RP 14E's $C = v_m\sqrt{\rho_\text{ns}}/1.22$, with the mixture velocity and the no-slip density, at each sample's grid point where it is largest (SMP-75's rule, from [WP §6]):

| Samples above | $C$ = 100 | $C$ = 150 |
|---|---|---|
| Flagged wells, `sol-2` (520); `nsol-2` (485) | 41.0%; 40.0% | 27.7%; 24.7% |
| The other wells, `sol-2` (4480); `nsol-2` (4515) | 12.9%; 11.3% | 4.6%; 4.0% |
| All | 15.8%; 14.0% | 7.0%; 6.0% |

- **Physically**, a flagged well is a productive well whose design rate the largest tubing cannot carry below the erosional velocity; in the field it would be choked back or completed larger (a monobore), and its open-choke operating points would not be run.
- **For the model** nothing breaks: the model has no erosion, and the friction at high velocity is in the momentum balance.
- **For the ML tasks**, the flagged wells, a tenth of the samples, hold two fifths of those above $C$ = 150 (144 of 348 and 120 of 300 of the re-solved ones); a user can filter by the flag, and by the velocity only with a re-solve, since the samples hold no velocities.

**Options.**

- **A. Keep the flag as a label** (recommended), with its share and the samples' $C$ in the datasheets. It keeps the productive wells, which L5 kept under coverage first (the platform oil wells' upper tail).
- **B. Redraw a flagged well**, as D-100 foresaw above a few percent: a tenth of the wells, the most productive ones, leave, and the productivity's upper tail with them (against L5).
- **C. Hold the samples below the erosional velocity**: a draw whose operating point exceeds it is redrawn, as one without an operating point is. SMP-75 already caps the design rates at the tubing's erosional rate, but not the samples' chokes. A new rule, which would condition every well's samples on their velocity as the redraws condition them on flowing, and remove about 6% to 7% of the samples' draws.

### 8.4 The lift capacity's floor (SMP-80, D-31)

**What happens.** SMP-80 scales the lift-gas capacity to the design liquid rate, $w_{lg,\max} = k_q\thinspace\text{GLR}_ \text{cap}$, with the GLR log-uniform over the part of [30, 1000] Sm³/Sm³ that gives a capacity within [0.1, 3.5] kg/s. Where no part does, because $1000\thinspace k_q < 0.1$ kg/s, the capacity is 0.1 kg/s.

| Kept lifted wells | `sol-2` | `nsol-2` |
|---|---|---|
| At 0.1 kg/s; at 3.5 kg/s | 419 of 3620 (11.6%); 0 | 669 of 5334 (12.5%); 0 |
| — of the lifted gas wells; condensates | 107 of 171 (63%); 64 of 163 (39%) | 158 of 242 (65%); 75 of 198 (38%) |
| — of the lifted black oils; volatile oils | 190 of 2625 (7.2%); 58 of 661 (8.8%) | 357 of 3996 (8.9%); 79 of 898 (8.8%) |
| — of the onshore lifted wells | 181 of 484 (37%) | 258 of 613 (42%) |
| GLR at capacity of the wells at 0.1: median, P95 (Sm³/Sm³) | 2737; 82,000 | 3113; 90,000 |
| Their lifted samples' injected GLR, median: oils, gas wells | 910; 1470 | 911; 3506 |
| The other lifted wells' samples, median: oils, gas wells | 156; 60 | 146; 123 |
| Liquid of the oils at 0.1, median (Sm³/d); lift gas, median (Sm³/d) | 5.2; 5500 | 5.4; 5400 |
| Formation gas of the gas wells at 0.1, median (Sm³/d); lift gas over formation gas, median and P95 | 28,800; 0.15 and 5.4 | 24,600; 0.22 and 5.7 |

- **Physically**, the floor binds where the well has little liquid: weak gas and condensate wells, whose lift unloads liquid, and small oil wells, mostly onshore, of a few Sm³/d. The oils' injected GLR, about 900 at the median, lies above the published optima of 430 to 800 ([WP §9]). The gas wells inject a fifth of their formation gas at the median, and five times it in a twentieth of their samples.
- **For the ML tasks**, the point mass sits in a well's parameter; the samples' `WGL` is continuous below it. For gas wells a GLR on the liquid says little.
- **The flag `lift_capacity_at_bound`** marks only 238 of the 419 (`sol-2`) and 271 of the 669 (`nsol-2`): `flags` holds the initial-stage draw's flags, and a well whose lift starts later takes its design from the late-stage draw (`life.py`), whose flags are not kept.

**Options.**

- **A. Keep the floor, and flag the design the well uses** (recommended): a bookkeeping fix, so that the flag marks every well at the bound.
- **B. Lower the floor**, so that the GLR's truncation works for small wells: the floor of 0.1 kg/s was set with subsea wells' injection rates (checkpoint F2), and onshore wells are absent from that data, so a lower one would be a guess until small wells' injection rates are found.
- **C. Scale a gas well's capacity to its gas rate**, since its lift unloads liquid: a new rule.

### 8.5 `nsol-2` drops a natural oil that cannot flow at its start (SMP-101, SMP-103)

**What happens.** SMP-103 lifts a `sol-2` natural oil that cannot flow at its date (`no_natural_flow`, D-101: operating practice), and drops it only if it fails with lift too. In `nsol-2`, SMP-101's rule 3 starts lift where a natural well fails at a scan point $i > 0$, and D-142 where a date is dead; at the first point the well is dropped as `not_producible`.

| | `sol-2` | `nsol-2` |
|---|---|---|
| Oils natural at the start (or date) that do not pass | lifted (`no_natural_flow`): 338 of 490 then kept (69%) | 184 dropped (2.7% of the drawn oils) |
| — of the oils whose lift the static check starts | – | 113 of 205 (55%) |

The static-check oils are the natural oils whose flow is weakest, so `nsol-2` loses most of them before their life starts.

**Options.**

- **A. Take SMP-103's `sol-2` rule at `nsol-2`'s first point** (recommended): a natural oil whose first point does not pass takes its life's lift design from $t = 0$ (`lift_start_rule` `no_natural_flow`) and is solved again, dropped only if it then fails. By `sol-2`'s 69%, about 120 more of every 10,000 drawn wells would be kept.
- **B. Keep**, and say so in the datasheet.

## 9. Thresholds for the final datasets (D-145)

D-145 replaced SMP-106's targets with a report, thresholds to be set from the beta. Four kinds are possible, each on the wells past the filters:

| Threshold | Beta, `sol-2`; `nsol-2` | Holds? |
|---|---|---|
| **A. Overall**: at most 2% dropped at the limits | 1.11%; 1.36% | yes |
| **B. The datasets' shift**: no drawn input's KS distance above 0.02, and no label's share moved by more than 1 point | 0.0098, 0.52 points; 0.0087, 0.77 points | yes |
| **C. The tails**: no decile of a drawn input (with 1000 wells or more) loses more than 10% | 9.7%; 7.3% | yes, `sol-2` narrowly |
| **D. The groups**: no group of 100 drawn wells or more loses more than 10% | 5.6% (`no_natural_flow`); 17% (static check) | `sol-2` yes; `nsol-2` no |

- **A bounds B.** With at most 2% dropped, no input's distribution can move by much more than 0.02, unless only a few wells have the input; B adds little to A.
- **C and D catch selectivity**, which A and B cannot: C by the inputs' values, D by the labels. The old targets of 1% per class stood in for them (D-145).

D fails on one group of the beta, `nsol-2`'s 205 static-check oils, of which 16 of the 92 past the filters are dropped, 14 of them at the redraw limit. `beta1_checks.py limit` generates the two most selective groups' wells at the redraw limit again with a limit of 3000 in place of 1000:

| Wells dropped at the redraw limit | `nsol-2`, static check (14) | `sol-2`, `no_natural_flow` (18) |
|---|---|---|
| At a limit of 3000 | 13 kept, 1 dropped after five dead dates | 11 kept, 5 at the new limit, 2 at the slot limit |
| CPU time, at 1000 (the run); at 3000 (two measurements) | 2444 s; 6563 s and 9469 s | 3711 s; 5064 s and 7877 s |
| The group's drops past the filters, now; at 3000 | 17%; 3% | 5.6%; 2.5% |

The outcomes are the same in both measurements; the CPU times differ with the machine's load (8 processes, then 14). The wells it keeps are conditioned harder: `sol-2`'s needed 1000 to 2900 redraws, 5 to 14 per sample. Applied to every well at the limit (13,866 s and 24,859 s of the run's CPU), the cost would rise by about 13 to 24 CPU-hours, 4% to 7% of the run, an extrapolation from these 32 wells, less the draws that the kept wells save.

**Recommended: A, C and D as the thresholds, at 2%, 10% and 10%, with the redraw limit raised to 3000 for the final run**, so that the weak wells' groups are covered (coverage first) at a small cost. The redraw limit is SMP-106's bound and needs sign-off. The alternatives: D at 20%, which the beta meets as it is; or D reported without a threshold, as B is.

## 10. Smaller findings

- **CasADi's search** found no operating point in 12 of the 400 rows and took another stable root in one; from the core's root it reaches the core's operating point in each (section 3; `backlog.md`, 2.9). The Rust core is the generator's backend, so this bears on the CasADi backend's search, not on the datasets.
- **The datasheets' numbers**: section 11 gives each placeholder's value for the beta. D-149 has the datasheets filled for the beta first and again for the final datasets; the recommendation of 2026-11-13 was to fill them once, from the final run, with this script, and keep the beta's numbers here.
- **The life stage of a `sol-2` tail well** labels the drawn stage, while its date is the life's end (SMP-92, `docs/datasets.md`): a tail well labelled `initial` is at its end-of-life state.

## 11. The datasheets' numbers for the beta

| Placeholder | `sol-2` | `nsol-2` |
|---|---|---|
| Rows of `geometry` and `pvt`; files' sizes | section 1 | section 1 |
| Wells drawn; rows of `redraws` | 12,299; 390,082 | 10,816; 443,850 |
| The tail's share of the accepted wells | 17.2% | 18.4% |
| Share of the drawn wells kept | 81.3% | 92.5% |
| Accepted by fluid type: black oil, volatile oil, condensate, gas | 51.9%, 14.8%, 15.8%, 17.5% | 48.9%, 13.2%, 17.8%, 20.0% |
| Accepted by well type: platform, subsea, onshore | 54.2%, 32.3%, 13.5% | 51.6%, 35.0%, 13.4% |
| Accepted by geometry class: vertical, slightly deviated, deviated, highly deviated | 23.8%, 25.4%, 24.9%, 25.9% | 24.8%, 24.8%, 25.0%, 25.4% |
| `T_end` (years): P5, P25, P50, P75, P95 | – | 1.0, 3.9, 7.4, 14.1, 25.5 |
| Wells in train; in test | 8042; 1958 | 8039; 1961 |
| Samples outside each correlation's range | section 7 | section 7 |
| Samples with a point off its table's grid | 1592 (0.08%) | 2224 (0.11%) |
| Redraws per sample, by outcome and reason; the shift | section 6 | section 6 |
| Samples with several stable roots | 0.82% | 1.29% |
| The re-solve check | section 3 | section 3 |
| The checks at generation | section 2 | section 2 |
| The seed | 2026 | 2027 |
| The run's dates, machine, processes, CPU hours | section 1 | section 1 |
| Drops at the limits by group; the shift of each drawn input | section 4 | section 4 |
| Dead dates: lifts started; lives cut, and the share cut | – | section 5 |
| Wells dropped by outcome; share of the attempts redrawn | section 4; 12.6% | section 4; 13.3% |

## 12. For Bjarne's ruling

1. The class label of a moved table fluid (8.1): A, B or C; with A, whether to raise the volatile oils' drawn share.
2. The `sol-2` tail's downstream pressure (8.2): A, B or C.
3. The tubing flag (8.3): A, B or C.
4. The lift capacity's floor (8.4): A, B or C.
5. `nsol-2`'s natural oils that cannot flow at the start (8.5): A or B.
6. The thresholds for the final datasets (9), and whether the redraw limit rises from 1000 to 3000.
7. Whether the datasheets are filled for the beta as well as for the final datasets (10).

Taken together, the recommendations change `specs/sampling.md` in SMP-52 (labels), SMP-89 or SMP-92 (the tail's separator), SMP-101 and SMP-103 (the start), SMP-106 (the thresholds and the redraw limit) and the flags' bookkeeping, and need a pilot before the final run: step 3 of the plan of 2026-11-13.
