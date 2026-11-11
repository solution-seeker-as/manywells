# Sampler redesign — the rework's pilots (2026-11-06)

*The rework implements Bjarne's checkpoint rulings of 2026-11-06 in the draft's scripts and `sampling-draft.md`. Its pilots set or check the values the rulings left to them, and led to further rulings the same day: D-105 (one life for both datasets), D2 with D-62 (the gas wells' rates), and changes to D-31 (the lift-gas capacity), D-47 (a floor on the lift-gas set point) and D-83 (the redraws' step), all in `decisions.md`. R6's measurements are in `r6-measurements.md`; this file gives the rework's. Feature 022 and the tables are not yet in these pilots: gas wells flow in through the interim productivity index and every fluid is the black oil. R9's rerun replaces both.*

## The runs

The scripts are R6's (`sampler_prototype.py`, `sampler_prototype_report.py`, `pilot_by_well_type.py`) at the rework's commits, with R6's scale and seeds, on the Rust core and without the CasADi retry (H1).

| Run | Rules | Seed | Wells drawn | Accepted | Wall time | Processes |
|---|---|---|---|---|---|---|
| `sol-2` | the checkpoint's rulings (`00d782d`) | 2026 | 438 | 400 | 32 min | 11 |
| `nsol-2` | the same | 2027 | 113 | 100 | about 9 min | 11 |
| `nsol-2` | with R4's change kept for redraws (D-83) | 2027 | 110 | 100 | about 8 min | 11 |
| `nsol-2` | a harsher end of a life, `nsol-2` only | 2027 | 110 | 100 | about 9 min | 10 |
| `nsol-2` | an exhausted end, stopped at the floor, and the lift-gas floor | 2027 | 112 | 100 | about 9 min | 10 |
| `sol-2` | every ruling of the day (`2f6184e`): D-105, D2, D-31, D-47, D-83 | 2026 | 486 | 400 | 26 min | 11 |
| `nsol-2` | the same | 2027 | 114 | 100 | about 9 min | 11 |

The last two are the rework's result; the others show the steps to it.

## What the rulings changed

**The decline of a life (D-41, D-105).** The regions measure each well's mean rate in each fifth of its life over its first year's; D-41's target for the median of the last fifth, about a tenth, is set with the stopped wells [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]. Measured the same way on the core `nsol-2` wells, median by group:

| Run | Last fifth, all | Oil, support | Oil, depletion | Gas, support | Gas, depletion |
|---|---|---|---|---|---|
| R6 | 0.38 | 0.19 | 0.52 | 0.77 | 0.36 |
| The rulings, first run | 0.38 | 0.16 | 0.64 | 0.77 | 0.37 |
| Harsher end, `nsol-2` only | 0.20 | 0.06 | 0.33 | 0.76 | 0.20 |
| One life, exhausted end, D2 (final) | 0.12 | 0.03 | 0.22 | 0.67 | 0.14 |

The steeper Arps clock alone moved nothing: a life's last fifths sit at its end state, which R3's late draw left at about a third of the start's rate. Running each life towards an exhausted reservoir and stopping it at the economic floor gives D-41's tenth overall; 21% of the final lives end at the floor. Gas wells with pressure support still decline little: their reservoir pressure stays high, and water at a WGR of 1000 Sm³ per 10⁶ Sm³ at the end does not stop them. That is open.

**One life for both datasets (D-105).** `sol-2`'s wells and `nsol-2`'s samples agree by stage, by construction (core wells, medians):

| Stage | Drive | $p_r/p_i$, `nsol-2` / `sol-2` | Oils' water cut, `nsol-2` / `sol-2` |
|---|---|---|---|
| initial | depletion | 0.78 / 0.84 | 0.00 / 0.02 |
| initial | pressure support | 0.98 / 0.96 | 0.01 / 0.04 |
| mid-life | depletion | 0.44 / 0.42 | 0.08 / 0.06 |
| mid-life | pressure support | 0.92 / 0.90 | 0.69 / 0.70 |
| late | depletion | 0.24 / 0.25 | 0.18 / 0.16 |
| late | pressure support | 0.89 / 0.86 | 0.95 / 0.96 |

A `sol-2` well whose date falls after it stopped paying is dropped as not producible, as an abandoned well is absent from a well stock: 72 of the 486 drawn wells (49 without an operating point at the best rate, 23 below half the floor).

**Rates.** Core offshore oil wells (GOR at most 1068 Sm³/Sm³), median-sample oil equivalent:

| Run | P10 / P50 / P90 (Sm³/d) | Above 1018 Sm³/d | Mean |
|---|---|---|---|
| EIA 2024, the Gulf of Mexico's oil wells (Table B32) | – | 5.5% | 167 |
| R6 | 9 / 197 / 2321 | 22.7% | 822 |
| The rulings, first run | 22 / 215 / 2431 | 24.6% | 881 |
| Final | 6 / 94 / 1852 | 19.4% | 634 |

By well type in the first run, the subsea oil wells' median rose by half against R6's (298 against 207 Sm³/d) and their lower quartile tripled, as D-100's permeability by well type intended; they stay below the regions' subsea wells at the median, and the platform wells' upper tail lies above every region's [own analysis of BSEE's well tests, ANP's production by well and NSTA's production by well]. The narrower permeability spread (1.2 in place of 1.5) did not move that tail: the rate's spread comes mostly from the other draws.

Core gas and condensate wells, median-sample gas rate: P10 / P50 / P90 of 0.17 / 1.15 / 5.5 million Sm³/d, 84% above 270,000 Sm³/d, in the first run; 30,000 / 185,000 / 1.2 million Sm³/d, 38% above 270,000, after D2 (EIA's Gulf of Mexico gas wells: a mean of 32,700 Sm³/d, 2.3% above 270,000, a mature shelf's calendar-day averages). D2's measurement with feature 022's inflow is `gas_rates_d62.py`'s (D-62).

**Subsea separator pressures (D-15).** Subsea wells' $p_{s,\text{nom}}$ in more than 1000 m of water lie within the range of Petrobras' subsea trees [own analysis of Petrobras' 3W Dataset]; the truncation at 250 bar binds in none.

**The lift-gas capacity (D-31).** Clipped to [0.1, 3.5] kg/s, 32% of the lifted core wells sat at the floor and 15% at the cap; with the GLR truncated, 8.5% sit at the floor (6% of the oils) and none at the cap.

**The redraws of `nsol-2` (D-83, D-47).** With D-47's smaller set-point steps also used for redraws, 7 of 113 drawn wells reached the redraw limit, all lifted black oils failing at low lift-gas set points; with R4's change kept for redraws, 4 of 110; with the lift-gas set point floored at 0.1 of the capacity and the final rules, 5 of 114, with 3 dead dates.

## Outcomes, final runs

| | `sol-2` | `nsol-2` |
|---|---|---|
| Wells kept | 400 of 486 (82.3%; core 85.2%, tail 71.0%) | 100 of 114 (87.7%) |
| Not producible | 72 | 6 |
| Redraw limit | 9 (2.3% of the core's drawn wells) | 5 |
| Solver-failure limit | 2 | 0 |
| Best-rate solver failure | 3 | 0 |
| Dead date | – | 3 (2.6%) |
| Redraws per sample | 0.110 | 0.103 (operating), 0.091 (tests) |
| Choked samples | 30.3% | – |
| Lifted wells | 33.0% | 48% lifted at some point |

Above the draft's targets, to settle before R11: the redraw-limit drops (D-81's 1% per class), the dead dates (D-83's 0.5%: a finer life scan is the draft's fix) and the best-rate solver failures (D-55's 0.5%). The deep states of an exhausted reservoir fail more often than R3's late states did: the redraws per sample rose from 0.036 (R6) to 0.110.

## Cost

| | R6 | Final |
|---|---|---|
| Rust solve, median / mean / P95 (s) | 0.159 / 0.189 / 0.415 | 0.156 / 0.173 / 0.341 |
| CPU per accepted `sol-2` well (s) | 40.7 | 41.3 |
| Solves per accepted `sol-2` well | 215 | 238 |
| Projected CPU-hours, 10,000 `sol-2` wells | 113 | 115 |

The CasADi retry is gone (H1): it cost nothing here and recovered nothing in R6.

## Open for the smaller checkpoint

- Gas wells with pressure support decline little over a life (0.67 at the last fifth).
- The redraw-limit drops, the dead dates and the best-rate solver failures above their targets.
- The platform oil wells' upper tail above every region's.
- R9's rerun with feature 022's inflow (D-61's fit) and the tables measures what the interim mapping hid.
