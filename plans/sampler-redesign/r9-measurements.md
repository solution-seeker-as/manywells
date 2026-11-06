# Sampler redesign — R9's pilots (2026-11-06)

*R9 brings feature 022's gas inflow and the per-well NeqSim tables into the prototype (`plans/evidence/README.md`, R9's integration). These pilots rerun the rework's (`rework-measurements.md`) with both, on the same seeds and draw indices, so that the differences are R9's alone: gas and condensate wells flow in by Forchheimer's law fitted as D-61 says, and volatile oils and condensates solve on their tables (D-63 to D-68), with D-106's repair of the tables' $p/Z$ (needs Bjarne's sign-off).*

## The runs

The scripts are the rework's at the branch's R9 commits, on the Rust core and without the CasADi retry (H1). The tables are built first, for the draw indices the pilots reach.

| Run | Seed | Draw indices | Result | Wall time | Processes |
|---|---|---|---|---|---|
| tables for `sol-2` (`table_fluids.py wells/build/axes`) | 2026 | 0 to 559 | 176 table wells (90 condensates, 86 volatile oils; 32 from published examples), all built and passing PVT-TAB-1 | 2.5 min | 8 |
| tables for `nsol-2` | 2027 | 0 to 149 | 49 table wells, 48 built; 1 volatile oil refused | 2.5 min | 8 |
| `sol-2` | 2026 | 0 to 486 | 400 kept of 487 | 92 min | 11 |
| `nsol-2` | 2027 | 0 to 114 | 100 kept of 115 | 38 min | 11 |

**The tables.** A table set takes 3.3 s (`sol-2`) or 9.7 s (`nsol-2`) to build at the median, with 2 GORs on most `sol-2` axes and 2 to 29 on `nsol-2`'s. D-106 dropped 33 pressures (each at one GOR) from 19 of the 176 `sol-2` sets, and 61 from 17 of the 48 `nsol-2` sets; one `nsol-2` volatile oil's set has $p/Z$ falling at 3 of 50 pressures at one GOR, above D-106's 5%, and the well is `table_failed`. The tables move the saturation pressure from R3's correlations by a factor of 0.46 to 2.2 (P10 to P90; median 0.99) and put it lower against the initial pressure: $p_\text{sat}/p_i$ has a median of 0.80 on the tables against R3's 0.93 (condensates) and 0.98 (volatile oils). So 53 of the 154 `sol-2` table wells change their saturation label, both ways.

## What R9 changed

**Outcomes.** The black oils solve as in the rework, bit for bit (the same 49,343 attempts and outcomes in `sol-2`), as they should. On the same draw indices:

| Class | `sol-2` kept, R9 / rework | `nsol-2` kept, R9 / rework | Attempts without an operating point, `sol-2`, R9 / rework |
|---|---|---|---|
| black oil | 188 / 188 | 41 / 41 | 20.9% / 20.9% |
| volatile oil | 67 / 66 | 15 / 16 | 12.5% / 12.3% |
| gas condensate | 67 / 69 | 18 / 18 | 2.1% / 3.6% |
| wet gas | 70 / 70 | 23 / 23 | 4.8% / 3.7% |
| dry gas | 7 / 7 | 3 / 3 | 12.6% / 12.2% |

Five volatile oils, two condensates and two wet gas wells change outcome in `sol-2`: the two condensates lose their operating point at the best rate, and the others move both ways. The volatile oils' 12 solver failures in the rework's attempts fall to none. In `nsol-2`, one volatile oil is `table_failed`, one becomes `short_life` (not producible before) and one becomes a dead date, one volatile oil's dead date is kept, and one wet gas well's dead date becomes a redraw-limit drop.

**Gas rates.** Median-sample gas rate of the wells kept in both runs, R9 over the rework, median and P10 to P90:

| Class | `sol-2` | `nsol-2` |
|---|---|---|
| gas condensate | 0.72 (0.48 to 0.94); oil 0.53 | 0.72 (0.54 to 0.95); oil 0.54 |
| wet gas | 0.91 (0.74 to 1.03) | 0.98 (0.80 to 1.07) |
| dry gas | 0.96 | 1.06 |
| volatile oil | 1.12 (0.44 to 1.64); oil 1.19 | 1.09 (0.86 to 1.97); oil 1.18 |

Forchheimer's inflow alone lowers the wet gas wells' rates by a tenth (the dry and wet gases have no table). The condensates lose a quarter of their gas and half their oil: the rest of that comes with their tables, whose saturation pressures, densities and liquid yields differ from the interim black oil's; which of these dominates is not measured. Core gas and condensate wells, median-sample gas rate P10 / P50 / P90: 25,000 / 154,000 / 973,000 Sm³/d, 32.4% above 270,000 Sm³/d (the rework: 30,000 / 185,000 / 1.2 million, 38.0%; EIA's Gulf of Mexico gas wells: a mean of 32,700 Sm³/d, 2.3% above 270,000, a mature shelf's calendar-day averages). The core offshore oil wells barely move: P10 / P50 / P90 of 6 / 91 / 1985 Sm³/d of oil equivalent, 19.0% above 1018 Sm³/d, a mean of 663 (the rework: 6 / 94 / 1852, 19.4%, 634).

**D-61's fit.** $b = 0$ in every attempt of at least half the gas and condensate wells (104 of 174 `sol-2` wells have it in some attempt), as the check run found (60% of the states).

**The decline of a life.** Measured as in `rework-measurements.md` on the core `nsol-2` wells, the median last fifth over the first year:

| Run | All | Oil, support | Oil, depletion | Gas, support | Gas, depletion |
|---|---|---|---|---|---|
| The rework (final) | 0.12 | 0.03 | 0.22 | 0.67 | 0.14 |
| R9 | 0.14 | 0.03 | 0.22 | 0.76 | 0.18 |

The fifths, all core wells: 0.98, 0.58, 0.34, 0.24, 0.14. The gas wells decline a little less with R9; the cause is not measured (D-61 refits $a$ and $b$ at each reservoir state). The gas wells with pressure support stay the open item they were.

**One life for both datasets (D-105).** The stage table of `rework-measurements.md` is unchanged to 0.01: $p_r/p_i$ and the oils' water cut of `sol-2`'s wells and `nsol-2`'s samples agree by stage and drive as before.

## Outcomes

| | `sol-2` | `nsol-2` |
|---|---|---|
| Wells kept | 400 of 487 (82.1%; core 85.3%, tail 70.0%) | 100 of 115 (87.0%) |
| Not producible | 73 | 5 |
| Redraw limit | 10 (8 core: 2.1% of the core's drawn wells) | 6 |
| Solver-failure limit | 2 | 0 |
| Best-rate solver failure | 2 | 0 |
| Dead date, short life | – | 2, 1 |
| Table failed | 0 | 1 |
| Redraws per sample | 0.106 | 0.110 |

The redraw-limit drops, the dead dates and the best-rate solver failures stay above the draft's targets, as in the rework; R9 neither adds to them nor removes them, apart from the `nsol-2` table well above.

## Cost

| | Rework (final) | R9 |
|---|---|---|
| Rust solve, median / mean / P95 (s) | 0.156 / 0.173 / 0.341 | 0.142 / 0.262 / 0.370 |
| Mean Rust solve, volatile oils (s) | 0.186 | 0.812 |
| CPU per accepted `sol-2` well (s) | 41.3 | 70.1 |
| CPU per accepted `nsol-2` well (s) | 50.4 | 106.4 |
| CPU outside the core and the retries, `sol-2` / `nsol-2` | 0.3% / 0.2% | 11.0% / 20.2% |
| Pool efficiency, `sol-2` / `nsol-2` | 0.96 / 0.92 | 0.46 / 0.42 |
| Projected CPU-hours, 10,000 wells, `sol-2` / `nsol-2` | 115 / 140 | 195 / 295 |
| Table builds, per 10,000 wells (CPU-hours, from the mean build) | – | about 6 (`sol-2`) and 18 (`nsol-2`) |

Two costs are new:

- **Slow volatile oils.** 8 of the 76 volatile oils on their tables in `sol-2` take 540 to 2930 s each, 37% of the run's CPU; the rest solve at the black oils' cost, about 1 ms per march. In the slow wells a march costs about 19 ms and a root search takes 2 to 3 times as many marches. The slow wells hold the pool at the end of each batch, which halves its efficiency. On draw 162, one solve of 11.3 s takes 0.41 s with Joule–Thomson cooling (THM-8) off, and 0.31 s with the black oil in place of the table and the cooling on; a single table at the attempt's GOR is as slow as the set, so the GOR interpolation is not the cause. With the cooling on, the core's temperature solve does 698,000 cell solves in that search, 78% of them fall back from the chord iteration to Brent and 55% end unsolved, for 27 million states, against 0.55 million for the black oil. DAK's Joule–Thomson factor is not near a pole: the tables' gases have reduced temperatures of at least 1.15 and $\lvert J\rvert \le 5$ over the wells' range, and neither relates to the cost. Why the energy row fails to solve on these tables is open.
- **Building an attempt.** `PVTTableSet.at_gor` interpolates a table at each attempt's GOR in Python, about 30 to 40 ms per attempt, and `nsol-2`'s life scans build more of them; that is the CPU outside the core.

## Open for the smaller checkpoint

- D-106, the repair of the tables' $p/Z$, and its 5% limit (one `nsol-2` well over it in these pilots).
- The slow volatile oils on their tables: the energy row's temperature solve with Joule–Thomson cooling, a Rust core finding to settle before R11. Until it is settled, the tables add 70% to `sol-2`'s CPU and double `nsol-2`'s.
- The tables' saturation pressures against R3's (a third of the table wells change their saturation label), and the condensates' lower liquid rates.
- $b = 0$ in D-61's fit for most gas wells.
- The rework's items: the gas wells with pressure support, the redraw-limit drops, the dead dates and the best-rate solver failures, and the platform oil wells' upper tail.
