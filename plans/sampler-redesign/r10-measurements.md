# Sampler redesign — R10's pilots (2026-11-10)

*R10 moved the sampler from the prototype scripts into the package, `manywells.sampling.v2` and `manywells.datasets.v2`, and changed the rules of decision entries D-108 to D-133 on the way. These pilots are the first datasets the package writes, with `scripts/data_generation/v2/generate_dataset.py`, end to end: the table fluids' build, the run in resumable chunks, and the assembly of the dataset and its companion logs. They are measured as R9's were (`r9-measurements.md`, The rerun after the second checkpoint), by `plans/evidence/r10_pilots.py`. Every number here is ManyWells' own, from synthetic wells.*

## The port

Each part of the package was proved equal, bit for bit, to the prototype that the earlier pilots ran, before the rules changed:

| Part | Compared |
|---|---|
| The well draws, the mapping to a well, Forchheimer's fit | 20,000 draws (wells 0 to 1999, seeds 2026 and 2027, every stage), 3.1 million values; 5,520 mapped cases; 2,760 fits |
| The life and the samples' draws | about 1,100 lives, 37,000 states, 590,000 attempts' inputs |
| The per-well generation | 56 wells, 12,700 attempts, 7,900 samples, every outcome and reason |
| The table build | 45 wells' fluids, manifests and tables, byte for byte; 228 GOR axes |
| The run and the assembly | 44 wells' tables and logs, about 1.2 million values; 31 values differ in the well log's row of one `table_failed` well, as D-125 records |
| Re-solving a sample from its rows alone | 646 samples, 274 of them on tables, to the bit |

Then the rules changed: D-108 to D-121 (R10's working assumptions, many from `guesses.md`'s triage), D-122 to D-126 (the schema), and D-127 to D-133 (an independent review's findings). D-108 alone draws different numbers for most wells, so these pilots are new draws, not R9's wells again.

## The runs

| Run | Seed | Wells kept / drawn | Wall time | Processes |
|---|---|---|---|---|
| `sol-2` | 2026 | 400 / 500 | 43 min, of which 4 min for the tables | 8 |
| `nsol-2` | 2027 | 100 / 116 | 31 min, of which 4 min for the tables | 6 |

Code `2e3c08d`, a clean tree; both datasets' records name it. The datasets and their logs are kept outside the repo, with the maintainer's generated data.

## Outcomes

| | `sol-2` | `nsol-2` |
|---|---|---|
| Wells kept | 400 of 500 (80.0%; core 84.6%, tail 62.5%) | 100 of 116 (86.2%; core 88.2%) |
| R9's rerun | 80.6% (core 84.5%, tail 66.0%) | 87.0% |
| Not producible | 88 | 6 |
| Redraw limit | 7 (1.5% of the core's drawn wells) | 6 |
| Solver-failure limit | 2 | 0 |
| Table failed | 3 | 2 |
| Dead date | – | 2 |
| Crash, invalid input | 0 | 0 |
| Redraws per sample | 0.119 | 0.102 |

Kept by class in `sol-2`: black oil 195 of 243, volatile oil 64 of 78, gas condensate 62 of 81, wet gas 72 of 88, dry gas 7 of 10. The tail keeps less than in R9's rerun, as its gas wells now reach a WGR of 10,000 (D-107). The redraw-limit drops, the dead dates and the table failures are against their targets on R11's pilot (H3).

## Rates

Core wells, median-sample rate:

| | R10 | R9's rerun |
|---|---|---|
| Offshore oil (GOR at most 1068), Sm³/d of oil equivalent, P10 / P50 / P90 | 4 / 74 / 2220 | 6 / 93 / 1965 |
| — above 1018 Sm³/d; mean | 18.6%; 707 | 18.8%; 664 |
| Gas and condensate, Sm³/d, P10 / P50 / P90 | 25,000 / 126,000 / 961,000 | 27,000 / 128,000 / 842,000 |
| — above 270,000 Sm³/d | 26.9% | 30.6% |

By well type (L5's measurement):

| Well type | Oil wells: P50 (Sm³/d oe), above 1018, below 16, lifted | Gas wells: P50 (Sm³/d), above 270,000 |
|---|---|---|
| platform | 52, 12.3%, 33.1%, 53.1% (130 wells) | 111,000, 18.3% (60) |
| subsea | 228, 29.5%, 11.5%, 43.6% (78) | 331,000, 51.4% (35) |
| onshore | 21, 0%, 47.4%, 42.1% (19) | 56,000, 0% (13) |

R9's rerun had platform oil wells at a P50 of 80 and subsea ones at 112. With 130 and 78 wells and rates that spread over three decades, a median is known to about a factor of 1.5, so the shift is within noise for subsea wells and at its edge for platform wells. Three rules lower a slanted or late platform well's rate: the slant skin (D-113: at 60° to 75° an oil well's productivity falls by a fifth at the median), the pressure's exponent (D-115) and the fewer deep-water dry trees (D-110). The platform wells' upper tail (L5) is lower than in R9's rerun.

## Flow regimes

| | Bottom | Top |
|---|---|---|
| `sol-2` | bubbly 38.0%, annular 35.7%, dispersed bubbly 15.3%, slug 10.2%, churn 0.8% | annular 57.8%, slug 29.1%, bubbly 7.7%, dispersed bubbly 2.9%, churn 2.5% |
| `nsol-2` | annular 43.4%, bubbly 31.6%, dispersed bubbly 12.7%, slug 11.2%, churn 1.1% | annular 61.6%, slug 27.8%, bubbly 7.2%, dispersed bubbly 1.7%, churn 1.7% |

Churn flow stays thin (G1).

## The decline of a life

Core `nsol-2` wells, the mean rate in each fifth of the life over the first year's, medians: 0.97, 0.54, 0.28, 0.19 and 0.10, the regions' tenth at the end [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well].

| Group | Wells | Last fifth (R10) | Last fifth (R9's rerun) |
|---|---|---|---|
| Oil, pressure support | 39 | 0.02 | 0.03 |
| Oil, depletion | 8 | 0.38 | 0.23 |
| Gas, pressure support | 18 | 0.27 | 0.25 |
| Gas, depletion | 17 | 0.21 | 0.19 |

The depletion oils are 8 wells; their decline moved with the pressure's exponent (D-115), whose uniform draw has a median of 1.5 where the material balance's is 0.95 (D-115's entry).

## One life for both datasets (D-105)

Core wells, medians, `nsol-2`'s samples against `sol-2`'s wells by stage:

| Stage | Drive | $p_r/p_i$ | Oils' water cut |
|---|---|---|---|
| initial | depletion | 0.73 / 0.83 | 0.00 / 0.00 |
| initial | pressure support | 0.97 / 0.97 | 0.05 / 0.05 |
| mid-life | depletion | 0.45 / 0.45 | 0.09 / 0.08 |
| mid-life | pressure support | 0.92 / 0.91 | 0.68 / 0.76 |
| late | depletion | 0.26 / 0.24 | 0.15 / 0.26 |
| late | pressure support | 0.89 / 0.87 | 0.95 / 0.96 |

`nsol-2`'s samples weigh a stage by its share of a life's dates and `sol-2`'s wells by the stage's draw, so the initial depletion stage, a year in which the pressure falls fast, differs in its median.

## Cost

| | Rework (final) | R9's rerun | R10 |
|---|---|---|---|
| Rust solve, median / mean / P95 (s), `sol-2` | 0.156 / 0.173 / 0.341 | – | 0.147 / 0.180 / 0.407 |
| Mean solve, volatile oils (s), `sol-2` | 0.186 | 0.221 | 0.241 |
| CPU per accepted `sol-2` well (s) | 41.3 | 53.6 | 42.9 |
| CPU per accepted `nsol-2` well (s) | 50.4 | 99.3 | 53.6 |
| Projected CPU-hours, 10,000 wells, `sol-2` / `nsol-2` | 115 / 140 | 149 / 276 | 119 / 149 |
| Table builds, CPU-hours | – | – | 0.26 for 500 drawn `sol-2` wells, 0.18 for 116 `nsol-2` wells |

The tables now cost the run almost nothing: an attempt's table is read in about 1 ms (R10's `PVTTableSet.at_gor`), where it took 37 ms in `sol-2` and 57 ms in `nsol-2`. With the Joule–Thomson guard (#40) the two datasets are back at the rework's cost: a full dataset of 10,000 wells is about 120 to 150 CPU-hours, a day on 6 processes.
