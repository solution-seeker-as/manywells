# Sampler redesign — the review before the full generation (2026-11-12)

*Bjarne asked for a check of `develop` for bugs and issues before the full generation started again. This file holds what the review measured, the rulings that followed, and the measurements of the fixes. The full generation of 2026-11-11 was stopped after 1000 drawn wells per dataset, when its process pool hung (`run.py`'s renewal, fixed in #44); those wells were made by the sampler that `develop` has, so they show what the full run would hold. The tables were built for every draw index, 13,000 of `sol-2` and 12,500 of `nsol-2`, so their numbers are the full run's. The measurements are `plans/evidence/preflight_checks.py`'s. Every number here is ManyWells' own, from synthetic wells.*

## What the review covered

| Part | How | Result |
|---|---|---|
| The run and its launcher | read; chunk 0 of `sol-2` run again from the run's own worktree, with the pool renewed every 18 wells | the 100 wells are the stopped run's to the bit (16,467 samples, 20,034 attempts), and the assembly writes the dataset from a clean tree |
| The draws, the life, the samples and the generation against SMP-50 to SMP-107 | read rule by rule; 104,000 draws without a solve; 7100 lives | the rules, units, seeding and limits are as specified; the findings below |
| The tables, the datasets' files and the assembly | read; the first 1000 wells of each dataset assembled; 4810 samples solved again from their rows | every sample is the stored one to the bit; the files match `docs/datasets.md` but for the points below |
| The Rust core's operating points | 72 samples of 18 `sol-2` wells, of every fluid class, lifted and natural, solved by the CasADi backend | 64 agree to $2.5\cdot10^{-7}$ relative; in 8, of two gas wells at low pressure, CasADi's own search finds no operating point and reaches the core's from it as a start (4 of 4 tried; `backlog.md`, 2.9) |
| The first 1000 wells of each dataset | outcomes, times and the samples' relations | no crash, no invalid input, no well slower than 413 s; 12,200 and 11,200 draws needed for 10,000 wells, against tables for 13,000 and 12,500 |

## The wells dropped at the limits

The first 1000 draws of each dataset, at `0e6720d`:

| | `sol-2` | `nsol-2` |
|---|---|---|
| Accepted | 822 | 891 |
| Not producible | 164 | 49 |
| `dead_date` | – | 28 (2.8%) |
| `redraw_limit` | 7 | 16 |
| `solver_failure_limit`; `best_rate_solver_failure` | 1; 1 | 1; 0 |
| `table_failed` | 5 | 11 |
| `short_life` | – | 4 |

Dropped at the limits, of the drawn wells of a class (`table_failed` included):

| Class | `sol-2` | `nsol-2` |
|---|---|---|
| Black oil | 7 of 494 (1.4%) | 33 of 492 (6.7%) |
| Volatile oil | 2 of 142 (1.4%) | 13 of 126 (10.3%), 9 of them tables |
| Gas condensate | 4 of 166 (2.4%) | 4 of 169 (2.4%) |
| Wet gas | 0 of 178 | 6 of 192 (3.1%) |
| Dry gas | 0 of 20 | 0 of 21 |

`nsol-2`'s wells that reached their slots, by the rule that starts their lift:

| Wells | Dropped at the limits |
|---|---|
| Oils, lift by the static check | 15 of 27 (56%): 12 dead dates, 3 at the redraw limit |
| Oils, lift at the end of natural flow (rule 3) | 18 of 142 (12.7%) |
| Oils, lift at a drawn date | 1 of 177 |
| Oils, lifted from the start | 3 of 117 |
| Oils that stay natural | 0 of 115 |
| Gas wells without a lift design | 6 of 301 (2.0%) |

- **The dead dates are the natural well's last dates before its lift.** Of the 28, 20 lie in a natural well shortly before its lift starts, 6 in a lifted well, 5 of them at the end of its life, and 2 in gas wells without a design. The static check's start lies at the first of its 41 dates at which a static column fails, and the solved well stops flowing in the interval before it; the life scan does not see it, since the scan's point there is already lifted. Under rule 3 the well flows at the scan's best controls and at none of a date's draws, with its drift. A stepped separator pressure makes a third kind: the well stops shortly before a step and flows again after it.
- **They are the well's, not the solver's.** On 72 attempts of 24 of these wells that the Rust core gave no root, at the two most open chokes of a dead slot and at its planned draw, the CasADi backend finds no root either.
- **A count hid it.** 2.8% of the wells is 9.8% of the oils whose lift starts during their life (34 of 346), and more than half of those the static check lifts.
- **`sol-2`** lies near D-81's 1%: its drops are lifted wells that flow in 1 to 16% of their draws, and tables.

## The table fluids

Over every draw index, at the repair's limit of 5% (D-139):

| | `sol-2` | `nsol-2` |
|---|---|---|
| Table wells | 3983 of 13,000 draws | 3788 of 12,500 |
| Volatile oils without a table | 31 of 1713 (1.8%) | 108 of 1618 (6.7%); core 5.7%, tail 8.2% |
| Gas condensates without a table | 30 of 2270 (1.3%) | 29 of 2170 (1.3%) |
| — the repair's limit ($p/Z$ falls at more than 5% of the pressures) | 33 | 113 |
| — a stock-tank oil lighter than 58 °API, refused by the draws' black oil | 24 | 19 |
| — the build (an `IndexError` in the saturation search, NeqSim's flash, no stock-tank oil, a GOR that is not finite) | 4 | 5 |

- **The repair.** The share of the pressures that a failed table would lose, at the GOR that failed, has a median of 6.5% and a largest of 15%. D-139 expected tail wells at the top of their GOR axis; in `nsol-2`, whose axes are long, core wells fail as often.
- **The light condensates.** A table's single-stage oil is lighter than the drawn, field-reported one by its offsets (SMP-113): 5.4% of the condensates drawn above 56 °API were refused, and 0.4% of the others.

## The rulings (Bjarne, 2026-11-12)

The first full generation is a **beta**, `manywells-sol-2-beta1` and `manywells-nsol-2-beta1`, most likely followed by a final run after its analysis (D-149). Before it:

| Entry | Ruling |
|---|---|
| D-142 | A dead date starts the lift of a natural well with a design, and else ends the life at the last earlier date |
| D-143 | A table's repair may drop a tenth of its pressures |
| D-144 | A table fluid's draws use the black oil at the end of its API range, so the light condensates keep their tables |
| D-145 | SMP-106's targets of 1% and 0.5% go; the drops are reported by group, and thresholds set after the beta |
| D-146 | A life that ends where its lift would start is natural |
| D-147 | The run keeps whether an operating point leaves its PVT table's grid |
| D-148 | The run's settings hold the versions of Python, numpy and scipy |

## The fixes, measured

**The dead dates** (D-142), on the stopped run's wells, each generated again with the rule:

| `nsol-2`, of the first 1000 draws | Before | With the rule |
|---|---|---|
| The 28 wells dropped as `dead_date` | – | 27 kept, 1 dropped as `dead_date` after five dead dates |
| — kept by a lift that starts at the dead date (`no_natural_flow`) | – | 20 |
| — kept by a life that ends at the date before | – | 7, one after two cuts |
| The 16 wells dropped at the redraw limit | – | 3 kept, by shorter lives, one of them with a lift from a dead date too; 13 dropped as before |
| Time of the 28 and of the 16 wells (s) | 2356; 2452 | 3970; 2703 |

- The 13 wells still at the redraw limit are wells that flow in few of their draws: their earliest open slot finds an operating point within its 100 attempts. The 3 kept ones had 10, 23 and 44 open slots, so the redraw limit had hidden their dead dates.
- Of the 30 kept wells, 20 keep the scan's life; 10 end earlier, by 1% to 43% of the scan's life, 4 of them by more than a tenth.
- The cost is 0.5 CPU-hours more in 1000 drawn wells, 4% of `nsol-2`'s, and 30 more wells are kept, so 10,000 wells need about 360 fewer draws. No kept well needed more than two dead dates.

**The tables** (D-143, D-144), the build's failed tables built again with the branch's code:

| Without a table | `sol-2`, before; after | `nsol-2`, before; after |
|---|---|---|
| Volatile oils | 31 (1.8%); 7 (0.4%) | 108 (6.7%); 19 (1.2%) |
| — core; tail, after | 0.2%; 0.8% | 0.6%; 2.1% |
| Gas condensates | 30 (1.3%); 4 (0.2%) | 29 (1.3%); 5 (0.2%) |
| Both classes, at the repair's limit of 10% | 7 | 18 |
| Condensates refused for a stock-tank oil lighter than 58 °API | 0 of 24 | 0 of 19 |

The tables that still fail the repair would lose 10% to 20% of their pressures at a GOR.

**The pilot**, at `917afe0`: the first 300 draws of each dataset, with the tables above, against the same draws of the stopped run (a well's times and the records that the branch adds left out):

| | `sol-2` | `nsol-2` |
|---|---|---|
| Wells the same to the bit | 299 of 300 | 283 of 300 |
| Wells changed | 1, a table well whose table still fails | 17: 8 dead dates and 1 well at the redraw limit now kept; 3 wells kept with the table they now have, and a fourth one's life too short; 3 still at the redraw limit; 1 natural where its life ended at its lift's start |
| Accepted, before; after | 245; 245 | 262; 274 |
| Dropped at the limits, after | 3 at the redraw limit, 1 table | 3 at the redraw limit, 1 at the solver-failure limit; no dead date |
| Dead dates that changed a life | – | 10, in 9 kept wells: 9 lift starts and 1 life end |
| Samples with a point off their table's grid, of the 48,000 of 240 wells | 214, in 2 wells | 0 |
| Wall time on 12 processes; CPU per drawn well | 16 min; 36 s | 20 min; 47 s |

Every well that changed did so for one of the rulings. The datasets, 240 wells each, are written from a clean tree with the beta's names and records, and no `nsol-2` well has a single lifted sample or a lift design without a lifted date.

## For after the beta

Four findings are questions of the rules, which the beta's data informs (D-149):

1. **Table fluids outside their class.** Where the flash puts a fluid in two phases at $p_i$, SMP-115 moves its GOR or takes the equilibrium phase, and the fluid keeps its class. Of the condensates' tables, 18.5% (`sol-2`) and 18.3% (`nsol-2`) have an initial GOR above the class's 8900 Sm³/Sm³, 2% above $10^6$, the largest $2\cdot10^{10}$ and $8\cdot10^{11}$; 23% of the volatile oils' lie below 356 Sm³/Sm³, down to about 50. Of 125 accepted `sol-2` condensates of the first 1000 draws, 25 lie above the class and 6 above the GOR of the dry gases' yield limit, with `R_si` up to $5.8\cdot10^7$. D-127 noted one such well.
2. **The `sol-2` tail's downstream pressure.** A tail well's date is the end of its life, where SMP-89 scales its $p_{s,\text{nom}} - \Delta p_\text{fl}$ by the late factor of 0.1 to 0.6: of 103 accepted platform and onshore tail wells, 27 sit at the floor of 1.5 bar, and none lies above 51 bar, where SMP-66's tail form reaches 120.
3. **The tubing flag.** `design_rate_above_tubing_capacity` is set on 7.5% and 8.7% of the core draws and 16% and 18% of the tail's, against D-100's "a few percent", above which a redraw was to be added.
4. **The lift capacity's floor.** SMP-80's fallback puts 31 of 288 (`sol-2`) and 58 of 480 (`nsol-2`) accepted lifted wells at exactly 0.1 kg/s, the point mass that D-31's truncation was to avoid.

Smaller points, with no change before the beta: about a tenth of the synthetic fluids miss SMP-113's match of 0.5 °API and 0.005 with a lever at its bound (a fifth to a third of the tail's), so `api` and `sg_gas` of a table well are the drawn targets and `field_api` and `field_sg_gas` the built fluid's; volatile oils heat above their table's top temperature in about 1.4% of their samples (D-147); the assembly holds the `pvt` table's text in every row and needs about 18 GB of address space for `nsol-2` at 10,000 wells, so the run's launcher caps it at 30 GB; and neither the Rust compiler's version nor the packages of NeqSim's environment are in the records.
