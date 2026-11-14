# The rulings on the first beta's analysis: measurements

*2026-11-14, on the branch `beta2-rulings`. The measurements behind D-158 to D-162 (`decisions.md`), the implementation of Bjarne's rulings of 2026-11-14 on the first beta's analysis (`beta1-analysis.md`, section 12), and a pilot of every rule that `develop` holds since the first beta. Plans are drafts, not specs.*

Everything here is measured on data that ManyWells generated: the first beta's datasets, folders of tables, and the pilot. No real-well value is given.

- **The pilot** is the datasets' whole pipeline (`scripts/data_generation/v2/generate_dataset.py`: the table fluids' build, the run by stratum and the assembly) for 400 `sol-2` wells and 300 `nsol-2` wells, with the first beta's seeds (2026 and 2027), on the sampler's code of `a0f15db`; the datasets record `24aadfa` with changes, which are of documents and evidence scripts. It holds every rule since the first beta: the choke sized to the well's rate and the start hurdle (D-150 to D-153), the lift gas kept on (D-154), the erosional cap (D-155), the quotas by stratum (D-157), and the rulings (D-158 to D-162).
- **The first beta's wells of the same draw indices** are what the pilot is compared with: a block's draws do not move when another block's rule changes (SMP-51), so a difference is a rule's. A dataset drawn by stratum does not hold the first draws, so the comparison takes the pilot's own draw indices.
- **The scripts** are in `plans/evidence/`: `beta2_rulings.py` prints sections 1 to 5 from a pilot and the beta, `fluid_redraws.py` the fluids' redraws from a folder of tables, `beta1_checks.py report` the drops by group and the thresholds, and `choke_rates_checks.py` the chokes and the rates.

## 1. The pilot's wells

| | `sol-2`, beta | `sol-2`, pilot | `nsol-2`, beta | `nsol-2`, pilot |
|---|---|---|---|---|
| Drawn wells | 619 | 619 | 398 | 398 |
| Accepted | 502 | 400 | 365 | 300 |
| Below the start hurdle | | 144 | | 75 |
| No fluid of its class | | 10 | | 3 |
| No operating point at the best rate or the sizing | 97 | 62 | 17 | 4 |
| Below the floor | 16 | 2 | 8 | 4 |
| A short life | | | 3 | 2 |
| At the redraw limit | 3 | 1 | 4 | 8 |
| Another limit (a failed table, the solver's failures) | 1 | 0 | 1 | 2 |

- **The quotas hold.** The accepted wells are 15.0% tail wells, and the core's are 55%, 35% and 10% platform, subsea and onshore in both datasets (D-157).
- **The fluid types** stay above the Goals' floor of about a tenth: 55.5% black oils, 16.0% volatile oils, 14.2% condensates and 14.2% gases in `sol-2`, and 50.0%, 16.7%, 16.0% and 17.3% in `nsol-2`.
- **Draw indices.** The 400 `sol-2` wells take the draw indices up to 815, the subsea wells' quota filling last, and the 300 `nsol-2` wells those up to 583: about 2.0 draw indices for a well, and so about 20,400 and 19,500 for 10,000 wells (D-157's estimate from the first beta: 21,000 and 18,100).

## 2. The table fluids (D-158)

The table wells of each folder of tables, by what the build did with their fluid (`fluid_redraws.py`):

| | First 1200 draw indices, `sol-2` | `nsol-2` | The pilot's folder, `sol-2` | `nsol-2` |
|---|---|---|---|---|
| Table wells | 354 | 359 | 384 | 459 |
| Keep their own draw | 80.8% | 82.2% | 81.2% | 84.1% |
| Take a redraw | 13.0% | 12.3% | 12.8% | 10.9% |
| No fluid of their class in 10 draws | 6.2% | 5.6% | 6.0% | 5.0% |
| Reservoir pressure (bar), median: own draw, redrawn, none | 466, 251, 116 | 489, 244, 108 | 483, 260, 113 | 490, 244, 109 |

- **A well's label is its fluid's.** Every accepted table well of the pilot has a field GOR of its class, 121 of 121 (`sol-2`) and 98 of 98 (`nsol-2`); the first beta's wells of the same draw indices have 127 of 148 and 101 of 115.
- **The redraws are few and early.** Of the 99 redrawn wells of the pilot's two folders, 55 keep their first redraw, 19 their second and 9 their third; none keeps the ninth.
- **A well without a fluid of its class has a low reservoir pressure**, about 110 bar at the median, where no volatile oil or condensate of the class is one phase. Most are tail wells, whose depths are drawn flat: of the pilot's 13 drawn wells without one, 9. The first beta had accepted 8 of the 13, with a moved fluid.
- **Their drop is a filter's** (`not_producible`, `no_fluid_of_class`): as `table_failed` they would count at the limits, and at about 6% of the table wells they would take most of the threshold of 2% alone.
- **A redrawn fluid's tables fail more often** by D-143's limit, p/Z falling at more than a tenth of a table's pressures: 4 of the 99 redrawn fluids' tables in the pilot's folders, and 1 of the 698 others'.
- **Cost.** The fluids' build takes a mean of 1.5 to 1.7 s per table well, and the build's first step 19 ms per draw index with the redraws.

## 3. A `sol-2` tail well's downstream pressure (D-159)

The accepted tail wells' nominal downstream pressure (bar), the first beta's wells of the same draw indices and the pilot's:

| | Beta: at 1.5 bar | Median | Largest | Pilot: at 1.5 bar | Median | Largest |
|---|---|---|---|---|---|---|
| Platform (27 and 22 wells) | 26% | 7.8 | 37 | 0% | 19.6 | 78 |
| Onshore (28 and 26) | 29% | 2.6 | 27 | 0% | 6.9 | 102 |
| Subsea (18 and 12) | 0% | 46.6 | 178 | 0% | 67.2 | 214 |

- **The tail's form is back.** Without the flowline's loss the pilot's tail wells have a median of 10.5 bar, a 95th percentile of 59 bar and a largest of 103 bar, where SMP-66's tail draw has 13, 97 and 120 bar.
- **Its top is thinned by the wells that do not flow.** Of the 114 tail wells drawn the pilot accepts 60 and the beta 73. The 13 lost are 5 below the start hurdle, 4 without a fluid of their class, and 4 without an operating point, at 14, 33, 106 and 120 bar downstream, 2.1 to 2.9 times the beta's pressure: of the 64 that the other two rules leave, D-159 costs 4.

## 4. The lift block's flags (D-160), and `nsol-2`'s start (D-161)

- **The flag marks every well at the floor.** Of the pilot's accepted lifted wells, 6 of 139 (`sol-2`) and 11 of 174 (`nsol-2`) have the capacity's floor of 0.1 kg/s, and each carries `lift_capacity_at_bound`: 2 wells that the producibility filter lifted and 4 whose lift starts at a drawn date among them, with the late-stage draw's design, which the first beta's flag missed. No natural well carries a flag of the lift block.
- **`nsol-2` lifts 7 natural oils from their start**, of 248 oils drawn; the first beta dropped 6 of them as not producible. Two are kept, core black oils with best rates of 46 and 197 Sm³/d that live their whole spans. Three are still not producible with their lift, and 2 reach the redraw limit.
- **`sol-2`'s filter** lifts 18 natural oils at their date, and 9 are kept.

## 5. The wells dropped at the limits (D-162)

| Threshold, of the wells past the filters | `sol-2`, pilot | `nsol-2`, pilot | First beta, `sol-2`; `nsol-2` |
|---|---|---|---|
| At most 2% dropped at the limits | 0.25% (1 of 401) | **3.2% (10 of 310)** | 1.11%; 1.36% |
| At most 10% in a tenth of a drawn input | no input with 1000 wells | no input with 1000 wells | 9.7%; 7.3% |
| At most 10% in a group of 100 drawn wells or more | 0.4% | 4.5% | 6.0%; 17.4% |

**The `nsol-2` pilot is past the first threshold.** Its 10 wells, against the first beta's 5 of 370 on the same draw indices:

- **8 at the redraw limit.** The beta dropped 2 of them there too and accepted 4. The other 2 are natural oils that D-161 lifts from their start: the beta dropped them at the filter, and with their lift they pass it and then need more than 1000 redraws.
- **1 at the solver-failure limit**, as in the beta, and **1 whose table failed**, a redrawn volatile oil (section 2).

The 4 wells that the beta accepted follow from the chokes. Over the wells accepted in both, the redraws per sample rise from 0.155 to 0.213 in `nsol-2`, and the attempts without an operating point at openings up to 0.25 from 27% to 38%: a choke sized to the well's rate, and held below the erosional velocity, leaves more small openings at which the well does not flow (D-150's and D-155's cost). In `sol-2` they fall, from 0.093 to 0.074, with D-154's lift gas kept on.

With 310 wells the share is uncertain: 10 wells could come from a true share between about 1.5% and 6%. A pilot cannot tell whether the full `nsol-2` would pass 2%, but it gives no reason to expect that it holds.

## 6. The chokes, the rates and what stays

| | `sol-2`, beta | `sol-2`, pilot | `nsol-2`, beta | `nsol-2`, pilot |
|---|---|---|---|---|
| Samples whose choke takes less than 1% of the wellhead pressure | 18.2% | 4.6% | 20.2% | 7.4% |
| Less than 3% | 27.7% | 12.6% | 31.0% | 16.1% |
| Less than 3%, by opening: to 0.25, to 0.5, to 0.75, above, fully open | 10%, 21%, 32%, 40%, 42% | 2%, 5%, 14%, 23%, 28% | 10%, 24%, 35%, 45%, 46% | 2%, 8%, 19%, 28%, 30% |
| At critical flow (`CHOKED`) | 26.6% | 31.5% | 23.8% | 33.1% |
| `PDC`/`PWH`: 5th, 25th, 50th, 75th and 95th percentile | 0.12, 0.51, 0.84, 0.98, 1.00 | 0.11, 0.44, 0.75, 0.92, 0.99 | 0.13, 0.57, 0.88, 0.98, 1.00 | 0.09, 0.41, 0.74, 0.93, 0.99 |
| Wells capped at the erosional rate | | 32.0% | | 40.7% |
| At critical flow: the capped wells' samples; the others' | | 73%; 12% | | 60%; 15% |
| Core offshore wells' median mass rate (kg/s) | 3.9 | 5.5 | 3.3 | 4.3 |
| `WTOT`, 99.5% (kg/s) | 118 | 76 | 113 | 71 |
| Samples with $p_r$ below hydrostatic | 54.2% | 52.2% | 61.1% | 61.3% |
| Samples with `PDC` below 10 bar | 14.3% | 14.6% | 14.1% | 16.7% |
| Fully open samples | 5.3% | 5.3% | 3.8% | 3.9% |
| Lifted samples | 30.7% | 30.5% | 39.2% | 40.8% |

- **The chokes grip.** With the lower end of 0.005 on a full pilot, the samples whose choke takes less than 1% are a quarter to a third of the first beta's, and they lie at the large openings.
- **The cap puts a third of the samples at critical flow.** It holds no well above its tubing's erosional velocity that the search could not bring down (none unmet), and three fifths to three quarters of a capped well's samples are at critical flow.
- **The top of the rates falls.** `WTOT`'s 99.5% quantile falls by a third against the first beta, and `nsol-2`'s core subsea oils' best 12 months have a 95th percentile of 4814 Sm³/d of oil equivalent, where the beta's had 8258; their median rises from 748 to 1261 with the hurdle.
- **`nsol-2`'s subsea wells still lose grip with age**: by fifth of the life, the core subsea wells' samples whose choke takes less than 1% are 5%, 7%, 11%, 12% and 16% (the beta's: 16%, 20%, 23%, 24% and 28%).
- **Cost.** The pilot's wells take 37.7 CPU-seconds for an accepted `sol-2` well and 52.3 for an `nsol-2` well, the dropped wells' solves included, where the first beta took 54 and 69: about 105 and 145 CPU-hours for 10,000 wells of each.

## 7. Open points

For Bjarne, with D-158 to D-162:

1. **`nsol-2`'s drops at the limits against the threshold of 2%** (section 5). The pilot has 3.2%. The redraw limit of 1000, the threshold, and the small openings of a sized and capped choke are the three things that can move.
2. **A well without a fluid of its class is a filter's drop** (D-158), a reading of the ruling: 5% to 6% of the table wells, most of them in the tail. The class drawn with the reservoir's pressure would remove the cause, and change every well's draws.
3. **The redraw's extent**: the fluid block with the composition, and 9 redraws (D-158).
4. **The schema**: `fluid_redraws` in `wells`, and the reason `no_fluid_of_class` (D-158).
5. **The flags of the injection depth** follow the capacity's flag to the design in use (D-160), beyond the ruling's one flag.
6. **The table build's draw indices** for a full generation: about 20,400 and 19,500 by the pilot.
