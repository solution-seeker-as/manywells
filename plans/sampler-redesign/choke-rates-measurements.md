# The choke's sizing and the start hurdle: measurements

*2026-11-14, on the branch `choke-sizing-and-rates`. The measurements behind D-150 to D-153 (`decisions.md`), which change SMP-77, SMP-102 and SMP-103 of `specs/sampling.md` after the first beta's analysis. Bjarne took two of their values on 2026-11-14 (section 6) and had the branch merged into `develop`; the rest are working assumptions for his ruling. Plans are drafts, not specs.*

Bjarne asked for two things to be changed after the first beta: the chokes' sizing and the distribution of the rates (his decision; a private comparison with field data informed it). Everything below is measured on data that ManyWells generated, the beta's datasets and the pilots, or stated in words from analyses of open well data, which are kept outside the repo (`AGENTS.md`, Data and sources). No real-well value is given here.

- **The beta** is `manywells-sol-2-beta1` and `manywells-nsol-2-beta1`, 10,000 wells of 200 samples each, at `e23705d`.
- **The first pilot** is the first 600 draws of `sol-2` (seed 2026) and the first 400 of `nsol-2` (seed 2027), generated with the two rules at their first values, a choke's share $\xi$ from 0.02 and hurdles of 50 and 500 Sm³/d, and the beta's tables. Sections 3 and 4 are its measurements, and "the pilots" there are its two datasets. They are compared with the same draws of the beta, so that every difference is a rule's: a block's draws do not move when another block's rule changes (SMP-51). They were run with start hurdles of 20 and 200 Sm³/d on the core wells and with four sizing chokes. The hurdle drops a well before any solve, and no well was sized with the third or the fourth choke, so the pilots' wells at or above a pair of hurdles are the wells that the code generates with them and that draw of $\xi$; the wells below them are taken out. Checked on the first 12 draws of `sol-2` and the first 6 of `nsol-2`, generated again with the code at the first values (`e97cf8b`): the outcomes are the pilots', and the 13 accepted wells' samples are the pilots' to the bit.
- **The values taken.** After the first pilot Bjarne took a platform hurdle of 20 Sm³/d for 50, and a lower end of 0.005 for 0.02 in the choke's share (D-150 and D-151, Ruling). Section 6 holds what is measured with them: the hurdle on the whole beta and on the first pilot's wells, and the lower end on the part of a second pilot that ran before it was stopped. A full pilot is to follow, with the rulings on the beta's analysis (`beta1-analysis.md`, section 12).
- **The whole beta with a hurdle** is the beta's wells whose design rate passes it: the hurdle changes no well that it leaves. It gives the hurdle's effect on 10,000 draws, with the beta's chokes.
- **The scripts** are in `plans/evidence/`: `choke_rates_checks.py` prints every figure of sections 1, 3 and 4 from a pilot and the beta, or from the whole beta with a hurdle; `choke_sizing_solves.py` counts the sizing's chokes and solves; `choke_design_point.py` measures section 2's alternative without a solve, `permeability_reweighting.py` section 5's higher permeability, and `choke_share_floor.py` section 6's lower end of the share, from two runs' well files.
- **Oil equivalent** is oil plus gas over 1000, in Sm³/d, without the lift gas (v2: scope and conventions).

## 1. What the beta shows

### The chokes

A choke takes the share $1 - p_s/p_N$ of the wellhead pressure $p_N$. In the beta it takes less than 1% in a fifth of the samples:

| | `sol-2` | `nsol-2` |
|---|---|---|
| Samples whose choke takes less than 1% of the wellhead pressure | 19.1% | 22.8% |
| Less than 3% | 29.0% | 33.1% |
| At critical flow (`CHOKED`) | 26.7% | 23.9% |
| The fully open samples' share: 5th, 25th, 50th, 75th and 95th percentile | 0.0001, 0.006, 0.046, 0.24, 0.80 | 0.0001, 0.004, 0.035, 0.21, 0.78 |

- **At small openings too.** Of the `sol-2` samples with the choke at 0.05 to 0.25, 10.5% have a choke that takes less than 3%; at 0.25 to 0.5, 22%; at 0.5 to 0.75, 33%; above 0.75, 41%; fully open, 44% (`nsol-2`: 13%, 26%, 38%, 46% and 48%).
- **It follows the well's rate.** By the well's best rate in oil equivalent, the `sol-2` samples whose choke takes less than 1% are 58% below 10 Sm³/d, 37% from 10 to 50, 19% from 50 to 200, 7% from 200 to 1000 and 1.7% above (`nsol-2`: 68%, 55%, 33%, 15% and 7%).
- **The cause is the tubing's smallest size.** D-27's bean is 0.15 to 0.6 of the tubing's diameter. The tubing is the smallest of five sizes that carries the design rate below the erosional velocity (SMP-75), and the smallest, 2-7/8 in, carries far more than most wells flow: 46% of the core wells take it, and their design rate is a median of 18% of its erosional capacity, and less than 7% in a quarter of them. Its smallest bean, 9.3 mm, is the same for a well of 5 Sm³/d and one of 500. The beta's median bean is 25 mm for the wells whose best rate is below 16 Sm³/d, 26 mm from 16 to 159, 31 mm from 159 to 1018 and 57 mm above: it does not follow the rate until the tubing does.
- **The fully open choke's share spreads over five decades**, from a choke that takes nothing to one at critical flow, where a choke sized to its well would take a share within one or two.

### The rates

The beta's wells start weak, most of all its subsea wells. `nsol-2`'s core wells, the best rate at the life's start and the best 12 months of operation (the largest yearly mean of the operating samples), 5th, 25th, 50th, 75th and 95th percentile in Sm³/d of oil equivalent:

| Core wells of `nsol-2` | Best rate at the start | Best 12 months |
|---|---|---|
| Platform, oils | 11, 98, 347, 1056, 6243 | 10, 77, 251, 776, 4397 |
| Platform, gas classes | 18, 83, 211, 486, 2824 | 15, 68, 162, 380, 2317 |
| Subsea, oils | 62, 327, 1033, 3675, 11,045 | 44, 211, 682, 2389, 8222 |
| Subsea, gas classes | 81, 246, 550, 1577, 6870 | 61, 185, 403, 1097, 5057 |

- **Against the regions' wells that start within the open data** [own analysis of BOEM's OGOR-A production and ANP's production by well], by their best 12 months: the platform wells' median lies within the regions', between the Gulf of Mexico's shelf and its deep-water platforms and a little below Brazil's post-salt wells', and their 5th percentile lies at the regions' lowest; the subsea wells lie well below both subsea regions, the Gulf's deep water and Brazil's pre-salt, at every quantile up to the median, the more so the lower the quantile, and above both at the 95th percentile.
- **The draws' design rates.** 10% of the drawn core platform wells have a design rate (SMP-75) below 20 Sm³/d of oil equivalent, 23% below 50 and 37% below 100; 23% of the drawn core subsea wells have one below 200 Sm³/d, 30% below 300 and 41% below 500 (`sol-2`; `nsol-2` within a point of these).
- **A `sol-2` well is weaker than an `nsol-2` well at its start**, since it is its life at one date: its core platform wells' best rate has a median of 118 Sm³/d, and 31% lie below 50 Sm³/d; its core subsea wells' has a median of 323 Sm³/d, and 14% lie below 50.
- **The choke held back little of the rate**: the median `sol-2` well's median sample makes 91% of its best rate.

## 2. The rules

**The choke (D-150, SMP-77).** The fully open choke takes a drawn share $\xi \sim \text{LogU}(0.005, 0.45)$ (the first pilot: from 0.02) of the pressure upstream of it at the well's own solved rate, where the dataset's well begins: an `nsol-2` life's start, a `sol-2` well's date. The well is solved there with its largest choke, D-27's largest bean, and the choke is the one that passes that rate with the drop of a choke that takes $\xi$; where the well has no operating point with the largest choke, a bean of half the diameter is tried.

**The start hurdle (D-151, SMP-102).** A platform well whose design rate in oil equivalent is below 20 Sm³/d (the first pilot: 50), and a subsea well whose design rate is below 500 Sm³/d, is not drilled, in the core and in the tail; onshore wells have no hurdle.

**An alternative without a solve** (D-150): the bean from the tubing's design point. Over 1491 black-oil `nsol-2` wells of the beta, the design mass rate over the solved fully open rate has a 5th percentile of 0.31, a median of 0.74 and a 95th percentile of 2.3 (a log standard deviation of 0.64), so a choke meant to take a tenth of its upstream pressure would take from 0.02 of it to more than critical flow allows between its 5th and 95th percentiles (`plans/evidence/choke_design_point.py`).

## 3. The first pilot against the beta's same draws

*With the first values: $\xi$ from 0.02, and hurdles of 50 and 500 Sm³/d. Section 6 has the values taken.*

### Outcomes

| The pilots' draws | `sol-2`, beta | `sol-2`, pilot | `nsol-2`, beta | `nsol-2`, pilot |
|---|---|---|---|---|
| Draws | 600 | 600 | 400 | 400 |
| Accepted | 488 | 396 | 364 | 289 |
| Below the start hurdle | | 146 | | 91 |
| No operating point at the best rate or at the sizing | 90 | 49 | 18 | 6 |
| Below the floor | 17 | 5 | 9 | 5 |
| At the redraw limit | 3 | 2 | 5 | 6 |
| Other (a failed table, a solver failure, a short life) | 2 | 2 | 4 | 3 |

- **The hurdle takes most of the wells that were dropped anyway.** Of the 146 `sol-2` draws below their hurdle, the beta accepted 90 and dropped 56, most of them without an operating point.
- **Lost and gained.** Of the beta's accepted wells the `sol-2` pilot loses 93, 90 to the hurdle and 3 without an operating point with a sized choke, and gains 1; the `nsol-2` pilot loses 78, 73 to the hurdle, 4 at the redraw limit and 1 whose life is too short, and gains 3.
- **The sizing alone**, on the beta's accepted wells that pass the run's hurdles of 20 and 200 Sm³/d: 4 of 445 `sol-2` wells are lost, without an operating point with a sized choke; 8 of 329 `nsol-2` wells, 6 at the redraw limit, 1 with a short life and 1 below its floor.
- **The sizing chokes.** Of the wells that reach the sizing, 404 `sol-2` wells and 301 `nsol-2` wells are sized with the largest choke, 4 and 3 with the quarter, and 45 and 5 have no operating point with either; of the 685 accepted wells, 678 and 7. No well was sized with the two smaller chokes that the run also tried (`choke_sizing_solves.py`).
- **The tail** is 18.2% (`sol-2`) and 19.7% (`nsol-2`) of the pilots' accepted wells, 16.4% and 18.4% in the beta's same draws.

### The chokes

Percentiles are the 5th, 25th, 50th, 75th and 95th.

| | `sol-2`, beta | `sol-2`, pilot | `nsol-2`, beta | `nsol-2`, pilot |
|---|---|---|---|---|
| Samples whose choke takes less than 1% of the wellhead pressure | 17.2% | 1.9% | 20.1% | 4.8% |
| Less than 3% | 26.4% | 5.9% | 30.1% | 13.7% |
| At critical flow (`CHOKED`) | 28.6% | 24.4% | 25.9% | 26.9% |
| `PDC`/`PWH` over the samples | 0.11, 0.47, 0.82, 0.97, 1.00 | 0.13, 0.55, 0.76, 0.90, 0.97 | 0.12, 0.52, 0.86, 0.98, 1.00 | 0.14, 0.51, 0.78, 0.93, 0.99 |
| The fully open samples' share | 0.0001, 0.008, 0.057, 0.27, 0.78 | 0.012, 0.042, 0.091, 0.19, 0.51 | 0.0002, 0.005, 0.045, 0.23, 0.73 | 0.005, 0.026, 0.081, 0.21, 0.63 |
| Less than 3%, by opening: to 0.25, to 0.5, to 0.75, above, fully open | 10%, 20%, 30%, 38%, 40% | 1%, 2%, 5%, 12%, 16% | 11%, 24%, 35%, 42%, 44% | 2%, 5%, 15%, 26%, 30% |
| Less than 1%, by the well's best rate: below 10 Sm³/d, to 50, to 200, to 1000, above | 67%, 36%, 17%, 4%, 2% | 7%, 5%, 2%, 1%, 0.2% | 78%, 57%, 28%, 16%, 6% | 3%, 4%, 5%, 4%, 6% |
| Median bean (mm), by best rate: below 16 Sm³/d, to 159, to 1018, above | 25, 26, 31, 56 | 8, 21, 34, 72 | 28, 23, 27, 54 | 5, 13, 27, 61 |
| Wells at their largest bean | | 23% | | 22% |

- **The bean follows the rate.** A weak well's bean is a third to a fifth of the beta's, a strong well's somewhat larger. Over the wells accepted in both, the bean is 0.24 to 2.9 of the beta's between its 5th and 95th percentiles in `sol-2` and 0.25 to 3.0 in `nsol-2`, with a median of 1.0, and the best rate is 0.8 to 1.4 of the beta's in both, with a median of 1.0.
- **What the fully open choke takes.** In the wells below their largest bean the fully open choke takes a median of 0.82 (`sol-2`: the well's fully open samples) and 0.75 (`nsol-2`: the first test's fully open sample) of the drawn share $\xi$, as SMP-77 expects: with its own choke the well flows a little less, at a higher wellhead pressure, than with the sizing choke. The wells at their largest bean take a median of 0.18 and 0.14 of the wellhead pressure.
- **`nsol-2` loses grip along the life.** The choke is the start's. The samples whose choke takes less than 1% are 2.4% of the initial stage's and 5.4% of the mid-life's and the late stage's. By fifth of the life they are 2%, 5%, 7%, 6% and 5% of the core platform wells' samples (the beta's: 18%, 19%, 19%, 17% and 15%) and 2%, 2%, 7%, 11% and 16% of the core subsea wells' (17%, 20%, 22%, 24% and 28%). In the last fifth the rate is a median of 0.54 (platform) and 0.56 (subsea) of the first fifth's, and the downstream pressure 0.38 of it on a platform, whose separator steps down, and 0.76 subsea: the subsea well's choke sees a rate that has halved against a back-pressure that has not. At the seventh choke test the fully open choke takes a median of 0.94 of what it took at the first, and a third of it or less in a quarter of the wells. In the Gulf's well tests, too, the share of the tests without grip rises along a well's test history [own analysis of BSEE's well tests].
- **A `sol-2` well's choke fits its date**, so `sol-2` has no such loss: its samples below 1% are 1.5%, 1.7% and 2.7% by life stage.

### The rates

**The hurdle on the whole beta.** `nsol-2`'s core wells' best 12 months of operation (the largest yearly mean of the operating samples), in Sm³/d of oil equivalent, for the beta and for its wells that pass each pair of hurdles (platform and subsea, Sm³/d), with the beta's chokes:

| Core wells of `nsol-2` | The beta | 20 and 200 | 20 and 500 (taken) | 50 and 500 (the first pilot's) | 100 and 750 |
|---|---|---|---|---|---|
| Platform, oils | 10, 77, 251, 776, 4397 | 19, 100, 282, 871, 4499 | 19, 100, 282, 871, 4499 | 37, 144, 334, 1004, 4843 | 59, 193, 427, 1232, 5190 |
| Platform, gas classes | 15, 68, 162, 380, 2317 | 44, 98, 197, 433, 2536 | 44, 98, 197, 433, 2536 | 77, 148, 271, 551, 3075 | 119, 212, 347, 706, 4139 |
| Subsea, oils | 44, 211, 682, 2389, 8222 | 107, 355, 963, 3014, 8699 | 166, 559, 1338, 3642, 9039 | 166, 559, 1338, 3642, 9039 | 221, 746, 1732, 4079, 9446 |
| Subsea, gas classes | 61, 185, 403, 1097, 5057 | 181, 376, 746, 1791, 6357 | 324, 733, 1249, 2524, 7630 | 324, 733, 1249, 2524, 7630 | 469, 994, 1763, 3071, 8616 |
| Accepted wells of the beta below the hurdles (`sol-2`, `nsol-2`) | | 10%, 13% | 16%, 19% | 21%, 25% | 32%, 35% |
| Draws for 10,000 wells (`sol-2`, `nsol-2`) | 12,299, 10,816 | 13,650, 12,430 | 14,580, 13,350 | 15,610, 14,440 | 17,950, 16,510 |
| Core offshore wells' median mass rate (kg/s; `sol-2`, `nsol-2`) | 3.0, 2.4 | 3.6, 3.1 | 3.8, 3.3 | 4.4, 4.0 | 5.7, 5.1 |
| `sol-2` core wells whose best rate is below 50 Sm³/d (platform, subsea) | 31%, 14% | 26%, 9% | 26%, 5% | 21%, 5% | 16%, 4% |
| Accepted core wells that are subsea (`sol-2`, `nsol-2`) | 33%, 35% | 30%, 32% | 25%, 27% | 27%, 29% | 26%, 29% |
| The tail's share of the accepted wells (`sol-2`, `nsol-2`) | 17%, 18% | 16%, 17% | 17%, 18% | 17%, 18% | 19%, 19% |

Against the starts of the regions' wells that begin within the open data, by their best 12 months [own analysis of BOEM's OGOR-A production and ANP's production by well]:

- **With the hurdles taken**, 20 and 500 Sm³/d, the platform oils' 5th percentile is near the Gulf of Mexico's shelf oil wells'; from the lower quartile to the upper they lie between the shelf wells' and the Gulf's deep-water platform wells', and their median is a little below that of Brazil's post-salt oil wells. The platform gas classes' lower quartile and median are near the shelf's, and their upper quartile lies between the shelf's and the deep-water platforms'. The subsea oils' and gas classes' medians are near the Gulf's deep-water subsea wells', the lower of the two subsea regions; the oils' 5th percentile and lower quartile stay below the Gulf's, and their upper quartile lies between the Gulf's and Brazil's pre-salt wells'.
- **With 50 and 500 Sm³/d**, the first pilot's, the platform oils' 5th percentile lies between the shelf oil wells' 5th percentile and their lower quartile: wells that start as weak as the shelf's weakest are then thinner in the datasets than on the shelf.
- **With 20 and 200 Sm³/d**, the subsea wells' lower quartile and median stay well below the Gulf's.
- **With 100 and 750 Sm³/d**, the platform oils' lower quartile lies above the shelf oil wells' upper quartile, and the subsea wells' medians above the Gulf's.
- **The 95th percentiles** lie above every region's with every pair, as in the beta: a hurdle takes weak wells only.

**The first pilot**, with both rules at their first values, on fewer wells: `nsol-2`'s core wells' best 12 months, the beta's same draws and the pilot:

| Core wells of `nsol-2` | The beta's same draws | The pilot |
|---|---|---|
| Platform, oils (114 and 104 wells) | 34, 127, 339, 1277, 4211 | 53, 130, 556, 1744, 5044 |
| Platform, gas classes (70 and 50) | 15, 74, 172, 400, 1589 | 74, 128, 341, 534, 2602 |
| Subsea, oils (61 and 44) | 52, 235, 662, 2070, 8366 | 125, 445, 1401, 3614, 11,554 |
| Subsea, gas classes (36 and 18) | 92, 193, 477, 875, 3837 | 201, 609, 914, 3262, 5247 |

- **The core offshore wells' median mass rate**, the median well's median sample, rises from 3.6 to 5.4 kg/s in `sol-2` and from 2.9 to 4.3 kg/s in `nsol-2`.
- **`sol-2`'s core wells whose best rate is below 50 Sm³/d** fall from 28% to 19% (platform) and from 14% to 5% (subsea). A `sol-2` well is its life at one date, so its low rates are mostly late lives', which the hurdle leaves.
- **Low rates stay as operating points.** The samples below 20 Sm³/d are 14.6% in `sol-2` (the beta's same draws: 16.9%) and 12.6% in `nsol-2` (15.4%); of the core platform wells' samples, 13% and 12%.
- **Short lives.** The `nsol-2` lives that end within two years fall from 10.7% to 6.9%, and those cut before their span from 28% to 22%.

### Cost

- **The sizing's solves.** One for a natural well and three for a lifted one, and as many again where the second choke is tried: 2.4 solves per `sol-2` well and 1.5 per `nsol-2` well that reaches the sizing, 1.2% and 0.5% of the wells' solves, with the run's four sizing chokes (`choke_sizing_solves.py`).
- **Redraws.** A choke that restricts leaves more small openings without an operating point. Over the wells accepted in both, the redraws per sample rise from 0.093 to 0.107 in `sol-2` and from 0.125 to 0.209 in `nsol-2`, in the same wells: the wells with a redraw are 30% and 48% before and after. In `nsol-2` the attempts without an operating point rise from 23% to 34% at openings below 0.25, from 10% to 18% at 0.25 to 0.5, and from 7% to 9% above 0.75. The accepted samples' openings keep their quartiles (section 4).
- **Draws.** With hurdles of 50 and 500 Sm³/d, 10,000 wells take about 15,600 draws of `sol-2` and 14,400 of `nsol-2`; with the hurdles taken, about 14,600 and 13,350. The table fluids are built for the first 13,000 and 12,500 draws, so a full generation first builds the tables of the draws beyond.

## 4. What stays, and what does not

*The first pilot's, as section 3.*

| | `sol-2`, beta | `sol-2`, pilot | `nsol-2`, beta | `nsol-2`, pilot |
|---|---|---|---|---|
| Samples with $p_r$ below hydrostatic | 53.2% | 52.5% | 60.1% | 58.9% |
| Samples with `PDC` below 10 bar | 16.2% | 19.4% | 15.5% | 17.3% |
| Fully open samples | 5.3% | 5.3% | 3.8% | 4.0% |
| The other samples' openings, quartiles | 0.30, 0.54, 0.77 | 0.31, 0.54, 0.77 | 0.31, 0.55, 0.79 | 0.31, 0.56, 0.79 |
| Lifted samples | 29.5% | 28.7% | 37.7% | 39.3% |
| Median `PWH` (bar) | 53.9 | 47.1 | 57.9 | 51.2 |
| Median `TWH` (K) | 319.7 | 325.0 | 318.5 | 326.2 |
| Median `PBH` (bar) | 216 | 214 | 188 | 207 |
| `PBH`, 0.5% to 99.5% (bar) | 7 to 683 | 9 to 659 | 10 to 725 | 14 to 689 |
| `WTOT`, 0.5% to 99.5% (kg/s) | 0.011 to 112 | 0.011 to 132 | 0.033 to 124 | 0.013 to 140 |
| Samples with `PBH` above 500 bar and `WTOT` above 60 kg/s | 0.9%, in 8 wells | 1.6%, in 8 wells | 0.9%, in 10 wells | 2.0%, in 12 wells |
| Samples with `WLR` above 0.85, `PWH` below 20 bar and `WTOT` above 15 kg/s | 0.4%, in 4 wells | 0.6%, in 6 wells | 0.3%, in 12 wells | 0.7%, in 14 wells |

**What stays.** The reservoirs below hydrostatic, the low downstream pressures, the fully open samples and the openings, the lift, and the samples' ranges. The wellheads are warmer and at a lower pressure because the wells are stronger and fewer of them are subsea.

**What does not improve.**

- **The strongest wells flow more.** A choke sized to the rate lets a strong well flow what its tubing passes, where a small bean of D-27 held some of them back: the samples at a bottomhole pressure above 500 bar and above 60 kg/s are about 40% (`sol-2`) and 80% (`nsol-2`) more, in nearly the same wells, and twice the share of the samples; `WTOT`'s 99.5% quantile rises from 112 to 132 kg/s (`sol-2`) and from 124 to 140 kg/s (`nsol-2`). Of the core subsea oils that pass the hurdle, 28% have a design rate above their largest tubing's capacity (D-100's flag), 20% in the beta. The rule for these wells is D-149's open point; it matters more with these two rules.
- **`nsol-2`'s subsea wells lose grip late in their lives** (section 3): 16% of their last fifth's samples have a choke that takes less than 1%.
- **The mix by well type moves.** Hurdles of 50 and 500 Sm³/d drop 24% of the drawn platform wells and 44% of the drawn subsea wells. By the whole beta's wells that pass, the accepted core wells are 61% platform, 27% subsea and 12% onshore in `sol-2` and 59%, 29% and 13% in `nsol-2`, where D-5 draws 55%, 35% and 10% and the beta accepted 58%, 33% and 9%, and 55%, 35% and 9%. Of all accepted wells, the onshore wells are 17% and 18% (the beta's: 13.5% and 13.4%), since they have no hurdle. Section 6 has the mix with the hurdles taken, which moves further.

## 5. Options not taken, and open points

For Bjarne's rulings, with D-150 to D-153:

1. **The platform hurdle** is 20 Sm³/d since Bjarne's ruling of 2026-11-14 (D-151); the first pilot's was 50. The subsea hurdle of 500 Sm³/d, the one that puts the subsea wells' medians near the Gulf's, is not ruled on.
2. **The lower end of the choke's share** is 0.005 since the same ruling (D-150); the first pilot's was 0.02. It is measured on a third of a pilot (section 6), and is to be confirmed on a full one.
3. **An `nsol-2` choke sized once.** A bean changed along the life, as the separator steps down or the lift starts, would keep the subsea wells' grip. It needs the choke's coefficient as a sample's input, a change of the schema and of SMP-88; not taken, for simplicity.
4. **The two datasets size at different states**, `sol-2` at the well's date and `nsol-2` at the life's start, because a `sol-2` table well's fluid is built for its date only (SMP-116). `sol-2`'s chokes therefore grip at every stage, and `nsol-2`'s lose grip with age.
5. **The well types' shares** (D-5). With the hurdles taken, drawn shares of about 0.46, 0.45 and 0.08 would give 55%, 35% and 10% among the accepted core wells; it is open whether D-5's shares are the draws' or the datasets'. A change of the shares changes the type of some wells, and so their table fluids. Since D-157 the datasets' wells are quotas by stratum, which hold D-5's shares among the core wells with no draw changed.
6. **A higher permeability in place of a hurdle** (D-151). By the beta's `nsol-2` wells weighted to the other draw (`permeability_reweighting.py`), a subsea median of 150 mD in place of 60 puts the core subsea oils' best 12 months at 83, 416, 1467, 4080 and 9848 Sm³/d and the flagged oils at 37%: it raises the top with the middle.
7. **The wells too strong for their tubing** are more of the wells after the hurdle and flow more after the sizing. On the beta's analysis Bjarne ruled that the flag stays a label (`beta1-analysis.md`, section 12), from the beta's shares; section 6 has the shares with the hurdles taken.
8. **The table build** is to be extended to about 15,500 and 14,500 draws before a full generation (about 21,000 and 18,100 with D-157's quotas), and further for the table fluids that the ruling on the beta's analysis draws again.
9. **The schema and the label** (D-152, D-153): two columns of `wells`, a reason's code, a record, and `beta2`.

## 6. The values taken on 2026-11-14

Bjarne took the recommendation that followed the first pilot: a platform hurdle of 20 Sm³/d for 50, and a lower end of 0.005 for 0.02 in the choke's share. The reason is the Goals' (Coverage first, shape second): a sample of real wells shows its own fields and areas, so its range is a lower bound on what the draws must cover, and its shape is weak evidence of what is typical. The open regions differ widely in both respects, in the rates at which their wells start [own analysis of BOEM's OGOR-A production, ANP's production by well and Sodir's FactPages] and in what their chokes take [own analysis of BSEE's well tests], and each first value left out wells that one region has: the shelf's weak starts, and the wells that are produced wide open.

### The platform hurdle of 20 Sm³/d

By the whole beta's wells that pass hurdles of 20 and 500 Sm³/d (section 3's table):

- **The rates.** The platform oils' 5th percentile is near the shelf oil wells', where the first pilot's hurdle put it between the shelf's 5th percentile and lower quartile; the quartiles stay between the shelf's and the deep-water platforms'. The subsea wells are the first pilot's.
- **The draws.** 12% of the drawn platform wells and 44% of the drawn subsea wells lie below their hurdle, and 16% (`sol-2`) and 19% (`nsol-2`) of the beta's accepted wells; 10,000 wells take about 14,600 and 13,350 draws.
- **The mix by well type** moves further than with the first pilot's hurdles, since the subsea hurdle now drops far more of its wells than the platform hurdle does: the accepted core wells are 64% platform, 25% subsea and 11% onshore in `sol-2`, and 62%, 27% and 12% in `nsol-2`, where D-5 draws 55%, 35% and 10%. Of all accepted wells, the onshore wells are 16% and 17%.
- **The tail** is 17% and 18% of the accepted wells; with a hurdle on the core wells alone it would be 20% and 22%.
- **The fluid types** all stay above the Goals' floor of about 10%: 53% black oils, 16% volatile oils, 15% condensates and 16% gases in `sol-2`, and 51%, 15%, 17% and 18% in `nsol-2`.
- **The wells too strong for their tubing** (D-100's flag) are 28% of the core subsea oils that pass, 20% in the beta, and 11% of the core offshore wells, 9% in the beta.

By the first pilot's wells that pass the same hurdles, with its chokes: 413 `sol-2` wells and 305 `nsol-2` wells are accepted, of the 488 and 364 that the beta accepted in the same draws. The core offshore wells' median mass rate is 4.6 kg/s in `sol-2` and 4.0 kg/s in `nsol-2` (the beta's same draws: 3.6 and 2.9; the first pilot's own hurdles: 5.4 and 4.3). The `sol-2` core wells whose best rate is below 50 Sm³/d are 23% (platform) and 5% (subsea), and the `nsol-2` lives that end within two years 7.5%. The tail is 18% and 19% of the accepted wells.

### The lower end of 0.005

A second pilot, of the same draws with the values taken, was stopped after 198 `sol-2` draws and 153 `nsol-2` draws had finished (draws 0 to 187 and 0 to 144, and a few beyond). On those draws, against the first pilot's wells with the same hurdles and against the beta (`choke_share_floor.py`):

| | `sol-2`, beta | `sol-2`, from 0.02 | `sol-2`, from 0.005 | `nsol-2`, beta | `nsol-2`, from 0.02 | `nsol-2`, from 0.005 |
|---|---|---|---|---|---|---|
| Accepted wells | 158 | 135 | 135 | 141 | 111 | 109 |
| Samples whose choke takes less than 1% of the wellhead pressure | 18.2% | 1.6% | 4.3% | 21.6% | 3.9% | 7.3% |
| Less than 3% | 28.0% | 5.3% | 13.1% | 32.2% | 12.2% | 18.1% |
| At critical flow (`CHOKED`) | 27.8% | 26.3% | 23.9% | 26.9% | 27.4% | 24.8% |
| `PDC`/`PWH`, 5th, 50th and 95th percentile | 0.09, 0.84, 1.00 | 0.09, 0.75, 0.97 | 0.09, 0.79, 0.99 | 0.12, 0.86, 1.00 | 0.14, 0.78, 0.99 | 0.14, 0.82, 0.99 |
| Less than 3%, by opening: to 0.25, to 0.5, to 0.75, above, fully open | 10%, 21%, 31%, 41%, 45% | 1%, 2%, 5%, 10%, 15% | 1%, 4%, 14%, 26%, 30% | 14%, 29%, 37%, 43%, 44% | 1%, 5%, 13%, 24%, 28% | 2%, 9%, 21%, 33%, 36% |

- **The samples whose choke takes less than 1% are two to three times the first pilot's**, and a quarter to a third of the beta's. They lie at the large openings: at openings below a half, 1% and 4% of the `sol-2` samples have a choke that takes less than 3%, where the beta's were 10% and 21%.
- **They are the wells with a small drawn share.** The wells with $\xi$ below 0.02 are 30% (`sol-2`) and 36% (`nsol-2`) of the accepted wells, as the draw gives three in ten; 9.5% and 12.7% of their samples have a choke that takes less than 1%, and 2.1% and 4.3% of the other wells'. A user who wants chokes that always restrict takes the wells above a `choke_dp_share`.
- **The accepted wells are nearly the same.** All 135 in `sol-2`. In `nsol-2`, 109 of the 111: three more wells reach the redraw limit, and one fewer has a life that is too short.
- **Not measured with the lower end of 0.005:** the other two thirds of the pilot's draws, section 4's table, the redraws and the cost. A full pilot is to follow, with the rulings on the beta's analysis.
