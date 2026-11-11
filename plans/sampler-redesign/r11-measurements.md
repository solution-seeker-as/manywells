# Sampler redesign — R11's pilot (2026-11-11)

*R11 took the two proposals of `guesses.md`'s triage that waited (G7, G30) and fixed the drops of R10's pilots against SMP-106's targets, as the working assumptions D-134 to D-140, and generated the pilot dataset that Bjarne reviews before the full generation (Step R11). The pilot ran twice: first with D-134 to D-139, then with D-140, which the first run's decline called for. Both runs are measured by `plans/evidence/r11_pilot.py`, which adds SMP-106's targets by class, the reservoirs against Nadeau, Sun and Ehrenberg (2023), the gas wells' separators and the decline by fifth to `r10_pilots.py`'s measurements. Every number here is ManyWells' own, from synthetic wells, unless it cites a publication.*

## The decisions

| Entry | Choice | From |
|---|---|---|
| D-134 | Offshore gas and condensate wells' $p_\text{sep} \sim \text{LogN}_ {[10, 120]}(65, 0.5)$ bar | G7 |
| D-135 | An oil's breakthrough at $U(0.1, 0.9)$ of its recovery, for both drives | G30 |
| D-136 | An `nsol-2` redraw keeps a lifted well's lift gas on | R10's redraw-limit drops |
| D-137 | Limits of 100 attempts in a slot and 1000 redraws | R10's redraw-limit drops |
| D-138 | A sign change next to a failed march is no operating point | R10's solver failures |
| D-139 | A table's repair may always drop one pressure | R10's table failures |
| D-140 | A third of the core oils under pressure support start wet | the first run's decline (G30) |

## The runs

| Run | Code | Seed | Wells kept / drawn | Wall time |
|---|---|---|---|---|
| `sol-2`, first run | `7354277` | 2026 | 400 / 498 | 52 min |
| `nsol-2`, first run | `7354277` | 2027 | 100 / 113 | 31 min |
| `sol-2` | `ad39f58` | 2026 | 400 / 507 | 69 min |
| `nsol-2` | `ad39f58` | 2027 | 100 / 113 | 35 min |

Each dataset ran on 5 processes, the two side by side; the second run shared the machine with the full test suite, which lengthened its wall times. The trees were clean, and both datasets' records name the commit. The datasets and their logs are kept outside the repo, with the maintainer's generated data; the second run is the pilot for review. The rest of this file gives the second run's numbers, with the first run's where they differ.

## Outcomes and SMP-106's targets

| | `sol-2` | `nsol-2` |
|---|---|---|
| Wells kept | 400 of 507 (78.9%; core 82.6%, tail 64.8%) | 100 of 113 (88.5%; core 90.0%, tail 82.6%) |
| The first run; R10 | 80.3%; 80.0% | 88.5%; 86.2% |
| Not producible | 103 (first run 94, R10 88) | 6 |
| Redraw limit | 3 (R10 7) | 1 (R10 6) |
| Solver-failure limit | 0 (R10 2) | 1 (R10 0) |
| Table failed | 1 (R10 3) | 2 (R10 2) |
| Dead date, short life | – | 2, 1 |
| Crash, invalid input | 0 | 0 |
| Redraws per sample | 0.135 (R10 0.119) | 0.239 (R10 0.102) |

The first run's drops were the same, well for well. D-140's wet starts make 8 more oils not producible in `sol-2`: lives that start wet and cannot lift their water at the drawn date.

SMP-106's targets, by class of the drawn wells (core and tail):

| Class | `sol-2`: drawn; dropped at the limits | `sol-2`: solver failures, attempts; wells | `nsol-2`: drawn; dropped at the limits | `nsol-2`: solver failures, attempts; wells |
|---|---|---|---|---|
| Black oil | 246; 3 (1.2%), redraw limit | 0 of 53,498; 0 | 48; 2 (4.2%), redraw limit and dead date | 0 of 14,261; 0 |
| Volatile oil | 81; 1 (1.2%), table failed | 0 of 14,257; 0 | 18; 2 (11.1%), table failed | 0 of 4226; 0 |
| Gas condensate | 82; 0 | 0 of 12,426; 0 | 18; 1 (5.6%), solver-failure limit | 0.3% of 4231; 1 well (5.6%) |
| Wet gas | 88; 0 | 0 of 13,808; 0 | 25; 1 (4.0%), dead date | 0 of 7678; 0 |
| Dry gas | 10; 0 | 0 of 1376; 0 | 4; 0 | 0 of 750; 0 |

- **`sol-2`.** No solver failure in 95,365 attempts, where R10 had them in 0.8% of the black oils (D-138); 73 attempts had a sign change rejected at the edge and count as no operating point. The two oil classes' drops, 1.2% each, are one well above the target of 1%: at a true rate of 1%, three or more drops among 246 wells happen in 45% of pilots, and one or more among 81 in 56%. The pilot cannot tell them from the target; the full dataset will.
- **`nsol-2`.** With 18 to 48 wells in a class, one well is 2 to 6% of it, so the pilot cannot judge a target of 1%. It shows which wells drop. Of R10's six redraw-limit wells, four are kept with 356 to 812 redraws (D-137), one ends at a dead date and one still reaches the limit. The two dead dates are 1.8% of the wells, as in R10, against a target of 0.5%; at a true 0.5%, two or more of 113 happen in 11% of pilots. The two table failures and `sol-2`'s one are the three tail volatile oils that D-139 leaves open. The solver-failure limit is one core gas condensate whose attempts find only an unstable root: near its critical point the gas and liquid densities meet, the march fails there, and the search misses the stable root. That is a model edge, not the sampler's.

## Rates

Core wells, median-sample rate:

| | R11 | The first run | R10 |
|---|---|---|---|
| Offshore oil (GOR at most 1068), Sm³/d of oil equivalent, P10 / P50 / P90 | 6 / 114 / 2160 | 6 / 142 / 2850 | 4 / 74 / 2220 |
| — above 1018 Sm³/d; mean | 20.4%; 720 | 23.6%; 885 | 18.6%; 707 |
| Gas and condensate, Sm³/d, P10 / P50 / P90 | 36,000 / 137,000 / 797,000 | the same | 25,000 / 126,000 / 961,000 |
| — above 270,000 Sm³/d | 29.9% | the same | 26.9% |

By well type (L5's measurement):

| Well type | Oil wells: P50 (Sm³/d oe), above 1018, below 16, lifted | Gas wells: P50 (Sm³/d), above 270,000 |
|---|---|---|
| platform | 68, 14.2%, 29.9%, 55.2% (134 wells) | 129,000, 21.2% (52) |
| subsea | 273, 30.1%, 10.8%, 43.4% (83) | 365,000, 56.2% (32) |
| onshore | 22, 0%, 38.9%, 33.3% (18) | 56,000, 0% (13) |

The oils' rates rose with their lower water: at their `sol-2` date, the mid-life oils under pressure support have a water cut of 0.45 at the median, where R10's had 0.76 (below). D-135 alone raised the offshore oils' median to 142 Sm³/d; D-140's wet starts bring it to 114. The share above 1018 Sm³/d stays above the 5.5% of the Gulf of Mexico's oil wells in 2024 (EIA 2025, Table B32), as R10's did. The platform oil wells' upper tail (L5) is 14.2% above 1018 Sm³/d, a little above R10's 12.3%.

The gas classes keep 130 of 180 drawn wells, where R10 kept 141 of 179: against D-134's higher line, the weakest gas wells cannot flow. With them gone the median rises a little and the upper tail falls. By well type the share above 270,000 Sm³/d stays near R10's (platform 21.2% against 18.3%, subsea 56.2% against 51.4%), within a well or two of 52 and 32. D-62's permeability factor, set at the old back-pressure, holds.

## The gas wells' line (D-134)

Core offshore wells, bar, P10 / P50 / P90:

| | $p_\text{sep}$ | Nominal pressure downstream of the choke |
|---|---|---|
| Gas classes (84 wells) | 31 / 60 / 95 | 20 / 66 / 130 |
| Oils (217 wells) | 19 / 32 / 59 | 11 / 36 / 121 |

## Reservoirs against Nadeau, Sun and Ehrenberg (2023)

Core `sol-2` wells against the reservoirs of the world's largest fields, by count of reservoirs; a comparison, not a target:

| Class | Sampler | Nadeau et al. (2023) |
|---|---|---|
| $T_r$ below 58 °C | 17.8% | 22.0% (Table 1) |
| $T_r$ 58 to 122 °C | 63.3% | 65.1% |
| $T_r$ above 122 °C | 19.0% | 12.9% |
| $p_i$ over fresh-water hydrostatic below 0.8 | 0.0% | 6.0% (Fig. 20) |
| 0.8 to 1.3 | 69.3% | 77.8% |
| above 1.3 | 30.7% | 16.2% |

The depth is from sea level offshore, from the surface onshore. The sampler draws about one and a half times the published share of hot reservoirs and twice the share above 1.3. Its overpressure, SMP-63's $\lambda_{50}(z)$, is fitted to one region's sands [own analysis of BOEM's sands atlas], an overpressured basin. Its underpressured reservoirs, $p_h\thinspace U(0.8, 1.0)$ with the well's own brine, all lie above 0.8 of fresh water's gradient. The first run's shares were the same.

## Flow regimes

| | Bottom | Top |
|---|---|---|
| `sol-2` | bubbly 36.8%, annular 34.9%, dispersed bubbly 16.2%, slug 11.0%, churn 1.1% | annular 58.4%, slug 27.5%, bubbly 8.1%, dispersed bubbly 3.0%, churn 2.9% |
| `nsol-2` | annular 42.5%, bubbly 32.0%, dispersed bubbly 12.7%, slug 11.7%, churn 1.1% | annular 61.4%, slug 28.1%, bubbly 6.2%, dispersed bubbly 2.3%, churn 2.0% |

Churn flow stays thin (G1).

## The decline of a life (D-135, D-140)

Core `nsol-2` wells, the mean rate in each fifth of the life over the first year's, medians; the oils' oil rate, the gas wells' oil equivalent:

| Group | Wells | R10 | The first run (D-135) | R11 (D-140) |
|---|---|---|---|---|
| Oil, pressure support | 39 | 0.87 / 0.29 / 0.08 / 0.03 / 0.02 | 0.97 / 0.73 / 0.22 / 0.05 / 0.03 | 0.96 / 0.61 / 0.22 / 0.06 / 0.03 |
| Oil, depletion | 8 | 1.04 / 0.53 / 0.28 / 0.21 / 0.16 | 0.99 / 0.61 / 0.31 / 0.22 / 0.16 | the same |
| Oil, both drives | 47 | 0.89 / 0.33 / 0.09 / 0.04 / 0.03 | 0.97 / 0.70 / 0.27 / 0.06 / 0.03 | 0.96 / 0.61 / 0.22 / 0.09 / 0.05 |
| Gas, pressure support | 18 | 0.98 / 0.88 / 0.74 / 0.52 / 0.27 | 0.98 / 0.88 / 0.72 / 0.46 / 0.28 | the same |
| Gas, depletion | 17; 16 | 1.11 / 0.71 / 0.49 / 0.32 / 0.21 | 1.15 / 0.77 / 0.56 / 0.35 / 0.28 | the same |

Against the regions' oil wells [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]:

- **R10's oils** lay at the bottom of the regions' span in the second fifth and below it from the third on.
- **D-135 alone** put the second fifth above every region's.
- **With D-140** the second and third fifths lie within the span, the second at its top, and the first fifth just above it.

Over 47 oils the pooled second fifth is known to 0.50 to 0.78 (90% bootstrap). On the pilot's own 39 lives under pressure support, D-140's proxy (`water_breakthrough.py`, $1 - \text{WLR}$ at a fixed liquid rate) gives 0.52, against 0.44 over its 600 lives, so the pilot's lives are on the slow side. The solve adds about 0.09 here, where under R10's rules it added nothing (the proxy's 0.31 on the same lives against the solved 0.29). Over the whole population the second fifth should lie between 0.44 and about 0.53, within the regions' span.

The fourth and fifth fifths stay below the regions' in every run. Their 90% bootstrap upper bounds, 0.11 and 0.06 for the pooled oils, lie below every region's median, as D-140's entry expected: under pressure support a life runs to a water cut of 0.98 (D-105), so in its last two fifths it makes a twentieth of its first year's oil. This is an open finding for Bjarne's ruling (below).

D-41's target, a last fifth at about a tenth of the first year's rate over all core wells: 0.16 (90% bootstrap 0.08 to 0.19), where R10 and the first run had 0.10. The median falls between the oils (0.05) and the gas wells (0.28), so it moves with the mix.

## One life for both datasets (D-105)

Core wells, medians, `nsol-2`'s samples against `sol-2`'s wells by stage:

| Stage | Drive | $p_r/p_i$ | Oils' water cut |
|---|---|---|---|
| initial | depletion | 0.74 / 0.83 | 0.03 / 0.00 |
| initial | pressure support | 0.97 / 0.97 | 0.03 / 0.04 |
| mid-life | depletion | 0.47 / 0.43 | 0.08 / 0.08 |
| mid-life | pressure support | 0.93 / 0.91 | 0.45 / 0.45 |
| late | depletion | 0.27 / 0.24 | 0.13 / 0.20 |
| late | pressure support | 0.89 / 0.87 | 0.95 / 0.96 |

The mid-life oils under pressure support had a water cut of 0.68 and 0.76 in R10 and 0.23 and 0.29 in the first run.

## Cost

| | R10 | The first run | R11 |
|---|---|---|---|
| Rust solve, median / mean / P95 (s), `sol-2` | 0.147 / 0.180 / 0.407 | 0.142 / 0.169 / 0.395 | 0.136 / 0.163 / 0.375 |
| Mean solve, volatile oils (s), `sol-2` | 0.241 | 0.242 | 0.228 |
| CPU per accepted `sol-2` well (s) | 42.9 | 40.4 | 38.9 |
| CPU per accepted `nsol-2` well (s) | 53.6 | 48.2 | 47.7 |
| Projected CPU-hours, 10,000 wells, `sol-2` / `nsol-2` | 119 / 149 | 112 / 134 | 108 / 132 |
| Table builds, CPU-hours, `sol-2` / `nsol-2` | 0.26 / 0.18 | 0.26 / 0.16 | 0.33 / 0.21 |

`nsol-2`'s redraws per sample more than doubled, from 0.10 to 0.24. Four wells that R10 dropped at the redraw limit are now kept, with 356 to 812 redraws each: that is D-137's cost. The CPU per well still fell. A full dataset of 10,000 wells is about 110 to 130 CPU-hours, a day on 5 or 6 processes.

## Checks

At `ad39f58` the full test suite passes (1619 tests), and both verifier candidates pass with 0 failures and a stable-root rate of 100%: `develop` and the Rust core, each in the `v1.0.0` configuration.

## Open findings

1. **The oils' late life.** Under pressure support their fourth and fifth fifths lie below the regions' oil wells, since a life ends at a water cut of 0.98 (D-105). A drawn end, such as $U(0.85, 0.98)$, the range D-41's ruling gave `nsol-2` before D-105, would raise them and keep the wettest wells in coverage; it revises D-105's end, so it waits for Bjarne.
2. **The tail volatile oils' tables** (D-139, Open): three tables fail, as D-139 expected.
3. **A near-critical gas condensate** (`nsol-2`): only an unstable root found where the densities meet; a model edge.
4. **The reservoirs** are hotter and more often overpressured than Nadeau et al.'s (2023) largest fields, and none lies below 0.8 of fresh water's gradient (SMP-63, SMP-65).
5. **The targets** that `nsol-2`'s pilot is too small to judge, and the oil classes' 1.2% in `sol-2`, are judged on the full dataset.
