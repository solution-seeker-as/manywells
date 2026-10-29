# 012 · Root set, stability label and operating point in the simulator

*Feature spec for Step 7 item 6 of `plans/manywells-v2-plan.md` (2026-10-01), with change 12 of `plans/develop_model_changes.md` (the initial-guess march). Status: Bjarne accepted the solver machinery under principle 7 on 2026-10-01: the starts 0.975 and 0.999, the march's fallback, the canonical gas-law row (006) and acceptance on `Solve_Succeeded`; and the march's row check (issue #7) on 2026-10-16. He chose the march's retry of point 0 (issue #10) over a changed bottom guess on 2026-10-28.*

## Motivation

The model's answer is a root set, and the operating point is its stable root (`specs/model/solution.md`, `specs/goals.md`). v1.0.0 and `develop` returned whichever root Ipopt reached from one start, which was the unstable trickle root in 20 of the verifier's 141 cases, and failed in 15.

## Delta

No change to `specs/model/`, whose SOL-1 to SOL-6 already define the answer; `solution.md` gets an informative section on `develop`'s search.

- `solution.py`: `Root` (state, label, normalized slope, CHOKED, flow regimes, reservoir rates), `RootSet` (roots sorted by $p_0$, operating point, the several-stable flag, the search record), `select_operating_point` (SOL-4 to SOL-6).
- `solvers/roots.py`: the multi-start search. Starts: an optional guess; the default march from $p_0 = p_r - 0.05(p_r - p_s)$; marches from $p_0 = p_s + f(p_r - p_s)$ for $f$ in 0.5, 0.7, 0.85, 0.975, 0.995, 0.999. A solve is accepted on `Solve_Succeeded` and SOL-1, solutions within `tol_x` are merged, and each root is labelled by SOL-3 with `label_min`, both copied from the verifier (`specs/architecture.md`, decision 5).
- `solvers/march.py`: the march solves each point by Newton's method from the previous point, and by Ipopt with v1.0.0's bounds where Newton fails. `solvers/ipopt.py`: the feasibility NLP, built once per well with the operating point as parameters.
- `SSDFSimulator(wp)`: `simulate(bc)` returns the operating point and raises `NoOperatingPoint` (a `SimError`, with the root set) without one; `root_set(bc)` returns every root found.

## Solver machinery and its measured gain (principle 7; for Bjarne's decision)

Measured on the verifier's 141 cases in the `v1.0.0` configuration, 2026-10-01, with the canonical gas-law row (006) and acceptance on `Solve_Succeeded`:

| Starts | March fallback | Stable-root rate | Unexpected failures | Incomplete root sets | Search per case |
|---|---|--:|--:|--:|--:|
| method A's five | no | 95.6% | 8 | 5 | 0.63 s |
| method A's five | yes | 97.1% | 5 | 5 | 0.73 s |
| eight (with 0.975, 0.99, 0.999) | no | 98.5% | 2 | 1 | 1.73 s |
| eight | yes | 100% | 0 | 0 | 1.96 s |
| **seven (method A's, 0.975, 0.999)** | **yes** | **100%** | **0** | **0** | **1.60 s** |
| six (without 0.7) | yes | 100% | 0 | 0 | 1.55 s |
| five (0.5, 0.85, 0.95, 0.975, 0.999) | yes | 100% | 3 | 3 | 1.21 s |

v1.0.0 from its default guess: 74.3%. Building a well's system costs 0.8 s per case, once. Accepted: the seven starts and the fallback. The extra starts and the fallback were chosen on this case set, which has no held-out part, so the rates above are optimistic for new wells; the stable-root rate on the regenerated datasets (Step 7 item 7) is a second measurement.

- **0.975.** In steep wells the stable root lies just above $f = 0.95$, where the default march, from below, cannot lift the rate; 0.975 found the stable root alone in 3 cases.
- **0.999.** Trickle roots lie at $f$ up to 0.99998; 0.999 found a trickle root alone in one case. It is the most expensive start (0.67 s on average), mostly from failed solves that run to Ipopt's 3000 iterations.
- **The fallback.** Newton fails where the classifier's slug–annular transition makes the state change steeply ($\alpha$ near 0.7); v1.0.0's bounded Ipopt march does not. It adds 0.2 s per case.
- **The row check (issue #7; Bjarne accepted it, with `NEWTON_ROW_TOL` = 1e-6, on 2026-10-16).** CasADi's Newton reports success where it steps into a region in which the rows are NaN, and the march passed that point on: a march that ran out of pressure stepped to $p < 0$, where the black-oil properties are NaN, and returned a finite state with negative pressures. The march now accepts Newton's point only if every row is finite and at most `NEWTON_ROW_TOL` = 1e-6 in magnitude, and otherwise solves it by Ipopt, as where Newton raises. Newton's converged points leave at most 2.6e-9 on the verifier's case set, where no row is NaN; there the check changes no root set, no attempt's outcome and no Ipopt iteration count, and the extra evaluation of the rows makes the march 11% slower (one process, the first 20 cases, every start), which is 0.016 s or 1% of the search time per case. On the comparison set of `tests/backend_cases.py` (6 wells per configuration, 180 cases), 101 marches in 64 of the 90 wells in the `develop` configuration ended at negative pressures; with the check none did, the search found the same 228 roots and operating points, and Ipopt's iterations in those 90 searches fell from 159,322 to 48,316. Of the attempts that changed, 73 Ipopt solves that failed from such a march (1013 s) became march failures (3 s). The 90 wells in the `v1.0.0` configuration with an overlay are unchanged. Times are on 24 processes; the iteration counts do not depend on the load.
- **The retry of point 0 (issue #10; Bjarne chose it on 2026-10-28).** Where all the reservoir gas is dissolved at the bottomhole, the smoothed phase rates of PVT-OIL-13 leave about $5\cdot10^{-4}$ kg/s of gas, so point 0's void fraction is $10^{-4}$ to $10^{-3}$. From the bottom guess's 0.5, Newton fails there and the Ipopt fallback reports the point infeasible. At well 87 of `develop`'s sampler (seed 2026, nominal point, `gor=40`) every march near the root failed at point 0, and the search found no operating point where the Rust core finds one at 304.40 bar. The march now retries point 0 from the phase rates' no-slip void fraction where it fails from the guess (`march.no_slip_guess`), so every start that worked is unchanged. Measured with `plans/evidence/all_liquid_bottomhole.py`, on `develop` at `ae754d8` with the retry added:
  - **Sampled wells.** The 263 cases of `plans/evidence/jt_wells.py`, at their GOR and at half and a quarter of it, which dissolves all the gas at the bottomhole more often (789 cases). The search finds all 116 operating points with an all-liquid bottomhole, where it missed 15 (4, 5 and 6 at the three GORs), and loses none; every operating point it finds equals the core's to $1.1\cdot10^{-6}$ bar, but in two cases with free gas at the bottomhole, where the two find different roots (0.05 and 0.06 bar), with and without the retry. The 9 operating points it still misses have gas-rich bottomholes: other causes of `plans/improvements.md` §2.9.
  - **Work.** On the 60 cases of the first 20 wells at their GOR, on one process, 4 of 420 starts change outcome, each from a failed march to a root, and the others are unchanged to the bit. Ipopt's iterations rise by 0.16% (26,454 against 26,411), all at those four starts.
  - **The verifier's case set.** The same 200 roots in the same 136 cases, to the bit, and the same search time (1.66 and 1.68 s per case with the retry, 1.68 and 1.64 s without, in two rounds on 24 processes); PASS and 100% for both candidates.
  - **Rejected for it:** guessing the no-slip void fraction at every start, which finds the same operating points but changes other starts' paths: 11 to 20% more Ipopt iterations, with two failed solves becoming more expensive.
- **The temperature bounds (issue #13; Bjarne chose them on 2026-10-29).** Ipopt's temperature bounds were v1.0.0's, $[\min(T_s, T_{lg}), T_r + 1\ \text{K}]$, which no root of v1.0.0's model reaches but roots of `develop`'s do. Frictional heating (THM-6) warms the flow above $T_r + 1$: at issue #13's well, 3 inch tubing at a high rate, the operating point is 3.2 K above $T_r$ at its warmest. Lift gas hotter than the reservoir does the same, and a surface warmer than the reservoir makes $T_0 = T_r$ fall below $T_s$. In each case the CasADi search found no root where the Rust core, which has no temperature bounds, finds one. The bounds are now $\min(T_r, T_s, T_{lg})$ and $\max(T_r, T_s, T_{lg}) + 50$ K (`T_MARGIN`), with the Joule–Thomson floor of 016 below where it is lower. Frictional heating warms the liquid by at most its frictional pressure drop over $\rho_l c_{pl}$, about 20 K for 300 bar.
  - **The verifier's case set**, against v1.0.0's bounds: the same 200 roots in the same 136 cases, within $7\cdot10^{-7}$; PASS and 100% for both candidates. Ipopt's iterations rise by 6.7% (54,568 against 51,163), and 515 starts reach a root against 512. Run back to back twice on 12 processes, the search takes 1.15 and 1.16 s per case against 1.28 and 1.32 s, and the candidate 26 and 27 s against 39 and 40 s.
  - **Rejected for them:** no upper bound, which costs 38% more iterations (70,698) and 27% more search time: 18 starts that found a root ran to Ipopt's 3000 iterations instead. A margin of 10 K costs about as much as 50 K (55,340 iterations) but lies within reach of frictional heating.
  - **Sampled wells** (`plans/evidence/gor_cap_wells.py`, the 263 cases of 88 wells on the CasADi backend): the same 253 operating points, within $6\cdot10^{-7}$ bar, and 2% more time in total.
  - **A known limit.** Without the Joule–Thomson term, the gravity term (THM-7) cools the gas by up to $g/\min(c_{pg}, c_{pl})$, about 5 K per km of depth, so where the geothermal gradient is below that, a root can lie below $\min(T_r, T_s, T_{lg})$, and only the Rust core finds it. With $T_r$ = 300 K and $T_s$ = 310 K the core's operating point is 2.8 K below $T_r$. Realistic gradients are 15 to 40 K/km, and the sampler's 32 to 37 K/km (SMP-1, SMP-14, SMP-15); Bjarne kept the lower bound as it is (2026-10-29).
- **Rejected:** KINSOL in the march (89.7%, against 95.6% with Newton, both with method A's starts and before the canonical gas row), Newton with sign constraints (no effect), and padding a failed march with its last state (100%, but two accepted roots violated mass conservation, at twice the time of the Ipopt fallback). Not proposed: an Ipopt iteration cap of 1000, which would save 19% of the search time with no root lost on the case set.

## Off in the `v1.0.0` configuration

Not a model option. The `v1.0.0` configuration is checked by the verifier with every check, including Root set and Stability.

## Acceptance

The verifier on `develop`'s candidate (`scripts/verification/develop_candidate.py`, CI job `verify`): PASS with no expected failures, 100% stable-root rate. `tests/test_roots.py` (the copies of the verifier's constants, SOL-1, SOL-4 to SOL-6), `tests/test_simulator.py` (two-root well, `NoOperatingPoint`), `tests/test_model_properties.py`.

## Out of scope

An early exit for `simulate` (it runs every start), dual warm starts, JIT compilation, a Newton polish of the full system (`plans/improvements.md` §4.2), and the Rust core's shooting search (`specs/architecture.md`).
