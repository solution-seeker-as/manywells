# 012 · Root set, stability label and operating point in the simulator

*Feature spec for Step 7 item 6 of `plans/manywells-v2-plan.md` (2026-10-01), with change 12 of `plans/develop_model_changes.md` (the initial-guess march). Status: Bjarne accepted the solver machinery under principle 7 on 2026-10-01: the starts 0.975 and 0.999, the march's fallback, the canonical gas-law row (006) and acceptance on `Solve_Succeeded`.*

## Motivation

The model's answer is a root set, and the operating point is its stable root (`specs/model/solution.md`, `specs/goals.md`). v1.0.0 and `develop` returned whichever root Ipopt reached from one start, which was the unstable trickle root in 20 of the verifier's 141 cases, and failed in 15.

## Delta

No change to `specs/model/`, whose SOL-1 to SOL-6 already define the answer; `solution.md` gets an informative section on `develop`'s search.

- `solution.py`: `Root` (state, label, normalized slope, CHOKED, flow regimes, reservoir rates), `RootSet` (roots sorted by $p_0$, operating point, the several-stable flag, the search record), `select_operating_point` (SOL-4 to SOL-6).
- `solvers/roots.py`: the multi-start search. Starts: an optional guess; the default march from $p_0 = p_r - 0.05\,(p_r - p_s)$; marches from $p_0 = p_s + f(p_r - p_s)$ for $f$ in 0.5, 0.7, 0.85, 0.975, 0.995, 0.999. A solve is accepted on `Solve_Succeeded` and SOL-1, solutions within `tol_x` are merged, and each root is labelled by SOL-3 with `label_min`, both copied from the verifier (`specs/architecture.md`, decision 5).
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
- **Rejected:** KINSOL in the march (89.7%, against 95.6% with Newton, both with method A's starts and before the canonical gas row), Newton with sign constraints (no effect), and padding a failed march with its last state (100%, but two accepted roots violated mass conservation, at twice the time of the Ipopt fallback). Not proposed: an Ipopt iteration cap of 1000, which would save 19% of the search time with no root lost on the case set.

## Off in the `v1.0.0` configuration

Not a model option. The `v1.0.0` configuration is checked by the verifier with every check, including Root set and Stability.

## Acceptance

The verifier on `develop`'s candidate (`scripts/verification/develop_candidate.py`, CI job `verify`): PASS with no expected failures, 100% stable-root rate. `tests/test_roots.py` (the copies of the verifier's constants, SOL-1, SOL-4 to SOL-6), `tests/test_simulator.py` (two-root well, `NoOperatingPoint`), `tests/test_model_properties.py`.

## Out of scope

An early exit for `simulate` (it runs every start), dual warm starts, JIT compilation, a Newton polish of the full system (`plans/improvements.md` §4.2), and the Rust core's shooting search (`specs/architecture.md`).
