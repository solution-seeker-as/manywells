# manywells_rs solver: findings and proposed improvements

*Draft 2026-09-29. Status: proposal. Reviewed code: branch `rust_implementation` @ 0e9e98b, mainly `manywells_rs/src/simulator.rs` and `manywells_rs/python/manywells_rs/__init__.py`.*

This note lists what a review of the Rust shooting solver found, with the evidence for each item and a proposed change. It feeds into Step 8 of `manywells-v2-plan.md` (bringing the Rust port under the verifier).

## How the findings were obtained

- A line-by-line JavaScript port of `manywells_rs/src/*.rs` was checked against the compiled crate on all 2,000 wells of `manywells-sol-1_config` (from Hugging Face) plus the default `WellProperties()`/`BoundaryConditions()`. The two agree on the number of roots for every well and on every root to within 1.1e-13 bar. All statistics below come from that port, run on the sol-1 configs.
- The CasADi comparison used `manywells/simulator.py` on the same branch, on 150 randomly chosen two-root wells (seed 0) plus well 977.
- "High-p0 root" and "low-p0 root" refer to the two roots of a two-root well. `simulate()` returns the high-p0 root first.

## Summary

| # | Finding | Impact | Proposed change |
|---|---|---|---|
| 1 | `simulate()[0]` is treated as "the old simulator's solution", but it is the unstable trickle root | Wrong well selection in data generation; misleading docs and plots | Return a stability label per root; select by label, not by position |
| 2 | The α fixed-point stop tests the step size, not the error | Low-p0 roots off by up to 0.32 bar | Error-based stop or a bracketed solve for α |
| 3 | The outer Brent tolerance is an absolute 1e-6 bar | Trickle roots with tiny drawdown are poorly resolved | Tolerance relative to drawdown; report a relative residual |
| 4 | Roots are accepted without checking R | A jump in R could be returned as a root | Add a residual acceptance test |
| 5 | The scan steps over dips narrower than (p_r − p_s)/100 and stops at the first negative region | Missed roots near the fold, and any third root | Refine the scan where R comes close to zero |
| 6 | `docs/simulator_in_rust.md` predates the current build | Confusing comparisons | Update numbers and answer its open questions |

Items 1 and 2 change results. Items 3 to 5 are robustness. Item 6 is documentation.

## 1. Solution order does not identify the CasADi solution

The package docstring says `simulate()[0]` "recovers the operating point the old simulator would converge to". On the sampled two-root wells that held for 12 of 151, including well 977. For the other 139, CasADi converged to the low-p0 root, which is `simulate()[-1]`. The high-p0 root is the statically unstable trickle state described in the background section below.

Code that relies on the order:

| Location | What it does | Effect |
|---|---|---|
| `scripts/data_generation/open_loop_nonstationary/generate_open_loop_nonstationary_well_data_rust_simulator.py:229-244` | Max-production check at u = 1 uses `simulate()[0]` and discards the well if `w_tot < 7` kg/s | The nsol generator samples its own wells, so the sol-1 configs serve as a proxy. At u = 1, 1,273 of 2,000 wells are discarded; with the lowest-p0 root, 71 would be. 1,202 of the 1,254 two-root wells are discarded only because `[0]` is the trickle root, so two-root wells are almost entirely filtered out of the Rust-generated nsol data. |
| same file, line 314 | `x_last = solutions[0]` is stored in the dump | The stored last state is the trickle root for two-root wells |
| `manywells_rs/python/manywells_rs/__init__.py:7-18` | Docstring and example use `[0]` as the primary solution | Readers pick the unstable root |
| `scripts/compare_simulators.py:252-262, 323` | `_primary_solution` and the no-reference fallback take `[0]` | Benchmark and fallback comparisons use the unstable root |
| `scripts/plot_simulator.py:22` | Plots `solutions[0]` | Shows the trickle state by default |
| `scripts/opt_with_rust_simulator.py:43-55` | Takes `solutions[1]` when there are two | Picks the stable root, but by position |

Proposed change:

- Return a stability label with each root. It costs nothing: with one crossing per bracket (true for every sol-1 well), a root from the right bracket `[p_neg, p_prev]` has dR/dp0 > 0 (unstable) and a root from the left bracket `[p_s + 1e-3, p_neg]` has dR/dp0 < 0 (stable).
- Fix the docstring and the comment at line 229 of the nonstationary script.
- Use the stable root in the max-production check. For stored datasets, keep all roots and add a stability column next to `solution_number`.
- Replace positional picks in the scripts with a picker that selects by label.

## 2. The α stopping rule measures the step, not the error

`solve_alpha` (`simulator.rs:199-231`) iterates α ← S(α) and stops when |α(k+1) − α(k)| < 1e-3, with at most 100 iterations. For a contraction with slope q = S′(α*), the remaining error after a step δ is about q/(1 − q)·δ.

Evidence:

- The iteration averages 3.0 steps and never fails at the 1e-3 tolerance, but converges slowly near α ≈ 0.7, where the classifier moves toward annular flow. In the 79 sampled states where a tight iteration failed (every fifth grid point of the low-p0 solution in the first 400 wells), S′ was 0.78 to 0.96. That puts the error at 3.5 to 24 times the tolerance. At well 6, cell 70, the solver accepts α = 0.7142 while the fixed point is 0.7194. All 79 states had a single fixed point.
- Root shifts against a bracketed reference solve: high-p0 roots at most 0.026 bar; low-p0 roots median 0.004 bar, 95th percentile 0.064 bar, maximum 0.32 bar.
- Tightening the tolerance alone makes it worse, because the iteration then hits the 100-iteration cap. Wells that lose every root: 20 at 1e-4, 39 at 1e-6, 81 at 1e-8, 160 at 1e-10.

Options:

1. **Bracketed solve.** Brent on g(α) = α − S(α) over [1e-6, 1 − 1e-6]: return 1e-6 if g(1e-6) ≥ 0 and 1 − 1e-6 if g(1 − 1e-6) ≤ 0. Tested: no failures, and the same root counts as the current code in all 2,000 wells. Cost: 3.1 times slower overall in the JavaScript port (14.3 s against 43.8 s for 2,001 configs).
2. **Error-based stop.** Estimate q from successive steps, stop when q/(1 − q)·|Δα| < tol, and raise the iteration cap. Untested.
3. **Accelerated iteration.** Aitken or Steffensen acceleration of the existing iteration, with option 1 as the fallback. Untested; likely close to the current cost.

Recommendation: option 3 with option 1 as the fallback, plus a regression test on a state near α ≈ 0.7 that must match the bracketed fixed point to 1e-8.

## 3. Trickle roots are resolved to an absolute 1e-6 bar

The outer Brent calls (`simulator.rs:464-476`) use xtol = 1e-6 bar. Most high-p0 roots have tiny drawdowns: 1,113 of 1,254 are below 1 bar and a quarter are below 0.04 bar. At the extreme, well 87 has a drawdown of 6.7e-5 bar and p_L − p_s = 1.5e-6 bar, the same size as xtol. There, |R| at the returned root is about 9e4 times w_m². Measured as |R|/(2w_m²), high-p0 roots have a median of 7e-4 and a 90th percentile of 0.33. For low-p0 roots the 90th percentile is 3e-8.

A related edge: the scan starts at p_r − 1e-6, so a root with a drawdown below 1e-6 bar is never bracketed.

Proposed change: solve the right bracket in the drawdown d = p_r − p0 with a relative tolerance, or set xtol proportional to the bracket's drawdown. Report |R|/w_m² per root so callers can see how well each root is resolved.

## 4. Roots are accepted without checking R

Roots are filtered only by the march's `failed` flag (`simulator.rs:468, 476`). Brent converges to any sign change, including a jump. A stopped march copies the last good state to the top, which makes R discontinuous. If Brent lands on such a jump, the returned point can sit on the non-failed side with R far from zero. This did not happen on the sol-1 configs (no stopped marches during shooting), but the guard is cheap.

Proposed change: accept a root only if |R| ≤ tol·(w_m² + (K_c σ(u))²·2ρΔp/Φ), i.e. relative to the size of the two terms of R.

## 5. The scan can step over a narrow dip or a third root

`shoot` samples R at 101 points and returns as soon as it has handled the first negative sample (`simulator.rs:441-483`). Two consequences:

- If R < 0 only on an interval narrower than one step, (p_r − p_s)/100, the scan can skip it and report no solution. This happens near the fold where the two roots merge. For well 977 the fold lies between p_r = 188.57 and 188.58 bar. At 188.58 the negative interval is 0.56 bar wide against a 1.19 bar step, and the scan still caught it, so the risk is confined to a few hundredths of a bar in p_r.
- A third root after the first negative region would be missed.

On the sol-1 configs the + − + shape held for every well: a full 101-sample scan found no sign changes beyond the returned roots, no sample was infeasible, and R(p_s + 1e-3) was always positive. Low priority.

Proposed change: when the scan finds no negative sample but R has a small positive local minimum between samples, run a golden-section search on R there before raising `SimError`.

## 6. `docs/simulator_in_rust.md` is out of date

- It gives the second root of well 977 as p0 = 168.99 bar. The current code returns 169.13 bar; the high root is unchanged at 208.02 bar.
- It describes α as solved "until convergence". The actual stop is the 1e-3 step test from item 2.
- Its two open questions have answers (see the background section): the ~280 K TWH spike comes from CasADi converging to the trickle root, and there is a physical reason for the second root.

## Background: why two roots, and which is stable

- **When there are two roots.** At zero rate the tubing is full of liquid. If p_r − p_s < ρ_l·g·L/10⁵, the static column cannot reach the separator and R > 0 near p0 = p_r. At moderate rates gas lightens the column and R turns negative. At high rates friction dominates and R rises again. On the sol-1 configs this criterion splits the wells without exception: all 1,254 two-root wells satisfy it and none of the 746 one-root wells do. If the dip never reaches below zero, there are no roots and the well cannot flow.
- **Which root is stable.** Take a quasi-static perturbation that raises the rate slightly, which lowers p0. At the high-p0 root dR/dp0 > 0, so R turns negative: the tubing delivers more wellhead pressure than the choke needs, and the flow accelerates further away. At the low-p0 root dR/dp0 < 0, so R turns positive and the flow decelerates back. The slope signs held in all 1,254 two-root wells. This is a static argument; heading-type dynamics are outside the model.
- **What the unstable root looks like.** Median drawdown 0.13 bar. The liquid rate is typically 217 times lower than at the stable root. The wellhead sits within a millibar of p_s (median 0.4 mbar). The median wellhead temperature is 278 K, against 347 K at the stable root.
- **Which root CasADi picks.** 139 of 151 sampled wells converged to the stable root and 12 to the trickle root. Those 12 have a median wellhead temperature of 279.5 K, which matches the TWH spike in the doc's histograms.
- **Controls that change the root count.** For well 977, lowering p_r merges the two roots near p_r ≈ 188.6 bar, and below that the well has no solution. A gas-lift rate of 0.3 kg/s removes the trickle root, because the column is gassy even at zero liquid rate.

## Relevance to the v2 plan

- **Step 2, case labelling.** Multiplicity and stability labels can be computed directly: the criterion above predicts whether a well has two roots, and the bracket a root came from gives its stability. That is cheaper and more reliable than inferring them from perturbed CasADi starts.
- **Step 2, invariants.** The trickle root has small but strictly positive velocities, so the "densities and velocities strictly positive" invariant does not exclude it. It is a genuine root of the model. The verifier needs an explicit rule for which roots a candidate must report.
- **Step 6, thermal model.** The Rust port integrates T(z) analytically, which decouples energy from momentum. That only works while the energy equation has no pressure-dependent terms. A richer v2 thermal model (for example with Joule–Thomson cooling) would need a per-cell temperature solve in the march.

## Cost profile (context for any speed work)

- A well needs 9 to 33 residual evaluations (median 21), because the scan stops at the first negative sample.
- Of all cell solves, 86% take the fast path (Brent on the narrow bracket), 5.6% the slow path (golden-section p*, then Brent), and 8.6% choke. Choked cells occur only in off-root trial marches, but each costs a golden-section search of about 20 evaluations of the cell equation.

## Suggested order of work

1. Item 1: stability label, fix the max-production check, docstring and positional picks. Regenerate any Rust-based nsol data produced with the current script.
2. Item 2: α solve, with a regression test near α ≈ 0.7.
3. Items 3 and 4 together, since both touch root acceptance in `shoot`.
4. Item 6, once 1 and 2 have landed so the numbers in the doc are final.
5. Item 5 if the verifier's case set turns up wells near the fold.
