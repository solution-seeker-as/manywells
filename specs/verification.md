# Verification

*Step 2 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: in use; tolerances and case set confirmed 2026-09-30.*

The verifier checks a candidate's roots, for each case in a fixed case set, against reference root sets computed from ManyWells v1.0.0. It holds no model: it depends on neither `manywells` nor CasADi, and it does not re-implement the equations. It checks `develop` in its v1-compatibility configuration (Step 7) and the Rust port (Step 8). How new model versions are checked, without a reference, is in the plan's "New model versions". The Distributions check is built in Step 3, and the real-well accuracy check is private.

Code: `verification/src/manywells_verify/`. Data: `verification/data/`. Build scripts: `verification/build/` and its README.

## State and cases

A root is the full state on the case's grid: `[p, v_g, v_l, alpha, rho_g, rho_l, T]` (bar, m/s, m/s, -, kg/m³, kg/m³, K) at each of the N + 1 grid points from bottomhole to wellhead, 7(N + 1) values in v1.0.0's order. A case is one well at one operating point on one grid, given by v1.0.0's well properties and boundary conditions:

| Parameter | Unit | Parameter | Unit |
|---|---|---|---|
| `L`, `D` | m | `f_g` | - |
| `rho_l` | kg/m³ | `K_c` | m² |
| `R_s`, `cp_g`, `cp_l` | J/(kg K) | `cpr` | - |
| `f_D` | - | `p_r`, `p_s` | bar |
| `h` | W/(m² K) | `T_r`, `T_s` | K |
| `w_l_max` | kg/s (Vogel) | `u` | - |
| `k_l` | kg/s/bar (productivity index) | `w_lg` | kg/s |

plus the variant (inflow `vogel` or `pi`; choke `simpson` or `bernoulli`; choke profile `linear`, `sigmoid`, `convex` or `concave`) and N.

## Case set

141 cases with 200 reference roots, committed in `verification/data/` (2.9 MB). Every random draw is seeded from the case's source, well and draw number, so the set can be rebuilt exactly or enlarged. 158 cases were drawn; 17 were left out (below).

| Source | Cases | Reference roots | Purpose |
|---|--:|---|---|
| `sol-1 u = 0.5` | 18 | 2 in 17, 1 in 1 | sol-1 wells whose first solve at u = 0.5, the generator's, lands on the trickle root (15), and stable ones (3) |
| `sol-1 drawn` | 39 | 2 in 9, 1 in 29, 0 in 1 | sol-1 wells at operating points drawn as the generator drew them, by two-root criterion and gas lift |
| `fresh sample_well` | 9 | 2 in 5, 1 in 4 | new wells from v1's sampler, for the failures and mostly-choked wells the generator's filters removed |
| `near fold` | 3 | 2 in 2, 1 in 1 | p_r just above where a well's two roots merge |
| `past fold` | 4 | 0 | p_r just below it |
| `gas lift on two-root well` | 10 | 2 in 1, 1 in 9 | gas lift on wells that meet the two-root criterion without it |
| `nsol-1 final state` | 17 | 2 in 16, 1 in 1 | nsol-1's stored final states, with the friction factor recovered from the stored state (below) |
| `synthetic variant` | 5 | 2 in 2, 1 in 3 | Bernoulli choke and productivity-index inflow, which no dataset uses |
| `convergence` | 36 | 2 in 12, 1 in 24 | 12 drawn cases at N, 2N and 4N |

There are no closed-loop cases (decided 2026-09-30). nsol-1's configs store `f_D = 0.05` for every well, although the generator draws it from U(0.01, 0.08); each nsol-1 case uses the friction factor recovered from its stored final state, which then solves v1's equations to 1e-7 or better.

**Reference root sets.** Every reference root is a v1.0.0 solution. Two searches find them. Method A solves v1 from its default guess, the dataset generator's start where there is one, the interpolated root one grid coarser for convergence members, and cellwise guesses at `p_s + f (p_r - p_s)` for f in {0.5, 0.7, 0.85, 0.95, 0.995}. Method B solves v1 from each root the Rust solver on `rust_implementation` finds. A root is accepted if Ipopt reports `Solve_Succeeded` and it passes Invariants.

The label is the sign of d(choke row)/dp0 along the manifold where every other row of v1's system holds, from one solve with v1's Jacobian with the choke row and the p0 column removed. It is normalized by `(p_r - p_s) / w_m`. Positive is unstable, negative stable. A magnitude of at most `label_min` is indeterminate.

A case is in the case set only if both methods give the same accepted roots, no label is indeterminate, and no v1 solution was rejected. Every other case is left out (Bjarne, 2026-09-30). In this build that left out 17: 13 where v1's own starts missed a root the Rust starts found, 1 where Rust missed roots v1 found, 2 near the fold with two stable roots (the model has more than two roots there), and 1 where v1 returned `Feasible_Point_Found` at a tangent point next to the fold. The list is written to `data/build/disagreements.md` on a rebuild. Rust returns at most two roots per case, so it does not test the assumption that there are at most two independently.

## Files

| File | Rows | Columns |
|---|---|---|
| `cases.parquet` | one per case | `schema_version`, `case_id`, `model`, `source`, `inflow`, `choke`, `profile`, `n_cells`, `group`, one column per parameter, `config_id`, `seed`, `note` |
| `reference_roots.parquet`, candidate files | one per root | `case_id`, `root`, `x` (list of 7(N + 1) floats), `label` (`stable`, `unstable`, `indeterminate` or empty), `operating_point`, `choked` (nullable); other columns are ignored |

A case with no rows in `reference_roots.parquet` has no root: the well cannot flow there. A case with no rows in a candidate file means the candidate reported no root. `group` links the N, 2N and 4N members of a convergence group. `v1_cold.parquet` holds v1.0.0's solutions from its default guess (for convergence members, from the interpolated coarser root), and `v1_dataset.parquet` its solutions from the start the dataset generator used. `verification/expected_failures.csv` lists v1.0.0's expected failures: Operating point where its default guess reaches the trickle root (20) or fails (15), written by `build/expected_failures.py`.

## Checks

The **state distance** between two states on the same grid is a scaled ∞-norm over every value: p by `p_r - p_s`, velocities and densities relative to the reference state (velocity floor 1e-3 m/s), alpha absolute, and T by `T_r - T_s`. A candidate root *matches* the nearest reference root if their distance is at most `tol_x`.

Per root:

- **Invariants.**
  - `0 ≤ alpha ≤ 1`.
  - Densities and velocities are positive.
  - Phase mass rates are constant along the well within `flux_rel` of the total rate.
  - Pressure never rises between neighbouring points by more than `p_slack`.
  - `p_N > p_s` and `p_0 < p_r`.
  - Temperature is not below the ambient profile `T_r - i (T_r - T_s)/N` by more than `T_slack`.
  - If the candidate reports CHOKED, it agrees with v1.0.0's `is_choked` (`p_s ≤ cpr p_N`) outside a dead band of `choke_band`.

Per case:

- **Operating point.** With one stable reference root, the candidate's operating point must pass Invariants and match that root. With no stable reference root, the candidate must report no operating point. Two stable reference roots are indeterminate until `specs/model/solution.md` settles that case.
- **Root set**, for candidates that report more than one root or any label. Every reference root must be matched by a valid candidate root.
- **Stability**, for candidates that label their roots. Each label must equal the label of the reference root that the root matches.
- **Findings.** A valid candidate root that matches no reference root is a finding for Bjarne, not a failure: either both searches missed it or the candidate is wrong, and the verifier cannot tell which.

Per convergence group:

- **Convergence.** For PBH, PWH, TWH and the phase rates at the bottom, the observed order `log2(|o_N - o_2N| / |o_2N - o_4N|)` must lie within `order`. Outputs whose change is at the noise level are not used, and groups within `conv_choke_margin` of the choke switch are skipped.

Each check is `pass`, `fail`, `indeterminate` or `n/a`. A failure listed in the expected-failure file (`case_id, check, reason`, with the group id for Convergence) counts as expected. The verdict is PASS when there are no unexpected failures.

## Tolerances

Confirmed by Bjarne on 2026-09-30, from the measurements of `build/tolerance_stats.py` on the case set:

| Name | Value | Meaning | Measured on the reference |
|---|---|---|---|
| `tol_x` | 1e-4 | state distance to a reference root (about 0.01 bar and 0.01 K) | one root reached from different v1 starts: at most 3.4e-6 apart; distinct roots: at least 0.10 apart |
| `p_slack` | 1e-6 bar | pressure rise between neighbouring points | pressure falls by at least 0.24 bar per cell |
| `flux_rel` | 1e-6 | phase mass-rate variation / total rate | at most 1.8e-8 |
| `T_slack` | 1e-6 K | temperature below ambient | at most 2e-10 K |
| `choke_band` | 5e-4 bar | CHOKED dead band | the smooth max in v1's choke row is about 5e-4 bar wide |
| `label_min` | 1e-3 | indeterminate label threshold, applied when the reference is built | smallest \|normalized dR/dp0\| in the case set: 8.6 |
| `order` | 0.8 to 1.25 | observed convergence order | 0.95 to 1.11 over 9 groups |
| `conv_choke_margin` | 1 bar | convergence groups this close to the choke switch are skipped | 3 of 12 groups |

## Report

`manywells-verify CANDIDATE --data DIR [--expected-failures CSV] [--json OUT]` prints a one-screen Markdown report. It exits with status 1 on unexpected failures, and `--json` writes every check of every root and case. The report contains:

- the verdict;
- the **stable-root rate**: of the cases with one stable reference root, the share where the candidate's operating point is that root;
- a table of counts per check;
- the largest distance of a passing operating point;
- unexpected failures, expected failures now passing, and findings for review.

## v1.0.0 on its own reference

`manywells-verify verification/data/v1_cold.parquet --data verification/data --expected-failures verification/expected_failures.csv` gives PASS: no unexpected failures, 35 expected failures, and a stable-root rate of 74.3% (101 of 136 cases with a stable root). The case set is harder than the datasets on purpose: it holds the trickle-root wells, the fold, and fresh wells the generators' filters would have removed. Started from the dataset generators' own starts instead of its default guess, v1 returns the stable root in all 67 cases that have such a start. The CI job `verify` in `.github/workflows/tests.yml` runs this command on every push and pull request.
