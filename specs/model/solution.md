# Solution set and operating point

## Purpose

What the model's answer is. The discretized system (DISC-6) can have several roots, and not all of them are states a well can operate at. This file defines the admissible states, the root set, a stability label for each root, and the operating point. The paper does not define these; they are Bjarne's decisions (plan, Step 4). Which root a solver happens to converge to is not part of the model.

## Interface

A case (well, operating point and grid) has a root set: zero or more states on its grid, each with a label `stable`, `unstable` or `indeterminate`. The operating point is at most one of them. Units are those of `nomenclature.md`.

## Equations

### SOL-1 · Admissible states

A state $x$ is admissible if at every grid point

$$p_s \le p_i \le p_r, \qquad 0 \le \alpha_i \le 1, \qquad v_{g,i},\ v_{l,i},\ \rho_{g,i},\ \rho_{l,i} > 0.$$

The strict positivity rules out the zero-flow state, a static liquid column with $v_l = 0$, but not the trickle root, whose velocities are small and positive. Temperature is not constrained. In `v1.0.0` with $T_s \le T_r$, every root satisfies $T_{a,i} \le T_i \le T_r$: DISC-5 with THM-1 makes each $T_i$ a convex combination of $T_{i-1}$ and $T_{a,i}$, starting from $T_0 = T_r$, and $T_a$ falls towards the wellhead.

### SOL-2 · Root set

The root set of a case is the set of admissible states at which every row of DISC-6 is zero. At every root $w_m > 0$, so the choke row (CHK-1) needs $p_N > p_c \ge p_s$, and the inflow (INF-1, INF-2) needs $p_0 < p_r$.

### SOL-3 · Stability label

Write the state as $x = (p_0, y)$, with $p_0$ the bottomhole pressure and $y$ the other $7(N+1) - 1$ unknowns. Remove the CHK-1 row from the system; the remaining rows $\tilde r(p_0, y) = 0$ define $y(p_0)$ near a root, where $\partial\tilde r/\partial y$ is nonsingular. The shooting residual is the CHK-1 row along that curve,

$$R(p_0) = r_\text{CHK-1}\big(p_0, y(p_0)\big) = w_m - w_c,$$

and its slope at the root is

$$s = \frac{dR}{dp_0} = \frac{\partial r_\text{CHK-1}}{\partial p_0} - \frac{\partial r_\text{CHK-1}}{\partial y}\left(\frac{\partial \tilde r}{\partial y}\right)^{-1}\frac{\partial \tilde r}{\partial p_0}.$$

The root is **stable** if $s < 0$ and **unstable** if $s > 0$. At $s = 0$, a fold where two roots merge, the label is **indeterminate**.

This is static (nodal-analysis) stability. A lower $p_0$ means a higher rate from the reservoir. At a stable root, a small increase in rate makes $R > 0$: the tubing delivers too low a wellhead pressure for the choke to pass that rate, and the flow falls back. At an unstable root the same increase makes $R < 0$, and the flow runs away. Heading and other dynamic instabilities are outside a steady-state model.

The sign of $s$ does not change if the CHK-1 row is multiplied by a positive function, such as the squared form of CHK-11, and it does not depend on the form or scaling of the other rows, because they define the same curve $y(p_0)$. The verifier normalizes $s$ by $(p_r - p_s)/w_m$ and treats labels of small magnitude as indeterminate (`specs/verification.md`, `label_min`).

### SOL-4 · Operating point

If the root set has exactly one stable root, the operating point is that root.

### SOL-5 · No stable root

If the root set has no stable root (no root at all, or only unstable ones), there is no operating point: the well cannot flow at these conditions. `simulate()` raises in that case, and the root set remains available (`specs/goals.md`).

### SOL-6 · More than one stable root

Decided by Bjarne, 2026-09-30 (`specs/discrepancies.md`, D-20). If the root set has more than one stable root, the operating point is the stable root with the lowest $p_0$, which is the highest rate. Every root stays in the root set with its label, and the case is flagged as having several stable roots.

Step 2 found this case near the fold: `fold-1505` has stable roots at $p_0$ = 115.547 and 115.908 bar, and `fold-0485` at 149.537 and 151.378 bar (`verification/data/build/disagreements.md`). Between two stable roots there must be at least one unstable root, so these cases have at least three roots, and neither root search resolved all of them. Both cases are out of the case set.

Why the lowest $p_0$: of the stable roots, it is the one a flowing well reaches when its rate falls from a higher rate, for example after kick-off or when it is choked back. Starting from shut-in, a well whose static column reaches the separator settles at the stable root with the highest $p_0$ instead, so the choice is a convention, not physics.

### SOL-7 · Two-root criterion

A tested property, not a definition. Without gas lift, a well has two roots, one stable and one unstable at higher $p_0$, if

$$p_r - p_s < \rho_l\, g\, L / c_\text{bar},$$

that is, if a static liquid column cannot reach the separator; otherwise it has one root, which is stable. The criterion split all 2,000 `sol-1` configs, solved with the Rust port (`plans/solver_description.md` §7), and held on 50 configs solved with v1.0.0 from six starts each (`plans/evidence/root_sets.py`). It fails near the fold (SOL-6) and is untested with gas lift, which can remove the trickle root: in Step 2's case set it did so in 9 of 10 two-root wells.

## Informative: v1.0.0's solver

Not part of the model; recorded because the verifier records which root v1.0.0 returns from its default guess (`specs/verification.md`, `v1_cold.parquet`).

- v1.0.0 solves DISC-6 as a feasibility NLP with Ipopt (objective zero), with bounds $p \in [p_s, p_r]$, $\alpha \le 1$, $T \in [T_s, T_r + 1]$ and all other unknowns $\ge 0$. The temperature bounds are never active at a root (SOL-1).
- Its default initial guess marches up the well: $p_0 = p_r - 0.05\,(p_r - p_s)$ and $T_0 = T_r$; the bottom point's $(v_g, v_l, \alpha)$ from INF-6, INF-7 and SLIP-1 by Ipopt, starting from $\alpha = 0.5$; then each point from DISC-2 to DISC-5 and the closures by Ipopt, starting from the previous point. A user-supplied `x_guess` replaces the march.
- It returns whichever root Ipopt reaches, which is the unstable trickle root in some cases (20 of the 141 cases in the case set). The dataset generators' warm starts are described in `specs/sampling.md`.

## Informative: `develop`'s root search

Not part of the model: only the operating point and the root set are specified (principle 6). Recorded so that a reader can follow `manywells.solvers` next to this file (principle 7).

- `SSDFSimulator(wp)` builds the well's system once (DISC-11), with the operating point as parameters, and an Ipopt feasibility NLP on it. `root_set(bc)` solves it from up to eight starts: a given guess (`x_guess`), the default march from $p_0 = p_r - 0.05\,(p_r - p_s)$, and marches from $p_0 = p_s + f\,(p_r - p_s)$ for $f$ in 0.5, 0.7, 0.85, 0.975, 0.995 and 0.999. The march solves point 0's rows at fixed $p_0$, then each point's rows up the well, by Newton's method, and by Ipopt with v1.0.0's bounds where Newton fails.
- A solve is accepted if Ipopt reports `Solve_Succeeded`, as the verifier's reference build requires, and the state is admissible (SOL-1). Solutions within the verifier's `tol_x` of each other are one root. Each root is labelled by SOL-3 from the residual's Jacobian, with the verifier's `label_min`. `simulate(bc)` returns the operating point of the root set (SOL-4 to SOL-6) and raises `NoOperatingPoint` without one.
- Measured on the verifier's case set in the `v1.0.0` configuration (2026-10-01): every reference root found with the right label and no other root, a stable-root rate of 100% against v1.0.0's 74.3%, at 0.8 s to build a well's system and 1.6 s to search, per case. The starts after v1.0.0's (method A of the reference build) and the march's fallback are solver machinery, with their measured gains in `manywells/solvers/roots.py` and `march.py` (`plans/manywells-v2-plan.md`, Step 7).

## Informative: the Rust core's search

Not part of the model; recorded so that a reader can follow `rust/src/shoot.rs` and `march.rs` next to this file (principle 7). Details and measured gains are in `specs/features/014-rust-solver.md`.

- **The search.** `SSDFSimulator(wp, backend='rust')` covers the `v1.0.0` configuration only, and shoots on $p_0$:
  - Given $p_0$, the march solves every row but CHK-1 point by point up the well, so $R(p_0)$, the CHK-1 row at the wellhead, is SOL-3's shooting residual.
  - Once the pressure falls below $p_s$, $R = w_m$ exactly (CHK-11).
  - The roots are the sign changes of $R$ on a scan of 101 points over $(p_s, p_r)$, with a ladder of halving drawdowns in the top interval and a search around each local minimum of $R \ge 0$.
  - Brent refines each sign change in the drawdown $p_r - p_0$.
- **Acceptance and label.** A root is accepted if its march solved every row and $|R| \le 10^{-3} w_m$. Its slope comes from a central difference of $R$, labelled by SOL-3 with the verifier's `label_min`.
- **Measured** on the verifier's case set (2026-10-01): every reference root found with the right label, a stable-root rate of 100%, and 65 ms per case on one core, with no build.

## Informative: roots on several void-fraction branches

Where SLIP-1 has several roots in $\alpha$ (`slip.md`, Open question), the root set holds roots on different branches at some points. A march follows one root per point, so the Rust core and the CasADi backend's starts each find some of them, and neither finds them all. SOL-6's choice of the lowest stable root then depends on which roots were found: at `sol-1` wells 91 and 95 (k = 2) of Step 8's regenerated samples, the two backends' stable roots differ by 0.20 and 0.29 bar, and each is a root of every row. Open, with `slip.md`.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| SOL-1 | — | `simulator.py` `simulate` (bounds) | verifier: Invariants |
| SOL-2 | §3.3 | `simulator.py` `simulate` | verifier: Root set |
| SOL-3 | — | `plans/evidence/stability_label.py`, `verification/build/label_cases.py` | verifier: Stability |
| SOL-4 | — | — | verifier: Operating point |
| SOL-5 | — | — | verifier: Operating point |
| SOL-6 | — | `verification/src/manywells_verify/checks.py` `check_operating_point` | verifier: Operating point (tested in `verification/tests/test_checks.py`, as the case set has no such case: both were left out for incomplete root sets) |
| SOL-7 | — | — | property: `plans/evidence/root_sets.py`, `plans/solver_description.md` §7; spec-only: a property of the model, not an equation |
