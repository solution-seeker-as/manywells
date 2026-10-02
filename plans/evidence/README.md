# Evidence for the v2 plan

Scripts and results behind the numbers in `../manywells-v2-plan.md`: the "Known v1 defect" bullet, the "Multiple roots and stability" section, and Step 2.4. Feature specs 016 and 017 have their own sections below. The v2 plan's scripts were run on 2026-09-30 against v1.0.0 (Python 3.11.13, casadi 3.6.4, numpy 1.24.3, pandas 1.5.3), using the published `manywells-sol-1` data and configs.

## Setup

The scripts use v1.0.0's API, not `develop`'s. Run them from the root of a v1.0.0 checkout, or set `MANYWELLS_V1` to one:

```console
git worktree add --detach ../manywells-v1 v1.0.0
cd ../manywells-v1
uv sync --frozen    # --locked re-resolves, and fails, if a global uv exclude-newer is set
.venv/bin/python <repo>/plans/evidence/stability_label.py --config <data>/manywells-sol-1_config.zip 977
```

`manywells-sol-1.zip` and `manywells-sol-1_config.zip` are in `data/` of the Hugging Face dataset `solution-seeker-as/manywells`.

## Scripts and results

| Script | What it shows | Result |
|---|---|---|
| `stability_label.py 977` | The stability label from v1's residual graph: drop the choke row, treat `p_0` as a parameter, and take the sign of d(choke row)/d`p_0` from one linear solve | v1's default guess lands on the trickle root (`p_0` = 208.02 bar, TWH 285.5 K, d/d`p_0` = +285: unstable). Other guesses reach `p_0` = 169.00 bar (TWH 332.2 K, −2.43: stable). |
| `root_sets.py 40 10` → `root_sets_sol1_50.csv` | Root sets from v1 run from six starting points, on 40 wells that meet the two-root criterion and 10 that do not (seed 1) | Every two-root well has one stable and one unstable root, the unstable one at higher `p_0`; every one-root well's root is stable. v1's default solve returns the trickle root in 1 of the 40 (well 1701). Median 3.1 s per well for the six starts. |
| `choke_sweep.py 41 977` | Why a v1-only sweep of the choke row over `p_0` cannot check completeness | The row is NaN where `p_L < p_s`, which is next to the trickle root, and the cellwise march fails at high rates. The sweep finds only the stable root. |
| `trickle_signature.py` | The trickle-root signature (PWH within 10 mbar of PDC) in the published data | 3.53% of samples, from 131 of 2,000 wells, with median TWH 279.4 K. They make up 33.5k of the 44.2k samples between 275 and 285 K. |

## Solver versions matter

The same batch was first run with casadi 3.8.1, the environment on `rust_implementation`. The roots agree with the v1.0.0 run to within 1e-7 K in TWH. The one difference is well 1847, where v1's default solve failed under 3.8.1 and converges to the stable root under 3.6.4. That is why Step 2 pins the CasADi version along with `manywells==1.0.0`.

## Feature 017: calibration

Scripts behind the measurements in `../../specs/features/017-calibration.md` that the twin study (`scripts/calibration/twins.py`) does not make, run on 2026-10-02 on the branch `calibration`, with 24 CPUs. Run from the project root as `uv run python plans/evidence/calibration_measurements.py <part>`.

| Part | What it shows | Result |
|---|---|---|
| `rows` | The Rust core's time for a row (the full search) on four sampled wells at 20, 50 and 100 cells, and 16 rows in 1 to 16 threads | 32 to 810 ms per row; 16 threads are 12 times faster than one |
| `step` | The forward difference of the predicted observations in the log of each parameter, for steps from $10^{-1}$ to $10^{-8}$ | Four digits from $10^{-4}$, five from $10^{-5}$ to $10^{-8}$: the Jacobian's step of $10^{-6}$ is safe |
| `valley` | Well 3 of seed 2026, periodic tests: the fit from the medians alone, with $\lvert z\rvert \le 4$, and from the start search | From the medians it stops at a roughness of 143 m (cost 119.6); with bounds at the bounds (cost 23,446); from the start search in the truth's basin (cost 27.6, 798 solves against 2,793) |
| `backends` | The root sets of both backends at the truth, at the rows of the backend comparison's wells 7 and 1 | Well 7: at $u = 0.2$ the CasADi search has two stable roots where the core has one. Well 1: it has no root at two rows where the core has a stable one |

## Feature 016: Joule–Thomson cooling

Scripts behind the measurements in `../../specs/features/016-joule-thomson.md`, run on 2026-10-02 against `develop`. `jt_wells.py` is the prototype that chose the design: it adds the term by a subclass of `ThermalModel`, so it runs on the CasADi backend only, with Papay's gas law. `dak_jt_default.py` measures the implemented feature.

| Script | What it shows | Result |
|---|---|---|
| `jt_factor.py` | The gas's Joule–Thomson factor $J = T(\partial\ln Z/\partial T)_p$ from Papay's z-factor and from the Dranchuk–Abou-Kassem equation of state, against CoolProp's reference equations of state for methane and two natural gases, over 5 to 460 bar and 280 to 425 K. Needs CoolProp: `uv run --no-project --with CoolProp --with numpy --with scipy python plans/evidence/jt_factor.py` | DAK follows the reference to 460 bar (within 0.07 for methane); Papay's factor has the wrong sign above about 300 bar. Papay's $Z$ at 460 bar is 11% to 29% above the reference. |
| `jt_wells.py 88 OUT.csv` | The term on 88 wells of `develop`'s sampler (seed 2026), three operating points each, with DAK's or Papay's factor, and with the cell's pressure gradient in place of $F + \rho_m g\cos\theta$ | See the feature spec's Measurements. About 20 minutes on 24 cores with one BLAS thread per process. |
| `dak_jt_default.py OUT.csv` | The implemented feature on the same cases, solved by the Rust core in four variants of `develop`'s default: Papay without the term (the old default), DAK without it, Papay with it, and DAK with it (the new default); with the core's time and work counters on one process | Each variant has an operating point in 259 of 263 cases. The term lowers TWH by 7.5 K at the median and by up to 45.9 K; the gas law alone moves PWH by up to 6.2 bar. About 5 minutes. |
