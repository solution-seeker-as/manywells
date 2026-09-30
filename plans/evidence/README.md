# Evidence for the v2 plan

Scripts and results behind the numbers in `../manywells-v2-plan.md`: the "Known v1 defect" bullet, the "Multiple roots and stability" section, and Step 2.4. They were run on 2026-09-30 against v1.0.0 (Python 3.11.13, casadi 3.6.4, numpy 1.24.3, pandas 1.5.3), using the published `manywells-sol-1` data and configs.

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
