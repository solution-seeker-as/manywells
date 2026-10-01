# 008 · Gas dissolving into oil

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 8 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

With black oil (007), part of the reservoir gas travels dissolved in the oil and comes out of solution up the well, so the phase mass rates change along the well; their total does not (paper §8, limitation 1).

## Delta

- `pvt/oil.md`: PVT-OIL-13, the dissolved gas $w_d = \operatorname{smin}(R_{so}\rho_{g,\text{sc}}/\rho_o\, x_o w_\text{res},\ w_{g,\text{res}})$ and the phase rates $w_g = \operatorname{smax}(w_{g,\text{res}} + w_{lg} - w_d, 0)$, $w_l = w_\text{res} + w_d$; with dead oil the rates are INF-5's, exactly, with no smoothing. `FluidModel.phase_rates(p, T, w_res, w_lg)`.
- `balances.md`: BAL-10, $\Gamma = (1/A)\, dw_g/dz$.
- `discretization.md`: DISC-7 and DISC-8, the flux-difference mass rows $(\alpha\rho_g v_g)_i - (\alpha\rho_g v_g)_{i-1} - (w_g(p_i, T_i) - w_g(p_{i-1}, T_{i-1}))/A$, one form for every fluid (`specs/architecture.md`, decision 4). They replace `develop`'s local rows $A\alpha\rho_g v_g = w_g(p_i, T_i)$, with the same roots. Bjarne signed off the spec text of decision 4 on 2026-10-01.
- The reservoir liquid rate $w_\text{res}$ is a function of $x_0$, passed to every point's rows; the hidden state `self._w_l_inflow` is gone (`plans/improvements.md` §2.4).
- Code: `FluidModel.phase_rates`, `discretization.cell_rows`.

## Off in the `v1.0.0` configuration

With dead oil the rate difference in DISC-7 and DISC-8 is identically zero, so they are DISC-2 and DISC-3 as functions of the state, and BAL-3 holds. Before Step 7, the smooth min gave $-\epsilon/(4w_g)$ instead of 0 at $R_{so} = 0$, and `develop`'s INF-6 and INF-7 rows differed from v1.0.0's by up to $2.3\cdot10^{-7}$ relative; now they match.

## Acceptance

The DISC-2, DISC-3, INF-6 and INF-7 row vectors in the `v1.0.0` configuration (they were skipped or a strict expected failure); the develop vectors of PVT-OIL-13; `tests/test_fluid.py` (exact dead-oil rates, conservation); `tests/test_model_properties.py` (total mass rate constant, local phase rates at every point, free gas increasing up the well).

## Out of scope

Lift gas dissolving into the oil, gas coming out of the water, and non-equilibrium mass transfer.
