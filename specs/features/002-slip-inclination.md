# 002 · Inclination in the slip model

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 2 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

In an inclined pipe, gas bubbles migrate to the upper wall: the transition from bubbly to slug flow comes earlier, and the Taylor bubble rises at a different velocity than in a vertical pipe (Hasan, Kabir and Sayarpour, 2010).

## Delta

- `slip.md`: SLIP-10, the deviation factor $\sqrt{\cos\theta}\,(1 + \sin\theta)^{1.2}$ on the Taylor-bubble rise velocity (Hasan et al. (A-10)); SLIP-11, the bubbly–slug threshold $0.25\cos\theta$ in the classifier's fourth feature. The classifier's weights are not refitted.
- The slip model's four regime constants become fields of `SlipModel`, with v1.0.0's values as defaults (`plans/improvements.md` §2.7).
- Step 7 dropped the guard $\cos\theta + 10^{-9}$ in the square root. GEO-3 already rules out $\cos\theta < 0$, and without the guard the factor is exactly 1 in a vertical cell. Signed off by Bjarne, 2026-10-01.
- Code: `slip.classify_flow_regime`, `SlipModel.identify_parameters`.

## Off in the `v1.0.0` configuration

Both terms are exactly v1.0.0's at $\cos\theta = 1$, which the `v1.0.0` configuration requires (vertical grid); the slip constants must be the defaults (`configurations.check`).

## Acceptance

- The SLIP-2/SLIP-3 vectors from v1.0.0 now pass exactly (they were a strict expected failure with the guard).
- The develop vectors of SLIP-10 and SLIP-11, and `tests/test_slip.py` (factor 1 in a vertical cell, no Taylor rise in a horizontal one, an earlier slug transition when inclined).

## Out of scope

Refitting the classifier for inclined pipes, and the four-regime model (after the plan; `scripts/flow_regimes/new_flow_regime_model.md`).
