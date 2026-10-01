# 013 · Components, frozen inputs, configurations and the API

*Feature spec for Step 7 of `plans/manywells-v2-plan.md` (2026-10-01), with change 13 of `plans/develop_model_changes.md`; implements `specs/architecture.md`. Status: draft for Bjarne's sign-off.*

## Motivation

`specs/architecture.md` (decided 2026-09-30) splits the simulator into components, discretization, solvers and solution, makes the inputs immutable, and builds a well's system once with the operating point as parameters. Not a model change.

## Delta

- **Components.** `WellProperties` holds one object per model part: `geometry`, `fluid`, `friction` (new), `thermal` (new; `h` moved here), `slip`, `inflow`, `choke`. The choke takes the wellhead state, `mass_flow_rate(u, p_s, s, A)`, and each model supplies its density and multiplier, so the simulator no longer dispatches on `isinstance` (`plans/improvements.md` §2.3).
- **CHK-11** is implemented: the choke passes no flow where $p_N \le p_c$, written with `if_else` so that the derivative is zero there, not NaN.
- **Frozen inputs** (§2.5, §2.1): every input dataclass is frozen and raises `ValueError` in `__post_init__`; the default Bernoulli choke is set by `WellProperties`, not by the simulator.
- **Discretization** (`discretization.py`): `build_system(wp)`, with the rows of DISC-11 defined once and used by the full residual and by the march; the operating point is the parameter vector $[p_r, p_s, T_r, T_s, T_{lg}, u, w_{lg}]$ (§4.1).
- **Configurations** (`configurations.py`): `v1_well(...)` from v1.0.0's parameters, and `check(wp, 'v1.0.0' | 'develop')`.
- **Closed loop** kept a frozen copy of the old simulator as its private base (`closed_loop/_base.py`; decision 2), pinned by `tests/test_closed_loop.py`, until `closed_loop/` was retired on 2026-10-01.
- **Logging**: no prints; failed starts are logged to `logging.getLogger('manywells')` (§2.2).

## Acceptance

`tests/test_simulator.py`, `tests/test_configurations.py`, `tests/test_closed_loop.py` (until closed loop's retirement), the row vectors, the verifier. Build-once speed-up: measured on the case set (`plans/manywells-v2-plan.md`, Step 7 status).

## Breaking changes

Each is in the CHANGELOG with an old→new snippet (`specs/goals.md`, API): `SSDFSimulator(wp)` with `simulate(bc)` returning a `Root` (the two-argument constructor works with a `DeprecationWarning`); `friction` and `thermal` objects in place of `f_D`, `roughness` and `h`; frozen inputs; `p_sep` and `p_bubble` in bar; `ChokeModel.mass_flow_rate(u, p_s, s, A)`; `FluidModel.surface_tension(p, T, rho_l)`; `ValueError` in place of `AssertionError`.

## Finding

The CI job `verify` piped the verifier into `tee` without `pipefail`, so it could not fail; Step 7 set `shell: bash` for the job.
