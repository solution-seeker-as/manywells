# Simulating a well

This guide shows how to set up a well, solve it at an operating point and read the result, and how to generate
datasets like the published ones. [`scripts/sim_examples/`](../scripts/sim_examples/) has complete examples, and
[`specs/model/`](../specs/model/README.md) specifies the model.

## One operating point

```python
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator

wp = WellProperties()                            # a 2000 m vertical well with develop's default model
sim = SSDFSimulator(wp)                          # builds the well's system once
bc = BoundaryConditions(p_r=170, p_s=20, u=0.5)  # bar, and the choke opening in [0, 1]

op = sim.simulate(bc)                            # the operating point, a Root
print(f'p_0 = {op.p_0:.1f} bar, {op.label}, choked: {op.choked}')
df = sim.solution_as_df(op)                      # one row per grid point, from the bottomhole
```

`WellProperties` holds one object per part of the model, and each has a default:

| Field | Class | Default |
|---|---|---|
| `geometry` | `WellGeometry` (`geometry.py`): `WellGeometry.vertical(length, n_cells)` or `WellGeometry.from_survey(...)` for a deviated well | vertical, 2000 m, 100 cells |
| `fluid` | `FluidModel` (`pvt/fluid.py`) | black oil, real gas, no water |
| `friction` | `RoughnessFriction` or `FixedFrictionFactor` (`friction.py`) | pipe roughness, Chen's correlation |
| `thermal` | `ThermalModel` (`thermal.py`) | heat loss, frictional heating, gravity term, lift-gas mixing |
| `slip` | `SlipModel` (`slip.py`) | drift-flux closure per flow regime |
| `inflow` | `ProductivityIndex`, `Vogel` or `FixedFlowRate` (`inflow.py`) | `ProductivityIndex(k_l=0.5)` |
| `choke` | `BernoulliChokeModel` or `SimpsonChokeModel` (`choke.py`) | Bernoulli, `K_c` 10% of the pipe's cross-section |

`BoundaryConditions` is the operating point: the reservoir and separator pressures `p_r` and `p_s` (bar), the
reservoir and surface temperatures `T_r` and `T_s` (K), the choke position `u`, the lift-gas rate `w_lg` (kg/s)
and the lift-gas temperature `T_lg` (K; `None` means `T_r`).

The inputs are frozen dataclasses, validated when they are built, and the simulator does not change them. To vary
one part, build a new object, for instance with `dataclasses.replace`. Pressures are in bar and temperatures in
Kelvin throughout.

## The result

`simulate` returns a `Root` (`solution.py`):

- `x`: the state, 7 values per grid point from the bottomhole, in the order `[p, v_g, v_l, alpha, rho_g, rho_l, T]`
  (bar, m/s, m/s, -, kg/m³, kg/m³, K); `state` is the same as an (N + 1, 7) array, and `p_0` the bottomhole
  pressure
- `label`: `stable`, `unstable` or `indeterminate`
- `choked`: whether the flow through the choke is critical
- `flow_regime`: the flow regime at each grid point
- `w_res`, `w_g_res`: the reservoir's liquid and gas mass rates (kg/s), without the lift gas

`sim.solution_as_df(op)` gives the state as a DataFrame with one row per grid point, the measured and true vertical
depths `md` and `tvd` (m, from the wellhead), and a `flow-regime` column.

## Several roots

At some operating points the steady-state equations have more than one root, typically a stable operating point and
a statically unstable root with a small rate ([`specs/model/solution.md`](../specs/model/solution.md)).
`sim.root_set(bc)` returns every root the search finds, sorted by bottomhole pressure and labelled, and the operating
point:

```python
rs = sim.root_set(BoundaryConditions(p_r=130, u=0.3))
for r in rs.roots:
    print(f'p_0 = {r.p_0:.1f} bar, {r.label}')  # 106.8 bar, stable; 128.8 bar, unstable
op = rs.operating_point                         # the stable root, or None
```

The operating point is the stable root, or the one with the lowest bottomhole pressure if there are several.
`simulate(bc)` returns it, and raises `NoOperatingPoint` (a `SimError`) if there is no stable root, for instance
when the reservoir pressure is too low for the well to flow. The exception holds the root set in `root_set`.

## Many operating points of one well

`SSDFSimulator(wp)` builds the well's system once, with the operating point as parameters, so solving the same well
again is cheap. The root of a nearby operating point can be passed as an extra start; it makes the search faster but
does not change the answer:

```python
for u in (0.4, 0.6, 0.8):
    op = sim.simulate(BoundaryConditions(u=u), x_guess=op.x)
```

The two-argument form `SSDFSimulator(wp, bc)` with `simulate()` still works, with a `DeprecationWarning`, and
returns the state as a flat list. It is removed in v2.0.0.

## Backends

`SSDFSimulator(wp, backend=...)` solves the same model in one of two ways:

- `'casadi'` (the default) builds the system as a CasADi graph and searches for roots with Ipopt from several
  starts.
- `'rust'` uses the Rust core (`manywells._core`), which marches from the bottomhole to the wellhead and searches
  the bottomhole pressure for the roots. It implements every option of the model and is about 10 to 30 times faster
  per case.

Both take the same inputs and return the same types. The Rust core has a closed set of component classes, so a well
with a user's own subclass of a component, such as an `InflowModel`, raises `ValueError` there and needs the CasADi
backend. [`backends.py`](../scripts/sim_examples/backends.py) solves one well with each and compares the roots.

## The v1.0.0 configuration

The defaults above are `develop`'s model, which has more physics than ManyWells v1.0.0, the version described in
the paper and used to generate the published datasets. `manywells.configurations.v1_well` builds a well in the
`v1.0.0` configuration, which reproduces v1.0.0's model, from v1.0.0's parameters:

```python
from manywells.choke import SimpsonChokeModel
from manywells.configurations import v1_well
from manywells.inflow import Vogel

wp = v1_well(L=2500, D=0.1554, rho_l=850, R_s=520, cp_g=2225, cp_l=4000, f_D=0.05, h=20, f_g=0.1,
             inflow=Vogel(w_l_max=20), choke=SimpsonChokeModel(K_c=0.002), n_cells=100)
```

`configurations.check(wp)` raises a `ValueError` that lists every difference from the configuration.
[`load_well_from_dataset.py`](../scripts/load_well_from_dataset.py) builds a well of the published
`manywells-sol-1` dataset from its config file and simulates it.

## Examples

Run the examples as modules from the project root, for instance
`uv run python -m scripts.sim_examples.vertical_well`:

| Script | What it shows |
|---|---|
| [`vertical_well.py`](../scripts/sim_examples/vertical_well.py) | One operating point of the default well, with profiles along the well |
| [`L_shaped_well.py`](../scripts/sim_examples/L_shaped_well.py) | An L-shaped well against a vertical well of the same depth |
| [`fixed_rate.py`](../scripts/sim_examples/fixed_rate.py) | A fixed liquid rate (`FixedFlowRate`), where the bottomhole pressure follows from the rate |
| [`gl_temp.py`](../scripts/sim_examples/gl_temp.py) | A gas-lift study, with the lift-gas temperature's effect on the wellhead temperature |
| [`backends.py`](../scripts/sim_examples/backends.py) | One well with each backend, and their roots compared |

## Generating datasets

Two generators sample wells and operating points with `manywells.sampling`, which implements the procedures of
`specs/sampling.md`:

```console
uv run python -m scripts.data_generation.open_loop_stationary.generate_well_data \
    --wells 2000 --samples 500 --seed 1 --configuration v1.0.0 --out data/manywells-sol-v1cfg
uv run python -m scripts.data_generation.open_loop_nonstationary.generate_open_loop_nonstationary_well_data \
    --wells 2000 --samples 500 --seed 1 --configuration v1.0.0 --out data/manywells-nsol-v1cfg
```

The first follows the procedure of `manywells-sol-1` (stationary, open loop), the second that of `manywells-nsol-1`
(non-stationary, open loop). The options:

- `--wells` and `--samples`: the number of wells and of samples per well (default 2000 and 500)
- `--seed`: the dataset's seed (required); a run is reproducible whatever the number of processes
- `--configuration`: `v1.0.0` (the default) or `develop`
- `--processes`: the number of worker processes (default: the number of cores less two)
- `--out`: the path of the dataset files, without suffix

A run writes `<out>.parquet` (the rows, with the features of [`datasets.md`](datasets.md)),
`<out>_config.parquet` (each well's draws) and `<out>_meta.json` (the seed, the configuration and the code version).

Every sample is at the operating point, the stable root. A dataset in the `v1.0.0` configuration therefore follows
`sol-1`'s procedure but is not a copy of it: `sol-1`'s seeds were not recorded, and v1.0.0 sometimes reached the
unstable root ([`corrigendum.md`](corrigendum.md)). The closed-loop dataset `manywells-nscl-1` was generated with
v1.0.0, whose closed-loop simulator and generator are not on `develop`; they are at the
[`v1.0.0` tag](https://github.com/solution-seeker-as/manywells/releases/tag/v1.0.0).

The published datasets are on [Hugging Face](https://huggingface.co/datasets/solution-seeker-as/manywells); the
[project README](../README.md#datasets) shows how to load them.
