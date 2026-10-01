# Changelog

## Unreleased (towards v2.0.0)

v2.0.0 may break any part of the API; every break is listed here with an old→new snippet (`specs/goals.md`). The `v1.0.0` configuration (`manywells.configurations`) reproduces v1.0.0's physics, not its API.

### Model

- `develop`'s model is specified in `specs/model/`, with v1.0.0 as the `v1.0.0` configuration and a feature spec per change since v1.0.0 (`specs/features/`): deviated and L-shaped wells, inclination in the slip model, black oil with dissolved gas, real gas, friction from roughness, frictional heating and a gravity term in the energy balance, and lift-gas temperature.
- The answer is a root set: `simulate` returns the operating point, the stable root, instead of whichever root the solver reached (`specs/model/solution.md`). In the `v1.0.0` configuration it returns the stable root in every case of the verifier's case set, where v1.0.0 returned it in 74.3%.
- The slip model's inclination factor no longer adds $10^{-9}$ to $\cos\theta$, so a vertical well is exactly v1.0.0's (`specs/features/002-slip-inclination.md`).
- `water_fvf` has the right sign: water shrinks under pressure (not used by the simulator).
- The Vazquez–Beggs separator gas-gravity correction takes $\log_{10}$ of $p_\text{sep}/114.7$, as the source does, not the natural log. `develop`'s black-oil $R_{so}$ at standard separator conditions rises by about 16% (`specs/features/007-black-oil.md`).

### API breaks

**Simulator.** The well's system is built once; the boundary conditions go to `simulate`, which returns a `Root`. The old form still works, with a `DeprecationWarning`, until the CasADi backend is retired.

```python
# old
sim = SSDFSimulator(wp, bc)
sim.x_guess = x_prev
x = sim.simulate()                  # list: whichever root Ipopt reached
df = sim.solution_as_df(x)

# new
sim = SSDFSimulator(wp)
op = sim.simulate(bc, x_guess=x_prev)   # Root: the stable root; raises NoOperatingPoint without one
x = op.x                                # numpy array, same order as before
df = sim.solution_as_df(op)
rs = sim.root_set(bc)                   # every root found, each labelled stable or unstable
```

**Friction and heat transfer** are components of the well.

```python
# old
wp = WellProperties(f_D=0.03, h=20.0)
wp = WellProperties(roughness=4.5e-5)

# new
from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.thermal import ThermalModel
wp = WellProperties(friction=FixedFrictionFactor(f_D=0.03), thermal=ThermalModel(h=20.0))
wp = WellProperties(friction=RoughnessFriction(roughness=4.5e-5))   # the default; correlation='chen' or 'haaland'
```

**Frozen inputs.** `WellProperties`, `BoundaryConditions`, `FluidModel`, `WellGeometry`, and the inflow, choke, slip, friction and thermal models are frozen dataclasses, so assigning to a field, or to a misspelt one, raises. Validation raises `ValueError`, not `AssertionError`. The default choke is set by `WellProperties` itself.

```python
# old
bc.u = 0.5
wp.choke = None; SSDFSimulator(wp, bc); wp.choke   # the simulator wrote the default choke into wp

# new
from dataclasses import replace
bc = replace(bc, u=0.5)
WellProperties().choke                             # BernoulliChokeModel(K_c=0.1 A)
```

**Pressures in bar.** `FluidModel.p_sep` and `p_bubble` are in bar, like every other interface.

```python
# old
FluidModel(p_bubble=250e5, p_sep=100 * CF_PSI)

# new
FluidModel(p_bubble=250.0, p_sep=100 * CF_PSI / CF_BAR)
```

**Choke rate.** `ChokeModel.mass_flow_rate` takes the wellhead state and the pipe area; each model chooses its density and multiplier. `choke_equation(u, p_in, p_out, rho, multiplier)` is unchanged.

```python
# old
BernoulliChokeModel().mass_flow_rate(u, p_in, p_out, rho_m)
SimpsonChokeModel().mass_flow_rate(u, p_in, p_out, x_g, rho_g, rho_l)

# new
from manywells.discretization import PointState
s = PointState(p=p_in, v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l, T=T)
choke.mass_flow_rate(u, p_out, s, A)
```

**Surface tension** takes the point's liquid density, which the `v1.0.0` option uses.

```python
# old
fluid.surface_tension(p, T)

# new
fluid.surface_tension(p, T, rho_l)            # surface_tension_model='oil' (default) ignores rho_l
```

**Phase rates** moved from the simulator to the fluid model.

```python
# old
sim._gas_and_liquid_flow_rate(p, T, w_l_inflow)

# new
fluid.phase_rates(p, T, w_res, w_lg)          # and fluid.reservoir_gas_rate(w_res)
```

**Closed loop** runs on a frozen copy of the old simulator; import its inputs from it.

```python
# old
from manywells.simulator import WellProperties, BoundaryConditions

# new
from manywells.closed_loop.cl_simulator import ClosedLoopWellSimulator, WellProperties, BoundaryConditions
```

### Breaks since v1.0.0 that predate this list

- `WellProperties(L=..., D=..., rho_l=..., R_s=..., cp_g=..., cp_l=...)` became `WellProperties(geometry=WellGeometry.vertical(L, n_cells, D), fluid=FluidModel(...))`; `manywells.configurations.v1_well` builds the v1.0.0 well from the old parameters.
- `SSDFSimulator(wp, bc, n_cells=N)`: the number of cells is in the geometry.
- `Vogel(w_l_max, f_g)` and `ProductivityIndex(k_l, f_g)` became `Vogel(w_l_max)` and `ProductivityIndex(k_l)`; the gas fraction is a property of the fluid. `FixedFlowRate` fixes the liquid rate only.
- `CF_PRES` became `CF_BAR`, in `manywells.units`.
