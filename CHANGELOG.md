# Changelog

## Unreleased (towards v2.0.0)

v2.0.0 may break any part of the API; every break is listed here with an old→new snippet (`specs/goals.md`). The `v1.0.0` configuration (`manywells.configurations`) reproduces v1.0.0's physics, not its API.

### Model

- `develop`'s model is specified in `specs/model/`, with v1.0.0 as the `v1.0.0` configuration and a feature spec per change since v1.0.0 (`specs/features/`): deviated and L-shaped wells, inclination in the slip model, black oil with dissolved gas, real gas, friction from roughness, frictional heating and a gravity term in the energy balance, and lift-gas temperature.
- The answer is a root set: `simulate` returns the operating point, the stable root, instead of whichever root the solver reached (`specs/model/solution.md`). In the `v1.0.0` configuration it returns the stable root in every case of the verifier's case set, where v1.0.0 returned it in 74.3%.
- The slip model's inclination factor no longer adds $10^{-9}$ to $\cos\theta$, so a vertical well is exactly v1.0.0's (`specs/features/002-slip-inclination.md`).
- `water_fvf` has the right sign: water shrinks under pressure (not used by the simulator).
- The Vazquez–Beggs separator gas-gravity correction takes $\log_{10}$ of $p_\text{sep}/114.7$, as the source does, not the natural log. `develop`'s black-oil $R_{so}$ at standard separator conditions rises by about 16% (`specs/features/007-black-oil.md`).
- A Rust core (`rust/`, built as `manywells._core`) solves the same model as the CasADi backend, with every option: `SSDFSimulator(wp, backend='rust')`. In the `v1.0.0` configuration it returns every reference root of the verifier's case set with its label (`specs/features/014-rust-solver.md`); with `develop`'s options it finds every root the CasADi backend finds, with the same label, except roots on another branch of a point's rows (`specs/features/015-rust-develop-model.md`). It is about 10 to 30 times faster per case than the CasADi backend (median and total, on one process). A well with a component class the core does not have, such as a user's subclass of `InflowModel`, raises `ValueError`; use the CasADi backend for it. The port it comes from (`rust_implementation`) is replaced: it used the exact solution of the energy ODE instead of v1.0.0's rows, and returned at most two unlabelled roots.
- A state with a supersonic cell is not a root (SOL-8): each cell's momentum row must rise with its pressure along the point's other rows, on the subsonic side of its sonic point. The CasADi search rejects such states, as the Rust core's cell solve never reaches them; none of the verifier's reference roots has one (`specs/features/015-rust-develop-model.md`, Finding 1).
- A fluid's PVT may come from a table: `FluidModel(pvt=PVTTable.read_csv('fluid.csv'), gor=...)`, with $R_{so}$, $B_o$ and $Z$ (and optionally $\mu_o$, $\mu_g$, $\sigma_{go}$) on a $(p, T)$ grid, interpolated bilinearly (`specs/model/pvt/table.md`). A fluid from an equation of state enters as a table, made from its flash's phase split by `PVTTable.from_phase_split`; tables of one stream at several GORs (`PVTTableSet`) are read at the fluid's GOR. The simulator refuses a reservoir off the table's grid, and a `Root` flags points off it (`off_table`). Both backends, which interpolate to the same bits (`specs/features/018-pvt-tables.md`).
- A PVT table may hold a gas condensate: its $R_{so}$ may fall as the pressure rises (retrograde condensation), and `PVTTable.from_phase_split` keeps a trace of liquid where the flash has none, a mass fraction `delta` of the stream ($10^{-4}$ by default), so that $B_o > 0$. A table is refused instead where the oil phase's mass, $\rho_o + R_{so}\rho_{g,\text{sc}}$, is not positive, or where the gas density $p/Z$ does not rise strictly with the pressure at every temperature (`specs/model/pvt/table.md`, PVT-TAB-1 and PVT-TAB-6).
- Each correlation's range, and what the model does outside it, is in `specs/model/validity.md`. Two inputs are refused with `ValueError`, in both backends: a `BlackOil` of API gravity outside 16 to 58, Beggs and Robinson's range (it refused 10 to 40, a range no source gives; a heavier oil needs a PVT table), and a well with `RoughnessFriction` whose relative roughness exceeds 0.05, beyond which the friction correlations have a pole and a spurious valley. The other correlations extrapolate outside their ranges, and a PVT table clamps to its grid. The `v1.0.0` configuration refuses neither.
- The lift gas can enter through a valve above the bottomhole: `WellProperties(injection_md=...)`, the valve's measured depth, at the nearest grid point. Below it the tubing carries the reservoir fluid alone; the lift gas mixes in by heat-capacity rates there, at the ambient temperature of that depth unless `T_lg` is given. `injection_md=None`, the default, keeps the bottomhole, and every result is as before (`specs/features/021-lift-gas-injection-depth.md`).
- A gas well can flow in through Forchheimer's law in the pressure squared, `Forchheimer(a, b)`: the inflow gives the gas rate, $p_r^2 - p_0^2 = a w_g + b w_g^2$, and the condensate and water come with the gas in the fluid's ratios. `Forchheimer.from_gas_rate` converts a well test's coefficients in Sm³/d. Both backends, and `v2.0.0`'s check accepts it; the liquid inflows and every result with them are as before (`specs/features/022-gas-rate-inflow.md`).
- The `v2.0.0` configuration names the model that v2's datasets are drawn in: `develop`'s defaults of 2026-10-30, with the five flow regimes and not the fitted classifier, and the black oil or a PVT table. `check(wp, 'v2.0.0')` fixes every model option and the slip law's constants, and leaves a well's inputs free; row vectors from both backends pin it (`specs/model/README.md`). `develop`'s defaults may move on without moving it. No result changes.
- Installing from source needs a Rust toolchain: the package is built by maturin instead of setuptools.

### Calibration

- `manywells.calibration` calibrates a well to its production data (`specs/calibration.md`, `docs/calibration.md`): the MAP estimate of the choke coefficient, the inflow productivity, friction (`f_D` or the roughness) and the heat-transfer coefficient, any of them free, each with a physics-based log-normal prior, from rows with well-test or MPFM rates, pressures and temperatures, with missing values and whatever instruments the well has. Each row is solved for the well's operating point, on either backend (the Rust core by default). `calibration.synthetic_data` simulates a well's rows at known parameters, and the recovery tests check that the fit finds them (`specs/features/017-calibration.md`).
- The slip law's profile parameter can be scaled towards no slip, $C_0 \to 1 + \kappa_0(C_0 - 1)$ with $\kappa_0 \ge 0$ (`SlipModel.profile_scale`, SLIP-18), and calibrated: `profile_scale` is a free parameter, with a prior centred on $C_0 = 1.1$. A holdup error then has a parameter of its own, rather than being absorbed by friction. The default, 1, leaves the model as it was (`specs/features/020-profile-scale.md`).
- `scipy` is a declared dependency; it was installed before as scikit-learn's.

### Sampling and datasets

- `specs/sampling.md` specifies the v2 procedure (SMP-50 to SMP-117), which draws the wells and operating points of the v2 datasets, `manywells-sol-2` and `manywells-nsol-2`, in the `v2.0.0` configuration: a realistic core weighted towards offshore wells and a labelled coverage tail, black oils, volatile oils, gas condensates and wet and dry gases, deviated wells and drains to the heel, and one life per well that both datasets use. Each rule names its decision entries and sources, and says what it rests on: data, physics, a design choice or a guess. It draws from generators of its own, so v1's procedure (SMP-1 to SMP-31) and its extension to `develop`'s inputs (SMP-40 to SMP-44), their draws and the published v1 datasets do not change.
- `manywells.sampling.v2` implements it: a well's draws, its life and its samples, its generation (the best rate and the producibility filter, `nsol-2`'s life scan, and the slots' redraws until each well has 200 operating points), the run over many wells in resumable chunks, the volatile oils' and condensates' PVT tables from NeqSim, built in an environment of their own, and the assembly of a dataset's tables and logs.
- `manywells.datasets.v2` holds the v2 files' label codes, column types and records: `read_dataset` reads a dataset's tables or its logs, and `sample_case` gives a sample's `WellProperties` and `BoundaryConditions` from the dataset's rows alone, so that `resolve_sample` solves it again to the same features. `docs/datasets.md` describes the schema: per dataset the tables `wells`, `samples`, `geometry` and `pvt`, and a companion folder with the well log and the redraw log. `docs/datasheet-sol-2.md` and `docs/datasheet-nsol-2.md` are the datasets' datasheets.
- `scripts/data_generation/v2/generate_dataset.py` makes a dataset end to end: the table fluids' build, the run and the assembly. A dataset to publish is written from a clean working tree, and records its seed, the configuration, the limits, the code version with its commit, the lock file's hash and the platform.
- The first full generation of the v2 datasets is a beta, `manywells-sol-2-beta1` and `manywells-nsol-2-beta1`, with `release` in their records (SMP-50); the final datasets take the plain names after its analysis. A review of `develop` before the run (`plans/sampler-redesign/preflight-measurements.md`) changed the procedure: in `nsol-2` a dead date, a date at which the well flows at none of its 100 draws, starts the well's lift or ends its life in place of dropping the well (SMP-90, SMP-106); SMP-106's targets for the dropped wells give way to a report by fluid class, well type and lift rule; a table's repair may drop a tenth of its pressures (SMP-116); a life that ends where its lift would start is natural (SMP-101); and a table condensate whose stock-tank oil is lighter than 58 °API keeps its table. The run records the versions of Python, numpy and scipy in its settings and refuses to resume or assemble under others, since a gas well's inflow coefficients differ in their last digits between them, and it keeps whether each operating point leaves its PVT table's grid.
- After the first beta's analysis the procedure changes in two rules, working assumptions of which Bjarne has taken the platform hurdle and the share's lower end (D-150 to D-153; `plans/sampler-redesign/choke-rates-measurements.md`), and the code generates `manywells-sol-2-beta2` and `manywells-nsol-2-beta2`. The choke's bean is sized to the well's rate (SMP-77): the well is solved with its largest choke, and the bean is the one whose fully open choke takes a drawn share, 0.5% to 45%, of the pressure upstream of it at that rate, in place of a share of the tubing's diameter, under which a fifth of the beta's samples had a choke that took less than 1% of the wellhead pressure. A platform or subsea well is drilled only where its design rate in oil equivalent reaches a start hurdle, 20 and 500 Sm³/d (SMP-102). The `wells` tables add `choke_dp_share` and `design_rate_oe`, and the well log the reason `below_start_hurdle`. A dataset's wells are quotas by stratum (SMP-110, D-157, a working assumption for Bjarne's ruling): 15% tail wells, and the core's platform, subsea and onshore wells at D-5's 0.55, 0.35 and 0.10, each stratum's accepted wells taken in draw order until its quota is reached, in place of the first 10,000 accepted wells, since the hurdle drops far more subsea wells than others; `metadata.json` records the quotas, and the well log holds each stratum's drawn wells.

### Versions

- Releases follow semantic versioning, with one version number for `manywells`, the Rust core and the verifier (`specs/constitution.md`, Versions; `AGENTS.md`, Releases). `develop` is `2.0.0.dev0`. It said `1.0.0` before, so a dataset generated on `develop` before this change records a `code_version` of `1.0.0`, followed by a commit after the tag, as in `1.0.0 (v1.0.0-223-ge44bb54)`, or alone where it ran without git.

### API breaks

**Simulator.** The well's system is built once; the boundary conditions go to `simulate`, which returns a `Root`. The old form still works, with a `DeprecationWarning`, until v2.0.0.

```python
# old
sim = SSDFSimulator(wp, bc)
sim.x_guess = x_prev
x = sim.simulate()  # list: whichever root Ipopt reached
df = sim.solution_as_df(x)

# new
sim = SSDFSimulator(wp)
op = sim.simulate(bc, x_guess=x_prev)  # Root: the stable root; raises NoOperatingPoint without one
x = op.x  # numpy array, same order as before
df = sim.solution_as_df(op)
rs = sim.root_set(bc)  # every root found, each labelled stable or unstable
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
wp = WellProperties(friction=RoughnessFriction(roughness=4.5e-5))  # the default; correlation='chen' or 'haaland'
```

**Frozen inputs.** `WellProperties`, `BoundaryConditions`, `FluidModel`, `WellGeometry`, and the inflow, choke, slip, friction and thermal models are frozen dataclasses, so assigning to a field, or to a misspelt one, raises. Validation raises `ValueError`, not `AssertionError`. The default choke is set by `WellProperties` itself.

```python
# old
bc.u = 0.5
wp.choke = None
SSDFSimulator(wp, bc)
wp.choke  # the simulator wrote the default choke into wp

# new
from dataclasses import replace

bc = replace(bc, u=0.5)
WellProperties().choke  # BernoulliChokeModel(K_c=0.1 A)
```

**Pressures in bar.** The black oil's `p_sep` and `p_bubble` are in bar, like every other interface.

```python
# old
FluidModel(p_bubble=250e5, p_sep=100 * CF_PSI)

# new
FluidModel(pvt=BlackOil(p_bubble=250.0, p_sep=100 * CF_PSI / CF_BAR))
```

**PVT component.** The hydrocarbon PVT is a component of `FluidModel`, `pvt`, with one class per option: `DeadOil` (`manywells.pvt.dead_oil`), `BlackOil` (`manywells.pvt.black_oil`) or `PVTTable` (`manywells.pvt.table`), behind the abstract `PVTModel` (`manywells.pvt.model`). `rho_o`, `rho_g`, `oil_model`, `ideal_gas`, `z_factor_model`, `p_sep`, `T_sep` and `p_bubble` move into the classes; `ideal_gas` and `z_factor_model` become one field, `gas_law` (`'ideal'`, `'dak'` or `'papay'`). `fluid.rho_o`, `fluid.rho_g` and every method keep working, and give the same values to the bit. `FluidModel.gas_viscosity` takes the pressure too (`specs/features/018-pvt-tables.md`).

```python
# old
FluidModel(rho_o=850, rho_g=0.8, gor=150, wlr=0.2, p_bubble=180, z_factor_model='papay')
FluidModel(rho_o=850, rho_g=0.8, gor=150, oil_model='dead_oil', ideal_gas=True)
replace(fluid, ideal_gas=True)
fluid.gas_viscosity(T, rho_g)

# new
FluidModel(pvt=BlackOil(rho_o=850, rho_g=0.8, p_bubble=180, gas_law='papay'), gor=150, wlr=0.2)
FluidModel(pvt=DeadOil(rho_o=850, rho_g=0.8, gas_law='ideal'), gor=150)
FluidModel(pvt=PVTTable.read_csv('fluid.csv'), gor=150, wlr=0.2)  # new: a PVT table
replace(fluid, pvt=replace(fluid.pvt, gas_law='ideal'))
fluid.gas_viscosity(p, T, rho_g)
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
fluid.surface_tension(p, T, rho_l)  # surface_tension_model='oil' (default) ignores rho_l
```

**Phase rates** moved from the simulator to the fluid model.

```python
# old
sim._gas_and_liquid_flow_rate(p, T, w_l_inflow)

# new
fluid.phase_rates(p, T, w_res, w_lg)  # and fluid.reservoir_gas_rate(w_res)
```

**Closed loop** is removed: closed-loop simulation is out of scope for v2 (`specs/goals.md`). `manywells.closed_loop`, the closed-loop and loaded-wells generators, and the v1.0.0 sampler they used (`scripts/data_generation/well.py`, `nonstationary_well.py`) are gone; v1.0.0 has them, and generated the published `manywells-nscl-1`.

```python
# old
from manywells.closed_loop.cl_simulator import ClosedLoopWellSimulator

# new: use v1.0.0
#   pip install git+ssh://git@github.com/solution-seeker-as/manywells.git@v1.0.0
```

**Calibration.** `calibration.choke_cal` and `calibration.inflow_cal` are removed. They fitted one component to columns that are not measured, such as the mixture density upstream of the choke; `calibrate` fits the whole well to what is measured.

```python
# old
from manywells.calibration.choke_cal import calibrate_simpson_choke_model

K_c = calibrate_simpson_choke_model(df_with_p_w_m_u_p_s_x_g_rho_g_rho_l, choke)

# new
from manywells.calibration import CalibrationData, calibrate

result = calibrate(wp, CalibrationData(df_with_CHK_PDC_PWH_TWH_QOIL_QGAS_QWAT), bc, free=['K_c'])
K_c = result.values['K_c']  # result.well is the calibrated WellProperties
```

**Lift gas at a depth** (021). `System.params(bc)` is an instance method, since the lift gas's default temperature depends on the well, and `solvers.rust.operating_point(bc)` became `operating_point(wp, bc)`.

```python
# old
params = System.params(bc)
op = operating_point(bc)
# new
params = system.params(bc)
op = operating_point(wp, bc)
```

**Inflow models take the fluid** (022). `InflowModel.liquid_mass_flow_rate(p, p_r, fluid=None)` takes the well's fluid, which the gas-rate inflow needs and the liquid inflows ignore, and the simulator passes it. Calls with two arguments still work for the liquid inflows; a user's subclass that overrides the method must accept the third argument.

```python
# old
class MyInflow(InflowModel):
    def liquid_mass_flow_rate(self, p, p_r): ...


# new
class MyInflow(InflowModel):
    def liquid_mass_flow_rate(self, p, p_r, fluid=None): ...
```

### Breaks since v1.0.0 that predate this list

- `WellProperties(L=..., D=..., rho_l=..., R_s=..., cp_g=..., cp_l=...)` became `WellProperties(geometry=WellGeometry.vertical(L, n_cells, D), fluid=FluidModel(...))`; `manywells.configurations.v1_well` builds the v1.0.0 well from the old parameters.
- `SSDFSimulator(wp, bc, n_cells=N)`: the number of cells is in the geometry.
- `Vogel(w_l_max, f_g)` and `ProductivityIndex(k_l, f_g)` became `Vogel(w_l_max)` and `ProductivityIndex(k_l)`; the gas fraction is a property of the fluid. `FixedFlowRate` fixes the liquid rate only.
- `CF_PRES` became `CF_BAR`, in `manywells.units`.
