# Tutorial on how to simulate a well

### Single datapoint
In order to simulate a single datapoint from single well a set of steps must be done. We refer to such an example in 
the `simulator.py` file found in:
```
Project folder
├── src/manywells/  # Package containing basic components of the simulator
│ └── simulator.py # Implementation of steady-state drift-flux model
```
The steps are as follows:
```
1. Define a wp = WellProperties() object
2. Define a sim = SSDFSimulator(wp) object, which builds the well's system once
3. Define a bc = BoundaryConditions() object
4. Simulate a data point op = sim.simulate(bc)
```
`op` is the operating point, the stable root (a `Root`, with the state in `op.x`), and `sim.solution_as_df(op)` gives it
per grid point. `simulate` raises `NoOperatingPoint` if the well has no stable root at these conditions. Further data
points of the same well reuse its system: `sim.simulate(bc2)`, optionally with `x_guess=op.x` as an extra start.
`sim.root_set(bc)` returns every root the search finds, each labelled stable or unstable (`specs/model/solution.md`).
`scripts/sim_examples/` has complete examples.

A well in the `v1.0.0` configuration, which reproduces ManyWells v1.0.0 and the published datasets, is built from
v1.0.0's parameters by `manywells.configurations.v1_well`.

### Multiple datapoints from multiple wells
In order to simulate datasets similar to those provided with the publication, refer to the following scripts:

```
Project folder
├── scripts # Script folder
│ ├── data_generation # Data generation folder
│ │ ├── open_loop_stationary # Folder data generation open-loop stationary data
│ │ │  └── generate_well_data.py # Data generation script stationary open-loop data
│ │ ├── open_loop_nonstationary # Folder data generation open-loop stationary data
│ │ │  └── generate_open_loop_nonstationary_well_data.py # Data generation script nonstationary open-loop data
│ │ ├── open_loop_nonstationary # Folder data generation closed-loop stationary data
│ │ │  └── generate_closed_loop_nonstationary_well_data.py # Data generation script nonstationary closed-loop data
```
Hence, `generate_well_data.py` generates open-loop stationary data, `generate_open_loop_nonstationary_well_data.py` 
generates open-loop nonstationary, and `generate_closed_loop_nonstationary_well_data.py` 
generates closed-loop nonstationary  data. Then, do the following steps:
```
1. Define n_wells (int, number of wells to be simulated)
2. Define n_sim (int, number of datapoints per well)
3. Define n_processes (int, number of parallell cpu cores generating data. Hardware dependent)
4. Run script
```
Default values for `n_wells`, `n_sim` and `n_processes` are given, and can be change according to the usecase
and hardware limitations.
