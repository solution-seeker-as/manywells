# PVT tables

This guide shows how to give a well the PVT of a fluid as a table, how to make one from a PVT report or an
equation of state, and how to choose its grid.
[`specs/model/pvt/table.md`](../specs/model/pvt/table.md) specifies the table (PVT-TAB-1 to PVT-TAB-7), and
[`specs/features/018-pvt-tables.md`](../specs/features/018-pvt-tables.md) has the design and its measurements.

## What a table holds

A table gives the oil's and the gas's properties at the nodes of a grid of pressures and temperatures:

| Column | Unit | Meaning | |
|---|---|---|---|
| `R_so` | Sm³/Sm³ | solution gas-oil ratio | required |
| `B_o` | – | oil formation volume factor | required |
| `Z` | – | the free gas's z-factor | required |
| `mu_o` | Pa s | oil viscosity | optional |
| `mu_g` | Pa s | gas viscosity | optional |
| `sigma` | J/m² | gas-oil surface tension | optional |

Two densities at standard conditions go with the columns: the stock-tank oil's, `rho_o`, and the gas's, `rho_g`
(kg/m³). The columns mean what they do in the balances, per Sm³ of stock-tank oil:
- the oil at $(p, T)$ carries the mass $\rho_o + R_{so}\rho_{g,\text{sc}}$ in the volume $B_o$;
- the free gas has the density $c_\text{bar}\thinspace p/(Z R_s T)$, where $R_s$ is the gas constant of
  $\rho_{g,\text{sc}}$.

So `R_so` and `B_o` must be on the basis of the separation that gives `rho_o` and `rho_g`. A laboratory report's
differential-liberation values must first be adjusted to the separator, as the report usually does.

Above the fluid's bubble point, `R_so` is the table's GOR, and `Z` is that of the gas that would appear first. Where a
column is missing, ManyWells uses its correlations for it: Beggs–Robinson's viscosities and the live-oil surface
tension at the table's `R_so`, and Lee–Gonzalez–Eakin's gas viscosity. The Joule–Thomson factor is always the
Dranchuk–Abou-Kassem equation of state's at the gas's gravity.

Between the nodes the columns are interpolated bilinearly. Outside the grid they keep the value at its nearest edge.
Water is not in the table: it mixes with the oil as with the black oil.

## Using a table

```python
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator

fluid = FluidModel(pvt=PVTTable.read_csv('fluid.csv'), gor=110.0, wlr=0.2)
sim = SSDFSimulator(WellProperties(fluid=fluid), backend='rust')   # or 'casadi'
op = sim.simulate(BoundaryConditions(p_r=250, T_r=360, p_s=20, u=0.6))
print(op.off_table)   # True if a point of the operating point lies off the table's grid
```

`FluidModel` takes the table as its `pvt`, in place of `BlackOil` or `DeadOil`. Its `gor` and `wlr` are the
stream's, as with the black oil. Both backends take a table and interpolate it to the same bits.

**The grid must cover the reservoir.** `simulate` and `root_set` raise `ValueError` if `p_r` or `T_r` lies outside
the grid. Elsewhere in the well a point may leave the grid, for instance at a wellhead pressure below the lowest
node. There the properties are those at the grid's edge, the root's `off_table` is `True`, and `simulate` logs a
warning on the `manywells` logger. Give the grid room: down to about 1 bar and to the coldest wellhead
temperature you expect.

## The file

A CSV file: `# key: value` lines, then the columns `p` (bar) and `T` (K) and the table's columns, one row per node of
a rectangular grid, in any order.

```text
# name: synthetic_eos_oil
# source: plans/evidence/pvt_table_eos.py; NeqSim 3.4.0, Peng-Robinson with Peneloux volume correction; ...
# separation: single stage at 1.01325 bar and 288.15 K
# rho_o: 866.4007
# rho_g: 1.035376
# gor: 108.7299
p,T,R_so,B_o,Z,mu_o,mu_g,sigma
1,330,-26.39282129,0.9767252985,0.8609510717,0.004864442699,1.13727943e-05,0.02415576601
10,330,16.87511237,1.053543663,1.044752939,0.00286113248,1.214911351e-05,0.02182602747
...
```

This is the start of `tests/data/pvt/synthetic_eos_oil.csv`, with its first line and part of its source left out.

`rho_o` and `rho_g` are required. `name`, `source`, `separation` and `gor` are for the reader;
`PVTTable.write_csv` writes them, and `read_csv` reads every value back to the bit. Field units and other formats,
such as Eclipse's `PVTO` and `PVDG` keywords, are converted to this one outside ManyWells.

A table can also be built in Python. Each column is an array of shape `(len(p), len(T))`:

```python
table = PVTTable(p=p, T=T, R_so=R_so, B_o=B_o, Z=Z, rho_o=866.4, rho_g=1.0354, gor=108.73)
```

The table is checked when it is built: every value finite, increasing grids, and positive `B_o`, `Z`, viscosities
and surface tension. `R_so` must not fall as the pressure rises, apart from rounding. It may be below 0 only if the
table has its own `mu_o` and `sigma`, because the live-oil correlations take `R_so` ≥ 0.

## A table from an equation of state

ManyWells has no equation of state; a fluid from one enters as a table. Flash the reservoir stream at each node with
your PVT program (NeqSim, PVTsim, ...). Separate it once to standard conditions, as your process does, which gives
the stock-tank oil density, the gas density at standard conditions and the GOR. Then
`PVTTable.from_phase_split` makes the table:

```python
table = PVTTable.from_phase_split(p, T, X, rho_L, rho_V, rho_o=866.4, rho_g=1.0354, gor=108.73,
                                  mu_o=mu_o, mu_g=mu_g, sigma=sigma)
```

At each node it takes three values from the flash:
- `X`, the vapour's mass fraction of the hydrocarbon stream;
- `rho_L`, the liquid's density, in kg/m³;
- `rho_V`, the vapour's density, in kg/m³.

Where the stream is all liquid, `X` is 0, and `rho_V` is the density of the gas that would appear first: the
incipient phase at the bubble point, or a stability analysis's trial phase.

The table then reproduces the flash's phase mass rates and densities at the nodes, at the stream's GOR (PVT-TAB-6).
At low pressure and high temperature its `R_so` can be negative and its `B_o` below 1, where the free gas carries
components that are oil at standard conditions. Such a table needs the flash's `mu_o` and `sigma`.

[`plans/evidence/pvt_table_eos.py`](../plans/evidence/pvt_table_eos.py) is a worked example with NeqSim. It wrote
`tests/data/pvt/synthetic_eos_oil.csv`, a synthetic oil, not a real fluid.

On sampled wells, this synthetic fluid's table and develop's black oil at the same stock-tank densities and GOR give
bottomhole pressures up to 10 bar apart, and wellhead pressures up to 16 bar apart. Matching the black oil's bubble
point to the table's does not close that gap: the black oil's shape between its end points is the difference.

## Choosing the grid

The interpolation's error falls with the grid's spacing. Over random points of 40 sampled black oils, and on 252
solved cases of sampled wells:

| Grid | $R_{so}$ (Sm³/Sm³) | $\rho_l$ | $\rho_g$ | PBH (bar) | PWH (bar) |
|---|---|---|---|---|---|
| 2 bar by 5 K | 0.13 | 0.008% | 0.07% | 0.005 | 0.004 |
| 10 bar by 10 K | 0.45 | 0.030% | 0.21% | 0.012 | 0.020 |
| 25 bar by 25 K | 3.2 | 0.19% | 1.5% | 0.059 | 0.088 |

The figures are the largest errors against the black oil the table was made from.

Put a node at the bubble point at each temperature. $R_{so}$ has a kink there, which the bilinear step keeps exactly
at a node and rounds off between nodes. On 10 bar by 10 K, a node at the bubble point cuts the largest $R_{so}$
error from 0.52 to 0.13 Sm³/Sm³. A coarse table can be refined before it is written, for instance by monotone
cubics, where the fluid is smooth; ManyWells does not refine it.

**One temperature.** A table may have one temperature, such as a laboratory report's at the reservoir temperature.
Its properties then do not depend on $T$, and the cooler fluid up the well is treated as if it were at $T_r$. On
sampled wells, a table of each well's black oil at its $T_r$ put the wellhead pressure 0.9 bar higher than the black
oil itself at the median, and up to 7.9 bar higher.

## Several GORs

Calibration rows, and the operating points of a well's life, differ in GOR. A table made at one GOR serves other
GORs by the black-oil rule: gas dissolves up to the smaller of `R_so` and the gas available. On the synthetic fluid
that errs by 1.2 to 1.7% in the mixture's no-slip density at 10% from the table's GOR, and by 2.4 to 3.8% at 20 to
25% from it.

A table can hold the stream at several GORs instead. Recombine the same separator gas and stock-tank oil at each
GOR, so that `rho_o` and `rho_g` are the same at every GOR, and give each GOR its own grid, with its own bubble
points. In the file, a `gor` column holds each row's GOR; `read_csv` then returns a `PVTTableSet`:

```python
import dataclasses

axis = PVTTable.read_csv('fluid_gor.csv')        # a PVTTableSet
fluid = FluidModel(pvt=axis, gor=120.0)          # read at 120 Sm³/Sm³
other = dataclasses.replace(fluid, gor=135.0)    # read again at 135
```

`FluidModel` reads the axis at its own GOR when it is built. Each column is interpolated linearly in GOR between the
two neighbouring GORs' tables (PVT-TAB-7), on the union of their grids. On the synthetic fluid this errs by 0.01 to
0.3% in the no-slip density at the median. A GOR outside the axis raises `ValueError`. A column only one of the two
neighbouring tables has is left out of the table between them, and the correlations stand in for it.

## Cost

The Rust core evaluates a table at the cost of the black oil: on 263 cases of sampled wells, a table of each well's
black oil took 1.00 times as long to solve as the black oil itself, at 0.3 s a case. The CasADi backend evaluates
each column through an interpolant, whose derivatives cost more: there the table took 1.8 times as long at the
median, 7.3 s a case against 3.7 s. Both backends interpolate a table to the same bits and find an operating point
in the same cases. In 258 of the 260 cases they agree on it to about $10^{-6}$ bar; in the other 2 they differ as
much with the black oil itself. Making a table of a black oil on 10 bar by 10 K takes about 0.02 s.

## Examples

[`tests/data/pvt/`](../tests/data/pvt/README.md) has four example tables, none from a real fluid:
- develop's black oil on 10 bar by 10 K;
- the synthetic equation-of-state oil, with every column;
- the same oil at one temperature, at a laboratory report's spacing;
- the same oil at five GORs.
