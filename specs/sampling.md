# Sampling

*Step 4 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: in use, 2026-09-30. The v1 procedure (SMP-1 to SMP-30) records v1.0.0's code, with the rulings of `specs/discrepancies.md`. Step 7 (2026-10-01) implemented the sampler (`manywells.sampling`, `manywells.datasets`) and wrote SMP-31 and SMP-40; Bjarne kept the starting points of SMP-41 to SMP-43 until the sampling redesign (2026-10-01).*

How ManyWells draws wells and operating points for its datasets. This is not physics, so it lives outside `specs/model/`. It records v1.0.0's procedure, which generated `manywells-sol-1` and `manywells-nsol-1`, and extends it to the inputs that `develop`'s model adds: trajectory, black-oil parameters and pipe roughness. The approach stays the same, independent draws. In the v1-compatibility configuration the sampler draws v1's inputs as before, so that the Distributions check (`specs/verification.md`, Step 3) can compare regenerated samples with the published datasets. Closed loop (`manywells-nscl-1`) is out of scope for v2 (`specs/goals.md`).

After this plan the procedure will be redesigned for the v2 datasets; that is a new version of this file.

## Conventions

- $U(a, b)$ is the uniform distribution. Every draw is independent of the others unless its definition says otherwise.
- A **well** is drawn once (SMP-1 to SMP-17). A **sample** is one operating point of a well, solved once; `sol-1` draws 500 per well (SMP-18 to SMP-22), `nsol-1` evolves 500 weekly samples (SMP-23 to SMP-27).
- The published datasets store the samples in their rows (`docs/datasets.md`) and the wells in their config files. A `sol-1` config holds the well draws, with $u = 0.5$ from the generator's first solve (SMP-28). An `nsol-1` config holds the well's state at its last sample, the last row of its data, because the generator updates the well in place (SMP-24 to SMP-27) and saves it at the end: the fractions, $\rho_l$, $c_{pl}$, the inflow's $f_g$, $p_r$, $p_s$, $u$, $w_{lg}$ and the decay rate $\gamma_{pr}$ are final values. The draws survive in its `ns_bhv` fields `pr_init`, `ps_init` and `init_fractions`. Both kinds store $f_D$ wrongly for `nsol-1` (S-8).
- Source: `scripts/data_generation/` at the `v1.0.0` tag. Where the code and the paper differ, `specs/discrepancies.md` (S-1 to S-12) records it; this file follows the code.

## Well draws

`sample_well` in `well.py`.

### SMP-1 · Depth

$L \sim U(1500, 4500)$ m. Paper (20).

### SMP-2 · Tubing diameter

$D$ uniform from the inner diameters {3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5} inches, each 0.5 inch less than the outer diameters {3.5, …, 7}; 1 inch = 0.0254 m. Paper (21).

### SMP-3 · Friction factor

$f_D \sim U(0.01, 0.08)$ (FRIC-2). Paper (22).

### SMP-4 · Heat transfer coefficient

$h \sim U(10, 40)$ W/(m² K). Paper (23).

### SMP-5 · Choke

A Simpson choke (CHK-5) with $K_c \sim U(K_c^\text{ref}/2,\ 2K_c^\text{ref})$ and $K_c^\text{ref} = 0.12A$. Paper (24).

### SMP-6 · Choke profile

Uniform from the four profiles CHK-7 to CHK-10. Paper §4.1.

### SMP-7 · Mass fractions

$(f_g, f_o, f_w) \sim \text{Dir}(1, 1, 1/2)$, the gas, oil and water mass fractions of the reservoir inflow. The well is discarded and redrawn if $f_g > 0.99$. Paper (25).

### SMP-8 · Inflow

Vogel inflow (INF-1) with $w_{l,\max} \sim U\big(20(1 - f_g),\ 200(1 - f_g)\big)$ kg/s, using the $f_g$ of SMP-7, and the gas fraction $f_g$ (INF-4). Paper (26).

### SMP-9 · Oil

$\rho_o \sim U(825, 925)$ kg/m³, $c_{po} = 2000$ J/(kg K). Paper (27).

### SMP-10 · Water

$\rho_w = 999.1$ kg/m³, $c_{pw} = 4184$ J/(kg K) (PVT-WAT-1). Paper §4.2.

### SMP-11 · Gas

$R_s \sim U(320, 520)$ J/(kg K), $c_{pg} = 2225$ J/(kg K). Paper (30).

### SMP-12 · Liquid

$\rho_l$ and $c_{pl}$ from the mixing rules PVT-MIX-2 to PVT-MIX-4, with $x_o = f_o/(f_o + f_w)$. Recomputed whenever the fractions change (SMP-22, SMP-24). Paper (28)–(31).

### SMP-13 · Nominal reservoir pressure

$p_{r,\text{nom}} = \rho_{sw}\, g\, L / c_\text{bar} + 1$ bar, with $\rho_{sw} = 1012.05$ kg/m³, the mean of seawater (1025) and water (999.1). Paper (33).

### SMP-14 · Reservoir temperature

$T_r = 333.15 + 0.03\,(L - 1500)$ K, that is 60 °C at 1500 m and 150 °C at 4500 m. Paper (36).

### SMP-15 · Surface temperature

$T_s = 277.15$ K (4 °C). Paper §4.3.

### SMP-16 · Nominal separator pressure

$p_{s,\text{nom}} \sim \text{LogNormal}(3, 1)$ bar, the logarithm having mean 3 and standard deviation 1. The well is discarded and redrawn if $p_{s,\text{nom}} \notin [10, 120]$ bar. Paper (34).

### SMP-17 · Gas-lift availability

If $f_g \le 0.2$, the well has gas lift with probability 1/2; otherwise it has none. Paper §4.5.

## Stationary samples (`sol-1`)

`Well.sample_new_conditions` in `well.py`, drawn around the well's nominal values for each sample.

### SMP-18 · Choke position

$u \sim U(0.05, 1)$. Paper (44).

### SMP-19 · Lift-gas rate

$w_{lg} \sim U(0, 5)$ kg/s if the well has gas lift, otherwise 0. Paper (45).

### SMP-20 · Separator pressure

$p_s \sim U(0.9\,p_{s,\text{nom}},\ 1.1\,p_{s,\text{nom}})$. Paper (35).

### SMP-21 · Reservoir pressure

$p_r \sim U(0.98\,p_{r,\text{nom}},\ 1.02\,p_{r,\text{nom}})$. It is not stored in the datasets. Paper (32).

### SMP-22 · Fractions

$f_g \sim U(0.95 f_{g,\text{nom}},\ 1.05 f_{g,\text{nom}})$, capped at 0.99; the water–liquid mass fraction $f_{wl} \sim U(0.95 f_{wl,\text{nom}},\ 1.05 f_{wl,\text{nom}})$, capped at 1, with $f_{wl,\text{nom}} = f_w/(f_o + f_w)$; then $f_w = (1 - f_g) f_{wl}$ and $f_o = 1 - f_g - f_w$. The liquid (SMP-12) and the inflow's $f_g$ follow; $w_{l,\max}$ does not change. Paper (53)–(55).

## Non-stationary samples (`nsol-1`)

`NonStationaryBehavior` and `NonStationaryWell` in `nonstationary_well.py`. Time $t_i$ counts weeks $i$ from the first sample; the reservoir pressure starts at $p_r(t_0) = p_{r,\text{nom}}$, without SMP-21's redraw.

### SMP-23 · Lifetime

$t_\text{life}/52 \sim U(10, 20)$ years.

### SMP-24 · Fraction random walk

Weekly drifts $\gamma_g = U\big(f_g(t_0)/2,\ f_g(t_0)\big)/t_\text{life}$ and $\gamma_o = U\big(f_o(t_0)/2,\ f_o(t_0)\big)/t_\text{life}$, drawn once per well. Each week

$$f_g \leftarrow \min\{0.99, \max\{f_g - \gamma_g + X_g, 0.002\}\}, \qquad f_o \leftarrow \min\{0.99, \max\{f_o - \gamma_o + X_o, 0.002\}\},$$

with $X_g, X_o \sim N(0, 0.015^2)$, both redrawn until $f_g + f_o \le 0.999$; then $f_w = 1 - f_g - f_o$, and the liquid and inflow follow (SMP-12). Paper (37)–(39).

The generator steps the walk on every attempt, not once per week (SMP-29). An attempt at week $i$, when the last accepted sample was at week $i'$, takes $\max(1, i - i')$ steps from the current fractions, which earlier attempts have already moved. The first attempt takes one step, so the fractions at $t_0$ are not the drawn ones. $k$ failed solves in a row, which advance the week, take $(k+1)(k+2)/2$ steps over $k + 1$ weeks, and a rejected sample, which does not advance it, takes as many steps again.

### SMP-25 · Reservoir pressure decay

$p_r(\infty) = p_r(t_0) - U\big(0.2\,p_r(t_0),\ 0.4\,p_r(t_0)\big)$ and $\gamma_{pr} = 1 - 0.01^{52/t_\text{life}}$ initially. At every attempt, $\gamma_{pr} \leftarrow \min\{0.9, \max\{0.1, \gamma_{pr} + \epsilon\}\}$ with $\epsilon \sim U(-\gamma_{pr}(t_0)/20,\ \gamma_{pr}(t_0)/20)$, and

$$p_r(t_i) = \big(p_r(t_0) - p_r(\infty)\big)(1 - \gamma_{pr})^{i/52} + p_r(\infty).$$

Paper (40)–(43).

### SMP-26 · Separator pressure

$p_s \sim U(0.9\,p_{s,\text{nom}},\ 1.1\,p_{s,\text{nom}})$ every week. Paper (35).

### SMP-27 · Controls

$u$ and $w_{lg}$ as in SMP-18 and SMP-19, every week (open loop). Paper §4.5.1.

## Generation

The solves, starts and acceptance rules. They are part of the procedure because they shape the datasets' distributions: a filter that discards wells, or a start that lands on the trickle root, changes what is published.

On `develop` (`sampling.generate`), every solve returns the well's operating point, the stable root (`specs/model/solution.md`), and the generator's start is an extra start of the root search (`x_guess`), not the only one; a solve without an operating point is a failed solve. Each sample's fractions change its fluid (SMP-22, SMP-24), so its well's system is built for the sample.

### SMP-28 · Stationary generation (`sol-1`)

`open_loop_stationary/generate_well_data.py`, 2,000 wells of 500 samples.

1. Solve the well at $u = 0.5$ with its nominal conditions; if that fails, discard the well. The root is the initial guess of every sample.
2. Draw samples (SMP-18 to SMP-22) and solve each. A failed solve, negative rates or fractions outside [0, 1] count as failures; a sample with $w_m < 0.1$ kg/s is dropped without counting. The well is discarded after $5 \cdot 500$ attempts or 100 failures.
3. Discard the well if the standard deviation of $u$ over its samples is below a fifth of $U(0.05, 1)$'s, $0.95/\sqrt{12}$; if more than 80% of its samples are choked (CHK-12); or if the coefficient of variation of its total standard volumetric rate (`QTOT`) is below 0.05.

### SMP-29 · Non-stationary generation (`nsol-1`)

`open_loop_nonstationary/generate_open_loop_nonstationary_well_data.py`, 2,000 wells of 500 samples.

1. Solve the well at $u$ = 0.1, 0.2, …, 1.0 in turn, each root the initial guess of the next; if one fails, discard the well. If $w_m < 7$ kg/s at $u = 1$, discard the well.
2. Keep a pool of starts, beginning with the root at $u = 1$. A root joins it if $w_m \ge 3$ kg/s and it lies more than 0.05 from every root in the pool, in the coordinates $(f_g, p_r/350\ \text{bar})$. Each sample starts from the pool's closest root in those coordinates.
3. Each attempt updates the conditions (SMP-24 to SMP-27), with the week $i$ and the week $i'$ of the last accepted sample, and solves. A failed solve counts as a failure and advances $i$. Every solved root joins the pool (step 2) before the checks below. Negative rates or fractions outside [0, 1] count as failures, and a sample with $w_m < 1$ kg/s is rejected without counting; neither advances $i$. An accepted sample sets $i' = i$ and advances $i$. The well is discarded after 200 failures, or after 50 failed solves in a row; any successful solve resets that count, even if the sample is then rejected. There is no cap on attempts.
4. The loop runs while the well has fewer than 500 samples, or while $i$ equals the lifetime in whole weeks, $\lfloor t_\text{life} \rfloor$; the second condition can add a 501st sample, which then discards the well.
5. Discard the well if its choke positions vary too little or its `QTOT` too little, as in SMP-28 step 3; there is no filter on choked samples.

### SMP-30 · Dataset rows

Each accepted sample becomes one dataset row, computed from the root: the features of `docs/datasets.md`, with standard volumetric rates at PVT-GAS-2's standard conditions, `CHOKED` from CHK-12 and the regimes from SLIP-8. `TBH` is $T_r$, which is the root's $T_0$ (THM-3).

### SMP-31 · Seeding

Every draw comes from a NumPy generator seeded from the dataset's seed and the draw's place in the dataset: well $k$'s v1 draws from $(\text{seed}, k, \texttt{well})$, its `develop` draws from $(\text{seed}, k, \texttt{develop})$, the $j$-th stationary sample from $(\text{seed}, k, j, \texttt{sample})$, and a non-stationary well's evolution, which is sequential, from $(\text{seed}, k, \texttt{nonstationary})$; a name is hashed to an integer by CRC-32 (`sampling.wells.rng_for`). A run gives the same draws whatever the order of the wells or the number of processes, and the `develop` draws do not shift the v1 draws. The generator draws wells $0, 1, 2, \dots$ and keeps the first it accepts, so the dataset does not depend on the number of processes either. Each dataset records its seed, its configuration, its grid and the code version (`datasets.io`).

v1.0.0 seeded NumPy from process ID × time for `sol-1` and from the process ID alone for `nsol-1` (S-9), so its datasets cannot be regenerated; that is why the Distributions check compares distributions, not rows.

## Extension to `develop`'s inputs

`develop`'s model takes inputs that v1's does not. The sampler draws them as further independent draws, from their own generator (SMP-31), and draws v1's inputs exactly as above in both configurations, so that the two configurations share the v1 draws. The values of SMP-41 to SMP-43 are the starting points of Step 4, implemented as they stand; Bjarne kept them until the sampling redesign after the plan (2026-10-01).

### SMP-40 · Configurations and mapping

- **`v1.0.0` configuration.** Draw SMP-1 to SMP-29 and map them to `develop`'s inputs with `configurations.v1_well`: a vertical `WellGeometry` of length $L$ and diameter $D$ with $N = 100$ cells; the fixed $f_D$ of SMP-3 and v1.0.0's thermal model with $h$; Vogel inflow and the Simpson choke; and the fluid of `configurations.v1_fluid`, a dead oil with the mixed liquid's $\rho_l$ and $c_{pl}$ (SMP-12) and no water, an ideal gas with $\rho_{g,\text{sc}} = p_\text{ref}/(R_s T_\text{ref})$ (PVT-GAS-2), the gas–oil ratio that gives $f_g$, and the surface tension from $\rho_l$ (PVT-MIX-5). It gives back $\rho_l$, $c_{pl}$, $f_g$ and $R_s$, to rounding. The liquid is mixed before the mapping, as v1.0.0 mixed it, so a sample whose liquid is all water (SMP-22 caps the water–liquid fraction at 1) maps as well.
- **`develop` default.** The same draws, with the oil and the water by their own densities at standard conditions: $\rho_o$, $\rho_w$, the water–liquid ratio $\alpha_{w,l}$ of PVT-MIX-2 and the gas–oil ratio $(f_g/\rho_{g,\text{sc}})/(f_o/\rho_o)$, which give back $f_g$, $\rho_{l,\text{sc}}$ and $c_{pl}$ (PVT-MIX-10); black oil, real gas and the surface tension from the oil (PVT-MIX-7); SMP-41 for the trajectory, SMP-42 for friction, and `develop`'s thermal model with $h$. A sample without oil ($f_o = 0$) has no gas–oil ratio and is a failed solve.

### SMP-41 · Trajectory

The bottomhole's true vertical depth is drawn as SMP-1, and SMP-13 and SMP-14 use it, since pressure and temperature follow depth. The trajectory is vertical with probability 1/2, deviated with probability 1/4 and L-shaped with probability 1/4:

- **Deviated:** vertical down to a kickoff depth $U(0.1, 0.5)$ times the true vertical depth, then straight at an inclination $\theta \sim U(10°, 60°)$ to the bottomhole.
- **L-shaped:** vertical to the bottomhole's depth, then a horizontal section of length $U(500, 2000)$ m.

The grid is uniform in measured depth with $N = 100$ cells (`WellGeometry.from_survey`). Bjarne kept these probabilities and ranges until the redesign (2026-10-01).

### SMP-42 · Pipe roughness

$\varepsilon \sim \text{LogUniform}(1.5\cdot10^{-6},\ 1.5\cdot10^{-4})$ m, from new to corroded tubing, with $f_D$ from the Reynolds number (Chen) in place of SMP-3. For comparison, SMP-3's range of $f_D$ reaches 0.08, which fully rough turbulent flow gives only at a relative roughness above 0.05.

### SMP-43 · Black-oil parameters

No new draws: the API gravity from SMP-9 (21.3 to 39.9, inside the Vazquez–Beggs range of 10 to 40), the gas gravity from SMP-11 (0.55 to 0.90), and the gas–oil and water–liquid ratios from SMP-40's mapping. There is no bubble-point cap: gas dissolves up to the gas available. Separator conditions are standard conditions (`FluidModel`'s defaults). Wells with a small oil fraction get very high gas–oil ratios (above $10^4$ Sm³/Sm³), outside the correlations' range; Bjarne accepted them until the redesign, with no cap (2026-10-01).

### SMP-44 · Lift-gas temperature

Not drawn: the lift gas enters at $T_r$, as in v1 (THM-3).

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| SMP-1 | (20) | `well.py` `sample_well` | verifier: Distributions |
| SMP-2 | (21) | `well.py` `sample_well` | verifier: Distributions |
| SMP-3 | (22) | `well.py` `sample_well` | verifier: Distributions |
| SMP-4 | (23) | `well.py` `sample_well` | verifier: Distributions |
| SMP-5 | (24) | `well.py` `sample_well` | verifier: Distributions |
| SMP-6 | §4.1 | `well.py` `sample_well` | verifier: Distributions |
| SMP-7 | (25) | `well.py` `sample_well` | verifier: Distributions |
| SMP-8 | (26) | `well.py` `sample_well` | verifier: Distributions |
| SMP-9 | (27) | `well.py` `sample_well` | verifier: Distributions |
| SMP-10 | §4.2 | `well.py` `sample_well` (`pvt.WATER`) | verifier: Distributions |
| SMP-11 | (30) | `well.py` `sample_well` | verifier: Distributions |
| SMP-12 | (28)–(31) | `well.py` (`pvt.liquid_mix`) | verifier: Distributions |
| SMP-13 | (33) | `well.py` `sample_well` | verifier: Distributions |
| SMP-14 | (36) | `well.py` `sample_well` | verifier: Distributions |
| SMP-15 | §4.3 | `simulator.py` `BoundaryConditions.T_s` | verifier: Distributions |
| SMP-16 | (34) | `well.py` `sample_well` | verifier: Distributions |
| SMP-17 | §4.5 | `well.py` `sample_well` | verifier: Distributions |
| SMP-18 | (44) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-19 | (45) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-20 | (35) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-21 | (32) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-22 | (53)–(55) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-23 | §4.4.1 | `nonstationary_well.py` `NonStationaryBehavior` | verifier: Distributions |
| SMP-24 | (37)–(39) | `nonstationary_well.py` `NonStationaryBehavior.mass_fractions` | verifier: Distributions |
| SMP-25 | (40)–(43) | `nonstationary_well.py` `NonStationaryBehavior.reservoir_pressure` | verifier: Distributions |
| SMP-26 | (35) | `nonstationary_well.py` `NonStationaryWell.update_conditions` | verifier: Distributions |
| SMP-27 | §4.5.1 | `nonstationary_well.py` `NonStationaryWell.update_conditions` | verifier: Distributions |
| SMP-28 | §5.1 | `open_loop_stationary/generate_well_data.py` | verifier: Distributions |
| SMP-29 | §5.2 | `open_loop_nonstationary/generate_open_loop_nonstationary_well_data.py`, `init_utils.py` | verifier: Distributions |
| SMP-30 | Table 3 | `open_loop_stationary/generate_well_data.py` `simulate_well` | verifier: Distributions |
| SMP-31 | — | none (S-9) | property: tests/test_sampling.py (a rerun gives the same draws) |
| SMP-40 | — | — | property: tests/test_sampling.py (the mapped well reproduces v1's inputs) |
| SMP-41 | — | — | property: tests/test_sampling.py (the drawn distributions) |
| SMP-42 | — | — | property: tests/test_sampling.py (the drawn distributions) |
| SMP-43 | — | — | property: tests/test_sampling.py (the mapped fluid) |
| SMP-44 | — | — | spec-only: a rule that nothing is drawn |
