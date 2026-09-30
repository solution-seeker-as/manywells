# Model changes on `develop` since v1.0.0

*Step 4 output of `manywells-v2-plan.md`, input to Step 7. Drafted 2026-09-30 from `git log v1.0.0..develop -- src/manywells` and a comparison of the two code trees.*

Step 7 adds each change to the component files in `specs/model/` as an option with new equation IDs, writes its feature spec (`specs/features/NNN-<name>.md`), tags the code, and adds a v1-compatibility configuration in which every change is off. This list says what changed, where, which v1.0.0 equations it touches, and whether `develop` can already switch it off.

The row vectors of `specs/model/discretization.md` measure the gap. `tests/test_spec_vectors.py` runs them on `develop`'s nearest configuration to v1.0.0 today: a vertical `WellGeometry`, fixed `f_D`, dead oil with `wlr = 0` and `rho_o = rho_l`, ideal gas. In that configuration the momentum row (DISC-4), the choke row (CHK-1), the inflow temperature (THM-3), the gas law (PVT-GAS-1) and the liquid density (PVT-MIX-1) reproduce v1.0.0 exactly. The other rows differ, as measured below at the vectors' perturbed states; each difference is a strict expected failure there.

## Summary

| # | Change | Spec files | v1.0.0 IDs touched | Off switch on `develop` today |
|---|---|---|---|---|
| 1 | Deviated and L-shaped wells | geometry, discretization, thermal, slip | GEO-1, DISC-1, DISC-4, DISC-5, BAL-6, THM-2 | yes: `WellGeometry.vertical` |
| 2 | Inclination in the slip model | slip | SLIP-3, SLIP-6 | exact only at $\cos\theta = 1$ without the $10^{-9}$ guard |
| 3 | Surface tension from the fluid model | pvt/mixture, pvt/oil | PVT-MIX-5 | no |
| 4 | Friction from roughness and viscosity | friction, pvt/* | FRIC-2 | yes: set `f_D` |
| 5 | Unified fluid model | pvt/*, nomenclature | PVT-GAS-1, PVT-MIX-1 to PVT-MIX-4, INF-4 | partly: `oil_model`, `ideal_gas` |
| 6 | Real gas | pvt/gas | PVT-GAS-1 | yes: `ideal_gas=True` |
| 7 | Black oil | pvt/oil, pvt/mixture | PVT-OIL-1, PVT-MIX-1 | yes: `oil_model='dead_oil'` |
| 8 | Gas dissolving into oil | balances, discretization, inflow | BAL-3, DISC-2, DISC-3, INF-5 to INF-7 | no |
| 9 | Frictional heating and a gravity term | balances, thermal, discretization | BAL-5, DISC-5 | no |
| 10 | Lift-gas temperature | thermal | THM-3 | yes: `T_lg=None` |
| 11 | Inflow returns the liquid rate only | inflow | INF-3, INF-4 | — (interface) |
| 12 | Initial guess by `ca.rootfinder` | — (solver) | — | — |
| 13 | Other: defaults, validation, units, output | — | — | — |

## Changes

### 1. Deviated and L-shaped wells

- **What.** `geometry.py`: `WellGeometry` from an (MD, TVD) survey, which is also the grid (non-uniform grids allowed), with per-cell $\Delta\text{MD}$ and $\cos\theta$, and $D$ moved into the geometry. `from_survey` interpolates a sparse survey to a uniform MD grid; `vertical` builds v1's pipe. In the simulator: friction over $\Delta\text{MD}$, gravity over $\Delta\text{TVD} = \Delta\text{MD}\cos\theta$, heat loss over $\Delta\text{MD}$, the ambient temperature linear in TVD, and each point's closures use the inclination of the cell below it (point 0 uses cell 1's).
- **Off.** `WellGeometry.vertical(L, N, D)` reproduces DISC-4 exactly.
- **Commits.** `be80b70`, `28d240e`, `153b6c0`, `e826a9f`.

### 2. Inclination in the slip model

- **What.** `slip.py`: the bubbly–slug threshold of SLIP-6 becomes $0.25\cos\theta$, and the Taylor velocity is multiplied by $\sqrt{\cos\theta + 10^{-9}}\,(1 + \sin\theta)^{1.2}$ (Hasan, Kabir and Sayarpour, 2010, Eq. (A-10)).
- **Gap.** At $\cos\theta = 1$ the $10^{-9}$ changes $v_{\infty T}$ by $5\cdot10^{-10}$ relative, so the SLIP-2/SLIP-3 vectors fail at their tolerance of $10^{-12}$. `WellGeometry` already rejects $\cos\theta < 0$, so the guard is not needed for real square roots; dropping it would make the vertical case exact. Decide in Step 7.
- **Commits.** `8385ad0`, `9267ed9`.

### 3. Surface tension from the fluid model

- **What.** `FluidModel.surface_tension(p, T)`: the dead-oil correlation at the oil's density at standard conditions, $\rho_o$, instead of at the state's liquid density $\rho_l$ (PVT-MIX-5, `specs/discrepancies.md` D-8), with a live-oil correction for black oil (Abdul-Majeed and Abu Al-Soof, 2000, Eqs. (4)–(5), blended by a sigmoid at $R_{so} = 50$ Sm³/Sm³ with rate 0.5). The coefficients match the source, but its two branches do not meet at $R_{so} = 50$: the ratio $\sigma_{lo}/\sigma_{od}$ is 0.425 by (4) and 0.375 by (5), a 12% step that the sigmoid smooths. `black_oil.py`'s docstring says they meet continuously; correct it when specifying the option.
- **Gap.** With `wlr = 0` and `rho_o = rho_l` the two agree at a root, but not away from it: the SLIP-1 rows differ by up to 13% (W1) at the perturbed states. With water in the liquid they differ at roots too. The v1-compatibility configuration needs $\sigma_{od}(\rho_l, T)$ with the state's $\rho_l$.
- **Commits.** `12be414`.

### 4. Friction from roughness and viscosity

- **What.** `friction.py`: $f_D$ from the Reynolds number $\rho_m\lvert v_m\rvert D/\mu_m$ and the relative roughness (default $4.5\cdot10^{-5}$ m), by Chen (1979); Haaland (1983) is implemented but cannot be chosen from `WellProperties`. Laminar $64/\text{Re}$ blended with the turbulent value by a sigmoid at Re = 3000 with rate 0.005, and smooth-max guards $\text{Re} \ge 1$ and $\text{Re} \ge 1000$ for the turbulent branch. The mixture viscosity is mass-weighted (Hasan et al. 2010, Eq. (A-3)); oil viscosity by Beggs–Robinson, dead and live (valid 100–295 °F, colder than which the wellhead often is); water by a Vogel–Fulcher–Tammann-type correlation; gas by Lee–Gonzalez–Eakin; the liquid's by volume with `wlr`.
- **Off.** Setting `WellProperties.f_D` gives FRIC-2.
- **Commits.** `8284ab2`, `e7d5e14`, `e70034b`, `04d5811`, `78747e1`, `8c7d7cc`.

### 5. Unified fluid model

- **What.** `pvt/fluid.py`: `FluidModel` takes densities at standard conditions ($\rho_o$, $\rho_{g,\text{sc}}$, $\rho_w$), a gas–oil ratio and a water–liquid ratio, and derives $f_g$, $R_s$, $c_{pl}$ and $\rho_l$; `oil_model` and `ideal_gas` choose the options. The gas law row is now $\rho_g - c_\text{bar}p/(Z R_s T)$, equivalent to PVT-GAS-1 (the vectors convert it exactly). `p_sep` and `p_bubble` are in Pa while every method takes bar (`improvements.md` §1.6).
- **Off.** The mapping from v1's parameters is in `specs/sampling.md` (SMP-40).
- **Commits.** `539fbd8`, `64bad4e`, `a77d02f`, `13c2ab5`, `28c5ca7`, `bbaafad`.

### 6. Real gas

- **What.** `pvt/gas.py`: Papay's z-factor with Sutton's pseudo-critical properties from the gas gravity (valid for $p_{pr} < 6$, $T_{pr} > 1.05$).
- **Off.** `ideal_gas=True`; PVT-GAS-1 rows match exactly.
- **Commits.** `8c9b179`.

### 7. Black oil

- **What.** `pvt/black_oil.py`: Vazquez–Beggs $R_{so}$ and $B_o$ with the separator gas-gravity correction, API gravity 10–40 enforced; an optional bubble-point cap by smooth min; live-oil density $(\rho_o + R_{so}\rho_{g,\text{sc}})/B_o$; the liquid density mixes it with water by the water–liquid ratio at standard conditions, not the in-situ one. `water_fvf` has the wrong sign and is unused (`improvements.md` §1.2); fix it before specifying it.
- **Off.** `oil_model='dead_oil'`; with `wlr = 0` and `rho_o = rho_l`, PVT-MIX-1 rows match exactly.
- **Commits.** `5708460`, `1e80f3a`.

### 8. Gas dissolving into oil

- **What.** `SSDFSimulator._gas_and_liquid_flow_rate`: dissolved gas $w_d = \operatorname{smin}(R_{so}\,\rho_{g,\text{sc}}/\rho_o \cdot w_o,\ w_g)$, gas rate $\operatorname{smax}(w_g + w_{lg} - w_d,\ 0)$, liquid rate $w_l + w_d$. The mass rows become $A\alpha\rho_g v_g = w_g(p_i, T_i)$ and the liquid equivalent at every point, replacing the flux continuity of DISC-2 and DISC-3. The liquid inflow rate reaches those rows through hidden state, `self._w_l_inflow` (`improvements.md` §2.4).
- **Gap.** With dead oil ($R_{so} = 0$) the smooth min is $-\epsilon/(4 w_g) \approx -2.5\cdot10^{-7}/w_g$ kg/s, not 0, and with $w_g = 0$ it is $-5\cdot10^{-4}$ kg/s, which makes a gas rate of $8\cdot10^{-4}$ kg/s out of nothing. Measured: INF-6 rows differ by up to $2.3\cdot10^{-7}$ relative and INF-7 by $1.6\cdot10^{-8}$ (W1). The v1-compatibility configuration needs $\Gamma = 0$ exactly (BAL-3), so the path must be bypassed, not only fed $R_{so} = 0$. The mass rows' form also differs, so DISC-2 and DISC-3 vectors are skipped until the configuration exists.
- **Commits.** `5708460`, `64bad4e`, `28c5ca7`.

### 9. Frictional heating and a gravity term in the energy balance

- **What.** `_differential_equations`: $dT = dT_\text{heat} - dT_\text{fric} + dT_\text{grav}$ with $dT_\text{fric} = \Delta\text{MD}\,(1-\alpha)v_l F/\text{cp\_flux}$ and $dT_\text{grav} = \Delta\text{TVD}\, g\,(\text{mass\_flux} - \text{liq\_flux}\,\rho_m)/\text{cp\_flux}$. Derivation in `docs/thermal_energy_modeling.md`.
- **Gap.** Always on; no switch. Measured: DISC-5 rows differ by up to 0.62 K per cell (W2). The v1-compatibility configuration needs a switch (plan, Step 7 item 2). Pin the terms with test vectors in `thermal.md` (`improvements.md` §3).
- **Commits.** `82344c6`, `be80b70` (the TVD in the gravity term).

### 10. Lift-gas temperature

- **What.** `BoundaryConditions.T_lg`; when $w_{lg} > 0$, $T_0$ is the heat-capacity-weighted mix of the reservoir fluid at $T_r$ and the lift gas at $T_{lg}$. The temperature lower bound becomes $\min(T_s, T_{lg})$.
- **Off.** `T_lg=None` means $T_r$; THM-3 rows match exactly.
- **Commits.** `f374b0c`, `f82f49c`, `8b05b18` (fixed a crash with gas lift).

### 11. Inflow returns the liquid rate only

- **What.** `inflow.py`: `liquid_mass_flow_rate(p, p_r)` replaces `mass_flow_rates`; the gas–liquid split comes from `FluidModel.f_g` (INF-4 moves into the simulator). `FixedFlowRate` fixes only the liquid rate, and its gas follows from $f_g$ (a change of INF-3). Vogel and the productivity index are unchanged (INF-1 and INF-2 vectors pass). The default inflow is `ProductivityIndex(k_l=0.5)`.
- **Commits.** `787d771`, `041b063`, `f75318f`.

### 12. Initial guess by `ca.rootfinder`

- **What.** The cellwise march uses CasADi's Newton rootfinder, without bounds, instead of a bounded Ipopt solve per cell, about 100 times faster. Not part of the model; which root `develop` reaches from its default guess has to be measured, not assumed to match v1.0.0 (plan, "Multiple roots and stability").
- **Commits.** `11cb4ac`.

### 13. Other

- Defaults: black oil, real gas, roughness-based friction and productivity-index inflow. The default Bernoulli choke with $K_c = 0.1A$ is set in `SSDFSimulator.__init__`, which writes it into the caller's `WellProperties` (`improvements.md` §2.5).
- Validation: $T_r$, $T_s$, $T_{lg} > 0$; roughness $> 0$; $f_D > 0$ when set; the geometry checks of `WellGeometry`.
- Units: `CF_PRES` became `CF_BAR`, with the same value, and the constants moved to `units.py`.
- Output: `solution_as_df` adds `md` and `tvd` columns, drops `z`, and uses the inclination in the regime label.
