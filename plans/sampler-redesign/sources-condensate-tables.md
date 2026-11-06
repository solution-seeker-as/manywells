# Sampler redesign — sources: gas condensates in tabular and black-oil PVT

*A source review for Bjarne's ruling on D-64 (`decisions-r5.md`), 2026-11-02. It asks how established simulators and libraries represent gas condensates in tabular or black-oil PVT, and what that means for PVT-TAB-1's two checks that refuse a condensate's table. Plans are drafts, not specs: every option below changes `specs/model/pvt/table.md` and needs Bjarne's sign-off.*

## Summary

- **Two families.** Reservoir simulators use the modified black oil: per phase, the solution gas–oil ratio $R_s$ and the vaporised oil–gas ratio $R_v$, as functions of pressure at one temperature. Pipe-flow simulators read tables on a $(p, T)$ grid that give the gas mass fraction of the hydrocarbon stream and each phase's density directly (OLGA's `.tab` files, written by PVTsim, Multiflash or NeqSim, an open-source Java library). ManyWells' PVT-TAB-6 is the second family written in black-oil units. Its $R_{so}$ is a linear function of the gas mass fraction $X$ at the table's GOR. The open-source pipe simulator Marlim3 converts flash results to $R_s$ with the same formula.
- **$R_{so} < 0$ is not $R_v$.** At one GOR, a modified black-oil state maps exactly onto PVT-TAB-1's $R_{so}$, $B_o$ and $Z$. The reverse does not hold. $R_{so}$ is a net figure: the gas dissolved in the liquid minus the oil vaporised in the gas, both in gas-equivalent units. The two forms differ off the table's GOR, with lift gas, and above the dew point. On Pedersen et al.'s Table 3.8 condensate, $R_{so} > 0$ at 143 of 385 two-phase nodes where the gas carries up to 47% of the stock-tank oil.
- **No tool found checks the monotonicity of a net quantity.** OPM Flow requires $R_s$ to increase strictly along its live-oil table, and $R_v$ to fall strictly within each undersaturated wet-gas record. It does not check the saturated $R_v$ against pressure. Whitson and Brulé's own condensate, and Pedersen's, have a minimum in $R_v$ at low pressure. No check was found on the gas mass fraction of OLGA's tables either.
- **Above the dew point** no established form needs a trace. They separate the amount of a phase from its intensive properties: OPM switches its primary variable to $R_v$ where the oil phase vanishes, and OLGA's tables carry $X = 1$ with the absent liquid's properties filled from the nearest node where it exists. ManyWells' $B_o$ holds both the amount and the volume of the oil phase, so its liquid density is $0/0$ where there is none. That is why R5 needed the trace.
- **Recommendation: option A′.** Keep R5's trace and drop the $R_{so}$ check. In its place, add checks that hold for every fluid, as established tools check theirs: the oil phase's mass is positive, and the gas density $p/Z$ rises with pressure at every temperature, as Eclipse's dry-gas table requires $B_g$ to fall. Both backends and the rows stay as they are. A declared kind (option B) can be added if Bjarne wants the old check back for oils. The $R_v$ form (C) costs a feature, and on the evidence here it loses PVT-TAB-6's exactness at the nodes. The mass-fraction form (D) is the cleaner long-term layout, but where the stream has no water it leaves the liquid's mass rate at 0, which the solver has not met.

## The question

PVT-TAB-1 holds $R_{so}$, $B_o$ and $Z$ at each node of a $(p, T)$ grid, at the table's GOR $R_{go}$. PVT-TAB-6 makes them from a flash's vapour mass fraction $X$ of the hydrocarbon stream and its phase densities $\rho_L$ and $\rho_V$, with $m = \rho_o + R_{go}\rho_{g,\text{sc}}$:

$$R_{so} = R_{go} - \frac{X m}{\rho_{g,\text{sc}}}, \qquad B_o = \frac{(1 - X) m}{\rho_L}, \qquad Z = \frac{c_\text{bar}\thinspace p}{\rho_V R_s T}.$$

A gas condensate breaks two of PVT-TAB-1's checks:

- **Retrograde region.** Below the dew point, liquid condenses as the pressure falls. $X$ rises towards 1 as the pressure rises towards the dew point, so $R_{so}$ falls. The check "$R_{so}$ nondecreasing in $p$" refuses this.
- **Above the dew point.** The stream is all vapour: $X = 1$, $R_{so} = -\rho_o/\rho_{g,\text{sc}}$ and $B_o = 0$. The check "$B_o > 0$" refuses this. Without water, PVT-MIX-6's liquid density is $0/0$.

R5 proposed a liquid trace $\delta = 10^{-4}$ and dropping the monotonicity check (D-64, `spec-changes.md`). The alternatives on the table were a declared kind and no condensate tables.

## How established tools represent condensates

### Across tools

| Tool | Representation | Liquid carried in the gas | Above the dew point | Retrograde region | Checks on the table | Source |
|---|---|---|---|---|---|---|
| Traditional black oil | $R_s$, $B_o$, $B_g$ against $p$ | none: "reservoir gas does not yield liquids" | not represented | not represented | – | Whitson and Brulé (2000), §7.2; Golan and Whitson (1991), §1.2, Eqs. 1.10–1.13 |
| Modified black oil | $R_s$, $B_o$ of the liquid; $R_v$ (Whitson and Brulé's $r_s$), $B_{gd}$ of the vapour; constant surface densities | $R_v$, stock-tank oil per surface gas of the vapour | undersaturated gas: $R_v$ below its saturated value | saturated $R_v(p)$ falls below the dew point; the split follows from mass balance (Eq. 7.6) | none stated; constant surface gravities (Eq. 7.7) assumed in most formulations | Whitson and Brulé (2000), §7.3, Eqs. 7.2–7.8, 7.17–7.20, Figs. 7.6–7.11 |
| Partial densities | four partial densities: mass of each surface component per reservoir volume of each phase | the partial density of surface oil in the gas | – | – | – | Kniazeff and Naville (1965), in Whitson and Brulé (2000), §7.5, Eqs. 7.48–7.49 |
| Eclipse format, OPM Flow | PVTO (live oil, keyed by $R_s$) and PVTG (wet gas, keyed by $p$), against pressure only; surface densities from DENSITY | $R_v$ (OPM: $r_{og}$) | gas-only cell: $r_{og}$ becomes the primary variable; PVTG's undersaturated records give $B_g(p, R_v)$ | saturated $R_v(p)$, any shape | PVTO: $R_s$ strictly increasing, undersaturated $p$ strictly increasing and $B_o$ strictly decreasing; PVTG: $p$ strictly increasing, saturated $R_v$ unchecked, undersaturated $R_v$ strictly decreasing; PVDG: $B_g$ strictly decreasing, $\mu_g$ nondecreasing; at least one undersaturated PVTG record; a non-monotone saturated $B_o$ is a warning | opm-common: `Tables.cpp` 503–512, 633–642, 1083–1087; `TableColumn.cpp` 58–63; `TableManager.cpp` 1658–1667, 1755–1759; `WetGasPvt.cpp` 64–68, 131–143; Rasmussen et al. (2021), §2.1, Eqs. 2 and 14 |
| MRST | reads Eclipse PVTO and PVTG; $\rho_g = b_g(\rho_{gS} + r_v\rho_{oS})$ | $r_v$ | undersaturated gas from PVTG | saturated $r_v(p)$, linear with $(0, 0)$ prepended | none in the assignment code | `BlackOilDensity.m` 32–41; `assignPVTG.m` 11–16, 29; `assignPVTO.m` 20–31; Lie (2019), §11.4.3, §11.5, p. 382 |
| NeqSim, black-oil converter | one temperature, a TP flash of the fixed composition at each pressure, each phase separated alone at standard conditions: $R_s$, $B_o$, $R_v$, $B_g$; writes PVTO and PVTG | $R_v$ of the vapour | no oil phase: $B_o = 1$ and $R_s = 0$ by default, $R_v$ of the stream's vapour; flags the fluid as a retrograde condensate | $R_v(p)$ as the flash gives it | the converter checks nothing; a separate validator (oil tables) errs on $R_s < 0$ and on non-monotone $R_s$ and $B_o$; the PVTG writer gives saturated records only | NeqSim `BlackOilConverter.java` 51–59, 129, 247–316; `EclipseEOSExporter.java` 457–472; `BlackOilTableValidator.java` 17–18, 134–152 |
| OLGA `.tab` (PVTsim, Multiflash, NeqSim) | $(p, T)$ grid: phase densities and their derivatives, gas mass fraction of gas plus oil (keyword `RS`), viscosities, enthalpies, surface tensions; dew and bubble pressures per temperature | in the gas mass fraction $X$ | $X = 1$; the liquid's columns filled from the nearest node where a liquid exists | $X(p)$ as the flash gives it | OLGA refuses a zero phase density; no check found on $X$ | pyfas `tab.py` 32–42; its sample files (`COLUMNS`, line 24; "GAS MASS FRACTION OF GAS + OIL", line 4552); NeqSim `OlgaTableGridFiller.java` 6–17, keyword writer 721–725, fixed writer 333–343 |
| LedaFlow | "Black Oil", "PVT tables" and "Composition tracking" among its thermodynamics options | – | – | – | – | LedaFlow web page; public material says no more |
| PIPESIM | black-oil correlations said to cover "light oil and condensate"; "PVT files" interpolated, from PVTsim, Multiflash and others; compositional EOS | – | – | – | – | PIPESIM 2017 Technical Reference (2018), p. 7; public material does not say how its black oil treats retrograde condensation |
| Prosper/GAP, tNavigator | Prosper: black oil and compositional; tNavigator: Eclipse's PVTG | $R_v$ in tNavigator | – | – | whitson+'s export: $R_s$ monotone, $R_v$ unique; "many reservoir simulators do not accept non-monotonic" tables | whitson+ manual, BOT module, Export formats; Petex product page; Prosper's retrograde-condensate model is not described in public material |
| Marlim3 (open-source pipe simulator) | flash-table mode from a `.tab` in the keyword format; black oil; compositional | black oil from flash results via the gas mass fraction | – | gas-dominated fluids: "condensate drops out of gas" by quality relations | – | Marlim3 documentation: Fluids; Developer guide, Fluid thermodynamics |
| `fluids`, `thermo` | no black-oil or table PVT; `fluids`' two-phase correlations take the mass quality $x$ and the phase densities; `thermo` flashes | – | – | – | – | `fluids/two_phase.py` 152 (`Beggs_Brill`), 2461; `thermo/flash/flash_vl.py` 61 |
| ManyWells PVT-TAB | $(p, T)$ grid: $R_{so}$, $B_o$, $Z$ (+ $\mu_o$, $\mu_g$, $\sigma_{go}$), at the table's GOR, with an optional GOR axis | net: $R_{so} < 0$ where the gas carries more oil than the liquid dissolves gas | $B_o = 0$, refused; R5: a trace $\delta$ | $R_{so}$ falls with $p$, refused | finite; increasing grids; $B_o, Z > 0$; $R_{so}$ nondecreasing in $p$; $R_{so} \ge 0$ without $\mu_o$ and $\sigma_{go}$ | `specs/model/pvt/table.md`, PVT-TAB-1 and PVT-TAB-6 |

Source files are cited at these commits: opm-common `7364486`, MRST `fa4ff98`, NeqSim `89a28c6` (the black-oil classes exist at its Java tag `v3.4.0` too), pyfas `d7432ec`, `fluids` `652a013`, `thermo` `2bb466e`.

### Notes per tool

- **Modified black oil** (Whitson and Torp 1983; Coats 1985; Whitson and Brulé 2000, §7.3). The equilibrium phases of a depletion experiment (constant composition expansion, constant volume depletion or differential liberation) pass separately through the separator (Fig. 7.7). The liquid gives $R_s$ and $B_o$ per stock-tank volume of its own oil. The vapour gives $R_v$ and $B_{gd}$ per surface volume of its own gas (Eq. 7.2). Most formulations assume one stock-tank oil gravity and one surface gas gravity for both phases (Eq. 7.7). Coats determines $R_v$ from a simulated depletion and then sets $B_{gd}$, $B_o$ and $R_s$ "so that reservoir-oil and -gas densities ... are honored". The phase densities follow from Eq. 7.8, and the volumes and masses of the two phases per stock-tank volume of total oil from Eqs. 7.17–7.20, with $F_{oo}$, the share of the stock-tank oil that comes from the liquid (Eq. 7.6). Section 7.4 notes that "multiphase flow in pipe is another obvious application" and gives the in-situ rates in tubing (Eq. 7.37). Fig. 7.10 shows the condensate NS-1's $R_v$ against pressure with a minimum near 1,000–1,500 psia. Its rise at lower pressure means $R_v$ is not monotone along the saturated curve. Section 7.5 adds the partial-density form. It keeps an exact mass balance because the surface gravities' variation enters the properties. The Drohm–Goldthorpe correction (Eq. 7.50) does the same for the modified black oil by scaling with per-phase surface densities.
- **OPM Flow and Eclipse format.** The black-oil equations carry surface oil in both phases as $b_o s_o + r_{og} b_g s_g$ and surface gas as $b_g s_g + r_{go} b_o s_o$ (Rasmussen et al. 2021, Eq. 2). Where a phase disappears, the third primary variable becomes $r_{go}$ (no gas) or $r_{og}$ (no oil) in place of $s_g$ (Eq. 14). So no trace is needed. The table schemas enforce the order of each column on input, with an `invalid_argument` "Non-monotonic values in keyword ..." (`TableColumn.cpp` 58–63). The saturated $R_v$ is declared `RANDOM`, with no order (`Tables.cpp` 511). The dew-point pressure of a gas with a given $R_v$ is found by Newton's method on the saturated $R_v(p)$ (`WetGasPvt.hpp` 328–379). A non-monotone $R_v(p)$ makes that inverse non-unique. That is a likely reason for whitson+'s "unique $R_v$" (our inference). PVTO and PVTG tabulate against pressure only. Lie (2019, §11.4.3) explains: "Since reservoirs are almost thermally inert, it is usually sufficient to express these quantities as function of pressure and not temperature."
- **NeqSim.** Its black-oil converter is Whitson and Torp's method on a constant-composition path at one temperature (`BlackOilConverter.java` 247–316). It flags a fluid that is gas at the top of the grid and drops out liquid below as a retrograde condensate: "A retrograde fluid needs a vaporised-oil (PVTG) treatment; a standard black-oil PVTO/PVDG table cannot represent its liquid dropout" (51–59). Its Eclipse writer gives one saturated PVTG row per pressure and no undersaturated records (`EclipseEOSExporter.java` 457–472). OPM refuses such a table: "PVTG tables are invalid: The last table must exhibit at least one entry for undersaturated gas!" (`WetGasPvt.cpp` 131–143). We infer this from the two codes and did not run them. Its OLGA writers set the gas mass fraction to 1 where there is no oil and to 0 where there is no gas. Negative values are clipped to 0. The absent phase's columns take the nearest present node's values: "the same convention commercial table generators use ... OLGA never evaluates them for a phase whose mass fraction is zero" (`OlgaTableGridFiller.java` 6–17).
- **OLGA's table format.** It has two layouts, fixed and keyword (pyfas `tab.py` 32–42). A PVTsim keyword file lists `COLUMNS = (PT,TM,ROG,ROHL,ROWT,DROGDP,...,RS,RSW,VISG,...)`, where `RS` is the gas mass fraction. Its header carries the dew and bubble pressures per temperature (pyfas sample file, lines 8–25). In the fixed twin, `RS` is labelled "GAS MASS FRACTION OF GAS + OIL" (line 4552). No public document was found on what OLGA checks beyond the nonzero densities.
- **whitson+** (a commercial PVT tool with a public manual) generates modified black-oil tables by a simulated constant-composition expansion. Each stage's equilibrium phases pass through the surface process. It writes PVTO and PVTG for Eclipse and PVTG for tNavigator. Its manual says non-monotonic $R_s$ "may occur for lean reservoir fluids (e.g. gas condensates)", that "many reservoir simulators do not accept non-monotonic" tables, and that a non-monotonic case "always leads to BO tables that are not entirely closed".
- **Marlim3** derives black-oil properties from flash results as $R_s = (\rho_{L,ST}/\rho_{V,ST})(w_{v,ST} - w_v)/(1 - w_{v,ST})$ and $B_o = (\rho_{L,ST}/\rho_L)(1 - w_v)/(1 - w_{v,ST})$ (Developer guide, Black-oil properties from compositional results). With $w_{v,ST} = R_{go}\rho_{g,\text{sc}}/m$ these are PVT-TAB-6's $R_{so}$ and $B_o$ exactly. Its documentation says the compositional model "is required for rich-gas/condensate systems". It also says latent heat in "extreme retrograde condensation" can make its solver unstable.
- **Textbooks on well performance.** Golan and Whitson (1991, §1.2) compute the phase rates along the flow path from the traditional black oil: $q_o B_o$ and $q_o(R - R_s)B_g$ (Eqs. 1.10–1.13). That is ManyWells' PVT-OIL-13. They note that "more complicated equations of state may be required for gas condensate fluids" (p. 12). Pedersen et al. (2015, §9.1, p. 197) describe PVT packages as "well suited for generation of PVT property tables needed as input to process and flow simulation programs". They warn that a model regressed at reservoir temperature need not hold at other conditions.

### How a flash becomes each form

- **Modified black oil.** Two extra flashes per node: each equilibrium phase is separated alone (Whitson–Torp). The surface densities are then either held constant (Eq. 7.7) or carried per phase (Eq. 7.50). The depletion path is a choice: CVD for a condensate reservoir, CCE in whitson+ and NeqSim.
- **OLGA-style table.** One flash per node: $X$, $\rho_L$, $\rho_V$ and the transport properties. The absent phase is filled from the nearest node where it exists.
- **PVT-TAB-6.** The same flash as OLGA's, written as $R_{so}$, $B_o$ and $Z$ at the stream's single-stage separation. It is exact at the nodes by construction: PVT-OIL-13 gives back $X$.

## What this means for ManyWells

### PVT-TAB-6 is the mass-fraction form in black-oil units

At the table's GOR, $X = (R_{go} - R_{so})\thinspace\rho_{g,\text{sc}}/m$, and the liquid density is $\rho_L = (\rho_o + R_{so}\rho_{g,\text{sc}})/B_o$. An OLGA table's `RS` and `ROHL` therefore map onto $R_{so}$ and $B_o$ at the nodes, and `ROG` maps onto $Z$. Marlim3's conversion is the same formula. At the table's GOR, PVT-TAB-6 carries the same information as such a table's gas mass fraction and phase densities. The difference is in one column: $B_o$ is the oil phase's volume per stock-tank volume of the stream. It holds both the amount of the phase and its density, and both vanish together above the dew point.

### $R_v$ and $R_{so} < 0$

Given a modified black-oil state $(R_s, B_o, R_v, B_{gd})$ at a node, with one stock-tank oil density $\rho_o$ and one surface gas density $\rho_{g,\text{sc}}$ (Eq. 7.7), PVT-TAB-1's columns at the GOR $R_{go}$ are

$$F_{oo} = \frac{1 - R_v R_{go}}{1 - R_s R_v}, \qquad R_{so} = F_{oo} R_s - (1 - F_{oo})\frac{\rho_o}{\rho_{g,\text{sc}}}, \qquad B_o^{\text{tab}} = F_{oo} B_o, \qquad \rho_V = \frac{\rho_{g,\text{sc}} + R_v\rho_o}{B_{gd}},$$

with $F_{oo}$ Whitson and Brulé's Eq. 7.6, the masses of Eq. 7.20 and the gas density of Eq. 7.8, and $Z$ from $\rho_V$ (PVT-TAB-6). The liquid's mass per stock-tank volume of the stream is $F_{oo}(\rho_o + R_s\rho_{g,\text{sc}}) = \rho_o + R_{so}\rho_{g,\text{sc}}$, so the mapping is exact in mass. Where the vapour carries no oil ($R_v = 0$), $F_{oo} = 1$ and $R_{so} = R_s$. Where the stream is all vapour ($R_v R_{go} = 1$), $F_{oo} = 0$ and $R_{so} = -\rho_o/\rho_{g,\text{sc}}$.

So $R_{so} < 0$ is neither $R_v$ nor a sign that the gas carries oil. It is the net transfer, and the two forms differ in four ways:

1. **One number for two.** The map from $(R_s, R_v)$ to $R_{so}$ is many to one. A volatile oil or a condensate at low pressure has $R_{so} > 0$ while its gas carries oil. On Pedersen et al.'s Table 3.8 condensate, $R_{so} > 0$ at 143 of 385 two-phase nodes, where the gas carries up to 47% of the stock-tank oil (evidence (a)). $R_{so}$ falls below 0 only where the oil vaporised outweighs the gas dissolved.
2. **Other GORs.** $R_s$ and $R_v$ are properties of the equilibrium phases, so the split at another GOR follows from Eq. 7.6. $R_{so}$ is tied to its table's GOR. Read at another GOR by PVT-OIL-13, it vaporises oil in proportion to the oil, where the modified black oil vaporises it in proportion to the gas. ManyWells' GOR axis (PVT-TAB-7) avoids the question by interpolating between flashes at several GORs.
3. **Lift gas.** In the modified black oil, dry lift gas can take up oil up to $R_v$. In PVT-OIL-13 it changes nothing in the oil. Lift gas in condensate wells is rare.
4. **Above the dew point.** The modified black oil keeps the liquid's density, $(\rho_o + R_s\rho_{g,\text{sc}})/B_o$, defined when the amount $F_{oo}$ is 0. OLGA's tables keep $\rho_L$ as its own column. $B_o^{\text{tab}} = F_{oo} B_o$ loses it, hence the trace.

### Measured on Pedersen et al.'s Table 3.8 condensate

`plans/evidence/condensate_rv_form.py` rebuilt the stream of `table_fluids.py`'s proof table (single-stage separation; GOR 2338 Sm³/Sm³; 531 nodes, 280–440 K, 1–460 bar). At each node it separated each phase of NeqSim's flash alone at standard conditions. The flash reproduces the table's $X$ to $2\cdot10^{-9}$.

- **(a) The mapping with one surface density per component.** At the stream's GOR, the modified black oil gives $X$ within 0.0064 at the median and 0.12 at most, and $R_{so}$ within 393 Sm³/Sm³ (the table's $R_{so}$ runs from −848 to 744). Its liquid density is 3.4% low at the median, from −7.4% to +3.1%. The main cause is the separation: separating each phase alone gives a GOR from −30% to +14% of the stream's. With that GOR in Eq. 7.6, $X$ is within 0.0049 at the median and 0.053 at most. The surface oils from the two phases range from 768 to 872 kg/m³ against the stream's 807. Eq. 7.6 itself holds to $3\cdot10^{-16}$. PVT-TAB-6 is exact at the nodes; a modified black-oil table made the usual way is not.
- **(b) Monotonicity in pressure, two-phase nodes, at each temperature.** $R_{so}$ falls at 245 pairs of neighbouring nodes and PVT-TAB-6's $B_o$ at 217. The liquid's own $R_s$ and $B_o$ fall at none. $R_v$ falls at 60: it has a minimum between 5 and 70 bar from 300 to 440 K, at a higher pressure the hotter the fluid, and at the lowest node, 1 bar, at 280 K. Whitson and Brulé's Fig. 7.10 has the same shape.
- **(c) Read at another GOR.** On the base table's two-phase nodes, the results against the recombined stream's own flash:

  | GOR | Rule | $X$: median / largest error | No-slip density: median / largest error |
  |---|---|---|---|
  | ×0.8 | $R_{so}$, black-oil rule (PVT-OIL-13) | 0.019 / 0.16 | 3.0% / 7.0% |
  | ×0.8 | $R_s$ and $R_v$, Eq. 7.6 | 0.029 / 0.24 | 0.44% / 3.7% |
  | ×1.25 | $R_{so}$, black-oil rule | 0.014 / 0.14 | 2.5% / 6.7% |
  | ×1.25 | $R_s$ and $R_v$, Eq. 7.6 | 0.022 / 0.14 | 0.45% / 5.1% |

  Neither form transfers within the GOR axis's 0.01–0.3% at the median (feature 018, design choice 5). The modified black oil's better density is partly its base error offsetting: at the table's own GOR it is already 0.07% off at the median and 4.0% at most.

### Checks that established tools apply, and which ManyWells could keep

| Check | Where | For ManyWells |
|---|---|---|
| $R_s$ strictly increasing (PVTO's key) | OPM | ManyWells has no $R_s$. Only an $R_v$ form would; whitson+ reports lean condensates that break it |
| saturated $R_v$ monotone in $p$ | not checked by OPM (`RANDOM`); whitson+ asks for a unique $R_v$ | no: Whitson and Brulé's NS-1 (Fig. 7.10) and Pedersen's condensate (60 falls) have a minimum |
| undersaturated $B_o$ strictly decreasing in $p$ (density rises with pressure in one liquid phase) | OPM | possible: the liquid density nondecreasing in $p$ where the stream is all liquid ($R_{so}$ the table's GOR) holds, to $10^{-6}$, on the six of the eight tables measured that have such nodes |
| $B_g$ strictly decreasing in $p$ (PVDG) | OPM | yes: the gas density $p/Z$ strictly increasing in $p$ at each $T$ holds at every pair of nodes of all eight tables |
| phase densities not zero | OLGA (as NeqSim reports) | yes: implied by $B_o > 0$ and $Z > 0$, and by a positive oil-phase mass, $\rho_o + R_{so}\rho_{g,\text{sc}} > 0$, which PVT-TAB-1 does not check today |
| $R_s \ge 0$; $B_o \ge 1$ (a warning) | NeqSim's validator, for oil tables | no: PVT-TAB-1 already allows $R_{so} < 0$ and $B_o < 1$ by design |
| $R_{so} \le R_{go}$ | – | no: `tests/data/pvt/vazquez_beggs.csv` has $R_{so}$ above its GOR at 188 nodes, as PVT-TAB-3 allows |

The eight tables are the five proof tables of `table_fluids.py` and `tests/data/pvt/`'s `synthetic_eos_oil.csv`, `vazquez_beggs.csv` and `synthetic_eos_oil_360K.csv` (evidence `tables`).

## Options for Bjarne

| Option | What changes | For | Against | Cost |
|---|---|---|---|---|
| **A**, R5 as drafted: a trace $\delta = 10^{-4}$ where the stream is all vapour; drop the $R_{so}$ check | PVT-TAB-1 and PVT-TAB-6 text (`spec-changes.md`); `table.py`: one check removed, `from_phase_split` clips $X$ at $1 - \delta$ and fills the liquid's columns from the nearest two-phase node at the same temperature; one test turned round | no model change; both backends unchanged; the fill is the absent-phase convention of OLGA's table generators | loses the only order check for oil tables; $B_o$ down to $5\cdot10^{-4}$ | about half a day |
| **A′**, A with checks that hold for every fluid (recommended) | as A, plus two checks in PVT-TAB-1: $\rho_o + R_{so}\rho_{g,\text{sc}} > 0$ at every node, and $p/Z$ strictly increasing in $p$ at every $T$; optionally the liquid density nondecreasing where the stream is all liquid | as A; keeps an order check, the one PVDG has, for oils and condensates alike; a positive oil-phase mass guards the new range of $R_{so}$ | the new checks catch fewer oil-table errors than the old one | about a day with tests |
| **B**, a declared kind | as A, plus a field (`kind`: `'oil'` or `'condensate'`, a `# kind:` header key, the same across a GOR axis); the $R_{so}$ check only for `'oil'` | the old check stays where it caught errors | a field whose only role is a check; a near-critical oil may need `'condensate'`; inferring the kind from a table is fragile (a condensate table below its dew point has no all-vapour node) | about a day; new API (MINOR) |
| **C**, an $R_v$ form in the model | a new PVT model: $R_s$, $B_o$, $R_v$, $B_{gd}$ of the saturated phases on $(p, T)$, with undersaturated data; PVT-OIL-13 becomes a two-way transfer with the split of Eq. 7.6 and smooth clamps at both saturation limits; the gas law reads $R_v$; new spec IDs; both backends; vectors; backend comparison | carries the phases' compositions; transfers across GOR and lift gas by mass balance; $R_s$ and $B_o$ are monotone here and can be checked | not exact at the nodes (evidence (a)); the separation path and the surface densities become choices; no established format holds it on $(p, T)$, since PVTO and PVTG tabulate against pressure only; the GOR axis already does better (evidence (c)); new kinks for the CasADi backend | a feature spec and weeks |
| **C1**, an Eclipse reader | PVTO and PVTG converted in Python to PVT-TAB-1 at the fluid's GOR by Eqs. 7.6–7.8, as PVT-TAB-7 reads an axis | lets users bring reservoir-simulator tables | still needs A, A′ or B for a condensate; one temperature only; 018 put readers out of scope | a few days, later |
| **D**, the mass-fraction form | $\rho_L$ in place of $B_o$ (or both, $B_o$ derived as $(\rho_o + R_{so}\rho_{g,\text{sc}})/\rho_L$); $X = 1$ allowed; PVT-MIX-6 gives $\rho_l = \rho_L$ directly where $\alpha_{w,l} = 0$ (a branch on a constant); `liquid_fvf` from the derived $B_o$; Rust `table.rs` and `fluid.rs`; the CSV schema; the `TAB` vectors; 018's GOR-axis measurement redone with $\rho_L$ interpolated | an OLGA table maps column for column; no artificial $B_o$; the liquid density is interpolated directly | the liquid's mass rate is still 0 above the dew point without water, which the drift-flux rows have not met; a floor on it would move the trace rather than remove it; breaks existing table files unless $B_o$ stays accepted | several days to a week, with measurements |
| **E**, no condensate tables | none now; condensates stay R3's interim black oil | no work | against the Goals' decision of 2026-10-30; the black oil keeps the condensate liquid (well 15: a free-gas share of 0.43 at $(p_r, T_r)$ against the table's 0.84) | none, and a ruling to revise the Goals |

Each option's effect on the parts of ManyWells:

- **The spec.** A, A′ and B change PVT-TAB-1's checks and PVT-TAB-6's text. D changes PVT-TAB-1's columns, PVT-TAB-3, PVT-TAB-6 and PVT-MIX-6 for tables. C adds a PVT model and replaces PVT-OIL-13's transfer.
- **The backends.** The Rust core checks only shapes ("Validation stays in Python", `table.rs`, header), so A, A′ and B touch Python alone. C and D change both.
- **CasADi smoothness.** A, A′ and B add nothing. The liquid density, a ratio of two bilinear interpolants with positive weights, stays between its nodes' densities, and the dew point is a node. C adds smooth clamps at two saturation limits per point. D interpolates $\rho_L$ directly.
- **`FluidModel`.** A, A′ and B need no change. D changes `liquid_density` and `liquid_fvf`. C changes `phase_rates` and the gas law.
- **The table pipeline.** A and A′ clip and fill in `from_phase_split`. B adds a header key. C needs two more flashes per node and a separation path. D writes $X$ and $\rho_L$ as the flash gives them.

## Recommendation

**A′.** The reasons:

1. **The established practice does not check a net transfer.** OPM orders the quantities that physics orders: its live-oil key $R_s$, and $p$, $B_o$ and $R_v$ within each undersaturated record, where one phase is compressed. It leaves the saturated $R_v$ free. OLGA's tables carry a gas mass fraction with no order check found. $R_{so}$ is a net transfer, and no order holds for it once the gas carries oil.
2. **The trace mirrors the industry convention.** OLGA's table generators fill an absent phase from the nearest node. The only extra in ManyWells is $X = 1 - \delta$ in place of 1, which its $B_o$ column needs. At $\delta = 10^{-4}$, the no-slip density moves by less than $10^{-4}$ of itself.
3. **It keeps PVT-TAB-6 exact at the nodes.** The modified black oil is not exact there (evidence (a)). The GOR axis already transfers across GOR better than $R_v$ would (evidence (c)).
4. **It changes neither backend, nor the rows, nor their smoothness.**

Add **B** if Bjarne wants the old check kept for oil tables. It is compatible with A′. Keep **D** in reserve, in case R6 finds that $B_o$ near $5\cdot10^{-4}$ or the trace liquid troubles either solver, or in case OLGA tables are to be read directly. The trace's alternative there is a floor on the liquid mass rate, and the drift-flux rows at $\alpha \to 1$ have not been tested without one (an assumption: no measurement here). Drop **C**: it costs a feature, and on this evidence it brings less accuracy than ManyWells' present form.

## Resource list (paywalled or proprietary; not consulted)

| Source | Bears on |
|---|---|
| Whitson, C.H. and Torp, S.B. (1983), "Evaluating constant-volume depletion data", *Journal of Petroleum Technology* 35(3), 610–620 | C: how $R_s$ and $R_v$ are made, and on which depletion path |
| Coats, K.H. (1985), "Simulation of gas condensate reservoir performance", *Journal of Petroleum Technology* 37(10), 1870–1886 | C: the modified black oil with constant surface gravities, against a compositional model |
| Fevang, Ø., Singh, K. and Whitson, C.H. (2000), "Guidelines for choosing compositional and black-oil models for volatile oil and gas-condensate reservoirs", SPE 63087 | C against A′: when black-oil tables at one composition suffice |
| Kniazeff, V.J. and Naville, S.A. (1965), "Two-phase flow of volatile hydrocarbons", *SPE Journal* 5(1), 37–44 | D: a mass-based table form, in its original |
| OLGA user manual, the PVT table file and its treatment of a phase with zero mass fraction | D and A′: what OLGA does where $X = 1$ |
| PVTsim Nova manual, the OLGA and LedaFlow table interface | A′: the fill convention for an absent phase, from its maker |
| Prosper user manual (Petex), the retrograde-condensate black oil | C: how a nodal-analysis tool treats condensates with black oil |
| Eclipse Reference Manual, PVTO and PVTG | the checks: OPM's source stands in for it here |

## References

- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall: §1.2, p. 12, Eqs. 1.10–1.13.
- Lie, K.-A. (2019), *An introduction to reservoir simulation using MATLAB/GNU Octave*, Cambridge University Press (open access, CC BY-NC-ND 4.0): Ch. 11, §11.4.3 and §11.5, p. 382.
- Pedersen, K.S., Christensen, P.L. and Shaikh, J.A. (2015), *Phase behavior of petroleum reservoir fluids*, 2nd ed., CRC Press: §9.1, p. 197; Tables 3.8 and 3.9 (the condensate of the evidence).
- Rasmussen, A.F. et al. (2021), "The Open Porous Media Flow reservoir simulator", *Computers & Mathematics with Applications* 81, 159–185 (arXiv:1910.06059): §2.1.1, Eqs. 1–2; §2.1.2, Eq. 14.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20: Ch. 7, §7.2, §7.3 (Eqs. 7.2–7.8, 7.12, 7.17–7.20; Figs. 7.6–7.11), §7.4 (Eq. 7.37), §7.5 (Eqs. 7.48–7.50).
- OPM, opm-common, commit `7364486` (github.com/OPM/opm-common): `opm/input/eclipse/EclipseState/Tables/Tables.cpp`, `TableColumn.cpp`, `TableManager.cpp`; `opm/material/fluidsystems/blackoilpvt/WetGasPvt.cpp`, `WetGasPvt.hpp`.
- MRST, commit `fa4ff98` (github.com/SINTEF-AppliedCompSci/MRST): `autodiff/ad-props/props/assignPVTG.m`, `assignPVTO.m`; `autodiff/ad-core/statefunctions/flowprops/BlackOilDensity.m`.
- NeqSim, an open-source Java library, repository `neqsim`, commit `89a28c6`: `src/main/java/neqsim/blackoil/BlackOilConverter.java`, `blackoil/io/EclipseEOSExporter.java`, `pvtsimulation/util/BlackOilTableValidator.java`, `thermodynamicoperations/propertygenerator/OlgaTableGridFiller.java`, `OLGApropertyTableGeneratorKeywordFormat.java`, `OLGApropertyTableGeneratorWater.java`.
- pyfas, commit `d7432ec` (github.com/gpagliuca/pyfas): `pyfas/tab.py`; `pyfas/test/test_files/3P_single-fluid_key.tab` and `3P_single-fluid_fixed.tab`.
- Bell, C., `fluids`, commit `652a013`, and `thermo`, commit `2bb466e` (github.com/CalebBell): `fluids/two_phase.py`, `thermo/flash/flash_vl.py`.
- Marlim3 documentation (marlim3.readthedocs.io): Single-branch model reference, Fluids; Developer guide, Fluid thermodynamics.
- whitson+ manual (manual.whitson.com), BOT module: Export formats; PVT table generation, advanced topics.
- SLB (2018), *PIPESIM 2017 steady-state multiphase flow simulator: technical reference*, p. 7.
- LedaFlow, "Why LedaFlow" (ledaflow.com), Thermodynamics.
- Petex, Prosper product page (petex.com).
- Evidence: `plans/evidence/condensate_rv_form.py` (`plans/evidence/README.md`, Sampler redesign: the model's gaps), on the proof tables of `plans/evidence/table_fluids.py`.
