# 007 · Black oil

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 7 of `plans/develop_model_changes.md`. Status: signed off by Bjarne on 2026-10-29.*

## Motivation

v1.0.0's liquid is incompressible and dissolves no gas (paper §8, limitation 1). Oil at reservoir pressure holds dissolved gas, which comes out of solution as the pressure falls up the well, and the oil swells with it.

## Delta

- `pvt/oil.md`: PVT-OIL-4 (black oil), PVT-OIL-5 (separator gas-gravity correction), PVT-OIL-6 ($R_{so}$, Vazquez–Beggs), PVT-OIL-7 (bubble-point cap), PVT-OIL-8 ($B_o$, Vazquez–Beggs), PVT-OIL-9 (live-oil density), PVT-OIL-14 (Standing's bubble point; not used by the simulator).
- `pvt/mixture.md`: PVT-MIX-6, the liquid density with black oil, mixed with water by in-situ volume.
- Code: `pvt.black_oil.BlackOilPVT`, `FluidModel(oil_model='black_oil')`.

## Off in the `v1.0.0` configuration

`FluidModel(oil_model='dead_oil')`: $R_{so} = 0$ and $B_o = 1$, so the liquid density is the constant of PVT-MIX-1, exactly.

## Acceptance

The PVT-MIX-1 row vectors in the `v1.0.0` configuration; the develop vectors of PVT-OIL-5 to PVT-OIL-9 and PVT-MIX-6; `tests/test_black_oil.py` and `tests/test_black_oil_consistency.py` (monotonicity, the bubble-point cap, consistency of $R_{so}$, $B_o$ and $B_g$); `tests/test_fluid.py` (the liquid density against the in-situ volume rate of PVT-OIL-13's mass rates).

## Rulings and open items

- **Separator correction (PVT-OIL-5).** The code took the natural log of $p_\text{sep}/114.7$; Vazquez and Beggs (1980) have $\log_{10}$. At standard separator conditions the code's corrected gas gravity was $0.75\gamma_g$ instead of $0.89\gamma_g$, so $R_{so}$ was 16% lower than the source's. Found in Step 7; Bjarne ruled on 2026-10-01 to follow the source, and the code now has $\log_{10}$, checked against hand values in `tests/test_black_oil.py`. A change of `develop`'s default model, not of the `v1.0.0` configuration.
- **Range (open).** API gravity is checked against 10 to 40; pressures and temperatures are not. Wells with a small oil fraction get gas–oil ratios above $10^4$ Sm³/Sm³ in the sampler (SMP-43).
- **Mixing by in-situ volume (PVT-MIX-6, PVT-MIX-8; issue #8).**
  - *Finding.* PVT-MIX-6 mixed the live oil and the water by the water–liquid ratio at standard conditions. Downhole the oil swells and the water does not, so that gave the water too much weight. It contradicted PVT-OIL-13's mass rates and the no-slip liquid: the density was not the liquid's mass over its in-situ volume.
  - *Ruling.* Bjarne ruled on 2026-10-28 to mix by in-situ volume. The density is the mass of one Sm³ of stock-tank liquid with its dissolved gas, over its volume $B_l$ at $(p, T)$. The viscosity is weighted by the same in-situ water fraction, so the two agree on how much water there is.
  - *Effect.* On 1493 cases of `develop`'s sampler, solved by the Rust core, the bottomhole liquid density falls by 1.0% at the median and by up to 6.4%. PBH changes by less than 0.3 bar in 80% of the cases and by up to 5.1 bar. An operating point is found in 1458 cases instead of 1457. The one case lost is a miss of the core's search; the CasADi backend finds that root (`plans/evidence/README.md`).
  - *Scope.* This changes `develop`'s default model, not the `v1.0.0` configuration: with dead oil, $B_l = 1$.
  - *Tests.* Two tests pinned values of the old model. The Joule–Thomson gas wells of `tests/test_rust_backend.py` were re-solved: their $p_0$ moved by 0.002 to 0.008 bar, and both backends agree. The two-root well of `tests/test_model_properties.py` lost its trickle root, because its lighter in-situ column can now reach the separator. That root had been 0.13 bar below $p_r$. Its separator pressure was raised from 25 to 29 bar, which puts the trickle root 2.5 bar below $p_r$.
  - *Not rerun.* The twin-study measurements of `017-calibration.md` were made with the old density.
- **Water's formation volume factor (open).** PVT-MIX-6 takes $B_w = 1$ (PVT-WAT-1). Downhole, heating expands water more than pressure compresses it: $B_w$ is about 1.03 to 1.04 at 380 K and 250 bar, so the water is itself 3–4% too dense. PVT-WAT-2 depends only on pressure and goes the wrong way. A temperature-dependent $B_w$ would be a separate option.
- **Dissolved gas in the density (issue #16).**
  - *Finding.* Where $R_{so}(p, T)$ exceeds the gas–oil ratio, the oil is undersaturated: all the reservoir gas is dissolved. PVT-MIX-6 still took $R_{so}$ and $B_o$ at $(p, T)$, and so did the viscosity and the surface tension, so they counted more dissolved gas and swelling than PVT-OIL-13 dissolves. The liquid was too light: 674.8 against 716.3 kg/m³ at 250 bar and 360 K with $\rho_o$ = 850 kg/m³ and a gas–oil ratio of 50 Sm³/Sm³.
  - *Ruling.* Bjarne ruled on 2026-10-29 to cap the black oil's $R_{so}$ at the stream's gas–oil ratio, PVT-OIL-15, so that $B_o$, the density, the viscosity, the surface tension and the mass rates all see the gas dissolved. The cap is the stream's bubble point; it is active wherever the oil is undersaturated, which in realistic oil wells is common: an API 35 oil at 100 Sm³/Sm³ has its bubble point at 235 to 267 bar. With it, PVT-OIL-13's own cap became a smooth min of two nearly equal mass rates, which left $5\cdot10^{-4}$ kg/s of reservoir gas undissolved whatever the rate; Bjarne ruled the same day to take that min in scf/STB, as PVT-OIL-7 and PVT-OIL-15 do.
  - *Effect* (`plans/evidence/gor_cap_wells.py`). On 1492 cases of `develop`'s sampler, solved by the Rust core, 109 of those with an operating point before and after are undersaturated at the bottomhole. There the bottomhole liquid density rises by 2.4% at the median and by up to 14%, PBH by 0.33 bar at the median and by up to 10.8 bar, and the reservoir liquid rate falls by 1.3% at the median and by up to 23%. In the other 1335 cases nothing changes by more than $10^{-3}$ bar. An operating point is found in 1444 cases instead of 1448. The four lost cases (wells 1 and 442) are the model's: low-GOR wells (16 to 30 Sm³/Sm³, 18 to 48% water) whose $R_{so}$ at $p_r$ is 3 to 10 times their GOR, and whose column was light enough to reach the separator only with the uncounted gas. After the change the core's $R(p_0)$ is positive on 4000 samples from $p_s$ to $p_r$, and neither backend finds a root. The CasADi backend, on 263 of the cases, loses the same one in its share and no other; the two backends agree on which cases have an operating point before and after, and their operating points agree within $1.7\cdot10^{-7}$ bar.
  - *Scope.* This changes `develop`'s default model, not the `v1.0.0` configuration, whose dead oil dissolves no gas. A PVT table is not capped (PVT-TAB-3): its $R_{so}$ and $B_o$ go together, and `tests/data/pvt/vazquez_beggs.csv`, a table of the black oil's correlations, keeps $R_{so}$ above its GOR below 360 K. The undersaturated oil's compressibility, which would lower $B_o$ above the bubble point, is not modelled.
  - *Tests.* `tests/test_fluid.py` checks the cap, the density, viscosity and surface tension at it, and that undersaturated oil dissolves all the reservoir gas; the vectors of PVT-OIL-6 to PVT-OIL-8 pin the correlations without the cap, and PVT-OIL-15 has its own. `tests/test_pvt_table.py` tabulates a sampled well's fluid at its GOR, not its correlations, when it compares the table's operating point with the black oil's. The Rust core's test `every_root_zeroes_every_row` allows the choke row 16 ulp of $p_r$ times the slope, not 8: at `w2_table_one_temperature`'s trickle root, 0.015 bar below $p_r$, the row was 0.73 times the old allowance before the change and 1.19 times it after, at a $p_0$ one ulp away, from rounding in its table.
  - *Sign-off.* Bjarne signed off on 2026-10-29 on what the ruling left open (#29): the regenerated `develop` vectors, the two changed tests of `tests/test_pvt_table.py` and the 16 ulp of `every_root_zeroes_every_row`.
