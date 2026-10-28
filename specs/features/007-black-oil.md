# 007 · Black oil

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 7 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

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
- **Dissolved gas in the density (open).** Where $R_{so}(p, T)$ exceeds the gas–oil ratio, PVT-MIX-6 still takes $R_{so}$ and $B_o$ at $(p, T)$, so it counts more dissolved gas and swelling than PVT-OIL-13 dissolves. This happens at the bottomhole in 117 of the 1456 sampled cases above. Evaluating both at the gas actually dissolved would fix it, and would change only $R_{so}$ and $B_o$ in PVT-MIX-6's formula.
