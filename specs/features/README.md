# Feature specs

One file per model feature, `NNN-<name>.md` (constitution, principle 4). A new feature's spec is approved by Bjarne before the implementation starts; it states the motivation, the delta to `specs/model/` with the new equation IDs, how the option is switched off in the `v1.0.0` configuration, the acceptance checks, and what is out of scope. `specs/model/` always shows the current model; the feature specs are the change record.

Specs 001 to 013 were written after the fact in Step 7 of `plans/manywells-v2-plan.md`, for the changes `develop` made after v1.0.0 (`plans/develop_model_changes.md`, numbered as there). They are drafts for Bjarne's sign-off. Spec 014 is Step 8's pilot, the Rust core, a backend rather than a model feature; spec 015 is Step 9's port of the rest of `develop`'s model to it.

| Spec | Feature | Spec files |
|---|---|---|
| [001](001-deviated-wells.md) | Deviated and L-shaped wells | geometry, balances, discretization, thermal |
| [002](002-slip-inclination.md) | Inclination in the slip model | slip |
| [003](003-surface-tension.md) | Surface tension from the fluid model | pvt/mixture, pvt/oil |
| [004](004-friction.md) | Friction from roughness and viscosity | friction, pvt/* |
| [005](005-fluid-model.md) | Unified fluid model | pvt/*, nomenclature |
| [006](006-real-gas.md) | Real gas | pvt/gas |
| [007](007-black-oil.md) | Black oil | pvt/oil, pvt/mixture |
| [008](008-dissolved-gas.md) | Gas dissolving into oil | balances, discretization, pvt/oil |
| [009](009-energy-balance.md) | Frictional heating and a gravity term | balances, thermal, discretization |
| [010](010-lift-gas-temperature.md) | Lift-gas temperature | thermal |
| [011](011-fixed-liquid-rate.md) | Inflow returns the liquid rate; fixed liquid rate | inflow |
| [012](012-root-search.md) | Root set, stability label and operating point in the simulator | solution (informative) |
| [013](013-components-and-api.md) | Components, frozen inputs, configurations and the API | README, architecture |
| [014](014-rust-solver.md) | The Rust core in the v1.0.0 configuration (Step 8) | none (architecture) |
| [015](015-rust-develop-model.md) | `develop`'s model in the Rust core (Step 9) | none (architecture) |
