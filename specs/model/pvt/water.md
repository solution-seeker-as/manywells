# Water

## Purpose

Water properties. In `v1.0.0`, water enters only through the liquid mixing of `pvt/mixture.md`, which the sampler applies.

## Interface

Constants only, in `v1.0.0`.

## Equations

### PVT-WAT-1 · Incompressible water

Water is incompressible, with density $\rho_w = 999.1$ kg/m³ and heat capacity $c_{pw} = 4184$ J/(kg K). The same density is the reference for specific gravity (PVT-OIL-2). Used by `v1.0.0`.

## Options

| Option | IDs | Used by |
|---|---|---|
| Incompressible water | PVT-WAT-1 | `v1.0.0`, `develop` |

`develop` adds a water formation volume factor with constant compressibility, which is not used by the simulator and has the wrong sign (`plans/improvements.md` §1.2); Step 7 fixes it before specifying it. `develop` also adds a water viscosity correlation for friction (Step 7).

## Safeguards

None.

## Sources

Paper §4.2.

## Test vectors

None: constants only. The mixing vectors in `pvt/mixture.md` use these values.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| PVT-WAT-1 | §4.2 | `pvt.py` `WATER` | spec-only: constants, which enter through PVT-MIX-2 to PVT-MIX-4 and PVT-OIL-2, which have vectors |
