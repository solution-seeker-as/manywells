# Thermal Energy Modeling

The simulator's energy equation governs the fluid temperature profile along the
wellbore.  Three physical mechanisms are modeled:

```
cp_flux * dT/dz = -4h(T - T_a)/D  +  (1-α)*v_l*F  +  g*(liq_flux*ρ_m - mass_flux)
                   ^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   heat transfer       liquid friction  gravitational cooling
                                       dissipation
```

where `cp_flux = cp_g*α*ρ_g*v_g + cp_l*(1-α)*ρ_l*v_l`.

## 1. Heat transfer to surroundings

Heat loss from the fluid to the formation through the wellbore wall:

```
dT_heat = Δz * 4h(T - T_a) / (D * cp_flux)
```

The ambient temperature follows a linear geothermal profile from the reservoir
temperature T_r (at z=0) to the surface temperature T_s (at z=L).

**Reference:** Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J.P. (2006).
"Unified Model of Heat Transfer in Gas/Liquid Pipe Flow."
*SPE Production & Operations*, 21(1), 114–122.

## 2. Frictional dissipation heating

Viscous friction converts mechanical energy to heat.  For an incompressible
liquid the frictional pressure drop directly heats the fluid; for an ideal gas
it does not (the Joule-Thomson coefficient is zero, so enthalpy is
pressure-independent).

```
F     = (f_D / (2D)) * ρ_m * v_m²
dT_fric = Δz * (1 - α) * v_l * F / cp_flux
```

### Limiting cases

- **Pure liquid (α = 0):** `dT_fric/dz = (f_D/(2D)) * v_l² / cp_l`
  (viscous dissipation heating).
- **Pure gas (α = 1):** the term vanishes.

### References

- Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J.P. (2006).
  "Unified Model of Heat Transfer in Gas/Liquid Pipe Flow."
  *SPE Production & Operations*, 21(1), 114–122.
  Eq. 5 presents the complete energy equation including the friction term.

- Hasan, A.R. and Kabir, C.S. (2002).
  *Fluid Flow and Heat Transfer in Wellbores*.
  Society of Petroleum Engineers.
  Chapter 2 derives the complete steady-state energy equation for wellbore
  flow, including viscous dissipation.

- Bird, R.B., Stewart, W.E. and Lightfoot, E.N. (2002).
  *Transport Phenomena*, 2nd ed., Wiley.
  Section 11.4 derives the viscous dissipation function from first principles.

## 3. Gravitational cooling (adiabatic lapse rate)

As the fluid rises, thermal energy is converted to gravitational potential
energy.  For a pure ideal gas this produces the classical adiabatic lapse rate
-g/cp.  For a pure incompressible liquid, hydrostatic pressure exactly balances
gravity and the term vanishes.

```
mass_flux = α*ρ_g*v_g + (1-α)*ρ_l*v_l
liq_flux  = (1 - α)*v_l
dT_grav   = Δz * g * (mass_flux - liq_flux * ρ_m) / cp_flux
```

### Limiting cases

- **Pure gas (α = 1):** `dT_grav/dz = g / cp_g ≈ 0.0044 K/m` for methane
  (cp_g = 2225 J/(kg K)), giving about 13 K of cooling over 3000 m.
- **Pure liquid (α = 0):** the term vanishes.

### References

- Ramey, H.J. Jr. (1962).
  "Wellbore Heat Transmission."
  *Journal of Petroleum Technology*, 14(4), 427–435. doi:10.2118/96-PA.
  The foundational paper on wellbore thermal modeling.  Eq. 4 includes the
  gravitational term g sin(θ) / (J cp).

- Hasan, A.R. and Kabir, C.S. (1994).
  "Aspects of Wellbore Heat Transfer During Two-Phase Flow."
  *SPE Production & Facilities*, 9(3), 211–216. doi:10.2118/22948-PA.
  Extends Ramey's analysis to two-phase flow in vertical pipes.

- Hasan, A.R. and Kabir, C.S. (2012).
  "Wellbore Heat-Transfer Modeling and Applications."
  *Journal of Petroleum Science and Engineering*, 86–87, 127–136.
  doi:10.1016/j.petrol.2012.03.021.
  Review paper consolidating the full energy equation for two-phase wellbore
  flow, including the gravitational work term.

## General two-phase energy equation

The combined equation follows from the steady-state enthalpy balance with
ideal gas (h_g = cp_g T) and incompressible liquid (h_l = cp_l T + p/ρ_l)
equations of state, neglecting kinetic energy changes:

- Shoham, O. (2006).
  *Mechanistic Modeling of Gas-Liquid Two-Phase Flow in Pipes*.
  Society of Petroleum Engineers.
  Chapter 3 derives the general energy equation for two-phase pipe flow and
  discusses why friction and gravity terms only affect one phase each.

- Hasan, A.R. and Kabir, C.S. (2002).
  *Fluid Flow and Heat Transfer in Wellbores*.
  Society of Petroleum Engineers.
  The most comprehensive single reference for the full derivation in a
  petroleum engineering context.
