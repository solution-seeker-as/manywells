# Thermal Energy Modeling

The simulator's energy equation governs the fluid temperature profile along the
wellbore. Three physical mechanisms are modeled:

```
cp_flux * dT/dz = -4h(T - T_a)/D  +  (1-α)*v_l*F  -  g*(mass_flux - liq_flux*ρ_m)
                   ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   heat transfer     liquid friction  gravitational cooling
                                    dissipation
```

where `cp_flux = cp_g*α*ρ_g*v_g + cp_l*(1-α)*ρ_l*v_l`.

The equation follows from the steady-state enthalpy balance with ideal-gas
(`h_g = cp_g T`) and incompressible-liquid (`h_l = cp_l T + p/ρ_l`) equations
of state, neglecting kinetic energy changes. Acceleration terms in the momentum
equation are also neglected when substituting `dp/dz` into the energy balance;
this is consistent with the simulator's low-fidelity, steady-state design.


## 1. Heat transfer to surroundings

Heat loss from the fluid to the formation through the wellbore wall:

```
dT_heat = Δz * 4h(T - T_a) / (D * cp_flux)
```

The ambient temperature T_a follows a linear geothermal profile from the reservoir
temperature T_r (at z=0) to the surface temperature T_s (at z=L).

### References

* Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J.P. (2006).
  "Unified Model of Heat Transfer in Gas/Liquid Pipe Flow."
  *SPE Production & Operations*, 21(1), 114–122.
  Eq. (13) gives the temperature gradient for bubbly/dispersed-bubble flow;
  Eq. (26) gives the corresponding expression for stratified/annular flow.
  Both have the form `dT/dl = -4U(T - T_O) / (d * cp_flux)`.

* Hasan, A.R. and Kabir, C.S. (2012).
  "Wellbore Heat-Transfer Modeling and Applications."
  *Journal of Petroleum Science and Engineering*, 86–87, 127–136.
  Eq. (7) presents the full single-conduit energy balance; the `∓Q/w` term
  represents the heat exchange with the surroundings.


## 2. Frictional dissipation heating

Viscous friction converts mechanical energy to heat. For an incompressible
liquid the frictional pressure drop directly heats the fluid; for an ideal gas
it does not (the enthalpy of an ideal gas is pressure-independent).

```
F       = (f_D / (2D)) * ρ_m * v_m²
dT_fric = Δz * (1 - α) * v_l * F / cp_flux
```

The term arises when the liquid enthalpy `h_l = cp_l T + p/ρ_l` is expanded
in the energy balance. The frictional pressure gradient `F` enters through the
`(1/ρ_l) dp/dz` contribution to `dh_l/dz`, weighted by the liquid volumetric
flux `(1-α) v_l`.

### Limiting cases

* **Pure liquid (α = 0):** `dT_fric/dz = (f_D/(2D)) * v_l² / cp_l`
  — standard viscous dissipation heating.
* **Pure gas (α = 1):** the term vanishes — correct for an ideal gas whose
  enthalpy is independent of pressure.

### References

* Hasan, A.R. and Kabir, C.S. (2012).
  "Wellbore Heat-Transfer Modeling and Applications."
  *Journal of Petroleum Science and Engineering*, 86–87, 127–136.
  Eq. (7): the `C_J dp/dz` (Joule–Thomson) term captures the thermodynamic coupling between pressure changes and temperature for single-phase flow. For an incompressible liquid `C_J = -1/(ρ_l cp_l)`, so this term reduces to friction heating.

* Hasan, A.R. and Kabir, C.S. (2002).
  *Fluid Flow and Heat Transfer in Wellbores*.
  Society of Petroleum Engineers.
  Chapter 2 derives the complete steady-state energy equation for wellbore flow, including viscous dissipation, from the general enthalpy balance.

## 3. Gravitational cooling (adiabatic lapse rate)

As the fluid rises, thermal energy is converted to gravitational potential
energy. For a pure ideal gas this produces the classical adiabatic lapse rate
`-g/cp`. For a pure incompressible liquid, hydrostatic pressure work exactly
compensates the gravitational potential energy change, so the net effect
vanishes.

```
mass_flux = α*ρ_g*v_g + (1-α)*ρ_l*v_l
liq_flux  = (1 - α)*v_l
dT_grav   = Δz_tvd * g * (mass_flux - liq_flux * ρ_m) / cp_flux
```

Note that only the vertical component of gravity contributes (`Δz_tvd = Δz cos(theta)`), whereas the frictional term acts along the measured depth (`Δz`).

### Limiting cases

* **Pure gas (α = 1):** `dT_grav/dz = g / cp_g ≈ 0.0044 K/m` for methane
  (cp_g = 2225 J/(kg K)), giving about 13 K of cooling over 3000 m.
* **Pure liquid (α = 0):** the term vanishes
  (`mass_flux - liq_flux * ρ_m = ρ_l v_l - v_l ρ_l = 0`).

### References

* Ramey, H.J. Jr. (1962).
  "Wellbore Heat Transmission."
  *Journal of Petroleum Technology*, 14(4), 427–435.
  The foundational paper on wellbore thermal modeling.

* Hasan, A.R. and Kabir, C.S. (2012).
  "Wellbore Heat-Transfer Modeling and Applications."
  *Journal of Petroleum Science and Engineering*, 86–87, 127–136.
  Eq. (7): the `g sinα / (J gc)` term represents gravitational work on the fluid. Note that `sinα` is used since `α` is the angle to the horizontal, and the conversion factors J and gc are both 1 when using SI units.


## Derivation sketch

Starting from the steady-state enthalpy balance for the mixture (no mass
transfer between phases, constant mass fluxes `w_g/A` and `w_l/A`):

```
d/dz [w_g/A · cp_g T  +  w_l/A · (cp_l T + p/ρ_l)]  =  -4h(T - T_a)/D  -  (w_g + w_l)/A · g
```

Expanding the left-hand side and substituting `dp/dz ≈ -F - ρ_m g`
(neglecting acceleration):

```
cp_flux · dT/dz  +  (1-α) v_l · (-F - ρ_m g)  =  -4h(T - T_a)/D  -  mass_flux · g
```

Rearranging:

```
cp_flux · dT/dz  =  -4h(T - T_a)/D  +  (1-α) v_l · F  +  g · (liq_flux · ρ_m - mass_flux)
```

The three terms on the right-hand side are heat transfer to surroundings
(§1), frictional dissipation (§2), and gravitational cooling (§3).