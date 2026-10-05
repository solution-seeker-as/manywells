# Thermal energy modeling

This note derives the temperature equation of ManyWells' model and explains its terms. The equations, with their IDs,
are specified in [`specs/model/thermal.md`](../specs/model/thermal.md) and
[`specs/model/balances.md`](../specs/model/balances.md); where this note and the spec differ, the spec holds.

## The temperature equation

Along the flow path, with $z$ the distance from the bottomhole and $\theta$ the inclination from vertical, the
temperature of the mixture obeys (BAL-13)

$$\frac{dT}{dz} = -H + \Phi_f - \Phi_g - \Phi_{JT},$$

where each term is in K/m and has the heat-capacity flux of the flow,

$$C = c_{pg}\alpha\rho_g v_g + c_{pl}\thinspace(1-\alpha)\rho_l v_l,$$

in its denominator:

| Term | ID | Expression | Effect | Switch |
|---|---|---|---|---|
| Heat loss | THM-1 | $H = \dfrac{4h\thinspace(T - T_a)}{D C}$ | cools the fluid while it is warmer than its surroundings | always on |
| Frictional heating | THM-6 | $\Phi_f = \dfrac{(1-\alpha)v_l F}{C}$ | heats | `frictional_heating` |
| Gravity term | THM-7 | $\Phi_g = \dfrac{g\cos\theta\thinspace\big(\alpha\rho_g v_g + (1-\alpha)\rho_l v_l - (1-\alpha)v_l\rho_m\big)}{C}$ | cools | `gravity_term` |
| Joule–Thomson term | THM-8 | $\Phi_{JT} = \dfrac{\alpha v_g J\thinspace(F + \rho_m g\cos\theta)}{C}$ | cools where $J > 0$, heats where $J < 0$ | `joule_thomson` |

$h$ is the overall heat transfer coefficient, $D$ the inner diameter, $T_a$ the ambient temperature, $F$ the
viscous pressure gradient (FRIC-1), $\rho_m$ the mixture density and $J = T(\partial \ln Z/\partial T)_ p$ the gas's
Joule–Thomson factor (PVT-GAS-10).

The switches are fields of `ThermalModel` (`src/manywells/thermal.py`). The `v1.0.0` configuration has heat loss
only (BAL-5), with an ambient temperature linear in $z$ (THM-2). `develop` has all four terms by default, with an
ambient temperature linear in true vertical depth (THM-4). $\Phi_{JT}$ is zero for an ideal gas, so it matters only
with a real gas (PVT-GAS-3 or PVT-GAS-11).

The temperature at the bottomhole is the reservoir temperature $T_r$ (THM-3), or, with `lift_gas_mixing`, the
mixture of the reservoir fluid and the lift gas weighted by their heat-capacity rates (THM-5). The discretized row
evaluates the gradient at the upper point of each cell (implicit Euler, DISC-10).

## Derivation

Write the phases' mass fluxes as $\dot m_g = \alpha\rho_g v_g$ and $\dot m_l = (1-\alpha)\rho_l v_l$ (kg/(m² s)),
so that $C = c_{pg}\dot m_g + c_{pl}\dot m_l$. In this section the pressure $p$ is in Pa; the code works in bar and
converts with $c_\text{bar}$.

**Energy balance.** At steady state, with no kinetic energy and no mass transfer between the phases, so that
$\dot m_g$ and $\dot m_l$ are constant, the total energy of the mixture changes by the heat lost through the wall
and the work done against gravity:

$$\frac{d}{dz}\big(\dot m_g h_g + \dot m_l h_l\big) = -\frac{4h\thinspace(T - T_a)}{D} - (\dot m_g + \dot m_l)g\cos\theta,$$

with $h_g$ and $h_l$ the phases' specific enthalpies. (The subscripted $h_g$ and $h_l$ are enthalpies; $h$ alone is
the heat transfer coefficient.) The factor $4/D$ is the pipe's perimeter over its cross-section.

**Enthalpies.** Both phases have constant heat capacities. The liquid is incompressible, and the gas obeys
$1/\rho_g = Z R_s T/p$:

$$dh_l = c_{pl}\thinspace dT + \frac{dp}{\rho_l}, \qquad dh_g = c_{pg}\thinspace dT + \left[\frac{1}{\rho_g} - T\left(\frac{\partial (1/\rho_g)}{\partial T}\right)_p\right]dp = c_{pg}\thinspace dT - \frac{J}{\rho_g}\thinspace dp.$$

The bracket is $-(R_s T^2/p)(\partial Z/\partial T)_ p = -J/\rho_g$, so the gas's Joule–Thomson coefficient is
$\mu_{JT} = J/(\rho_g c_{pg})$. For an ideal gas $Z = 1$ and $J = 0$, so its enthalpy does not depend on pressure.
The liquid's enthalpy does, through $p/\rho_l$: an incompressible liquid heats as it expands.

**Substitution.** With $\dot m_l/\rho_l = (1-\alpha)v_l$ and $\dot m_g J/\rho_g = \alpha v_g J$, the balance
becomes

$$C\frac{dT}{dz} + \big[(1-\alpha)v_l - \alpha v_g J\big]\frac{dp}{dz} = -\frac{4h\thinspace(T - T_a)}{D} - (\dot m_g + \dot m_l)g\cos\theta.$$

The momentum balance (BAL-11) without its acceleration term gives the pressure gradient,
$dp/dz = -(F + \rho_m g\cos\theta)$. Substituting it and dividing by $C$:

$$\frac{dT}{dz} = -\underbrace{\frac{4h\thinspace(T - T_a)}{D C}}_{H} + \underbrace{\frac{(1-\alpha)v_l F}{C}}_{\Phi_f} - \underbrace{\frac{g\cos\theta\thinspace\big(\dot m_g + \dot m_l - (1-\alpha)v_l\rho_m\big)}{C}}_{\Phi_g} - \underbrace{\frac{\alpha v_g J\thinspace(F + \rho_m g\cos\theta)}{C}}_{\Phi_{JT}}.$$

## The terms

**Heat loss ($H$).** The form of Zhang et al. (2006): heat flows to the formation in proportion to the temperature
difference, with one overall coefficient $h$ for the whole well. The ambient temperature falls linearly from $T_r$ at
the bottomhole to $T_s$ at the surface, in $z$ (THM-2) or in true vertical depth (THM-4), which is the same in a
vertical well.

**Frictional heating ($\Phi_f$).** Friction lowers the pressure, and the liquid's enthalpy carries the lost pressure
as heat, through its $p/\rho_l$ term, weighted by the liquid's volumetric flux $(1-\alpha)v_l$. For pure liquid it is
$\Phi_f = F/(\rho_l c_{pl})$, viscous dissipation. The gas's share is in $\Phi_{JT}$, and it is zero for an ideal gas.

**Gravity term ($\Phi_g$).** Lifting the flow converts thermal energy into potential energy. The liquid's share is
paid by the hydrostatic pressure drop through its $p/\rho_l$ term, so only the gas's share is left; with
$\rho_m = \alpha\rho_g + (1-\alpha)\rho_l$,

$$\Phi_g = \frac{\alpha g\cos\theta\thinspace\big(\rho_g v_g + (1-\alpha)v_l\thinspace(\rho_l - \rho_g)\big)}{C} \ge 0.$$

For pure liquid it vanishes. For pure gas it is the adiabatic lapse rate $g\cos\theta/c_{pg}$: about 0.0044 K/m with
$c_{pg} = 2225$ J/(kg K), or 13 K over 3000 m of vertical depth.

**Joule–Thomson term ($\Phi_{JT}$).** A real gas's enthalpy depends on pressure, so the gas cools as it expands where
$Z$ rises with $T$ ($J > 0$), and heats where $Z$ falls with $T$, above the inversion pressure. In the range the
sampler draws, $J > 0$ below about 300 bar. The pressure gradient $F + \rho_m g\cos\theta$ is positive at every
admissible state, so $\Phi_{JT}$ has the sign of $J$. It vanishes for an ideal gas and for pure liquid, and for pure
gas it is $\mu_{JT}(F + \rho_g g\cos\theta)$, the gas's Joule–Thomson cooling along its pressure drop. $J$ comes from
the Dranchuk–Abou-Kassem equation of state in closed form at the state's gas density (PVT-GAS-10).

With this term the fluid can become colder than its surroundings in gas-rich wells, and heat then flows in from the
formation. Near a gas well's choked wellhead it can give a cell's energy row two roots in the temperature; only the
one where the row rises in the temperature is a root of the model (SOL-9).

## Assumptions and limitations

- **One temperature.** The phases are in thermal equilibrium at every point.
- **No kinetic energy** in the energy balance.
- **No acceleration in the substituted pressure gradient.** $\Phi_f$, $\Phi_g$ and $\Phi_{JT}$ use
  $F + \rho_m g\cos\theta$, while the momentum row (BAL-11) keeps the acceleration. Using the cell's actual pressure
  gradient in $\Phi_{JT}$ instead changes the wellhead temperature by at most 0.38 K (feature spec 016, design
  choice 1).
- **Constant heat capacities.** $c_{pg}$ and $c_{pl}$ do not depend on pressure or temperature, although a real gas's
  heat capacity rises with pressure.
- **Incompressible liquid.** The liquid has no thermal expansion, so its Joule–Thomson coefficient is
  $-1/(\rho_l c_{pl})$, which gives $\Phi_f$ and the liquid's share of $\Phi_g$.
- **Constant mass fluxes in the derivation.** With dissolved gas (BAL-10) the phases' mass fluxes change along the
  well. The equation uses the local fluxes and leaves out the enthalpy the gas carries as it leaves solution, the
  heat of solution.
- **Simple heat transfer.** One overall coefficient $h$ for the whole well and a linear ambient profile, with no
  transient conduction into the formation as in Ramey (1962).
- **The gas law's range.** Nothing checks that a state is in the Dranchuk–Abou-Kassem equation's range. Below a
  pseudo-reduced temperature of 1.05, near the critical point, $J$ can have a pole (`specs/model/pvt/gas.md`,
  Safeguards).
- **Nothing downstream of the wellhead.** The Joule–Thomson cooling across the choke is outside the model.

## References

- Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J.P. (2006). "Unified model of heat transfer in gas/liquid pipe flow."
  *SPE Production & Operations* 21(1), 114–122. Eqs. (13) and (26) give the temperature gradient for
  bubbly/dispersed-bubble and for stratified/annular flow, both of the form $dT/dl = -4U(T - T_O)/(d\thinspace C)$: THM-1.
- Ramey, H.J. Jr. (1962). "Wellbore heat transmission." *Journal of Petroleum Technology* 14(4), 427–435. The
  foundational paper on wellbore temperatures.
- Hasan, A.R. and Kabir, C.S. (2002). *Fluid Flow and Heat Transfer in Wellbores*. Society of Petroleum Engineers.
  Chapter 2 derives the steady-state energy equation of wellbore flow, with viscous dissipation, from the enthalpy
  balance.
- Hasan, A.R. and Kabir, C.S. (2012). "Wellbore heat-transfer modeling and applications." *Journal of Petroleum
  Science and Engineering* 86–87, 127–136. Eq. (7) is the single-conduit energy balance: its $\mp Q/w$ term is the
  heat exchange (THM-1); its $C_J\thinspace dp/dz$ term the Joule–Thomson effect, which for an incompressible liquid,
  $C_J = -1/(\rho_l c_{pl})$, gives the frictional heating (THM-6) and for a real gas THM-8; its
  $g\sin\alpha/(J g_c)$ term the gravitational work (THM-7), where $\alpha$ is the angle from horizontal and
  $J = g_c = 1$ in SI units.
- Hasan, A.R. and Kabir, C.S. (2018). *Fluid Flow and Heat Transfer in Wellbores*, 2nd ed. Society of Petroleum
  Engineers. §6.4.2 gives the Joule–Thomson coefficient of a liquid, a real gas and a two-phase mixture, weighted by
  mass; §6.4.1 advises against neglecting it, "because gas in most wells is rarely ideal".
- Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z factors for natural gases using equations of
  state." *Journal of Canadian Petroleum Technology* 14(3), 34–36. The equation of state behind $Z$ and $J$
  (PVT-GAS-9 to PVT-GAS-11).
