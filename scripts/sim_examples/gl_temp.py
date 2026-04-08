"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Gas lift study: step up gas lift rate from a low to a medium level and
analyse the steady-state production response.

Each operating point is a separate steady-state simulation with the choke
valve position and reservoir/separator boundary conditions held fixed.

The user can change the lift gas temperature to see how it affects the wellhead temperature.
"""

import numpy as np
import matplotlib.pyplot as plt

from manywells.simulator import SSDFSimulator, WellProperties, BoundaryConditions, SimError
from manywells.pvt.fluid import FluidModel
from manywells.inflow import ProductivityIndex

from scripts.plt_config import set_seaborn_config

set_seaborn_config()

# -- Well configuration -----------------------------------------------
fluid = FluidModel(wlr=0.5)
wp = WellProperties(fluid=fluid, inflow=ProductivityIndex(k_l=1.0))

# -- Fixed boundary conditions ----------------------------------------
U_CHOKE = 0.8
P_R = 120.0    # Reservoir pressure (bar) – depleted reservoir
P_S = 20.0     # Separator pressure (bar)
T_R = 373.15   # Reservoir temperature (K) – 100 °C
T_S = 277.15   # Ambient / sea-floor temperature (K) – 4 °C
# T_LG = 373.15  # Gas lift temperature at injection point (K) – 100 °C
T_LG = 308.15  # Gas lift temperature at injection point (K) – 35 °C

# -- Gas lift schedule: low → medium ----------------------------------
w_lg_values = np.linspace(1.0, 3.0, 8)

# -- Run simulations --------------------------------------------------
w_lg_list, w_oil_list, w_gas_list, w_water_list, T_wh_list = [], [], [], [], []

prev_solution = None
for w_lg in w_lg_values:
    bc = BoundaryConditions(p_r=P_R, p_s=P_S, T_r=T_R, T_s=T_S, u=U_CHOKE, w_lg=w_lg, T_lg=T_LG)
    sim = SSDFSimulator(wp, bc)

    if prev_solution is not None:
        sim.x_guess = prev_solution

    try:
        x = sim.simulate()
    except SimError as e:
        print(f"Simulation failed at w_lg = {w_lg:.2f} kg/s: {e}")
        continue

    prev_solution = x
    df = sim.solution_as_df(x)

    A = wp.geometry.A

    # Wellhead = last row (top of well)
    wh = df.iloc[-1]
    w_gas_wh = A * wh["alpha"] * wh["rho_g"] * wh["v_g"]

    # Bottom-hole pressure → reservoir inflow
    p_bh = float(df["p"].iloc[0])
    w_l_inflow = float(wp.inflow.liquid_mass_flow_rate(p_bh, bc.p_r))
    w_oil = fluid.f_o_in_liquid * w_l_inflow
    w_water = (1 - fluid.f_o_in_liquid) * w_l_inflow

    w_lg_list.append(w_lg)
    w_oil_list.append(w_oil)
    w_gas_list.append(w_gas_wh)
    w_water_list.append(w_water)
    T_wh_list.append(wh["T"] - 273.15)

w_lg_arr = np.array(w_lg_list)
w_oil_arr = np.array(w_oil_list)
w_gas_arr = np.array(w_gas_list)
w_water_arr = np.array(w_water_list)
T_wh_arr = np.array(T_wh_list)

# -- Plot results -----------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

step = np.arange(len(w_lg_arr))

axes[0].step(step, w_lg_arr, where="mid", color="k", linewidth=1.5)
axes[0].set_ylabel("Gas lift rate\n(kg/s)")

axes[1].plot(step, w_oil_arr, "o-", color="green", markersize=4)
axes[1].set_ylabel("Oil rate\n(kg/s)")

axes[2].plot(step, w_gas_arr, "o-", color="red", markersize=4)
axes[2].set_ylabel("Gas rate\n(kg/s)")

axes[3].plot(step, w_water_arr, "o-", color="royalblue", markersize=4)
axes[3].set_ylabel("Water rate\n(kg/s)")

axes[4].plot(step, T_wh_arr, "o-", color="darkorange", markersize=4)
axes[4].set_ylabel("Wellhead\ntemperature (°C)")
axes[4].set_xlabel("Operating point")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Gas lift study  (choke {U_CHOKE*100:.0f}%,  "
    f"$p_r$={P_R:.0f} bar,  $p_s$={P_S:.0f} bar,  $T_{{lg}}$={T_LG - 273.15:.0f} °C)",
    fontsize=16,
)
fig.tight_layout()
plt.savefig("gl_study.png", dpi=150, bbox_inches="tight")
plt.show()
