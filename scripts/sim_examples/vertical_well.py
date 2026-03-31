"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Simulate a vertical well.
"""

import pandas as pd
import matplotlib.pyplot as plt
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator


###########################################################################
# Create a new well (using default values)
###########################################################################
well_properties = WellProperties()
boundary_conditions = BoundaryConditions(u=0.5)
sim = SSDFSimulator(well_properties, boundary_conditions)

###########################################################################
# Simulate
###########################################################################
x = sim.simulate()

# Convert solution to DataFrame
df = sim.solution_as_df(x)

###########################################################################
# Plot results
###########################################################################

# Plot solution from cell-wise simulation
A = well_properties.geometry.A
df['w_g'] = A * df['alpha'] * df['rho_g'] * df['v_g']  # Gas mass flow rates
df['w_l'] = A * (1 - df['alpha']) * df['rho_l'] * df['v_l']  # Liquid mass flow rates
pd.set_option('display.max_columns', None)
print(df)

df.plot()
plt.xlabel('Cell index')
plt.show()
