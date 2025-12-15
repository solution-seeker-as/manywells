"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 15 December 2025
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 
"""

import pandas as pd

from manywells.choke import SimpsonChokeModel
from manywells.inflow import Vogel
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator, SimError
from manywells.pvt import GasProperties, LiquidProperties
from scripts.data_generation.well import Well


def load_well(well_id, df_config) -> Well:
    assert well_id in df_config['ID'].tolist(), 'ID not found'

    # Load parameters of well
    w_config = df_config[df_config['ID'] == well_id].to_dict('records')[0]

    # Create objects for well properties and boundary conditions
    if w_config['wp.inflow.class_name'] != 'Vogel':
        raise ValueError('Inflow model class must be "Vogel"')
    inflow_model = Vogel(w_l_max=w_config['wp.inflow.w_l_max'], f_g=w_config['wp.inflow.f_g'])

    if w_config['wp.choke.class_name'] != 'SimpsonChokeModel':
        raise ValueError('Choke model class must be "SimpsonChokeModel"')
    choke_model = SimpsonChokeModel(K_c=w_config['wp.choke.K_c'], chk_profile=w_config['wp.choke.chk_profile'])

    wp = WellProperties(L=w_config['wp.L'], D=w_config['wp.D'], rho_l=w_config['wp.rho_l'], R_s=w_config['wp.R_s'],
                        cp_g=w_config['wp.cp_g'], cp_l=w_config['wp.cp_l'], f_D=w_config['wp.f_D'], h=w_config['wp.h'],
                        inflow=inflow_model, choke=choke_model)

    bc = BoundaryConditions(p_r=w_config['bc.p_r'], p_s=w_config['bc.p_s'], T_r=w_config['bc.T_r'], T_s=w_config['bc.T_s'],
                            u=w_config['bc.u'], w_lg=w_config['bc.w_lg'])

    gas = GasProperties(w_config['gas.name'], w_config['gas.R_s'], w_config['gas.cp'])
    oil = LiquidProperties(w_config['oil.name'], w_config['oil.rho'], w_config['oil.cp'])
    water = LiquidProperties(w_config['water.name'], w_config['water.rho'], w_config['water.cp'])
    fractions = (w_config['fraction.gas'], w_config['fraction.oil'], w_config['fraction.water'])
    has_gas_lift = w_config['has_gas_lift']

    well = Well(wp=wp, bc=bc, gas=gas, oil=oil, water=water, fractions=fractions, has_gas_lift=has_gas_lift)

    return well


if __name__ == "__main__":
    # Load config
    df_meta = pd.read_csv('./data/manywells-sol/manywells-sol-1_config.zip', compression='zip')

    # Load specific well
    well = load_well(0, df_meta)  # Load well 0
    print('Well parameters:')
    print(well)

    try:
        # Simulate well
        well.bc.u = 0.5  # Set choke opening to 50%
        sim = SSDFSimulator(well.wp, well.bc)
        x = sim.simulate()
        
        # Print solution
        df_x = sim.solution_as_df(x)
        print('Solution:')
        print(df_x)

    except SimError as e:
        print('Could not simulate well:', e)
