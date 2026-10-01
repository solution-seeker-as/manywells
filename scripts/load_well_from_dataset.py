"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 15 December 2025
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Load a well of the published manywells-sol-1 dataset and simulate it in the v1.0.0 configuration. From the project
root:

    uv run python -m scripts.load_well_from_dataset <manywells-sol-1_config.zip> [ID]
"""

import sys

import pandas as pd

from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import WellDraw, well_from_config, well_properties
from manywells.simulator import SimError, SSDFSimulator


def load_well(well_id, df_config) -> WellDraw:
    """The v1 draws of a published sol-1 well."""
    assert well_id in df_config['ID'].tolist(), 'ID not found'
    return well_from_config(df_config[df_config['ID'] == well_id].to_dict('records')[0])


if __name__ == "__main__":
    df_meta = pd.read_csv(sys.argv[1], compression='zip')
    draw = load_well(int(sys.argv[2]) if len(sys.argv) > 2 else 0, df_meta)
    print('Well:', draw)
    try:
        sim = SSDFSimulator(well_properties(draw))   # the v1.0.0 configuration
        op = sim.simulate(nominal_conditions(draw, u=0.5))
        print(sim.solution_as_df(op))
    except SimError as e:
        print('Could not simulate well:', e)
