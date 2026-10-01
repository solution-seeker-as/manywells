"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Pins ClosedLoopWellSimulator's open-loop solve (feedback=False) to its values before Step 7 of
plans/manywells-v2-plan.md restructured the simulator. Closed loop is out of v2 and runs on a frozen
copy of the old simulator (manywells/closed_loop/_base.py), so these values must not change.

Both solves land on a trickle root (wellhead pressure at the separator's). feedback=True, the
default, has failed since the lift-gas temperature was added (`if bc.w_lg > 0` on a symbolic
w_lg), so it is not pinned.

The copy shares develop's components, so a ruled physics fix in one reaches closed loop: the
black-oil values were re-pinned after the separator correction took log10 (PVT-OIL-5, 2026-10-01).
"""

import numpy as np
import pytest

from manywells.closed_loop.cl_simulator import BoundaryConditions, ClosedLoopWellSimulator, WellProperties
from manywells.geometry import WellGeometry
from manywells.pvt.fluid import FluidModel

# Recorded on develop at c5fedf2, black oil again after PVT-OIL-5's fix: p_0, p_N (bar), T_N (K), v_l at point 0,
# v_g at point N (m/s), sum of alpha
PINNED = {
    'black oil': (169.98722001814562, 20.000000177710053, 277.2246445391096,
                  0.0004875841035515878, 0.20883104381377535, 0.1660340414253444),
    'dead oil': (169.7424013978306, 20.000075457170624, 278.5916279554026,
                 0.00847441984786894, 0.3157200427015511, 2.2101016303742087),
}


@pytest.mark.slow
@pytest.mark.parametrize('fluid', PINNED)
def test_open_loop_solve_is_pinned(fluid):
    np.random.seed(0)  # _initial_guess adds unseeded noise from NumPy's global state
    fl = FluidModel() if fluid == 'black oil' else FluidModel(oil_model='dead_oil', ideal_gas=True)
    wp = WellProperties(geometry=WellGeometry.vertical(2000, 20), fluid=fl)
    sim = ClosedLoopWellSimulator(wp, BoundaryConditions(u=0.8), feedback=False)
    x, _ = sim.simulate()
    X = np.array(x).reshape(-1, 7)
    got = (X[0, 0], X[-1, 0], X[-1, 6], X[0, 2], X[-1, 1], X[:, 3].sum())
    np.testing.assert_allclose(got, PINNED[fluid], rtol=1e-6)
