"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Writes the example PVT tables of tests/data/pvt/ that ManyWells makes itself (specs/model/pvt/table.md,
tests/data/pvt/README.md):

  vazquez_beggs.csv           develop's black oil, API 35, gas gravity 0.75, GOR 120, capped at its bubble point,
                              235.688 bar, where R_so is the GOR at 360 K; Z from DAK. 1, 10, ..., 400 bar and p_b;
                              280 to 400 K by 10
  synthetic_eos_oil_360K.csv  synthetic_eos_oil.csv at 360 K only, on 15 of its pressures (a laboratory report's
                              spacing), without the viscosities and the surface tension

The two equation-of-state tables, synthetic_eos_oil.csv and synthetic_eos_oil_gor.csv, come from NeqSim, by
plans/evidence/pvt_table_eos.py. From the repository root:

    uv run python specs/tools/make_pvt_tables.py
"""

import pathlib

import numpy as np

from manywells.pvt import gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.table import PVTTable

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / 'tests' / 'data' / 'pvt'

VB = BlackOil(rho_o=850.0, rho_g=gas_density_from_sg(0.75), p_bubble=235.688)
VB_GOR = 120.0
VB_P = np.unique(np.concatenate([[1.0], np.arange(10.0, 400.0 + 1e-9, 10.0), [VB.p_bubble]]))
VB_T = np.arange(280.0, 400.0 + 1e-9, 10.0)
LAB_P = (10, 30, 50, 70, 90, 110, 130, 150, 170, 178.346, 200, 250, 300, 350, 400)


def vazquez_beggs() -> PVTTable:
    """develop's black oil tabulated at the nodes: R_so, B_o and Z, each the black oil's to the bit."""
    P, T = np.meshgrid(VB_P, VB_T, indexing='ij')
    columns = {
        name: np.vectorize(lambda p, t, f=f: float(f(p, t)))(P, T)
        for name, f in (('R_so', VB.rs), ('B_o', VB.bo), ('Z', VB.z_factor))
    }
    return PVTTable(
        p=VB_P,
        T=VB_T,
        rho_o=VB.rho_o,
        rho_g=VB.rho_g,
        gor=VB_GOR,
        name='vazquez_beggs',
        source=(
            f"specs/tools/make_pvt_tables.py; develop's black oil: BlackOil(rho_o=850, "
            f'rho_g=gas_density_from_sg(0.75), p_bubble={VB.p_bubble}) at gor {VB_GOR}, R_so capped at '
            f'the bubble point (PVT-OIL-7), where R_so is the GOR at 360 K; Z from DAK (PVT-GAS-11)'
        ),
        separation="standard conditions (BlackOil's p_sep and T_sep)",
        **columns,
    )


def eos_360K() -> PVTTable:
    """synthetic_eos_oil.csv at 360 K, on LAB_P, with R_so, B_o and Z only."""
    eos = PVTTable.read_csv(DATA / 'synthetic_eos_oil.csv')
    k = eos.T.index(360.0)
    rows = [eos.p.index(p) for p in LAB_P]
    columns = {c: np.array(getattr(eos, c))[rows, k : k + 1] for c in ('R_so', 'B_o', 'Z')}
    return PVTTable(
        p=[eos.p[i] for i in rows],
        T=[360.0],
        rho_o=eos.rho_o,
        rho_g=eos.rho_g,
        gor=eos.gor,
        name='synthetic_eos_oil_360K',
        separation=eos.separation,
        source=(
            'specs/tools/make_pvt_tables.py; synthetic_eos_oil.csv at 360 K only, on 15 of its '
            "pressures (a laboratory report's spacing), without the viscosities and the surface "
            'tension'
        ),
        **columns,
    )


def main():
    for name, make in (('vazquez_beggs', vazquez_beggs), ('synthetic_eos_oil_360K', eos_360K)):
        make().write_csv(DATA / f'{name}.csv')
        print(f'wrote tests/data/pvt/{name}.csv')


if __name__ == '__main__':
    main()
