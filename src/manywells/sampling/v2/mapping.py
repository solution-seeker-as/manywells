"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The map from a v2 well's draws and an attempt's inputs to the model's inputs in the v2.0.0 configuration
(specs/sampling.md, SMP-111).
"""

import math

from manywells.choke import SimpsonChokeModel
from manywells.configurations import V2_SLIP
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Forchheimer, ProductivityIndex, Vogel
from manywells.pvt.fluid import FluidModel
from manywells.sampling.v2.draws import _black_oil
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.thermal import ThermalModel

_N_CELLS = 100  # spec: SMP-111


class WellBuilder:  # spec: SMP-111
    """
    A drawn well's fixed parts (the survey's grid, the PVT model and the choke), and the WellProperties and
    BoundaryConditions of an attempt.

    :param w: The well's draws (draws.draw_well)
    :param stations: Its survey, (md, tvd)
    :param design: The draws that hold the well's lift design (`injection_md` and `T_lg`), or None for a well that
        is never lifted
    :param pvt: A table fluid's PVTTableSet, or None for the well's black oil
    """

    def __init__(self, w, stations, design=None, pvt=None):
        self.w, self.design = w, design
        self.geometry = WellGeometry.from_survey(stations[0], stations[1], _N_CELLS, D=w['D'])
        self.pvt = _black_oil(w) if pvt is None else pvt
        self.choke = SimpsonChokeModel(K_c=w['K_c'], chk_profile=w['chk_profile'])

    def fluid(self, gor, wlr) -> FluidModel:
        """The well's fluid at a gas-oil ratio (Sm3/Sm3) and a water-liquid ratio."""
        w = self.w
        return FluidModel(
            pvt=self.pvt, rho_w=w['rho_w'], gor=gor, wlr=wlr, cp_g=w['cp_g'], cp_o=w['cp_o'], cp_w=w['cp_w']
        )

    def case(self, inp) -> tuple:
        """
        The WellProperties and BoundaryConditions of an attempt.

        :param inp: The attempt's inputs: `p_r` and `p_s` (bar), `u`, `w_lg` (kg/s), `lift_on`, `gor` (Sm3/Sm3),
            `wlr`, `h` (W/(m2 K)), `roughness` (m), and the inflow's coefficients (inflow._inflow_at): `forchheimer_a`
            and `forchheimer_b`, or `w_l_max`, or `k_l`, the others NaN or absent
        """
        w, lift = self.w, bool(inp['lift_on'])
        if not math.isnan(inp.get('forchheimer_a', math.nan)):
            inflow = Forchheimer(a=inp['forchheimer_a'], b=inp['forchheimer_b'])
        elif not math.isnan(inp.get('w_l_max', math.nan)):
            inflow = Vogel(w_l_max=inp['w_l_max'])
        else:
            inflow = ProductivityIndex(k_l=inp['k_l'])
        wp = WellProperties(
            geometry=self.geometry,
            fluid=self.fluid(inp['gor'], inp['wlr']),
            friction=RoughnessFriction(roughness=inp['roughness'], correlation='chen'),
            thermal=ThermalModel(h=inp['h']),
            slip=V2_SLIP,
            inflow=inflow,
            choke=self.choke,
            injection_md=self.design['injection_md'] if lift else None,
        )
        bc = BoundaryConditions(
            p_r=inp['p_r'],
            p_s=inp['p_s'],
            T_r=w['T_r'],
            T_s=w['T_s'],
            T_lg=self.design['T_lg'] if lift else None,
            u=inp['u'],
            w_lg=inp['w_lg'] if lift else 0.0,
        )
        return wp, bc
