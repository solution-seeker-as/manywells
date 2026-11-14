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
from types import SimpleNamespace

from manywells.choke import SimpsonChokeModel
from manywells.configurations import V2_SLIP
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Forchheimer, ProductivityIndex, Vogel
from manywells.pvt.fluid import FluidModel
from manywells.sampling.v2.draws import _black_oil, _size_bean
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.thermal import ThermalModel
from manywells.units import CF_BAR

_N_CELLS = 100  # spec: SMP-111


class WellBuilder:  # spec: SMP-111
    """
    A drawn well's fixed parts (the survey's grid, the PVT model and the choke), and the WellProperties and
    BoundaryConditions of an attempt. A well whose bean is not sized yet (SMP-77) has its largest choke, with which the
    generation solves it to size the bean (size_choke).

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
        self.choke = None
        if w['K_c'] is None:
            self.widest_choke()
        else:
            self.choke = SimpsonChokeModel(K_c=w['K_c'], chk_profile=w['chk_profile'])

    def widest_choke(self, share: float = 1.0) -> float:  # spec: SMP-77
        """Gives the well its largest choke, `K_c_max`, or the given share of it, and returns the choke's
        coefficient (m2)."""
        self.choke = SimpsonChokeModel(K_c=share * self.w['K_c_max'], chk_profile=self.w['chk_profile'])
        return self.choke.K_c

    def size_choke(self, w_m: float, wellhead, p_dn: float, K_c_most: float = math.inf) -> float:  # spec: SMP-77
        """
        Sizes the well's choke: fully open, it passes the mass rate w_m at the wellhead's state with the share
        `xi_choke` of the pressure upstream of it lost across it, down to p_dn, by the model's choke equation (CHK-2
        with Simpson's density, CHK-5). It is at most K_c_most and the well's largest choke (draws._size_bean), and
        the well's draws take it.

        :param w_m: Mass rate (kg/s)
        :param wellhead: The state at the wellhead, (p, v_g, v_l, alpha, rho_g, rho_l, T)
        :param p_dn: Pressure downstream of the choke (bar)
        :param K_c_most: The largest coefficient the choke may take (m2): the sizing's own choke
        :return: The choke's coefficient K_c (m2)
        """
        _p, v_g, v_l, alpha, rho_g, rho_l, _T = (float(v) for v in wellhead)
        s = SimpleNamespace(v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l)
        rho, phi = SimpsonChokeModel(K_c=1.0).density_and_multiplier(s, self.geometry.A)
        xi = self.w['xi_choke']
        dp = CF_BAR * p_dn * xi / (1 - xi)  # Pa: p_up - p_dn at p_dn = (1 - xi) p_up
        return self.take_choke(min(w_m / math.sqrt(2 * rho / float(phi) * dp), K_c_most))

    def take_choke(self, K_c: float) -> float:  # spec: SMP-77
        """Gives the well the choke K_c (m2), at most its largest (draws._size_bean), which the well's draws take, and
        returns its coefficient."""
        K_c = _size_bean(self.w, K_c)
        self.choke = SimpsonChokeModel(K_c=K_c, chk_profile=self.w['chk_profile'])
        return K_c

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
