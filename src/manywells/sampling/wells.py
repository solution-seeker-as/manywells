"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Well draws (specs/sampling.md, SMP-1 to SMP-17 and SMP-41 to SMP-43), their seeding (SMP-31), and the map from a
draw to the inputs of a configuration (SMP-40).
"""

import zlib
from dataclasses import dataclass, replace

import numpy as np

from manywells.choke import CHOKE_PROFILES, SimpsonChokeModel
from manywells.configurations import DEVELOP, V1, v1_well
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Vogel
from manywells.pvt import LiquidProperties, liquid_mix
from manywells.pvt.fluid import FluidModel
from manywells.simulator import WellProperties
from manywells.thermal import ThermalModel
from manywells.units import CF_BAR, P_REF, STD_GRAVITY, T_REF

INCH = 0.0254                                               # m
OUTER_DIAMETERS = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)  # inch
RHO_SEAWATER, RHO_WATER, CP_WATER = 1025.0, 999.1, 4184.0   # kg/m³, J/(kg K)  # spec: SMP-10
CP_OIL, CP_GAS = 2000.0, 2225.0                             # J/(kg K)
N_CELLS = 100


def rng_for(seed: int, *parts) -> np.random.Generator:  # spec: SMP-31
    """
    The random generator of one draw: seeded from the dataset's seed and the draw's place in it, such as
    (well index, 'well') or (well index, sample index, 'sample'). Runs are reproducible whatever the order or the
    number of processes, and the develop draws of a well do not shift its v1 draws.
    """
    key = [zlib.crc32(str(p).encode()) & 0xFFFFFFFF if isinstance(p, str) else int(p) for p in parts]
    return np.random.default_rng([int(seed)] + key)


@dataclass(frozen=True)
class WellDraw:
    """
    The draws of one well. The v1 draws are v1.0.0's well and its nominal operating point (SMP-1 to SMP-17); the
    develop draws are the inputs that only develop's model takes (SMP-41 to SMP-43), drawn from their own generator.
    """
    # v1 draws
    L: float                        # Pipe length, or the bottomhole's true vertical depth (m), SMP-1
    D: float                        # Inner diameter (m), SMP-2
    f_D: float                      # Darcy friction factor, SMP-3
    h: float                        # Heat transfer coefficient (W/(m² K)), SMP-4
    K_c: float                      # Choke coefficient (m²), SMP-5
    chk_profile: str                # Choke profile, SMP-6
    fractions: tuple                # (f_g, f_o, f_w), mass fractions of the reservoir inflow, SMP-7
    w_l_max: float                  # Vogel's maximum liquid rate (kg/s), SMP-8
    rho_o: float                    # Oil density (kg/m³), SMP-9
    R_s: float                      # Specific gas constant (J/(kg K)), SMP-11
    p_r: float                      # Nominal reservoir pressure (bar), SMP-13
    T_r: float                      # Reservoir temperature (K), SMP-14
    p_s: float                      # Nominal separator pressure (bar), SMP-16
    has_gas_lift: bool              # SMP-17
    T_s: float = 277.15             # Surface temperature (K)  # spec: SMP-15
    cp_o: float = CP_OIL            # SMP-9
    rho_w: float = RHO_WATER        # SMP-10
    cp_w: float = CP_WATER          # SMP-10
    cp_g: float = CP_GAS            # SMP-11
    # develop draws
    trajectory: tuple = ('vertical',)   # SMP-41: ('vertical',), ('deviated', kickoff fraction, inclination in degrees),
                                        # or ('l_shaped', horizontal length in m)
    roughness: float = 4.5e-5           # Pipe wall roughness (m), SMP-42

    @property
    def x_o(self) -> float:
        """Oil mass fraction of the liquid."""
        f_g, f_o, f_w = self.fractions
        return f_o / (f_o + f_w)


def _v1_draws(rng):
    """One attempt at v1.0.0's well draws, in v1.0.0's order (`sample_well`), or None if the well is discarded."""
    L = rng.uniform(1500, 4500)                                                          # spec: SMP-1
    D = rng.choice([INCH * (od - 0.5) for od in OUTER_DIAMETERS])                        # spec: SMP-2
    p_r = (1 / CF_BAR) * ((RHO_SEAWATER + RHO_WATER) / 2) * STD_GRAVITY * L + 1         # spec: SMP-13
    T_r = 273.15 + 60 + (L - 1500) * 0.03                                               # spec: SMP-14
    p_s = rng.lognormal(3, 1)                                                            # spec: SMP-16
    if p_s < 10 or p_s > 120:
        return None
    h = rng.uniform(10, 40)                                                              # spec: SMP-4
    f_D = rng.uniform(0.01, 0.08)                                                        # spec: SMP-3
    f_g, f_o, f_w = rng.dirichlet(alpha=(1.0, 1.0, 0.5))                                # spec: SMP-7
    if f_g > 0.99:
        return None
    R_s = rng.uniform(320, 520)                                                          # spec: SMP-11
    rho_o = rng.uniform(825, 925)                                                        # spec: SMP-9
    w_l_max = (1 - f_g) * rng.uniform(20, 200)                                          # spec: SMP-8
    A = np.pi * (D / 2) ** 2
    K_c = rng.uniform(0.06 * A, 0.24 * A)                                               # spec: SMP-5
    chk_profile = str(rng.choice(CHOKE_PROFILES))                                       # spec: SMP-6
    has_gas_lift = bool(f_g <= 0.2 and rng.choice([0, 1]) == 0)                          # spec: SMP-17
    return WellDraw(L=float(L), D=float(D), f_D=float(f_D), h=float(h), K_c=float(K_c), chk_profile=chk_profile,
                    fractions=(float(f_g), float(f_o), float(f_w)), w_l_max=float(w_l_max), rho_o=float(rho_o),
                    R_s=float(R_s), p_r=float(p_r), T_r=float(T_r), p_s=float(p_s), has_gas_lift=has_gas_lift)


def _develop_draws(rng):
    """develop's further draws (SMP-41, SMP-42): Step 4's starting points, kept until the sampling redesign."""
    kind = rng.choice(['vertical', 'deviated', 'l_shaped'], p=[0.5, 0.25, 0.25])       # spec: SMP-41
    if kind == 'deviated':
        trajectory = ('deviated', float(rng.uniform(0.1, 0.5)), float(rng.uniform(10, 60)))
    elif kind == 'l_shaped':
        trajectory = ('l_shaped', float(rng.uniform(500, 2000)))
    else:
        trajectory = ('vertical',)
    roughness = float(np.exp(rng.uniform(np.log(1.5e-6), np.log(1.5e-4))))              # spec: SMP-42
    return {'trajectory': trajectory, 'roughness': roughness}


def sample_well(seed: int, well: int) -> WellDraw:
    """
    The draws of well number `well` of a dataset with seed `seed`. A discarded attempt (SMP-7, SMP-16) is redrawn
    from the same generator, as v1.0.0 redrew it.
    """
    rng = rng_for(seed, well, 'well')
    draw = None
    while draw is None:
        draw = _v1_draws(rng)
    return replace(draw, **_develop_draws(rng_for(seed, well, 'develop')))


def well_from_config(row) -> WellDraw:
    """The v1 draws of a well of a published sol-1 config (a row of manywells-sol-1_config.zip, as a dict)."""
    if row['wp.inflow.class_name'] != 'Vogel' or row['wp.choke.class_name'] != 'SimpsonChokeModel':
        raise ValueError('a sol-1 well has Vogel inflow and a Simpson choke')
    return WellDraw(L=row['wp.L'], D=row['wp.D'], f_D=row['wp.f_D'], h=row['wp.h'], K_c=row['wp.choke.K_c'],
                    chk_profile=row['wp.choke.chk_profile'],
                    fractions=(row['fraction.gas'], row['fraction.oil'], row['fraction.water']),
                    w_l_max=row['wp.inflow.w_l_max'], rho_o=row['oil.rho'], R_s=row['gas.R_s'], p_r=row['bc.p_r'],
                    T_r=row['bc.T_r'], p_s=row['bc.p_s'], has_gas_lift=bool(row['has_gas_lift']), T_s=row['bc.T_s'],
                    cp_o=row['oil.cp'], rho_w=row['water.rho'], cp_w=row['water.cp'], cp_g=row['gas.cp'])


def liquid(draw: WellDraw, fractions=None) -> LiquidProperties:  # spec: SMP-12
    """The mixed liquid of the draw's oil and water at the given fractions (PVT-MIX-2 to PVT-MIX-4)."""
    f_g, f_o, f_w = fractions or draw.fractions
    return liquid_mix(LiquidProperties('oil', draw.rho_o, draw.cp_o), LiquidProperties('water', draw.rho_w, draw.cp_w),
                      f_o / (f_o + f_w))


def geometry(draw: WellDraw, n_cells: int = N_CELLS) -> WellGeometry:  # spec: SMP-41
    """The trajectory of the develop draws; its bottomhole's true vertical depth is L."""
    kind, *prm = draw.trajectory
    if kind == 'vertical':
        return WellGeometry.vertical(draw.L, n_cells, D=draw.D)
    if kind == 'deviated':
        kickoff, theta = prm[0] * draw.L, np.radians(prm[1])
        md = kickoff + (draw.L - kickoff) / np.cos(theta)
        return WellGeometry.from_survey([0.0, kickoff, md], [0.0, kickoff, draw.L], n_cells, D=draw.D)
    if kind == 'l_shaped':
        return WellGeometry.from_survey([0.0, draw.L, draw.L + prm[0]], [0.0, draw.L, draw.L], n_cells, D=draw.D)
    raise ValueError(f'unknown trajectory {kind!r}')


def well_properties(draw: WellDraw, fractions=None, configuration: str = V1,
                    n_cells: int = N_CELLS) -> WellProperties:  # spec: SMP-40
    """
    A draw's well in a configuration, at the given fractions (the draw's by default).

    v1.0.0: the vertical pipe of length L, the mixed liquid as a dead oil with the gas-oil ratio that gives f_g
    (configurations.v1_fluid), the fixed f_D, and v1.0.0's thermal model with h. develop: the trajectory, black oil
    and water by their densities at standard conditions and the water-liquid ratio, real gas, friction from the
    roughness, and develop's thermal model with h. Both: Vogel inflow and a Simpson choke.
    """
    f_g, f_o, f_w = fractions or draw.fractions
    inflow = Vogel(w_l_max=draw.w_l_max)
    choke = SimpsonChokeModel(K_c=draw.K_c, chk_profile=draw.chk_profile)
    if configuration == V1:
        mix = liquid(draw, (f_g, f_o, f_w))
        return v1_well(L=draw.L, D=draw.D, rho_l=mix.rho, R_s=draw.R_s, cp_g=draw.cp_g, cp_l=mix.cp, f_D=draw.f_D,
                       h=draw.h, f_g=f_g, inflow=inflow, choke=choke, n_cells=n_cells)
    if configuration == DEVELOP:
        if f_o <= 0:
            raise ValueError('develop configuration: no oil to give a gas-oil ratio (specs/sampling.md, SMP-43)')
        # spec: SMP-43. Black oil and real gas from the v1 draws: API from rho_o, gas gravity from R_s
        rho_g = P_REF / (draw.R_s * T_REF)                                               # PVT-GAS-2
        wlr = (f_w / draw.rho_w) / (f_w / draw.rho_w + f_o / draw.rho_o)                 # PVT-MIX-2
        fluid = FluidModel(rho_o=draw.rho_o, rho_g=rho_g, rho_w=draw.rho_w, gor=(f_g / rho_g) / (f_o / draw.rho_o),
                           wlr=wlr, cp_g=draw.cp_g, cp_o=draw.cp_o, cp_w=draw.cp_w)
        return WellProperties(geometry=geometry(draw, n_cells), fluid=fluid,
                              friction=RoughnessFriction(roughness=draw.roughness), thermal=ThermalModel(h=draw.h),
                              inflow=inflow, choke=choke)
    raise ValueError(f'unknown configuration {configuration!r}')
