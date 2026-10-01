"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of the steady-state drift flux model for two-phase flow in wellbores

The simulator builds a well's discretized system once (manywells.discretization), with the operating point as
parameters, and finds its roots (manywells.solvers). The model's answer is a root set, each root labelled stable
or unstable; the operating point is the stable root (specs/model/solution.md).
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from manywells.choke import ChokeModel, BernoulliChokeModel
from manywells.discretization import STATE, DIM_X, build_system
from manywells.friction import FrictionModel, RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import InflowModel, ProductivityIndex
from manywells.pvt.fluid import FluidModel
from manywells.slip import SlipModel
from manywells.solution import Root, RootSet
from manywells.solvers.roots import RootFinder
from manywells.thermal import ThermalModel

log = logging.getLogger('manywells')


class SimError(Exception):
    """ Exception caused by simulator """
    pass


class NoOperatingPoint(SimError):
    """The well has no stable root at these conditions, so it cannot flow (SOL-5). The root set is in root_set."""

    def __init__(self, root_set: RootSet):
        n = len(root_set.roots)
        super().__init__(f'no stable root: the search found {n} root(s), none of them stable')
        self.root_set = root_set


@dataclass(frozen=True)
class WellProperties:
    """A well: one object per model part. The defaults are develop's model."""

    # Well geometry
    geometry: WellGeometry = field(default_factory=lambda: WellGeometry.vertical(length=2000, n_cells=100))

    # Fluid model
    fluid: FluidModel = field(default_factory=FluidModel)

    # Friction
    friction: FrictionModel = field(default_factory=RoughnessFriction)

    # Heat transfer
    thermal: ThermalModel = field(default_factory=ThermalModel)

    # Slip relation
    slip: SlipModel = field(default_factory=SlipModel)

    # Productivity
    inflow: InflowModel = field(default_factory=lambda: ProductivityIndex(k_l=0.5))

    # Choke model. None means a Bernoulli choke with K_c = 10% of the pipe's cross-section.
    choke: ChokeModel = None

    def __post_init__(self):
        for name, cls in (('geometry', WellGeometry), ('fluid', FluidModel), ('friction', FrictionModel),
                          ('thermal', ThermalModel), ('slip', SlipModel), ('inflow', InflowModel)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f'{name} must be a {cls.__name__}, got {type(getattr(self, name)).__name__}')
        if self.choke is None:
            object.__setattr__(self, 'choke', BernoulliChokeModel(K_c=0.1 * self.geometry.A))
        elif not isinstance(self.choke, ChokeModel):
            raise ValueError(f'choke must be a ChokeModel, got {type(self.choke).__name__}')


@dataclass(frozen=True)
class BoundaryConditions:
    # Pressures
    p_r: float = 170          # Upstream reservoir pressure (bar)
    p_s: float = 20           # Downstream separator pressure (bar)

    # Temperatures
    T_r: float = 373.15       # Reservoir temperature (K). Default value corresponds to 100 decC.
    T_s: float = 277.15       # Ambient temperature (K) at the surface z=L. Default value corresponds to 4 degC.
    T_lg: float = None        # Lift gas temperature (K) at injection point. None means T_r (no mixing effect).

    # Controls
    u: float = 1.             # Choke position (dimensionless). Must be in [0,1].
    w_lg: float = 0.          # Lift gas mass flow rate (kg/s). Default value is 0. Assumed injected at z=0.

    def __post_init__(self):
        checks = [
            (self.p_r > 0, 'Reservoir pressure must be positive'),
            (self.p_s > 0, 'Separator pressure must be positive'),
            (self.T_r > 0, 'Reservoir temperature must be positive'),
            (self.T_s > 0, 'Ambient temperature must be positive'),
            (self.T_lg is None or self.T_lg > 0, 'Lift gas temperature must be positive'),
            (0 <= self.u <= 1, 'Choke opening must be in [0, 1]'),
            (0 <= self.w_lg, 'Gas lift rate must be non-negative'),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


class SSDFSimulator:
    """
    Implementation of the steady-state drift-flux model

        sim = SSDFSimulator(wp)          # validates and builds the well's system once
        op = sim.simulate(bc)            # the operating point, a Root; raises NoOperatingPoint
        rs = sim.root_set(bc)            # every root found, labelled; empty if the well cannot flow
        df = sim.solution_as_df(op)      # per point: state, md, tvd, flow regime

    The pipe is discretized into n cells (see WellGeometry object). The state holds the following variables at each
    of the n + 1 grid points (in the given order), from the bottomhole to the wellhead:
        x = [p, v_g, v_l, alpha, rho_g, rho_l, T],
    where p is pressure, v_g and v_l are gas and liquid velocities, alpha is the volumetric fraction of gas,
    rho_g and rho_l are the gas and liquid densities, and T is temperature.

    Deprecated: SSDFSimulator(wp, bc) with simulate() and the x_guess attribute, which returns the operating point's
    state as a flat list. It is removed with the CasADi backend.
    """

    def __init__(self, well_properties: WellProperties, boundary_conditions: BoundaryConditions = None):
        """
        :param well_properties: Well properties (object of type WellProperties)
        :param boundary_conditions: Deprecated; pass the boundary conditions to simulate() instead
        """
        if boundary_conditions is not None:
            warnings.warn('SSDFSimulator(wp, bc) is deprecated: use SSDFSimulator(wp) and simulate(bc), which returns '
                          'the operating point as a Root', DeprecationWarning, stacklevel=2)
        self.wp = well_properties       # Well properties
        self.bc = boundary_conditions   # Boundary conditions of the deprecated simulate()
        self.x_guess = None             # Initial guess of the deprecated simulate()

        self.geo = well_properties.geometry  # Convenience alias
        self.n_cells = self.geo.n_cells
        self.dim_x = DIM_X
        self.variable_names = list(STATE)  # Ordering is important

        self.system = build_system(well_properties)
        self._roots = RootFinder(self.system)

    def root_set(self, bc: BoundaryConditions, x_guess=None) -> RootSet:
        """
        Every root the search finds at the operating point bc, each labelled stable, unstable or indeterminate,
        sorted by bottomhole pressure, and the operating point.

        :param bc: Boundary conditions
        :param x_guess: An extra start for the search, such as the root of a nearby operating point
        :return: The root set
        """
        rs = self._roots.find(bc, x_guess=x_guess)
        failed = [a for a in rs.search if not a.outcome.startswith('root')]
        if failed:
            log.debug('%d of %d starts found no root: %s', len(failed), len(rs.search),
                      '; '.join(f'{a.start}: {a.outcome}' for a in failed))
        return rs

    def simulate(self, bc: BoundaryConditions = None, x_guess=None):
        """
        The operating point at bc: the stable root, or the one with the lowest bottomhole pressure if there are
        several (SOL-4 to SOL-6).

        :param bc: Boundary conditions. Deprecated: None, with the boundary conditions given to the constructor
        :param x_guess: An extra start for the search; it makes the search faster, not the answer different
        :return: The operating point (Root); a flat list of its state in the deprecated form
        :raises NoOperatingPoint: if there is no stable root
        """
        legacy = bc is None
        if legacy:
            if self.bc is None:
                raise TypeError('simulate() needs the boundary conditions')
            bc, x_guess = self.bc, self.x_guess
        rs = self.root_set(bc, x_guess=x_guess)
        if rs.operating_point is None:
            raise NoOperatingPoint(rs)
        if rs.several_stable:
            log.info('%d stable roots; the operating point is the one with the lowest p_0 (SOL-6)',
                     sum(r.label == 'stable' for r in rs.roots))
        return rs.operating_point.x.tolist() if legacy else rs.operating_point

    def solution_as_df(self, x):
        """
        Represent a solution as a DataFrame, one row per grid point from the bottomhole

        :param x: A Root, or its state as a flat list
        :return: Solution as DataFrame: the state, md, tvd and the flow regime
        """
        if isinstance(x, Root):
            regimes = x.flow_regime
            X = x.state
        else:
            X = np.asarray(x, dtype=float).reshape(-1, DIM_X)
            regimes = self.system.flow_regimes(X.ravel())

        df = pd.DataFrame(X, columns=self.variable_names)

        # Add geometry columns (simulator order: bottom to top)
        df.insert(loc=1, column='md', value=np.array(self.geo.md))
        df.insert(loc=2, column='tvd', value=np.array(self.geo.tvd))

        df['flow-regime'] = list(regimes)
        return df
