"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The model's answer (specs/model/solution.md): a case's roots, each labelled stable, unstable or indeterminate, and
its operating point, the stable root.
"""

from dataclasses import dataclass, field

import numpy as np

DIM_X = 7
LABELS = ('stable', 'unstable', 'indeterminate')


@dataclass(frozen=True, eq=False)
class Root:
    """
    One steady-state root of a well at an operating point.

    x holds the 7(N + 1) state values, point by point from the bottomhole, in the order
    [p, v_g, v_l, alpha, rho_g, rho_l, T] (bar, m/s, m/s, -, kg/m³, kg/m³, K).
    """
    x: np.ndarray
    label: str                   # One of LABELS (SOL-3)
    slope: float                 # dR/dp_0 normalized by (p_r - p_s) / w_m; positive is unstable
    choked: bool                 # CHK-12, at the root
    flow_regime: tuple           # Regime label at each point (SLIP-8), bottomhole first
    w_res: float                 # Reservoir liquid mass rate (kg/s)
    w_g_res: float               # Reservoir gas mass rate (kg/s), without the lift gas

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f'label {self.label!r} is not one of {LABELS}')
        object.__setattr__(self, 'x', np.asarray(self.x, dtype=float))

    @property
    def state(self) -> np.ndarray:
        """The state as an (N + 1, 7) array, one row per grid point."""
        return self.x.reshape(-1, DIM_X)

    @property
    def p_0(self) -> float:
        """Bottomhole pressure (bar)."""
        return float(self.x[0])


def select_operating_point(roots):  # spec: SOL-4, SOL-5, SOL-6
    """
    The operating point of a root set: its stable root, or the stable root with the lowest p_0 if there are
    several, or None if there is none.

    :param roots: Roots (Root)
    :return: (operating point or None, whether there are several stable roots)
    """
    stable = sorted((r for r in roots if r.label == 'stable'), key=lambda r: r.p_0)
    return (stable[0] if stable else None), len(stable) > 1


@dataclass(frozen=True)
class RootSet:
    """
    Every root a search found for one well at one operating point, sorted by p_0, and the operating point.

    search records each start the search tried and its outcome; it is not part of the model.
    """
    roots: tuple
    operating_point: Root = None
    several_stable: bool = False
    search: tuple = field(default=(), compare=False)

    @classmethod
    def of(cls, roots, search=()):
        roots = tuple(sorted(roots, key=lambda r: r.p_0))
        op, several = select_operating_point(roots)
        return cls(roots=roots, operating_point=op, several_stable=several, search=tuple(search))
