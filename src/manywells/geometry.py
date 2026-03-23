"""
Well geometry module.

Provides the WellGeometry dataclass for representing vertical and deviated
wellbore trajectories.  A geometry is specified by a list of (MD, TVD)-pairs
where MD is measured depth and TVD is true vertical depth, both in meters,
with origin at the surface (top of the well).

The geometry pre-computes the computational grid and per-cell inclination
angles used by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WellGeometry:
    """Wellbore geometry defined by an (MD, TVD) survey.

    Parameters
    ----------
    md_survey : Sequence[float]
        Measured depth at survey stations (m), starting at 0, increasing.
    tvd_survey : Sequence[float]
        True vertical depth at survey stations (m), starting at 0, increasing.
    n_cells : int
        Number of computational cells for the spatial discretisation.
    D : float
        Inner pipe diameter (m).  Default ~6.1 inch.

    Pre-computed attributes (set in ``__post_init__``, simulator order:
    index 0 = bottom, index *n_cells* = top/surface):

    md : tuple[float, ...]
        Grid-node measured depths (length *n_cells* + 1).
    tvd : tuple[float, ...]
        Grid-node true vertical depths (length *n_cells* + 1).
    delta_md : float
        Uniform cell length along the wellbore.
    cos_incl : tuple[float, ...]
        Cosine of the inclination from vertical for each cell
        (length *n_cells*).
    tvd_frac : tuple[float, ...]
        Fractional TVD at each grid node relative to the deepest point
        (length *n_cells* + 1).  Used for the geothermal gradient.
    """

    md_survey: Sequence[float]
    tvd_survey: Sequence[float]
    n_cells: int
    D: float = 0.1554

    md: tuple = None
    tvd: tuple = None
    delta_md: float = None
    cos_incl: tuple = None
    tvd_frac: tuple = None

    def __post_init__(self):
        md_arr = np.asarray(self.md_survey, dtype=float)
        tvd_arr = np.asarray(self.tvd_survey, dtype=float)

        # --- validation ---
        if len(md_arr) != len(tvd_arr):
            raise ValueError("md_survey and tvd_survey must have the same length")
        if len(md_arr) < 2:
            raise ValueError("Survey must have at least two stations")
        if md_arr[0] != 0.0:
            raise ValueError("md_survey must start at 0 (surface)")
        if tvd_arr[0] != 0.0:
            raise ValueError("tvd_survey must start at 0 (surface)")
        if not np.all(np.diff(md_arr) > 0):
            raise ValueError("md_survey must be strictly increasing")
        if np.any(md_arr < tvd_arr - 1e-12):
            raise ValueError("MD must be >= TVD at every survey station")
        if self.D <= 0:
            raise ValueError("Pipe diameter D must be positive")
        if self.n_cells < 1:
            raise ValueError("n_cells must be >= 1")

        # store inputs as immutable tuples
        object.__setattr__(self, "md_survey", tuple(md_arr))
        object.__setattr__(self, "tvd_survey", tuple(tvd_arr))

        # --- build uniform grid in survey order (surface → bottom) ---
        md_grid = np.linspace(0.0, md_arr[-1], self.n_cells + 1)
        tvd_grid = np.interp(md_grid, md_arr, tvd_arr)

        # cell length (uniform)
        object.__setattr__(self, "delta_md", float(md_arr[-1] / self.n_cells))

        # per-cell cos(inclination) in survey order
        d_tvd = np.diff(tvd_grid)
        d_md = np.diff(md_grid)
        cos_survey = d_tvd / d_md

        # --- convert to simulator order (bottom → top) ---
        object.__setattr__(self, "md", tuple(md_grid[::-1].tolist()))
        object.__setattr__(self, "tvd", tuple(tvd_grid[::-1].tolist()))
        object.__setattr__(self, "cos_incl", tuple(cos_survey[::-1].tolist()))

        # tvd_frac: fraction of max TVD at each node (simulator order)
        tvd_sim = tvd_grid[::-1]
        max_tvd = tvd_sim[0]
        if max_tvd > 0:
            frac = tvd_sim / max_tvd
        else:
            frac = np.zeros_like(tvd_sim)
        object.__setattr__(self, "tvd_frac", tuple(frac.tolist()))

    # ----- properties ---------------------------------------------------

    @property
    def L(self) -> float:
        """Total measured depth (m)."""
        return self.md_survey[-1]

    @property
    def A(self) -> float:
        """Cross-sectional area of the pipe (m²)."""
        return np.pi * (self.D / 2) ** 2

    # ----- factory methods -----------------------------------------------

    @classmethod
    def vertical(cls, length: float, n_cells: int, D: float = 0.1554) -> WellGeometry:
        """Create a vertical well geometry (MD == TVD everywhere)."""
        md = (0.0, float(length))
        tvd = (0.0, float(length))
        return cls(md_survey=md, tvd_survey=tvd, n_cells=n_cells, D=D)
