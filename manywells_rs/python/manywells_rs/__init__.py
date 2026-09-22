"""
manywells_rs: Rust port of the steady-state drift-flux well simulator.

The compiled extension (``manywells_rs.manywells_rs``) exposes its own
``WellProperties`` / ``BoundaryConditions`` / inflow / choke classes. This package
re-exports them and wraps the simulator in a thin Python ``SSDFSimulator``. It is
close to a drop-in for ``manywells.simulator.SSDFSimulator``, but with one
deliberate difference: ``simulate()`` returns a *list* of solutions, since a well
can have multiple steady operating points (the old simulator ignored this). The
solutions are ordered highest bottomhole pressure (least drawdown) first, so
``simulate()[0]`` recovers the operating point the old simulator would converge to.

    # from manywells.simulator import SSDFSimulator
    from manywells_rs import SSDFSimulator

    sim = SSDFSimulator(well.wp, well.bc)   # plain manywells objects, no conversion
    solutions = sim.simulate()              # list of solutions (>= 1)
    df = sim.solution_as_df(solutions[0])   # highest-p0 operating point

The wrapper converts the manywells ``wp``/``bc`` (and their inflow/choke models) to
the corresponding Rust objects internally. Conversion happens on every
``simulate()`` call, so mutating ``sim.wp``/``sim.bc`` (or their attributes) between
calls behaves exactly like the pure-Python simulator. Passing the Rust objects
directly is also supported (they are used as-is).
"""

from .manywells_rs import (  # noqa: F401  (re-exported)
    BoundaryConditions,
    WellProperties,
    Vogel,
    ProductivityIndex,
    SimpsonChokeModel,
    BernoulliChokeModel,
)
from .manywells_rs import SSDFSimulator as _NativeSSDFSimulator
from .manywells_rs import SimError as _NativeSimError

try:
    # Prefer the manywells SimError so `except SimError` in code written against
    # manywells.simulator keeps catching failures unchanged (a true drop-in). Falls
    # back to the native exception when manywells is not installed (standalone use).
    from manywells.simulator import SimError
except Exception:  # pragma: no cover - manywells not installed
    SimError = _NativeSimError

__all__ = [
    "SSDFSimulator",
    "WellProperties",
    "BoundaryConditions",
    "Vogel",
    "ProductivityIndex",
    "SimpsonChokeModel",
    "BernoulliChokeModel",
    "SimError",
]


def _to_native_inflow(inflow):
    """Convert a manywells inflow model to its manywells_rs counterpart.

    Rust inflow objects are returned unchanged. Duck-typed by class name +
    attributes so importing manywells is not required.
    """
    if isinstance(inflow, (Vogel, ProductivityIndex)):
        return inflow
    name = type(inflow).__name__
    if name == "Vogel":
        return Vogel(w_l_max=inflow.w_l_max, f_g=inflow.f_g)
    if name == "ProductivityIndex":
        return ProductivityIndex(k_l=inflow.k_l, f_g=inflow.f_g)
    raise TypeError(f"Unsupported inflow model: {name}")


def _to_native_choke(choke):
    """Convert a manywells choke model to its manywells_rs counterpart.

    Rust choke objects are returned unchanged. Simpson vs. Bernoulli is
    distinguished by class name (they share every attribute).
    """
    if isinstance(choke, (SimpsonChokeModel, BernoulliChokeModel)):
        return choke
    name = type(choke).__name__
    if name == "SimpsonChokeModel":
        return SimpsonChokeModel(K_c=choke.K_c, chk_profile=choke.chk_profile)
    if name == "BernoulliChokeModel":
        return BernoulliChokeModel(K_c=choke.K_c, chk_profile=choke.chk_profile)
    raise TypeError(f"Unsupported choke model: {name}")


def _to_native_wp_bc(wp, bc):
    """Convert manywells WellProperties/BoundaryConditions to manywells_rs objects.

    Objects that are already the Rust types are passed through unchanged.
    """
    if isinstance(wp, WellProperties):
        rwp = wp
    else:
        rwp = WellProperties(
            L=wp.L, D=wp.D, rho_l=wp.rho_l, R_s=wp.R_s, cp_g=wp.cp_g, cp_l=wp.cp_l,
            f_D=wp.f_D, h=wp.h,
            inflow=_to_native_inflow(wp.inflow), choke=_to_native_choke(wp.choke),
        )

    if isinstance(bc, BoundaryConditions):
        rbc = bc
    else:
        rbc = BoundaryConditions(p_r=bc.p_r, p_s=bc.p_s, T_r=bc.T_r, T_s=bc.T_s, u=bc.u, w_lg=bc.w_lg)

    return rwp, rbc


class SSDFSimulator:
    """Drop-in replacement for manywells.simulator.SSDFSimulator backed by Rust.

    Holds the original ``wp``/``bc`` (which may be manywells or manywells_rs
    objects) and rebuilds the underlying Rust simulator on each ``simulate()``,
    so in-place mutation of ``wp``/``bc`` between calls is picked up. ``x_guess``
    is accepted for interface parity but ignored (the shooting method uses its
    own initial guess).
    """

    dim_x = 7  # state: [p, v_g, v_l, alpha, rho_g, rho_l, T]

    def __init__(self, well_properties, boundary_conditions, n_cells: int = 100):
        self.wp = well_properties
        self.bc = boundary_conditions
        self.n_cells = n_cells
        self.x_guess = None
        self._sim = None

    @property
    def delta_z(self) -> float:
        return self.wp.L / self.n_cells

    def _build(self):
        rwp, rbc = _to_native_wp_bc(self.wp, self.bc)
        self._sim = _NativeSSDFSimulator(rwp, rbc, self.n_cells)
        return self._sim

    def _ensure(self):
        return self._sim if self._sim is not None else self._build()

    def _residual(self, p0):
        return self._ensure()._residual(p0)

    def simulate(self):
        """Return a list of steady-state solutions (>= 1), highest bottomhole
        pressure first. Raises SimError if no valid solution exists."""
        try:
            return self._build().simulate()
        except _NativeSimError as e:
            if SimError is _NativeSimError:
                raise
            raise SimError(str(e)) from e

    def solution_as_df(self, x):
        return self._ensure().solution_as_df(x)

    # def _simulate_inner(self, p0):
    #     return self._ensure()._simulate_inner(p0)

    def _right_boundary_eqs(self, x):
        return self._ensure()._right_boundary_eqs(x)
