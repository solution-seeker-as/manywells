"""
Solution-level tests for the steady-state drift-flux simulator.

The idea: you describe a well (``WellProperties`` + ``BoundaryConditions``), say how
many steady-state solutions you expect, and (optionally) roughly where those
solutions should be. The harness runs ``simulate()`` and checks:

  1. the number of solutions matches ``expected_n_solutions``; and
  2. each solution you describe is reproduced within tolerance.

To add a test, append a ``Case`` to ``CASES`` below and fill in the details. You do
not have to pin every quantity: an ``Expect`` only checks the fields you set
(``p_bh``, ``p_wh``, ``w_l``, ``w_g``, ``w_m``), everything left as ``None`` is
ignored.

Run with:

    uv run pytest manywells_rs/tests/test_simulator.py -v

A solution is summarized by a few scalars (see ``summarize``):

  * ``p_bh`` -- bottomhole pressure, p at z = 0        (bar)
  * ``p_wh`` -- wellhead pressure, p at z = L          (bar)
  * ``w_l``  -- liquid mass rate at the wellhead       (kg/s)
  * ``w_g``  -- gas mass rate at the wellhead          (kg/s)
  * ``w_m``  -- total mass rate at the wellhead        (kg/s)

``simulate()`` returns solutions ordered highest-``p_bh`` first, so list the
entries of ``expected`` in that same order (highest bottomhole pressure first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest

import manywells_rs as mrs
from manywells_rs import SimError


# --------------------------------------------------------------------------- #
# Solution summary + expectation types
# --------------------------------------------------------------------------- #

# State layout of one cell in the flat solution list: [p, v_g, v_l, alpha, rho_g, rho_l, T]
DIM_X = 7


@dataclass
class Summary:
    """Scalar fingerprint of a single steady-state solution."""

    p_bh: float  # bottomhole pressure, p(z=0)   (bar)
    p_wh: float  # wellhead pressure,   p(z=L)   (bar)
    w_l: float   # liquid mass rate at wellhead  (kg/s)
    w_g: float   # gas mass rate at wellhead     (kg/s)
    w_m: float   # total mass rate at wellhead   (kg/s)


def summarize(sim, x) -> Summary:
    """Reduce a flat solution list ``x`` to a :class:`Summary`."""
    n = len(x) // DIM_X
    assert n * DIM_X == len(x), "solution length is not a multiple of DIM_X"

    bottom = x[0:DIM_X]
    top = x[(n - 1) * DIM_X : n * DIM_X]

    area = math.pi * (sim.wp.D / 2.0) ** 2  # cross-sectional flow area (m^2)

    # Mass rates are conserved along the well; evaluate them at the wellhead cell.
    _p, v_g, v_l, alpha, rho_g, rho_l, _T = top
    w_g = area * alpha * rho_g * v_g
    w_l = area * (1.0 - alpha) * rho_l * v_l

    return Summary(
        p_bh=bottom[0],
        p_wh=top[0],
        w_l=w_l,
        w_g=w_g,
        w_m=w_l + w_g,
    )


@dataclass
class Expect:
    """One expected solution. Only the fields you set are checked."""

    p_bh: Optional[float] = None
    p_wh: Optional[float] = None
    w_l: Optional[float] = None
    w_g: Optional[float] = None
    w_m: Optional[float] = None
    # Tolerances applied to every set field (relative OR absolute passes).
    rtol: float = 0.02
    atol: float = 1e-6


@dataclass
class Case:
    """A well + boundary conditions + what we expect ``simulate()`` to return.

    Set ``expected_n_solutions=0`` to assert that the well does not simulate
    (``simulate()`` raises ``SimError``); ``expected`` is then ignored.
    """

    name: str
    wp: object  # manywells_rs.WellProperties (or manywells equivalent)
    bc: object  # manywells_rs.BoundaryConditions (or manywells equivalent)
    expected_n_solutions: int
    expected: list[Expect] = field(default_factory=list)
    n_cells: int = 100


# --------------------------------------------------------------------------- #
# Cases -- fill in / add your own here
# --------------------------------------------------------------------------- #
CASES: list[Case] = [
    # Productivity-index inflow + Simpson choke.
    Case(
        name="pi_simpson_u0.5",
        wp=mrs.WellProperties(L=2000, D=0.1554, rho_l=850, R_s=518.3, cp_g=2225, cp_l=4180, f_D=0.05, h=20.0, inflow=mrs.ProductivityIndex(k_l=0.5, f_g=0.1379), choke=mrs.BernoulliChokeModel(K_c=0.0018966705911591126, cpr=0.544465891827854, chk_profile='linear')),
        bc=mrs.BoundaryConditions(p_r=170, p_s=20, T_r=373.15, T_s=277.15, u=0.5, w_lg=0.0),
        expected_n_solutions=2,
        expected=[
            Expect(p_bh=169.732375, p_wh=20.0),
            Expect(p_bh=130.565907, p_wh=43.087602),
        ],
    ),
    Case(
        name="vogel_simpson_u0.63",
        wp=mrs.WellProperties(L=1721.9603339481268, D=0.1397, rho_l=930.743487963018, R_s=363.4427150793591, cp_g=2225, cp_l=3156.504075437806, f_D=0.05, h=12.67595099338305, inflow=mrs.Vogel(w_l_max=106.717070619435, f_g=0.2599255592113066), choke=mrs.SimpsonChokeModel(K_c=0.0014075302875897, cpr=0.544465891827854, chk_profile='convex')),
        bc=mrs.BoundaryConditions(p_r=115.18137045160027, p_s=18.19728485632093, T_r=339.8088100184438, T_s=277.15, u=0.6323478195871421, w_lg=4.802525514813793),
        expected_n_solutions=1,
        expected=[
            Expect(p_bh=109.94, p_wh=59.77),
        ],
    ),
    Case(
        name="no solution",
        wp=mrs.WellProperties(L=3464.592636891714, D=0.0761999999999999, rho_l=972.9309195918396, R_s=355.44652560268554, cp_g=2225, cp_l=3828.209929393251, f_D=0.05, h=22.77347807279142,
            inflow=mrs.Vogel(w_l_max=24.759037189495423, f_g=0.2052894249538803),
            choke=mrs.SimpsonChokeModel(K_c=0.0009286106405242, cpr=0.544465891827854, chk_profile='linear')
        ),
        bc=mrs.BoundaryConditions(p_r=222.35772124368995, p_s=42.0459274200113, T_r=392.0877791067514, T_s=277.15, u=0.3594533898306623, w_lg=0.0),
        expected_n_solutions=0,
        expected=[],
    ),
    # --- TEMPLATE: copy this block, rename, and fill in ------------------------
    # Case(
    #     name="my_well",
    #     wp=mrs.WellProperties(
    #         # L=2000.0, D=0.1554, rho_l=850.0, R_s=518.3, cp_g=2225.0, cp_l=4180.0, f_D=0.05, h=20.0,
    #         inflow=mrs.Vogel(w_l_max=25.0, f_g=0.3),   # or mrs.ProductivityIndex(k_l=..., f_g=...)
    #         choke=mrs.BernoulliChokeModel(chk_profile="linear"),  # or mrs.SimpsonChokeModel(...)
    #     ),
    #     bc=mrs.BoundaryConditions(p_r=170.0, p_s=20.0, T_r=373.15, T_s=277.15, u=1.0, w_lg=0.0),
    #     expected_n_solutions=1,
    #     expected=[
    #         Expect(p_bh=..., w_m=...),  # set only the quantities you want to pin
    #     ],
    # ),
]


# --------------------------------------------------------------------------- #
# Test driver
# --------------------------------------------------------------------------- #

_FIELDS = ("p_bh", "p_wh", "w_l", "w_g", "w_m")


def _assert_close(case: Case, idx: int, exp: Expect, got: Summary) -> None:
    for name in _FIELDS:
        want = getattr(exp, name)
        if want is None:
            continue
        actual = getattr(got, name)
        if not math.isclose(actual, want, rel_tol=exp.rtol, abs_tol=exp.atol):
            raise AssertionError(
                f"[{case.name}] solution #{idx}: {name} = {actual:.6g}, "
                f"expected {want:.6g} (rtol={exp.rtol}, atol={exp.atol})"
            )


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_simulator_solutions(case: Case) -> None:
    sim = mrs.SSDFSimulator(case.wp, case.bc, case.n_cells)

    # A well we expect not to simulate should raise SimError.
    if case.expected_n_solutions == 0:
        with pytest.raises(SimError):
            sim.simulate()
        return

    solutions = sim.simulate()

    assert len(solutions) == case.expected_n_solutions, (
        f"[{case.name}] expected {case.expected_n_solutions} solution(s), "
        f"got {len(solutions)}"
    )

    if not case.expected:
        return  # only the count was being checked

    assert len(case.expected) == len(solutions), (
        f"[{case.name}] gave {len(case.expected)} expected solution(s) but the "
        f"solver returned {len(solutions)}; list one Expect per solution "
        f"(highest bottomhole pressure first) or leave `expected` empty to "
        f"check the count only"
    )

    # solutions are ordered highest-p_bh first; expected must follow the same order.
    summaries = [summarize(sim, x) for x in solutions]
    for idx, (exp, got) in enumerate(zip(case.expected, summaries)):
        _assert_close(case, idx, exp, got)
