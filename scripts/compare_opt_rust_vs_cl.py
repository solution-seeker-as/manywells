"""
Compare closed-loop optimization problem solutions: Rust black-box (solve_optimization_problem)
vs. the old ClosedLoopWellSimulator joint NLP.

Loads well configs from manywells-nscl via scripts.load_well_from_dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manywells.closed_loop.cl_simulator import ClosedLoopWellSimulator, SimError as ClSimError
from manywells_rs import SSDFSimulator
from scripts.load_well_from_dataset import load_well
from scripts.opt_with_rust_simulator import solve_optimization_problem

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CONFIG = DATA_ROOT / "manywells-nscl" / "manywells-nscl-1_config.zip"
DEFAULT_DATA = DATA_ROOT / "manywells-nscl" / "manywells-nscl-1.zip"


def _controls_from_s(s: float) -> tuple[float, float]:
    return min(s, 1.0), max(s - 1.0, 0.0)


def solve_old_cl(well, w_ref: float) -> tuple[float, float, float]:
    """Run ClosedLoopWellSimulator feedback. Returns (u, w_lg, objective)."""
    n_cells = int(well.wp.L / 10)
    sim = ClosedLoopWellSimulator(well.wp, well.bc, feedback=True, n_cells=n_cells)
    with contextlib.redirect_stdout(io.StringIO()):
        x, obj = sim.simulate(wtot_ref=w_ref, well=well)
    s = float(np.asarray(x[-1]).reshape(-1)[0])
    u, w_lg = _controls_from_s(s)
    # obj may be a CasADi DM
    f = float(np.asarray(obj).reshape(-1)[0]) if obj is not None else float("nan")
    return u, w_lg, f


def solve_rust_opt(well, w_ref: float) -> tuple[float, float]:
    """Run 1D optimization problem with the Rust simulator."""
    n_cells = int(well.wp.L / 10)
    sim = SSDFSimulator(well.wp, well.bc, n_cells=n_cells)
    return solve_optimization_problem(sim, w_ref, has_gas_lift=well.has_gas_lift)


def default_w_ref(well_id: int, df_data: pd.DataFrame | None, fallback: float) -> float:
    """Use median WTOT from dataset rows for this well, else fallback."""
    if df_data is None:
        return fallback
    rows = df_data[df_data["ID"] == well_id]
    if rows.empty:
        return fallback
    return float(rows["WTOT"].median())


def compare_wells(
    well_ids: list[int],
    df_config: pd.DataFrame,
    df_data: pd.DataFrame | None,
    w_ref: float | None,
    w_ref_fallback: float,
) -> pd.DataFrame:
    records = []
    for well_id in well_ids:
        well = load_well(well_id, df_config)
        ref = w_ref if w_ref is not None else default_w_ref(well_id, df_data, w_ref_fallback)

        rec = {
            "well_id": well_id,
            "has_gas_lift": bool(well.has_gas_lift),
            "w_ref": ref,
            "old_ok": False,
            "rust_ok": False,
            "u_old": np.nan,
            "w_lg_old": np.nan,
            "f_old": np.nan,
            "u_rust": np.nan,
            "w_lg_rust": np.nan,
            "t_old": np.nan,
            "t_rust": np.nan,
            "dt": np.nan,
            "speedup": np.nan,
            "du": np.nan,
            "dw_lg": np.nan,
        }

        t0 = time.perf_counter()
        try:
            u_old, w_lg_old, f_old = solve_old_cl(well, ref)
            rec.update(old_ok=True, u_old=u_old, w_lg_old=w_lg_old, f_old=f_old)
        except (ClSimError, RuntimeError, ValueError) as e:
            rec["old_err"] = str(e)
        rec["t_old"] = time.perf_counter() - t0

        # Reload so bc.u / bc.w_lg are not left in a mutated CasADi state
        well = load_well(well_id, df_config)

        t0 = time.perf_counter()
        try:
            u_rust, w_lg_rust = solve_rust_opt(well, ref)
            rec.update(rust_ok=True, u_rust=u_rust, w_lg_rust=w_lg_rust)
        except Exception as e:
            rec["rust_err"] = str(e)
        rec["t_rust"] = time.perf_counter() - t0

        if rec["old_ok"] and rec["rust_ok"]:
            rec["du"] = rec["u_rust"] - rec["u_old"]
            rec["dw_lg"] = rec["w_lg_rust"] - rec["w_lg_old"]
            rec["dt"] = rec["t_old"] - rec["t_rust"]
            rec["speedup"] = rec["t_old"] / rec["t_rust"] if rec["t_rust"] > 0 else float("nan")

        status = (
            "ok" if rec["old_ok"] and rec["rust_ok"]
            else "old_fail" if not rec["old_ok"] and rec["rust_ok"]
            else "rust_fail" if rec["old_ok"] and not rec["rust_ok"]
            else "both_fail"
        )
        print(
            f"well {well_id:4d}  w_ref={ref:7.3f}  [{status}]  "
            f"u: {rec['u_old']:.4f} vs {rec['u_rust']:.4f}  "
            f"w_lg: {rec['w_lg_old']:.4f} vs {rec['w_lg_rust']:.4f}  "
            f"du={rec['du']:+.4f}  dw_lg={rec['dw_lg']:+.4f}  "
            f"dt={rec['dt']:+.2f}s  speedup={rec['speedup']:.1f}x"
        )
        records.append(rec)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Rust opt vs old ClosedLoopWellSimulator on manywells-nscl.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to *_config.zip")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data zip (for w_ref)")
    parser.add_argument("--wells", type=int, nargs="+", default=None, help="Well IDs (default: first N)")
    parser.add_argument("--max-wells", type=int, default=10, help="How many wells if --wells omitted")
    parser.add_argument(
        "--w-ref", type=float, default=None,
        help="Fixed w_ref for all wells (default: median WTOT from data per well)",
    )
    parser.add_argument(
        "--w-ref-fallback", type=float, default=15.0,
        help="w_ref if data row missing (default: 15)",
    )
    parser.add_argument(
        "--gas-lift-only", action="store_true",
        help="Only compare wells with has_gas_lift=True",
    )
    args = parser.parse_args()

    df_config = pd.read_csv(args.config, compression="zip")
    df_data = pd.read_csv(args.data, compression="zip") if args.data.is_file() else None

    if args.wells is not None:
        well_ids = list(args.wells)
    else:
        well_ids = sorted(df_config["ID"].unique())
        if args.gas_lift_only:
            mask = df_config.set_index("ID")["has_gas_lift"].astype(bool)
            well_ids = [i for i in well_ids if mask.get(i, False)]
        well_ids = well_ids[: args.max_wells]

    print(f"Config: {args.config}")
    print(f"Comparing {len(well_ids)} wells  (w_ref={'fixed '+str(args.w_ref) if args.w_ref is not None else 'median WTOT'})")
    print("-" * 90)

    results = compare_wells(well_ids, df_config, df_data, args.w_ref, args.w_ref_fallback)

    both = results[results["old_ok"] & results["rust_ok"]]
    print("-" * 90)
    print(f"Both succeeded: {len(both)}/{len(results)}")
    if len(both):
        print(
            f"  |du|    mean={both['du'].abs().mean():.4f}  "
            f"median={both['du'].abs().median():.4f}  max={both['du'].abs().max():.4f}"
        )
        print(
            f"  |dw_lg| mean={both['dw_lg'].abs().mean():.4f}  "
            f"median={both['dw_lg'].abs().median():.4f}  max={both['dw_lg'].abs().max():.4f}"
        )
        print(
            f"  time    old mean={both['t_old'].mean():.2f}s  "
            f"rust mean={both['t_rust'].mean():.2f}s"
        )
        print(
            f"  dt      mean={both['dt'].mean():+.2f}s  "
            f"median={both['dt'].median():+.2f}s"
        )
        print(
            f"  speedup mean={both['speedup'].mean():.1f}x  "
            f"median={both['speedup'].median():.1f}x"
        )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    cols = [
        "well_id", "has_gas_lift", "w_ref", "old_ok", "rust_ok",
        "u_old", "u_rust", "w_lg_old", "w_lg_rust", "du", "dw_lg", "f_old",
        "t_old", "t_rust", "dt", "speedup",
    ]
    print("\n" + results[cols].to_string(index=False))


if __name__ == "__main__":
    main()
