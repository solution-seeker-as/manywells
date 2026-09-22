"""
Compare and benchmark the steady-state drift-flux well simulators.

This single module gathers the three comparison tools that used to live in
separate scripts:

  1. benchmark   -- time simulate() across many dataset wells for each backend
  2. convergence -- Rust vs. Ipopt wellhead (z = L) difference as N (cells) grows
  3. profile     -- compare full solution profiles/features at one operating point

Backends
--------
    simulator    manywells/simulator.py   -- SSDFSimulator (CasADi/Ipopt)
    rust         manywells_rs             -- Rust port (shooting method + implicit Euler)

CLI (run from the project root)
-------------------------------
    uv run python scripts/compare_simulators.py benchmark
    uv run python scripts/compare_simulators.py benchmark --simulators simulator rust --max-wells 20
    uv run python scripts/compare_simulators.py benchmark --simulators rust --show-failed
    uv run python scripts/compare_simulators.py convergence --wells 0 1 2 --n-cells 25 50 100 200 --save convergence_p.pdf
    uv run python scripts/compare_simulators.py profile --well 200 --save-pdf

Notebook usage
--------------
    from scripts.compare_simulators import (
        run_benchmark, load_config, run_convergence, plot_convergence, compare_operating_point,
    )

    df_config = load_config()
    results = run_convergence(df_config, well_ids=[0, 1, 2], n_cells_list=[25, 50, 100, 200])
    plot_convergence(results, variable='p')

    df_features, df_old, df_news, fig = compare_operating_point(200)
"""

import argparse
import contextlib
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running/importing as `scripts.compare_simulators` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import manywells_rs as mrs  # noqa: E402
import manywells.pvt as pvt  # noqa: E402
from manywells.simulator import SSDFSimulator, SimError  # noqa: E402
from scripts.load_well_from_dataset import load_well, load_well_config  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = DATA_ROOT / "manywells-sol" / "manywells-sol-1_config.zip"
DATA_DIR = str(DATA_ROOT / "manywells-sol")
WELL_ID = 200

# State layout shared by all simulators: x = [p, v_g, v_l, alpha, rho_g, rho_l, T].
VARIABLE_NAMES = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']
DIM_X = len(VARIABLE_NAMES)


# ===========================================================================
# Simulator registry (used by the benchmark)
# ===========================================================================

SIMULATORS: dict[str, callable] = {}


def simulator(name: str):
    """Register a simulator prepare-function under a CLI-selectable name."""

    def decorator(prepare):
        SIMULATORS[name] = prepare
        return prepare

    return decorator


@simulator("simulator")
def _prepare_simulator(well):
    """manywells/simulator.py: SSDFSimulator (CasADi/Ipopt)."""
    return SSDFSimulator(well.wp, well.bc).simulate


@simulator("rust")
def _prepare_rust(well):
    """manywells_rs: Rust port (accepts the manywells wp/bc directly)."""
    return mrs.SSDFSimulator(well.wp, well.bc).simulate


# ===========================================================================
# Benchmark: time simulate() across many wells
# ===========================================================================

def find_config_files(dataset: str | None = None) -> list[Path]:
    pattern = "**/*_config.zip" if dataset is None else f"{dataset}/*_config.zip"
    return sorted(DATA_ROOT.glob(pattern))


def _stats(times: list[float], n_success: int, n_fail: int, failed_ids: list | None = None) -> dict:
    return {
        "n_wells": len(times),
        "n_success": n_success,
        "n_fail": n_fail,
        "failed_ids": list(failed_ids) if failed_ids else [],
        "total_time": sum(times),
        "mean_time": statistics.mean(times) if times else 0.0,
        "median_time": statistics.median(times) if times else 0.0,
        "min_time": min(times) if times else 0.0,
        "max_time": max(times) if times else 0.0,
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def benchmark_config(config_path: Path, max_wells: int | None, sim_names: list[str]) -> dict[str, dict]:
    """Run every selected simulator on the same wells of one config file.

    Returns {simulator name: stats dict}. Only sim.simulate() is timed; loading,
    conversion, and construction happen outside the timer. Solver failures
    (exceptions from the timed call) are counted, not raised; stdout from the
    solvers is suppressed so prints don't pollute the timing or the report.
    """
    df_config = pd.read_csv(config_path, compression="zip")
    well_ids = sorted(df_config["ID"].unique())
    if max_wells is not None:
        well_ids = well_ids[:max_wells]

    times = {name: [] for name in sim_names}
    n_success = {name: 0 for name in sim_names}
    n_fail = {name: 0 for name in sim_names}
    failed_ids = {name: [] for name in sim_names}

    for well_id in well_ids:
        with contextlib.redirect_stdout(io.StringIO()):
            well = load_well(well_id, df_config)

        for name in sim_names:
            # Errors here (imports, conversion) are bugs, not solver failures -- let them raise.
            with contextlib.redirect_stdout(io.StringIO()):
                run = SIMULATORS[name](well)

            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run()
                n_success[name] += 1
            except Exception:
                n_fail[name] += 1
                failed_ids[name].append(well_id)
            times[name].append(time.perf_counter() - t0)

    return {name: _stats(times[name], n_success[name], n_fail[name], failed_ids[name]) for name in sim_names}


def print_config_results(config_path: Path, results: dict[str, dict], show_failed: bool = False) -> None:
    print(f"\n{config_path}")
    print(
        f"  {'simulator':<12} {'wells':>6} {'success':>8} {'fail':>6}"
        f" {'total(s)':>10} {'mean(s)':>9} {'median(s)':>10} {'max(s)':>9}"
    )
    for name, r in results.items():
        print(
            f"  {name:<12} {r['n_wells']:>6} {r['n_success']:>8} {r['n_fail']:>6}"
            f" {r['total_time']:>10.3f} {r['mean_time']:>9.4f} {r['median_time']:>10.4f} {r['max_time']:>9.4f}"
        )
    if show_failed:
        for name, r in results.items():
            if r["failed_ids"]:
                print(f"    {name} failed wells ({len(r['failed_ids'])}): {r['failed_ids']}")


def print_overall(all_results: list[dict[str, dict]], sim_names: list[str], show_failed: bool = False) -> None:
    print("\nOverall")
    totals = {}
    for name in sim_names:
        totals[name] = {
            "n_wells": sum(r[name]["n_wells"] for r in all_results),
            "n_success": sum(r[name]["n_success"] for r in all_results),
            "n_fail": sum(r[name]["n_fail"] for r in all_results),
            "total_time": sum(r[name]["total_time"] for r in all_results),
        }

    slowest = max((t["total_time"] / t["n_wells"] for t in totals.values() if t["n_wells"]), default=0.0)
    print(
        f"  {'simulator':<12} {'wells':>6} {'success':>8} {'fail':>6}"
        f" {'total(s)':>10} {'mean/well(s)':>13} {'speedup':>8}"
    )
    for name in sim_names:
        t = totals[name]
        mean = t["total_time"] / t["n_wells"] if t["n_wells"] else 0.0
        speedup = slowest / mean if mean > 0 else float("nan")
        print(
            f"  {name:<12} {t['n_wells']:>6} {t['n_success']:>8} {t['n_fail']:>6}"
            f" {t['total_time']:>10.3f} {mean:>13.4f} {speedup:>7.1f}x"
        )

    if show_failed:
        for name in sim_names:
            # De-duplicated across config files (IDs are per-dataset, so the same ID
            # in different datasets is collapsed here -- see the per-config lists above
            # for the dataset-specific breakdown).
            failed = sorted({wid for r in all_results for wid in r[name]["failed_ids"]})
            if failed:
                print(f"    {name} failed wells ({len(failed)}): {failed}")


def run_benchmark(sim_names=None, dataset=None, max_wells=None, show_failed=False) -> list[dict[str, dict]]:
    """Benchmark the selected simulators over every matching config file.

    :param sim_names: which backends to run (defaults to all registered)
    :param dataset: restrict to one dataset folder under ./data (default: all found)
    :param max_wells: cap wells per config file (default: all)
    :param show_failed: also print the well IDs each simulator failed on
    :return: list of per-config results dicts (also printed to stdout)
    """
    sim_names = list(SIMULATORS) if sim_names is None else list(sim_names)

    config_files = find_config_files(dataset)
    if not config_files:
        raise SystemExit(f"No config files found under {DATA_ROOT.resolve()}")

    print(f"Simulator benchmark: {', '.join(sim_names)}")
    print(f"Data root: {DATA_ROOT.resolve()}")
    if max_wells is not None:
        print(f"Max wells per config: {max_wells}")

    all_results = []
    for config_path in config_files:
        results = benchmark_config(config_path, max_wells, sim_names)
        all_results.append(results)
        print_config_results(config_path, results, show_failed=show_failed)

    print_overall(all_results, sim_names, show_failed=show_failed)
    return all_results


# ===========================================================================
# Convergence test: Rust vs. Ipopt wellhead difference as N grows
# ===========================================================================

def load_config(path=DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Load the well-config DataFrame from the zipped CSV."""
    return pd.read_csv(path, compression='zip')


def _primary_solution(x):
    """Normalize a simulate() result to a single flat solution list.

    The manywells simulator returns one flat list; manywells_rs returns a list of
    solutions (highest bottomhole pressure first). Return the primary solution in
    both cases.
    """
    if x and isinstance(x[0], (list, tuple, np.ndarray)):
        return x[0]
    return x


def _all_solutions(x):
    """Normalize a simulate() result to a *list* of flat solution lists.

    The manywells simulator returns one flat list; manywells_rs returns a list of
    solutions. Return a list of solutions in both cases.
    """
    if x and isinstance(x[0], (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _wellhead_state(x):
    """Extract the state at z = L (the last cell) from a flat solution list."""
    last = np.asarray(x[-DIM_X:], dtype=float)
    return dict(zip(VARIABLE_NAMES, last))


def _bottomhole_pressure(x):
    """Bottomhole pressure (p at z = 0, i.e. the first cell) of a flat solution list."""
    return float(x[0])


def _simulate(sim_factory):
    """Run a simulator built by ``sim_factory()``; return (solutions, seconds).

    ``solutions`` is a list of flat solution lists (one per steady operating point;
    the old simulator yields a single-element list, the Rust one may yield several).
    On failure returns ([], elapsed) so the caller can record NaNs and keep going.
    """
    t0 = time.perf_counter()
    try:
        sim = sim_factory()
        solutions = _all_solutions(sim.simulate())
        elapsed = time.perf_counter() - t0
        return solutions, elapsed
    except Exception:
        return [], time.perf_counter() - t0


def compare_well_convergence(well_id, df_config, n_cells_list) -> list:
    """Compare the Rust and Ipopt simulators for a single well across cell counts.

    When the Rust solver returns multiple solutions, the one whose bottomhole pressure
    is closest to the old (Ipopt) solution is used for the comparison.

    :return: list of record dicts, one per n_cells value.
    """
    wp, bc = load_well_config(well_id, df_config)

    records = []
    for n in n_cells_list:
        old_sols, t_old = _simulate(lambda: SSDFSimulator(wp, bc, n_cells=n))
        rust_sols, t_rust = _simulate(lambda: mrs.SSDFSimulator(wp, bc, n))

        old_sol = old_sols[0] if old_sols else None
        if old_sol is not None and rust_sols:
            pbh_old = _bottomhole_pressure(old_sol)
            rust_sol = min(rust_sols, key=lambda s: abs(_bottomhole_pressure(s) - pbh_old))
        else:
            rust_sol = rust_sols[0] if rust_sols else None  # no reference: highest-p0

        old_state = _wellhead_state(old_sol) if old_sol is not None else {}
        rust_state = _wellhead_state(rust_sol) if rust_sol is not None else {}

        rec = {
            'well_id': well_id,
            'n_cells': n,
            't_old': t_old,
            't_rust': t_rust,
            'old_ok': bool(old_state),
            'rust_ok': bool(rust_state),
        }
        for name in VARIABLE_NAMES:
            ov = old_state.get(name, np.nan)
            rv = rust_state.get(name, np.nan)
            rec[f'{name}_old'] = ov
            rec[f'{name}_rust'] = rv
            rec[f'{name}_absdiff'] = abs(rv - ov)
        records.append(rec)

    return records


def run_convergence(df_config, well_ids=None, n_cells_list=(25, 50, 100, 200), verbose=True) -> pd.DataFrame:
    """Run the convergence comparison over multiple wells.

    :param df_config: config DataFrame (from :func:`load_config`)
    :param well_ids: iterable of well IDs; defaults to the first 5 in the config
    :param n_cells_list: cell counts N to sweep
    :param verbose: print progress
    :return: tidy DataFrame, one row per (well_id, n_cells). Columns include
        ``<var>_old``, ``<var>_rust`` and ``<var>_absdiff`` for each state variable,
        the wellhead difference at z = L, plus per-solver wall-clock times and
        success flags.
    """
    if well_ids is None:
        well_ids = df_config['ID'].tolist()[:5]

    n_cells_list = list(n_cells_list)
    all_records = []
    for well_id in well_ids:
        if verbose:
            print(f'Well {well_id}: sweeping N in {n_cells_list} ...')
        try:
            all_records.extend(compare_well_convergence(well_id, df_config, n_cells_list))
        except Exception as exc:
            if verbose:
                print(f'  skipping well {well_id}: {type(exc).__name__}: {exc}')

    return pd.DataFrame(all_records)


def plot_convergence(results, variable='p', ax=None, savepath=None):
    """Plot the wellhead (z = L) convergence curve from :func:`run_convergence`.

    One curve per well: the absolute difference ``<variable>_absdiff`` between the
    Rust and old simulator versus N (the number of cells), on log-log axes so the
    convergence rate shows up as the slope. A dashed guide line indicating ideal
    first-order (O(1/N)) convergence is overlaid for reference.

    :param results: DataFrame returned by :func:`run_convergence`
    :param variable: state variable to plot (one of VARIABLE_NAMES), default 'p'
    :param ax: optional matplotlib Axes to draw on; a new figure is created if None
    :param savepath: if given, save the figure to this path
    :return: the matplotlib Axes
    """
    import matplotlib.pyplot as plt

    col = f'{variable}_absdiff'
    if col not in results.columns:
        raise ValueError(f'Unknown variable {variable!r}; expected one of {VARIABLE_NAMES}')

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for well_id, g in results.groupby('well_id'):
        g = g.sort_values('n_cells')
        mask = g[col].notna() & (g[col] > 0)
        if mask.any():
            ax.loglog(g.loc[mask, 'n_cells'], g.loc[mask, col], marker='o', label=f'well {well_id}')

    # First-order O(1/N) reference line, anchored to the first plotted point.
    finite = results[results[col].notna() & (results[col] > 0)]
    if not finite.empty:
        n0 = finite['n_cells'].min()
        d0 = finite.loc[finite['n_cells'] == n0, col].max()
        n_ref = np.array(sorted(results['n_cells'].unique()), dtype=float)
        ax.loglog(n_ref, d0 * n0 / n_ref, 'k--', alpha=0.6, label='O(1/N)')

    ax.set_xlabel('N (number of cells)')
    ax.set_ylabel(f'|{variable}_rust - {variable}_old| at z = L')
    ax.set_title(f'Convergence of {variable} (Rust vs. old simulator)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    if savepath is not None:
        ax.figure.savefig(savepath, bbox_inches='tight')
        print(f'Saved plot to {savepath}')

    return ax


# ===========================================================================
# Operating-point comparison: full profiles/features at one dataset row
# ===========================================================================

def apply_row_conditions(well, row):
    """
    Reconstruct the operating conditions a dataset row was generated with.

    Each data row was simulated after scripts/data_generation/well.py::sample_new_conditions
    resampled the operating point, so the config's boundary conditions alone do NOT describe
    the row. Recoverable from the row: choke opening (CHK), separator pressure (PDC), lift
    gas rate (WGL), inflow gas mass fraction (FGAS), and the oil/water mix -- which sets the
    liquid density and heat capacity via pvt.liquid_mix, as in well.py. NOT recoverable: the
    reservoir pressure p_r was resampled +/- 2% but is not stored in the row, so it stays at
    the config value -- expect a correspondingly small residual offset vs. the dataset
    (visible mainly in PBH and the rates).
    """
    well.bc.u = float(row['CHK'])
    well.bc.p_s = float(row['PDC'])
    well.bc.w_lg = float(row['WGL'])
    well.wp.inflow.f_g = float(row['FGAS'])

    f_o, f_w = float(row['FOIL']), float(row['FWAT'])
    if f_o + f_w > 0:
        liquid_mix = pvt.liquid_mix(well.oil, well.water, f_o / (f_o + f_w))
        well.wp.rho_l = liquid_mix.rho
        well.wp.cp_l = liquid_mix.cp

    return well


def features_from_profile(df_x, wp, bc):
    """
    Compute the dataset's feature columns from a cellwise solution profile, using the same
    formulas as scripts/data_generation/open_loop_stationary/generate_well_data.py.

    The oil/water split features (WOIL/WWAT/QOIL/QWAT/FOIL/FWAT) are not computed: the
    simulators only know the total liquid phase, so those would just echo the dataset's own
    sampled split back. CHK and PDC are input echoes and likewise omitted.
    """
    # Mass flow rates at the wellhead cell (constant along the well in steady state)
    w_g = float(wp.A * df_x['alpha'].iloc[-1] * df_x['rho_g'].iloc[-1] * df_x['v_g'].iloc[-1])  # Incl. lift gas
    w_l = float(wp.A * (1 - df_x['alpha'].iloc[-1]) * df_x['rho_l'].iloc[-1] * df_x['v_l'].iloc[-1])
    w_lg = bc.w_lg
    w_tot = w_g + w_l

    # Volumetric flow rates at standard reference conditions, in Sm³/h
    rho_g_std = pvt.gas_density(wp.R_s)
    SECONDS_PER_HOUR = 3600
    q_g = w_g / rho_g_std * SECONDS_PER_HOUR
    q_lg = w_lg / rho_g_std * SECONDS_PER_HOUR
    q_l = w_l / wp.rho_l * SECONDS_PER_HOUR

    pwh = float(df_x['p'].iloc[-1])

    return {
        'PBH': float(df_x['p'].iloc[0]),
        'PWH': pwh,
        'TBH': bc.T_r,
        'TWH': float(df_x['T'].iloc[-1]),
        'WGL': w_lg,
        'WGAS': w_g - w_lg,
        'WLIQ': w_l,
        'WTOT': w_tot,
        'QGL': q_lg,
        'QGAS': q_g - q_lg,
        'QLIQ': q_l,
        'QTOT': q_g + q_l,
        'FGAS': (w_g - w_lg) / (w_tot - w_lg) if w_tot > w_lg else float('nan'),
        'CHOKED': wp.choke.is_choked(pwh, bc.p_s),
        'FRBH': str(df_x['flow-regime'].iloc[0]),
        'FRWH': str(df_x['flow-regime'].iloc[-1]),
    }


def compare_operating_point(well_id, row_idx=0, data_dir=DATA_DIR, df_config=None, df_data=None, save_pdf=False):
    """
    Run the old (Ipopt) and Rust simulators on one well/operating point and plot the
    profiles side by side.

    :param well_id: Well ID in the manywells-sol dataset
    :param row_idx: Which of the well's data rows (operating points) to use
    :param data_dir: Directory holding the dataset zips (defaults to <repo>/data/manywells-sol)
    :param df_config: Preloaded config DataFrame (loaded from data_dir if None)
    :param df_data: Preloaded data DataFrame (loaded from data_dir if None)
    :param save_pdf: Also save the figure to compare_operating_point.pdf
    :return: (df_features, df_old, df_news, fig) where df_news is a list of DataFrames,
        one per Rust solution (empty if the Rust simulator failed). All solutions are
        plotted.
    """
    import matplotlib.pyplot as plt

    # -- Load config + a specific operating point (row) for this well --------
    if df_config is None:
        df_config = pd.read_csv(f'{data_dir}/manywells-sol-1_config.zip', compression='zip')
    if df_data is None:
        df_data = pd.read_csv(f'{data_dir}/manywells-sol-1.zip', compression='zip')
    row = df_data[df_data['ID'] == well_id].iloc[row_idx]

    # -- Old simulator (CasADi/Ipopt) -----------------------------------------
    well = load_well(well_id, df_config)
    well = apply_row_conditions(well, row)

    old_sim = SSDFSimulator(well.wp, well.bc)
    x_old = old_sim.simulate()
    df_old = old_sim.solution_as_df(x_old)

    # -- Rust simulator (manywells_rs) -----------------------------------------
    # manywells_rs.SSDFSimulator accepts the manywells wp/bc directly (it converts
    # them internally), so the feature formulas below can reuse well.wp/well.bc for
    # both simulators.
    new_sim = mrs.SSDFSimulator(well.wp, well.bc)
    df_news = []  # one DataFrame per Rust solution (a well can have several)
    try:
        x_news = _all_solutions(new_sim.simulate())  # all steady operating points
        df_news = [new_sim.solution_as_df(x) for x in x_news]
    except (SimError, mrs.SimError) as e:
        print(f"simerror: {e}")

    def _new_label(i):
        return f'new (rust) #{i}' if len(df_news) > 1 else 'new (rust)'

    # -- Compare features against the dataset row ------------------------------
    feat_old = features_from_profile(df_old, well.wp, well.bc)
    df_features_dict = {
        'dataset': {k: row[k] for k in feat_old},
        'old (Ipopt)': feat_old,
    }
    for i, df_new in enumerate(df_news):
        df_features_dict[_new_label(i)] = features_from_profile(df_new, well.wp, well.bc)
    df_features = pd.DataFrame(df_features_dict)

    print(f"Well {well_id}, row {row_idx}, CHK={row['CHK']:.4f}")
    print(df_features)

    # Flow regime is categorical (bubbly/slug-churn/annular); encode as an int so it can be
    # stepped through like the numeric quantities, with the y-axis labeled back to names.
    FLOW_REGIMES = ['bubbly', 'slug-churn', 'annular']
    regime_to_int = {r: i for i, r in enumerate(FLOW_REGIMES)}

    cols = ['p', 'T', 'alpha', 'v_g', 'v_l', 'rho_g']
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    # Dataset features are per-operating-point scalars at the well ends, not profiles --
    # show them as endpoint markers on the panels where they exist.
    z_ends = [df_old['z'].iloc[0], df_old['z'].iloc[-1]]
    dataset_markers = {'p': [row['PBH'], row['PWH']], 'T': [row['TBH'], row['TWH']]}

    for ax, col in zip(axes[:6], cols):
        ax.plot(df_old['z'], df_old[col], label='old (Ipopt)')
        for i, df_new in enumerate(df_news):
            ax.plot(df_new['z'], df_new[col], '--', label=_new_label(i), alpha=0.5)
        if col in dataset_markers:
            ax.plot(z_ends, dataset_markers[col], 'k*', markersize=10, linestyle='none', label='dataset')
        ax.set_xlabel('z (m)')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(alpha=0.3)

    ax_fr = axes[6]
    ax_fr.step(df_old['z'], df_old['flow-regime'].map(regime_to_int), where='post', label='old (Ipopt)')
    for i, df_new in enumerate(df_news):
        ax_fr.step(df_new['z'], df_new['flow-regime'].map(regime_to_int), '--', where='post', label=_new_label(i))
    ax_fr.plot(z_ends, [regime_to_int[row['FRBH']], regime_to_int[row['FRWH']]],
               'k*', markersize=10, linestyle='none', label='dataset')
    ax_fr.set_xlabel('z (m)')
    ax_fr.set_ylabel('flow regime')
    ax_fr.set_yticks(range(len(FLOW_REGIMES)))
    ax_fr.set_yticklabels(FLOW_REGIMES)
    ax_fr.set_ylim(-0.5, len(FLOW_REGIMES) - 0.5)
    ax_fr.legend()
    ax_fr.grid(alpha=0.3)

    axes[7].axis('off')  # unused 8th slot in the 2x4 grid

    fig.suptitle(f"Well {well_id}, CHK={row['CHK']:.3f}")
    plt.tight_layout()
    if save_pdf:
        fig.savefig('compare_operating_point.pdf')
        print("Saved comparison plot to compare_operating_point.pdf")

    return df_features, df_old, df_news, fig


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare / benchmark the well simulators (Ipopt, Rust)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Time simulate() across many dataset wells.")
    p_bench.add_argument(
        "--simulators", nargs="+", choices=sorted(SIMULATORS), default=sorted(SIMULATORS), metavar="NAME",
        help=f"Which simulators to benchmark (default: all). Choices: {', '.join(sorted(SIMULATORS))}",
    )
    p_bench.add_argument(
        "--dataset", choices=["manywells-sol", "manywells-nsol", "manywells-nscl"], default=None,
        help="Benchmark only this dataset folder (default: all found in ./data).",
    )
    p_bench.add_argument(
        "--max-wells", type=int, default=None,
        help="Maximum number of wells to simulate per config file (default: all).",
    )
    p_bench.add_argument(
        "--show-failed", action="store_true",
        help="Also print the well IDs each simulator failed on.",
    )

    p_conv = sub.add_parser("convergence", help="Rust vs. Ipopt wellhead difference as N grows.")
    p_conv.add_argument("--wells", type=int, nargs="+", default=None, help="Well IDs (default: first 3).")
    p_conv.add_argument("--n-cells", type=int, nargs="+", default=[25, 50, 100, 200], help="Cell counts N to sweep.")
    p_conv.add_argument("--save", default=None, help="Save the pressure convergence plot to this path.")

    p_prof = sub.add_parser("profile", help="Compare full profiles at one dataset operating point.")
    p_prof.add_argument("--well", type=int, default=WELL_ID, help="Well ID (default: %(default)s).")
    p_prof.add_argument("--row", type=int, default=0, help="Operating-point row index (default: 0).")
    p_prof.add_argument("--save-pdf", action="store_true", help="Save the profile plot to a PDF.")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(sim_names=args.simulators, dataset=args.dataset, max_wells=args.max_wells,
                      show_failed=args.show_failed)

    elif args.command == "convergence":
        df_config = load_config()
        well_ids = args.wells if args.wells is not None else df_config['ID'].tolist()[:3]
        results = run_convergence(df_config, well_ids=well_ids, n_cells_list=args.n_cells)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        summary_cols = ['well_id', 'n_cells', 'p_old', 'p_rust', 'p_absdiff',
                        'alpha_absdiff', 'T_absdiff', 't_old', 't_rust', 'old_ok', 'rust_ok']
        print('\nConvergence summary (difference at z = L):')
        print(results[summary_cols].to_string(index=False))

        if args.save:
            plot_convergence(results, variable='p', savepath=args.save)

    elif args.command == "profile":
        compare_operating_point(args.well, row_idx=args.row, save_pdf=args.save_pdf)


if __name__ == "__main__":
    main()
