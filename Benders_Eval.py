#!/usr/bin/env python3

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_parse_investment(x: Any):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def investment_to_tuple(x: Any, ndigits: int = 6) -> Optional[Tuple[float, ...]]:
    vals = safe_parse_investment(x)
    if vals is None:
        return None
    out = []
    for v in vals:
        try:
            out.append(round(float(v), ndigits))
        except Exception:
            out.append(v)
    return tuple(out)


def sample_id_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"sample(\d+)", path.name)
    if m:
        return int(m.group(1))
    return None


def load_logs(folder: Path) -> Dict[int, pd.DataFrame]:
    logs = {}
    for fp in sorted(folder.glob("*.csv")):
        sid = sample_id_from_filename(fp)
        if sid is None:
            continue

        df = pd.read_csv(fp).copy()

        required = ["iter", "UB", "LB", "gap_abs", "gap_rel", "exact_mode", "investment"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing columns: {missing}")

        df["sample"] = sid if "sample" not in df.columns else df["sample"]
        df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
        df["UB"] = pd.to_numeric(df["UB"], errors="coerce")
        df["LB"] = pd.to_numeric(df["LB"], errors="coerce")
        df["gap_abs"] = pd.to_numeric(df["gap_abs"], errors="coerce")
        df["gap_rel"] = pd.to_numeric(df["gap_rel"], errors="coerce")

        if "t_master" in df.columns:
            df["t_master"] = pd.to_numeric(df["t_master"], errors="coerce").fillna(0.0)
        else:
            df["t_master"] = 0.0

        if "t_sub" in df.columns:
            df["t_sub"] = pd.to_numeric(df["t_sub"], errors="coerce").fillna(0.0)
        else:
            df["t_sub"] = 0.0

        if "t_iter_wall" in df.columns:
            df["t_iter_wall"] = pd.to_numeric(df["t_iter_wall"], errors="coerce").fillna(df["t_master"] + df["t_sub"])
        else:
            df["t_iter_wall"] = df["t_master"] + df["t_sub"]

        df["cum_wall_time"] = df["t_iter_wall"].cumsum()
        df["investment_tuple"] = df["investment"].apply(investment_to_tuple)

        if len(df) <= 1:
            df["iter_norm"] = 1.0
        else:
            df["iter_norm"] = df["iter"] / float(df["iter"].max())

        logs[sid] = df

    return logs


def resample_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, x, y)


def _ensure_fair_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "t_master" not in df.columns:
        df["t_master"] = 0.0
    if "t_sub" not in df.columns:
        df["t_sub"] = 0.0

    df["t_master"] = pd.to_numeric(df["t_master"], errors="coerce").fillna(0.0)
    df["t_sub"] = pd.to_numeric(df["t_sub"], errors="coerce").fillna(0.0)

    df["t_iter_fair"] = df["t_master"] + df["t_sub"]
    df["cum_fair_time"] = df["t_iter_fair"].cumsum()
    return df


def _compute_mean_median_curve_on_time_grid(
    logs: Dict[int, pd.DataFrame],
    grid: np.ndarray,
    time_col: str = "cum_fair_time",
    y_col: str = "gap_rel",
) -> Tuple[np.ndarray, np.ndarray]:
    curves = []

    for _, df0 in logs.items():
        df = _ensure_fair_time_columns(df0)

        x = df[time_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        y = np.maximum(y, 1e-16)

        if len(x) == 0:
            continue
        elif len(x) == 1:
            y_interp = np.full_like(grid, y[0], dtype=float)
        else:
            y_interp = np.interp(grid, x, y)

        curves.append(y_interp)

    if not curves:
        raise ValueError("No valid curves found for plotting.")

    arr = np.vstack(curves)
    mean_curve = np.nanmean(arr, axis=0)
    median_curve = np.nanmedian(arr, axis=0)
    return mean_curve, median_curve


def _find_xmax_from_y_threshold(
    grid: np.ndarray,
    curves: List[np.ndarray],
    y_threshold: float = 1e-6,
    buffer_frac: float = 0.03,
) -> float:
    arr = np.vstack(curves)
    mask = np.all(arr <= y_threshold, axis=0)

    idx = None
    for i, ok in enumerate(mask):
        if ok:
            idx = i
            break

    if idx is None:
        return float(grid[-1])

    xmax = float(grid[idx])
    span = float(grid[-1] - grid[0])
    xmax = min(float(grid[-1]), xmax + buffer_frac * span)
    return xmax


def _get_crossover_points(
    logs: Dict[int, pd.DataFrame],
    time_col: str = "cum_fair_time",
    y_col: str = "gap_rel",
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Return:
      mean_point   = (mean crossover time, mean crossover y)
      median_point = (median crossover time, median crossover y)

    Crossover is defined as the first row where exact_mode switches False -> True.
    """
    times = []
    ys = []

    for _, df0 in logs.items():
        df = _ensure_fair_time_columns(df0)

        if "exact_mode" not in df.columns or len(df) < 2:
            continue

        flags = df["exact_mode"].astype(bool).to_numpy()
        switch_idx = None
        for i in range(1, len(flags)):
            if (not flags[i - 1]) and flags[i]:
                switch_idx = i
                break

        if switch_idx is None:
            continue

        t = float(df.iloc[switch_idx][time_col])
        y = float(df.iloc[switch_idx][y_col])
        y = max(y, 1e-16)

        times.append(t)
        ys.append(y)

    if not times:
        return None, None

    mean_point = (float(np.mean(times)), float(np.mean(ys)))
    median_point = (float(np.median(times)), float(np.median(ys)))
    return mean_point, median_point


def plot_mean_gap_vs_time(
    method_logs: Dict[str, Dict[int, pd.DataFrame]],
    figures_dir: Path,
    filename: str = "mean_relative_duality_gap_vs_time.png",
    grid_size: int = 400,
    title: str = "Mean relative duality gap vs time (averaged across all samples)",
    y_stop_threshold: float = 1e-6,
    annotate_crossover: bool = True,
    show_median: bool = True,
    log_y: bool = True,
):
    """
    Parameters
    ----------
    show_median : bool
        If True (default), plot both mean and median curves per method.
        If False, plot only the mean curves.

    log_y : bool
        If True (default), plot the y-axis on log scale (gap spans many
        orders of magnitude). If False, use a linear y-axis.
    """
    ensure_dir(figures_dir)

    time_col = "cum_wall_time"   # match LB plot
    time_col = "cum_fair_time"   # switch to fair time for all methods (t_master + t_sub)

    # make sure cum_wall_time exists
    processed_method_logs = {}
    for method_name, logs in method_logs.items():
        processed_method_logs[method_name] = {}
        for sid, df in logs.items():
            df2 = _ensure_fair_time_columns(df)

            if "cum_wall_time" not in df2.columns:
                if "t_iter_wall" in df2.columns:
                    df2["t_iter_wall"] = pd.to_numeric(
                        df2["t_iter_wall"], errors="coerce"
                    ).fillna(df2["t_master"] + df2["t_sub"])
                else:
                    df2["t_iter_wall"] = df2["t_master"] + df2["t_sub"]

                df2["cum_wall_time"] = df2["t_iter_wall"].cumsum()

            processed_method_logs[method_name][sid] = df2

    max_time = 0.0
    for logs in processed_method_logs.values():
        for df in logs.values():
            if len(df) > 0:
                max_time = max(max_time, float(df[time_col].iloc[-1]))

    grid = np.linspace(0.0, max_time, grid_size)

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(10, 6))

    all_curves = []

    for i, (method_name, logs) in enumerate(processed_method_logs.items()):
        color = cmap(i)

        mean_curve, median_curve = _compute_mean_median_curve_on_time_grid(
            logs=logs,
            grid=grid,
            time_col=time_col,
            y_col="gap_rel",
        )

        all_curves.append(mean_curve)

        # mean curve label depends on whether median is also shown
        mean_label = f"{method_name} (mean)" if show_median else method_name

        plt.plot(
            grid,
            mean_curve,
            color=color,
            linewidth=2.8,
            linestyle="-",
            label=mean_label,
        )

        if show_median:
            all_curves.append(median_curve)
            plt.plot(
                grid,
                median_curve,
                color=color,
                linewidth=2.8,
                linestyle="--",
                alpha=0.95,
                label=f"{method_name} (median)",
            )

        # annotate crossover only for inexact-refine methods
        if annotate_crossover and ("exact" not in method_name.lower()):
            mean_point, median_point = _get_crossover_points(
                logs=logs,
                time_col=time_col,
                y_col="gap_rel",
            )

            if mean_point is not None:
                # vertical line at the mean crossover time, with the gap
                # value at crossover (as a real percentage) reported in the legend
                plt.axvline(
                    x=mean_point[0],
                    color=color,
                    linestyle=":",
                    linewidth=1.8,
                    alpha=0.8,
                    zorder=4,
                    label=f"{method_name} crossover ({mean_point[1] * 100:.2f}%)",
                )

    xmax = _find_xmax_from_y_threshold(
        grid=grid,
        curves=all_curves,
        y_threshold=y_stop_threshold,
        buffer_frac=0.03,
    )

    ymax = max(curve[0] for curve in all_curves) * 1.15

    plt.xlim(0.0, xmax)
    if log_y:
        plt.ylim(y_stop_threshold, ymax)
        plt.yscale("log")
    else:
        plt.ylim(0.0, ymax)

    plt.xlabel("Cumulative time (s)")
    plt.ylabel(r"Duality gap $(|UB-LB|)/|UB|$")
    plt.title(title)
    plt.grid(True, alpha=0.3, which="both" if log_y else "major")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()


def get_lb_star_map(exact_logs: Dict[int, pd.DataFrame]) -> Dict[int, float]:
    """Use the final LB from Exact Benders as LB* for each sample."""
    lb_star = {}
    for sid, df in exact_logs.items():
        if len(df) == 0:
            continue
        lb_star[sid] = float(df["LB"].iloc[-1])
    return lb_star


def get_first_crossover_index(df: pd.DataFrame) -> Optional[int]:
    """First index where exact_mode switches from False to True."""
    flags = df["exact_mode"].astype(bool).to_numpy()
    for i in range(1, len(flags)):
        if (not flags[i - 1]) and flags[i]:
            return i
    return None


def aggregate_lb_percent_curves(
    logs: Dict[int, pd.DataFrame],
    lb_star_map: Dict[int, float],
    grid: np.ndarray,
    x_mode: str = "time_raw",
    as_optimality_gap: bool = False,
    time_col: str = "cum_fair_time",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-sample curves across a shared grid.

    Parameters
    ----------
    as_optimality_gap : bool
        If False (default): y = 100 * LB / LB*  (LB progress as % of optimum).
        If True:            y = 100 * (LB* - LB) / LB*  (optimality gap %, -> 0).
    """
    curves = []

    for sid, df in logs.items():
        if sid not in lb_star_map:
            continue

        lb_star = lb_star_map[sid]

        if lb_star is None or abs(lb_star) < 1e-12:
            continue

        # if x_mode == "time_raw":
        #     x = df["cum_wall_time"].to_numpy(dtype=float)
        if x_mode == "time_raw":
            df = _ensure_fair_time_columns(df)
            x = df[time_col].to_numpy(dtype=float)
        elif x_mode == "iter_raw":
            x = df["iter"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Unsupported x_mode: {x_mode}")

        lb_arr = df["LB"].to_numpy(dtype=float)
        if as_optimality_gap:
            y = 100.0 * (lb_star - lb_arr) / lb_star
        else:
            y = 100.0 * lb_arr / lb_star

        curves.append(resample_curve(x, y, grid))

    if len(curves) == 0:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan)

    arr = np.vstack(curves)
    mean_curve = np.nanmean(arr, axis=0)
    median_curve = np.nanmedian(arr, axis=0)
    return mean_curve, median_curve


def aggregate_crossover_lb_percent(
    logs: Dict[int, pd.DataFrame],
    lb_star_map: Dict[int, float],
    as_optimality_gap: bool = False,
    time_col: str = "cum_fair_time",
) -> Dict[str, Optional[float]]:
    """
    Mean/median crossover point in normalized terms.

    If as_optimality_gap is True, the crossover y-value is reported as
    100 * (LB* - LB) / LB* (i.e. the optimality gap at the moment of switch).
    Otherwise it is reported as 100 * LB / LB*.
    """
    cross_times = []
    cross_vals = []

    for sid, df in logs.items():
        if sid not in lb_star_map:
            continue

        lb_star = lb_star_map[sid]
        if lb_star is None or abs(lb_star) < 1e-12:
            continue

        idx = get_first_crossover_index(df)
        if idx is None:
            continue

        # t_cross = float(df.iloc[idx]["cum_wall_time"])
        df = _ensure_fair_time_columns(df)
        t_cross = float(df.iloc[idx][time_col])
        lb_cross = float(df.iloc[idx]["LB"])

        if as_optimality_gap:
            val_cross = 100.0 * (lb_star - lb_cross) / lb_star
        else:
            val_cross = 100.0 * lb_cross / lb_star

        cross_times.append(t_cross)
        cross_vals.append(val_cross)

    if len(cross_times) == 0:
        return {
            "mean_time": None,
            "mean_val": None,
            "median_time": None,
            "median_val": None,
        }

    return {
        "mean_time": float(np.mean(cross_times)),
        "mean_val": float(np.mean(cross_vals)),
        "median_time": float(np.median(cross_times)),
        "median_val": float(np.median(cross_vals)),
    }


def plot_mean_lb_percent_vs_time(
    method_logs: Dict[str, Dict[int, pd.DataFrame]],
    exact_logs: Dict[int, pd.DataFrame],
    figures_dir: Path,
    filename: str = "mean_lb_percent_vs_time.png",
    grid_size: int = 400,
    title: Optional[str] = None,
    use_median_crossover: bool = False,
    show_median: bool = True,
    plot_optimality_gap: bool = True,
    log_y: bool = True,
    y_stop_threshold: float = 1e-4,
    trim_x_at_threshold: bool = True,
):
    """
    Plot lower-bound progress or optimality gap vs cumulative fair time.

    Fair time is defined as cumulative sum of:
        t_master + t_sub

    Parameters
    ----------
    show_median : bool
        If True, plot both mean and median curves per method.
        If False, plot only mean curves.

    plot_optimality_gap : bool
        If True:
            y = 100 * (LB* - LB) / LB*
        If False:
            y = 100 * LB / LB*

    log_y : bool
        If True, use log-scale y-axis.

    y_stop_threshold : float
        Threshold used to trim the x-axis when plot_optimality_gap=True.
        For example, 1e-4 means the plot stops once all mean curves are below 1e-4.

    trim_x_at_threshold : bool
        If True, trims x-axis when all mean curves reach y_stop_threshold.
    """
    ensure_dir(figures_dir)

    time_col = "cum_fair_time"
    lb_star_map = get_lb_star_map(exact_logs)

    # Shared fair-time grid
    max_time = 0.0
    for logs in method_logs.values():
        for sid, df in logs.items():
            if sid in lb_star_map and len(df) > 0:
                df = _ensure_fair_time_columns(df)
                max_time = max(max_time, float(df[time_col].iloc[-1]))

    grid = np.linspace(0.0, max_time, grid_size)

    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_mean_curves = []

    for idx, (method_name, logs) in enumerate(method_logs.items()):
        color = color_cycle[idx % len(color_cycle)]

        mean_curve, median_curve = aggregate_lb_percent_curves(
            logs=logs,
            lb_star_map=lb_star_map,
            grid=grid,
            x_mode="time_raw",
            as_optimality_gap=plot_optimality_gap,
            time_col=time_col,
        )

        # Clip for log-scale stability
        if log_y:
            mean_curve = np.maximum(mean_curve, 1e-6)
            median_curve = np.maximum(median_curve, 1e-6)

        all_mean_curves.append(mean_curve)

        mean_label = f"{method_name} (mean)" if show_median else method_name

        plt.plot(
            grid,
            mean_curve,
            linewidth=2.5,
            linestyle="-",
            color=color,
            label=mean_label,
        )

        if show_median:
            plt.plot(
                grid,
                median_curve,
                linewidth=2.2,
                linestyle="--",
                color=color,
                alpha=0.9,
                label=f"{method_name} (median)",
            )

        # Crossover line: skip pure Exact Benders
        if "Exact Benders" not in method_name or "(Kmeans)" in method_name:
            cross_stats = aggregate_crossover_lb_percent(
                logs=logs,
                lb_star_map=lb_star_map,
                as_optimality_gap=plot_optimality_gap,
                time_col=time_col,
            )

            if use_median_crossover:
                x_cross = cross_stats["median_time"]
                y_cross = cross_stats["median_val"]
            else:
                x_cross = cross_stats["mean_time"]
                y_cross = cross_stats["mean_val"]

            if x_cross is not None and y_cross is not None:
                plt.axvline(
                    x=x_cross,
                    color=color,
                    linestyle=":",
                    linewidth=1.8,
                    alpha=0.8,
                    zorder=4,
                    label=f"{method_name} crossover ({y_cross:.2f}%)",
                )

    if trim_x_at_threshold and plot_optimality_gap and all_mean_curves:
        xmax = _find_xmax_from_y_threshold(
            grid=grid,
            curves=all_mean_curves,
            y_threshold=y_stop_threshold,
            buffer_frac=0.03,
        )
        plt.xlim(0.0, xmax)
    else:
        plt.xlim(0.0, max_time)

    plt.xlabel("Cumulative solver time (s)")

    if plot_optimality_gap:
        default_title = r"Optimality gap vs time:  $100 \cdot (LB^* - LB)/LB^*$ (%)"
        plt.ylabel(r"Optimality gap $100 \cdot (LB^* - LB)/LB^*$ (%)")
    else:
        default_title = r"Lower bound progress vs time as percentage of $LB^*$"
        plt.ylabel(r"$100 \cdot LB / LB^*$ (%)")

    if log_y:
        plt.yscale("log")
        if plot_optimality_gap:
            plt.ylim(y_stop_threshold, None)

    plt.title(title if title is not None else default_title)
    plt.grid(True, alpha=0.3, which="both" if log_y else "major")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    BASE = "outputs/Benders/3Node/Sample_120"
    exact_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact"))
    dCAB_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_single"))
    dCAB_class_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_Classify_single"))
    dUniform_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_single"))

    dCAB_NR50 = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB-NR50_single"))
    dCAB_NR90 = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB-NR90_single"))

    dCAB_Kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_kmeans"))
    exact_Kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact_kmeans"))

    dCABCap_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_Capacity_1_0_Base_single"))


    exact_full_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact_full"))
    dCAB_full_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_full"))
    dUniform_full_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_uniform_full"))
    dCABCap_full_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_Capacity_1_0_Base_full"))
    # ---- Duality gap plot: now configurable to hide median ----
    plot_mean_gap_vs_time(
        method_logs={
            "Exact Benders": exact_logs,
            "D_CAB Self-Supervised": dCAB_logs,
            # "D_Uniform": dUniform_logs,
            "D_CAB Supervised": dCAB_class_logs,
            # "D_CABCap": dCABCap_logs,
            # "Exact Full": exact_full_logs,
            # "D_CAB Full": dCAB_full_logs,
            # "D_Uniform Full": dUniform_full_logs,
            # "D_CABCap Full": dCABCap_full_logs,
        },
        figures_dir=Path("comparison_gap_time_figures"),
        title="Mean relative duality gap vs time (averaged across all samples), Single Cut",
        y_stop_threshold=1e-4,
        annotate_crossover=True,
        show_median=False,   # set True to also plot the median curves
        log_y=True,         # log scale is recommended for the gap plot because it spans many orders of magnitude
    )

    # ---- LB plot, now showing OPTIMALITY GAP (LB*-LB)/LB* on log scale ----
    plot_mean_lb_percent_vs_time(
        method_logs={
            "Exact Benders": exact_logs,
            "D_CAB Self-Supervised": dCAB_logs,
            # "D_Uniform": dUniform_logs,
            "D_CAB Supervised": dCAB_class_logs,
            # "D_CABCap": dCABCap_logs,
            # "Exact Full": exact_full_logs,
            # "D_CAB Full": dCAB_full_logs,
            # "D_Uniform Full": dUniform_full_logs,       
            # "D_CABCap Full": dCABCap_full_logs,
        },
        exact_logs=exact_logs,
        figures_dir=Path("comparison_lb_time_figures"),
        filename="mean_optimality_gap_vs_time_with_crossover.png",
        title="Mean optimality gap vs time (averaged across all samples), Single Cut",
        show_median=False,           # set True to also plot the median curves
        plot_optimality_gap=True,    # set False to recover the old 100*LB/LB* plot,
        log_y=True,                 # log scale is recommended for the optimality gap plot
        y_stop_threshold=1e-2,
        trim_x_at_threshold=True,
    )

    for method_name, logs in {"D_CAB": dCAB_logs, "D_Uniform": dUniform_logs}.items():
        switch_iters = []
        n_inexact_logged = []
        n_exact_logged = []

        for sid, df in logs.items():
            flags = df["exact_mode"].astype(bool).to_numpy()

            idx = get_first_crossover_index(df)
            if idx is not None:
                switch_iters.append(idx)

            n_inexact_logged.append(int((~flags).sum()))
            n_exact_logged.append(int(flags.sum()))

        print(f"\n{method_name}")
        print("mean switch idx:", np.mean(switch_iters))
        print("median switch idx:", np.median(switch_iters))
        print("mean inexact logged:", np.mean(n_inexact_logged))
        print("mean exact logged:", np.mean(n_exact_logged))