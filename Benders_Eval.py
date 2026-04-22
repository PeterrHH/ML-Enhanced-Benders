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

        # if "t_iter_wall" in df.columns:
        #     df["t_iter_wall"] = pd.to_numeric(df["t_iter_wall"], errors="coerce").fillna(df["t_master"] + df["t_sub"])
        # else:
        df["t_iter_wall"] = df["t_master"] + df["t_sub"]

        df["cum_wall_time"] = df["t_iter_wall"].cumsum()
        df["investment_tuple"] = df["investment"].apply(investment_to_tuple)

        if len(df) <= 1:
            df["iter_norm"] = 1.0
        else:
            df["iter_norm"] = df["iter"] / float(df["iter"].max())

        logs[sid] = df

    return logs


def common_prefix_length(a: List[Any], b: List[Any]) -> int:
    n = min(len(a), len(b))
    k = 0
    for i in range(n):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k


def first_divergence(a: List[Any], b: List[Any]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def first_index_of(seq: List[Any], value: Any) -> Optional[int]:
    for i, x in enumerate(seq):
        if x == value:
            return i
    return None


def get_refinement_start(df: pd.DataFrame) -> Tuple[Optional[int], Optional[Tuple[float, ...]]]:
    flags = df["exact_mode"].astype(bool).tolist()
    invs = df["investment_tuple"].tolist()

    seen_false = False
    for i, flag in enumerate(flags):
        if not flag:
            seen_false = True
        elif flag and seen_false:
            return int(df.iloc[i]["iter"]), invs[i]

    return None, None


def area_under_curve(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    if len(y) == 0:
        return np.nan
    if len(y) == 1:
        return float(y[0])
    if x is None:
        x = np.arange(len(y))
    return float(np.trapz(y, x))


def first_hit_threshold(df: pd.DataFrame, col: str, thresholds: List[float]) -> Dict[str, Any]:
    out = {}
    for thr in thresholds:
        sub = df[df[col] <= thr]
        out[f"{col}_first_le_{thr}_iter"] = None if sub.empty else int(sub.iloc[0]["iter"])
        out[f"{col}_first_le_{thr}_time"] = None if sub.empty else float(sub.iloc[0]["cum_wall_time"])
    return out


def summarize_run(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    last = df.iloc[-1]

    res = {
        f"{prefix}_total_iterations": int(last["iter"]) + 1,
        f"{prefix}_final_iter": int(last["iter"]),
        f"{prefix}_final_UB": float(last["UB"]),
        f"{prefix}_final_LB": float(last["LB"]),
        f"{prefix}_final_gap_abs": float(last["gap_abs"]),
        f"{prefix}_final_gap_rel": float(last["gap_rel"]),
        f"{prefix}_final_investment": list(last["investment_tuple"]) if last["investment_tuple"] is not None else None,
        f"{prefix}_total_t_master": float(df["t_master"].sum()),
        f"{prefix}_total_t_sub": float(df["t_sub"].sum()),
        f"{prefix}_total_t_wall": float(df["t_iter_wall"].sum()),
        f"{prefix}_gap_abs_auc_iter": area_under_curve(df["gap_abs"].to_numpy(), df["iter"].to_numpy()),
        f"{prefix}_gap_rel_auc_iter": area_under_curve(df["gap_rel"].to_numpy(), df["iter"].to_numpy()),
        f"{prefix}_LB_auc_iter": area_under_curve(df["LB"].to_numpy(), df["iter"].to_numpy()),
        f"{prefix}_gap_abs_auc_time": area_under_curve(df["gap_abs"].to_numpy(), df["cum_wall_time"].to_numpy()),
        f"{prefix}_gap_rel_auc_time": area_under_curve(df["gap_rel"].to_numpy(), df["cum_wall_time"].to_numpy()),
        f"{prefix}_LB_auc_time": area_under_curve(df["LB"].to_numpy(), df["cum_wall_time"].to_numpy()),
    }

    thresholds = [0.5, 0.2, 0.1, 0.05, 0.01]
    prefixed_hits = first_hit_threshold(df, "gap_rel", thresholds)
    prefixed_hits = {f"{prefix}_{k}": v for k, v in prefixed_hits.items()}
    res.update(prefixed_hits)

    best_ub_idx = df["UB"].idxmin()
    res[f"{prefix}_best_UB"] = float(df.loc[best_ub_idx, "UB"])
    res[f"{prefix}_best_UB_iter"] = int(df.loc[best_ub_idx, "iter"])
    res[f"{prefix}_best_UB_time"] = float(df.loc[best_ub_idx, "cum_wall_time"])

    return res


def classify_sample(row: pd.Series, other_name: str) -> str:
    same_final = bool(row["same_final_investment"])
    diverged = row["first_divergence_iter"] is not None
    exact_iters = row["exact_total_iterations"]
    other_iters = row[f"{other_name}_total_iterations"]

    if same_final and diverged and other_iters < exact_iters:
        return f"different_path_{other_name.lower()}_faster_same_final"
    if same_final and diverged and other_iters > exact_iters:
        return "different_path_exact_faster_same_final"
    if same_final and not diverged:
        if other_iters < exact_iters:
            return f"same_path_{other_name.lower()}_faster"
        if other_iters > exact_iters:
            return "same_path_exact_faster"
        return "same_path_same_speed"
    if (not same_final) and other_iters < exact_iters:
        return f"different_final_{other_name.lower()}_faster"
    if (not same_final) and other_iters > exact_iters:
        return "different_final_exact_faster"
    return "other"


def compare_sample(sample_id: int, exact_df: pd.DataFrame, other_df: pd.DataFrame, other_name: str) -> Dict[str, Any]:
    row = {"sample": sample_id}
    row.update(summarize_run(exact_df, "exact"))
    row.update(summarize_run(other_df, other_name))

    exact_inv = exact_df["investment_tuple"].tolist()
    other_inv = other_df["investment_tuple"].tolist()

    row["common_prefix_len"] = common_prefix_length(exact_inv, other_inv)
    row["first_divergence_iter"] = first_divergence(exact_inv, other_inv)
    row["same_final_investment"] = exact_inv[-1] == other_inv[-1]

    refine_iter, refine_inv = get_refinement_start(other_df)
    row[f"{other_name}_refine_start_iter"] = refine_iter
    row[f"{other_name}_refine_start_investment"] = list(refine_inv) if refine_inv is not None else None

    if refine_inv is not None:
        exact_hit = first_index_of(exact_inv, refine_inv)
        row["exact_hits_other_refine_investment"] = exact_hit is not None
        row["exact_hits_other_refine_investment_iter"] = exact_hit
    else:
        row["exact_hits_other_refine_investment"] = None
        row["exact_hits_other_refine_investment_iter"] = None

    other_flags = other_df["exact_mode"].astype(bool)
    row[f"{other_name}_exact_iterations_logged"] = int(other_flags.sum())
    row[f"{other_name}_inexact_iterations_logged"] = int((~other_flags).sum())

    row[f"delta_iterations_{other_name}_minus_exact"] = row[f"{other_name}_total_iterations"] - row["exact_total_iterations"]
    row[f"delta_time_{other_name}_minus_exact"] = row[f"{other_name}_total_t_wall"] - row["exact_total_t_wall"]
    row[f"delta_gap_rel_{other_name}_minus_exact"] = row[f"{other_name}_final_gap_rel"] - row["exact_final_gap_rel"]
    row["sample_class"] = classify_sample(pd.Series(row), other_name)

    return row


def resample_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, x, y)


def build_shared_grid(exact_logs: Dict[int, pd.DataFrame],
                      other_logs: Dict[int, pd.DataFrame],
                      x_mode: str,
                      grid_size: int = 300) -> np.ndarray:
    if x_mode == "iter_raw":
        max_iter_exact = max(int(df["iter"].max()) for df in exact_logs.values())
        max_iter_other = max(int(df["iter"].max()) for df in other_logs.values())
        max_iter = max(max_iter_exact, max_iter_other)
        return np.arange(max_iter + 1, dtype=float)

    if x_mode == "iter_norm":
        return np.linspace(0.0, 1.0, grid_size)

    if x_mode == "time_norm":
        return np.linspace(0.0, 1.0, grid_size)

    if x_mode == "time_raw":
        max_time_exact = max(float(df["cum_wall_time"].iloc[-1]) for df in exact_logs.values())
        max_time_other = max(float(df["cum_wall_time"].iloc[-1]) for df in other_logs.values())
        max_time = max(max_time_exact, max_time_other)
        return np.linspace(0.0, max_time, grid_size)

    raise ValueError(f"Unknown x_mode: {x_mode}")


def aggregate_curves(logs: Dict[int, pd.DataFrame],
                     x_mode: str,
                     y_col: str,
                     grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    curves = []

    for df in logs.values():
        if x_mode == "iter_raw":
            x = df["iter"].to_numpy(dtype=float)

        elif x_mode == "iter_norm":
            x = df["iter_norm"].to_numpy(dtype=float)

        elif x_mode == "time_norm":
            x = df["cum_wall_time"].to_numpy(dtype=float)
            xmax = x[-1] if len(x) > 0 else 0.0
            x = np.zeros_like(x) if xmax <= 0 else x / xmax

        elif x_mode == "time_raw":
            x = df["cum_wall_time"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Unknown x_mode: {x_mode}")

        y = df[y_col].to_numpy(dtype=float)
        curves.append(resample_curve(x, y, grid))

    arr = np.vstack(curves)
    mean_curve = np.nanmean(arr, axis=0)
    median_curve = np.nanmedian(arr, axis=0)
    return mean_curve, median_curve


def find_useful_xmax(grid: np.ndarray,
                     curves: List[np.ndarray],
                     threshold: float,
                     buffer_frac: float = 0.05,
                     min_x: float = 0.1) -> float:
    arr = np.vstack(curves)
    below = np.all(arr <= threshold, axis=0)

    idx = None
    for i, flag in enumerate(below):
        if flag:
            idx = i
            break

    if idx is None:
        return float(grid[-1])

    xmax = float(grid[idx])
    full_span = float(grid[-1] - grid[0])
    xmax += buffer_frac * full_span
    xmax = min(float(grid[-1]), xmax)
    xmax = max(min_x, xmax)
    return xmax


def plot_aggregated(exact_logs: Dict[int, pd.DataFrame],
                    other_logs: Dict[int, pd.DataFrame],
                    other_name: str,
                    figures_dir: Path):
    configs = [
        ("gap_rel", "iter_raw", "Relative duality gap", "gap_rel_vs_iter.png"),
        ("gap_abs", "iter_raw", "Absolute duality gap", "gap_abs_vs_iter.png"),
        ("LB", "iter_raw", "Lower bound", "lb_vs_iter.png"),

        ("gap_rel", "iter_norm", "Relative duality gap", "gap_rel_vs_iter_norm.png"),
        ("gap_abs", "iter_norm", "Absolute duality gap", "gap_abs_vs_iter_norm.png"),
        ("LB", "iter_norm", "Lower bound", "lb_vs_iter_norm.png"),

        ("gap_rel", "time_norm", "Relative duality gap", "gap_rel_vs_time_norm.png"),
        ("gap_abs", "time_norm", "Absolute duality gap", "gap_abs_vs_time_norm.png"),
        ("LB", "time_norm", "Lower bound", "lb_vs_time_norm.png"),

        ("gap_rel", "time_raw", "Relative duality gap", "gap_rel_vs_time_raw.png"),
    ]

    for y_col, x_mode, ylabel, filename in configs:
        grid = build_shared_grid(exact_logs, other_logs, x_mode, grid_size=200)
        ex_mean, ex_median = aggregate_curves(exact_logs, x_mode, y_col, grid)
        ot_mean, ot_median = aggregate_curves(other_logs, x_mode, y_col, grid)

        # full plot
        plt.figure(figsize=(8, 5))
        plt.plot(grid, ex_mean, label="Exact mean", linewidth=2)
        plt.plot(grid, ex_median, label="Exact median", linewidth=2, linestyle="--")
        plt.plot(grid, ot_mean, label=f"{other_name} mean", linewidth=2)
        plt.plot(grid, ot_median, label=f"{other_name} median", linewidth=2, linestyle="--")

        if x_mode == "iter_raw":
            plt.xlabel("Iteration")
        elif x_mode == "iter_norm":
            plt.xlabel("Normalized iteration")
        else:
            plt.xlabel("Normalized cumulative time")

        plt.ylabel(ylabel)
        plt.title(f"{ylabel} evolution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
        plt.close()

        # cropped plots for gap
        if y_col in {"gap_abs", "gap_rel"}:
            plt.figure(figsize=(8, 5))
            plt.plot(grid, ex_mean, label="Exact mean", linewidth=2)
            plt.plot(grid, ex_median, label="Exact median", linewidth=2, linestyle="--")
            plt.plot(grid, ot_mean, label=f"{other_name} mean", linewidth=2)
            plt.plot(grid, ot_median, label=f"{other_name} median", linewidth=2, linestyle="--")

            if y_col == "gap_rel":
                plt.yscale("log")

            if x_mode == "iter_raw":
                plt.xlabel("Iteration")
            elif x_mode == "iter_norm":
                plt.xlabel("Normalized iteration")
            elif x_mode == "time_norm":
                plt.xlabel("Normalized cumulative time")
            else:
                plt.xlabel("Normalized cumulative time")

            plt.ylabel(ylabel)
            plt.title(f"{ylabel} evolution (cropped)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            if y_col == "gap_abs":
                xmax = find_useful_xmax(
                    grid,
                    [ex_mean, ex_median, ot_mean, ot_median],
                    threshold=1e6,
                    buffer_frac=0.05,
                    min_x=0.1,
                )
            else:
                xmax = find_useful_xmax(
                    grid,
                    [ex_mean, ex_median, ot_mean, ot_median],
                    threshold=0.01,
                    buffer_frac=0.05,
                    min_x=0.1,
                )

            plt.xlim(grid[0], xmax)
            plt.tight_layout()
            cropped_name = filename.replace(".png", "_cropped.png")
            plt.savefig(figures_dir / cropped_name, dpi=200, bbox_inches="tight")
            plt.close()


def plot_per_sample_overlays(exact_logs: Dict[int, pd.DataFrame],
                             other_logs: Dict[int, pd.DataFrame],
                             other_name: str,
                             figures_dir: Path,
                             max_samples: int = 10):
    overlay_dir = figures_dir / "sample_overlays"
    ensure_dir(overlay_dir)

    samples = sorted(set(exact_logs.keys()) & set(other_logs.keys()))[:max_samples]

    for s in samples:
        edf = exact_logs[s]
        odf = other_logs[s]

        plt.figure(figsize=(9, 5))
        plt.plot(edf["iter"], edf["gap_rel"], marker="o", label="Exact")
        plt.plot(odf["iter"], odf["gap_rel"], marker="o", label=other_name)
        plt.xlabel("Iteration")
        plt.ylabel("Relative gap")
        plt.title(f"Sample {s}: relative gap")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_dir / f"sample_{s}_gap_rel.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(edf["iter"], edf["LB"], marker="o", label="Exact")
        plt.plot(odf["iter"], odf["LB"], marker="o", label=other_name)
        plt.xlabel("Iteration")
        plt.ylabel("Lower bound")
        plt.title(f"Sample {s}: LB progression")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_dir / f"sample_{s}_lb.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(edf["cum_wall_time"], edf["gap_rel"], marker="o", label="Exact")
        plt.plot(odf["cum_wall_time"], odf["gap_rel"], marker="o", label=other_name)
        plt.xlabel("Cumulative time (s)")
        plt.ylabel("Relative gap")
        plt.title(f"Sample {s}: relative gap vs time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_dir / f"sample_{s}_gap_rel_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close()

def get_mean_crossover_point(
    logs: Dict[int, pd.DataFrame],
    y_col: str = "gap_rel",
) -> Optional[Tuple[float, float]]:
    """
    Return the mean crossover point across samples.

    Crossover = first iteration where exact_mode switches from False to True.
    Runs that start already in exact mode have no crossover.
    """
    cross_times = []
    cross_vals = []

    for df in logs.values():
        if len(df) < 2:
            continue

        exact_flags = df["exact_mode"].astype(bool).to_numpy()

        # If the run starts in exact mode, there is no crossover
        if exact_flags[0]:
            continue

        # Find first False -> True transition
        cross_idx = None
        for i in range(1, len(exact_flags)):
            if (not exact_flags[i - 1]) and exact_flags[i]:
                cross_idx = i
                break

        if cross_idx is None:
            continue

        row = df.iloc[cross_idx]
        t_cross = float(row["cum_wall_time"])
        y_cross = float(row[y_col])

        if np.isfinite(t_cross) and np.isfinite(y_cross) and y_cross > 0:
            cross_times.append(t_cross)
            cross_vals.append(y_cross)

    if not cross_times:
        return None

    return float(np.mean(cross_times)), float(np.mean(cross_vals))


def plot_mean_gap_vs_time(
    method_logs: Dict[str, Dict[int, pd.DataFrame]],
    figures_dir: Path,
    filename: str = "mean_relative_duality_gap_vs_time.png",
    grid_size: int = 400,
    title: str = "Mean relative duality gap vs time (averaged across all samples)",
    add_crossover_dot: bool = True,
):
    """
    method_logs:
        {
            "Exact Benders": exact_logs,
            "D_CAB": dCAB_logs,
            "D_Uniform": dUniform_logs,
            ...
        }

    Style:
      - same method = same color
      - mean = solid
      - median = dashed
      - crossover = large dot on the mean curve
    """
    ensure_dir(figures_dir)

    # global time horizon
    max_time = 0.0
    for logs in method_logs.values():
        for df in logs.values():
            if len(df) > 0:
                max_time = max(max_time, float(df["cum_wall_time"].iloc[-1]))

    grid = np.linspace(0.0, max_time, grid_size)

    plt.figure(figsize=(10, 6))

    # use deterministic colors from matplotlib default cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (method_name, logs) in enumerate(method_logs.items()):
        color = color_cycle[idx % len(color_cycle)]

        mean_curve, median_curve = aggregate_curves(
            logs=logs,
            x_mode="time_raw",
            y_col="gap_rel",
            grid=grid,
        )

        # same method, same color
        plt.plot(
            grid,
            mean_curve,
            color=color,
            linewidth=2.6,
            linestyle="-",
            label=f"{method_name} (mean)",
        )
        plt.plot(
            grid,
            median_curve,
            color=color,
            linewidth=2.2,
            linestyle="--",
            alpha=0.95,
            label=f"{method_name} (median)",
        )

        # crossover dot on mean curve
        if add_crossover_dot:
            cross_pt = get_mean_crossover_point(logs, y_col="gap_rel")
            if cross_pt is not None:
                cross_time, _ = cross_pt

                # put dot on the plotted mean curve value at that time
                y_dot = np.interp(cross_time, grid, mean_curve)

                if np.isfinite(y_dot) and y_dot > 0:
                    plt.scatter(
                        [cross_time],
                        [y_dot],
                        s=140,
                        color=color,
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=10,
                    )

                    # optional small annotation
                    plt.annotate(
                        f"{method_name}\n{cross_time:.2f}s",
                        xy=(cross_time, y_dot),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=9,
                        color=color,
                    )

    plt.yscale("log")
    plt.xlabel("Cumulative time (s)")
    plt.ylabel(r"$\log((|UB-LB|)/|UB|)$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()

def make_aggregate_stats(summary_df: pd.DataFrame, other_name: str) -> pd.DataFrame:
    cols = [
        "exact_total_iterations",
        f"{other_name}_total_iterations",
        f"{other_name}_exact_iterations_logged",
        f"{other_name}_inexact_iterations_logged",
        "exact_total_t_wall",
        f"{other_name}_total_t_wall",
        "exact_final_gap_rel",
        f"{other_name}_final_gap_rel",
        "common_prefix_len",
    ]

    rows = []
    for c in cols:
        vals = pd.to_numeric(summary_df[c], errors="coerce")
        rows.append(
            {
                "metric": c,
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
            }
        )
    return pd.DataFrame(rows)


def main(default_exact_dir="outputs/Benders/3Node/Sample_120/iter_logs_exact_0",
         default_other_dir="outputs/Benders/3Node/Sample_120/iter_logs_inexact_refine_ConstD_1",
         default_other_name="D_CAB",
         default_out_dir="comparison_exact_vs_d1"):
    default_figures_dir = None
    default_max_overlay_samples = 10

    parser = argparse.ArgumentParser()

    parser.add_argument("--exact_dir", type=str, default=default_exact_dir,
                        help="Folder with exact iterlog csvs")
    parser.add_argument("--other_dir", type=str, default=default_other_dir,
                        help="Folder with other iterlog csvs")
    parser.add_argument("--other_name", type=str, default=default_other_name,
                        help="Name of the second method")
    parser.add_argument("--out_dir", type=str, default=default_out_dir,
                        help="Directory to save csv analysis")
    parser.add_argument("--figures_dir", type=str, default=default_figures_dir,
                        help="Directory to save figures; default = out_dir/figures")
    parser.add_argument("--max_overlay_samples", type=int, default=default_max_overlay_samples,
                        help="Number of sample overlays to save")

    args = parser.parse_args()

    exact_dir = Path(args.exact_dir)
    other_dir = Path(args.other_dir)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_dir / "figures"

    ensure_dir(out_dir)
    ensure_dir(figures_dir)

    exact_logs = load_logs(exact_dir)
    other_logs = load_logs(other_dir)

    common_samples = sorted(set(exact_logs.keys()) & set(other_logs.keys()))
    if not common_samples:
        raise ValueError(
            f"No common sample IDs found between folders:\n"
            f"  exact_dir = {exact_dir}\n"
            f"  other_dir = {other_dir}"
        )

    rows = []
    for sid in common_samples:
        rows.append(compare_sample(sid, exact_logs[sid], other_logs[sid], args.other_name))

    summary_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    summary_df.to_csv(out_dir / "per_sample_summary.csv", index=False)

    agg_df = make_aggregate_stats(summary_df, args.other_name)
    agg_df.to_csv(out_dir / "aggregate_summary_stats.csv", index=False)

    class_counts = (
        summary_df["sample_class"]
        .value_counts(dropna=False)
        .rename_axis("sample_class")
        .reset_index(name="count")
    )
    class_counts.to_csv(out_dir / "sample_class_counts.csv", index=False)

    wins_df = pd.DataFrame(
        [
            {
                "n_samples": len(summary_df),
                "other_fewer_iterations_count": int(
                    (summary_df[f"{args.other_name}_total_iterations"] < summary_df["exact_total_iterations"]).sum()
                ),
                "exact_fewer_iterations_count": int(
                    (summary_df[f"{args.other_name}_total_iterations"] > summary_df["exact_total_iterations"]).sum()
                ),
                "same_iterations_count": int(
                    (summary_df[f"{args.other_name}_total_iterations"] == summary_df["exact_total_iterations"]).sum()
                ),
                "other_faster_wall_time_count": int(
                    (summary_df[f"{args.other_name}_total_t_wall"] < summary_df["exact_total_t_wall"]).sum()
                ),
                "exact_faster_wall_time_count": int(
                    (summary_df[f"{args.other_name}_total_t_wall"] > summary_df["exact_total_t_wall"]).sum()
                ),
                "same_final_investment_count": int(summary_df["same_final_investment"].fillna(False).sum()),
                "exact_hits_other_refine_investment_count": int(
                    summary_df["exact_hits_other_refine_investment"].fillna(False).sum()
                ),
            }
        ]
    )
    wins_df.to_csv(out_dir / "win_loss_summary.csv", index=False)

    stacked = []
    for sid in common_samples:
        e = exact_logs[sid].copy()
        e["method"] = "Exact"
        o = other_logs[sid].copy()
        o["method"] = args.other_name
        stacked.append(e)
        stacked.append(o)

    pd.concat(stacked, ignore_index=True).to_csv(out_dir / "stacked_iteration_logs.csv", index=False)

    plot_aggregated(exact_logs, other_logs, args.other_name, figures_dir)
    # plot_per_sample_overlays(
    #     exact_logs,
    #     other_logs,
    #     args.other_name,
    #     figures_dir,
    #     max_samples=args.max_overlay_samples,
    # )

    meta = {
        "exact_dir": str(exact_dir),
        "other_dir": str(other_dir),
        "other_name": args.other_name,
        "n_exact_logs": len(exact_logs),
        "n_other_logs": len(other_logs),
        "n_common_samples": len(common_samples),
    }
    with open(out_dir / "comparison_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved per-sample summary to: {out_dir / 'per_sample_summary.csv'}")
    print(f"Saved aggregate stats to:    {out_dir / 'aggregate_summary_stats.csv'}")
    print(f"Saved class counts to:       {out_dir / 'sample_class_counts.csv'}")
    print(f"Saved win/loss summary to:   {out_dir / 'win_loss_summary.csv'}")
    print(f"Saved figures to:            {figures_dir}")
def get_first_crossover_point(df: pd.DataFrame):
    """
    Return the first point where exact_mode switches from False to True.

    Returns
    -------
    cross_time : float | None
    cross_lb   : float | None
    """
    flags = df["exact_mode"].astype(bool).to_numpy()

    for i in range(1, len(flags)):
        if (flags[i - 1] == False) and (flags[i] == True):
            return float(df.iloc[i]["cum_wall_time"]), float(df.iloc[i]["LB"])

    return None, None

def get_lb_star_map(exact_logs: Dict[int, pd.DataFrame]) -> Dict[int, float]:
    """
    Use the final LB from Exact Benders as LB* for each sample.
    """
    lb_star = {}
    for sid, df in exact_logs.items():
        if len(df) == 0:
            continue
        lb_star[sid] = float(df["LB"].iloc[-1])
    return lb_star


def get_first_crossover_index(df: pd.DataFrame) -> Optional[int]:
    """
    First index where exact_mode switches from False to True.
    """
    flags = df["exact_mode"].astype(bool).to_numpy()
    for i in range(1, len(flags)):
        if (flags[i - 1] == False) and (flags[i] == True):
            return i
    return None


def aggregate_lb_percent_curves(
    logs: Dict[int, pd.DataFrame],
    lb_star_map: Dict[int, float],
    grid: np.ndarray,
    x_mode: str = "time_raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate curves of 100 * LB / LB* across samples.
    """
    curves = []

    for sid, df in logs.items():
        if sid not in lb_star_map:
            continue

        lb_star = lb_star_map[sid]

        # avoid division by zero or near-zero
        if lb_star is None or abs(lb_star) < 1e-12:
            continue

        if x_mode == "time_raw":
            x = df["cum_wall_time"].to_numpy(dtype=float)
        elif x_mode == "iter_raw":
            x = df["iter"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Unsupported x_mode: {x_mode}")

        y = 100.0 * df["LB"].to_numpy(dtype=float) / lb_star
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
) -> Dict[str, Optional[float]]:
    """
    Compute mean/median crossover point in normalized LB percentage terms.
    Only for methods that actually switch from inexact to exact.
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

        t_cross = float(df.iloc[idx]["cum_wall_time"])
        lb_cross = float(df.iloc[idx]["LB"])
        lb_cross_pct = 100.0 * lb_cross / lb_star

        cross_times.append(t_cross)
        cross_vals.append(lb_cross_pct)

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

def aggregate_lb_curves(logs: Dict[int, pd.DataFrame], grid: np.ndarray):
    """
    Aggregate LB curves across samples on a shared time grid.
    """
    mean_curve, median_curve = aggregate_curves(
        logs=logs,
        x_mode="time_raw",
        y_col="LB",
        grid=grid,
    )
    return mean_curve, median_curve


def aggregate_crossover_points(logs: Dict[int, pd.DataFrame]):
    """
    Collect crossover times/LBs for all samples that have a crossover.
    """
    cross_times = []
    cross_lbs = []

    for df in logs.values():
        t, lb = get_first_crossover_point(df)
        if t is not None and lb is not None:
            cross_times.append(t)
            cross_lbs.append(lb)

    if len(cross_times) == 0:
        return {
            "mean_time": None,
            "median_time": None,
            "mean_lb": None,
            "median_lb": None,
        }

    return {
        "mean_time": float(np.mean(cross_times)),
        "median_time": float(np.median(cross_times)),
        "mean_lb": float(np.mean(cross_lbs)),
        "median_lb": float(np.median(cross_lbs)),
    }

def plot_mean_lb_percent_vs_time(
    method_logs: Dict[str, Dict[int, pd.DataFrame]],
    exact_logs: Dict[int, pd.DataFrame],
    figures_dir: Path,
    filename: str = "mean_lb_percent_vs_time.png",
    grid_size: int = 400,
    title: str = r"Lower bound progress vs time as percentage of $LB^*$",
    use_median_crossover: bool = False,
):
    """
    Plot 100 * LB / LB* vs cumulative wall time.

    Parameters
    ----------
    method_logs : dict
        Example:
        {
            "Exact Benders": exact_logs,
            "D_CAB": dCAB_logs,
            "D_Uniform": dUniform_logs,
            "Exact Benders (Kmeans)": exact_Kmeans_logs,
            "D_CAB (Kmeans)": dCAB_Kmeans_logs,
        }

    exact_logs : dict
        Exact Benders logs used to define LB* per sample.
    """
    ensure_dir(figures_dir)

    lb_star_map = get_lb_star_map(exact_logs)

    # shared time grid
    max_time = 0.0
    for logs in method_logs.values():
        for sid, df in logs.items():
            if sid in lb_star_map and len(df) > 0:
                max_time = max(max_time, float(df["cum_wall_time"].iloc[-1]))

    grid = np.linspace(0.0, max_time, grid_size)

    plt.figure(figsize=(10, 6))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (method_name, logs) in enumerate(method_logs.items()):
        color = color_cycle[idx % len(color_cycle)]

        mean_curve, median_curve = aggregate_lb_percent_curves(
            logs=logs,
            lb_star_map=lb_star_map,
            grid=grid,
            x_mode="time_raw",
        )

        # same method -> same color
        plt.plot(
            grid,
            mean_curve,
            linewidth=2.5,
            linestyle="-",
            color=color,
            label=f"{method_name} (mean)",
        )
        plt.plot(
            grid,
            median_curve,
            linewidth=2.2,
            linestyle="--",
            color=color,
            alpha=0.9,
            label=f"{method_name} (median)",
        )

        # Exact Benders has no crossover marker
        if "Exact Benders" not in method_name or "(Kmeans)" in method_name:
            # still only add a crossover if the method truly has one
            cross_stats = aggregate_crossover_lb_percent(logs, lb_star_map)

            if use_median_crossover:
                x_cross = cross_stats["median_time"]
                y_cross = cross_stats["median_val"]
            else:
                x_cross = cross_stats["mean_time"]
                y_cross = cross_stats["mean_val"]

            if x_cross is not None and y_cross is not None:
                plt.scatter(
                    x_cross,
                    y_cross,
                    s=140,
                    color=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=5,
                    label=f"{method_name} crossover",
                )

    plt.xlabel("Cumulative time (s)")
    plt.ylabel(r"$100 \cdot LB / LB^*$ (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    BASE = "outputs/Benders/3Node/Sample_120"
    exact_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact"))
    #exact_kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact_Kmeans8"))
    dCAB_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_single"))
    # dCAB_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_ConstD_1"))
    # dCAB_kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_ConstD_1_kmeans10"))
    dUniform_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_uniform_single"))
    # cap_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_Capacity-Bound-Class-FullCut"))

    dCAB_NR50 = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB-NR50_single"))
    dCAB_NR90 = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB-NR90_single"))
    

    # dCAB_Kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Inexact_Refine_D_CAB_kmeans"))
    # exact_Kmeans_logs = load_logs(Path(f"{BASE}/iter_logs_Exact_Exact_kmeans"))



    plot_mean_gap_vs_time(
        method_logs={
            "Exact Benders": exact_logs,
            # "Exact Kmeans8": exact_kmeans_logs,
            "D_CAB": dCAB_logs,
            "D_Uniform": dUniform_logs,
            # "D_CAB_NR50": dCAB_NR50,
            # "D_CAB_NR90": dCAB_NR90,
            # "Cap_Class": cap_logs,
            # "Exact Benders (Kmeans)": exact_Kmeans_logs,
            # "D_CAB (Kmeans)": dCAB_kmeans_logs,
        },
        figures_dir=Path("comparison_gap_time_figures"),
        title="Mean relative duality gap vs time (averaged across all samples), Kmeans",
    )

    plot_mean_lb_percent_vs_time(
        method_logs={
            "Exact Benders": exact_logs,
            "D_CAB": dCAB_logs,
            "D_Uniform": dUniform_logs,
        },
        exact_logs=exact_logs,
        figures_dir=Path("comparison_lb_time_figures"),
        filename="mean_lb_vs_time_with_crossover.png",
        title="Mean lower bound vs time (with crossover point)",
    )