import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Helpers ──────────────────────────────────────────────────────
def load_method_logs(folder_path):
    """Load all per-sample CSVs from a method's iter_log folder."""
    dfs = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, fname))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_per_instance_stats(df):
    """Compute summary stats per sample from iter logs."""
    stats = df.groupby('sample').agg(
        total_iters=('iter', 'max'),
        inexact_iters=('exact_mode', lambda x: (~x.astype(bool)).sum()),
        exact_iters=('exact_mode', lambda x: x.astype(bool).sum()),
    ).reset_index()

    # Total time per sample
    time_per_sample = df.groupby('sample').apply(
        lambda g: g['t_master'].sum() + g['t_sub'].sum()
    ).reset_index(name='total_time')
    stats = stats.merge(time_per_sample, on='sample')

    return stats


# ═══════════════════════════════════════════════════════════════
# PLOT 1: Duality gap vs CUMULATIVE TIME (supervisor's request)
#         Compares Exact vs Inexact on a single sample
# ═══════════════════════════════════════════════════════════════
def plot_gap_vs_time_single_sample(method_csvs, sample_id=0, figsize=(10, 6)):
    """
    Plot duality gap vs cumulative wall-clock time for multiple methods on one sample.
    
    method_csvs: dict of {method_name: csv_path} where csv is per-sample iter log
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for method_name, csv_path in method_csvs.items():
        df = pd.read_csv(csv_path).sort_values('iter')

        gap = (df['UB'] - df['LB']).to_numpy()
        gap_rel = df['gap_rel'].to_numpy() if 'gap_rel' in df.columns else gap / np.abs(df['UB'].to_numpy() + 1e-12)
        cum_time = (df['t_master'] + df['t_sub']).cumsum().to_numpy()

        # Mark inexact→exact switch
        is_exact = df['exact_mode'].astype(bool).to_numpy()
        switch_idx = None
        if not is_exact[0] and is_exact.any():
            switch_idx = np.argmax(is_exact)

        # Absolute gap vs time
        ax = axes[0]
        line, = ax.plot(cum_time, gap, marker='o', markersize=3, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(cum_time[switch_idx], linestyle='--', color=line.get_color(), alpha=0.5)

        # Relative gap vs time
        ax = axes[1]
        line, = ax.plot(cum_time, gap_rel, marker='o', markersize=3, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(cum_time[switch_idx], linestyle='--', color=line.get_color(), alpha=0.5)

    axes[0].set_xlabel('Cumulative time (s)')
    axes[0].set_ylabel('Duality gap (absolute)')
    axes[0].set_title(f'Duality gap vs time (Sample {sample_id})')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Cumulative time (s)')
    axes[1].set_ylabel('Duality gap (relative)')
    axes[1].set_title(f'Relative gap vs time (Sample {sample_id})')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def compute_switch_times(folder):
    """
    For each sample, compute cumulative wall-clock time until the first exact iteration.
    Returns a DataFrame with columns: sample, switch_time
    """
    df = load_method_logs(folder)
    results = []

    for sample, g in df.groupby('sample'):
        g = g.sort_values('iter').copy()
        g['cum_time'] = (g['t_master'] + g['t_sub']).cumsum()

        is_exact = g['exact_mode'].astype(bool).values
        if (not is_exact[0]) and is_exact.any():
            switch_idx = np.argmax(is_exact)
            switch_time = g['cum_time'].iloc[switch_idx]
            results.append({
                'sample': sample,
                'switch_time': switch_time
            })

    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════
# PLOT 2: Mean duality gap vs time AVERAGED across all samples
# ═══════════════════════════════════════════════════════════════
def plot_mean_gap_vs_time(method_folders, max_time=None, n_time_bins=200, figsize=(10, 5),
                         relative=False, save_path=None, show_median=True, show_switch_lines=False):
    """
    Average the gap-vs-time trajectory across all samples for each method.
    Uses time binning since samples have different total times.

    relative: if True, gap = (UB - LB) / max(1, |UB|); if False, gap = UB - LB
    show_median: if True, plot dashed median curve
    show_switch_lines: if True, plot vertical line at mean switch time for inexact methods
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for method_name, folder in method_folders.items():
        df = load_method_logs(folder)

        all_times = []
        all_gaps = []
        for sample, group in df.groupby('sample'):
            group = group.sort_values('iter')
            cum_time = (group['t_master'] + group['t_sub']).cumsum().values
            ub = group['UB'].values
            lb = group['LB'].values
            if relative:
                gap = (ub - lb) / np.maximum(1.0, np.abs(ub))
            else:
                gap = ub - lb
            all_times.append(cum_time)
            all_gaps.append(gap)

        method_max_time = max(t[-1] for t in all_times)
        t_max = max_time if max_time else method_max_time
        time_bins = np.linspace(0, t_max, n_time_bins)

        interp_gaps = []
        for times, gaps in zip(all_times, all_gaps):
            interp = np.interp(time_bins, times, gaps, left=gaps[0], right=gaps[-1])
            interp_gaps.append(interp)

        mean_gap = np.mean(interp_gaps, axis=0)
        median_gap = np.median(interp_gaps, axis=0)

        line, = ax.plot(time_bins, mean_gap, linewidth=2, label=f'{method_name} (mean)')

        if show_median:
            ax.plot(time_bins, median_gap, linewidth=1, linestyle='--', alpha=0.6,
                    label=f'{method_name} (median)')

        if show_switch_lines:
            switch_df = compute_switch_times(folder)
            if not switch_df.empty:
                mean_switch_time = switch_df['switch_time'].mean()
                ax.axvline(mean_switch_time, linestyle='--', alpha=0.5, color=line.get_color())
                print(f"{method_name}: mean switch time = {mean_switch_time:.4f}s")

    ax.set_xlabel('Cumulative time (s)')
    if relative:
        ax.set_ylabel('Relative duality gap  (UB − LB) / max(1, |UB|)')
        ax.set_title('Mean relative duality gap vs time')
    else:
        ax.set_ylabel('Duality gap  (UB − LB)')
        ax.set_title('Mean duality gap vs time')

    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════
# PLOT 3: Per-instance head-to-head scatter + bar comparison
#         (Inexact iters, total runtime, ordered by difficulty)
# ═══════════════════════════════════════════════════════════════
def plot_head_to_head(method_folders, method_a_name, method_b_name, figsize=(14, 10)):
    """
    Compare two methods head-to-head using:
      - scatter of inexact iterations
      - scatter of total runtime
      - per-instance difference in inexact iterations
      - per-instance difference in total runtime

    Bottom-row interpretation:
      diff = method_b - method_a
      > 0 means method_b uses more / is slower
      < 0 means method_b uses fewer / is faster
    """
    all_stats = {}
    for name, folder in method_folders.items():
        df = load_method_logs(folder)
        all_stats[name] = compute_per_instance_stats(df).set_index('sample')

    a = all_stats[method_a_name]
    b = all_stats[method_b_name]

    # Keep only common samples
    common = a.index.intersection(b.index)
    a = a.loc[common].copy()
    b = b.loc[common].copy()

    # Sort by difficulty
    if 'Exact Benders' in all_stats:
        exact = all_stats['Exact Benders']
        common = common.intersection(exact.index)
        a = a.loc[common]
        b = b.loc[common]
        exact = exact.loc[common]
        difficulty_order = exact['total_time'].sort_values().index
    else:
        difficulty_order = a['total_time'].sort_values().index
        exact = None

    # Ordered views
    a_ord = a.loc[difficulty_order]
    b_ord = b.loc[difficulty_order]

    x = np.arange(len(difficulty_order))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ── Scatter: inexact iterations ──
    ax = axes[0, 0]
    a_inex = a_ord['inexact_iters']
    b_inex = b_ord['inexact_iters']
    ax.scatter(a_inex, b_inex, alpha=0.65, edgecolors='white', linewidths=0.3)
    lim_max = max(a_inex.max(), b_inex.max()) + 2
    ax.plot([0, lim_max], [0, lim_max], 'r--', alpha=0.5, label='Equal')
    ax.set_xlabel(f'{method_a_name} inexact iters')
    ax.set_ylabel(f'{method_b_name} inexact iters')
    ax.set_title(f'Inexact iterations\n(below diagonal = {method_b_name} uses fewer)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # ── Scatter: total runtime ──
    ax = axes[0, 1]
    a_time = a_ord['total_time']
    b_time = b_ord['total_time']
    ax.scatter(a_time, b_time, alpha=0.65, edgecolors='white', linewidths=0.3)
    lim_min = min(a_time.min(), b_time.min()) * 0.9
    lim_max = max(a_time.max(), b_time.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='Equal')
    ax.set_xlabel(f'{method_a_name} total time (s)')
    ax.set_ylabel(f'{method_b_name} total time (s)')
    ax.set_title(f'Total runtime\n(below diagonal = {method_b_name} faster)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # ── Difference plot: inexact iterations ──
    ax = axes[1, 0]
    diff_inex = b_ord['inexact_iters'].values - a_ord['inexact_iters'].values
    colors = ['tab:red' if d > 0 else 'tab:green' for d in diff_inex]

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.vlines(x, 0, diff_inex, colors=colors, alpha=0.6, linewidth=1.5)
    ax.scatter(x, diff_inex, c=colors, s=28, zorder=3)
    ax.set_xlabel('Instance (sorted by difficulty →)')
    ax.set_ylabel(f'{method_b_name} - {method_a_name}')
    ax.set_title('Difference in inexact iterations\n(above 0 = more inexact iters, below 0 = fewer)')
    ax.grid(True, alpha=0.2)

    # ── Difference plot: runtime ──
    ax = axes[1, 1]
    diff_time = b_ord['total_time'].values - a_ord['total_time'].values
    colors = ['tab:red' if d > 0 else 'tab:green' for d in diff_time]

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.vlines(x, 0, diff_time, colors=colors, alpha=0.6, linewidth=1.5)
    ax.scatter(x, diff_time, c=colors, s=28, zorder=3)
    ax.set_xlabel('Instance (sorted by difficulty →)')
    ax.set_ylabel(f'{method_b_name} - {method_a_name} (s)')
    ax.set_title('Difference in total runtime\n(above 0 = slower, below 0 = faster)')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # ── Print summary ──
    diff_inex_all = b['inexact_iters'] - a['inexact_iters']
    diff_time_all = b['total_time'] - a['total_time']

    print(f"{'='*60}")
    print(f"HEAD-TO-HEAD: {method_b_name} vs {method_a_name}")
    print(f"{'='*60}")

    print(f"\nInexact iterations ({method_b_name} - {method_a_name}):")
    print(f"  {method_b_name} more:  {(diff_inex_all > 0).sum()} / {len(diff_inex_all)}")
    print(f"  {method_b_name} fewer: {(diff_inex_all < 0).sum()} / {len(diff_inex_all)}")
    print(f"  Equal:                 {(diff_inex_all == 0).sum()} / {len(diff_inex_all)}")
    print(f"  Mean diff:             {diff_inex_all.mean():.2f}")
    print(f"  Median diff:           {diff_inex_all.median():.2f}")

    print(f"\nTotal runtime ({method_b_name} - {method_a_name}):")
    print(f"  {method_b_name} slower: {(diff_time_all > 0).sum()} / {len(diff_time_all)}")
    print(f"  {method_b_name} faster: {(diff_time_all < 0).sum()} / {len(diff_time_all)}")
    print(f"  Equal:                  {(diff_time_all == 0).sum()} / {len(diff_time_all)}")
    print(f"  Mean diff:              {diff_time_all.mean():.4f}s")
    print(f"  Median diff:            {diff_time_all.median():.4f}s")

    if exact is not None:
        exact_mean = exact['total_time'].mean()
        print(f"\nExact Benders mean time: {exact_mean:.4f}s")
        print(f"{method_a_name} mean time:        {a['total_time'].mean():.4f}s  "
              f"(speedup: {(1 - a['total_time'].mean()/exact_mean)*100:.1f}%)")
        print(f"{method_b_name} mean time:        {b['total_time'].mean():.4f}s  "
              f"(speedup: {(1 - b['total_time'].mean()/exact_mean)*100:.1f}%)")

    return fig


# ═══════════════════════════════════════════════════════════════
# PLOT 4: Single-sample 2x2 diagnostic (your existing plot, enhanced)
#         Now with overlay of Exact Benders trajectory
# ═══════════════════════════════════════════════════════════════
def plot_single_sample_diagnostic(method_csvs, sample_id=0, figsize=(12, 10)):
    """
    2x2 diagnostic for a single sample, overlaying multiple methods.
    Bottom row uses RELATIVE duality gap:
        gap_rel = (UB - LB) / max(1, |UB|)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = plt.cm.tab10.colors

    for idx, (method_name, csv_path) in enumerate(method_csvs.items()):
        df = pd.read_csv(csv_path).sort_values('iter')
        color = colors[idx % len(colors)]

        iters = df['iter'].to_numpy()
        ub = df['UB'].to_numpy()
        lb = df['LB'].to_numpy()
        gap_abs = ub - lb
        gap_rel = gap_abs / np.maximum(1.0, np.abs(ub))
        cum_time = (df['t_master'] + df['t_sub']).cumsum().to_numpy()

        is_exact = df['exact_mode'].astype(bool).to_numpy()
        switch_idx = None
        if not is_exact[0] and is_exact.any():
            switch_idx = np.argmax(is_exact)

        # UB vs iteration
        ax = axes[0, 0]
        ax.plot(iters, ub, marker='o', markersize=3, color=color, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(iters[switch_idx], linestyle='--', color=color, alpha=0.4)

        # LB vs iteration
        ax = axes[0, 1]
        ax.plot(iters, lb, marker='o', markersize=3, color=color, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(iters[switch_idx], linestyle='--', color=color, alpha=0.4)

        # Relative gap vs TIME
        ax = axes[1, 0]
        ax.plot(cum_time, gap_rel, marker='o', markersize=3, color=color, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(cum_time[switch_idx], linestyle='--', color=color, alpha=0.4)

        # Relative gap vs iteration
        ax = axes[1, 1]
        ax.plot(iters, gap_rel, marker='o', markersize=3, color=color, linewidth=1.5, label=method_name)
        if switch_idx is not None:
            ax.axvline(iters[switch_idx], linestyle='--', color=color, alpha=0.4)

    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Upper bound')
    axes[0, 0].set_title('Upper bound vs iteration')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Lower bound')
    axes[0, 1].set_title('Lower bound vs iteration (cut quality)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Cumulative time (s)')
    axes[1, 0].set_ylabel('Relative duality gap')
    axes[1, 0].set_title('Relative duality gap vs time')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Relative duality gap')
    axes[1, 1].set_title('Relative duality gap vs iteration')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Benders convergence comparison (Sample {sample_id})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def compute_switch_metrics(folder):
    """
    For each sample:
    - gap at switch (relative)
    - fraction of inexact iterations
    """
    df = load_method_logs(folder)

    results = []

    for sample, g in df.groupby('sample'):
        g = g.sort_values('iter')

        is_exact = g['exact_mode'].astype(bool).values
        ub = g['UB'].values
        lb = g['LB'].values

        # Find switch index
        switch_idx = None
        if not is_exact[0] and is_exact.any():
            switch_idx = np.argmax(is_exact)

        if switch_idx is None:
            continue  # skip if no switch

        # Relative gap at switch
        gap_rel = (ub - lb) / (np.abs(ub) + 1e-12)
        gap_switch = gap_rel[switch_idx]

        # Inexact iteration ratio
        total_iters = len(g)
        inexact_iters = (~is_exact).sum()
        inexact_ratio = inexact_iters / total_iters

        results.append({
            'sample': sample,
            'gap_switch': gap_switch,
            'inexact_ratio': inexact_ratio
        })

    return pd.DataFrame(results)


def plot_switch_analysis(method_folders, figsize=(12, 5), save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    all_data = {}

    for name, folder in method_folders.items():
        stats = compute_switch_metrics(folder)
        all_data[name] = stats

        # Scatter: ratio vs gap
        axes[0].scatter(
            stats['inexact_ratio'],
            stats['gap_switch'],
            alpha=0.6,
            label=name
        )

        # Boxplot prep
        axes[1].boxplot(
            stats['gap_switch'],
            positions=[len(all_data)],
            widths=0.6
        )

    # ── Scatter plot ──
    axes[0].set_xlabel('Fraction of iterations in inexact mode')
    axes[0].set_ylabel('Relative gap at switch')
    axes[0].set_title('Switch quality vs inexact effort')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Boxplot ──
    axes[1].set_xticks(range(1, len(all_data) + 1))
    axes[1].set_xticklabels(list(all_data.keys()))
    axes[1].set_ylabel('Relative gap at switch')
    axes[1].set_title('Gap at switch distribution')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig

def print_switch_summary(method_folders):
    print("="*60)
    print("SWITCH ANALYSIS SUMMARY")
    print("="*60)

    for name, folder in method_folders.items():
        stats = compute_switch_metrics(folder)

        print(f"\n{name}:")
        print(f"  Mean gap at switch:   {stats['gap_switch'].mean():.4f}")
        print(f"  Median gap at switch: {stats['gap_switch'].median():.4f}")
        print(f"  Mean inexact ratio:   {stats['inexact_ratio'].mean():.3f}")
        print(f"  Median inexact ratio: {stats['inexact_ratio'].median():.3f}")


def plot_speedup_vs_switch_gap(exact_folder, method_folders, figsize=(12, 5), save_path=None):
    """
    For each inexact-refine method:
      x = relative gap at switch
      y = speedup vs Exact Benders on that same sample

    speedup = (exact_time - method_time) / exact_time
    so:
      > 0  => method faster than Exact Benders
      < 0  => method slower
    """
    exact_df = load_method_logs(exact_folder)
    exact_stats = compute_per_instance_stats(exact_df).set_index('sample')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    summary_rows = []

    for name, folder in method_folders.items():
        method_df = load_method_logs(folder)
        method_stats = compute_per_instance_stats(method_df).set_index('sample')
        switch_stats = compute_switch_metrics(folder).set_index('sample')

        # Keep only common samples
        common = exact_stats.index.intersection(method_stats.index).intersection(switch_stats.index)

        merged = pd.DataFrame({
            'exact_time': exact_stats.loc[common, 'total_time'],
            'method_time': method_stats.loc[common, 'total_time'],
            'gap_switch': switch_stats.loc[common, 'gap_switch'],
            'inexact_ratio': switch_stats.loc[common, 'inexact_ratio'],
        })

        merged['speedup_pct'] = (merged['exact_time'] - merged['method_time']) / (merged['exact_time'] + 1e-12)
        merged['speedup_ratio'] = merged['exact_time'] / (merged['method_time'] + 1e-12)

        # Scatter 1: speedup vs switch gap
        axes[0].scatter(
            merged['gap_switch'],
            merged['speedup_pct'],
            alpha=0.65,
            label=name
        )

        # Scatter 2: speedup vs switch gap, colored by inexact ratio via point size
        sizes = 40 + 180 * merged['inexact_ratio'].to_numpy()
        axes[1].scatter(
            merged['gap_switch'],
            merged['speedup_pct'],
            s=sizes,
            alpha=0.55,
            label=name
        )

        corr = merged[['gap_switch', 'speedup_pct']].corr().iloc[0, 1]
        summary_rows.append({
            'method': name,
            'mean_gap_switch': merged['gap_switch'].mean(),
            'std_gap_switch': merged['gap_switch'].std(ddof=1),
            'median_gap_switch': merged['gap_switch'].median(),
            'mean_speedup_pct': merged['speedup_pct'].mean(),
            'median_speedup_pct': merged['speedup_pct'].median(),
            'corr_gap_vs_speedup': corr,
        })

    # Left plot
    axes[0].axhline(0.0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Relative gap at switch')
    axes[0].set_ylabel('Speedup vs Exact Benders  (fraction)')
    axes[0].set_title('Speedup vs switch gap per instance')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right plot
    axes[1].axhline(0.0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Relative gap at switch')
    axes[1].set_ylabel('Speedup vs Exact Benders  (fraction)')
    axes[1].set_title('Same plot, point size = inexact iteration ratio')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    summary_df = pd.DataFrame(summary_rows)
    print("=" * 80)
    print("SPEEDUP VS SWITCH GAP SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")

    return fig, summary_df

# ═══════════════════════════════════════════════════════════════
# MAIN: Example usage for 3-node case
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BASE = "outputs/Benders/3Node/Sample_1752"
    SAVE_BASE = "figures/Benders"
    os.makedirs(SAVE_BASE, exist_ok=True)

    # ── Define methods with their iter_log folders ──
    iter_log_folders = {
        "Exact Benders":  f"{BASE}/iter_logs_Exact_Exact",
        r"D$_{Uniform}$":             f"{BASE}/iter_logs_Inexact_Refine_BendersBaseline",
        r"D$_{CAB}$":             f"{BASE}/iter_logs_inexact_refine_ConstD_1",
        # "D_1_sephead":  f"{BASE}/iter_logs_Inexact_Refine_ConstSeperateHead_NormbyDemand",
    }

    # ── Plot 1: Gap vs time for a specific sample ──
    sample_id = 0
    sample_csvs = {}
    for name, folder in iter_log_folders.items():
        csv = os.path.join(folder, f"iterlog_sample{sample_id}_start_exactFalse_refTrue.csv")
        # Exact Benders has a different filename pattern
        if name == "Exact Benders":
            csv = os.path.join(folder, f"iterlog_sample{sample_id}_start_exactTrue_refFalse.csv")
        if os.path.exists(csv):
            sample_csvs[name] = csv

    if sample_csvs:
        fig1 = plot_gap_vs_time_single_sample(sample_csvs, sample_id=sample_id)
        fig1.savefig(f'{SAVE_BASE}/gap_vs_time_sample{sample_id}.png', dpi=200, bbox_inches='tight')
        print(f"Saved gap_vs_time_sample{sample_id}.png")

    # ── Plot 2a: Mean ABSOLUTE gap vs time across all samples ──
    fig2b = plot_mean_gap_vs_time(
        iter_log_folders,
        relative=False,
        show_median=False,
        show_switch_lines=True,
        save_path=f'{SAVE_BASE}/mean_gap_absolute_vs_time_with_switch.png'
    )

    # ── Plot 2b: Mean RELATIVE gap vs time across all samples ──
    fig2b = plot_mean_gap_vs_time(
        iter_log_folders,
        relative=True,
        show_median=False,
        show_switch_lines=True,
        save_path=f'{SAVE_BASE}/mean_gap_relative_vs_time_with_switch.png'
    )
    # ── Plot 3: Head-to-head D₀ vs D₁ ──
    # fig3 = plot_head_to_head(iter_log_folders, "D₀", "D₁")
    # fig3.savefig(f'{SAVE_BASE}/head_to_head_D0_vs_D1.png', dpi=200, bbox_inches='tight')
    # print("Saved head_to_head_D0_vs_D1.png")


    ## ── Plot 3: Head-to-head D_1 vs D_1 Seperate Head ──
    # fig3 = plot_head_to_head(iter_log_folders, "D_1", "D_1_sephead")
    # fig3.savefig(f'{SAVE_BASE}/head_to_head_D1_vs_D1SepHead.png', dpi=200, bbox_inches='tight')
    # print("Saved head_to_head_D1_vs_D1SepHead.png")

    

    # # ── Run for multiple representative samples ──
    # for sid in [0, 10, 36, 72]:  # easy, medium, hard, last
    #     sample_csvs_i = {}
    #     for name, folder in iter_log_folders.items():
    #         if name == "Exact Benders":
    #             csv = os.path.join(folder, f"iterlog_sample{sid}_start_exactTrue_refFalse.csv")
    #         else:
    #             csv = os.path.join(folder, f"iterlog_sample{sid}_start_exactFalse_refTrue.csv")
    #         if os.path.exists(csv):
    #             sample_csvs_i[name] = csv
    #     if len(sample_csvs_i) >= 2:
    #         fig = plot_single_sample_diagnostic(sample_csvs_i, sample_id=sid)
    #         fig.savefig(f'{SAVE_BASE}/diagnostic_sample{sid}.png', dpi=200, bbox_inches='tight')
    #         print(f"Saved diagnostic_sample{sid}.png")

    # plt.show()

    # TODO
    # --- Evalation compare the normalized duality gap (%) reached when inexact iters switched to exact iters 
    # And also how many percentage of iteration out of total iter is done in inexact mode
    # and also plot it out
    iter_log_folders_inexact = {
        # "Exact Benders":  f"{BASE}/iter_logs_inexact_refine_Exact_Solve",
        r"D$_{uniform}$":             f"{BASE}/iter_logs_Inexact_Refine_BendersBaseline",

        r"D$_{CAB}$":             f"{BASE}/iter_logs_inexact_refine_ConstD_1",
        # "D_1_sephead":  f"{BASE}/iter_logs_Inexact_Refine_ConstSeperateHead_NormbyDemand",
        # "D_2":             f"{BASE}/iter_logs_Inexact_Refine_ConstRenewUB10000",
        # "D_3":             f"{BASE}/iter_logs_Inexact_Refine_Const10000Perc50",
        # "D_4":             f"{BASE}/iter_logs_Inexact_Refine_Const10000Perc90"
    }

    print_switch_summary(iter_log_folders_inexact)

    fig_switch = plot_switch_analysis(
        iter_log_folders_inexact,
        save_path=f"{SAVE_BASE}/switch_analysis.png"
    )

    # fig3 = plot_head_to_head(iter_log_folders_inexact, "D_4", "D_1")
    # fig3.savefig(f'{SAVE_BASE}/head_to_head_D4_vs_D1.png', dpi=200, bbox_inches='tight')
    # print("Saved head_to_head_D4_vs_D1.png")


    fig_speed, speed_summary = plot_speedup_vs_switch_gap(
        exact_folder=f"{BASE}/iter_logs_Exact_Exact",
        method_folders=iter_log_folders_inexact,
        save_path=f"{SAVE_BASE}/speedup_vs_switch_gap.png"
    )

    fig_speed_d1, _ = plot_speedup_vs_switch_gap(
        exact_folder=f"{BASE}/iter_logs_Exact_Exact",
        method_folders={
            r"D$_{uniform}$": f"{BASE}/iter_logs_Inexact_Refine_BendersBaseline",
            r"D$_{CAB}$": f"{BASE}/iter_logs_inexact_refine_ConstD_1"
        },
        save_path=f"{SAVE_BASE}/speedup_vs_switch_gap_D0_D1.png"
    )