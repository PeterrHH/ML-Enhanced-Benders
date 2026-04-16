import numpy as np
import matplotlib.pyplot as plt

def plot_benders_cuts(show_approximate_second_cut=True):
    # Investment axis
    x = np.linspace(0, 10, 400)

    # Convex true total cost
    def f(x):
        return 0.22 * (x - 5.2)**2 + 6.2 + 0.15 * x

    def df(x):
        return 2 * 0.22 * (x - 5.2) + 0.15

    total_cost = f(x)

    # Two cut points
    xk1 = 2.2
    xk2 = 7.0

    # ---- Cut 1 ----
    y1 = f(xk1)
    if show_approximate_second_cut:
        # Slightly approximate but still close
        m1 = df(xk1) * 0.95
        offset1 = 0.5
        cut1 = (y1 - offset1) + m1 * (x - xk1)
        cut1_label = "Approximate cut 1"
    else:
        # Exact / tight
        m1 = df(xk1)
        cut1 = y1 + m1 * (x - xk1)
        cut1_label = "Exact cut 1"

    # ---- Cut 2 ----
    y2 = f(xk2)
    if show_approximate_second_cut:
        # More visibly approximate
        m2 = df(xk2) * 0.85
        offset2 = 0.75
        cut2 = (y2 - offset2) + m2 * (x - xk2)
        cut2_label = "Approximate cut 2"
    else:
        # Exact / tight
        m2 = df(xk2)
        cut2 = y2 + m2 * (x - xk2)
        cut2_label = "Exact cut 2"

    # Lower approximation
    lower_approx = np.maximum(cut1, cut2)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(x, total_cost, color="black", linewidth=3, label="True total cost")
    ax.plot(x, cut1, "--", linewidth=2.8, color="tab:blue", label=cut1_label)
    ax.plot(x, cut2, "--", linewidth=2.8, color="tab:orange", label=cut2_label)
    ax.plot(x, lower_approx, color="tab:red", linewidth=3, label="Current lower approximation")

    # Points on the true curve
    ax.scatter([xk1], [f(xk1)], color="tab:blue", edgecolor="black", s=120, zorder=6)
    ax.scatter([xk2], [f(xk2)], color="tab:orange", edgecolor="black", s=120, zorder=6)

    # Show vertical gaps for approximate case
    if show_approximate_second_cut:
        y_cut1_at_xk1 = cut1[np.argmin(np.abs(x - xk1))]
        y_cut2_at_xk2 = cut2[np.argmin(np.abs(x - xk2))]

        ax.plot(
            [xk1, xk1],
            [y_cut1_at_xk1, f(xk1)],
            color="tab:blue",
            linestyle=":",
            linewidth=2
        )
        ax.plot(
            [xk2, xk2],
            [y_cut2_at_xk2, f(xk2)],
            color="tab:orange",
            linestyle=":",
            linewidth=2
        )

    ax.set_xlabel("Investment decision", fontsize=18, fontweight="bold")
    ax.set_ylabel("Total cost", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_benders_cuts(show_approximate_second_cut=True)