import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

depths = [1, 2, 3, 4, 5, 6]

ternary_dot = {
    4096: [920.382, 1019.45, 1109.75, 1186.27, 981.706, 1159.35],
    8192: [932.068, 1021.22, 1118.21,  903.389, 850.814, 1092.20],
}

binary_dot = {
    4096: [932.068, 981.946, 1135.51, 1271.00, 1092.20, 1085.35],
    8192: [869.934, 978.675, 1092.20, 1198.37, 1043.66, 1089.03],
}

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette = sns.color_palette("muted")
c4096 = palette[0]
c8192 = palette[1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("PLut Dot-Product Throughput by LUT Depth", fontsize=14, fontweight="bold", y=1.01)

x = np.arange(len(depths))
bar_w = 0.35

def draw_panel(ax, data, title, baseline_4096, baseline_8192):
    vals_4096 = data[4096]
    vals_8192 = data[8192]

    bars1 = ax.bar(x - bar_w / 2, vals_4096, bar_w, label="seq 4096", color=c4096, edgecolor="white")
    bars2 = ax.bar(x + bar_w / 2, vals_8192, bar_w, label="seq 8192", color=c8192, edgecolor="white")

    # Baseline (depth-1) dashed reference lines
    ax.axhline(baseline_4096, color=c4096, linewidth=1.1, linestyle="--", alpha=0.6)
    ax.axhline(baseline_8192, color=c8192, linewidth=1.1, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("LUT Depth", fontsize=10)
    ax.set_ylabel("Throughput (M items/s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.2f} G" if v >= 1000 else f"{v:.0f} M"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate recommended depth
    recommended = {"Ternary": 3, "Binary": 4}[title]
    rec_idx = recommended - 1
    for bar in [bars1[rec_idx], bars2[rec_idx]]:
        ax.annotate(
            "★",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="goldenrod",
        )

draw_panel(axes[0], ternary_dot, "Ternary", ternary_dot[4096][0], ternary_dot[8192][0])
draw_panel(axes[1], binary_dot,  "Binary",  binary_dot[4096][0],  binary_dot[8192][0])

fig.tight_layout()
plt.savefig("benchmark_plot.png", dpi=180, bbox_inches="tight")
plt.savefig("benchmark_plot.pdf", bbox_inches="tight")
print("Saved benchmark_plot.png and benchmark_plot.pdf")
