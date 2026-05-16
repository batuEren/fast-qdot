import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# ── Load CSV ───────────────────────────────────────────────────────────────────

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "procedural_lut.csv"

def load_dot_data(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("name,"))
    df = pd.read_csv(csv_path, skiprows=header_idx)

    pattern = re.compile(r"BM_PLut(Ternary|Binary)DotOnly<(\d+)>/(\d+)")
    ternary, binary = {}, {}
    for _, row in df.iterrows():
        m = pattern.search(row["name"])
        if not m:
            continue
        kind, depth, seq = m.group(1), int(m.group(2)), int(m.group(3))
        throughput_M = row["items_per_second"] / 1e6
        target = ternary if kind == "Ternary" else binary
        target.setdefault(seq, {})[depth] = throughput_M

    depths = sorted(next(iter(ternary.values())).keys())
    ternary = {seq: [vals[d] for d in depths] for seq, vals in ternary.items()}
    binary  = {seq: [vals[d] for d in depths] for seq, vals in binary.items()}
    return depths, ternary, binary

depths, ternary_dot, binary_dot = load_dot_data(CSV_PATH)
seq_sizes = sorted(ternary_dot.keys())

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette = sns.color_palette("muted", len(seq_sizes))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("PLut Dot-Product Throughput by LUT Depth", fontsize=14, fontweight="bold", y=1.01)

x = np.arange(len(depths))
bar_w = 0.8 / len(seq_sizes)

def draw_panel(ax, data, title):
    bars_per_seq = []
    for i, seq in enumerate(seq_sizes):
        offset = (i - (len(seq_sizes) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, data[seq], bar_w, label=f"seq {seq}",
                      color=palette[i], edgecolor="white")
        bars_per_seq.append((seq, bars))
        ax.axhline(data[seq][0], color=palette[i], linewidth=1.1, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("LUT Depth", fontsize=10)
    ax.set_ylabel("Throughput (M items/s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1000:.2f} G" if v >= 1000 else f"{v:.0f} M")
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    recommended = {"Ternary": 3, "Binary": 4}[title]
    rec_idx = recommended - 1
    for _, bars in bars_per_seq:
        bar = bars[rec_idx]
        ax.annotate(
            "★",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="goldenrod",
        )

draw_panel(axes[0], ternary_dot, "Ternary")
draw_panel(axes[1], binary_dot,  "Binary")

fig.tight_layout()
plt.savefig("benchmark_plot.png", dpi=180, bbox_inches="tight")
plt.savefig("benchmark_plot.pdf", bbox_inches="tight")
print("Saved benchmark_plot.png and benchmark_plot.pdf")
