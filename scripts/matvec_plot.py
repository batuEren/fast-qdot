import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# ── Load CSV ───────────────────────────────────────────────────────────────────

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "matvec.csv"

# BM_PLutTernaryMatVec<3>/128/1024
PATTERN = re.compile(r"BM_PLut(Ternary|Binary)MatVec<(\d+)>/(\d+)/(\d+)")

def load(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("name,"))
    df = pd.read_csv(csv_path, skiprows=header_idx)

    records = []
    for _, row in df.iterrows():
        m = PATTERN.search(row["name"])
        if not m:
            continue
        kind, depth, rows_m, cols_n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        records.append({"kind": kind, "depth": depth,
                        "shape": f"{rows_m}×{cols_n}",
                        "throughput_M": row["items_per_second"] / 1e6})
    return pd.DataFrame(records)

df = load(CSV_PATH)
depths = sorted(df["depth"].unique())
shapes = sorted(df["shape"].unique(), key=lambda s: int(s.split("×")[0]))

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette = dict(zip(shapes, sns.color_palette("muted", len(shapes))))
markers = ["o", "s", "D", "^", "v"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("PLut Matrix-Vector Throughput by LUT Depth", fontsize=14,
             fontweight="bold", y=1.01)

x = np.arange(len(depths))
bar_w = 0.8 / len(shapes)

def draw_panel(ax, kind):
    sub = df[df["kind"] == kind]
    for i, shape in enumerate(shapes):
        vals = [sub[(sub["depth"] == d) & (sub["shape"] == shape)]["throughput_M"].values[0]
                for d in depths]
        offset = (i - (len(shapes) - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, label=shape,
               color=palette[shape], edgecolor="white")

    ax.set_title(kind, fontsize=12, fontweight="bold")
    ax.set_xlabel("LUT Depth", fontsize=10)
    ax.set_ylabel("Throughput (M items/s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1000:.1f} G" if v >= 1000 else f"{v:.0f} M")
    )
    ax.legend(title="Matrix shape", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_panel(axes[0], "Ternary")
draw_panel(axes[1], "Binary")

fig.tight_layout()
plt.savefig("matvec_plot.png", dpi=180, bbox_inches="tight")
plt.savefig("matvec_plot.pdf", bbox_inches="tight")
print("Saved matvec_plot.png and matvec_plot.pdf")
