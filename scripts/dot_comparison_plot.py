import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Load CSV ───────────────────────────────────────────────────────────────────

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "dot_comparison.csv"

IMPL_ORDER = ["Naive", "Mad", "MadAVX2", "Lut", "PLut"]
IMPL_LABELS = {"Naive": "Naive", "Mad": "MAD", "MadAVX2": "MAD AVX2",
               "Lut": "LUT", "PLut": "PLut"}

def parse_name(name):
    # Normalise: MadTernaryAVX2 → impl=MadAVX2, kind=Ternary
    m = re.search(r"BM_(Mad)(Ternary|Binary)(AVX2)?/(\d+)", name)
    if m:
        return ("Mad" if not m.group(3) else "MadAVX2"), m.group(2), int(m.group(4))
    m = re.search(r"BM_(Naive|Lut|PLut)(Ternary|Binary)/(\d+)", name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None

def load(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("name,"))
    df = pd.read_csv(csv_path, skiprows=header_idx)

    records = []
    for _, row in df.iterrows():
        parsed = parse_name(row["name"])
        if parsed is None:
            continue
        impl, kind, n = parsed
        records.append({"impl": impl, "kind": kind, "n": n,
                        "throughput_M": row["items_per_second"] / 1e6})
    return pd.DataFrame(records)

df = load(CSV_PATH)

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette = dict(zip(IMPL_ORDER, sns.color_palette("tab10", len(IMPL_ORDER))))
markers = {"Naive": "o", "Mad": "s", "MadAVX2": "D", "Lut": "^", "PLut": "v"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Dot-Product Throughput: Implementation Comparison", fontsize=14,
             fontweight="bold", y=1.01)

def draw_panel(ax, kind):
    sub = df[df["kind"] == kind].copy()
    for impl in IMPL_ORDER:
        rows = sub[sub["impl"] == impl].sort_values("n")
        if rows.empty:
            continue
        ax.plot(rows["n"], rows["throughput_M"],
                label=IMPL_LABELS[impl],
                color=palette[impl],
                marker=markers[impl],
                linewidth=1.8, markersize=6)

    ax.set_title(kind, fontsize=12, fontweight="bold")
    ax.set_xlabel("Vector size (n)", fontsize=10)
    ax.set_ylabel("Throughput (M items/s)", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1000:.1f} G" if v >= 1000 else f"{v:.0f} M")
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_panel(axes[0], "Ternary")
draw_panel(axes[1], "Binary")

fig.tight_layout()
plt.savefig("dot_comparison_plot.png", dpi=180, bbox_inches="tight")
plt.savefig("dot_comparison_plot.pdf", bbox_inches="tight")
print("Saved dot_comparison_plot.png and dot_comparison_plot.pdf")
