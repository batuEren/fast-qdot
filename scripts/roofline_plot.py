import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ── Data & Specifications ─────────────────────────────────────────────────────

# Hardware Specifications
PEAK_PERF = 512.0  # Peak Performance (GFLOPS)
MEM_BW = 64.0  # Memory Bandwidth (GB/s)

# Kernel Arithmetic Intensities (FLOPs/Byte) and Measured Performances (GFLOPS)
kernels = ["GEMM", "Conv2D", "FFT", "SpMV", "AXPY"]
intensities = [8.0, 4.5, 1.2, 0.15, 0.05]
performances = [410.0, 220.0, 70.0, 9.1, 3.1]

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette = sns.color_palette("muted")
c_roof = "#2c3e50"  # Dark slate for the roofline boundary
c_dots = palette[0]  # Soft blue for the application kernels

fig, ax = plt.subplots(figsize=(7.5, 6))
fig.suptitle(
    "Hardware Roofline Model & Application Bounds",
    fontsize=14,
    fontweight="bold",
    y=0.96,
)

# ── Roofline Math ─────────────────────────────────────────────────────────────

# Define the X-axis range (Arithmetic Intensity)
x_min, x_max = 0.01, 100.0
x_space = np.logspace(np.log10(x_min), np.log10(x_max), 500)

# Calculate the roofline bound: Min(Memory Bound, Compute Bound)
roofline = np.minimum(MEM_BW * x_space, PEAK_PERF)

# Ridge point where the bottleneck shifts (Inflexion point)
ridge_intensity = PEAK_PERF / MEM_BW

# ── Plotting ──────────────────────────────────────────────────────────────────

# 1. Draw the Roofline Bound
ax.plot(
    x_space,
    roofline,
    color=c_roof,
    linewidth=2.5,
    label=f"Peak Bounds ({PEAK_PERF} GFLOPS / {MEM_BW} GB/s)",
    zorder=2,
)

# 2. Draw Kernel Data Points
scatter = ax.scatter(
    intensities,
    performances,
    color=c_dots,
    edgecolor="white",
    s=90,
    linewidth=1.2,
    label="Measured Kernels",
    zorder=4,
)

# ── Formatting & Annotations ──────────────────────────────────────────────────

# Log-Log scaling is standard for Roofline Models
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_min, x_max)
ax.set_ylim(0.1, PEAK_PERF * 2)

ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=10, fontweight="bold")
ax.set_ylabel(
    "Attainable Performance (GFLOPS)", fontsize=10, fontweight="bold"
)

# Clean, explicit grid lines for log scale
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Kernel Annotations
for i, name in enumerate(kernels):
    # Determine offset based on spacing so text doesn't overlap points
    y_offset = -12 if name in ["SpMV", "AXPY"] else 8
    ax.annotate(
        name,
        xy=(intensities[i], performances[i]),
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        weight="semibold",
        alpha=0.85,
    )

# Highlight Ridge Point (The design inflection sweet spot)
ax.plot(
    [ridge_intensity, ridge_intensity],
    [0.1, PEAK_PERF],
    color="crimson",
    linestyle=":",
    linewidth=1.2,
    alpha=0.7,
)
ax.annotate(
    f"Ridge Point\n({ridge_intensity:.2f} FLOP/B)",
    xy=(ridge_intensity, 0.15),
    xytext=(10, 0),
    textcoords="offset points",
    ha="left",
    fontsize=8,
    color="crimson",
    weight="semibold",
)

# Region Text Labels
ax.text(
    0.02,
    0.6,
    "Memory-Bound Area",
    fontsize=9,
    fontstyle="italic",
    alpha=0.6,
    rotation=36,
)
ax.text(
    15.0,
    PEAK_PERF * 1.1,
    "Compute-Bound Area",
    fontsize=9,
    fontstyle="italic",
    alpha=0.6,
)

# Elegant Legend Placement
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

# ── Save Outputs ──────────────────────────────────────────────────────────────

fig.tight_layout()
plt.savefig("roofline_model.png", dpi=180, bbox_inches="tight")
plt.savefig("roofline_model.pdf", bbox_inches="tight")
print("Saved roofline_model.png and roofline_model.pdf")