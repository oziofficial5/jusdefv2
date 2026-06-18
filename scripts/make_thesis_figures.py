"""
Produce thesis-grade PDF figures from the JSON logs in outputs/logs/.

Figures (those producible from existing data):
    fig1_per_regime_ablation_heatmap.pdf
        Variants × density-bins, color = macro-F1. The marquee figure for
        the per-regime ablation argument.
    fig2_density_stratified_bars.pdf
        Density bins × {mean baseline, v3}, 5-seed mean +/- std bars.
        The principal positive-finding figure.
    fig3_aggregate_vs_regime_inversion.pdf
        Two-panel: aggregate F1 vs 10-20% bin F1, for v3 vs hard_attention.
        Makes the inversion narrative visually obvious.
    fig4_compute_vs_gain_pareto.pdf
        Parameter count vs 10-20% bin macro-F1 across all v3 variants and
        the mean baseline. Calibrates the "is the extra capacity worth it"
        question.

Figures requiring additional data collection (NOT produced here):
    op_coef_trajectory      -- needs per-epoch coef logging (not currently saved)
    attention_entropy       -- needs eval-time attention weight dump
    bin_sensitivity         -- needs analyse_ledgar_bin_sensitivity.py to run

Usage:
    python scripts/make_thesis_figures.py
Output:
    outputs/figures/fig1..fig4.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Styling (consistent across all figures)
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette (colour-blind friendly, Wong 2011)
C_MEAN   = "#0072B2"  # blue
C_V3     = "#D55E00"  # vermillion
C_ABL    = "#999999"  # neutral grey for ablations
C_HARD   = "#CC79A7"  # pink — hard attention specifically
C_HIGHLIGHT = "#E69F00"  # amber — operating regime marker

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path("outputs/logs/ledgar_per_regime_ablation_analysis.json")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_per_regime():
    with open(LOG_PATH) as f:
        d = json.load(f)
    return d


# Bin display order and labels
BIN_KEYS = ["all", "0pct", "5_10", "10_20", "20_30", "30_50", "50_plus"]
BIN_LABELS = ["all", "0\\%", "5--10\\%", "10--20\\%", "20--30\\%", "30--50\\%", "$\\geq 50$\\%"]

VARIANT_DISPLAY = {
    "mean":                    "mean baseline",
    "v3_pilot":                "v3 (main)",
    "v3_abl_shared_w_revert":  "$-$ shared $W$",
    "v3_abl_hard_attention":   "$-$ soft attention",
    "v3_abl_unit_coefs":       "$-$ signed init",
    "v3_abl_no_drift_reg":     "$-$ drift reg",
}
VARIANT_ORDER = list(VARIANT_DISPLAY.keys())


# ---------------------------------------------------------------------------
# Figure 1: Per-regime ablation heatmap
# ---------------------------------------------------------------------------

def fig1_per_regime_heatmap(data):
    agg = data["aggregate"]
    # Build matrix: rows = variants, cols = bins
    mat = np.full((len(VARIANT_ORDER), len(BIN_KEYS)), np.nan)
    for i, v in enumerate(VARIANT_ORDER):
        for j, b in enumerate(BIN_KEYS):
            m = agg[v][b]["mean"]
            if m is not None:
                mat[i, j] = m

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.55, vmax=0.85, aspect="auto")

    # Annotate cells with macro-F1 to 3 decimals
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                # Choose text colour by background brightness
                txt_color = "black" if 0.62 < mat[i, j] < 0.78 else "white"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color=txt_color, fontsize=8.5)

    ax.set_xticks(range(len(BIN_KEYS)))
    ax.set_xticklabels(BIN_LABELS)
    ax.set_yticks(range(len(VARIANT_ORDER)))
    ax.set_yticklabels([VARIANT_DISPLAY[v] for v in VARIANT_ORDER])
    ax.set_xlabel("non-AFF density bin (5-seed mean macro-F1)")

    # Highlight the operating regime column
    ax.add_patch(plt.Rectangle((3 - 0.5, -0.5), 1, len(VARIANT_ORDER),
                               fill=False, edgecolor=C_HIGHLIGHT, lw=2.2))

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("macro-F1", rotation=270, labelpad=14)

    fig.tight_layout()
    out = OUT_DIR / "fig1_per_regime_ablation_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Density-stratified bars (mean baseline vs v3)
# ---------------------------------------------------------------------------

def fig2_density_stratified_bars(data):
    agg = data["aggregate"]
    bins = ["5_10", "10_20", "20_30", "30_50", "50_plus"]
    bin_labels = ["5--10\\%", "10--20\\%", "20--30\\%", "30--50\\%", "$\\geq 50$\\%"]

    mean_vals = [agg["mean"][b]["mean"] for b in bins]
    mean_stds = [agg["mean"][b]["std"] for b in bins]
    v3_vals   = [agg["v3_pilot"][b]["mean"] for b in bins]
    v3_stds   = [agg["v3_pilot"][b]["std"] for b in bins]

    x = np.arange(len(bins))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(x - w/2, mean_vals, w, yerr=mean_stds, capsize=3,
           color=C_MEAN, label="mean baseline", edgecolor="black", lw=0.4)
    ax.bar(x + w/2, v3_vals, w, yerr=v3_stds, capsize=3,
           color=C_V3, label="v3 (main)", edgecolor="black", lw=0.4)

    # Highlight the operating regime bar group
    ax.axvspan(0.5, 1.5, color=C_HIGHLIGHT, alpha=0.12, zorder=0)
    ax.text(1, 0.83, "operating\\nregime", ha="center", va="top",
            fontsize=8.5, color="#8a6500")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("non-AFF density bin")
    ax.set_ylabel("test macro-F1 (5-seed mean $\\pm$ std)")
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    out = OUT_DIR / "fig2_density_stratified_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Aggregate vs regime inversion (hard_attention case)
# ---------------------------------------------------------------------------

def fig3_inversion(data):
    agg = data["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

    variants = ["v3_pilot", "v3_abl_hard_attention"]
    labels = ["v3 (main)", "$-$ soft attention"]
    colors = [C_V3, C_HARD]

    # Panel 1: aggregate
    means = [agg[v]["all"]["mean"] for v in variants]
    stds  = [agg[v]["all"]["std"] for v in variants]
    x = np.arange(len(variants))
    axes[0].bar(x, means, yerr=stds, capsize=4, color=colors,
                edgecolor="black", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("aggregate test set", fontsize=10)
    axes[0].set_ylabel("test macro-F1 (5-seed mean $\\pm$ std)")
    delta_agg = means[1] - means[0]
    axes[0].text(0.5, 0.72, f"$\\Delta$ = ${delta_agg:+.4f}$",
                 ha="center", transform=axes[0].transAxes,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # Panel 2: 10-20% bin
    means = [agg[v]["10_20"]["mean"] for v in variants]
    stds  = [agg[v]["10_20"]["std"] for v in variants]
    axes[1].bar(x, means, yerr=stds, capsize=4, color=colors,
                edgecolor="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("10--20\\% non-AFF density bin (operating regime)",
                      fontsize=10)
    delta_reg = means[1] - means[0]
    axes[1].text(0.5, 0.72, f"$\\Delta$ = ${delta_reg:+.4f}$",
                 ha="center", transform=axes[1].transAxes,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    for ax in axes:
        ax.set_ylim(0.55, 0.78)

    fig.suptitle("Aggregate-vs-regime inversion: hard attention", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fig3_aggregate_vs_regime_inversion.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Compute-vs-gain on the operating regime
# ---------------------------------------------------------------------------

# Approximate parameter counts (from train_ledgar.py output, rounded)
PARAMS_M = {
    "mean":                    0.71,  # input_proj + classifier only
    "v3_pilot":                2.08,
    "v3_abl_shared_w_revert":  2.51,  # 4 W's instead of 1 shared
    "v3_abl_hard_attention":   2.08,
    "v3_abl_unit_coefs":       2.08,
    "v3_abl_no_drift_reg":     2.08,
}

def fig4_pareto(data):
    agg = data["aggregate"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    for v in VARIANT_ORDER:
        m = agg[v]["10_20"]["mean"]
        s = agg[v]["10_20"]["std"]
        if m is None:
            continue
        p = PARAMS_M[v]
        if v == "mean":
            color = C_MEAN
        elif v == "v3_pilot":
            color = C_V3
        elif v == "v3_abl_hard_attention":
            color = C_HARD
        else:
            color = C_ABL
        ax.errorbar(p, m, yerr=s, fmt="o", color=color, markersize=8,
                    capsize=4, markeredgecolor="black", markeredgewidth=0.6,
                    ecolor=color, alpha=0.9, zorder=3)
        # Label point
        dy = 0.005 if v != "mean" else -0.012
        ax.annotate(VARIANT_DISPLAY[v], (p, m), xytext=(p + 0.05, m + dy),
                    fontsize=8.5)

    ax.set_xlabel("model parameters (M)")
    ax.set_ylabel("test macro-F1 on 10--20\\% bin\n(5-seed mean $\\pm$ std)")
    ax.set_title("Compute-vs-gain on the operating regime")
    ax.set_xlim(0.4, 2.9)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    out = OUT_DIR / "fig4_compute_vs_gain_pareto.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    if not LOG_PATH.is_file():
        print(f"ERROR: {LOG_PATH} is missing. Run "
              f"scripts/analyse_ledgar_ablations_stratified.py first.")
        return
    data = load_per_regime()
    print(f"Loaded {LOG_PATH}")
    print("Producing figures:")
    fig1_per_regime_heatmap(data)
    fig2_density_stratified_bars(data)
    fig3_inversion(data)
    fig4_pareto(data)
    print(f"\nAll figures written to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
