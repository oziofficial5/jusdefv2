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
# Figure 5: Bin-sensitivity sweep
# ---------------------------------------------------------------------------

BIN_SENS_PATH = Path("outputs/logs/ledgar_bin_sensitivity_analysis.json")


def fig5_bin_sensitivity():
    if not BIN_SENS_PATH.is_file():
        print(f"  SKIP fig5: {BIN_SENS_PATH} not present "
              f"(run analyse_ledgar_bin_sensitivity.py first)")
        return
    with open(BIN_SENS_PATH) as f:
        data = json.load(f)

    # Display order: symmetric shifts first, then asymmetric
    display_order = ["8-22", "9-21", "10-20", "11-19", "12-18", "8-15", "15-25"]
    available = [k for k in display_order if k in data["per_bin"]
                 and data["per_bin"][k].get("delta_mean") is not None]

    bin_centres = []
    deltas = []
    stds = []
    n_paragraphs = []
    labels = []
    for k in available:
        rec = data["per_bin"][k]
        bin_centres.append((rec["lo"] + rec["hi"]) / 2 * 100)
        deltas.append(rec["delta_mean"])
        stds.append(rec["delta_std"])
        n_paragraphs.append(rec["n_paragraphs"])
        labels.append(k + "\\%")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    # Highlight zero line
    ax.axhline(0, color="grey", lw=0.7, linestyle="--", zorder=1)

    # Bars coloured by canonical-or-not
    colors = [C_HIGHLIGHT if k == "10-20" else C_V3 for k in available]
    x = np.arange(len(available))
    bars = ax.bar(x, deltas, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", lw=0.5)

    # Annotate N per bar
    ymax = max(d + s for d, s in zip(deltas, stds)) * 1.15
    for i, (d, s, n) in enumerate(zip(deltas, stds, n_paragraphs)):
        ax.text(i, d + s + 0.005, f"N={n}", ha="center", va="bottom",
                fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("non-AFF density bin (\\% range)")
    ax.set_ylabel("$\\Delta$ macro-F1 (v3 $-$ mean), 5-seed mean $\\pm$ std")
    ax.set_title("Bin-sensitivity of the v3 win around the 10--20\\% operating regime")
    ax.set_ylim(min(0, min(d - s for d, s in zip(deltas, stds)) * 1.3), ymax)

    fig.tight_layout()
    out = OUT_DIR / "fig5_bin_sensitivity.pdf"
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
    fig5_bin_sensitivity()
    fig10_v4_variants_comparison()
    fig11_v4_twostage_per_seed_deltas()
    fig14_counterfactual_per_bin_heatmap()
    fig15_v4_training_curves()
    fig16_cross_corpus_density()
    fig17_op_coef_trajectory()
    fig18_confusion_10_20()
    print(f"\nAll figures written to {OUT_DIR.resolve()}/")


# ---------------------------------------------------------------------------
# Figure 17: op_coef trajectory during v3 training (single seed)
# ---------------------------------------------------------------------------

def fig17_op_coef_trajectory():
    # Look for a v3 LEDGAR training log JSON with op_coef_trajectory populated.
    candidates = [
        Path("outputs/logs/ledgar_v3_traj_s42.json"),
        Path("outputs/logs/ledgar_v3_pilot_s42.json"),
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        print("  SKIP fig17: no v3 training log with op_coef_trajectory found")
        return
    with open(src) as f:
        d = json.load(f)
    traj = d.get("op_coef_trajectory")
    if not traj:
        print(f"  SKIP fig17: {src} has no op_coef_trajectory field")
        return

    epochs = [e["epoch"] for e in traj]
    # Detect layers from keys
    layer_idxs = sorted({int(k.split("_")[0].replace("layer", ""))
                         for k in traj[0].keys() if k.startswith("layer")})
    fig, axes = plt.subplots(1, len(layer_idxs), figsize=(4.5 * len(layer_idxs), 3.4),
                              sharey=True)
    if len(layer_idxs) == 1:
        axes = [axes]
    op_colors = {"AFF": "#0072B2", "NEG": "#D55E00", "EXC": "#E69F00", "OVR": "#009E73"}
    op_init = {"AFF": 1.0, "NEG": -1.0, "EXC": -0.5, "OVR": 1.0}

    for ax, li in zip(axes, layer_idxs):
        for op_name in ["AFF", "NEG", "EXC", "OVR"]:
            key = f"layer{li}_{op_name}"
            ys = [e.get(key) for e in traj]
            ax.plot(epochs, ys, marker="o", markersize=3, lw=1.2,
                    color=op_colors[op_name], label=f"$c_\\mathsf{{{op_name}}}$")
            ax.axhline(op_init[op_name], color=op_colors[op_name], lw=0.6,
                       linestyle=":", alpha=0.6)
        ax.set_title(f"layer {li}")
        ax.set_xlabel("epoch")
        ax.set_ylim(-1.4, 1.4)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--", zorder=1)
        ax.grid(True, alpha=0.25, linestyle=":")
    axes[0].set_ylabel("$c_\\omega$ (signed coefficient)")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("v3 signed-coefficient trajectory across training (seed 42, LEDGAR)\n"
                 "dotted horizontal lines mark the semantic initialisation",
                 fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "fig17_op_coef_trajectory.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 10: v4 variants comparison (aggregate vs 10-20% bin)
# ---------------------------------------------------------------------------

V4_ANALYSIS_PATH = Path("outputs/logs/ledgar_v4_analysis.json")


def fig10_v4_variants_comparison():
    if not V4_ANALYSIS_PATH.is_file():
        print(f"  SKIP fig10: {V4_ANALYSIS_PATH} missing")
        return
    with open(V4_ANALYSIS_PATH) as f:
        d = json.load(f)
    agg = d["aggregate"]

    variants = ["mean", "v3_pilot", "v4_hard", "v4_soft", "v4_twostage"]
    labels = ["mean", "v3", "v4\\_hard", "v4\\_soft", "v4\\_twostage"]
    colors = ["#0072B2", "#D55E00", "#999999", "#009E73", "#CC79A7"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, bin_key, title in [(axes[0], "all", "Aggregate test set"),
                                (axes[1], "10_20", "10--20\\% operating regime")]:
        means = [agg[v][bin_key]["mean"] for v in variants]
        stds  = [agg[v][bin_key]["std"]  for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors,
               edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("test macro-F1 (5-seed mean $\\pm$ std)")
    axes[1].axhline(agg["v3_pilot"]["10_20"]["mean"], color="grey",
                    lw=0.7, linestyle=":", zorder=1)
    fig.suptitle("v4 variants comparison on LEDGAR", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fig10_v4_variants_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 11: v4_twostage vs v3 per-seed deltas on 10-20% bin
# ---------------------------------------------------------------------------

def fig11_v4_twostage_per_seed_deltas():
    if not V4_ANALYSIS_PATH.is_file():
        print(f"  SKIP fig11: {V4_ANALYSIS_PATH} missing")
        return
    with open(V4_ANALYSIS_PATH) as f:
        d = json.load(f)
    per_seed = d["per_seed"]
    seeds = sorted(int(s) for s in per_seed["v3_pilot"].keys())
    deltas = []
    for s in seeds:
        v3 = per_seed["v3_pilot"][str(s)]["10_20"]["macro_f1"]
        v4 = per_seed["v4_twostage"][str(s)]["10_20"]["macro_f1"]
        if v3 is None or v4 is None:
            deltas.append(None)
        else:
            deltas.append(v4 - v3)
    if all(x is None for x in deltas):
        print("  SKIP fig11: no v4_twostage data")
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(seeds))
    colors = ["#0072B2" if d is None or d >= 0 else "#D55E00" for d in deltas]
    bars = ax.bar(x, [d if d is not None else 0 for d in deltas],
                  color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    for i, d in enumerate(deltas):
        if d is None:
            continue
        ymax = max([x for x in deltas if x is not None]) or 0.05
        y_off = 0.003 if d >= 0 else -0.005
        sign = "+" if d >= 0 else "-"
        ax.text(i, d + y_off, f"{sign}{abs(d):.4f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("$\\Delta$ macro-F1 (v4\\_twostage $-$ v3) on 10--20\\% bin")
    ax.set_title("Per-seed regime gain of v4\\_twostage over v3")
    fig.tight_layout()
    out = OUT_DIR / "fig11_v4_twostage_per_seed_deltas.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 14: counterfactual sensitivity per-density-bin heatmap
# ---------------------------------------------------------------------------

CF_PATH = Path("outputs/logs/ledgar_counterfactual_sensitivity.json")


def fig14_counterfactual_per_bin_heatmap():
    if not CF_PATH.is_file():
        print(f"  SKIP fig14: {CF_PATH} missing")
        return
    with open(CF_PATH) as f:
        d = json.load(f)
    per_seed = d["per_seed"]

    variants = [v for v in ["mean", "v3_pilot", "v4_soft", "v4_hard", "v4_twostage"]
                if v in per_seed and per_seed[v]]
    labels = {"mean": "mean", "v3_pilot": "v3",
              "v4_soft": "v4\\_soft", "v4_hard": "v4\\_hard",
              "v4_twostage": "v4\\_twostage"}
    bin_names = ["<10%", "10-20%", "20-50%", ">=50%"]
    bin_labels = ["<10\\%", "10--20\\%", "20--50\\%", "$\\geq$50\\%"]

    # 5-seed mean true_shift per variant per bin
    mat = np.full((len(variants), len(bin_names)), np.nan)
    for i, v in enumerate(variants):
        for j, b in enumerate(bin_names):
            vals = []
            for seed_str, rec in per_seed[v].items():
                if rec is None:
                    continue
                pb = rec.get("per_density_bin", {})
                if b in pb and pb[b].get("mean_true_shift") is not None:
                    vals.append(pb[b]["mean_true_shift"])
            if vals:
                mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    vmax = max(0.05, float(np.nanmax(mat) * 1.05))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "white" if mat[i, j] > vmax * 0.6 else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=8.5, color=color)
    ax.set_xticks(range(len(bin_names)))
    ax.set_xticklabels(bin_labels)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([labels.get(v, v) for v in variants])
    ax.set_xlabel("non-AFF density bin")
    ax.set_title("Counterfactual sensitivity ($|\\Delta P(\\mathrm{true})|$) by density bin")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # Highlight 10-20% column
    ax.add_patch(plt.Rectangle((1 - 0.5, -0.5), 1, len(variants),
                                fill=False, edgecolor="#E69F00", lw=2.2))
    fig.tight_layout()
    out = OUT_DIR / "fig14_counterfactual_per_bin_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 15: training curves -- single-stage v4_hard vs v4_twostage
# ---------------------------------------------------------------------------

import re

def _parse_training_log(path):
    """Extract (epoch, val_macro) pairs from a training log."""
    if not path.is_file():
        return None
    pat = re.compile(r"Epoch\s+(\d+)[^|]*\|\s*loss=[\d.]+\s*\|\s*val_macro=([\d.]+)")
    epochs, vals = [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            vals.append(float(m.group(2)))
    if not epochs:
        return None
    return epochs, vals


def fig15_v4_training_curves():
    candidates = [
        ("v4\\_hard (single-stage)", Path("outputs/logs/v4_hard_s42.log"),    "#999999"),
        ("v4\\_twostage",            Path("outputs/logs/ledgar_v4_twostage_s42.log"), "#CC79A7"),
        ("v3 (reference)",           Path("outputs/logs/d3_v3_pilot_s42.log"), "#D55E00"),
    ]
    curves = []
    for label, p, color in candidates:
        parsed = _parse_training_log(p)
        if parsed is not None:
            curves.append((label, parsed[0], parsed[1], color))
    if not curves:
        print(f"  SKIP fig15: no training-log files found")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for label, epochs, vals, color in curves:
        ax.plot(epochs, vals, marker="o", markersize=3, lw=1.2, color=color,
                label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Training trajectories: single-stage vs two-stage v4 vs v3")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    out = OUT_DIR / "fig15_v4_training_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 16: cross-corpus density histograms
# ---------------------------------------------------------------------------

CORPUS_DENSITY_PATH = Path("outputs/logs/cross_corpus_density.json")


def fig16_cross_corpus_density():
    if not CORPUS_DENSITY_PATH.is_file():
        print(f"  SKIP fig16: {CORPUS_DENSITY_PATH} missing -- "
              f"run scripts/analyse_cross_corpus_density.py first")
        return
    with open(CORPUS_DENSITY_PATH) as f:
        d = json.load(f)
    corpora = [c for c in ["EUR-Lex", "ECtHR", "LEDGAR"] if c in d]
    if not corpora:
        return

    fig, axes = plt.subplots(1, len(corpora), figsize=(3.4 * len(corpora), 3.2),
                              sharey=False)
    if len(corpora) == 1:
        axes = [axes]
    colors = {"EUR-Lex": "#0072B2", "ECtHR": "#009E73", "LEDGAR": "#D55E00"}
    for ax, name in zip(axes, corpora):
        densities = np.array(d[name]["densities"])
        ax.hist(densities, bins=40, color=colors.get(name, "#999999"),
                edgecolor="black", lw=0.3, alpha=0.85)
        # Highlight 10-20% operating regime
        ax.axvspan(0.10, 0.20, color="#E69F00", alpha=0.18)
        ax.set_title(f"{name}\n(n={d[name]['n_documents']}, "
                     f"mean={d[name]['mean_density']*100:.2f}\\%)",
                     fontsize=9.5)
        ax.set_xlabel("non-AFF density")
        ax.set_xlim(0, max(0.3, np.percentile(densities, 99) * 1.05))
        ax.set_yscale("log")
    axes[0].set_ylabel("\\# documents (log scale)")
    fig.suptitle("Cross-corpus non-AFF density distributions\n"
                 "(highlighted band = 10--20\\% operating regime)",
                 fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "fig16_cross_corpus_density.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 18: 10-20% bin confusion-style: per-paragraph correctness map
# ---------------------------------------------------------------------------

PRED_PATH = Path("outputs/logs/predictions_10_20_bin.json")


def fig18_confusion_10_20():
    if not PRED_PATH.is_file():
        print(f"  SKIP fig18: {PRED_PATH} missing -- "
              f"run scripts/dump_predictions_10_20.py first")
        return
    with open(PRED_PATH) as f:
        d = json.load(f)
    truths = d["true_labels"]
    n = len(truths)

    variants = [v for v in ["mean", "v3_pilot", "v4_twostage"] if v in d]
    if not variants:
        return
    # For each variant, majority vote across 5 seeds per paragraph -> single prediction
    majority = {}
    for v in variants:
        seed_preds = d[v]
        if not seed_preds:
            continue
        seeds = sorted(seed_preds.keys(), key=lambda s: int(s))
        per_para = list(zip(*(seed_preds[s] for s in seeds)))
        majority[v] = [max(set(votes), key=votes.count) for votes in per_para]

    # Build agreement matrix: correctness rows v3 vs v4_twostage
    if "v3_pilot" not in majority or "v4_twostage" not in majority:
        print("  SKIP fig18: need v3_pilot and v4_twostage majority predictions")
        return

    v3_correct = np.array([majority["v3_pilot"][i] == truths[i] for i in range(n)])
    v4_correct = np.array([majority["v4_twostage"][i] == truths[i] for i in range(n)])
    both = (v3_correct & v4_correct).sum()
    only_v3 = (v3_correct & ~v4_correct).sum()
    only_v4 = (~v3_correct & v4_correct).sum()
    neither = (~v3_correct & ~v4_correct).sum()

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    mat = np.array([[both, only_v3], [only_v4, neither]])
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.05)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if mat[i, j] > mat.max() * 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v3 correct", "v3 wrong"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["v4\\_twostage correct", "v4\\_twostage wrong"])
    ax.set_title(f"Agreement on the 10--20\\% bin (n={n} paragraphs)\n"
                 f"v4\\_twostage rescues {only_v4} that v3 got wrong; "
                 f"loses {only_v3} that v3 got right.",
                 fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig18_confusion_10_20.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
