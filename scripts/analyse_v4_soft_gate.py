"""
Diagnostic: what did v4_soft actually learn?

Loads the 5 v4_soft LEDGAR checkpoints, forward-passes the test set,
intercepts the per-paragraph gate value, and reports the learned gate
as a function of paragraph non-AFF density.

The critical questions:
  1. Is the learned gate ~1 everywhere (router didn't differentiate),
     or does it vary with density (router IS routing)?
  2. If it varies, does it peak near the 10-20% operating regime
     (matches our discovered regime) or peak elsewhere (mismatch)?
  3. Does the per-paragraph gate predict per-paragraph F1 improvement?

Usage:
    python scripts/analyse_v4_soft_gate.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_v4_soft_s{42..46}.pt

Outputs:
    outputs/logs/ledgar_v4_soft_gate_analysis.json
    outputs/figures/fig6_v4_soft_learned_gate.pdf
"""
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


SEEDS = [42, 43, 44, 45, 46]

# Density bins for aggregation
DENSITY_BINS = [
    (0.00, 0.001, "0%"),
    (0.001, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, 0.20, "15-20%"),
    (0.20, 0.30, "20-30%"),
    (0.30, 0.50, "30-50%"),
    (0.50, 1.01, "50%+"),
]


@torch.no_grad()
def collect_gate_values(model, test_data, device):
    """Run forward pass for each paragraph, extract the per-paragraph gate."""
    model.eval()
    gates = []
    for p in test_data:
        sent_embs = p["embeddings"].to(device)
        sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
        operators = torch.tensor(p["operators"], dtype=torch.long, device=device)

        # Replicate the model's density computation
        density, n_non_aff, n_total = model._compute_density_features(
            operators, sent_to_para, num_paragraphs=1
        )
        gate = model.router(density, n_non_aff, n_total)  # (1, 1)
        gates.append(float(gate.item()))
    return np.array(gates)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading test data...")
    with open("data/processed_ledgar/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"  {len(test_data)} test paragraphs")

    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in test_data
    ])

    # Collect gate values for each seed
    print("\nForward-passing the test set for each seed...")
    ckpt_dir = Path("outputs/checkpoints")
    per_seed_gates = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"ledgar_v4_soft_s{seed}.pt"
        if not ckpt.is_file():
            print(f"  seed {seed}: MISSING ({ckpt})")
            continue
        model = JusDefLEDGAR(
            in_dim=768, hidden_dim=512, num_classes=100, num_layers=1,
            dmp_variant="v4_soft",
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        gates = collect_gate_values(model, test_data, device)
        per_seed_gates[seed] = gates
        print(f"  seed {seed}: mean gate {gates.mean():.4f}  "
              f"min {gates.min():.4f}  max {gates.max():.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not per_seed_gates:
        print("No seeds loaded; aborting.")
        return

    # Aggregate per-bin
    print("\n" + "=" * 80)
    print(" LEARNED GATE BY DENSITY BIN (5-seed mean +/- std)")
    print("=" * 80)
    print(f"{'bin':<10} {'N':>6} {'gate_mean':>12} {'gate_std':>10} "
          f"{'gate_min':>10} {'gate_max':>10}")
    print("-" * 80)

    per_bin = {}
    for lo, hi, name in DENSITY_BINS:
        mask = (density >= lo) & (density < hi)
        n = int(mask.sum())
        if n < 5:
            print(f"{name:<10} {n:>6}  (too few)")
            continue
        # 5-seed mean gate in this bin
        per_seed_means = [g[mask].mean() for g in per_seed_gates.values()]
        per_seed_all = np.concatenate([g[mask] for g in per_seed_gates.values()])
        per_bin[name] = {
            "n_paragraphs": n,
            "gate_mean_across_seeds": float(np.mean(per_seed_means)),
            "gate_std_across_seeds":  float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0,
            "gate_min": float(per_seed_all.min()),
            "gate_max": float(per_seed_all.max()),
        }
        print(f"{name:<10} {n:>6} "
              f"{per_bin[name]['gate_mean_across_seeds']:>12.4f} "
              f"{per_bin[name]['gate_std_across_seeds']:>10.4f} "
              f"{per_bin[name]['gate_min']:>10.4f} "
              f"{per_bin[name]['gate_max']:>10.4f}")

    # Verdict
    print("\n" + "=" * 80)
    print(" GATE VERDICT")
    print("=" * 80)
    mean_gate_overall = np.mean([g.mean() for g in per_seed_gates.values()])
    std_gate_overall = np.mean([g.std() for g in per_seed_gates.values()])
    print(f"Overall mean gate (averaged over seeds): {mean_gate_overall:.4f}")
    print(f"Overall std of gate within each seed:    {std_gate_overall:.4f}")

    if std_gate_overall < 0.02:
        print("VERDICT: router output is essentially constant. The MLP learned")
        print("to ignore its inputs. v4_soft is effectively v3 with a useless")
        print("scaling factor. Aggregate gain (+0.0022) is noise.")
    elif mean_gate_overall > 0.95:
        print("VERDICT: gate close to 1 everywhere. Router learned that the v3")
        print("contribution is approximately always helpful; the small aggregate")
        print("gain comes from the few paragraphs where gate dropped below 1.")
    elif 0.5 < mean_gate_overall < 0.9:
        print("VERDICT: router is actively routing. Inspect per-bin pattern:")
        if "10-15%" in per_bin and "0%" in per_bin:
            g_in = per_bin["10-15%"]["gate_mean_across_seeds"]
            g_out = per_bin["0%"]["gate_mean_across_seeds"]
            if g_in > g_out + 0.05:
                print("  -> gate at 10-15% bin exceeds gate at 0% bin by "
                      f"{g_in - g_out:.3f}. Routing matches discovered regime.")
            else:
                print("  -> per-bin gate does NOT track the discovered regime.")
                print("     The router is doing something, but not the intended routing.")
    else:
        print("VERDICT: router suppressing v3 strongly. Inspect per-bin pattern.")

    # Save
    out_json = Path("outputs/logs/ledgar_v4_soft_gate_analysis.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "seeds": list(per_seed_gates.keys()),
            "overall": {
                "gate_mean_across_seeds": float(mean_gate_overall),
                "gate_std_within_seed":   float(std_gate_overall),
            },
            "per_bin": per_bin,
        }, f, indent=2)
    print(f"\nSaved {out_json}")

    # Figure
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 10, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })

    names = list(per_bin.keys())
    means = [per_bin[n]["gate_mean_across_seeds"] for n in names]
    stds = [per_bin[n]["gate_std_across_seeds"] for n in names]
    ns = [per_bin[n]["n_paragraphs"] for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(names))
    # Highlight the operating-regime bins (10-15 and 15-20)
    colors = ["#E69F00" if n in ("10-15%", "15-20%") else "#0072B2" for n in names]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", lw=0.5)
    for i, n in enumerate(ns):
        ax.text(i, means[i] + stds[i] + 0.01, f"N={n}", ha="center",
                va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_xlabel("non-AFF density bin")
    ax.set_ylabel("v4\\_soft learned gate (5-seed mean $\\pm$ std)")
    ax.set_ylim(0, max(1.05, max(m + s for m, s in zip(means, stds)) * 1.1))
    ax.axhline(1.0, color="grey", lw=0.7, linestyle="--", zorder=1)
    ax.set_title("v4\\_soft: learned gate as a function of paragraph density")

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig = fig_dir / "fig6_v4_soft_learned_gate.pdf"
    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
