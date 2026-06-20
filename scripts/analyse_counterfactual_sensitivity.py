"""
Counterfactual operator-intervention sensitivity test.

For each LEDGAR test paragraph that contains at least one non-AFF operator,
we run two forward passes:

  ORIGINAL:        model(sent_embs, sent_to_para, operators_original)
  COUNTERFACTUAL:  model(sent_embs, sent_to_para, operators_all_AFF)

The counterfactual keeps the sentence embeddings identical and replaces every
non-AFF operator label with AFF (0). If a model uses operator information,
the prediction should shift; if a model ignores operators (like the mean
baseline by construction), the prediction is identical.

We measure two sensitivity metrics:

  1. Class-switch rate: fraction of paragraphs where argmax changes
  2. Mean absolute softmax shift on the true label

A v3 (and v4) sensitivity substantially above the mean baseline's zero
sensitivity is evidence that the architecture is *using* operator information,
not just consuming it as inert input. Chapter 6 currently lacks this evidence.

Usage:
    python scripts/analyse_counterfactual_sensitivity.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_baseline_mean_s{42..46}.pt
    outputs/checkpoints/ledgar_v3_pilot_s{42..46}.pt
    outputs/checkpoints/ledgar_v4_soft_s{42..46}.pt          (if available)

Outputs:
    outputs/logs/ledgar_counterfactual_sensitivity.json
    outputs/figures/fig9_counterfactual_sensitivity.pdf
"""
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


SEEDS = [42, 43, 44, 45, 46]

# variant_tag -> JusDefLEDGAR kwargs and file prefix
VARIANTS = {
    "mean":          ({"dmp_variant": "mean"},              "baseline_mean"),
    "v3_pilot":      ({"dmp_variant": "v3"},                "v3_pilot"),
    "v4_soft":       ({"dmp_variant": "v4_soft"},           "v4_soft"),
    "v4_hard":       ({"dmp_variant": "v4_hard",
                       "v4_density_lo": 0.10,
                       "v4_density_hi": 0.20},              "v4_hard"),
    "v4_twostage":   ({"dmp_variant": "v4_hard",
                       "v4_density_lo": 0.10,
                       "v4_density_hi": 0.20},              "v4_twostage"),
}


def build_model(variant_tag, device):
    kwargs, _ = VARIANTS[variant_tag]
    full = dict(in_dim=768, hidden_dim=512, num_classes=100, num_layers=1)
    full.update(kwargs)
    return JusDefLEDGAR(**full).to(device)


@torch.no_grad()
def predict_softmax(model, sent_embs, sent_to_para, operators, device):
    """Forward pass returning softmax probabilities."""
    model.eval()
    logits = model(sent_embs.to(device), sent_to_para.to(device),
                   operators.to(device), num_paragraphs=1)
    return F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading test data...")
    with open("data/processed_ledgar/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"  {len(test_data)} test paragraphs")

    # Pre-compute density per paragraph for filtering and stratification
    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in test_data
    ])

    # Only test paragraphs with at least one non-AFF operator (counterfactual
    # would be a no-op otherwise)
    eligible = [i for i, p in enumerate(test_data) if any(o != 0 for o in p["operators"])]
    print(f"  Eligible (has non-AFF operator): {len(eligible)} / {len(test_data)}")

    results = {}
    ckpt_dir = Path("outputs/checkpoints")

    for variant_tag, (_, file_prefix) in VARIANTS.items():
        print(f"\n=== Variant: {variant_tag} ===")
        results[variant_tag] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"ledgar_{file_prefix}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  seed {seed}: MISSING ({ckpt})")
                continue
            model = build_model(variant_tag, device)
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device))
            except Exception as e:
                print(f"  seed {seed}: load failed: {e}")
                continue

            n_class_switch = 0
            shifts_true_label = []
            shifts_l1_total = []
            for idx in eligible:
                p = test_data[idx]
                sent_embs = p["embeddings"]
                ops_orig = torch.tensor(p["operators"], dtype=torch.long)
                ops_cf = torch.zeros_like(ops_orig)
                sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long)
                true_label = int(p["label"])

                probs_orig = predict_softmax(model, sent_embs, sent_to_para,
                                             ops_orig, device)
                probs_cf = predict_softmax(model, sent_embs, sent_to_para,
                                           ops_cf, device)

                if int(np.argmax(probs_orig)) != int(np.argmax(probs_cf)):
                    n_class_switch += 1
                shifts_true_label.append(abs(probs_orig[true_label] - probs_cf[true_label]))
                shifts_l1_total.append(np.abs(probs_orig - probs_cf).sum())

            n = len(eligible)
            class_switch_rate = n_class_switch / n
            mean_true_shift = float(np.mean(shifts_true_label))
            mean_l1_shift = float(np.mean(shifts_l1_total))
            print(f"  seed {seed}: class_switch={class_switch_rate:.4f} "
                  f"true_shift={mean_true_shift:.4f} "
                  f"L1_shift={mean_l1_shift:.4f}")

            # Per-density-bin breakdown
            per_bin = {}
            for lo, hi, name in [(0.001, 0.10, "<10%"), (0.10, 0.20, "10-20%"),
                                  (0.20, 0.50, "20-50%"), (0.50, 1.01, ">=50%")]:
                mask = np.array([(density[idx] >= lo and density[idx] < hi)
                                 for idx in eligible])
                if mask.sum() < 5:
                    continue
                masked = mask
                sub_true = np.array(shifts_true_label)[masked]
                sub_l1 = np.array(shifts_l1_total)[masked]
                per_bin[name] = {
                    "n": int(masked.sum()),
                    "mean_true_shift": float(sub_true.mean()),
                    "mean_l1_shift": float(sub_l1.mean()),
                }

            results[variant_tag][seed] = {
                "n_eligible": n,
                "class_switch_rate": class_switch_rate,
                "mean_true_shift": mean_true_shift,
                "mean_l1_shift": mean_l1_shift,
                "per_density_bin": per_bin,
            }
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Aggregate 5-seed mean +/- std per variant
    print("\n" + "=" * 80)
    print(" COUNTERFACTUAL SENSITIVITY (5-seed mean +/- std)")
    print("=" * 80)
    print(f"{'variant':<14}{'class_switch':>18}{'true_shift':>18}{'L1_shift':>18}")
    print("-" * 80)

    agg = {}
    for variant_tag in VARIANTS:
        seeds_data = [results[variant_tag][s] for s in SEEDS
                      if s in results[variant_tag]]
        if not seeds_data:
            continue
        cs  = np.array([d["class_switch_rate"] for d in seeds_data])
        ts  = np.array([d["mean_true_shift"] for d in seeds_data])
        l1  = np.array([d["mean_l1_shift"] for d in seeds_data])
        agg[variant_tag] = {
            "class_switch_rate": {"mean": float(cs.mean()), "std": float(cs.std(ddof=1) if len(cs) > 1 else 0.0)},
            "mean_true_shift":  {"mean": float(ts.mean()), "std": float(ts.std(ddof=1) if len(ts) > 1 else 0.0)},
            "mean_l1_shift":    {"mean": float(l1.mean()), "std": float(l1.std(ddof=1) if len(l1) > 1 else 0.0)},
            "n_seeds": len(seeds_data),
        }
        print(f"{variant_tag:<14}"
              f"{cs.mean():.4f}+-{cs.std(ddof=1) if len(cs)>1 else 0:.3f}"
              f"      "
              f"{ts.mean():.4f}+-{ts.std(ddof=1) if len(ts)>1 else 0:.3f}"
              f"      "
              f"{l1.mean():.4f}+-{l1.std(ddof=1) if len(l1)>1 else 0:.3f}")

    # Verdict
    print("\n" + "=" * 80)
    print(" SENSITIVITY VERDICT")
    print("=" * 80)
    mean_sens = agg.get("mean", {}).get("class_switch_rate", {}).get("mean", 0.0)
    v3_sens = agg.get("v3_pilot", {}).get("class_switch_rate", {}).get("mean", 0.0)
    if mean_sens > 0.001:
        print(f"  Mean baseline class_switch_rate {mean_sens:.4f} > 0 -- this is")
        print("  unexpected; mean baseline should ignore operators completely.")
        print("  Check predict_softmax was called with the right inputs.")
    else:
        print(f"  Mean baseline class_switch_rate {mean_sens:.4f} (~0): confirms")
        print("  mean baseline IGNORES operators (as expected by construction).")
    if v3_sens > 0.05:
        print(f"  v3 class_switch_rate {v3_sens:.4f}: v3 USES operator info.")
        print("  The architecture is sensitive to its operator inputs --")
        print("  evidence that v3 is not trivially ignoring operators.")
    elif v3_sens > 0.01:
        print(f"  v3 class_switch_rate {v3_sens:.4f}: weak operator sensitivity.")
        print("  v3 uses operator info on some paragraphs but the effect is small.")
    else:
        print(f"  v3 class_switch_rate {v3_sens:.4f}: v3 LARGELY IGNORES operators.")
        print("  This is a major finding: the architectural claim that v3 uses")
        print("  operator information empirically is NOT SUPPORTED on this test.")

    # Save
    out_json = Path("outputs/logs/ledgar_counterfactual_sensitivity.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "n_eligible_paragraphs": len(eligible),
            "per_seed": {
                v: {str(s): results[v][s] for s in SEEDS if s in results[v]}
                for v in VARIANTS
            },
            "aggregate": agg,
        }, f, indent=2)
    print(f"\nSaved {out_json}")

    # Figure
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 10, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    variants_in_fig = [v for v in ["mean", "v3_pilot", "v4_soft", "v4_hard"]
                       if v in agg]
    labels = {"mean": "mean baseline", "v3_pilot": "v3 (main)",
              "v4_soft": "v4\\_soft", "v4_hard": "v4\\_hard"}
    colors = {"mean": "#0072B2", "v3_pilot": "#D55E00",
              "v4_soft": "#009E73", "v4_hard": "#CC79A7"}

    if variants_in_fig:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

        for ax, metric, title in [
            (axes[0], "class_switch_rate", "Class-switch rate"),
            (axes[1], "mean_true_shift",   "Mean $|\\Delta P(\\mathrm{true})|$"),
        ]:
            means = [agg[v][metric]["mean"] for v in variants_in_fig]
            stds = [agg[v][metric]["std"] for v in variants_in_fig]
            xs = np.arange(len(variants_in_fig))
            ax.bar(xs, means, yerr=stds, capsize=4,
                   color=[colors.get(v, "#999999") for v in variants_in_fig],
                   edgecolor="black", lw=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([labels.get(v, v) for v in variants_in_fig])
            ax.set_title(title)
            ax.set_ylabel("5-seed mean $\\pm$ std")

        fig.suptitle("Operator-intervention counterfactual sensitivity on LEDGAR",
                     fontsize=11)
        fig.tight_layout()
        out_fig = Path("outputs/figures/fig9_counterfactual_sensitivity.pdf")
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
