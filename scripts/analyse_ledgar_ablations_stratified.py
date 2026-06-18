"""
LEDGAR per-regime ablation evaluation.

For each of {v3_pilot, v3_abl_shared_w_revert, v3_abl_hard_attention,
v3_abl_unit_coefs, v3_abl_no_drift_reg} × seeds {42..46}, load the checkpoint
and run density-stratified macro-F1. Output a (variant × density-bin) table
of 5-seed mean +/- std plus per-bin delta vs v3_pilot.

The 10-20% bin row is the critical one: if any ablation matches or beats
v3_pilot there (within 5-seed std), the corresponding architectural choice
has no per-regime justification and Chapter 7's mechanistic claim for that
component needs to be reframed.

Usage:
    python scripts/analyse_ledgar_ablations_stratified.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_baseline_mean_s{SEED}.pt           for SEED in {42..46}
    outputs/checkpoints/ledgar_v3_pilot_s{SEED}.pt                for SEED in {42..46}
    outputs/checkpoints/ledgar_v3_abl_{ABL}_s{SEED}.pt            for ABL × SEED

Outputs:
    outputs/logs/ledgar_per_regime_ablation_analysis.json
"""
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


SEEDS = [42, 43, 44, 45, 46]

# variant_tag -> kwargs for JusDefLEDGAR
VARIANTS = {
    "mean":                    {"dmp_variant": "mean"},
    "v3_pilot":                {"dmp_variant": "v3"},
    "v3_abl_shared_w_revert":  {"dmp_variant": "v3", "v3_shared_w_revert": True},
    "v3_abl_hard_attention":   {"dmp_variant": "v3", "v3_hard_attention": True},
    "v3_abl_unit_coefs":       {"dmp_variant": "v3", "v3_init_coefs": (1.0, 1.0, 1.0, 1.0)},
    "v3_abl_no_drift_reg":     {"dmp_variant": "v3", "v3_coef_reg_strength": 0.0},
}

# Checkpoint tag prefixes (the file names use "baseline_mean", "v3_pilot",
# "v3_abl_X" — same as the train_ledgar.py --tag values)
TAG_FILE_PREFIX = {
    "mean":                    "baseline_mean",
    "v3_pilot":                "v3_pilot",
    "v3_abl_shared_w_revert":  "v3_abl_shared_w_revert",
    "v3_abl_hard_attention":   "v3_abl_hard_attention",
    "v3_abl_unit_coefs":       "v3_abl_unit_coefs",
    "v3_abl_no_drift_reg":     "v3_abl_no_drift_reg",
}

BINS = [
    ("all",        0.00, 1.01),
    ("0pct",       0.00, 0.001),
    ("5_10",       0.05, 0.10),
    ("10_20",      0.10, 0.20),
    ("20_30",      0.20, 0.30),
    ("30_50",      0.30, 0.50),
    ("50_plus",    0.50, 1.01),
]


def build_model(variant_tag, device):
    kwargs = dict(in_dim=768, hidden_dim=512, num_classes=100, num_layers=1)
    kwargs.update(VARIANTS[variant_tag])
    return JusDefLEDGAR(**kwargs).to(device)


@torch.no_grad()
def predict_all(model, test_data, device):
    model.eval()
    preds = []
    for p in test_data:
        sent_embs = p["embeddings"].to(device)
        sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
        operators = torch.tensor(p["operators"], dtype=torch.long, device=device)
        logits = model(sent_embs, sent_to_para, operators, num_paragraphs=1)
        preds.append(int(logits.argmax(dim=-1).item()))
    return np.array(preds)


def stratified_f1(preds, labels_true, density):
    """Return dict bin_name -> macro_f1."""
    out = {}
    for name, lo, hi in BINS:
        mask = (density >= lo) & (density < hi)
        n = int(mask.sum())
        if n < 5:
            out[name] = {"n": n, "macro_f1": None}
            continue
        labels_sub = labels_true[mask]
        preds_sub = preds[mask]
        labels_present = sorted(set(labels_sub.tolist()))
        f1 = f1_score(labels_sub, preds_sub, labels=labels_present,
                      average="macro", zero_division=0)
        out[name] = {"n": n, "macro_f1": float(f1)}
    return out


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
    labels_true = np.array([d["label"] for d in test_data])

    # variant -> seed -> bin -> {n, macro_f1}
    results = {}

    ckpt_dir = Path("outputs/checkpoints")
    for variant in VARIANTS:
        print(f"\n=== Variant: {variant} ===")
        results[variant] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"ledgar_{TAG_FILE_PREFIX[variant]}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  seed {seed}: MISSING ({ckpt})")
                results[variant][seed] = None
                continue
            model = build_model(variant, device)
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device))
            except Exception as e:
                print(f"  seed {seed}: load failed: {e}")
                results[variant][seed] = None
                continue
            preds = predict_all(model, test_data, device)
            bin_results = stratified_f1(preds, labels_true, density)
            results[variant][seed] = bin_results
            f1_all = bin_results["all"]["macro_f1"]
            f1_10_20 = bin_results["10_20"]["macro_f1"]
            print(f"  seed {seed}: all={f1_all:.4f}  10-20%={f1_10_20:.4f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Aggregate: variant -> bin -> {mean, std, n_seeds_used}
    aggregate = {}
    for variant, seed_results in results.items():
        aggregate[variant] = {}
        for name, _, _ in BINS:
            vals = [
                seed_results[s][name]["macro_f1"]
                for s in SEEDS
                if seed_results[s] is not None
                   and seed_results[s][name]["macro_f1"] is not None
            ]
            if not vals:
                aggregate[variant][name] = {"mean": None, "std": None, "n_seeds": 0}
            else:
                aggregate[variant][name] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }

    # Display table: ablation vs v3_pilot per bin
    print("\n" + "=" * 100)
    print(" PER-REGIME ABLATION TABLE (5-seed mean +/- std macro-F1)")
    print("=" * 100)
    header = f"{'variant':<28}" + "".join(f"{name:>16}" for name, _, _ in BINS)
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        row = f"{variant:<28}"
        for name, _, _ in BINS:
            cell = aggregate[variant][name]
            if cell["mean"] is None:
                row += f"{'---':>16}"
            else:
                row += f"  {cell['mean']:.4f}+-{cell['std']:.3f}"
        print(row)

    # Delta vs v3_pilot for the 10-20% bin — the critical row
    print("\n" + "=" * 80)
    print(" DELTA vs v3_pilot ON 10-20% BIN (the critical regime)")
    print("=" * 80)
    base_10_20 = aggregate["v3_pilot"]["10_20"]
    for variant in VARIANTS:
        if variant == "v3_pilot":
            continue
        cell = aggregate[variant]["10_20"]
        if cell["mean"] is None or base_10_20["mean"] is None:
            print(f"  {variant:<28}  (incomplete)")
            continue
        delta = cell["mean"] - base_10_20["mean"]
        marker = ""
        # The "hurt" we want to see for ablations:
        if variant.startswith("v3_abl_"):
            if delta < -0.01:
                marker = "  <-- ablation HURTS v3 (architectural choice justified)"
            elif delta > 0.01:
                marker = "  <-- ablation HELPS — choice unjustified on this regime"
            else:
                marker = "  <-- ablation flat (within +-1pt)"
        print(f"  {variant:<28}  v3 {base_10_20['mean']:.4f}  abl {cell['mean']:.4f}  "
              f"delta {delta:+.4f}{marker}")

    # Save full result
    out_path = Path("outputs/logs/ledgar_per_regime_ablation_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "bins": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in BINS],
            "per_seed": {
                v: {str(s): results[v][s] for s in SEEDS}
                for v in VARIANTS
            },
            "aggregate": aggregate,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
