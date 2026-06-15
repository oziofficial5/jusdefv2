"""
LEDGAR Y_exc-equivalent analysis.

Compares mean baseline vs v3 on density-stratified subsets of the LEDGAR
test set. If v3 outperforms the baseline on defeasibility-dense paragraphs
(even if it underperforms overall), this is the positive empirical result.

Usage:
    python scripts/analyse_ledgar_density_subset.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_baseline_mean_s42.pt
    outputs/checkpoints/ledgar_v3_pilot_s42.pt

Outputs:
    outputs/logs/ledgar_density_stratified_analysis.json
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


def collate_one(p, device):
    sent_embs = p["embeddings"].to(device)
    sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
    operators = torch.tensor(p["operators"], dtype=torch.long, device=device)
    return sent_embs, sent_to_para, operators


@torch.no_grad()
def predict_all(model, test_data, device):
    model.eval()
    preds = []
    for p in test_data:
        sent_embs, sent_to_para, operators = collate_one(p, device)
        logits = model(sent_embs, sent_to_para, operators, num_paragraphs=1)
        preds.append(int(logits.argmax(dim=-1).item()))
    return preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test data
    print("Loading test data...")
    with open("data/processed_ledgar/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"  {len(test_data)} test paragraphs")

    # Compute non-AFF density per paragraph
    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in test_data
    ])
    labels_true = np.array([d["label"] for d in test_data])

    print(f"\nDensity distribution across test set:")
    for thresh in [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]:
        n = (density >= thresh).sum()
        print(f"  density >= {thresh*100:5.1f}%: {n:5d} paragraphs ({n/len(density)*100:5.1f}%)")

    # Load both models
    print("\nLoading models...")
    model_mean = JusDefLEDGAR(
        in_dim=768, hidden_dim=512, num_classes=100,
        num_layers=1, dmp_variant="mean"
    ).to(device)
    model_mean.load_state_dict(torch.load(
        "outputs/checkpoints/ledgar_baseline_mean_s42.pt", map_location=device
    ))

    model_v3 = JusDefLEDGAR(
        in_dim=768, hidden_dim=512, num_classes=100,
        num_layers=1, dmp_variant="v3"
    ).to(device)
    model_v3.load_state_dict(torch.load(
        "outputs/checkpoints/ledgar_v3_pilot_s42.pt", map_location=device
    ))

    # Get predictions
    print("Running mean baseline predictions...")
    preds_mean = predict_all(model_mean, test_data, device)
    print("Running v3 predictions...")
    preds_v3 = predict_all(model_v3, test_data, device)

    preds_mean = np.array(preds_mean)
    preds_v3 = np.array(preds_v3)

    # Stratified evaluation
    print("\n" + "=" * 70)
    print(" DENSITY-STRATIFIED EVALUATION")
    print("=" * 70)
    print(f"{'Density range':<25} {'N':>8} {'mean_F1':>10} {'v3_F1':>10} {'Δ (v3-mean)':>14}")
    print("-" * 70)

    bins = [
        ("all paragraphs", 0.00, 1.01),
        ("0% (AFF only)", 0.00, 0.001),
        ("0-5%", 0.001, 0.05),
        ("5-10%", 0.05, 0.10),
        ("10-20%", 0.10, 0.20),
        ("20-30%", 0.20, 0.30),
        ("30-50%", 0.30, 0.50),
        ("50%+", 0.50, 1.01),
        ("== >= 10% (Y_exc-like)", 0.10, 1.01),
        ("== >= 20% dense subset", 0.20, 1.01),
        ("== >= 30% very dense", 0.30, 1.01),
    ]

    results = {}
    for name, lo, hi in bins:
        mask = (density >= lo) & (density < hi)
        n = mask.sum()
        if n < 5:
            print(f"{name:<25} {n:>8} {'(too few)':>10}")
            continue
        labels_sub = labels_true[mask]
        mean_sub = preds_mean[mask]
        v3_sub = preds_v3[mask]
        # Macro-F1 over only the labels present in this subset
        labels_present = sorted(set(labels_sub.tolist()))
        f1_mean = f1_score(labels_sub, mean_sub, labels=labels_present,
                           average="macro", zero_division=0)
        f1_v3 = f1_score(labels_sub, v3_sub, labels=labels_present,
                         average="macro", zero_division=0)
        delta = f1_v3 - f1_mean
        marker = " <-- v3 wins" if delta > 0 else ""
        print(f"{name:<25} {n:>8} {f1_mean:>10.4f} {f1_v3:>10.4f} {delta:>14.4f}{marker}")
        results[name] = {
            "n_paragraphs": int(n),
            "mean_macro_f1": float(f1_mean),
            "v3_macro_f1": float(f1_v3),
            "delta": float(delta),
            "n_labels_present": len(labels_present),
        }

    # Save full result
    out_path = Path("outputs/logs/ledgar_density_stratified_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_test_paragraphs": len(test_data),
            "density_distribution": {
                str(t): int((density >= t).sum())
                for t in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
            },
            "stratified_f1": results,
        }, f, indent=2)
    print(f"\nFull analysis saved to {out_path}")

    # Decision summary
    print("\n" + "=" * 70)
    print(" DECISION SUMMARY")
    print("=" * 70)
    overall_delta = results["all paragraphs"]["delta"]
    dense_keys = [k for k in results if "==" in k]
    dense_wins = [k for k in dense_keys if results[k]["delta"] > 0]
    if dense_wins and overall_delta < 0:
        print("VERDICT: v3 wins on defeasibility-dense subsets but loses overall.")
        print("         This is a publishable POSITIVE result with calibration.")
        for k in dense_wins:
            print(f"         {k}: v3 +{results[k]['delta']:.4f} F1")
    elif overall_delta > 0:
        print("VERDICT: v3 wins overall. Strong positive empirical result.")
    else:
        print("VERDICT: v3 loses across the board. Commit to negative-result thesis.")


if __name__ == "__main__":
    main()
