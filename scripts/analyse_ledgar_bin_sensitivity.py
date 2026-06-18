"""
Bin-sensitivity sweep for the LEDGAR operating regime.

Question this script answers: is the +0.0624 macro-F1 gain of v3 over the
mean baseline on the 10-20% non-AFF density bin specific to the (0.10, 0.20)
threshold choice, or does it persist across nearby bin definitions? A
reviewer's "cherry-picked threshold" objection is defused only if the gain
is robust to small shifts in the bin boundaries.

Method: for a panel of (lo, hi) bin definitions centred near the 10-20%
window, compute the 5-seed delta (v3 macro-F1 - mean macro-F1) on the
test paragraphs whose non-AFF density falls in [lo, hi). Each seed
contributes one delta; the panel reports mean +/- std across seeds.

Usage:
    python scripts/analyse_ledgar_bin_sensitivity.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_baseline_mean_s{42..46}.pt
    outputs/checkpoints/ledgar_v3_pilot_s{42..46}.pt

Outputs:
    outputs/logs/ledgar_bin_sensitivity_analysis.json
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

# Sweep: shifted bins around the 10-20% window plus two wider/narrower controls
BIN_SWEEP = [
    ("8-22",   0.08, 0.22),
    ("9-21",   0.09, 0.21),
    ("10-20",  0.10, 0.20),   # canonical
    ("11-19",  0.11, 0.19),
    ("12-18",  0.12, 0.18),
    ("8-15",   0.08, 0.15),   # asymmetric: lower
    ("15-25",  0.15, 0.25),   # asymmetric: upper
]


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


def f1_on_subset(preds, labels_true, mask):
    n = int(mask.sum())
    if n < 5:
        return None, n
    labels_sub = labels_true[mask]
    preds_sub = preds[mask]
    labels_present = sorted(set(labels_sub.tolist()))
    return float(f1_score(labels_sub, preds_sub, labels=labels_present,
                          average="macro", zero_division=0)), n


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

    # Per-seed predictions for v3 and mean
    print("\nCollecting predictions for each seed...")
    ckpt_dir = Path("outputs/checkpoints")

    preds_mean_per_seed = {}
    preds_v3_per_seed = {}

    for seed in SEEDS:
        for tag, store, kwargs in [
            ("baseline_mean", preds_mean_per_seed, {"dmp_variant": "mean"}),
            ("v3_pilot",      preds_v3_per_seed,   {"dmp_variant": "v3"}),
        ]:
            ckpt = ckpt_dir / f"ledgar_{tag}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  MISSING: {ckpt}")
                store[seed] = None
                continue
            model = JusDefLEDGAR(
                in_dim=768, hidden_dim=512, num_classes=100, num_layers=1,
                **kwargs,
            ).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            store[seed] = predict_all(model, test_data, device)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  seed {seed} {tag}: predictions cached")

    # Sweep bins
    print("\n" + "=" * 90)
    print(" BIN-SENSITIVITY SWEEP (5-seed mean +/- std delta)")
    print("=" * 90)
    print(f"{'bin':<10} {'N':>6} {'v3 mean':>12} {'mean mean':>12} {'delta':>20}")
    print("-" * 90)

    per_bin = {}
    for name, lo, hi in BIN_SWEEP:
        mask = (density >= lo) & (density < hi)
        n = int(mask.sum())
        per_seed_deltas = []
        per_seed_v3 = []
        per_seed_mean = []
        for seed in SEEDS:
            pm = preds_mean_per_seed[seed]
            pv = preds_v3_per_seed[seed]
            if pm is None or pv is None:
                continue
            f1_v3, _   = f1_on_subset(pv, labels_true, mask)
            f1_mean, _ = f1_on_subset(pm, labels_true, mask)
            if f1_v3 is None or f1_mean is None:
                continue
            per_seed_deltas.append(f1_v3 - f1_mean)
            per_seed_v3.append(f1_v3)
            per_seed_mean.append(f1_mean)

        if per_seed_deltas:
            d_mean = float(np.mean(per_seed_deltas))
            d_std  = float(np.std(per_seed_deltas, ddof=1)) if len(per_seed_deltas) > 1 else 0.0
            v3_mean = float(np.mean(per_seed_v3))
            mean_mean = float(np.mean(per_seed_mean))
            sign = "+" if d_mean >= 0 else "-"
            print(f"{name:<10} {n:>6} {v3_mean:>12.4f} {mean_mean:>12.4f} "
                  f"{sign}{abs(d_mean):.4f} +- {d_std:.4f}")
            per_bin[name] = {
                "lo": lo, "hi": hi, "n_paragraphs": n,
                "v3_mean": v3_mean, "mean_mean": mean_mean,
                "delta_mean": d_mean, "delta_std": d_std,
                "per_seed_delta": per_seed_deltas,
                "n_seeds": len(per_seed_deltas),
            }
        else:
            print(f"{name:<10} {n:>6}  (no usable seeds)")
            per_bin[name] = {"lo": lo, "hi": hi, "n_paragraphs": n,
                             "delta_mean": None, "delta_std": None}

    # Save
    out_path = Path("outputs/logs/ledgar_bin_sensitivity_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "sweep": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in BIN_SWEEP],
            "per_bin": per_bin,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Quick verdict
    print("\n" + "=" * 50)
    print(" VERDICT")
    print("=" * 50)
    deltas_5pt_wide = [per_bin[k]["delta_mean"] for k in
                       ["8-22", "9-21", "10-20", "11-19", "12-18"]
                       if per_bin.get(k, {}).get("delta_mean") is not None]
    if deltas_5pt_wide and all(d > 0 for d in deltas_5pt_wide):
        print(f"v3 wins on ALL 5 shifted bins (8-22, 9-21, 10-20, 11-19, 12-18).")
        print(f"Range of deltas: [{min(deltas_5pt_wide):+.4f}, "
              f"{max(deltas_5pt_wide):+.4f}]")
        print("Cherry-pick objection defused: the operating-regime win is")
        print("robust to symmetric shifts in the bin boundaries.")
    else:
        print("v3 win does NOT persist across all shifted bins. The 10-20%")
        print("choice may be threshold-specific; reframe accordingly in Ch. 8.")


if __name__ == "__main__":
    main()
