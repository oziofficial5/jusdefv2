"""
Density-stratified evaluation on CUAD, mirroring the LEDGAR analysis.

Loads converged v3 and mean-baseline CUAD checkpoints, runs forward passes
on the CUAD test set, and reports per-density-bin macro-F1 with the
v3 vs mean delta per bin. Cross-corpus regime test against LEDGAR's
+0.0624 finding on the 10-20% bin.

Usage:
    python scripts/analyse_cuad_density_subset.py
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

VARIANTS = {
    "mean":     ({"dmp_variant": "mean"}, "cuad_baseline_mean"),
    "v3_pilot": ({"dmp_variant": "v3"},   "cuad_v3"),
}

BINS = [
    ("all",       0.00, 1.01),
    ("0%",        0.00, 0.001),
    ("0-5%",      0.001, 0.05),
    ("5-10%",     0.05, 0.10),
    ("10-20%",    0.10, 0.20),
    ("20-30%",    0.20, 0.30),
    ("30-50%",    0.30, 0.50),
    ("50%+",      0.50, 1.01),
]


def detect_num_classes(test_data):
    return int(max(p["label"] for p in test_data)) + 1


def build_model(tag, hidden_dim, num_classes, device):
    kwargs, _ = VARIANTS[tag]
    full = dict(in_dim=768, hidden_dim=hidden_dim, num_classes=num_classes,
                num_layers=1)
    full.update(kwargs)
    return JusDefLEDGAR(**full).to(device)


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open("data/processed_cuad/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"  CUAD test paragraphs: {len(test_data)}")

    num_classes = detect_num_classes(test_data)
    print(f"  num_classes: {num_classes}")

    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in test_data
    ])
    labels_true = np.array([d["label"] for d in test_data])

    print("\nCUAD test density distribution:")
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        n = int((density >= thresh).sum())
        print(f"  density >= {thresh*100:5.1f}%: {n:4d} paragraphs "
              f"({n/len(test_data)*100:5.2f}%)")
    print(f"  10-20% bin: {int(((density >= 0.10) & (density < 0.20)).sum())} paragraphs")

    ckpt_dir = Path("outputs/checkpoints")
    results = {"num_classes": num_classes, "per_variant": {}}

    for tag, (_, file_prefix) in VARIANTS.items():
        print(f"\n=== {tag} ===")
        results["per_variant"][tag] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"{file_prefix}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  [skip] s{seed}: {ckpt} missing")
                continue
            # detect hidden_dim from state dict
            state = torch.load(ckpt, map_location=device)
            hidden_dim = state.get("input_proj.weight",
                                   torch.zeros(512, 768)).shape[0]
            model = build_model(tag, hidden_dim, num_classes, device)
            model.load_state_dict(state, strict=False)
            preds = predict_all(model, test_data, device)
            per_bin = {}
            for name, lo, hi in BINS:
                mask = (density >= lo) & (density < hi)
                n = int(mask.sum())
                if n < 5:
                    per_bin[name] = {"n": n, "macro_f1": None}
                    continue
                labels_sub = labels_true[mask]
                preds_sub = preds[mask]
                labels_present = sorted(set(labels_sub.tolist()))
                f1 = f1_score(labels_sub, preds_sub, labels=labels_present,
                              average="macro", zero_division=0)
                per_bin[name] = {"n": n, "macro_f1": float(f1)}
            results["per_variant"][tag][seed] = per_bin
            print(f"  s{seed}: all={per_bin['all']['macro_f1']:.4f}  "
                  f"10-20%={per_bin['10-20%']['macro_f1']} "
                  f"(n={per_bin['10-20%']['n']})")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Aggregate
    print("\n" + "=" * 90)
    print(" CUAD PER-REGIME EVALUATION (5-seed mean +- std)")
    print("=" * 90)
    header = f"{'variant':<14}" + "".join(f"{n:>14}" for n, _, _ in BINS)
    print(header)
    print("-" * len(header))
    agg = {}
    for tag in VARIANTS:
        row = f"{tag:<14}"
        agg[tag] = {}
        for name, _, _ in BINS:
            vals = [results["per_variant"][tag][s][name]["macro_f1"]
                    for s in SEEDS
                    if s in results["per_variant"][tag]
                    and results["per_variant"][tag][s][name]["macro_f1"] is not None]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                agg[tag][name] = {"mean": m, "std": s, "n_seeds": len(vals)}
                row += f"  {m:.4f}+-{s:.3f}"
            else:
                agg[tag][name] = {"mean": None, "std": None, "n_seeds": 0}
                row += f"{'---':>14}"
        print(row)

    # Cross-corpus regime verdict
    if "v3_pilot" in agg and "mean" in agg:
        print("\n" + "=" * 70)
        print(" CUAD CROSS-CORPUS REGIME VERDICT")
        print("=" * 70)
        agg_v3 = agg["v3_pilot"]
        agg_mean = agg["mean"]
        for name in ["all", "10-20%"]:
            v3m = agg_v3[name]["mean"]
            mnm = agg_mean[name]["mean"]
            if v3m is None or mnm is None:
                continue
            d = v3m - mnm
            sign = "+" if d >= 0 else "-"
            print(f"  {name:<10} v3 {v3m:.4f}  mean {mnm:.4f}  delta {sign}{abs(d):.4f}")
        d_regime = agg_v3.get("10-20%", {}).get("mean")
        m_regime = agg_mean.get("10-20%", {}).get("mean")
        if d_regime is not None and m_regime is not None:
            delta = d_regime - m_regime
            if delta > 0.02:
                print(f"\n  -> CUAD REGIME WIN: v3 outperforms mean by +{delta:.4f}")
                print(f"     Cross-corpus replication of LEDGAR finding SUPPORTED")
            elif delta < -0.02:
                print(f"\n  -> CUAD REGIME LOSS: v3 underperforms mean by {delta:.4f}")
                print(f"     Cross-corpus replication NOT supported")
            else:
                print(f"\n  -> CUAD REGIME NEUTRAL: delta = {delta:+.4f}")
                print(f"     Within seed-to-seed noise; inconclusive")

    out_path = Path("outputs/logs/cuad_density_stratified.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"aggregate": agg, "per_seed": results}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
