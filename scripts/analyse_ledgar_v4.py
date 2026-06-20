"""
Density-stratified evaluation of v4 routers against v3, mean baseline, and
the four v3 ablations.

For each of {mean, v3_pilot, v4_hard, v4_soft} (and optionally
v4_hard_alt threshold variants) x seeds {42..46} x density bins, load
the checkpoint and report 5-seed mean +/- std macro-F1 plus delta vs
v3_pilot on every bin.

The critical checks:
    aggregate row:
        v4 should improve over v3 (since 65% of paragraphs route to mean)
    10-20% bin row:
        v4 should preserve v3's regime win

Usage:
    python scripts/analyse_ledgar_v4.py

Reads:
    data/processed_ledgar/test_processed.pkl
    outputs/checkpoints/ledgar_{baseline_mean,v3_pilot,v4_hard,v4_soft}_s{42..46}.pt

Outputs:
    outputs/logs/ledgar_v4_analysis.json
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

# variant_tag -> JusDefLEDGAR kwargs (excluding hidden_dim/in_dim/num_classes/num_layers)
VARIANTS = {
    "mean":          {"dmp_variant": "mean"},
    "v3_pilot":      {"dmp_variant": "v3"},
    "v4_hard":       {"dmp_variant": "v4_hard", "v4_density_lo": 0.10, "v4_density_hi": 0.20},
    "v4_soft":       {"dmp_variant": "v4_soft"},
    "v4_twostage":   {"dmp_variant": "v4_hard", "v4_density_lo": 0.10, "v4_density_hi": 0.20},
}

TAG_FILE_PREFIX = {
    "mean":          "baseline_mean",
    "v3_pilot":      "v3_pilot",
    "v4_hard":       "v4_hard",
    "v4_soft":       "v4_soft",
    "v4_twostage":   "v4_twostage",
}

BINS = [
    ("all",       0.00, 1.01),
    ("0pct",      0.00, 0.001),
    ("5_10",      0.05, 0.10),
    ("10_20",     0.10, 0.20),
    ("20_30",     0.20, 0.30),
    ("30_50",     0.30, 0.50),
    ("50_plus",   0.50, 1.01),
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

    # Print v4 summary table
    print("\n" + "=" * 100)
    print(" V4 PER-REGIME EVALUATION (5-seed mean +/- std macro-F1)")
    print("=" * 100)
    header = f"{'variant':<14}" + "".join(f"{name:>16}" for name, _, _ in BINS)
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        row = f"{variant:<14}"
        for name, _, _ in BINS:
            cell = aggregate[variant][name]
            if cell["mean"] is None:
                row += f"{'---':>16}"
            else:
                row += f"  {cell['mean']:.4f}+-{cell['std']:.3f}"
        print(row)

    # Verdict: v4 vs v3 on aggregate and on 10-20% bin
    print("\n" + "=" * 80)
    print(" V4 VERDICT")
    print("=" * 80)
    v3_all = aggregate["v3_pilot"]["all"]
    v3_10_20 = aggregate["v3_pilot"]["10_20"]
    mean_all = aggregate["mean"]["all"]
    for v in ["v4_hard", "v4_soft"]:
        a = aggregate[v]["all"]
        r = aggregate[v]["10_20"]
        if a["mean"] is None:
            print(f"  {v}: no usable seeds")
            continue
        d_agg = a["mean"] - v3_all["mean"]
        d_mean = a["mean"] - mean_all["mean"]
        d_reg = r["mean"] - v3_10_20["mean"]
        print(f"  {v}:")
        print(f"    aggregate:   {a['mean']:.4f}  "
              f"(vs v3 {d_agg:+.4f}, vs mean {d_mean:+.4f})")
        print(f"    10-20% bin:  {r['mean']:.4f}  "
              f"(vs v3 {d_reg:+.4f})")
        if d_agg > 0.005 and abs(d_reg) < 0.01:
            print(f"    -> STRONG WIN: aggregate gain, regime preserved")
        elif d_agg > 0 and d_reg > -0.02:
            print(f"    -> OK: small aggregate gain, regime mostly preserved")
        elif d_agg < 0 and d_reg < -0.02:
            print(f"    -> FAIL: lost on both aggregate and regime; reconsider")
        else:
            print(f"    -> MIXED: examine the per-seed table before concluding")

    out_path = Path("outputs/logs/ledgar_v4_analysis.json")
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
