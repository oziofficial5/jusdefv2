"""
Paired-bootstrap significance for v4_twostage on the LEDGAR 10-20% operating regime.

Three comparisons, each with 10,000 paired resamples on per-paragraph F1
contributions, reported per seed and pooled across seeds:

    1. v4_twostage vs v3 on the 10-20% bin (156 paragraphs)
    2. v4_twostage vs mean baseline on the 10-20% bin
    3. v4_twostage vs v3 on the aggregate test set (10,000 paragraphs)

For multi-class macro-F1 with per-paragraph predictions, "per-paragraph
F1 contribution" reduces to the per-paragraph correctness indicator: a
paired bootstrap on `pred == label` directly measures whether the two
variants differ in classification accuracy, which is what dominates macro-F1
on a multi-class single-label task with this many classes.

Usage:
    python scripts/bootstrap_v4_twostage.py

Outputs:
    outputs/logs/bootstrap_v4_twostage.json
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


import os as _os
SEEDS = [int(x) for x in _os.environ.get(
    "SEEDS", "42 43 44 45 46 47 48 49 50 51").split()]
N_BOOTSTRAP = 10000
RNG_SEED = 12345

VARIANTS = {
    "mean":         ({"dmp_variant": "mean"},          "baseline_mean"),
    "v3_pilot":     ({"dmp_variant": "v3"},            "v3_pilot"),
    "v4_twostage":  ({"dmp_variant": "v4_hard",
                      "v4_density_lo": 0.10,
                      "v4_density_hi": 0.20},          "v4_twostage"),
}


def build_model(tag, device):
    kwargs, _ = VARIANTS[tag]
    full = dict(in_dim=768, hidden_dim=512, num_classes=100, num_layers=1)
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


def paired_bootstrap_pvalue(preds_a, preds_b, labels, indices, rng,
                             n_resamples=N_BOOTSTRAP):
    """One-sided paired bootstrap: P(macro-F1(a) <= macro-F1(b))."""
    n = len(indices)
    deltas = np.zeros(n_resamples)
    # Base macro-F1 difference (a - b) on the subset
    labels_sub = labels[indices]
    a_sub = preds_a[indices]
    b_sub = preds_b[indices]
    labels_present = sorted(set(labels_sub.tolist()))
    base_a = f1_score(labels_sub, a_sub, labels=labels_present, average="macro", zero_division=0)
    base_b = f1_score(labels_sub, b_sub, labels=labels_present, average="macro", zero_division=0)
    base_delta = base_a - base_b

    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        l = labels_sub[idx]
        a = a_sub[idx]
        b = b_sub[idx]
        lp = sorted(set(l.tolist()))
        fa = f1_score(l, a, labels=lp, average="macro", zero_division=0)
        fb = f1_score(l, b, labels=lp, average="macro", zero_division=0)
        deltas[i] = fa - fb

    # Two-sided p-value: fraction of resamples where the sign flips
    if base_delta > 0:
        p_one_sided = float((deltas <= 0).mean())
    else:
        p_one_sided = float((deltas >= 0).mean())
    p_two_sided = 2 * p_one_sided
    p_two_sided = min(p_two_sided, 1.0)

    return {
        "base_delta": float(base_delta),
        "base_macro_f1_a": float(base_a),
        "base_macro_f1_b": float(base_b),
        "p_one_sided": p_one_sided,
        "p_two_sided": float(p_two_sided),
        "delta_mean_bootstrap": float(deltas.mean()),
        "delta_std_bootstrap": float(deltas.std()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }


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
    mask_10_20 = (density >= 0.10) & (density < 0.20)
    idx_10_20 = np.where(mask_10_20)[0]
    idx_all = np.arange(len(test_data))
    print(f"  10-20% bin: {len(idx_10_20)} paragraphs")
    print(f"  Aggregate: {len(idx_all)} paragraphs")

    # Collect per-seed predictions for all three variants
    ckpt_dir = Path("outputs/checkpoints")
    all_preds = {}
    for tag, (_, file_prefix) in VARIANTS.items():
        all_preds[tag] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"ledgar_{file_prefix}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  [skip] {tag} s{seed}: {ckpt} missing")
                continue
            model = build_model(tag, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            preds = predict_all(model, test_data, device)
            all_preds[tag][seed] = preds
            print(f"  cached {tag} s{seed}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    rng = np.random.default_rng(RNG_SEED)

    results = {
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "comparisons": {},
    }

    for cmp_name, a_tag, b_tag, mask_idx in [
        ("v4_twostage_vs_v3_on_10_20",     "v4_twostage", "v3_pilot", idx_10_20),
        ("v4_twostage_vs_mean_on_10_20",   "v4_twostage", "mean",     idx_10_20),
        ("v4_twostage_vs_v3_on_aggregate", "v4_twostage", "v3_pilot", idx_all),
    ]:
        print(f"\n--- {cmp_name} ---")
        per_seed = {}
        for seed in SEEDS:
            if seed not in all_preds[a_tag] or seed not in all_preds[b_tag]:
                continue
            r = paired_bootstrap_pvalue(
                all_preds[a_tag][seed], all_preds[b_tag][seed],
                labels_true, mask_idx, rng,
            )
            per_seed[seed] = r
            print(f"  seed {seed}: base_delta={r['base_delta']:+.4f}  "
                  f"p_two_sided={r['p_two_sided']:.4f}  "
                  f"95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
        # Pooled: stack predictions across seeds, run one bootstrap with seed labels as factor
        # Simpler aggregation: report Fisher-combined p-value
        ps = [r["p_two_sided"] for r in per_seed.values() if r["p_two_sided"] > 0]
        if ps:
            chi2 = -2 * sum(np.log(p) for p in ps)
            df = 2 * len(ps)
            from scipy.stats import chi2 as chi2_dist
            p_fisher = float(1 - chi2_dist.cdf(chi2, df))
            print(f"  Fisher-pooled p-value (n_seeds={len(ps)}): {p_fisher:.5f}")
        else:
            p_fisher = None
            print("  (no non-zero p-values to pool)")
        results["comparisons"][cmp_name] = {
            "per_seed": per_seed,
            "fisher_pooled_p_value": p_fisher,
        }

    out_path = Path("outputs/logs/bootstrap_v4_twostage.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Verdict
    print("\n" + "=" * 70)
    print(" BOOTSTRAP VERDICT")
    print("=" * 70)
    for cmp_name, rec in results["comparisons"].items():
        p = rec.get("fisher_pooled_p_value")
        n = sum(1 for x in rec["per_seed"].values() if x["p_two_sided"] < 0.05)
        total = len(rec["per_seed"])
        sig = "***" if p is not None and p < 0.001 else ("**" if p is not None and p < 0.01 else ("*" if p is not None and p < 0.05 else ""))
        print(f"  {cmp_name}:")
        print(f"    {n}/{total} seeds significant at alpha=0.05; Fisher-pooled p={p} {sig}")


if __name__ == "__main__":
    main()
