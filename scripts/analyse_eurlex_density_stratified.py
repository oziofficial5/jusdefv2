"""
Within-corpus density-stratified evaluation on EUR-Lex.

The LEDGAR operating-regime finding (10-20% non-AFF density bin where v3
outperforms a mean-aggregation baseline) needs replication evidence. A full
second-corpus pipeline (HUDOC, CUAD) is not tractable in the available
budget; this script provides the next-best evidence by stratifying the
EXISTING EUR-Lex test set by per-document non-AFF density and comparing
v3 to R-GCN on each density slice.

Hypothesis: if the operating-regime hypothesis is corpus-independent and
truly density-driven, EUR-Lex documents with density in the 10-20%
operating window should also show a v3 vs.\ R-GCN gap that mirrors the
LEDGAR finding. Documents below the regime (most of EUR-Lex) should show
v3 underperforming, consistent with the LEDGAR aggregate.

This is the within-corpus version of the operating-regime test using
existing v3 and R-GCN EUR-Lex checkpoints. No new training is required.

Usage:
    python scripts/analyse_eurlex_density_stratified.py

Reads:
    data/processed/graphs/test_graphs.pt
    outputs/checkpoints/jusdef_v3_pilot_s{42,43,44}.pt   (v3 EUR-Lex)
    outputs/checkpoints/best_rgcn_seed{42,43,44}.pt      (R-GCN EUR-Lex)

Outputs:
    outputs/logs/eurlex_density_stratified.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# We avoid importing the JusDef model here and instead reuse a thin
# wrapper around the eval pipeline. Predictions are loaded from per-checkpoint
# eval JSONs if they already exist; otherwise we compute them in-process.

SEEDS = [42, 43, 44]


def _get_edge_attr(g, edge_type, attr_name):
    """Defensive HeteroData edge-attr accessor."""
    if hasattr(g, "edge_attr_dict"):
        d = g.edge_attr_dict
        if edge_type in d and isinstance(d[edge_type], dict) and attr_name in d[edge_type]:
            return d[edge_type][attr_name]
    try:
        store = g[edge_type]
    except (KeyError, AttributeError):
        return None
    if hasattr(store, attr_name):
        return getattr(store, attr_name)
    return None


def compute_per_document_density(graphs):
    """Per-document non-AFF density on r2 mention edges."""
    r2 = ("sec", "mentions", "conc")
    densities = []
    for g in graphs:
        ops = _get_edge_attr(g, r2, "operator")
        if ops is None:
            densities.append(0.0)
            continue
        ops_t = ops if isinstance(ops, torch.Tensor) else torch.tensor(ops)
        n = ops_t.numel()
        if n == 0:
            densities.append(0.0)
            continue
        n_non_aff = int((ops_t != 0).sum().item())
        densities.append(n_non_aff / n)
    return np.array(densities)


def collect_predictions_from_eval_logs(seeds, tag):
    """Look for per-document predictions in existing eval JSONs.

    The full JusDef eval is heavy; for this analysis we accept any of:
      outputs/logs/eval_<tag>_s<seed>.json
      outputs/logs/jusdef_<tag>_s<seed>.json
    that contain a 'per_document_predictions' field.

    Returns (preds_per_seed, labels) where preds_per_seed[seed] is an array
    of label predictions per document and labels is the gold array. If no
    eval JSONs are found, returns None and the caller can decide how to
    proceed (re-run eval, or skip).
    """
    eval_dir = Path("outputs/logs")
    preds_per_seed = {}
    labels = None
    for seed in seeds:
        candidates = [
            eval_dir / f"eval_{tag}_s{seed}.json",
            eval_dir / f"jusdef_{tag}_s{seed}.json",
        ]
        found = None
        for c in candidates:
            if c.is_file():
                try:
                    with open(c) as f:
                        d = json.load(f)
                except json.JSONDecodeError:
                    continue
                if "per_document_predictions" in d and "per_document_labels" in d:
                    found = d
                    break
        if found is None:
            continue
        preds_per_seed[seed] = np.array(found["per_document_predictions"])
        if labels is None:
            labels = np.array(found["per_document_labels"])
    if not preds_per_seed:
        return None, None
    return preds_per_seed, labels


def stratified_f1(preds, labels, mask):
    """Multi-label macro-F1 if labels are 2D, else multi-class macro-F1."""
    if labels.ndim == 2:
        # multi-label: per-label F1, average macro
        labels_sub = labels[mask]
        preds_sub = preds[mask]
        # F1 per label
        n_labels = labels.shape[1]
        f1s = []
        for j in range(n_labels):
            if labels_sub[:, j].sum() == 0 and preds_sub[:, j].sum() == 0:
                continue
            f1s.append(f1_score(labels_sub[:, j], preds_sub[:, j], zero_division=0))
        return float(np.mean(f1s)) if f1s else None, int(mask.sum())
    else:
        labels_sub = labels[mask]
        preds_sub = preds[mask]
        labels_present = sorted(set(labels_sub.tolist()))
        if not labels_present:
            return None, int(mask.sum())
        return float(f1_score(labels_sub, preds_sub, labels=labels_present,
                              average="macro", zero_division=0)), int(mask.sum())


def main():
    print("Loading EUR-Lex test graphs...")
    graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
    n_docs = len(graphs)
    print(f"  {n_docs} test documents")

    density = compute_per_document_density(graphs)

    # Report distribution
    print("\nEUR-Lex per-document non-AFF density distribution:")
    for thresh in [0.005, 0.01, 0.05, 0.10, 0.15, 0.20]:
        n = int((density >= thresh).sum())
        print(f"  density >= {thresh*100:5.1f}%: {n:5d} documents ({n/n_docs*100:5.2f}%)")

    # Density bins
    BINS = [
        ("all",     0.00, 1.01),
        ("0%",      0.00, 0.001),
        ("0-1%",    0.001, 0.01),
        ("1-5%",    0.01, 0.05),
        ("5-10%",   0.05, 0.10),
        ("10-20%",  0.10, 0.20),
        (">=20%",   0.20, 1.01),
    ]

    results = {
        "n_test_documents": n_docs,
        "density_distribution": {f"{t}": int((density >= t).sum())
                                 for t in [0.005, 0.01, 0.05, 0.10, 0.15, 0.20]},
        "per_variant": {},
    }

    for tag, file_template in [
        ("v3_pilot",      "jusdef_v3_pilot"),
        ("rgcn_h512",     "best_rgcn_seed"),
    ]:
        print(f"\n=== Variant: {tag} ===")
        preds_per_seed, labels = collect_predictions_from_eval_logs(SEEDS, file_template)
        if preds_per_seed is None:
            print(f"  SKIP {tag}: no per-document predictions found in existing "
                  f"eval JSONs. Run scripts/eval_all_jusdef.py first to produce "
                  f"the per-document predictions.")
            continue
        results["per_variant"][tag] = {"per_seed_per_bin": {}}
        for seed, preds in preds_per_seed.items():
            results["per_variant"][tag]["per_seed_per_bin"][seed] = {}
            for name, lo, hi in BINS:
                mask = (density >= lo) & (density < hi)
                f1, n = stratified_f1(preds, labels, mask)
                results["per_variant"][tag]["per_seed_per_bin"][seed][name] = {
                    "n_documents": n,
                    "macro_f1": f1,
                }
            print(f"  seed {seed}: bin counts and F1s recorded")

    # Compute deltas (v3 - rgcn) per bin per seed, then 5-seed mean/std
    if "v3_pilot" in results["per_variant"] and "rgcn_h512" in results["per_variant"]:
        print("\n" + "=" * 70)
        print(" EUR-LEX WITHIN-CORPUS DENSITY-STRATIFIED DELTAS (v3 - R-GCN)")
        print("=" * 70)
        per_bin_summary = {}
        for name, _, _ in BINS:
            deltas = []
            for seed in SEEDS:
                v3 = results["per_variant"]["v3_pilot"]["per_seed_per_bin"].get(seed, {}).get(name)
                rg = results["per_variant"]["rgcn_h512"]["per_seed_per_bin"].get(seed, {}).get(name)
                if v3 and rg and v3["macro_f1"] is not None and rg["macro_f1"] is not None:
                    deltas.append(v3["macro_f1"] - rg["macro_f1"])
            if deltas:
                n_docs_bin = v3["n_documents"]
                m = float(np.mean(deltas))
                s = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                marker = "  <-- regime if positive" if name == "10-20%" else ""
                sign = "+" if m >= 0 else "-"
                print(f"  {name:<10} N={n_docs_bin:<5} delta_mean = {sign}{abs(m):.4f} +- {s:.4f}{marker}")
                per_bin_summary[name] = {"delta_mean": m, "delta_std": s,
                                          "n_documents": n_docs_bin,
                                          "n_seeds": len(deltas)}
        results["v3_minus_rgcn_summary"] = per_bin_summary

        if "10-20%" in per_bin_summary and per_bin_summary["10-20%"]["delta_mean"] > 0:
            print("\nVERDICT: EUR-Lex within-corpus replication SUPPORTS the regime hypothesis.")
        elif "10-20%" in per_bin_summary:
            print("\nVERDICT: EUR-Lex 10-20% bin does NOT show v3 outperforming R-GCN.")
            print("         The regime may be corpus-specific, or the bin may be too small.")
        else:
            print("\nVERDICT: 10-20% bin has insufficient documents on EUR-Lex; "
                  "regime test is inconclusive.")

    out_path = Path("outputs/logs/eurlex_density_stratified.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
