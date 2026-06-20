"""
Compute per-document non-AFF operator density distributions for each
corpus and save as JSON for the cross-corpus density histogram figure.

Outputs:
    outputs/logs/cross_corpus_density.json

Reads (per corpus, if available):
    LEDGAR:  data/processed_ledgar/test_processed.pkl
    EUR-Lex: data/processed/graphs/test_graphs.pt
    ECtHR:   data/processed_ecthr/test_processed.pkl  (optional)
"""
import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def ledgar_density():
    p = Path("data/processed_ledgar/test_processed.pkl")
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    densities = []
    for d in data:
        ops = d.get("operators")
        if ops is None or len(ops) == 0:
            continue
        n_non_aff = sum(1 for o in ops if o != 0)
        densities.append(n_non_aff / len(ops))
    return densities


def _get_edge_attr(g, edge_type, attr_name):
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


def eurlex_density():
    p = Path("data/processed/graphs/test_graphs.pt")
    if not p.is_file():
        return None
    graphs = torch.load(p, map_location="cpu")
    r2 = ("sec", "mentions", "conc")
    densities = []
    for g in graphs:
        ops = _get_edge_attr(g, r2, "operator")
        if ops is None:
            continue
        ops_t = ops if isinstance(ops, torch.Tensor) else torch.tensor(ops)
        if ops_t.numel() == 0:
            continue
        n_non_aff = int((ops_t != 0).sum().item())
        densities.append(n_non_aff / ops_t.numel())
    return densities


def ecthr_density():
    p = Path("data/processed_ecthr/test_processed.pkl")
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    densities = []
    for d in data:
        ops = d.get("operators")
        if ops is None or len(ops) == 0:
            continue
        n_non_aff = sum(1 for o in ops if o != 0)
        densities.append(n_non_aff / len(ops))
    return densities


def main():
    result = {}
    for name, fn in [("LEDGAR", ledgar_density),
                     ("EUR-Lex", eurlex_density),
                     ("ECtHR", ecthr_density)]:
        ds = fn()
        if ds is None:
            print(f"  {name}: skip (input not found)")
            continue
        ds_arr = np.array(ds)
        result[name] = {
            "n_documents": len(ds),
            "mean_density": float(ds_arr.mean()),
            "median_density": float(np.median(ds_arr)),
            "fraction_in_10_20": float(((ds_arr >= 0.10) & (ds_arr < 0.20)).mean()),
            "fraction_below_1pct": float((ds_arr < 0.01).mean()),
            "densities": ds_arr.tolist(),
        }
        print(f"  {name}: n={len(ds)}, mean={ds_arr.mean():.4f}, "
              f"in [10,20)%: {result[name]['fraction_in_10_20']*100:.2f}%")

    out_path = Path("outputs/logs/cross_corpus_density.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
