"""
EUR-Lex within-corpus density-stratified regime test, INLINE predictions.

This script does what analyse_eurlex_density_stratified.py would have done
if per-document predictions were already cached: it loads the v3 and R-GCN
EUR-Lex checkpoints, runs forward passes on the test graphs, and reports
multi-label macro-F1 stratified by per-document non-AFF density.

Critical context: EUR-Lex per-document density (computed earlier) shows:
    >=10%: 1068 documents (21.36% of test set)
    >=20%:  326 documents
    in 10-20%: ~742 documents

If the operating-regime hypothesis is corpus-independent, v3 should
outperform R-GCN on the 742 EUR-Lex documents in the 10-20% bin. This is
the within-corpus replication test: same pipeline, same models, the only
varying condition is the per-document density slice.

Usage:
    python scripts/analyse_eurlex_regime_inline.py

Reads:
    data/processed/graphs/test_graphs.pt
    data/processed/label_adj.pt                       (for v3 label scoring)
    outputs/checkpoints/jusdef_v3_pilot_s{42,43,44}.pt
    outputs/checkpoints/best_rgcn_seed{42,43,44}.pt

Outputs:
    outputs/logs/eurlex_regime_inline.json
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

from src.model.jusdef import JusDef
from src.model.baselines import RGCN


SEEDS = [42, 43, 44]
LABEL_THRESHOLD = 0.5

BINS = [
    ("all",     0.00, 1.01),
    ("0%",      0.00, 0.001),
    ("0-1%",    0.001, 0.01),
    ("1-5%",    0.01, 0.05),
    ("5-10%",   0.05, 0.10),
    ("10-20%",  0.10, 0.20),
    (">=20%",   0.20, 1.01),
]


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


def compute_per_doc_density(graphs):
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


def collect_labels(graphs):
    """Return (N_docs, N_labels) binary label matrix."""
    label_lists = []
    for g in graphs:
        # Try common locations for labels
        for attr in ("label", "y", "labels"):
            if hasattr(g, attr):
                lab = getattr(g, attr)
                if isinstance(lab, torch.Tensor):
                    label_lists.append(lab.flatten().cpu().numpy())
                    break
                elif isinstance(lab, (list, np.ndarray)):
                    label_lists.append(np.array(lab).flatten())
                    break
        else:
            label_lists.append(np.array([], dtype=int))
    # Build a 2D matrix if labels are indices; else assume already binary
    if not label_lists:
        return None
    n_labels = 0
    for ll in label_lists:
        if ll.size:
            n_labels = max(n_labels, int(ll.max()) + 1)
    if n_labels == 0:
        return None
    n_docs = len(label_lists)
    mat = np.zeros((n_docs, n_labels), dtype=np.int8)
    for i, ll in enumerate(label_lists):
        if ll.size:
            mat[i, ll] = 1
    return mat


@torch.no_grad()
def predict_jusdef(model, graphs, label_embs, device, threshold=LABEL_THRESHOLD):
    """Return (N_docs, N_labels) binary predictions for v3 JusDef."""
    model.eval()
    preds_list = []
    for g in graphs:
        x_dict = {nt: x.to(device) for nt, x in g.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in g.edge_index_dict.items()}
        # Build edge_attr_dict the JusDef forward expects
        edge_attr_dict = {}
        for k in edge_index_dict:
            if hasattr(g, "edge_attr_dict") and k in g.edge_attr_dict:
                edge_attr_dict[k] = {
                    f: t.to(device) if isinstance(t, torch.Tensor) else t
                    for f, t in g.edge_attr_dict[k].items()
                }
            else:
                # Pull attrs that exist as direct attributes on the edge store
                attrs = {}
                try:
                    store = g[k]
                    for name in ("operator", "priority", "auth_type",
                                 "auth_level", "auth_recency"):
                        if hasattr(store, name):
                            v = getattr(store, name)
                            if isinstance(v, torch.Tensor):
                                attrs[name] = v.to(device)
                except (KeyError, AttributeError):
                    pass
                if attrs:
                    edge_attr_dict[k] = attrs

        h, _ = model(x_dict, edge_index_dict, edge_attr_dict)
        doc_emb = model.pool_document(h["sec"])  # (1, hidden_dim)
        logits = model.score(doc_emb, label_embs.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
        preds_list.append((probs >= threshold).astype(np.int8))
    return np.stack(preds_list, axis=0)


@torch.no_grad()
def predict_rgcn(model, graphs, label_embs, device, threshold=LABEL_THRESHOLD):
    """Return (N_docs, N_labels) binary predictions for R-GCN baseline."""
    model.eval()
    preds_list = []
    for g in graphs:
        x_dict = {nt: x.to(device) for nt, x in g.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in g.edge_index_dict.items()}
        h = model(x_dict, edge_index_dict)
        # Pool document via mean of section embeddings (RGCN baseline approach)
        sec_emb = h["sec"].mean(dim=0, keepdim=True)
        logits = model.score(sec_emb, label_embs.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
        preds_list.append((probs >= threshold).astype(np.int8))
    return np.stack(preds_list, axis=0)


def stratified_macro_f1(preds, labels, mask):
    """Per-label F1 averaged macro on documents in mask."""
    if mask.sum() < 5:
        return None, int(mask.sum())
    labels_sub = labels[mask]
    preds_sub = preds[mask]
    n_labels = labels.shape[1]
    f1s = []
    for j in range(n_labels):
        # skip labels with zero positives in both pred and gold for this subset
        if labels_sub[:, j].sum() == 0 and preds_sub[:, j].sum() == 0:
            continue
        f1s.append(f1_score(labels_sub[:, j], preds_sub[:, j], zero_division=0))
    return float(np.mean(f1s)) if f1s else None, int(mask.sum())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading EUR-Lex test graphs...")
    graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
    n_docs = len(graphs)
    print(f"  {n_docs} test documents")

    print("Loading label embeddings...")
    # The thesis uses label embeddings stored separately
    label_emb_path = Path("data/processed/label_embs.pt")
    if not label_emb_path.is_file():
        label_emb_path = Path("data/processed/label_embeddings.pt")
    if not label_emb_path.is_file():
        # Try to find any plausible label-embeddings file
        for cand in Path("data/processed").glob("label*.pt"):
            label_emb_path = cand
            break
    if not label_emb_path.is_file():
        print(f"ERROR: no label embeddings file found in data/processed/")
        return
    label_embs = torch.load(label_emb_path, map_location="cpu")
    if isinstance(label_embs, dict) and "embeddings" in label_embs:
        label_embs = label_embs["embeddings"]
    print(f"  label embeddings shape: {tuple(label_embs.shape)}")

    density = compute_per_doc_density(graphs)
    labels = collect_labels(graphs)
    if labels is None:
        print("ERROR: could not extract labels from graphs")
        return
    print(f"  labels matrix shape: {labels.shape}")

    ckpt_dir = Path("outputs/checkpoints")
    results = {"n_docs": n_docs, "per_variant": {}}

    # === v3 EUR-Lex ===
    print("\n=== v3_pilot ===")
    results["per_variant"]["v3_pilot"] = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"jusdef_v3_pilot_s{seed}.pt"
        if not ckpt.is_file():
            print(f"  [skip] s{seed}: {ckpt} missing")
            continue
        model = JusDef(
            in_dim=768, hidden_dim=512, num_layers=2,
            dropout=0.3, temperature=5.0,
            use_dmp=True, use_authority=True, dmp_variant="v3",
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        try:
            preds = predict_jusdef(model, graphs, label_embs, device)
        except Exception as e:
            print(f"  s{seed}: forward failed: {e}")
            del model
            continue
        # Stratified F1
        per_bin = {}
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            f1, n = stratified_macro_f1(preds, labels, mask)
            per_bin[name] = {"n_documents": n, "macro_f1": f1}
        results["per_variant"]["v3_pilot"][seed] = per_bin
        print(f"  s{seed}: all={per_bin['all']['macro_f1']:.4f}  "
              f"10-20%={per_bin['10-20%']['macro_f1']} (n={per_bin['10-20%']['n_documents']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # === R-GCN baseline ===
    print("\n=== rgcn_h512 ===")
    results["per_variant"]["rgcn_h512"] = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"best_rgcn_seed{seed}.pt"
        if not ckpt.is_file():
            print(f"  [skip] s{seed}: {ckpt} missing")
            continue
        model = RGCN(in_dim=768, hidden_dim=512, out_dim=512,
                     num_layers=2, dropout=0.3).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        try:
            model.load_state_dict(state, strict=False)
            preds = predict_rgcn(model, graphs, label_embs, device)
        except Exception as e:
            print(f"  s{seed}: load or forward failed: {e}")
            del model
            continue
        per_bin = {}
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            f1, n = stratified_macro_f1(preds, labels, mask)
            per_bin[name] = {"n_documents": n, "macro_f1": f1}
        results["per_variant"]["rgcn_h512"][seed] = per_bin
        print(f"  s{seed}: all={per_bin['all']['macro_f1']:.4f}  "
              f"10-20%={per_bin['10-20%']['macro_f1']} (n={per_bin['10-20%']['n_documents']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # === Compute deltas (v3 - rgcn) ===
    if results["per_variant"]["v3_pilot"] and results["per_variant"]["rgcn_h512"]:
        print("\n" + "=" * 70)
        print(" EUR-LEX WITHIN-CORPUS DENSITY-STRATIFIED DELTAS (v3 - R-GCN)")
        print("=" * 70)
        summary = {}
        for name, _, _ in BINS:
            deltas = []
            for seed in SEEDS:
                v3 = results["per_variant"]["v3_pilot"].get(seed, {}).get(name)
                rg = results["per_variant"]["rgcn_h512"].get(seed, {}).get(name)
                if v3 and rg and v3["macro_f1"] is not None and rg["macro_f1"] is not None:
                    deltas.append(v3["macro_f1"] - rg["macro_f1"])
            if deltas:
                m = float(np.mean(deltas))
                s = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                n = v3["n_documents"]
                sign = "+" if m >= 0 else "-"
                marker = "  <-- regime if positive" if name == "10-20%" else ""
                print(f"  {name:<10} N={n:<6} delta = {sign}{abs(m):.4f} +- {s:.4f}{marker}")
                summary[name] = {"delta_mean": m, "delta_std": s,
                                  "n_documents": n, "n_seeds": len(deltas)}
        results["v3_minus_rgcn_summary"] = summary

        print("\n" + "=" * 70)
        print(" VERDICT")
        print("=" * 70)
        regime = summary.get("10-20%")
        if regime is None:
            print("  10-20%% bin had no usable seeds; inconclusive")
        elif regime["delta_mean"] > 0.005:
            print(f"  Positive delta of +{regime['delta_mean']:.4f} on EUR-Lex 10-20%% bin")
            print(f"  ({regime['n_documents']} documents, {regime['n_seeds']} seeds)")
            print(f"  -> within-corpus replication SUPPORTS the operating-regime hypothesis")
        elif regime["delta_mean"] < -0.005:
            print(f"  Negative delta of {regime['delta_mean']:.4f} on EUR-Lex 10-20%% bin")
            print(f"  ({regime['n_documents']} documents, {regime['n_seeds']} seeds)")
            print(f"  -> within-corpus replication does NOT support regime hypothesis")
            print(f"     (regime may be LEDGAR-specific, or per-document density on EUR-Lex")
            print(f"      is not the right granularity)")
        else:
            print(f"  Near-zero delta of {regime['delta_mean']:+.4f} on EUR-Lex 10-20%% bin")
            print(f"  ({regime['n_documents']} documents)")
            print(f"  -> inconclusive on within-corpus replication")

    out_path = Path("outputs/logs/eurlex_regime_inline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
