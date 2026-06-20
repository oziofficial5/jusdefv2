"""
EUR-Lex within-corpus density-stratified regime test, INLINE predictions.

Uses the same forward-pass and label-extraction pattern as
scripts/eval_all_jusdef.py: forward through JusDef.forward() with proper
edge_attr_dict, pool_document over section embeddings, score against the
graph's own h["label"] embeddings, then sigmoid + threshold.

Reports multi-label macro-F1 stratified by per-document non-AFF density,
plus the v3 - R-GCN delta per bin.

Usage:
    python scripts/analyse_eurlex_regime_inline.py

Outputs:
    outputs/logs/eurlex_regime_inline.json
"""
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef import JusDef
from src.model.baselines import RGCN, tune_threshold
from src.train.trainer import forward_one_graph
import torch.nn.functional as F


SEEDS = [42, 43, 44]


# Patch RGCN.forward to be compatible with newer torch_geometric versions
# (older code did `if k in conv.convs` with k a tuple, which now triggers
#  AttributeError because newer ModuleDict tries to call key.replace on it).
def _patched_rgcn_forward(self, x_dict, edge_index_dict):
    h = {}
    for nt in x_dict:
        if nt in self.input_proj:
            h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
        else:
            h[nt] = x_dict[nt]
    for conv in self.convs:
        valid_edges = {k: v for k, v in edge_index_dict.items() if v.size(1) > 0}
        if valid_edges:
            try:
                new_h = conv(h, valid_edges)
            except (KeyError, AttributeError):
                continue
            for k in new_h:
                h[k] = self.dropout(F.relu(new_h[k]))
    h = {k: self.out_proj(v) for k, v in h.items()}
    return h

RGCN.forward = _patched_rgcn_forward


def detect_rgcn_hidden_dim(state):
    """Read hidden_dim from a checkpoint state dict (handles h=512 and h=768)."""
    for key in ("input_proj.doc.weight", "input_proj.sec.weight"):
        if key in state:
            return state[key].shape[0]
    return 512  # fallback

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


def collect_y(graphs):
    """Return (N_docs, N_labels) binary target matrix from g.y."""
    targets = []
    for g in graphs:
        y = g.y if hasattr(g, "y") else None
        if y is None:
            return None
        targets.append(y.cpu().view(-1).numpy().astype(np.int8))
    return np.stack(targets, axis=0)


@torch.no_grad()
def collect_logits_jusdef(model, graphs, device):
    """Return (N_docs, N_labels) raw logits."""
    model.eval()
    out = []
    for g in graphs:
        scores, _, _ = forward_one_graph(model, g, device)
        out.append(scores.cpu().view(-1).numpy())
    return np.stack(out, axis=0)


@torch.no_grad()
def collect_logits_rgcn(model, graphs, device):
    """Return (N_docs, N_labels) raw logits for the RGCN baseline."""
    model.eval()
    out = []
    for g in graphs:
        g = g.to(device)
        x_dict = {nt: g[nt].x for nt in g.node_types}
        ei_dict = {et: g[et].edge_index for et in g.edge_types
                   if g[et].edge_index.size(1) > 0}
        h = model(x_dict, ei_dict)
        doc_emb = model.pool_document(h["sec"]) if hasattr(model, "pool_document") \
                  else h["sec"].mean(dim=0, keepdim=True)
        if "label" in h:
            scores = model.score(doc_emb, h["label"])
        else:
            # Fallback: use input label embeddings if RGCN didn't process them
            label_x = g["label"].x if "label" in g.node_types else None
            scores = (doc_emb @ label_x.T) if label_x is not None else None
        out.append(scores.cpu().view(-1).numpy())
    return np.stack(out, axis=0)


def stratified_macro_f1(preds, labels, mask, label_subset=None):
    if mask.sum() < 5:
        return None, int(mask.sum())
    labels_sub = labels[mask]
    preds_sub = preds[mask]
    n_labels = labels.shape[1]
    label_iter = label_subset if label_subset is not None else range(n_labels)
    f1s = []
    for j in label_iter:
        if labels_sub[:, j].sum() == 0 and preds_sub[:, j].sum() == 0:
            continue
        f1s.append(f1_score(labels_sub[:, j], preds_sub[:, j], zero_division=0))
    return float(np.mean(f1s)) if f1s else None, int(mask.sum())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading EUR-Lex test + validation graphs...")
    test_graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
    val_graphs = torch.load("data/processed/graphs/validation_graphs.pt", map_location="cpu")
    print(f"  test: {len(test_graphs)}  val: {len(val_graphs)}")

    density = compute_per_doc_density(test_graphs)
    labels = collect_y(test_graphs)
    val_labels = collect_y(val_graphs)
    if labels is None or val_labels is None:
        print("ERROR: no g.y on graphs")
        return
    print(f"  labels matrix: {labels.shape}")

    ckpt_dir = Path("outputs/checkpoints")
    results = {"n_docs": len(test_graphs), "per_variant": {}}

    # ====== v3 EUR-Lex ======
    print("\n=== v3_pilot ===")
    results["per_variant"]["v3_pilot"] = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"jusdef_v3_pilot_s{seed}.pt"
        if not ckpt.is_file():
            print(f"  [skip] s{seed}: {ckpt} missing")
            continue
        try:
            model = JusDef(in_dim=768, hidden_dim=512, num_layers=2,
                           dropout=0.3, temperature=5.0,
                           use_dmp=True, use_authority=True, dmp_variant="v3").to(device)
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state, strict=False)
            # Tune threshold on validation
            val_logits = collect_logits_jusdef(model, val_graphs, device)
            val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
            best_t, _ = tune_threshold(val_probs, val_labels)
            print(f"  s{seed}: val-tuned threshold = {best_t:.4f}")
            # Test predictions
            test_logits = collect_logits_jusdef(model, test_graphs, device)
            test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40, 40)))
            preds = (test_probs >= best_t).astype(np.int8)
        except Exception as e:
            import traceback
            print(f"  s{seed}: failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        per_bin = {}
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            f1, n = stratified_macro_f1(preds, labels, mask)
            per_bin[name] = {"n_documents": n, "macro_f1": f1}
        results["per_variant"]["v3_pilot"][seed] = per_bin
        print(f"    all={per_bin['all']['macro_f1']:.4f}  "
              f"10-20%={per_bin['10-20%']['macro_f1']} "
              f"(n={per_bin['10-20%']['n_documents']})  "
              f">=20%={per_bin['>=20%']['macro_f1']} (n={per_bin['>=20%']['n_documents']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ====== R-GCN baseline ======
    print("\n=== rgcn ===")
    results["per_variant"]["rgcn"] = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"best_rgcn_seed{seed}.pt"
        if not ckpt.is_file():
            print(f"  [skip] s{seed}: {ckpt} missing")
            continue
        try:
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            hd = detect_rgcn_hidden_dim(state)
            print(f"  s{seed}: auto-detected hidden_dim={hd}")
            model = RGCN(in_dim=768, hidden_dim=hd, out_dim=hd,
                         num_layers=2, dropout=0.3).to(device)
            model.load_state_dict(state, strict=False)
            val_logits = collect_logits_rgcn(model, val_graphs, device)
            val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
            best_t, _ = tune_threshold(val_probs, val_labels)
            print(f"  s{seed}: val-tuned threshold = {best_t:.4f}")
            test_logits = collect_logits_rgcn(model, test_graphs, device)
            test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40, 40)))
            preds = (test_probs >= best_t).astype(np.int8)
        except Exception as e:
            import traceback
            print(f"  s{seed}: failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        per_bin = {}
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            f1, n = stratified_macro_f1(preds, labels, mask)
            per_bin[name] = {"n_documents": n, "macro_f1": f1}
        results["per_variant"]["rgcn_h512"][seed] = per_bin
        print(f"    all={per_bin['all']['macro_f1']:.4f}  "
              f"10-20%={per_bin['10-20%']['macro_f1']} "
              f"(n={per_bin['10-20%']['n_documents']})  "
              f">=20%={per_bin['>=20%']['macro_f1']} (n={per_bin['>=20%']['n_documents']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ====== Deltas v3 - R-GCN ======
    if results["per_variant"]["v3_pilot"] and results["per_variant"]["rgcn"]:
        print("\n" + "=" * 70)
        print(" EUR-LEX WITHIN-CORPUS DENSITY-STRATIFIED DELTAS (v3 - R-GCN)")
        print("=" * 70)
        summary = {}
        for name, _, _ in BINS:
            deltas = []
            n_docs = None
            for seed in SEEDS:
                v3 = results["per_variant"]["v3_pilot"].get(seed, {}).get(name)
                rg = results["per_variant"]["rgcn"].get(seed, {}).get(name)
                if v3 and rg and v3["macro_f1"] is not None and rg["macro_f1"] is not None:
                    deltas.append(v3["macro_f1"] - rg["macro_f1"])
                    n_docs = v3["n_documents"]
            if deltas:
                m = float(np.mean(deltas))
                s = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                sign = "+" if m >= 0 else "-"
                marker = "  <-- regime" if name == "10-20%" else ""
                print(f"  {name:<10} N={n_docs:<6} delta = {sign}{abs(m):.4f} +- {s:.4f}{marker}")
                summary[name] = {"delta_mean": m, "delta_std": s,
                                  "n_documents": n_docs, "n_seeds": len(deltas)}
        results["v3_minus_rgcn_summary"] = summary

        print("\n" + "=" * 70)
        print(" VERDICT")
        print("=" * 70)
        regime = summary.get("10-20%")
        if regime is None:
            print("  10-20%% bin: inconclusive")
        elif regime["delta_mean"] > 0.005:
            print(f"  +{regime['delta_mean']:.4f} on EUR-Lex 10-20%% bin "
                  f"(N={regime['n_documents']}, {regime['n_seeds']} seeds)")
            print(f"  -> WITHIN-CORPUS REPLICATION SUPPORTS the operating-regime hypothesis")
        elif regime["delta_mean"] < -0.005:
            print(f"  {regime['delta_mean']:+.4f} on EUR-Lex 10-20%% bin "
                  f"(N={regime['n_documents']}, {regime['n_seeds']} seeds)")
            print(f"  -> within-corpus replication does NOT support the hypothesis")
        else:
            print(f"  {regime['delta_mean']:+.4f} on EUR-Lex 10-20%% bin (near zero)")
            print(f"  -> inconclusive")

    out_path = Path("outputs/logs/eurlex_regime_inline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
