"""
Direct empirical measurement of the three named failure modes on existing
v2 EUR-Lex checkpoints.

Currently Chapter 6 INFERS these failure modes from controlled interventions.
A hostile examiner can say "you don't measure them, you only infer them". This
script measures all three directly, transforming Chapter 6 from inferential to
empirical.

Measured quantities:

  Failure mode 1 (STE bias on hard gates -- Liu Gapped STE 2022):
      For v2's DMP defeat mask, the STE backward gradient is
          d_sigmoid(-T * gap) / d_gap = T * sig * (1 - sig)
      which peaks at gap=0 (gradient T/4) and decays to ~0 for |gap| >> 1/T.
      We measure the distribution of priority_gap on the EUR-Lex test set
      and report the fraction of edges in the "saturated" regime (|gap| > 0.5,
      where sigmoid gradient is <10% of peak). High saturation = STE bias.

  Failure mode 2 (sign cancellation -- Zhu 2024):
      Applies to v3, not v2 (v2 has no signed coefficients). We measure the
      v3 op_coef trajectory across layers when num_layers > 1 if a multi-
      layer v3 checkpoint is available; otherwise we report this measurement
      as not applicable to single-layer v3 and infer from the per-layer op_coef
      values at training end.

  Failure mode 3 (W-undertraining -- diagnosed in this thesis):
      Read W_op[0..3] weight matrices directly from the v2 state_dict and
      report ||W_op||_F per operator. Read the training graph data and count
      mention edges per operator. The Frobenius norm ratio + edge-count
      ratio together characterise the asymmetric training.

Usage:
    python scripts/analyse_v2_failure_modes.py

Reads:
    data/processed/graphs/{train,test}_graphs.pt
    outputs/checkpoints/jusdef_v2_keyword_s{42,43,44}.pt
    outputs/checkpoints/jusdef_v2_neural_s{42,43,44}.pt
    outputs/checkpoints/jusdef_v3_eurlex_s{42,43,44}.pt   (for failure mode 2)

Outputs:
    outputs/logs/v2_failure_mode_analysis.json
    outputs/figures/fig7_ste_saturation.pdf
    outputs/figures/fig8_w_op_imbalance.pdf
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef import JusDef
from src.model.dmp_layer import compute_defeat_mask


OPERATOR_NAMES = ["AFF", "NEG", "EXC", "OVR"]

# Checkpoint tag mapping for v2/v3 EUR-Lex (file naming on Ampere)
V2_TAGS = ["full", "full_neural"]    # v2 keyword and v2 neural
V3_TAGS = ["v3_pilot"]               # v3 EUR-Lex
SEEDS = [42, 43, 44]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_v2(device, dmp_variant="hard"):
    return JusDef(
        in_dim=768, hidden_dim=512, num_layers=2,
        dropout=0.3, temperature=5.0, use_dmp=True,
        use_authority=True, dmp_variant=dmp_variant,
    ).to(device)


def load_state(model, ckpt):
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    return model


# ---------------------------------------------------------------------------
# Failure mode 3: W_op norm imbalance + operator-frequency asymmetry
# ---------------------------------------------------------------------------

def measure_w_op_norms(model):
    """Read ||W_op||_F per operator from v2's DMP layers."""
    norms_per_layer = []
    for layer in model.dmp_layers:
        if not hasattr(layer, "W_op"):
            continue
        layer_norms = {}
        for op_idx in range(4):
            W = layer.W_op[op_idx].weight.detach()
            layer_norms[OPERATOR_NAMES[op_idx]] = float(torch.norm(W, p="fro").item())
        norms_per_layer.append(layer_norms)
    return norms_per_layer


def _get_edge_attr(g, edge_type, attr_name):
    """Defensive accessor: try multiple HeteroData edge-attr access patterns."""
    # Pattern A: explicit edge_attr_dict attached to the graph
    if hasattr(g, "edge_attr_dict"):
        d = g.edge_attr_dict
        if edge_type in d and isinstance(d[edge_type], dict) and attr_name in d[edge_type]:
            return d[edge_type][attr_name]
    # Pattern B: direct attribute on the edge store
    try:
        store = g[edge_type]
    except (KeyError, AttributeError):
        return None
    if hasattr(store, attr_name):
        return getattr(store, attr_name)
    return None


def measure_operator_frequency(graphs, max_graphs=2000):
    """Count r2 mention edges per operator across (a subset of) training graphs."""
    r2_key = ("sec", "mentions", "conc")
    counts = np.zeros(4, dtype=np.int64)
    n_used = 0
    for g in graphs[:max_graphs]:
        ops = _get_edge_attr(g, r2_key, "operator")
        if ops is None:
            continue
        ops_t = ops if isinstance(ops, torch.Tensor) else torch.tensor(ops)
        for op_idx in range(4):
            counts[op_idx] += int((ops_t == op_idx).sum().item())
        n_used += 1
    return counts, n_used


# ---------------------------------------------------------------------------
# Failure mode 1: STE saturation rate on v2's defeat gates
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_ste_saturation(model, graphs, device, max_graphs=500):
    """Forward-pass v2 on test graphs and log the priority_gap distribution
    that the STE sees. Compute the fraction of edges in the saturated regime
    where the sigmoid gradient is <10% of peak (|gap| > 0.5 with T=5)."""
    r2_key = ("sec", "mentions", "conc")
    all_gaps = []
    all_ops = []

    model.eval()
    for g in graphs[:max_graphs]:
        edge_index_dict = {k: v.to(device) for k, v in g.edge_index_dict.items()}
        if r2_key not in edge_index_dict or edge_index_dict[r2_key].size(1) == 0:
            continue
        r2_ei = edge_index_dict[r2_key]

        ops = _get_edge_attr(g, r2_key, "operator")
        pri = _get_edge_attr(g, r2_key, "priority")
        if ops is None or pri is None:
            continue
        ops = ops.to(device) if isinstance(ops, torch.Tensor) else torch.tensor(ops, device=device)
        pri = pri.to(device) if isinstance(pri, torch.Tensor) else torch.tensor(pri, device=device, dtype=torch.float32)

        # Replicate the same priority_gap computation as DMPLayer
        defeat_score = ops.float() * 1000.0 + pri
        dst = r2_ei[1]
        num_groups = int(dst.max().item()) + 1
        base = torch.full((num_groups,), -1e9, device=device)
        group_max = base.scatter_reduce(
            0, dst, defeat_score, reduce="amax", include_self=True,
        )
        gap = group_max[dst] - defeat_score - 0.001

        all_gaps.append(gap.cpu().numpy())
        all_ops.append(ops.cpu().numpy())

    if not all_gaps:
        return None

    gaps = np.concatenate(all_gaps)
    ops_arr = np.concatenate(all_ops)

    T = 5.0  # STE temperature in DMPLayer default
    # Sigmoid gradient: T * sig(-T*gap) * (1 - sig(-T*gap)); peak at gap=0 is T/4
    peak = T / 4.0
    sig = 1.0 / (1.0 + np.exp(T * gaps))
    grad = T * sig * (1 - sig)
    saturation_rate = float((grad < 0.1 * peak).mean())

    per_op = {}
    for op_idx in range(4):
        m = (ops_arr == op_idx)
        if m.sum() < 10:
            continue
        sat_op = float(((grad[m]) < 0.1 * peak).mean())
        per_op[OPERATOR_NAMES[op_idx]] = {
            "n_edges": int(m.sum()),
            "saturation_rate": sat_op,
            "mean_gap": float(gaps[m].mean()),
            "median_abs_gap": float(np.median(np.abs(gaps[m]))),
        }

    return {
        "n_edges_total": int(len(gaps)),
        "overall_saturation_rate": saturation_rate,
        "mean_gap": float(gaps.mean()),
        "median_abs_gap": float(np.median(np.abs(gaps))),
        "per_operator": per_op,
    }


# ---------------------------------------------------------------------------
# Failure mode 2: op_coef trajectory across v3 layers
# ---------------------------------------------------------------------------

def measure_op_coef_drift(model):
    """For v3 (multi-layer), report ||op_coef||_2 per layer and the
    per-operator coef values. If multiple layers exist, this characterises
    sign-cancellation drift across depth."""
    out = []
    for i, layer in enumerate(model.dmp_layers):
        if not hasattr(layer, "op_coef"):
            continue
        coef = layer.op_coef.detach().cpu().numpy()
        out.append({
            "layer": i,
            "AFF": float(coef[0]),
            "NEG": float(coef[1]),
            "EXC": float(coef[2]),
            "OVR": float(coef[3]),
            "norm_l2": float(np.linalg.norm(coef)),
        })
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- Load training and test graphs -----
    graph_dir = Path("data/processed/graphs")
    if not (graph_dir / "test_graphs.pt").is_file():
        print(f"ERROR: {graph_dir/'test_graphs.pt'} missing")
        return
    test_graphs = torch.load(graph_dir / "test_graphs.pt", map_location="cpu")
    print(f"  Test graphs: {len(test_graphs)}")
    train_graphs = None
    if (graph_dir / "train_graphs.pt").is_file():
        train_graphs = torch.load(graph_dir / "train_graphs.pt", map_location="cpu")
        print(f"  Train graphs: {len(train_graphs)}")

    results = {}
    ckpt_dir = Path("outputs/checkpoints")

    # ----- v2 analysis (failure modes 1 + 3) -----
    for tag in V2_TAGS:
        # Display name maps "full" -> "v2_keyword", "full_neural" -> "v2_neural"
        display = {"full": "v2_keyword", "full_neural": "v2_neural"}.get(tag, tag)
        results[display] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"jusdef_{tag}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  [skip] {ckpt} missing")
                continue
            print(f"\n--- {display} (file tag '{tag}') seed {seed} ---")
            model = build_v2(device, dmp_variant="hard")
            try:
                load_state(model, ckpt)
            except Exception as e:
                print(f"  load failed: {e}")
                continue

            w_norms = measure_w_op_norms(model)
            print(f"  ||W_op||_F per layer: {w_norms}")

            ste = measure_ste_saturation(model, test_graphs, device)
            if ste is not None:
                print(f"  STE saturation: overall {ste['overall_saturation_rate']:.3f} "
                      f"(n={ste['n_edges_total']})")
                for op_name, op_data in ste["per_operator"].items():
                    print(f"    {op_name}: sat={op_data['saturation_rate']:.3f} "
                          f"(n={op_data['n_edges']})")
            else:
                print("  STE: no r2 edges found in test graphs")

            results[display][seed] = {
                "w_op_norms_per_layer": w_norms,
                "ste_saturation": ste,
            }
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ----- Operator frequency on training data -----
    if train_graphs is not None:
        print("\n--- Operator frequency on training graphs (failure mode 3) ---")
        counts, n_used = measure_operator_frequency(train_graphs, max_graphs=2000)
        total = max(int(counts.sum()), 1)
        freqs = (counts.astype(float) / total).tolist()
        print(f"  Across {n_used} training graphs: total edges {total}")
        for i, name in enumerate(OPERATOR_NAMES):
            print(f"    {name}: {int(counts[i]):>8d}  ({freqs[i]*100:.2f}%)")
        results["operator_frequency"] = {
            "n_graphs_used": n_used,
            "counts": counts.tolist(),
            "fractions": freqs,
        }

    # ----- v3 op_coef trajectory (failure mode 2) -----
    print("\n--- v3 op_coef per-layer values (failure mode 2) ---")
    results["v3_op_coef_drift"] = {}
    for tag in V3_TAGS:
        results["v3_op_coef_drift"][tag] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"jusdef_{tag}_s{seed}.pt"
            if not ckpt.is_file():
                continue
            model = build_v2(device, dmp_variant="v3")
            try:
                load_state(model, ckpt)
            except Exception as e:
                print(f"  {tag} s{seed} load failed: {e}")
                continue
            coefs = measure_op_coef_drift(model)
            print(f"  {tag} s{seed}: {coefs}")
            results["v3_op_coef_drift"][tag][seed] = coefs
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ----- Save JSON -----
    out_json = Path("outputs/logs/v2_failure_mode_analysis.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Figures -----
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 10, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure: STE saturation per operator (v2_neural seed 42)
    primary = results.get("v2_neural", {}).get(42, {}) or results.get("v2_keyword", {}).get(42, {})
    if primary and primary.get("ste_saturation"):
        per_op = primary["ste_saturation"]["per_operator"]
        names = [n for n in OPERATOR_NAMES if n in per_op]
        sats = [per_op[n]["saturation_rate"] for n in names]
        ns   = [per_op[n]["n_edges"] for n in names]
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        bars = ax.bar(names, sats, color=["#0072B2", "#D55E00", "#E69F00", "#009E73"],
                      edgecolor="black", lw=0.5)
        for i, n in enumerate(ns):
            ax.text(i, sats[i] + 0.01, f"N={n}", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel("STE saturation rate (gap with $|grad| < 10$\\% peak)")
        ax.set_title("v2 defeat-gate STE saturation rate by operator (seed 42)")
        ax.axhline(0.8, color="grey", lw=0.7, linestyle="--", zorder=1)
        ax.set_ylim(0, max(1.0, max(sats) * 1.15))
        fig.tight_layout()
        out_fig = fig_dir / "fig7_ste_saturation.pdf"
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"Saved {out_fig}")

    # Figure: ||W_op||_F imbalance for v2 (averaged across seeds, layer 0)
    v2_display_keys = [k for k in ("v2_keyword", "v2_neural") if results.get(k)]
    if v2_display_keys:
        tag_to_norms = {}
        for tag in v2_display_keys:
            seed_norms = {n: [] for n in OPERATOR_NAMES}
            for seed in SEEDS:
                wn = results.get(tag, {}).get(seed, {}).get("w_op_norms_per_layer")
                if wn and len(wn) > 0:
                    for n in OPERATOR_NAMES:
                        v = wn[0].get(n)
                        if v is not None:
                            seed_norms[n].append(v)
            if any(seed_norms[n] for n in OPERATOR_NAMES):
                tag_to_norms[tag] = {
                    n: (float(np.mean(seed_norms[n])) if seed_norms[n] else None,
                        float(np.std(seed_norms[n])) if len(seed_norms[n]) > 1 else 0.0)
                    for n in OPERATOR_NAMES
                }
        if tag_to_norms:
            x = np.arange(4)
            w = 0.36
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            colors = {"v2_keyword": "#0072B2", "v2_neural": "#D55E00"}
            for offset, tag in enumerate(sorted(tag_to_norms.keys())):
                means = [tag_to_norms[tag][n][0] for n in OPERATOR_NAMES]
                stds = [tag_to_norms[tag][n][1] for n in OPERATOR_NAMES]
                ax.bar(x + (offset - 0.5) * w, means, w, yerr=stds, capsize=3,
                       color=colors.get(tag, "#999999"), label=tag,
                       edgecolor="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(OPERATOR_NAMES)
            ax.set_ylabel("$\\|W_\\omega\\|_F$ (mean over seeds)")
            ax.set_title("v2 per-operator weight-matrix norm")
            ax.legend(frameon=False)
            fig.tight_layout()
            out_fig = fig_dir / "fig8_w_op_imbalance.pdf"
            fig.savefig(out_fig)
            plt.close(fig)
            print(f"Saved {out_fig}")

    # ----- Print verdict -----
    print("\n" + "=" * 70)
    print(" FAILURE-MODE VERDICT")
    print("=" * 70)
    if primary and primary.get("ste_saturation"):
        sr = primary["ste_saturation"]["overall_saturation_rate"]
        if sr > 0.7:
            print(f"  STE BIAS:    overall saturation {sr:.2%} -> CONFIRMED")
        elif sr > 0.4:
            print(f"  STE BIAS:    overall saturation {sr:.2%} -> PARTIAL")
        else:
            print(f"  STE BIAS:    overall saturation {sr:.2%} -> NOT SUPPORTED")
    if results.get("operator_frequency"):
        f = results["operator_frequency"]["fractions"]
        ratio = f[0] / max(min(f[1], f[2], f[3]), 1e-6)
        print(f"  W ASYMMETRY: AFF/{min(['NEG','EXC','OVR'], key=lambda x: f[{'NEG':1,'EXC':2,'OVR':3}[x]])} "
              f"edge ratio = {ratio:.0f}x -> {'CONFIRMED' if ratio > 10 else 'PARTIAL' if ratio > 3 else 'NOT SUPPORTED'}")
    if results.get("v3_op_coef_drift"):
        any_drift = False
        for tag in results["v3_op_coef_drift"]:
            for seed, coefs in results["v3_op_coef_drift"][tag].items():
                if not coefs:
                    continue
                # Final layer NEG coef close to zero indicates sign collapse
                neg_last = coefs[-1].get("NEG")
                if neg_last is not None and abs(neg_last) < 0.2:
                    any_drift = True
        if any_drift:
            print(f"  SIGN COLLAPSE: NEG coef drifted close to 0 -> CONFIRMED on some seeds")
        else:
            print(f"  SIGN COLLAPSE: NEG coef preserved (regulariser working) -> NOT supported on v3")


if __name__ == "__main__":
    main()
