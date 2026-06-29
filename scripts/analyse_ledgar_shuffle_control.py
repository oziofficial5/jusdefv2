"""
Operator-permutation control analysis for the LEDGAR operating-regime claim.

For each seed it compares, on the TRUE 10-20% non-AFF density bin:
    - mean baseline          (operator-agnostic)
    - v3 with REAL operators  (the headline model)
    - v3 with SHUFFLED operators (the control)

Binning uses the TRUE operators from data/processed_ledgar (real density). The
shuffled-v3 model is fed operators from data/processed_ledgar_shuffleop, which
is index-aligned to the original (same paragraph order, same per-paragraph
length) — see make_shuffled_operators.py.

Decisive reading:
    delta_real  = F1(v3 real)     - F1(mean)   -> expected POSITIVE (the claim)
    delta_shuf  = F1(v3 shuffled) - F1(mean)   -> expected ~ZERO   (no real ops)
If delta_real > 0 across seeds while delta_shuf collapses to ~0, the regime gain
is from operator SEMANTICS, not capacity / inductive bias.

Checkpoint naming expected (from train_ledgar.py --tag):
    mean      : outputs/checkpoints/ledgar_baseline_mean_s<seed>.pt
    v3 real   : outputs/checkpoints/ledgar_v3_pilot_s<seed>.pt
    v3 shuffle: outputs/checkpoints/ledgar_v3_shuffleop_s<seed>.pt

Usage:
    python scripts/analyse_ledgar_shuffle_control.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(variant, device):
    m = JusDefLEDGAR(
        in_dim=768, hidden_dim=512, num_classes=100, num_layers=1,
        dmp_variant=variant,
    ).to(device)
    return m


@torch.no_grad()
def predict(model, embeddings_data, ops_source, device):
    """Predict per paragraph; embeddings from embeddings_data[i], operators from
    ops_source[i] (a list of int operator ids aligned to that paragraph)."""
    model.eval()
    preds = []
    for emb_p, ops in zip(embeddings_data, ops_source):
        sent_embs = emb_p["embeddings"].to(device)
        sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
        operators = torch.tensor(ops, dtype=torch.long, device=device)
        logits = model(sent_embs, sent_to_para, operators, num_paragraphs=1)
        preds.append(int(logits.argmax(dim=-1).item()))
    return np.array(preds)


def macro_on_mask(labels, preds, mask):
    labels_sub = labels[mask]
    preds_sub = preds[mask]
    present = sorted(set(labels_sub.tolist()))
    return f1_score(labels_sub, preds_sub, labels=present, average="macro", zero_division=0)


def load_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--orig_dir", default="data/processed_ledgar")
    ap.add_argument("--shuf_dir", default="data/processed_ledgar_shuffleop")
    ap.add_argument("--mean_tag", default="baseline_mean")
    ap.add_argument("--real_tag", default="v3_pilot")
    ap.add_argument("--shuf_tag", default="v3_shuffleop")
    ap.add_argument("--bin_lo", type=float, default=0.10)
    ap.add_argument("--bin_hi", type=float, default=0.20)
    ap.add_argument("--ckpt_dir", default="outputs/checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    orig_test = load_pkl(Path(args.orig_dir) / "test_processed.pkl")
    shuf_test = load_pkl(Path(args.shuf_dir) / "test_processed.pkl")
    assert len(orig_test) == len(shuf_test), "orig/shuffled test length mismatch"
    for a, b in zip(orig_test, shuf_test):
        assert len(a["operators"]) == len(b["operators"]), "per-paragraph length mismatch"

    # TRUE density + labels from the original corpus
    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in orig_test
    ])
    labels = np.array([d["label"] for d in orig_test])
    mask = (density >= args.bin_lo) & (density < args.bin_hi)
    n_bin = int(mask.sum())
    print(f"True {args.bin_lo*100:.0f}-{args.bin_hi*100:.0f}% density bin: {n_bin} paragraphs\n")

    real_ops = [list(d["operators"]) for d in orig_test]   # true operators
    shuf_ops = [list(d["operators"]) for d in shuf_test]    # permuted operators

    ckpt = Path(args.ckpt_dir)
    rows = []
    for s in args.seeds:
        p_mean = ckpt / f"ledgar_{args.mean_tag}_s{s}.pt"
        p_real = ckpt / f"ledgar_{args.real_tag}_s{s}.pt"
        p_shuf = ckpt / f"ledgar_{args.shuf_tag}_s{s}.pt"
        if not (p_mean.exists() and p_real.exists() and p_shuf.exists()):
            miss = [str(p) for p in (p_mean, p_real, p_shuf) if not p.exists()]
            print(f"[seed {s}] SKIP — missing checkpoints: {miss}")
            continue

        m_mean = load_state(build_model("mean", device), p_mean, device)
        m_real = load_state(build_model("v3", device), p_real, device)
        m_shuf = load_state(build_model("v3", device), p_shuf, device)

        preds_mean = predict(m_mean, orig_test, real_ops, device)   # mean ignores ops
        preds_real = predict(m_real, orig_test, real_ops, device)   # v3 + TRUE ops
        preds_shuf = predict(m_shuf, orig_test, shuf_ops, device)   # v3 + SHUFFLED ops

        f_mean = macro_on_mask(labels, preds_mean, mask)
        f_real = macro_on_mask(labels, preds_real, mask)
        f_shuf = macro_on_mask(labels, preds_shuf, mask)
        rows.append({
            "seed": s,
            "mean": f_mean, "v3_real": f_real, "v3_shuf": f_shuf,
            "delta_real": f_real - f_mean, "delta_shuf": f_shuf - f_mean,
        })
        print(f"[seed {s}] mean={f_mean:.4f}  v3_real={f_real:.4f} (Δ{f_real-f_mean:+.4f})  "
              f"v3_shuf={f_shuf:.4f} (Δ{f_shuf-f_mean:+.4f})")

    if not rows:
        print("\nNo complete seed sets found. Train the checkpoints first.")
        return

    dr = np.array([r["delta_real"] for r in rows])
    ds = np.array([r["delta_shuf"] for r in rows])
    n = len(rows)
    print("\n" + "=" * 64)
    print(f" SHUFFLE-CONTROL SUMMARY  (n={n} seeds, {args.bin_lo*100:.0f}-{args.bin_hi*100:.0f}% bin)")
    print("=" * 64)
    print(f"  delta_real (v3 TRUE ops vs mean):     {dr.mean():+.4f} ± {dr.std():.4f}  "
          f"| seeds >0: {int((dr>0).sum())}/{n}")
    print(f"  delta_shuf (v3 SHUFFLED ops vs mean): {ds.mean():+.4f} ± {ds.std():.4f}  "
          f"| seeds >0: {int((ds>0).sum())}/{n}")
    print(f"  semantic gain (real - shuffled):      {(dr-ds).mean():+.4f} ± {(dr-ds).std():.4f}")

    print("\n VERDICT")
    if dr.mean() > 0 and ds.mean() <= 0.5 * dr.mean():
        print("  PASS — the regime gain depends on REAL operator semantics.")
        print("  Shuffling operators collapses the gain; capacity/inductive-bias")
        print("  alone does NOT reproduce it. This defuses the 'capacity not")
        print("  defeasibility' challenge for the LEDGAR regime.")
    elif dr.mean() <= 0:
        print("  INCONCLUSIVE — real-operator gain not reproduced this run; check")
        print("  seeds / that v3_pilot checkpoints are the intended ones.")
    else:
        print("  WARNING — shuffled operators retain much of the gain. The 10-20%")
        print("  effect may be partly capacity / inductive bias, NOT pure semantics.")
        print("  Report this honestly; it reshapes the central claim.")

    out = Path("outputs/logs/ledgar_shuffle_control.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_seeds": n, "bin": [args.bin_lo, args.bin_hi], "n_bin_paragraphs": n_bin,
            "per_seed": rows,
            "delta_real_mean": float(dr.mean()), "delta_real_std": float(dr.std()),
            "delta_shuf_mean": float(ds.mean()), "delta_shuf_std": float(ds.std()),
        }, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
