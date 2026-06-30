"""
Generic multi-seed comparison of any LEDGAR variant vs the mean baseline, on
BOTH the aggregate test set and the true 10-20% non-AFF density regime.

Reusable across the 6-day architecture plan (F1 residual-anchored gate, F3
operator-attention, F4 multi-task, ...). It answers the two questions that
matter for "beating baselines":

  1. aggregate Δ (variant - mean)  >= 0 ?   (does it stop losing overall?)
  2. regime  Δ (variant - mean)    >  0 ?   (does it keep the 10-20% win?)

Checkpoints expected: outputs/checkpoints/ledgar_<tag>_s<seed>.pt
                      outputs/checkpoints/ledgar_baseline_mean_s<seed>.pt

Usage:
    # F1 (baseline-anchored soft gate):
    python scripts/analyse_ledgar_variant.py --tag v4_soft_anchor --variant v4_soft \
        --seeds 42 43 44 45 46 47 48 49 50 51
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


def build_model(variant, device, init_bias=-5.0):
    return JusDefLEDGAR(
        in_dim=768, hidden_dim=512, num_classes=100, num_layers=1,
        dmp_variant=variant, v4_soft_init_bias=init_bias,
    ).to(device)


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    preds = []
    for p in data:
        sent_embs = p["embeddings"].to(device)
        sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
        operators = torch.tensor(p["operators"], dtype=torch.long, device=device)
        logits = model(sent_embs, sent_to_para, operators, num_paragraphs=1)
        preds.append(int(logits.argmax(dim=-1).item()))
    return np.array(preds)


def macro_all(labels, preds):
    return f1_score(labels, preds, average="macro", zero_division=0)


def macro_subset(labels, preds, mask):
    ls, ps = labels[mask], preds[mask]
    present = sorted(set(ls.tolist()))
    return f1_score(ls, ps, labels=present, average="macro", zero_division=0)


def summ(name, arr):
    a = np.array(arr); n = len(a)
    sd = a.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    print(f"  {name:<28} {a.mean():+.4f} ± {sd:.4f} (SE {se:.4f}) | seeds >0: {int((a>0).sum())}/{n}")
    return {"mean": float(a.mean()), "se": float(se), "n_pos": int((a > 0).sum()), "n": n}


def load_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="variant tag, e.g. v4_soft_anchor")
    ap.add_argument("--variant", required=True,
                    choices=["v3", "mean", "v4_hard", "v4_soft"],
                    help="dmp_variant used to build the model for loading")
    ap.add_argument("--init_bias", type=float, default=-5.0,
                    help="v4_soft init bias used at training (only affects module build, "
                         "overwritten by load_state_dict)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--mean_tag", default="baseline_mean")
    ap.add_argument("--data_dir", default="data/processed_ledgar")
    ap.add_argument("--bin_lo", type=float, default=0.10)
    ap.add_argument("--bin_hi", type=float, default=0.20)
    ap.add_argument("--ckpt_dir", default="outputs/checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  | variant '{args.tag}' ({args.variant}) vs mean baseline")

    with open(Path(args.data_dir) / "test_processed.pkl", "rb") as f:
        test = pickle.load(f)
    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1) for d in test
    ])
    labels = np.array([d["label"] for d in test])
    mask = (density >= args.bin_lo) & (density < args.bin_hi)
    print(f"Test paragraphs: {len(test)} | {args.bin_lo*100:.0f}-{args.bin_hi*100:.0f}% bin: {int(mask.sum())}\n")

    ckpt = Path(args.ckpt_dir)
    rows, dagg, dbin = [], [], []
    for s in args.seeds:
        p_mean = ckpt / f"ledgar_{args.mean_tag}_s{s}.pt"
        p_var = ckpt / f"ledgar_{args.tag}_s{s}.pt"
        if not (p_mean.exists() and p_var.exists()):
            miss = [str(p) for p in (p_mean, p_var) if not p.exists()]
            print(f"[seed {s}] SKIP — missing {miss}")
            continue
        m_mean = load_state(build_model("mean", device), p_mean, device)
        m_var = load_state(build_model(args.variant, device, args.init_bias), p_var, device)
        pm, pv = predict(m_mean, test, device), predict(m_var, test, device)
        agg = macro_all(labels, pv) - macro_all(labels, pm)
        b = macro_subset(labels, pv, mask) - macro_subset(labels, pm, mask)
        rows.append({"seed": s, "delta_agg": agg, "delta_bin": b})
        dagg.append(agg); dbin.append(b)
        print(f"[seed {s}] Δagg={agg:+.4f}  Δ(10-20%)={b:+.4f}")

    if not rows:
        print("\nNo complete seed pairs found — train the variant first."); return

    print("\n" + "=" * 64)
    print(f" VARIANT vs MEAN  (n={len(rows)} seeds)  —  tag '{args.tag}'")
    print("=" * 64)
    sa = summ("aggregate  Δ (var-mean)", dagg)
    sb = summ("regime 10-20% Δ (var-mean)", dbin)

    print("\n VERDICT")
    agg_ok = sa["mean"] - sa["se"] >= 0 or sa["mean"] >= 0    # no aggregate loss
    bin_ok = sb["mean"] > 0 and sb["n_pos"] >= 0.7 * sb["n"]  # keeps regime win
    if agg_ok and bin_ok:
        print("  SUCCESS — no aggregate loss AND regime win preserved.")
        print("  This is the F1 target: beats/matches mean overall, keeps the 10-20% gain.")
    elif bin_ok and not agg_ok:
        print("  PARTIAL — regime win kept but still loses aggregate. Gate not suppressing v3 enough.")
    elif agg_ok and not bin_ok:
        print("  PARTIAL — aggregate fixed but regime win lost. Gate over-suppressing v3.")
    else:
        print("  FAIL — neither aggregate nor regime improved vs mean.")

    out = Path(f"outputs/logs/ledgar_variant_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tag": args.tag, "variant": args.variant, "per_seed": rows,
                   "delta_aggregate": sa, "delta_regime": sb}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
