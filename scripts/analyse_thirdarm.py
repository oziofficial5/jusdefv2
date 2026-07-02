"""
Three-arm permutation decomposition on the LEDGAR 10-20% regime.

Compares, on the TRUE 10-20% density bin, v3 trained+evaluated with:
  true operators      (data/processed_ledgar,           tag v3_pilot)
  shuffled operators  (data/processed_ledgar_shuffleop,  tag v3_shuffleop)
  zeroed operators    (data/processed_ledgar_zeroop,     tag v3_zeroop)
each vs the mean baseline (tag baseline_mean).

Reports:
  semantics    = delta_true - delta_shuf   (operator identity)
  rate effect  = delta_shuf - delta_zero   (operator presence/rate)
  bias         = delta_zero                (pure inductive bias)
This answers the reviewer challenge that the shuffled arm's gain is operator
presence, not content-free inductive bias.

Usage: python scripts/analyse_thirdarm.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.model.jusdef_ledgar import JusDefLEDGAR


def load(path):
    return pickle.load(open(path, "rb"))


def model(variant, device):
    return JusDefLEDGAR(in_dim=768, hidden_dim=512, num_classes=100,
                        num_layers=1, dmp_variant=variant).to(device)


@torch.no_grad()
def predict(m, embdata, ops, device):
    m.eval(); preds = []
    for emb, o in zip(embdata, ops):
        se = emb["embeddings"].to(device)
        s2p = torch.zeros(se.size(0), dtype=torch.long, device=device)
        op = torch.tensor(o, dtype=torch.long, device=device)
        preds.append(int(m(se, s2p, op, num_paragraphs=1).argmax(-1).item()))
    return np.array(preds)


def macro(labels, preds, mask):
    ls, ps = labels[mask], preds[mask]
    present = sorted(set(ls.tolist()))
    return f1_score(ls, ps, labels=present, average="macro", zero_division=0)


def state(m, p, device):
    m.load_state_dict(torch.load(p, map_location=device)); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--ckpt", default="outputs/checkpoints")
    ap.add_argument("--bin_lo", type=float, default=0.10)
    ap.add_argument("--bin_hi", type=float, default=0.20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    orig = load("data/processed_ledgar/test_processed.pkl")
    shuf = load("data/processed_ledgar_shuffleop/test_processed.pkl")
    zero = load("data/processed_ledgar_zeroop/test_processed.pkl")
    density = np.array([sum(1 for o in d["operators"] if o != 0) /
                        max(len(d["operators"]), 1) for d in orig])
    labels = np.array([d["label"] for d in orig])
    mask = (density >= args.bin_lo) & (density < args.bin_hi)
    ops_true = [list(d["operators"]) for d in orig]
    ops_shuf = [list(d["operators"]) for d in shuf]
    ops_zero = [list(d["operators"]) for d in zero]
    print(f"10-20% bin: {int(mask.sum())} paragraphs\n")

    ck = Path(args.ckpt)
    D = {"true": [], "shuf": [], "zero": []}
    for s in args.seeds:
        paths = {k: ck / f"ledgar_{t}_s{s}.pt" for k, t in
                 [("mean", "baseline_mean"), ("true", "v3_pilot"),
                  ("shuf", "v3_shuffleop"), ("zero", "v3_zeroop")]}
        if not all(p.exists() for p in paths.values()):
            miss = [k for k, p in paths.items() if not p.exists()]
            print(f"[seed {s}] SKIP missing {miss}"); continue
        pm = predict(state(model("mean", dev), paths["mean"], dev), orig, ops_true, dev)
        fm = macro(labels, pm, mask)
        for k, opsrc in [("true", ops_true), ("shuf", ops_shuf), ("zero", ops_zero)]:
            pv = predict(state(model("v3", dev), paths[k], dev), orig, opsrc, dev)
            D[k].append(macro(labels, pv, mask) - fm)
        print(f"[seed {s}] d_true={D['true'][-1]:+.4f} d_shuf={D['shuf'][-1]:+.4f} "
              f"d_zero={D['zero'][-1]:+.4f}")

    if not D["true"]:
        print("\nNo complete seed sets (train v3_zeroop first)."); return
    t, sh, z = np.array(D["true"]), np.array(D["shuf"]), np.array(D["zero"])
    print("\n" + "=" * 60)
    print(f" THREE-ARM DECOMPOSITION (n={len(t)} seeds, 10-20% bin)")
    print("=" * 60)
    print(f"  true operators   : {t.mean():+.4f} ± {t.std(ddof=1):.4f}")
    print(f"  shuffled (rate)  : {sh.mean():+.4f} ± {sh.std(ddof=1):.4f}")
    print(f"  zeroed (bias)    : {z.mean():+.4f} ± {z.std(ddof=1):.4f}")
    print("  ---")
    print(f"  semantics (true-shuf) : {(t-sh).mean():+.4f}")
    print(f"  rate      (shuf-zero) : {(sh-z).mean():+.4f}")
    print(f"  bias      (zero)      : {z.mean():+.4f}")
    json.dump({"true": t.tolist(), "shuf": sh.tolist(), "zero": z.tolist(),
               "semantics": float((t-sh).mean()), "rate": float((sh-z).mean()),
               "bias": float(z.mean())},
              open("outputs/logs/ledgar_thirdarm.json", "w"), indent=2)
    print("\nSaved outputs/logs/ledgar_thirdarm.json")


if __name__ == "__main__":
    main()
