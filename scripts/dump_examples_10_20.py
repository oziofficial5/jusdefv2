"""
Extract real LEDGAR clause examples from the 10-20% regime where the operator-aware
model helps or hurts, for the qualitative box in the paper.

Prints paragraphs in the 10-20% density bin where:
  - v3 predicts the gold label and mean does not  (operator prior HELPS)
  - mean predicts the gold label and v3 does not   (operator prior HURTS)
showing the sentences, their operators, gold label id, and both predictions.

Usage:
    python scripts/dump_examples_10_20.py --seed 42 --k 5
"""
import argparse, os, pickle, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.model.jusdef_ledgar import JusDefLEDGAR

INT2OP = {0: "AFF", 1: "NEG", 2: "EXC", 3: "OVR"}


def model(v, d):
    return JusDefLEDGAR(in_dim=768, hidden_dim=512, num_classes=100,
                        num_layers=1, dmp_variant=v).to(d)


@torch.no_grad()
def predict_one(m, p, d):
    se = p["embeddings"].to(d)
    s2p = torch.zeros(se.size(0), dtype=torch.long, device=d)
    op = torch.tensor(p["operators"], dtype=torch.long, device=d)
    return int(m(se, s2p, op, num_paragraphs=1).argmax(-1).item())


def show(p, gold, pm, pv):
    for s, o in zip(p["sentences"], p["operators"]):
        print(f"      [{INT2OP[int(o)]}] {s.strip()[:140]}")
    print(f"      gold={gold}  mean_pred={pm}  v3_pred={pv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ckpt", default="outputs/checkpoints")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path(args.ckpt)

    test = pickle.load(open("data/processed_ledgar/test_processed.pkl", "rb"))
    density = np.array([sum(1 for o in d["operators"] if o != 0) /
                        max(len(d["operators"]), 1) for d in test])
    idx = [i for i in range(len(test)) if 0.10 <= density[i] < 0.20]

    mm = model("mean", dev); mm.load_state_dict(
        torch.load(ck / f"ledgar_baseline_mean_s{args.seed}.pt", map_location=dev))
    vv = model("v3", dev); vv.load_state_dict(
        torch.load(ck / f"ledgar_v3_pilot_s{args.seed}.pt", map_location=dev))

    helps, hurts = [], []
    for i in idx:
        p = test[i]; gold = p["label"]
        pm = predict_one(mm, p, dev); pv = predict_one(vv, p, dev)
        if pv == gold and pm != gold:
            helps.append((i, gold, pm, pv))
        elif pm == gold and pv != gold:
            hurts.append((i, gold, pm, pv))

    print(f"10-20% bin: {len(idx)} paragraphs | v3 helps: {len(helps)} | v3 hurts: {len(hurts)}\n")
    print("=" * 70); print(" OPERATOR PRIOR HELPS (v3 right, mean wrong)"); print("=" * 70)
    for i, g, pm, pv in helps[:args.k]:
        print(f"  --- test #{i} ---"); show(test[i], g, pm, pv); print()
    print("=" * 70); print(" OPERATOR PRIOR HURTS (mean right, v3 wrong)"); print("=" * 70)
    for i, g, pm, pv in hurts[:args.k]:
        print(f"  --- test #{i} ---"); show(test[i], g, pm, pv); print()


if __name__ == "__main__":
    main()
