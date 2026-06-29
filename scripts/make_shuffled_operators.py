"""
Build an operator-permutation control corpus for the LEDGAR regime experiment.

Takes data/processed_ledgar/{train,validation,test}_processed.pkl and writes a
copy to data/processed_ledgar_shuffleop/ in which the per-sentence operator
labels are GLOBALLY PERMUTED within each split.

Why a global permutation (not per-paragraph):
  - It preserves the split's marginal operator distribution EXACTLY (it is a
    permutation of the full operator vector), so the model sees the same
    AFF/NEG/EXC/OVR frequencies it always did.
  - It destroys the correspondence between a sentence's CONTENT and its
    operator, so any operator information the model exploits is now noise.
  - Paragraph order and each paragraph's sentence count are preserved, so index
    i in the shuffled dir aligns with index i in the original dir. This lets the
    control analysis bin paragraphs by their TRUE density while feeding the
    model the shuffled operators.

If v3 trained on this control no longer beats the mean baseline on the true
10-20% density bin, the regime gain is genuinely from operator SEMANTICS, not
from added capacity or v3's inductive bias. That is the decisive control for the
single most dangerous viva question.

Usage:
    python scripts/make_shuffled_operators.py
    python scripts/make_shuffled_operators.py --shuffle_seed 999
"""
import argparse
import pickle
from collections import Counter
from pathlib import Path

# fixed per-split offsets so splits get different (but reproducible) permutations
SPLIT_OFFSET = {"train": 0, "validation": 1, "test": 2}


def shuffle_split(data, seed):
    import random
    rng = random.Random(seed)
    flat = []
    lengths = []
    for p in data:
        ops = [int(o) for o in p["operators"]]
        lengths.append(len(ops))
        flat.extend(ops)
    before = Counter(flat)
    rng.shuffle(flat)
    out = []
    idx = 0
    for p, L in zip(data, lengths):
        newp = dict(p)                      # shallow copy keeps embeddings + label
        newp["operators"] = flat[idx:idx + L]
        idx += L
        out.append(newp)
    assert idx == len(flat), "redistribution length mismatch"
    after = Counter(o for p in out for o in p["operators"])
    assert before == after, "marginal operator distribution changed — bug!"
    return out, before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed_ledgar")
    ap.add_argument("--dst", default="data/processed_ledgar_shuffleop")
    ap.add_argument("--shuffle_seed", type=int, default=12345)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        with open(src / f"{split}_processed.pkl", "rb") as f:
            data = pickle.load(f)
        sh, marg = shuffle_split(data, args.shuffle_seed + SPLIT_OFFSET[split])
        with open(dst / f"{split}_processed.pkl", "wb") as f:
            pickle.dump(sh, f)
        total = sum(marg.values())
        dist = {k: f"{v}({v / total * 100:.1f}%)" for k, v in sorted(marg.items())}
        print(f"[{split:<10}] {len(sh):>6} paras | operators preserved: {dist}")

    print(f"\nShuffled control corpus written to {dst}/")
    print("Operator legend: 0=AFF 1=NEG 2=EXC 3=OVR")


if __name__ == "__main__":
    main()
