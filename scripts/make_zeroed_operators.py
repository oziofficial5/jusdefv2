"""
Build an operators-ZEROED control corpus (third arm of the permutation control).

Sets every sentence operator to AFF (0), removing ALL operator information (not
just the sentence-operator correspondence, as the shuffle does). This is the
control the reviewers asked for: it separates "responds to operator PRESENCE/rate"
from "responds to operator IDENTITY/semantics".

Decomposition on the true 10-20% regime:
  true ops      : full gain            (semantics + rate + inductive bias)
  shuffled ops  : rate + inductive bias (identity destroyed, rate preserved)
  zeroed ops    : inductive bias only   (all operator info removed)
So: semantics = true - shuffled ; rate effect = shuffled - zeroed.

Paragraph order and length preserved (index-aligned to the original) so binning
uses TRUE density.

Usage: python scripts/make_zeroed_operators.py
"""
import argparse
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed_ledgar")
    ap.add_argument("--dst", default="data/processed_ledgar_zeroop")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        data = pickle.load(open(src / f"{split}_processed.pkl", "rb"))
        out = []
        for p in data:
            newp = dict(p)
            newp["operators"] = [0] * len(list(p["operators"]))  # all AFF
            out.append(newp)
        pickle.dump(out, open(dst / f"{split}_processed.pkl", "wb"))
        print(f"[{split:<10}] {len(out)} paras -> all operators set to AFF (0)")
    print(f"\nZeroed control corpus written to {dst}/")


if __name__ == "__main__":
    main()
