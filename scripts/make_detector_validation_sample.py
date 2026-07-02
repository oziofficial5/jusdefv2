"""
Detector cross-genre validation on LEDGAR (reviewer's load-bearing request).

The operator detector was validated only on EUR-Lex; the LEDGAR density bins depend
on it. This script (1) samples LEDGAR sentences stratified by the detector's
predicted operator so non-AFF classes are well represented, and writes a TSV for
manual annotation; (2) once the human column is filled, computes accuracy and
Cohen's kappa between detector and human.

Step 1 (create sample):
    python scripts/make_detector_validation_sample.py --n 160 \
        --out outputs/logs/ledgar_detector_sample.tsv
    # -> open the TSV, fill the 'human_op' column with AFF/NEG/EXC/OVR

Step 2 (score after annotation):
    python scripts/make_detector_validation_sample.py --score \
        outputs/logs/ledgar_detector_sample.tsv
"""
import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

INT2OP = {0: "AFF", 1: "NEG", 2: "EXC", 3: "OVR"}
OP2INT = {v: k for k, v in INT2OP.items()}


def make_sample(args):
    data = pickle.load(open(Path(args.data_dir) / "train_processed.pkl", "rb"))
    rng = random.Random(args.seed)
    by_op = defaultdict(list)
    for p in data:
        for sent, op in zip(p["sentences"], p["operators"]):
            s = sent.strip().replace("\t", " ").replace("\n", " ")
            if 20 <= len(s) <= 300:
                by_op[int(op)].append(s)
    per = max(args.n // 4, 1)
    rows = []
    for op in range(4):
        pool = by_op.get(op, [])
        rng.shuffle(pool)
        for s in pool[:per]:
            rows.append((INT2OP[op], s))
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("idx\tdetector_op\thuman_op\tsentence\n")
        for i, (op, s) in enumerate(rows):
            f.write(f"{i}\t{op}\t\t{s}\n")
    print(f"Wrote {len(rows)} sentences to {out}")
    print("Fill the 'human_op' column (AFF/NEG/EXC/OVR), then run with --score.")


def score(path):
    det, hum = [], []
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        d, h = parts[1].strip().upper(), parts[2].strip().upper()
        if h in OP2INT:
            det.append(d); hum.append(h)
    if not hum:
        print("No annotated rows (human_op empty)."); return
    n = len(hum)
    acc = sum(1 for a, b in zip(det, hum) if a == b) / n
    # Cohen's kappa
    labels = ["AFF", "NEG", "EXC", "OVR"]
    po = acc
    pe = sum((det.count(l) / n) * (hum.count(l) / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"Annotated: {n} | detector-vs-human accuracy: {acc:.3f} | "
          f"Cohen's kappa: {kappa:.3f}")
    print("Report this kappa in the paper as LEDGAR detector validation "
          "(compare to the EUR-Lex kappa=0.77).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", nargs="?", const=None)
    ap.add_argument("--data_dir", default="data/processed_ledgar")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/logs/ledgar_detector_sample.tsv")
    args = ap.parse_args()
    import sys
    if "--score" in sys.argv:
        # path is the positional value after --score
        idx = sys.argv.index("--score")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else args.out
        score(path)
    else:
        make_sample(args)


if __name__ == "__main__":
    main()
