"""
TF-IDF + LinearSVC baseline on LEDGAR (LexGLUE), reporting macro AND micro F1.

Uses the same ``lex_glue/ledgar`` train/test splits as train_legalbert_ledgar.py,
so the number is directly comparable to the published LexGLUE baselines
(Chalkidis et al. 2022: LegalBERT 0.819 macro / 0.883 micro; TF-IDF+SVM 0.814
macro). This confirms/quantifies the paper's classical baseline honestly.

CPU-only, runs in a few minutes.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=200000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tag", default="tfidf_svm")
    args = ap.parse_args()

    print("Loading LEDGAR via lex_glue...", flush=True)
    tr = load_dataset("lex_glue", "ledgar", split="train")
    te = load_dataset("lex_glue", "ledgar", split="test")
    print(f"  train {len(tr)} | test {len(te)}", flush=True)

    vec = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, args.ngram_max),
        min_df=args.min_df, max_features=args.max_features,
        lowercase=True, strip_accents="unicode",
    )
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    print(f"  features: {Xtr.shape[1]}", flush=True)

    clf = LinearSVC(C=args.C)
    clf.fit(Xtr, tr["label"])
    pred = clf.predict(Xte)

    macro = float(f1_score(te["label"], pred, average="macro", zero_division=0))
    micro = float(f1_score(te["label"], pred, average="micro", zero_division=0))

    res = {
        "model": "tfidf_linearsvc",
        "ngram_max": args.ngram_max, "min_df": args.min_df,
        "max_features": args.max_features, "C": args.C,
        "n_features": int(Xtr.shape[1]),
        "test_macro_f1": round(macro, 4),
        "test_micro_f1": round(micro, 4),
    }
    out = Path("outputs/logs") / f"ledgar_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    print("\nRESULTS: TF-IDF + LinearSVC on LEDGAR")
    print(f"  test_macro_f1: {macro:.4f}")
    print(f"  test_micro_f1: {micro:.4f}")
    print("  (LexGLUE published: TF-IDF+SVM 0.814 macro; "
          "LegalBERT 0.819 macro / 0.883 micro)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
