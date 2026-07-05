"""
Classical / linear baselines for LEDGAR, to complete the structure-vs-scale
comparison spectrum (classical -> frozen-linear -> frozen-GNN -> fine-tuned).

Two baselines, both cheap (CPU, no fine-tuning):
  1. TF-IDF (1-2 grams) + LinearSVC on the raw paragraph text   [classical floor]
  2. frozen LegalBERT [CLS] mean-pool + Logistic Regression      [frozen linear probe]

These bracket the frozen-GNN family (mean/v3/v4 ~0.69-0.70) from below (classical)
and contextualise it against the fine-tuned transformers (~0.80-0.82).

Usage:
    python scripts/train_ledgar_classical.py
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def tfidf_svm():
    from datasets import load_dataset
    tr = load_dataset("lex_glue", "ledgar", split="train")
    te = load_dataset("lex_glue", "ledgar", split="test")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
                          strip_accents="unicode")
    Xtr = vec.fit_transform(tr["text"]); Xte = vec.transform(te["text"])
    clf = LinearSVC(C=1.0)
    clf.fit(Xtr, tr["label"])
    p = clf.predict(Xte)
    return {"macro_f1": float(f1_score(te["label"], p, average="macro")),
            "micro_f1": float(f1_score(te["label"], p, average="micro"))}


def frozen_probe(data_dir):
    tr = pickle.load(open(Path(data_dir) / "train_processed.pkl", "rb"))
    te = pickle.load(open(Path(data_dir) / "test_processed.pkl", "rb"))
    Xtr = np.stack([p["embeddings"].mean(0).numpy() for p in tr]); ytr = [p["label"] for p in tr]
    Xte = np.stack([p["embeddings"].mean(0).numpy() for p in te]); yte = [p["label"] for p in te]
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    return {"macro_f1": float(f1_score(yte, p, average="macro")),
            "micro_f1": float(f1_score(yte, p, average="micro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_ledgar")
    args = ap.parse_args()
    res = {}
    print("=== TF-IDF + LinearSVC (classical floor) ===", flush=True)
    res["tfidf_svm"] = tfidf_svm()
    print(f"  macro-F1 {res['tfidf_svm']['macro_f1']:.4f} | micro-F1 {res['tfidf_svm']['micro_f1']:.4f}")
    print("=== frozen LegalBERT [CLS] mean + LogReg (linear probe) ===", flush=True)
    res["frozen_probe"] = frozen_probe(args.data_dir)
    print(f"  macro-F1 {res['frozen_probe']['macro_f1']:.4f} | micro-F1 {res['frozen_probe']['micro_f1']:.4f}")
    out = Path("outputs/logs/ledgar_classical.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved {out}")
    print("\nSpectrum context (LEDGAR macro-F1): classical (above) -> frozen mean ~0.708 / "
          "v3 ~0.688 / v4 ~0.702 -> fine-tuned LegalBERT ~0.80-0.82.")


if __name__ == "__main__":
    main()
