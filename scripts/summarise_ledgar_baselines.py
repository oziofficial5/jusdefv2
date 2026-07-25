"""
Summarise the re-tuned LEDGAR baselines for the paper's structure-vs-scale
section: properly-tuned paragraph LegalBERT-FT (5 seeds) vs TF-IDF+SVM vs the
published LexGLUE numbers. Prints the honest ordering so the "bag-of-words vs
neural" claim can be stated correctly. No GPU.
"""
import json
import statistics as st
from pathlib import Path

LOG = Path("outputs/logs")


def main():
    lb = []
    for f in sorted(LOG.glob("legalbert_ledgar_ft_tuned_s*.json")):
        d = json.load(open(f))
        lb.append((d["test_macro_f1"], d["test_micro_f1"]))

    tf = None
    tp = LOG / "ledgar_tfidf_svm.json"
    if tp.exists():
        tf = json.load(open(tp))

    print("=" * 66)
    print(" LEDGAR baselines (re-tuned) -- structure vs. scale")
    print("=" * 66)

    if lb:
        ma = [x[0] for x in lb]
        mi = [x[1] for x in lb]
        print(f"LegalBERT-FT (tuned, {len(lb)} seeds): "
              f"macro {st.mean(ma):.4f} +/- {st.pstdev(ma):.4f} | "
              f"micro {st.mean(mi):.4f} +/- {st.pstdev(mi):.4f} | "
              f"best macro {max(ma):.4f}")
    else:
        print("LegalBERT-FT (tuned): no result files yet.")

    if tf:
        print(f"TF-IDF + LinearSVC:            "
              f"macro {tf['test_macro_f1']:.4f} | micro {tf['test_micro_f1']:.4f} "
              f"({tf['n_features']} feats)")
    else:
        print("TF-IDF + LinearSVC: no result file yet.")

    print("-" * 66)
    print("Published LexGLUE (Chalkidis 2022): LegalBERT 0.819 macro / 0.883 micro"
          "; TF-IDF+SVM 0.814 macro")

    if lb and tf:
        d = st.mean([x[0] for x in lb]) - tf["test_macro_f1"]
        print("-" * 66)
        print(f"Delta macro (tuned LegalBERT-FT  -  TF-IDF) = {d:+.4f}")
        if d >= -0.002:
            print(" -> re-tuned transformer >= TF-IDF (within noise). DROP the "
                  "'bag-of-words beats every neural model' framing; reframe Section 9 "
                  "to 'a classical TF-IDF model is competitive with a fine-tuned "
                  "transformer on this lexical task'.")
        else:
            print(" -> TF-IDF still ahead of the properly-tuned transformer. The "
                  "'competitive/ahead' claim holds; report the honest (small) gap and "
                  "keep the ordering.")


if __name__ == "__main__":
    main()
