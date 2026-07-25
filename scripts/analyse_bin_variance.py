"""
Reviewer re-analysis of the 10-20% regime bin (headline v3 vs mean): per-class
support, micro-F1 (accuracy, far less variance-pathological than macro over 100
classes), PER-SEED signs, and a paragraph-level bootstrap CI on the macro-F1 delta.

Run after dump_predictions_10_20.py (SEEDS 42-51). Prints everything needed to
harden / correct the paper's "+0.075, positive in all N seeds" claim. No GPU.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

PRED = Path("outputs/logs/predictions_10_20_bin.json")


def main():
    d = json.load(open(PRED))
    y = np.asarray(d["true_labels"])
    N = len(y)
    seeds = sorted(d["mean"].keys(), key=int)
    print(f"N paragraphs = {N} | seeds ({len(seeds)}) = {seeds}")

    sup = Counter(y.tolist())
    classes = sorted(sup)
    n1 = sum(1 for c in classes if sup[c] == 1)
    nle2 = sum(1 for c in classes if sup[c] <= 2)
    print(f"\nclasses present: {len(classes)}/100 | singletons {n1} | "
          f"support<=2: {nle2} ({100*nle2/len(classes):.0f}% of present classes)")

    def macro(p): return f1_score(y, p, average="macro", zero_division=0)
    def micro(p): return accuracy_score(y, p)  # micro-F1 == accuracy (single-label)

    dmac, dmic = [], []
    print("\nseed | mean_macro v3_macro  dMACRO | mean_micro v3_micro dMICRO")
    for s in seeds:
        pm, pv = np.asarray(d["mean"][s]), np.asarray(d["v3_pilot"][s])
        a, b, c, e = macro(pm), macro(pv), micro(pm), micro(pv)
        dmac.append(b - a); dmic.append(e - c)
        print(f"  {s} |   {a:.3f}    {b:.3f}   {b-a:+.3f} |   {c:.3f}    {e:.3f}  {e-c:+.3f}")

    dmac, dmic = np.array(dmac), np.array(dmic)
    npos_mac = int((dmac > 0).sum()); npos_mic = int((dmic > 0).sum())
    print(f"\nMACRO delta: {dmac.mean():+.4f} +/- {dmac.std():.4f} | "
          f"seeds>0: {npos_mac}/{len(seeds)}")
    print(f"MICRO delta: {dmic.mean():+.4f} +/- {dmic.std():.4f} | "
          f"seeds>0: {npos_mic}/{len(seeds)}")

    # paragraph-level bootstrap CI on the pooled-seed macro delta
    rng = np.random.default_rng(0); B = 10000
    pm = {s: np.asarray(d["mean"][s]) for s in seeds}
    pv = {s: np.asarray(d["v3_pilot"][s]) for s in seeds}
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, N, N); yb = y[idx]
        boot[i] = np.mean([
            f1_score(yb, pv[s][idx], average="macro", zero_division=0) -
            f1_score(yb, pm[s][idx], average="macro", zero_division=0) for s in seeds])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nBootstrap 95% CI on macro delta ({B} paragraph resamples, pooled seeds):")
    print(f"  point {dmac.mean():+.4f} | 95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"width {hi-lo:.3f} | P(delta>0)={np.mean(boot>0):.3f}")


if __name__ == "__main__":
    main()
