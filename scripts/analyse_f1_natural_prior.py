"""Reweight the two-stratum blind annotation into a natural-prior estimate.

Stage 3 of run_day2_ampere.sh drew 400 sentences from a 60,000-sentence uniform
random sample of the EUR-Lex test split: 250 the detector called AFF, 150 it
called non-AFF. The strata have very different sampling fractions, so the raw
count of non-AFF sentences in the 400 means nothing on its own -- it has to be
reweighted by the population each stratum was drawn from.

That reweighting is what answers F1. It gives the true per-sentence non-AFF rate
with an interval, and it gives the detector's precision and its false-positive
rate on AFF -- the quantity the prediction-stratified LEDGAR design cannot reach,
and the one that decides whether the 4.40% the detector predicts is a measurement
or its own noise floor.

Usage:
    python scripts/analyse_f1_natural_prior.py \
        --blind outputs/day2/f1_eurlex_validation_blind.tsv \
        --key   outputs/day2/f1_eurlex_validation_KEY.tsv \
        --weights outputs/day2/f1_weights.json
"""
import io
import os
import json
import math
import argparse
import collections

NONAFF = {"NEG", "EXC", "OVR"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", default="outputs/day2/f1_eurlex_validation_blind.tsv")
    ap.add_argument("--key", default="outputs/day2/f1_eurlex_validation_KEY.tsv")
    ap.add_argument("--weights", default="outputs/day2/f1_weights.json")
    ap.add_argument("--out", default="outputs/logs/f1_natural_prior.json")
    a = ap.parse_args()

    for p in (a.blind, a.key, a.weights):
        if not os.path.exists(p):
            print("missing: %s" % p)
            return 1

    def tsv(p):
        rows = [l.rstrip("\n").split("\t") for l in io.open(p, encoding="utf-8")]
        hdr = [h.strip() for h in rows[0]]
        return hdr, [r for r in rows[1:] if len(r) >= len(hdr)]

    bh, br = tsv(a.blind)
    kh, kr = tsv(a.key)
    W = json.load(open(a.weights))

    i_bidx, i_hum = bh.index("idx"), bh.index("human_op")
    i_kidx, i_str, i_det = kh.index("idx"), kh.index("stratum"), kh.index("detector_op")
    key = {r[i_kidx]: (r[i_str], r[i_det].strip().upper()) for r in kr}

    unlabelled = [r[i_bidx] for r in br if not r[i_hum].strip()]
    if unlabelled:
        print("%d of %d rows are not yet annotated. Complete the human_op column first."
              % (len(unlabelled), len(br)))
        print("first few: %s" % ", ".join(unlabelled[:10]))
        return 1

    st = collections.defaultdict(lambda: {"n": 0, "nonaff": 0, "conf": collections.Counter()})
    for r in br:
        idx = r[i_bidx]
        h = r[i_hum].strip().upper()
        if idx not in key:
            print("row %s not in key -- files do not correspond" % idx)
            return 1
        stratum, det = key[idx]
        d = st[stratum]
        d["n"] += 1
        if h in NONAFF:
            d["nonaff"] += 1
        d["conf"][(det, h)] += 1

    NA = W["pred_AFF_population"]
    NB = W["pred_nonAFF_population"]
    N = NA + NB
    A, B = st["A_pred_AFF"], st["B_pred_nonAFF"]
    nA, nB = A["n"], B["n"]
    pA = A["nonaff"] / float(nA)          # detector said AFF, human says non-AFF
    pB = B["nonaff"] / float(nB)          # detector said non-AFF, human agrees

    print("=" * 72)
    print("  STRATUM SUMMARY")
    print("=" * 72)
    print("  A  detector says AFF     : %4d sampled from %6d  -> %d truly non-AFF (%.2f%%)"
          % (nA, NA, A["nonaff"], 100 * pA))
    print("  B  detector says non-AFF : %4d sampled from %6d  -> %d truly non-AFF (%.2f%%)"
          % (nB, NB, B["nonaff"], 100 * pB))

    # stratified estimate with finite-population correction
    est = (NA * pA + NB * pB) / float(N)
    vA = pA * (1 - pA) / nA * (1 - nA / float(NA)) if nA else 0.0
    vB = pB * (1 - pB) / nB * (1 - nB / float(NB)) if nB else 0.0
    var = (NA ** 2 * vA + NB ** 2 * vB) / float(N ** 2)
    se = math.sqrt(var)
    lo, hi = est - 1.96 * se, est + 1.96 * se

    print()
    print("=" * 72)
    print("  TRUE PER-SENTENCE NON-AFF DENSITY, EUR-Lex test split")
    print("=" * 72)
    print("  estimate  %.4f%%   95%% CI [%.4f%%, %.4f%%]   (SE %.4f%%)"
          % (100 * est, 100 * lo, 100 * hi, 100 * se))
    print("  detector predicted 4.3983%% on the same 60,000 sentences.")
    if hi < 0.043983:
        print("  --> the detector OVER-predicts: its 4.40%% is above the true rate.")
        print("      Chapter 3's class-prior argument is confirmed and the")
        print("      per-sentence figure must be revised DOWN to this estimate.")
    elif lo > 0.043983:
        print("  --> the detector UNDER-predicts: the true rate is higher still.")
        print("      This runs against the class-prior argument and for the")
        print("      over-segmentation argument of the segmentation section.")
    else:
        print("  --> the interval contains 4.3983%%: no evidence the detector's")
        print("      predicted rate is biased in either direction.")

    print()
    print("=" * 72)
    print("  DETECTOR BEHAVIOUR AT THE NATURAL PRIOR")
    print("=" * 72)
    tp = NB * pB
    fp = NB * (1 - pB)
    fn = NA * pA
    tn = NA * (1 - pA)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    plo, phi = wilson(B["nonaff"], nB)
    print("  non-AFF precision : %.4f   95%% CI [%.3f, %.3f]" % (prec, plo, phi))
    print("  non-AFF recall    : %.4f" % rec)
    print("  false-positive rate on true AFF : %.4f%%" % (100 * fpr))
    print()
    print("  This is the number the LEDGAR validation could not produce, because")
    print("  stratifying by detector prediction fixes the predicted marginals.")
    print("  It is what decides whether the sub-1%% EUR-Lex floor is a measurement")
    print("  or an artefact of the detector's own error rate.")

    print()
    print("  confusion, detector vs human, within each stratum:")
    for name, d in (("A_pred_AFF", A), ("B_pred_nonAFF", B)):
        print("    %s" % name)
        for (det, hum), v in sorted(d["conf"].items(), key=lambda x: -x[1]):
            flag = "" if det == hum else "   <- disagreement"
            print("      detector %-4s human %-4s  %3d%s" % (det, hum, v, flag))

    os.makedirs("outputs/logs", exist_ok=True)
    json.dump({
        "n_sampled": {"A_pred_AFF": nA, "B_pred_nonAFF": nB},
        "population": {"A_pred_AFF": NA, "B_pred_nonAFF": NB, "total": N},
        "stratum_nonaff_rate": {"A_pred_AFF": pA, "B_pred_nonAFF": pB},
        "true_nonaff_rate": est, "se": se, "ci95": [lo, hi],
        "detector_predicted_rate": W.get("pointwise_nonaff_rate"),
        "precision_nonaff": prec, "recall_nonaff": rec, "fpr_on_aff": fpr,
    }, open(a.out, "w"), indent=2)
    print("\n  written to %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
