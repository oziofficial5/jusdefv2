"""
Density / paragraph-length confound control for the LEDGAR operating regime.

The per-paragraph operator density used to define the operating regime is

    d_p = k / n,    k = number of non-AFF sentences,  n = number of sentences.

Because k and n are both integers and n is small for contract clauses, the
density bins are not free to range over paragraph length: each bin admits only
certain (n, k) pairs. This script reports

  (A) analytically, the exact set of paragraph lengths each density bin can
      contain, which needs no data at all; and
  (B) empirically, the LEDGAR test-split paragraph-length distribution and how
      much of the test set is even eligible for each bin on length grounds.

Together these establish whether "the 10-20% density bin" is separable from
"the medium-length-paragraph bin", which is the alternative explanation for the
operating-regime result that the thesis does not currently control for.

Sentence segmentation is identical to scripts/preprocess_ledgar.py so that the
paragraph lengths here are the same ones the density measurement saw.

Usage:
    python scripts/analyse_ledgar_length_confound.py

Outputs:
    outputs/logs/ledgar_length_confound.json
"""
import os
import re
import sys
import json
from pathlib import Path
from fractions import Fraction
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Bin edges as used in Chapter 7, Table 7.8. Lower bound inclusive, upper
# exclusive, except the top bin which is closed.
BINS = [
    ("0%",       Fraction(0),     Fraction(0),      True),
    # 0-5% is strictly positive: the exact-zero paragraphs are their own bin.
    ("0-5%",     None,            Fraction(1, 20),  False),
    ("5-10%",    Fraction(1, 20), Fraction(1, 10),  False),
    ("10-20%",   Fraction(1, 10), Fraction(1, 5),   False),
    ("20-30%",   Fraction(1, 5),  Fraction(3, 10),  False),
    ("30-50%",   Fraction(3, 10), Fraction(1, 2),   False),
    ("50-100%",  Fraction(1, 2),  Fraction(1),      True),
]

MAX_N = 60  # paragraph lengths to enumerate over


def sentence_split(text):
    """Identical to scripts/preprocess_ledgar.py."""
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def feasible_lengths(lo, hi, closed, max_n=MAX_N):
    """Paragraph lengths n for which some integer k gives d = k/n in the bin."""
    ok = []
    for n in range(1, max_n + 1):
        for k in range(0, n + 1):
            d = Fraction(k, n)
            if lo is None:                     # strictly positive, below hi
                hit = (Fraction(0) < d < hi)
            elif lo == hi:                     # the exact-zero bin
                hit = (d == lo)
            elif closed:
                hit = (lo <= d <= hi)
            else:
                hit = (lo <= d < hi)
            if hit:
                ok.append(n)
                break
    return ok


def main():
    report = {}

    # ---------- (A) analytical bin/length feasibility ----------
    print("=" * 74)
    print("(A) Which paragraph lengths can each density bin contain?")
    print("    Arithmetic only. No data involved.")
    print("=" * 74)
    print("%-10s %8s   %s" % ("bin", "min n", "admissible paragraph lengths n"))
    print("-" * 74)

    analytic = {}
    for name, lo, hi, closed in BINS:
        ok = feasible_lengths(lo, hi, closed)
        # excluded lengths below MAX_N, for readability
        lo_n = min(ok) if ok else None
        if name == "0%":
            desc = "any n (k=0)"
        else:
            head = [n for n in ok if n <= 24]
            desc = ", ".join(str(n) for n in head)
            if len(ok) > len(head):
                desc += ", ..."
        analytic[name] = {"min_n": lo_n, "admissible_n_upto_%d" % MAX_N: ok}
        print("%-10s %8s   %s" % (name, lo_n, desc))

    print()
    print("The 10-20%% bin is unreachable for any paragraph of fewer than %d"
          % analytic["10-20%"]["min_n"], "sentences.")
    print("The >=50%% bin is unreachable for any paragraph of more than 2"
          " sentences unless it is almost entirely non-AFF.")
    report["analytic"] = {k: {"min_n": v["min_n"]} for k, v in analytic.items()}

    # ---------- (B) empirical paragraph-length distribution ----------
    print()
    print("=" * 74)
    print("(B) LEDGAR test-split paragraph-length distribution")
    print("=" * 74)

    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="test")
    lengths = [len(sentence_split(t)) for t in ds["text"]]
    N = len(lengths)
    cnt = Counter(lengths)

    print("paragraphs: %d   mean length %.2f   median %d   max %d"
          % (N, sum(lengths) / N, sorted(lengths)[N // 2], max(lengths)))
    print()
    print("%-8s %8s %8s" % ("n sents", "count", "% of test"))
    print("-" * 26)
    for n in sorted(cnt):
        if n <= 14:
            print("%-8d %8d %7.2f%%" % (n, cnt[n], 100 * cnt[n] / N))
    tail = sum(v for k, v in cnt.items() if k > 14)
    print("%-8s %8d %7.2f%%" % (">14", tail, 100 * tail / N))
    report["length_distribution"] = {str(k): v for k, v in sorted(cnt.items())}
    report["n_test_paragraphs"] = N
    report["mean_length"] = sum(lengths) / N

    # ---------- (C) eligibility of the test set per bin, on length alone ----------
    print()
    print("=" * 74)
    print("(C) How much of the test set is eligible for each bin, by length alone?")
    print("    'Eligible' means the paragraph's sentence count admits some k")
    print("    placing it in the bin. Nothing about its operators is used.")
    print("=" * 74)
    print("%-10s %8s %10s %12s" % ("bin", "min n", "eligible", "% of test"))
    print("-" * 46)
    elig = {}
    for name, lo, hi, closed in BINS:
        ok = set(analytic[name]["admissible_n_upto_%d" % MAX_N])
        e = sum(v for k, v in cnt.items() if k in ok)
        elig[name] = e
        print("%-10s %8s %10d %11.2f%%"
              % (name, analytic[name]["min_n"], e, 100 * e / N))
    report["eligible_by_length"] = elig

    # mean length of the paragraphs eligible for the regime bin
    ok10 = set(analytic["10-20%"]["admissible_n_upto_%d" % MAX_N])
    reg_lengths = [n for n in lengths if n in ok10]
    if reg_lengths:
        print()
        print("Paragraphs eligible for the 10-20%% regime bin: n=%d, mean length %.2f"
              % (len(reg_lengths), sum(reg_lengths) / len(reg_lengths)))
        print("All other paragraphs:                          n=%d, mean length %.2f"
              % (N - len(reg_lengths),
                 (sum(lengths) - sum(reg_lengths)) / (N - len(reg_lengths))))
        report["regime_eligible_mean_length"] = sum(reg_lengths) / len(reg_lengths)
        report["other_mean_length"] = (sum(lengths) - sum(reg_lengths)) / (N - len(reg_lengths))

    out = Path("outputs/logs/ledgar_length_confound.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
