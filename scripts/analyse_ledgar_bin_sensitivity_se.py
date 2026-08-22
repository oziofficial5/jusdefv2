"""
Standard errors and interval estimates for the LEDGAR bin-sensitivity sweep.

Table 7.10 of the thesis reports a five-seed mean delta per candidate bin with a
+/- that is the seed-to-seed standard deviation. A standard deviation describes
the spread of the five runs; it is not the precision of their mean, and readers
comparing bins need the latter. This script recomputes, from the per-seed deltas
already logged by scripts/analyse_ledgar_bin_sensitivity.py:

    mean, sample SD, standard error, 95% CI (t, df=4), one-sample t vs 0,
    seeds positive, and the relationship between bin population and effect size.

The last of these matters: the deltas grow as the bins narrow, which is the
signature both of a genuinely concentrated effect and of noise-mining, and the
thesis should say which it can distinguish.

Usage:
    python scripts/analyse_ledgar_bin_sensitivity_se.py

Reads:   outputs/logs/ledgar_bin_sensitivity_analysis.json
Outputs: outputs/logs/ledgar_bin_sensitivity_se.json
"""
import json
import math
from pathlib import Path

# two-sided t critical value at 95%, df = n-1
T_CRIT = {4: 2.776445, 9: 2.262157}

SYMMETRIC = ["8-22", "9-21", "10-20", "11-19", "12-18"]
ASYMMETRIC = ["8-15", "15-25"]


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)      # sample variance
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    tc = T_CRIT[n - 1]
    t = m / se if se > 0 else float("inf")
    return dict(n=n, mean=m, sd=sd, se=se, lo=m - tc * se, hi=m + tc * se,
                t=t, pos=sum(1 for x in xs if x > 0))


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def main():
    src = Path("outputs/logs/ledgar_bin_sensitivity_analysis.json")
    d = json.loads(src.read_text())
    per_bin = d["per_bin"]

    print("=" * 96)
    print("Bin-sensitivity sweep: mean, SD, SE and 95% CI on the five-seed delta")
    print("=" * 96)
    print("%-8s %6s %9s %9s %9s %-20s %7s %6s" %
          ("bin", "N", "mean", "SD", "SE", "95% CI", "t(4)", "pos"))
    print("-" * 96)

    out = {}
    for name in SYMMETRIC + ASYMMETRIC:
        b = per_bin[name]
        s = stats(b["per_seed_delta"])
        out[name] = dict(n_paragraphs=b["n_paragraphs"], **s)
        if name == ASYMMETRIC[0]:
            print("-" * 96)
        star = "*" if s["lo"] > 0 else " "
        print("%-8s %6d %+9.4f %9.4f %9.4f  [%+.4f, %+.4f]%s %7.2f %5d/5" %
              (name, b["n_paragraphs"], s["mean"], s["sd"], s["se"],
               s["lo"], s["hi"], star, s["t"], s["pos"]))

    print()
    print("* = 95% CI excludes zero.")
    print("The seven bins overlap heavily (8-22 contains 10-20 contains 12-18),")
    print("so these are not seven independent tests and no correction across them")
    print("would be meaningful. The sweep probes stability, not significance.")

    # ---- does the effect size track the bin population? ----
    print()
    print("=" * 96)
    print("Effect size against bin population")
    print("=" * 96)
    ns = [per_bin[k]["n_paragraphs"] for k in SYMMETRIC]
    ds = [out[k]["mean"] for k in SYMMETRIC]
    r_sym = pearson(ns, ds)
    ns_all = [per_bin[k]["n_paragraphs"] for k in SYMMETRIC + ASYMMETRIC]
    ds_all = [out[k]["mean"] for k in SYMMETRIC + ASYMMETRIC]
    r_all = pearson(ns_all, ds_all)

    print("symmetric shifts only : Pearson r(N, delta) = %+.3f  (k=%d)" % (r_sym, len(ns)))
    print("including asymmetric  : Pearson r(N, delta) = %+.3f  (k=%d)" % (r_all, len(ns_all)))
    print()
    print("Widest symmetric bin  8-22: N=%d, delta=%+.4f" % (ns[0], ds[0]))
    print("Narrowest symmetric  12-18: N=%d, delta=%+.4f" % (ns[-1], ds[-1]))
    print("ratio of deltas: %.2fx on %.2fx the paragraphs"
          % (ds[-1] / ds[0], ns[-1] / ns[0]))
    out["_correlation_N_delta_symmetric"] = r_sym
    out["_correlation_N_delta_all"] = r_all

    dst = Path("outputs/logs/ledgar_bin_sensitivity_se.json")
    dst.write_text(json.dumps(out, indent=2))
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
