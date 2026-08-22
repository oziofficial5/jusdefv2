"""
Recompute the dispersion statistics behind Tables 7.9 and 7.10 on one convention.

Two logs report the standard deviation of the same five per-seed deltas and
disagree, because analyse_ledgar_bin_sensitivity.py uses the sample standard
deviation (ddof=1) and the density-stratified analysis uses the population one
(ddof=0). For n = 5 runs treated as a sample from the seed distribution, ddof=1
is the correct choice, and it is the one adopted here.

The script prints, for every bin in both tables, the mean delta with its sample
SD, its standard error, and a 95% t interval, so that the thesis can report the
precision of each mean rather than only the spread of its five runs.

Usage:
    python scripts/recompute_ledgar_dispersion.py

Reads:
    outputs/logs/ledgar_5seed_stratified.json
    outputs/logs/ledgar_bin_sensitivity_analysis.json
Outputs:
    outputs/logs/ledgar_dispersion_recomputed.json
"""
import json
import math
from pathlib import Path

T95_DF4 = 2.776445


def summarise(xs):
    n = len(xs)
    m = sum(xs) / n
    sd1 = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))   # sample, ddof=1
    sd0 = math.sqrt(sum((x - m) ** 2 for x in xs) / n)         # population
    se = sd1 / math.sqrt(n)
    return dict(n=n, mean=m, sd_sample=sd1, sd_population=sd0, se=se,
                ci_lo=m - T95_DF4 * se, ci_hi=m + T95_DF4 * se,
                pos=sum(1 for x in xs if x > 0))


def show(title, rows):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    print("%-10s %7s %9s %9s %9s %9s %-22s %4s" %
          ("bin", "N", "mean", "SD(n-1)", "SD(n)", "SE", "95% CI", "pos"))
    print("-" * 92)
    for name, N, s in rows:
        star = "*" if s["ci_lo"] > 0 or s["ci_hi"] < 0 else " "
        print("%-10s %7s %+9.4f %9.4f %9.4f %9.4f  [%+.4f, %+.4f]%s %3d/5" %
              (name, N, s["mean"], s["sd_sample"], s["sd_population"], s["se"],
               s["ci_lo"], s["ci_hi"], star, s["pos"]))


def main():
    out = {}

    strat = json.loads(Path("outputs/logs/ledgar_5seed_stratified.json").read_text())
    rows = []
    for e in strat:
        s = summarise(e["per_seed_deltas"])
        rows.append((e["bin"], e["N"], s))
        out.setdefault("stratified", {})[e["bin"]] = dict(N=e["N"], **s)
    show("Table 7.9  density-stratified deltas (JusDef-SP minus mean baseline)", rows)

    sweep = json.loads(Path("outputs/logs/ledgar_bin_sensitivity_analysis.json").read_text())
    rows = []
    for name in ["8-22", "9-21", "10-20", "11-19", "12-18", "8-15", "15-25"]:
        b = sweep["per_bin"][name]
        s = summarise(b["per_seed_delta"])
        rows.append((name, b["n_paragraphs"], s))
        out.setdefault("bin_sweep", {})[name] = dict(N=b["n_paragraphs"], **s)
    show("Table 7.10  bin-sensitivity sweep", rows)

    print()
    print("* = 95% interval excludes zero.")
    print("SD(n-1) is the sample standard deviation and is the convention adopted;")
    print("SD(n) is shown only to identify which figure a given table currently reports.")

    dst = Path("outputs/logs/ledgar_dispersion_recomputed.json")
    dst.write_text(json.dumps(out, indent=2))
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
