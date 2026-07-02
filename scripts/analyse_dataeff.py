"""
Data-efficiency analysis: operator benefit vs training-set size.

Reads outputs/logs/ledgar_ft_<agg>_n<size>_s<seed>.json (from run_dataeff.sh) and
reports, per training size, Delta(operator-aware - mean) on the aggregate test set
and on the 10-20% regime. It also emits pgfplots coordinates ready to paste into
the paper.

Prediction under the paper's thesis: Delta rises as size falls (the operator prior
substitutes for encoder adaptation when data is scarce).

Usage:
    python scripts/analyse_dataeff.py --sizes 500 1500 5000 15000 60000 --seeds 42 43 44
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path


def read(agg, size, seed, logdir):
    p = Path(logdir) / f"ledgar_ft_{agg}_n{size}_s{seed}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["test_macro_f1"], d["test_macro_f1_10_20"]


def ms(v):
    n = len(v)
    if n == 0:
        return float("nan"), 0.0, 0
    return st.mean(v), (st.stdev(v) / math.sqrt(n) if n > 1 else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1500, 5000, 15000, 60000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--ops", default="v4_soft", help="operator-aware tag component")
    ap.add_argument("--logdir", default="outputs/logs")
    args = ap.parse_args()

    print("=" * 74)
    print(" DATA-EFFICIENCY: operator benefit vs training size")
    print("=" * 74)
    print(f"{'size':>8} {'mean_agg':>10} {'ops_agg':>10} {'d_agg':>9} "
          f"{'mean_reg':>10} {'ops_reg':>10} {'d_reg':>9} {'n':>3}")
    print("-" * 74)

    rows = []
    for size in args.sizes:
        magg, mreg, oagg, oreg = [], [], [], []
        for s in args.seeds:
            rm = read("mean", size, s, args.logdir)
            ro = read(args.ops, size, s, args.logdir)
            if rm is None or ro is None:
                continue
            magg.append(rm[0]); mreg.append(rm[1])
            oagg.append(ro[0]); oreg.append(ro[1])
        n = len(magg)
        if n == 0:
            print(f"{size:>8}  (no data)")
            continue
        dagg = [o - m for o, m in zip(oagg, magg)]
        dreg = [o - m for o, m in zip(oreg, mreg)]
        ma, _, _ = ms(magg); oa, _, _ = ms(oagg)
        mr, _, _ = ms(mreg); orr, _, _ = ms(oreg)
        da, dase, _ = ms(dagg); dr, drse, _ = ms(dreg)
        print(f"{size:>8} {ma:>10.4f} {oa:>10.4f} {da:>+9.4f} "
              f"{mr:>10.4f} {orr:>10.4f} {dr:>+9.4f} {n:>3}")
        rows.append((size, da, dase, dr, drse))

    if len(rows) >= 2:
        trend = "RISES as data shrinks (supports the thesis)" \
            if rows[0][3] > rows[-1][3] else "does NOT rise as data shrinks"
        print("-" * 74)
        print(f" Regime operator benefit {trend}.")
        print("\n pgfplots coordinates (regime Delta vs size), for the paper:")
        print("   \\addplot+[error bars/y dir=both,error bars/y explicit] coordinates {")
        for size, da, dase, dr, drse in rows:
            print(f"     ({size},{dr:.4f}) +- (0,{drse:.4f})")
        print("   };")

    out = Path("outputs/logs/ledgar_dataeff.json")
    json.dump({"rows": [{"size": s, "d_agg": da, "d_agg_se": dase,
                         "d_reg": dr, "d_reg_se": drse}
                        for s, da, dase, dr, drse in rows]},
              open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
