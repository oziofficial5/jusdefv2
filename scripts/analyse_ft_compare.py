"""
Compare F2 fine-tuned-encoder variants from their result JSONs (no GPU).

Reads outputs/logs/ledgar_ft_<agg>_s<seed>.json (written by train_ledgar_ft.py)
and reports, per aggregation head, the aggregate and 10-20% regime macro-F1, plus
the delta of each operator head vs the FT+mean control.

The thesis question F2 answers:
    Does an operator-aware head beat FT+mean ON TOP of a fine-tuned encoder,
    especially in the 10-20% regime?

Usage:
    python scripts/analyse_ft_compare.py --aggs mean v4_soft --seeds 42 43 44
"""
import argparse
import json
from pathlib import Path
import statistics as st
import math


def load(agg, seeds, logdir):
    rows = {}
    for s in seeds:
        p = Path(logdir) / f"ledgar_ft_{agg}_s{s}.json"
        if p.exists():
            d = json.load(open(p))
            rows[s] = (d["test_macro_f1"], d["test_macro_f1_10_20"])
    return rows


def ms(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), 0.0, 0)
    sd = st.stdev(vals) if n > 1 else 0.0
    return (st.mean(vals), sd / math.sqrt(n) if n > 1 else 0.0, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggs", nargs="+", default=["mean", "v4_soft"])
    ap.add_argument("--ref", default="mean", help="reference head to delta against")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--logdir", default="outputs/logs")
    args = ap.parse_args()

    data = {agg: load(agg, args.seeds, args.logdir) for agg in args.aggs}
    if args.ref not in data or not data[args.ref]:
        print(f"No reference '{args.ref}' results found."); return

    print("=" * 70)
    print(f" F2 FINE-TUNED-ENCODER COMPARISON  (seeds {args.seeds})")
    print("=" * 70)
    print(f"{'head':<12}{'agg macro':>16}{'10-20% macro':>18}{'n':>5}")
    print("-" * 70)
    for agg in args.aggs:
        macs = [v[0] for v in data[agg].values()]
        bins = [v[1] for v in data[agg].values()]
        ma, mse, n = ms(macs); ba, bse, _ = ms(bins)
        print(f"{agg:<12}{ma:>10.4f}±{mse:.3f}{ba:>11.4f}±{bse:.3f}{n:>5}")

    ref = data[args.ref]
    print("\n DELTAS vs FT+" + args.ref + " (paired by seed)")
    print("-" * 70)
    for agg in args.aggs:
        if agg == args.ref:
            continue
        common = sorted(set(ref) & set(data[agg]))
        if not common:
            print(f"  {agg}: no common seeds"); continue
        d_agg = [data[agg][s][0] - ref[s][0] for s in common]
        d_bin = [data[agg][s][1] - ref[s][1] for s in common]
        am, ase, n = ms(d_agg); bm, bse, _ = ms(d_bin)
        print(f"  {agg} vs {args.ref}:")
        print(f"     aggregate  Δ {am:+.4f} ± {ase:.4f}  | seeds>0: {sum(1 for x in d_agg if x>0)}/{n}")
        print(f"     10-20%     Δ {bm:+.4f} ± {bse:.4f}  | seeds>0: {sum(1 for x in d_bin if x>0)}/{n}")
        verdict = []
        verdict.append("regime WIN" if bm > 0 else "regime no-gain")
        verdict.append("aggregate WIN" if am > 0 else ("aggregate tie" if abs(am) <= ase else "aggregate loss"))
        print(f"     -> {', '.join(verdict)}")

    print("\n (reference: paragraph-level LegalBERT-FT ~0.80 macro from "
          "train_legalbert_ledgar.py — compare the agg-macro column above to it.)")


if __name__ == "__main__":
    main()
