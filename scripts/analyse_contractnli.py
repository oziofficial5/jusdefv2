"""
Compare operator-aware (v3) vs baseline (mean) on ContractNLI, from result JSONs.

Reports, across seeds, macro-F1 (3 classes), mentioned-only macro-F1
(Entailment vs Contradiction -- where defeasibility matters most), and per-class F1,
plus the paired Delta (v3 - mean). A positive mentioned-only Delta is the headline
signal that the operator prior helps where defeat determines the label.

Usage: python scripts/analyse_contractnli.py --seeds 42 43 44
"""
import argparse, json, math
import statistics as st
from pathlib import Path

KEYS = ["macro_f1", "mentioned_macro_f1", "f1_NM", "f1_Ent", "f1_Con"]


def load(agg, seeds, logdir):
    out = {}
    for s in seeds:
        p = Path(logdir) / f"cnli_{agg}_s{s}.json"
        if p.exists():
            out[s] = json.load(open(p))
    return out


def ms(v):
    n = len(v)
    return (st.mean(v), (st.stdev(v) / math.sqrt(n) if n > 1 else 0.0), n) if n else (float("nan"), 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--logdir", default="outputs/logs")
    args = ap.parse_args()
    M, V = load("mean", args.seeds, args.logdir), load("v3", args.seeds, args.logdir)
    if not M or not V:
        print("Missing results (train both variants first)."); return

    print("=" * 68)
    print(f" CONTRACTNLI: operator-aware (v3) vs baseline (mean), seeds {args.seeds}")
    print("=" * 68)
    print(f"{'metric':<22}{'mean':>14}{'v3':>14}{'Delta (v3-mean)':>18}")
    print("-" * 68)
    common = sorted(set(M) & set(V))
    for k in KEYS:
        mv = [M[s][k] for s in common]; vv = [V[s][k] for s in common]
        dm = [V[s][k] - M[s][k] for s in common]
        ma, mse, _ = ms(mv); va, vse, _ = ms(vv); da, dse, n = ms(dm)
        star = " *" if (da > 0 and abs(da) > dse) else ""
        print(f"{k:<22}{ma:>9.4f}±{mse:.3f}{va:>9.4f}±{vse:.3f}{da:>+13.4f}±{dse:.3f}{star}")

    dm = [V[s]["mentioned_macro_f1"] - M[s]["mentioned_macro_f1"] for s in common]
    da, dse, n = ms(dm)
    print("\n VERDICT (mentioned-only = Entailment vs Contradiction, the defeasible subset)")
    if da > 0 and abs(da) > dse:
        print(f"  Operator prior HELPS where defeat determines the label: "
              f"+{da:.4f} (SE {dse:.3f}, {sum(1 for x in dm if x>0)}/{n} seeds).")
        print("  This is the positive result Pivot B was after --- report it.")
    elif da > 0:
        print(f"  Operator prior shows a positive but noisy mentioned-only trend "
              f"(+{da:.4f}, SE {dse:.3f}); add seeds.")
    else:
        print(f"  No operator benefit on the defeasible subset ({da:+.4f}); honest negative "
              "--- ContractNLI's exceptions are not exploited by the current head.")
    if V:
        print("\n learned v3 operator coefficients (per seed):")
        for s in common:
            c = V[s].get("coef")
            if c: print(f"   seed {s}: AFF {c[0]:+.2f} NEG {c[1]:+.2f} EXC {c[2]:+.2f} OVR {c[3]:+.2f}")


if __name__ == "__main__":
    main()
