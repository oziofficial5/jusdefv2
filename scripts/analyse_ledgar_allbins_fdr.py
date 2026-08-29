"""
All-density-bins significance with Benjamini-Hochberg correction (reviewer request:
guard the 10-20% regime against multiple-comparison / selection bias).

For each density bin, compute the per-seed delta (v3 - mean) macro-F1, a one-sample
t-test against 0 across seeds, then BH-correct the p-values across all bins.
Fills Table (all bins) in the paper.

Usage:
    python scripts/analyse_ledgar_allbins_fdr.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.model.jusdef_ledgar import JusDefLEDGAR

BINS = [("0%", 0.0, 0.001), ("5-10%", 0.05, 0.10), ("10-20%", 0.10, 0.20),
        ("20-30%", 0.20, 0.30), ("30-50%", 0.30, 0.50), (">=50%", 0.50, 1.01)]


def model(v, d):
    return JusDefLEDGAR(in_dim=768, hidden_dim=512, num_classes=100,
                        num_layers=1, dmp_variant=v).to(d)


@torch.no_grad()
def predict(m, data, d):
    m.eval(); out = []
    for p in data:
        se = p["embeddings"].to(d)
        s2p = torch.zeros(se.size(0), dtype=torch.long, device=d)
        op = torch.tensor(p["operators"], dtype=torch.long, device=d)
        out.append(int(m(se, s2p, op, num_paragraphs=1).argmax(-1).item()))
    return np.array(out)


def macro(labels, preds, mask):
    ls, ps = labels[mask], preds[mask]
    present = sorted(set(ls.tolist()))
    return f1_score(ls, ps, labels=present, average="macro", zero_division=0)


def bh(pvals):
    """Benjamini-Hochberg corrected q-values."""
    p = np.array(pvals); n = len(p); order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank
        val = min(prev, p[idx] * n / i)
        q[idx] = val; prev = val
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--ckpt", default="outputs/checkpoints")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from scipy.stats import ttest_1samp

    test = pickle.load(open("data/processed_ledgar/test_processed.pkl", "rb"))
    density = np.array([sum(1 for o in d["operators"] if o != 0) /
                        max(len(d["operators"]), 1) for d in test])
    labels = np.array([d["label"] for d in test])
    ck = Path(args.ckpt)

    per_bin = {name: [] for name, _, _ in BINS}
    for s in args.seeds:
        pm = ck / f"ledgar_baseline_mean_s{s}.pt"
        pv = ck / f"ledgar_v3_pilot_s{s}.pt"
        if not (pm.exists() and pv.exists()):
            print(f"[seed {s}] skip (missing)"); continue
        mm = model("mean", dev); mm.load_state_dict(torch.load(pm, map_location=dev))
        vv = model("v3", dev); vv.load_state_dict(torch.load(pv, map_location=dev))
        pm_ = predict(mm, test, dev); pv_ = predict(vv, test, dev)
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            if mask.sum() < 3:
                per_bin[name].append(np.nan); continue
            per_bin[name].append(macro(labels, pv_, mask) - macro(labels, pm_, mask))

    names, deltas, ps, ns = [], [], [], []
    for name, lo, hi in BINS:
        vals = np.array([x for x in per_bin[name] if not np.isnan(x)])
        mask = (density >= lo) & (density < hi)
        if len(vals) < 2:
            names.append(name); deltas.append(np.nan); ps.append(1.0); ns.append(int(mask.sum())); continue
        t, p = ttest_1samp(vals, 0.0)
        # one-sided (positive) p when mean>0
        p_one = p / 2 if vals.mean() > 0 else 1 - p / 2
        names.append(name); deltas.append(vals.mean()); ps.append(p_one); ns.append(int(mask.sum()))
    q = bh(ps)

    n_seeds = max(len([x for x in per_bin[b] if not np.isnan(x)]) for b, _, _ in BINS)
    print("=" * 78)
    print(f" ALL-BINS SIGNIFICANCE (Benjamini-Hochberg across {len(BINS)} bins,"
          f" {n_seeds} seeds)")
    print("=" * 78)
    print(f"{'bin':<10}{'N':>7}{'delta':>10}{'sd':>9}{'p(1-sided)':>13}{'q(BH)':>10}  sig")
    rows = []
    for name, d_, p_, q_, n_ in zip(names, deltas, ps, q, ns):
        vals = np.array([x for x in per_bin[name] if not np.isnan(x)])
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        sig = "*" if (q_ < 0.05 and d_ > 0) else ""
        print(f"{name:<10}{n_:>7}{d_:>+10.4f}{sd:>9.4f}{p_:>13.4f}{q_:>10.4f}  {sig}")
        rows.append({"bin": name, "N": n_, "delta": float(d_), "sd": sd,
                     "n_seeds": int(len(vals)),
                     "seeds_positive": int((vals > 0).sum()),
                     "p_one_sided": float(p_), "q_bh": float(q_),
                     "significant_positive": bool(q_ < 0.05 and d_ > 0),
                     "per_seed_deltas": [float(x) for x in vals]})
    print("\n(Only bins with q<0.05 AND delta>0 are significant positives.)")

    # Persist, so the correction is an artefact rather than a console message.
    # Chapter 7's multiple-comparison section cites these numbers; they should
    # be regenerable, and the per-bin deltas here are also the data behind the
    # density-stratified headline figure.
    out = Path("outputs/logs/ledgar_allbins_fdr.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"seeds": list(args.seeds), "n_seeds_used": n_seeds,
         "fdr_level": 0.05, "test": "one-sided one-sample t on across-seed deltas",
         "bins": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
