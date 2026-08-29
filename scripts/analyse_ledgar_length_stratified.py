"""
Sentence-count stratification of the LEDGAR operating regime.

This is the experiment identified as outstanding in the thesis (Ch. 7,
Sec. "The density measure is confounded with paragraph length"; Ch. 9,
external validity; Ch. 10, future work). It decides between two accounts of
the 10-20% operating regime that the density-stratified evaluation cannot
separate, because the bin that selects for one also selects for the other:

  (i)  operator account  - JusDef-SP wins where a paragraph carries a small
                           number of explicit defeasible operators;
  (ii) length account    - JusDef-SP wins where a paragraph is long enough for
                           a signed soft-attention pooler to beat a mean pooler,
                           and the operators are along for the ride.

The confound is arithmetic. With d_p = k/n over integers, membership of the
10-20% band requires 5k < n <= 10k and therefore n >= 6, so no paragraph of
five or fewer sentences can enter the regime whatever its operators say.

The design below is a 2x2 rather than a single stratification, because that is
what identifies the effect:

  (A) marginal    - delta per sentence-count bin, ignoring density. If the
                    advantage appears in every long-paragraph bin regardless of
                    density, the regime is a length effect.
  (B) 2x2 control - restricted to paragraphs with n >= 6 (i.e. all of them
                    eligible for the band on length grounds), compare those
                    *inside* the 10-20% density band against those *outside*
                    it. Length is held approximately fixed by construction;
                    only density varies. This is the decisive cell.
  (C) within-bin  - delta per sentence-count bin *within* the 10-20% band, to
                    check whether the in-band advantage itself grows with
                    length.

Verdicts printed at the end follow directly from (B): a positive delta on the
in-band subset and a null on the out-of-band subset supports the operator
account; comparable positive deltas on both supports the length account.

Requires no training. Inference only, over the released test split with the
existing ten-seed LEDGAR checkpoints.

Usage:
    python scripts/analyse_ledgar_length_stratified.py \
        --seeds 42 43 44 45 46 47 48 49 50 51

Outputs:
    outputs/logs/ledgar_length_stratified.json
"""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.model.jusdef_ledgar import JusDefLEDGAR  # noqa: E402

# Sentence-count bins. The boundary at 6 is the one that matters: it is the
# minimum length admitting membership of the 10-20% density band.
LENGTH_BINS = [
    ("n=1", 1, 1),
    ("n=2", 2, 2),
    ("n=3", 3, 3),
    ("n=4-5", 4, 5),
    ("n=6-10", 6, 10),
    ("n>=11", 11, 10 ** 6),
]

REGIME_LO, REGIME_HI = 0.10, 0.20
MIN_ELIGIBLE_N = 6  # from 5k < n <= 10k with k >= 1


def model(variant, device):
    return JusDefLEDGAR(in_dim=768, hidden_dim=512, num_classes=100,
                        num_layers=1, dmp_variant=variant).to(device)


@torch.no_grad()
def predict(m, data, device):
    m.eval()
    out = []
    for p in data:
        se = p["embeddings"].to(device)
        s2p = torch.zeros(se.size(0), dtype=torch.long, device=device)
        op = torch.tensor(p["operators"], dtype=torch.long, device=device)
        out.append(int(m(se, s2p, op, num_paragraphs=1).argmax(-1).item()))
    return np.array(out)


def macro_f1(labels, preds, mask):
    ls, ps = labels[mask], preds[mask]
    present = sorted(set(ls.tolist()))
    return f1_score(ls, ps, labels=present, average="macro", zero_division=0)


def accuracy(labels, preds, mask):
    return float((labels[mask] == preds[mask]).mean())


def summarise(deltas):
    """Mean, sample sd, se and a 95% t interval, matching the thesis convention."""
    vals = [d for d in deltas if d is not None and not math.isnan(d)]
    n = len(vals)
    if n < 2:
        return {"n_seeds": n, "mean": (vals[0] if n else None), "sd": None,
                "se": None, "ci_lo": None, "ci_hi": None, "seeds_positive": None}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    # two-sided 95% t critical values for the panel sizes we ever use here
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return {"n_seeds": n, "mean": m, "sd": sd, "se": se,
            "ci_lo": m - tcrit * se, "ci_hi": m + tcrit * se,
            "seeds_positive": sum(1 for v in vals if v > 0)}


def row(name, N, s, extra=""):
    if s["mean"] is None:
        return "%-22s %7d   %s" % (name, N, "(insufficient seeds)")
    if s["sd"] is None:
        return "%-22s %7d  %+8.4f %s" % (name, N, s["mean"], extra)
    return ("%-22s %7d  %+8.4f  %7.4f  %7.4f  [%+.4f, %+.4f]  %2d/%-2d %s"
            % (name, N, s["mean"], s["sd"], s["se"], s["ci_lo"], s["ci_hi"],
               s["seeds_positive"], s["n_seeds"], extra))


HEAD = ("%-22s %7s  %8s  %7s  %7s  %-20s %5s"
        % ("subset", "N", "delta", "sd", "se", "95% CI", "+/n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--ckpt", default="outputs/checkpoints")
    ap.add_argument("--test", default="data/processed_ledgar/test_processed.pkl")
    ap.add_argument("--metric", choices=["macro_f1", "accuracy", "both"],
                    default="both")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test = pickle.load(open(args.test, "rb"))
    labels = np.array([d["label"] for d in test])
    n_sent = np.array([max(len(d["operators"]), 1) for d in test])
    k_nonaff = np.array([sum(1 for o in d["operators"] if o != 0) for d in test])
    density = k_nonaff / n_sent
    in_band = (density >= REGIME_LO) & (density < REGIME_HI)
    eligible = n_sent >= MIN_ELIGIBLE_N

    print("test paragraphs: %d  mean length %.2f  median %d  max %d"
          % (len(test), n_sent.mean(), int(np.median(n_sent)), n_sent.max()))
    print("in 10-20%% band: %d    length-eligible (n>=%d): %d"
          % (in_band.sum(), MIN_ELIGIBLE_N, eligible.sum()))

    # ---- masks ------------------------------------------------------------
    masks = {}
    for name, lo, hi in LENGTH_BINS:
        masks[("A", name)] = (n_sent >= lo) & (n_sent <= hi)
    masks[("B", "n>=6 & in band")] = eligible & in_band
    masks[("B", "n>=6 & out of band")] = eligible & ~in_band
    masks[("B", "n<6 (band impossible)")] = ~eligible
    for name, lo, hi in LENGTH_BINS:
        if lo >= MIN_ELIGIBLE_N:
            masks[("C", name + " & in band")] = (n_sent >= lo) & (n_sent <= hi) & in_band
    masks[("REF", "all paragraphs")] = np.ones(len(test), dtype=bool)
    masks[("REF", "10-20% band")] = in_band

    # ---- per-seed inference ----------------------------------------------
    per_seed = {k: {"macro_f1": [], "accuracy": []} for k in masks}
    seeds_used = []
    for s in args.seeds:
        pm = Path(args.ckpt) / f"ledgar_baseline_mean_s{s}.pt"
        pv = Path(args.ckpt) / f"ledgar_v3_pilot_s{s}.pt"
        if not (pm.exists() and pv.exists()):
            print(f"[seed {s}] skip (checkpoint missing)")
            continue
        mm = model("mean", device)
        mm.load_state_dict(torch.load(pm, map_location=device))
        vv = model("v3", device)
        vv.load_state_dict(torch.load(pv, map_location=device))
        p_mean = predict(mm, test, device)
        p_v3 = predict(vv, test, device)
        seeds_used.append(s)
        for key, mask in masks.items():
            if mask.sum() < 3:
                per_seed[key]["macro_f1"].append(float("nan"))
                per_seed[key]["accuracy"].append(float("nan"))
                continue
            per_seed[key]["macro_f1"].append(
                macro_f1(labels, p_v3, mask) - macro_f1(labels, p_mean, mask))
            per_seed[key]["accuracy"].append(
                accuracy(labels, p_v3, mask) - accuracy(labels, p_mean, mask))
        print(f"[seed {s}] done")

    if not seeds_used:
        raise SystemExit("No checkpoints found - nothing to do.")
    if len(seeds_used) < len(args.seeds):
        missing = [s for s in args.seeds if s not in seeds_used]
        print()
        print("!" * 78)
        print(" WARNING: ran on %d of %d requested seeds. Missing: %s"
              % (len(seeds_used), len(args.seeds), missing))
        print(" Every figure and number derived from this run is a %d-SEED result."
              % len(seeds_used))
        print(" Label it as such. Do not quote it beside the ten-seed headline")
        print(" without saying which panel each number comes from.")
        print("!" * 78)

    # ---- report -----------------------------------------------------------
    report = {"seeds": seeds_used, "n_test": int(len(test)),
              "mean_sentences": float(n_sent.mean()),
              "n_in_band": int(in_band.sum()),
              "n_length_eligible": int(eligible.sum()),
              "regime_bin": [REGIME_LO, REGIME_HI],
              "min_eligible_n": MIN_ELIGIBLE_N,
              "results": {}}

    metrics = ["macro_f1", "accuracy"] if args.metric == "both" else [args.metric]

    for metric in metrics:
        print()
        print("=" * 96)
        print(" JusDef-SP minus mean baseline, %s, %d seeds" % (metric, len(seeds_used)))
        print("=" * 96)

        for section, title in [
            ("REF", "reference (as reported in the thesis)"),
            ("A", "(A) marginal: by sentence count, density ignored"),
            ("B", "(B) 2x2 control: length held above the band threshold"),
            ("C", "(C) within the 10-20%% band, by sentence count"),
        ]:
            keys = [k for k in masks if k[0] == section]
            if not keys:
                continue
            print()
            print(title)
            print(HEAD)
            print("-" * 96)
            for key in keys:
                s = summarise(per_seed[key][metric])
                N = int(masks[key].sum())
                print(row(key[1], N, s))
                report["results"].setdefault(metric, {})[key[1]] = {
                    "section": section, "N": N,
                    "per_seed": per_seed[key][metric], **s}

    # ---- verdict ----------------------------------------------------------
    band = summarise(per_seed[("B", "n>=6 & in band")]["macro_f1"])
    outb = summarise(per_seed[("B", "n>=6 & out of band")]["macro_f1"])
    print()
    print("=" * 96)
    print(" VERDICT (macro-F1, section B)")
    print("=" * 96)
    if band["mean"] is None or outb["mean"] is None:
        print(" insufficient data")
    else:
        band_pos = band["ci_lo"] is not None and band["ci_lo"] > 0
        out_pos = outb["ci_lo"] is not None and outb["ci_lo"] > 0
        print(" among paragraphs long enough to enter the band (n >= %d):" % MIN_ELIGIBLE_N)
        print("   inside  the 10-20%% band: %+.4f  %s"
              % (band["mean"], "(interval excludes 0)" if band_pos else "(interval includes 0)"))
        print("   outside the 10-20%% band: %+.4f  %s"
              % (outb["mean"], "(interval excludes 0)" if out_pos else "(interval includes 0)"))
        print()
        if band_pos and not out_pos:
            print(" -> Supports the OPERATOR account. At approximately fixed length the")
            print("    advantage is confined to the density band. C8 stands as written.")
        elif out_pos and band_pos:
            print(" -> Supports the LENGTH account. The advantage appears on long")
            print("    paragraphs whether or not they are in the density band.")
            print("    C8 must be restated as a paragraph-length effect.")
        elif out_pos and not band_pos:
            print(" -> Supports the LENGTH account, strongly and against the thesis:")
            print("    the advantage is on long out-of-band paragraphs only.")
        else:
            print(" -> INCONCLUSIVE at this panel size. Neither subset separates from")
            print("    zero; report as such rather than as support for either account.")
        print()
        print(" difference (in-band minus out-of-band): %+.4f" % (band["mean"] - outb["mean"]))

    out = Path("outputs/logs/ledgar_length_stratified.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
