"""
Per-class decomposition of the 10-20% regime gain on LEDGAR.

Reads outputs/logs/predictions_10_20_bin.json (written by
dump_predictions_10_20.py) and computes, for each LEDGAR clause type present in
the 10-20% non-AFF density bin, the one-vs-rest F1 of the operator-aware model
(v3_pilot) and of the mean baseline, averaged over seeds. Reports the clause
types that gain and lose the most under the operator prior.

This answers the reviewer-anticipated question "which clause types actually
benefit from operator-aware aggregation?" and decomposes the +0.075 regime
macro-F1 by class.

Outputs:
    outputs/logs/perclass_regime_gain.json  (full per-class table + macro sanity)

No GPU required.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PRED = Path("outputs/logs/predictions_10_20_bin.json")
OUT = Path("outputs/logs/perclass_regime_gain.json")
NAMES = Path("data/processed_ledgar/label_names.json")  # optional {idx: name}


def per_seed_class_f1(d, tag, y_true, classes):
    """tag in {mean, v3_pilot}. Returns {class: [f1 per seed]}."""
    acc = {c: [] for c in classes}
    for _seed, preds in d[tag].items():
        f1s = f1_score(y_true, np.asarray(preds), labels=classes,
                       average=None, zero_division=0)
        for c, f in zip(classes, f1s):
            acc[c].append(float(f))
    return acc


def main():
    if not PRED.exists():
        raise SystemExit(f"missing {PRED}; run dump_predictions_10_20.py first")
    d = json.load(open(PRED))

    y_true = np.asarray(d["true_labels"])
    classes = sorted(set(int(c) for c in y_true.tolist()))
    n_seeds_v3 = len(d.get("v3_pilot", {}))
    n_seeds_mean = len(d.get("mean", {}))
    print(f"paragraphs: {len(y_true)} | classes present: {len(classes)} | "
          f"seeds (mean/v3): {n_seeds_mean}/{n_seeds_v3}")

    f1_mean = per_seed_class_f1(d, "mean", y_true, classes)
    f1_v3 = per_seed_class_f1(d, "v3_pilot", y_true, classes)

    names = {}
    if NAMES.exists():
        names = {int(k): v for k, v in json.load(open(NAMES)).items()}

    support = {c: int((y_true == c).sum()) for c in classes}
    rows = []
    for c in classes:
        m = float(np.mean(f1_mean[c]))
        v = float(np.mean(f1_v3[c]))
        rows.append({"class": c, "name": names.get(c, ""), "support": support[c],
                     "f1_mean": m, "f1_v3": v, "delta": v - m})
    rows.sort(key=lambda r: r["delta"], reverse=True)

    def fmt(r):
        return (f"  cls {r['class']:>3} {r['name'][:34]:<34} n={r['support']:>3}  "
                f"mean={r['f1_mean']:.3f}  v3={r['f1_v3']:.3f}  d={r['delta']:+.3f}")

    print("\n=== TOP 10 clause types by gain (v3 - mean) ===")
    for r in rows[:10]:
        print(fmt(r))
    print("\n=== BOTTOM 10 clause types (v3 - mean) ===")
    for r in rows[-10:]:
        print(fmt(r))

    macro_mean = float(np.mean([r["f1_mean"] for r in rows]))
    macro_v3 = float(np.mean([r["f1_v3"] for r in rows]))
    print(f"\nmacro-F1 over {len(rows)} present classes: "
          f"mean={macro_mean:.4f}  v3={macro_v3:.4f}  "
          f"delta={macro_v3 - macro_mean:+.4f}  (sanity vs the +0.075 regime gain)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_paragraphs": int(len(y_true)),
               "n_classes_present": len(classes),
               "macro_mean": macro_mean, "macro_v3": macro_v3,
               "macro_delta": macro_v3 - macro_mean, "rows": rows},
              open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
