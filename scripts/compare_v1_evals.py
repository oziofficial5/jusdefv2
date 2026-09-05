"""Why does eval_v1_yexc.py report v1 overall 0.2026 when run_thesis_gaps.sh
PART 2 reported 0.1717 +/- 0.0609, from the same checkpoints?

Both numbers cannot stand. This prints the per-seed values and the thresholds
each route chose, which identifies the cause without re-running anything.

label_adj is already ruled out: it is consumed by the loss (trainer.py:86), not
by forward_one_graph, so it cannot affect inference.

The live hypothesis is threshold selection. train_jusdef.py tunes tau during
training and reports test macro-F1 at that tau; eval_v1_yexc.py re-tunes tau on
the validation split from the saved checkpoint. If one seed picked a
pathological tau during training, PART 2's mean drops and its spread widens --
which matches what we see: PART 2 sd 0.0609 against 0.008 here.
"""
import json
import glob
import os

print("=" * 74)
print("  PART 2 training logs (outputs/logs/jusdef_v1_corrected*_s*.json)")
print("=" * 74)
train = {}
for f in sorted(glob.glob("outputs/logs/jusdef_v1_corrected*_s*.json")):
    if "smoke" in os.path.basename(f):
        continue
    d = json.load(open(f))
    seed = d.get("config", {}).get("seed", "?")
    train[str(seed)] = d
    print("  seed %-4s test_macro_f1 %-9s val_threshold %-8s best_val %s"
          % (seed,
             round(d.get("test_macro_f1", float("nan")), 4),
             d.get("val_threshold", "n/a"),
             round(d.get("best_val_macro_f1", float("nan")), 4)))
if train:
    v = [d["test_macro_f1"] for d in train.values() if "test_macro_f1" in d]
    if v:
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        print("  mean %.4f  sd %.4f  (n=%d)" % (m, sd, len(v)))

print()
print("=" * 74)
print("  eval_v1_yexc.py (outputs/logs/v1_yexc.json)")
print("=" * 74)
p = "outputs/logs/v1_yexc.json"
if not os.path.exists(p):
    print("  not found -- run scripts/eval_v1_yexc.py first")
    raise SystemExit(1)
ev = json.load(open(p))
for tag, r in sorted(ev.items()):
    print("  %-22s test_macro_f1 %-9s threshold %-8s Y_exc %s"
          % (tag, round(r["test_macro_f1"], 4), round(r["threshold"], 4),
             round(r["test_yexc_macro_f1"], 4)))
v = [r["test_macro_f1"] for r in ev.values()]
m = sum(v) / len(v)
sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
print("  mean %.4f  sd %.4f  (n=%d)" % (m, sd, len(v)))

print()
print("=" * 74)
print("  READING")
print("=" * 74)
print("""  Line up the two tables seed by seed.

  Thresholds differ, one seed far off in PART 2
      -> threshold selection is the cause. The re-tuned figures here follow the
         thesis's own stated protocol (tune tau on val, freeze for test,
         Ch. 5 sec:v2-eval) more faithfully than a tau chosen mid-training, so
         they are the ones to report -- but then v1 = 0.2026, not 0.1717, and
         Ch. 6 and SUPERVISOR-NOTE.md both need the corrected number.

  Thresholds match but test_macro_f1 differs
      -> the saved checkpoint is not the one PART 2 scored (best-val vs final
         epoch). Check whether train_jusdef.py saves at best val or at the end;
         if at the end, PART 2's reported figure came from a model state that no
         longer exists on disk, and the eval here is the reproducible one.

  Both match and only the means differ
      -> a smoke or extra seed is being averaged in somewhere.

  Either way the v1 vs v2 vs R-GCN comparison must use ONE protocol for all
  three. v2 (0.1822) and R-GCN (0.2731) come from the training-time route, so if
  the re-tuned route is adopted for v1 it has to be re-run for those two as well
  before any of the three numbers are compared.""")
