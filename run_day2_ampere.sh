#!/usr/bin/env bash
# =============================================================================
#  run_day2_ampere.sh  --  Chapter 3 revision, Day 2
# =============================================================================
#
#  USAGE (on Ampere):
#
#      cd ~/jusdefv2
#      tmux new -s day2
#      bash run_day2_ampere.sh 2>&1 | tee logs/day2_$(date +%Y%m%d_%H%M).log
#      #  detach with  ctrl-b d      reattach with  tmux attach -t day2
#
#  Stages 1-4 are diagnostics and finish in minutes. Stage 5 is inference over
#  the v1 checkpoints run_thesis_gaps.sh PART 2 already produced -- it does NOT
#  retrain. Every stage writes a sentinel to outputs/sentinels/ and is skipped on
#  a re-run, so you can safely re-invoke after fixing any single stage.
#
#  Set SKIP_TRAIN=1 to run only the diagnostics:
#      SKIP_TRAIN=1 bash run_day2_ampere.sh
# =============================================================================

set -uo pipefail

PY="${PY:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SENT="outputs/sentinels"
OUT="outputs/day2"
mkdir -p "$SENT" "$OUT" logs outputs/logs

hr()  { printf '\n%s\n' "-------------------------------------------------------------------------------"; }
say() { hr; printf '  %s\n' "$*"; hr; }
done_already() { [ -f "$SENT/$1.done" ]; }
mark() { touch "$SENT/$1.done"; }

say "Day 2 starting -- $(date)"

# =============================================================================
# STAGE 0  --  inventory
# =============================================================================
say "STAGE 0  Inventory"

$PY - <<'PYEOF'
import os, glob
def show(p, what):
    if os.path.isdir(p):
        n = len(os.listdir(p)); print("  OK      %-42s %d entries" % (what, n))
    elif os.path.isfile(p):
        print("  OK      %-42s %.1f MB" % (what, os.path.getsize(p)/1e6))
    else:
        print("  MISSING %-42s <-- needed" % what)

show("data/processed/graphs/train_graphs.pt",        "keyword graphs (train)")
show("data/processed/graphs/validation_graphs.pt",   "keyword graphs (validation)")
show("data/processed/graphs/test_graphs.pt",         "keyword graphs (test)")
show("data/processed/embeddings",                    "embeddings")
show("data/processed/label_adj.pt",                  "label adjacency")
show("outputs/checkpoints/operator_detector_neural.pt", "neural operator detector")
show("ledgar_detector_sample.tsv",                   "LEDGAR 160-sentence sample")
for split in ("train", "validation", "test"):
    p = "data/processed/%s_processed.pkl" % split
    if os.path.isfile(p):
        import pickle
        d = pickle.load(open(p, "rb"))
        print("  OK      %-42s %d docs" % ("%s_processed.pkl" % split, len(d)))
    else:
        print("  MISSING %-42s <-- needed" % ("%s_processed.pkl" % split))
PYEOF

# =============================================================================
# STAGE 1  --  F7 : sentence segmentation audit
# =============================================================================
if done_already day2_f7; then say "STAGE 1  F7 segmentation  [skipped, sentinel present]"; else
say "STAGE 1  F7  Sentence segmentation audit"

$PY - <<'PYEOF' | tee outputs/day2/f7_segmentation.txt
# The deployed segmenter is  re.split(r"(?<=[.;])\s+", text)  from
# src/preprocess/operator_detector.py:62. It splits on semicolons and has no
# abbreviation guard. n is the denominator of d_p, so any systematic error here
# propagates straight into the density-stratified protocol.
import pickle, re, os, statistics

DEPLOYED = re.compile(r"(?<=[.;])\s+")
PERIOD   = re.compile(r"(?<=[.])\s+")
ABBREV = re.compile(r"(?:art|arts|no|nos|para|paras|pp|cf|eg|e\.g|ie|i\.e|reg|dir|"
                    r"sec|ch|vol|fig|al|etc|approx|ibid)\.$", re.I)

def guarded(text):
    """Period split, then re-join a fragment whose predecessor ended in a known
    abbreviation or a bare numeral (e.g. 'Article 5.' inside a citation)."""
    parts = PERIOD.split(text)
    out = []
    for p in parts:
        if out and (ABBREV.search(out[-1].strip()) or re.search(r"\b\d+\.$", out[-1].strip())):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out

for split in ("test", "validation", "train"):
    path = "data/processed/%s_processed.pkl" % split
    if not os.path.isfile(path):
        continue
    docs = pickle.load(open(path, "rb"))
    dep, per, gua, semis, shortfrag, lower_start = [], [], [], 0, 0, 0
    nsec = 0
    for d in docs[:2000]:
        for s in d.get("sections", []):
            t = s.get("text") or ""
            if not t.strip():
                continue
            nsec += 1
            a = DEPLOYED.split(t); b = PERIOD.split(t); c = guarded(t)
            dep.append(len(a)); per.append(len(b)); gua.append(len(c))
            semis += len(a) - len(b)
            for frag in a:
                f = frag.strip()
                if f and len(f) < 25:
                    shortfrag += 1
                if f and f[0].islower():
                    lower_start += 1
    if not nsec:
        continue
    periods = sum(1 for d in docs[:2000] for s in d.get("sections", [])
                  if "." in (s.get("text") or ""))
    print("\n=== %s split : %d sections sampled ===" % (split, nsec))
    if periods < nsec * 0.5 or statistics.mean(dep) < 1.5:
        print("  *** INPUT LOOKS TRUNCATED: only %d of %d sections contain a period," % (periods, nsec))
        print("  *** and the mean segment count is %.2f. The local dev checkout stores" % statistics.mean(dep))
        print("  *** section text clipped to 500 characters; the figures below are")
        print("  *** meaningless on such a copy. Re-run against the full pipeline output.")
        continue
    print("  mean segments/section, deployed (split on . and ;) : %.3f" % statistics.mean(dep))
    print("  mean segments/section, period only                 : %.3f" % statistics.mean(per))
    print("  mean segments/section, period + abbreviation guard  : %.3f" % statistics.mean(gua))
    infl_semi = 100.0 * (statistics.mean(dep) / statistics.mean(per) - 1)
    infl_abbr = 100.0 * (statistics.mean(per) / statistics.mean(gua) - 1)
    print("  --> semicolon splitting inflates n by %+.1f%%" % infl_semi)
    print("  --> unguarded abbreviations inflate n by a further %+.1f%%" % infl_abbr)
    print("  --> d_p = k/n is therefore DEFLATED by roughly %.1f%% overall" %
          (100.0 * (1 - statistics.mean(gua) / statistics.mean(dep))))
    print("  segments shorter than 25 chars : %d (%.1f%% of all segments)"
          % (shortfrag, 100.0 * shortfrag / max(1, sum(dep))))
    print("  segments starting lower-case   : %d (%.1f%%)  <- likely spurious splits"
          % (lower_start, 100.0 * lower_start / max(1, sum(dep))))
PYEOF
mark day2_f7; fi

# =============================================================================
# STAGE 2  --  F5 : detector baselines on the 160 blind LEDGAR sentences
# =============================================================================
if done_already day2_f5; then say "STAGE 2  F5 detector baselines  [skipped]"; else
say "STAGE 2  F5  Keyword vs neural detector against human LEDGAR labels"

$PY - <<'PYEOF' | tee outputs/day2/f5_baselines.txt
import io, os, sys, math, collections
sys.path.insert(0, os.getcwd())
from src.preprocess.operator_detector import detect_operator

PATH = "ledgar_detector_sample.tsv"
rows = [l.rstrip("\n").split("\t") for l in io.open(PATH, encoding="utf-8")]
hdr = [h.strip() for h in rows[0]]
# Resolve by header name: the file gained a keyword_op column when the recovered
# human annotations were installed, so fixed indices are not safe.
try:
    I_DET = hdr.index("detector_op")
    I_HUM = hdr.index("human_op")
    I_SENT = hdr.index("sentence")
except ValueError:
    print("  *** unexpected header: %s" % hdr)
    print("  *** expected columns detector_op, human_op, sentence")
    raise SystemExit(1)
need = max(I_DET, I_HUM, I_SENT) + 1
data = [r for r in rows[1:] if len(r) >= need]
print("loaded %d sentences from %s" % (len(data), PATH))
print("columns: %s" % hdr)

human = [r[I_HUM].strip().upper() for r in data]
neural = [r[I_DET].strip().upper() for r in data]
sents = [r[I_SENT] for r in data]
keyword = [detect_operator(s) for s in sents]

if sents and all(len(x.strip()) <= 4 for x in sents[:10]):
    print("  *** the sentence column looks like labels, not text -- column")
    print("  *** resolution failed. Aborting rather than reporting nonsense.")
    raise SystemExit(1)

io.open("outputs/day2/f5_keyword_predictions.tsv", "w", encoding="utf-8").write(
    "idx\tdetector_op\tkeyword_op\thuman_op\tsentence\n" +
    "".join("%s\t%s\t%s\t%s\t%s\n" % (r[0], n, k, h, s)
            for r, n, k, h, s in zip(data, neural, keyword, human, sents)))
print("keyword predictions written to outputs/day2/f5_keyword_predictions.tsv")
print("keyword distribution : %s" % dict(collections.Counter(keyword)))
print("neural  distribution : %s" % dict(collections.Counter(neural)))

filled = sum(1 for h in human if h)
if filled < len(data):
    print("\n  *** human_op is filled on only %d of %d rows. ***" % (filled, len(data)))
    print("  The annotations behind kappa = 0.867 are NOT in this checkout.")
    print("  Stage 4 searches the filesystem for them. Once found, drop the file in")
    print("  as ledgar_detector_sample.tsv (with human_op populated) and re-run:")
    print("      rm outputs/sentinels/day2_f5.done && bash run_day2_ampere.sh")
    raise SystemExit(0)

def kappa(pairs):
    n = len(pairs); po = sum(1 for a, b in pairs if a == b) / float(n)
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / float(n * n)
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return po, k, se

def report(name, pred):
    po, k, se = kappa(list(zip(pred, human)))
    print("\n=== %s vs human (N=%d) ===" % (name, len(human)))
    print("  accuracy %.4f   kappa %.4f   95%% CI [%.3f, %.3f]"
          % (po, k, k - 1.96 * se, k + 1.96 * se))
    for op in ("AFF", "NEG", "EXC", "OVR"):
        tp = sum(1 for p, h in zip(pred, human) if p == op and h == op)
        fp = sum(1 for p, h in zip(pred, human) if p == op and h != op)
        fn = sum(1 for p, h in zip(pred, human) if p != op and h == op)
        P = tp / float(tp + fp) if tp + fp else 0.0
        R = tp / float(tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print("  %-4s P %.3f  R %.3f  F1 %.3f" % (op, P, R, F))

report("NEURAL detector", neural)
report("KEYWORD detector", keyword)
print("\nC4 claims the neural detector improves on the keyword rule. The two rows")
print("above are the first direct evidence either way, on a common human reference.")
PYEOF
mark day2_f5; fi

# =============================================================================
# STAGE 3  --  F1 : natural-prior validation sample for blind annotation
# =============================================================================
if done_already day2_f1; then say "STAGE 3  F1 validation sample  [skipped]"; else
say "STAGE 3  F1  Build natural-prior detector-validation sample"

$PY - <<'PYEOF' | tee outputs/day2/f1_sample_build.txt
# The LEDGAR validation is stratified BY DETECTOR PREDICTION, which fixes the
# predicted marginals and so cannot estimate P(predicted non-AFF | true AFF) --
# the quantity that decides whether EUR-Lex 0.71% is a measurement or the
# detector's own false-positive floor.
#
# This draws a two-stratum sample WITH KNOWN WEIGHTS so both directions are
# estimable and can be reweighted to a natural-prior figure with an interval:
#   stratum A: 250 edges the detector calls AFF      -> bounds missed density
#   stratum B: 150 edges the detector calls non-AFF  -> bounds precision
import pickle, re, os, random, io, collections

random.seed(20260904)
N_AFF, N_NONAFF = 250, 150
SPLIT = "test"

path = "data/processed/%s_processed.pkl" % SPLIT
docs = pickle.load(open(path, "rb"))
print("loaded %d docs from %s" % (len(docs), path))

# The LexGLUE EUR-Lex test split holds 5,000 documents. Anything far short of
# that is the truncated dev stub, whose section text is clipped to 500 chars --
# a sample drawn from it is not a sample of the test split and must not be
# annotated. The full graphs can be present while these pkls are stubs.
if len(docs) < 1000:
    print("""
  *** ABORTING: %d documents, expected ~5,000.
  *** This is the truncated dev copy, not the full processed test split, so any
  *** sample drawn here would misrepresent the corpus. Stage 1 hits the same
  *** wall for the same reason.
  ***
  *** A real text source is needed. Likely candidates on this machine:
  ***     data/raw/                        (gitignored; the preprocessing input)
  ***     the LexGLUE eurlex split via datasets.load_dataset("coastalcph/lex_glue","eurlex")
  *** Re-run stages 1 and 3 with --source pointed at whichever exists:
  ***     rm outputs/sentinels/day2_f7.done outputs/sentinels/day2_f1.done
  """ % len(docs))
    raise SystemExit(0)

SENT = re.compile(r"(?<=[.;])\s+")

def governing(text, start):
    """Same rule as detect_operators_in_section, so the sampled unit is exactly
    the unit the deployed detector scores."""
    cc = 0
    for s in SENT.split(text):
        cc += len(s) + 1
        if cc >= start:
            return s
    return text

edges = []
for di, d in enumerate(docs):
    for si, s in enumerate(d.get("sections", [])):
        t = s.get("text") or ""
        for ci, c in enumerate(s.get("concepts", []) or []):
            op = (c.get("operator") or "AFF").upper()
            edges.append((di, si, ci, op, int(c.get("span_start") or 0), c.get("phrase", "")))

print("total mention edges in %s split : %d" % (SPLIT, len(edges)))
dist = collections.Counter(e[3] for e in edges)
tot = float(len(edges))
nonaff = sum(v for k, v in dist.items() if k != "AFF")
print("operator distribution: %s" % dict(dist))
print("non-AFF density on this split : %.4f%%  (%d edges)" % (100.0 * nonaff / tot, nonaff))

pool_aff = [e for e in edges if e[3] == "AFF"]
pool_non = [e for e in edges if e[3] != "AFF"]
sa = random.sample(pool_aff, min(N_AFF, len(pool_aff)))
sb = random.sample(pool_non, min(N_NONAFF, len(pool_non)))
print("stratum A (predicted AFF)     : %d sampled from %d" % (len(sa), len(pool_aff)))
print("stratum B (predicted non-AFF) : %d sampled from %d" % (len(sb), len(pool_non)))

rows = []
for stratum, sample in (("A_pred_AFF", sa), ("B_pred_nonAFF", sb)):
    for (di, si, ci, op, start, phrase) in sample:
        text = docs[di]["sections"][si].get("text") or ""
        rows.append((stratum, op, phrase, governing(text, start).replace("\t", " ").strip()))
random.shuffle(rows)

out = "outputs/day2/f1_eurlex_validation_blind.tsv"
with io.open(out, "w", encoding="utf-8") as f:
    f.write("idx\thuman_op\tconcept\tsentence\n")
    for i, (st, op, phrase, sent) in enumerate(rows, 1):
        f.write("%d\t\t%s\t%s\n" % (i, phrase, sent))
print("\nBLIND annotation file (no labels, shuffled): %s" % out)

key = "outputs/day2/f1_eurlex_validation_KEY.tsv"
with io.open(key, "w", encoding="utf-8") as f:
    f.write("idx\tstratum\tdetector_op\tconcept\tsentence\n")
    for i, (st, op, phrase, sent) in enumerate(rows, 1):
        f.write("%d\t%s\t%s\t%s\t%s\n" % (i, st, op, phrase, sent))
print("KEY (detector labels, do NOT open before annotating): %s" % key)

with io.open("outputs/day2/f1_weights.txt", "w", encoding="utf-8") as f:
    f.write("N_pred_AFF_population=%d\nN_pred_nonAFF_population=%d\n"
            "n_pred_AFF_sampled=%d\nn_pred_nonAFF_sampled=%d\n"
            % (len(pool_aff), len(pool_non), len(sa), len(sb)))
print("reweighting weights: outputs/day2/f1_weights.txt")
print("""
NEXT: annotate the blind file's human_op column against the guidelines
(Appendix A), WITHOUT looking at the KEY. Then the two strata reweight to a
natural-prior estimate of the true non-AFF density with a Wilson interval,
which is what F1 asks for. Expected direction of the result FAVOURS the thesis:
a lower true floor strengthens every negative EUR-Lex result.""")
PYEOF
mark day2_f1; fi

# =============================================================================
# STAGE 4  --  F11 : locate the LEDGAR validation artefacts
# =============================================================================
if done_already day2_f11; then say "STAGE 4  F11 artefact hunt  [skipped]"; else
say "STAGE 4  F11  Locate the LEDGAR cross-genre validation (kappa = 0.867)"

{
  echo "Searching for the completed 160-sentence LEDGAR annotation..."
  echo "kappa = 0.867 carries 'cross-genre-validated' in the abstract and is the"
  echo "only headline agreement statistic with no reproducible artefact."
  echo
  ROOTS="$ROOT $HOME/jusdefv2 $HOME/Downloads $HOME/annotations $HOME/data"
  for R in $ROOTS; do
    [ -d "$R" ] || continue
    timeout 60 find "$R" -maxdepth 4 \( -iname "*ledgar*detector*" -o -iname "*ledgar*sample*" \
         -o -iname "*ledgar*annot*" -o -iname "*ledgar*iaa*" -o -iname "*cross_genre*" \) \
         -not -path "*/.git/*" 2>/dev/null
  done | sort -u | head -40
  echo
  echo "--- files whose contents mention 0.867 ---"
  for R in $ROOTS; do
    [ -d "$R" ] || continue
    timeout 60 grep -rl "0\.867" "$R" --include=*.json --include=*.tsv --include=*.csv \
         --include=*.txt --include=*.md 2>/dev/null
  done | sort -u | head -20
} | tee outputs/day2/f11_artefact_hunt.txt
mark day2_f11; fi

# =============================================================================
# STAGE 5  --  Y_exc on the v1 checkpoints
# =============================================================================
say "STAGE 5  Y_exc evaluation of v1"

# NOTE: v1 is already trained. scripts/run_thesis_gaps.sh PART 2 ran three seeds
# under --ablate all plus the three single-correction ablations (~16 h), giving
#     v1 0.1717 +/- 0.0609 | v2 0.1822 +/- 0.0163 | R-GCN 0.2731
# which dissolves Chapter 6's anomaly. Do NOT retrain that here.
#
# What PART 2 left open: it measured OVERALL macro-F1, but the v1 workshop
# paper's claim was +6.0 on Y_exc specifically. That is inference over existing
# checkpoints -- minutes, not hours.

V1_LOGS=$(ls outputs/logs/jusdef_v1_corrected*_s*.json 2>/dev/null | wc -l)
V1_CKPTS=$(ls outputs/checkpoints/jusdef_v1_corrected*_s*.pt 2>/dev/null | wc -l)
echo "  v1 training logs found : $V1_LOGS"
echo "  v1 checkpoints found   : $V1_CKPTS"

if [ "$V1_LOGS" -eq 0 ]; then
  echo
  echo "  No v1 results present. PART 2 has not run in this checkout:"
  echo "      PARTS=2 bash scripts/run_thesis_gaps.sh 2>&1 | tee outputs/logs/gaps_part2.log"
  echo "  That is the ~16 h job. Run it before this stage."
  exit 1
fi

if [ "$V1_CKPTS" -eq 0 ]; then
  echo
  echo "  v1 logs exist but the checkpoints do not, so Y_exc cannot be computed"
  echo "  without retraining. If the checkpoints were cleaned up, re-run PART 2."
  exit 1
fi

if done_already day2_yexc; then
  say "  Y_exc already computed, skipping"
else
  $PY scripts/eval_v1_yexc.py 2>&1 | tee outputs/day2/v1_yexc.txt
  if [ -f outputs/logs/v1_yexc.json ]; then mark day2_yexc; fi
fi

say "Summary"
$PY - <<'PYEOF'
import json, glob, os

def grab(d):
    for k in ("test_macro_f1", "test_macro_F1", "macro_f1", "test_f1"):
        if isinstance(d, dict) and k in d:
            return d[k]
    for v in (d.values() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            r = grab(v)
            if r is not None:
                return r
    return None

print("  overall test macro-F1, for reference:")
for pat, name in (("outputs/logs/baseline_rgcn_seed*.json",    "R-GCN"),
                  ("outputs/logs/jusdef_full_s*.json",         "v2"),
                  ("outputs/logs/jusdef_v1_corrected*_s*.json", "v1")):
    vals = []
    for f in sorted(glob.glob(pat)):
        try:
            v = grab(json.load(open(f)))
        except Exception:
            v = None
        if v is not None:
            vals.append(v)
    if vals:
        m = sum(vals) / len(vals)
        print("    %-8s %.4f  (n=%d)" % (name, m, len(vals)))

p = "outputs/logs/v1_yexc.json"
if os.path.exists(p):
    d = json.load(open(p))
    ys = [r["test_yexc_macro_f1"] for r in d.values()]
    print("\n  v1 Y_exc macro-F1: %.4f over %d seeds" % (sum(ys) / len(ys), len(ys)))
    print("""
  The workshop paper claimed +6.0 over R-GCN on Y_exc. Compare the figure above
  against R-GCN's Y_exc under the same protocol (scripts/eval_rgcn_detailed.py
  computes it) to settle whether that claim survives the corrected protocol.
  Either answer is reportable; below R-GCN is the cleaner one for the thesis,
  because it makes the negative EUR-Lex result uniform across metrics.""")
PYEOF

say "Day 2 complete -- $(date)"
