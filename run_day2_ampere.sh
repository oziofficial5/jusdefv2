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
# abbreviation guard. n is the denominator of d_p, so error here propagates
# straight into the density-stratified protocol -- and differentially, because
# EUR-Lex enumerations are semicolon-heavy in a way contract clauses are not.
import re, statistics, collections

def lex(cfg, split):
    """trust_remote_code is required by some datasets versions and rejected by
    others; try both rather than fail unattended."""
    from datasets import load_dataset
    try:
        return load_dataset("coastalcph/lex_glue", cfg, split=split, trust_remote_code=True)
    except TypeError:
        return load_dataset("coastalcph/lex_glue", cfg, split=split)

DEPLOYED = re.compile(r"(?<=[.;])\s+")
PERIOD   = re.compile(r"(?<=[.])\s+")
ABBREV = re.compile(r"(?:art|arts|no|nos|para|paras|pp|cf|eg|e\.g|ie|i\.e|reg|dir|"
                    r"sec|ch|vol|fig|al|etc|approx|ibid)\.$", re.I)

def guarded(text):
    parts, out = PERIOD.split(text), []
    for q in parts:
        if out and (ABBREV.search(out[-1].strip()) or re.search(r"\b\d+\.$", out[-1].strip())):
            out[-1] = out[-1] + " " + q
        else:
            out.append(q)
    return out

for cfg, split, cap in (("eurlex", "test", 5000), ("ledgar", "test", 10000)):
    print("\n=== %s / %s ===" % (cfg, split), flush=True)
    try:
        ds = lex(cfg, split)
    except Exception as e:
        print("  could not load: %s" % e); continue
    texts = [t for t in ds["text"][:cap] if t and t.strip()]
    dep = [len(DEPLOYED.split(t)) for t in texts]
    per = [len(PERIOD.split(t)) for t in texts]
    gua = [len(guarded(t)) for t in texts]
    md, mp, mg = statistics.mean(dep), statistics.mean(per), statistics.mean(gua)
    print("  documents/paragraphs        : %d" % len(texts))
    print("  total segments, deployed    : %d" % sum(dep))
    print("  mean segments, deployed     : %.4f" % md)
    print("  mean segments, period only  : %.4f" % mp)
    print("  mean segments, period+guard : %.4f" % mg)
    print("  --> semicolons inflate n by  %+.1f%%" % (100.0 * (md / mp - 1)))
    print("  --> abbreviations inflate by %+.1f%%" % (100.0 * (mp / mg - 1)))
    print("  --> d_p = k/n deflated by    %.1f%% overall" % (100.0 * (1 - mg / md)))
    if cfg == "ledgar":
        c = collections.Counter(dep)
        one = 100.0 * c[1] / len(dep)
        print("  single-segment paragraphs   : %d (%.1f%%)   [thesis: 4,415 = 44.2%%]" % (c[1], one))
        print("  thesis Table 3.x claims 22,615 sentences over 10,000 paragraphs, mean 2.26")
        print("  measured here: %d segments, mean %.4f" % (sum(dep), md))
        elig = sum(1 for n in dep if n >= 6)
        print("  n >= 6 (10-20%% band eligible): %d   [thesis: 523]" % elig)
print("""
The number that matters is the DIFFERENCE between the two corpora. d_p is
compared across them, so a segmenter that inflates n more on one than the other
biases the comparison itself, independently of how it treats either alone.""")
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
# the quantity deciding whether EUR-Lex 0.71% is a measurement or the detector's
# own false-positive floor.
#
# Sampling here is per SENTENCE, not per mention edge, because the processed
# pkls are stubs and carry no concept spans. That is deliberate: 3.6.1 flags
# re-expressing EUR-Lex per-sentence as an outstanding item, so this closes it
# as well. Two strata with known weights make both error directions estimable.
import re, io, os, random, collections, math, json
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

def lex(cfg, split):
    """trust_remote_code is required by some datasets versions and rejected by
    others; try both rather than fail unattended."""
    from datasets import load_dataset
    try:
        return load_dataset("coastalcph/lex_glue", cfg, split=split, trust_remote_code=True)
    except TypeError:
        return load_dataset("coastalcph/lex_glue", cfg, split=split)

random.seed(20260904)
N_AFF, N_NONAFF, POOL = 250, 150, 60000
CKPT = "outputs/checkpoints/operator_detector_neural.pt"
SENT = re.compile(r"(?<=[.;])\s+")

class OperatorClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))

INT2 = {0: "AFF", 1: "NEG", 2: "EXC", 3: "OVR"}

if not os.path.exists(CKPT):
    print("  detector checkpoint missing at %s -- cannot sample." % CKPT); raise SystemExit(1)
state = torch.load(CKPT, map_location="cpu")
backbone = state["config"]["backbone"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device %s | backbone %s | detector val_macro_f1 %s"
      % (device, backbone, state.get("val_macro_f1", "n/a")), flush=True)
tok = AutoTokenizer.from_pretrained(backbone)
model = OperatorClassifier(backbone).to(device)
model.load_state_dict(state["state_dict"])
model.eval()

ds = lex("eurlex", "test")
sents = []
for t in ds["text"]:
    if t and t.strip():
        for x in SENT.split(t):
            x = x.strip()
            if 20 <= len(x) <= 1200:
                sents.append(x)
print("EUR-Lex test: %d documents -> %d sentences" % (len(ds), len(sents)), flush=True)

random.shuffle(sents)
pool = sents[:POOL]
print("scoring a uniform random pool of %d sentences ..." % len(pool), flush=True)

@torch.no_grad()
def predict(batch):
    enc = tok(batch, truncation=True, padding="max_length", max_length=128,
              return_tensors="pt").to(device)
    return model(enc["input_ids"], enc["attention_mask"]).argmax(-1).tolist()

preds = []
for i in range(0, len(pool), 128):
    preds.extend(predict(pool[i:i + 128]))
    if i % 12800 == 0:
        print("  %d/%d" % (i, len(pool)), flush=True)
labels = [INT2[p] for p in preds]

dist = collections.Counter(labels)
n = len(labels)
k = sum(v for lab, v in dist.items() if lab != "AFF")
phat = k / float(n)
z = 1.96
den = 1 + z * z / n
cen = (phat + z * z / (2 * n)) / den
half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
print("\n=== per-sentence non-AFF density on a uniform random sample ===")
print("  distribution : %s" % dict(dist))
print("  non-AFF      : %d/%d = %.4f%%" % (k, n, 100 * phat))
print("  95%% Wilson   : [%.4f%%, %.4f%%]" % (100 * (cen - half), 100 * (cen + half)))
print("  thesis reports 0.71%% over MENTION EDGES; this is the per-sentence")
print("  figure 3.6.1 lists as an outstanding recomputation.")

aff = [s for s, l in zip(pool, labels) if l == "AFF"]
non = [(s, l) for s, l in zip(pool, labels) if l != "AFF"]
sa = random.sample(aff, min(N_AFF, len(aff)))
sb = random.sample(non, min(N_NONAFF, len(non)))
print("\n  stratum A (predicted AFF)     : %d of %d" % (len(sa), len(aff)))
print("  stratum B (predicted non-AFF) : %d of %d" % (len(sb), len(non)))

rows = [("A_pred_AFF", "AFF", x) for x in sa] + [("B_pred_nonAFF", l, x) for x, l in sb]
random.shuffle(rows)
os.makedirs("outputs/day2", exist_ok=True)
with io.open("outputs/day2/f1_eurlex_validation_blind.tsv", "w", encoding="utf-8") as f:
    f.write("idx\thuman_op\tsentence\n")
    for i, (st, l, x) in enumerate(rows, 1):
        f.write("%d\t\t%s\n" % (i, x.replace("\t", " ")))
with io.open("outputs/day2/f1_eurlex_validation_KEY.tsv", "w", encoding="utf-8") as f:
    f.write("idx\tstratum\tdetector_op\tsentence\n")
    for i, (st, l, x) in enumerate(rows, 1):
        f.write("%d\t%s\t%s\t%s\n" % (i, st, l, x.replace("\t", " ")))
json.dump({"pool_scored": n, "pred_AFF_population": len(aff),
           "pred_nonAFF_population": len(non), "n_sampled_AFF": len(sa),
           "n_sampled_nonAFF": len(sb), "pointwise_nonaff_rate": phat},
          open("outputs/day2/f1_weights.json", "w"), indent=2)
print("""
  blind file : outputs/day2/f1_eurlex_validation_blind.tsv
  key        : outputs/day2/f1_eurlex_validation_KEY.tsv  (do not open first)
  weights    : outputs/day2/f1_weights.json

NEXT: annotate human_op in the blind file against the guidelines (Appendix A),
without the key. The two strata then reweight to a natural-prior estimate of the
true non-AFF rate with an interval -- which is what F1 asks for, and which the
prediction-stratified LEDGAR design cannot give. Expected direction favours the
thesis: a lower true floor strengthens every negative EUR-Lex result.""")
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
