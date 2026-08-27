#!/bin/bash
# =============================================================================
# Two outstanding thesis experiments, in one job.
#
# Both close gaps that the thesis names as open. Neither needs new data.
#
# -----------------------------------------------------------------------------
# PART 1 — sentence-count stratification of the LEDGAR operating regime
# -----------------------------------------------------------------------------
# Thesis: Ch. 7 (length confound), Ch. 9 (external validity), Ch. 10 (future
# work), which call this "the single most informative outstanding run".
#
# Question: is the 10-20% operating regime a density effect or a paragraph-
# length effect? Because d_p = k/n over integers forces n >= 6 for band
# membership, the density bin also selects for long paragraphs and the two
# accounts are not separable by any experiment currently in the thesis.
#
# Mechanism: inference only, over the released test split with the existing
# ten-seed checkpoints. Stratifies by sentence count instead of density, then
# runs the 2x2 that identifies the effect — among paragraphs with n >= 6
# (all band-eligible on length grounds), in-band vs out-of-band.
#
# Success metric (printed as a VERDICT block by the analysis script):
#   in-band delta > 0 with interval excluding zero AND out-of-band delta null
#       -> operator account; C8 stands as written
#   both positive and comparable
#       -> length account; C8 must be restated as a pooling-architecture result
#
# Either outcome is reportable. This is the run that cannot fail to be
# informative, which is why it goes first and why it is cheap.
#
# -----------------------------------------------------------------------------
# PART 2 — v1 under the corrected protocol, and per-correction attribution
# -----------------------------------------------------------------------------
# Thesis: Ch. 6 opens from v2 underperforming R-GCN by nine macro-F1 points and
# calls this *anomalous* because v2 strictly extends v1. That framing presumes
# v1 was ahead of the same baseline under the same protocol — and v1's corrected
# number appears nowhere in the thesis. Ch. 5 forward-references Ch. 6 for it;
# Ch. 6 contains no v1 run.
#
# Question 2a: is there an anomaly at all? Three seeds of v1 under the corrected
# threshold protocol either establish it or dissolve it.
#   - if v1 >> v2, the anomaly is real and Ch. 6's diagnostic chain is motivated
#   - if v1 ~= v2 ~= 0.18, there is no anomaly, only a weak architecture family,
#     and Ch. 6's framing needs rewriting (a real result, not a failure)
#
# Question 2b: v1 -> v2 changes three things at once (thesis fixes F1, F2, F3a)
# and performance collapses. The falsification chain tests five *other*
# hypotheses and never asks whether one of the corrections is itself harmful.
# One single-seed run per correction closes that hole.
#
# NOTE ON NAMING: the ablation targets are named by mechanism, not by the thesis
# labels F1/F2/F3a, because scripts/run_f1.sh and run_f2.sh in this repo already
# use those names for unrelated gate experiments. The mapping is:
#   reverse_edges   = thesis fix F1  (conc,mentions_rev,sec relation)
#   authority_grad  = thesis fix F2  (authority scorer in the gradient path)
#   ontology_edges  = thesis fix F3a (conc,ontology,conc + EuroVoc label adj)
#
# Graphs: keyword-detector graphs (data/processed/graphs), because the nine-point
# gap the thesis quotes is v2-keyword 0.1822 against R-GCN 0.2731.
#
# -----------------------------------------------------------------------------
# Launch:
#   cd ~/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   SMOKE=1 bash scripts/run_thesis_gaps.sh          # wiring check, ~15 min
#   sbatch run_thesis_gaps.slurm                     # the real thing
#
# Run one part only:
#   PARTS=1 bash scripts/run_thesis_gaps.sh
#   PARTS=2 bash scripts/run_thesis_gaps.sh
#
# Outputs:
#   outputs/logs/ledgar_length_stratified.json
#   outputs/logs/jusdef_v1_corrected_s{42,43,44}.json
#   outputs/logs/jusdef_abl_{reverse_edges,authority_grad,ontology_edges}_s42.json
#   outputs/logs/run_thesis_gaps_summary.txt
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CKPT=outputs/checkpoints
LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p "$CKPT" "$LOG" "$SENT"

PARTS="${PARTS:-1 2}"
LEDGAR_SEEDS="${LEDGAR_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
V1_SEEDS="${V1_SEEDS:-42 43 44}"
ABL_SEEDS="${ABL_SEEDS:-42}"
GRAPHS="${GRAPHS:-data/processed/graphs}"

EXTRA=""; SFX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="--max_train 2000 --epochs 3 --patience 2 --stage1_end 1 --stage2_end 2"
    SFX="_smoke"
    V1_SEEDS="42"
    ABL_SEEDS="42"
    LEDGAR_SEEDS="42 43"
    echo "*** SMOKE MODE (throwaway, tag suffix '$SFX') ***"
fi

SUMMARY="$LOG/run_thesis_gaps_summary${SFX}.txt"
: > "$SUMMARY"

say() { echo "$@" | tee -a "$SUMMARY"; }

say "=============================================================="
say " run_thesis_gaps.sh   started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say " parts: $PARTS   graphs: $GRAPHS"
say "=============================================================="

# -----------------------------------------------------------------------------
# PART 1 — length stratification (inference only)
# -----------------------------------------------------------------------------
if [[ " $PARTS " == *" 1 "* ]]; then
    say ""
    say "--- PART 1: sentence-count stratification of the 10-20% regime ---"
    s="$SENT/length_stratified${SFX}.done"
    if [[ -f "$s" ]]; then
        say "  sentinel present, skipping"
    else
        python scripts/analyse_ledgar_length_stratified.py \
            --seeds $LEDGAR_SEEDS 2>&1 | tee -a "$SUMMARY"
        touch "$s"
    fi
fi

# -----------------------------------------------------------------------------
# PART 2 — v1 under the corrected protocol, then per-correction ablations
# -----------------------------------------------------------------------------
if [[ " $PARTS " == *" 2 "* ]]; then
    say ""
    say "--- PART 2a: v1 (all three corrections off) under the corrected protocol ---"
    for seed in $V1_SEEDS; do
        tag="v1_corrected${SFX}"
        s="$SENT/${tag}_s${seed}.done"
        if [[ -f "$s" ]]; then say "  seed $seed sentinel present, skipping"; continue; fi
        say "  training $tag seed $seed"
        python scripts/train_jusdef.py \
            --seed "$seed" --tag "$tag" \
            --graph_dir "$GRAPHS" \
            --dmp_variant hard \
            --ablate all \
            $EXTRA 2>&1 | tee -a "$SUMMARY"
        touch "$s"
    done

    say ""
    say "--- PART 2b: one correction removed at a time (v2 minus X) ---"
    for abl in reverse_edges authority_grad ontology_edges; do
        for seed in $ABL_SEEDS; do
            tag="abl_${abl}${SFX}"
            s="$SENT/${tag}_s${seed}.done"
            if [[ -f "$s" ]]; then say "  $abl seed $seed sentinel present, skipping"; continue; fi
            say "  training $tag seed $seed"
            python scripts/train_jusdef.py \
                --seed "$seed" --tag "$tag" \
                --graph_dir "$GRAPHS" \
                --dmp_variant hard \
                --ablate "$abl" \
                $EXTRA 2>&1 | tee -a "$SUMMARY"
            touch "$s"
        done
    done

    # -------------------------------------------------------------------------
    # Collate: put the new numbers next to the ones already in the thesis.
    # -------------------------------------------------------------------------
    say ""
    say "--- PART 2c: collated comparison ---"
    python - "$SFX" <<'PY' 2>&1 | tee -a "$SUMMARY"
import json, sys, math, glob, os
sfx = sys.argv[1] if len(sys.argv) > 1 else ""

def load(pat):
    out = {}
    for p in sorted(glob.glob(pat)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        s = d.get("config", {}).get("seed")
        v = d.get("test_macro_f1")
        if s is not None and v is not None:
            out[s] = v
    return out

def stat(d):
    v = list(d.values())
    if not v: return None, None, 0
    m = sum(v)/len(v)
    sd = math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)) if len(v) > 1 else None
    return m, sd, len(v)

rows = [
    ("v1 (all corrections off)", f"outputs/logs/jusdef_v1_corrected{sfx}_s*.json"),
    ("v2 minus reverse_edges",   f"outputs/logs/jusdef_abl_reverse_edges{sfx}_s*.json"),
    ("v2 minus authority_grad",  f"outputs/logs/jusdef_abl_authority_grad{sfx}_s*.json"),
    ("v2 minus ontology_edges",  f"outputs/logs/jusdef_abl_ontology_edges{sfx}_s*.json"),
    ("v2 full (existing)",       "outputs/logs/jusdef_full_s*.json"),
]
RGCN = 0.2731  # Table 7.1, three-seed mean, h=512
V2   = 0.1822  # Table 7.3, three-seed mean, keyword operators

print()
print("%-28s %8s %8s %6s %12s %12s" % ("variant", "mean", "sd", "n", "d vs R-GCN", "d vs v2"))
print("-" * 82)
for name, pat in rows:
    m, sd, n = stat(load(pat))
    if m is None:
        print("%-28s %8s" % (name, "(no runs)")); continue
    print("%-28s %8.4f %8s %6d %+12.4f %+12.4f"
          % (name, m, ("%.4f" % sd) if sd else "  --  ", n, m - RGCN, m - V2))
print("-" * 82)
print("reference: R-GCN h=512 = %.4f, v2 keyword = %.4f (thesis Tables 7.1, 7.3)" % (RGCN, V2))
print()
m, _, n = stat(load(rows[0][1]))
if m is not None and n:
    gap = RGCN - m
    print("READING FOR CHAPTER 6:")
    if m > V2 + 0.03:
        print("  v1 (%.4f) is well ahead of v2 (%.4f). The anomaly is real: the" % (m, V2))
        print("  corrections cost %.3f macro-F1 and Ch. 6's framing is motivated." % (m - V2))
        print("  Next question is which correction — see the per-correction rows.")
    elif abs(m - V2) <= 0.03:
        print("  v1 (%.4f) is level with v2 (%.4f). There is NO anomaly to explain:" % (m, V2))
        print("  the architecture family sits around 0.18 with or without the")
        print("  corrections, and Ch. 6 should be reframed from 'the corrections")
        print("  broke it' to 'the family never worked on EUR-Lex'. This is a")
        print("  reportable result, not a failed run.")
    else:
        print("  v1 (%.4f) is BELOW v2 (%.4f): the corrections helped, and the" % (m, V2))
        print("  nine-point gap to R-GCN predates them entirely.")
    print("  v1 trails R-GCN by %.4f." % gap)
PY
fi

say ""
say "=============================================================="
say " finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say " summary: $SUMMARY"
say "=============================================================="
