#!/bin/bash
# =============================================================================
#  STAGE 1b — the length/density experiment, done correctly this time.
# =============================================================================
#
#  WHY THIS EXISTS: the first attempt was invalid, and the fault is in the run
#  file, not in the architecture.
#
#  preprocess_cuad_whole.py line 64 does this:
#
#       4. Discard blocks with < min_sentences (avoids headings, signature
#          lines).
#
#  --min_sentences DISCARDS. It does not re-segment. So arm A at
#  --min_sentences 6 threw away every block under six sentences, and arm B at
#  --min_sentences 10 threw away everything under ten. What survived was a
#  fraction of the corpus:
#
#       CUAD-whole (2-12)   1,222 test paragraphs
#       arm A      (6-7)      546 test  (45%)
#       arm B      (10-20)    317 test  (26%),  1,086 train
#
#  A 41-class task trained on ~1,000 paragraphs is roughly 26 examples per
#  class. Both arms landed at 0.10-0.12 test macro-F1, against CUAD-whole's
#  0.33 and LEDGAR's 0.70. At that level there is no headroom in which an
#  aggregation difference could show itself, so the nulls measure data
#  starvation, not the absence of a regime.
#
#  Arm B also failed the power gate written into the run file: its 10-20% bin
#  held 81 paragraphs against the stated minimum of ~100. It should have been
#  abandoned rather than trained. Arm A passed on bin size (184) but not on
#  training data.
#
#  DO NOT REPORT THE FIRST ATTEMPT AS A NEGATIVE RESULT. It is inconclusive.
#  Reporting it as "the regime is LEDGAR-only, confirmed" would be exactly the
#  error the gate was written to prevent.
#
#  -----------------------------------------------------------------------
#  THE CORRECTED DESIGN
#  -----------------------------------------------------------------------
#  Hold --min_sentences at 2 for BOTH arms, the CUAD-whole default, so nothing
#  that CUAD-whole kept is discarded. Vary only --max_sentences, which chops
#  long blocks into different window sizes. Same corpus, same documents, two
#  paragraph-length distributions.
#
#      arm S   --min_sentences 2  --max_sentences 6
#              nothing longer than six sentences; in-band paragraphs sit at
#              exactly the arithmetic floor (n = 6, k = 1, d_p = 16.7%)
#
#      arm L   --min_sentences 2  --max_sentences 20
#              paragraphs up to twenty sentences; in-band paragraphs span a
#              range of lengths, as they do on LEDGAR
#
#  Chopping produces MORE paragraphs in arm S, not fewer, so both arms train on
#  a corpus at least the size of CUAD-whole's. That is the property the first
#  attempt lacked.
#
#  This is the comparison Chapters 9 and 10 ask for: the in-band advantage at
#  two different in-band length distributions, on one corpus, with training
#  data held comparable.
#
#  -----------------------------------------------------------------------
#  BUDGET: ~16 GPU-hours. Day 6 of 9, so this fits with a day to spare.
#  -----------------------------------------------------------------------
#
#  Usage:   bash run_stage1b_fix.sh
#
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

LOG=outputs/logs
CKPT=outputs/checkpoints
SENT=outputs/sentinels
mkdir -p "$LOG" "$CKPT" "$SENT" logs

PY=${PY:-python}
SEEDS_5="42 43 44 45 46"

run_once () {
    local name="$1"; shift
    if [[ -f "$SENT/$name.done" ]]; then echo "[skip]  $name"; return 0; fi
    echo "[run ]  $name"
    echo "        $*"
    "$@"
    touch "$SENT/$name.done"
}

echo "=============== STAGE 1b — corrected length/density design ==============="

# --- preprocess: min_sentences fixed at 2, only max_sentences varies ---------
for ARM in "S 2 6" "L 2 20"; do
    set -- $ARM; NAME=$1; LO=$2; HI=$3
    run_once "s1b_preprocess_$NAME" \
        $PY scripts/preprocess_cuad_whole.py \
            --output_dir "data/processed_cuadw_win$NAME" \
            --min_sentences "$LO" --max_sentences "$HI" \
            --detector_ckpt "$CKPT/operator_detector_neural.pt"
done

# -----------------------------------------------------------------------------
#  GATE — ENFORCE IT THIS TIME. Two conditions, both must hold in both arms.
#
#    (a) training set within ~2x of CUAD-whole's, so the model can learn at all
#    (b) 10-20% bin >= 100 paragraphs, so the comparison is powered
#
#  The preprocessing script prints Train/Val/Test counts. The training script
#  prints the bin count. If either condition fails in an arm, abandon that arm
#  and say so -- do not train it and report its null.
#
#  Reference points:
#    CUAD-whole   1,222 test paragraphs, 123 in the 10-20% bin, 0.33 macro-F1
#    LEDGAR      10,000 test paragraphs, 156 in the bin,        0.70 macro-F1
#    first try      317 test (arm B),     81 in the bin,        0.12 macro-F1
# -----------------------------------------------------------------------------
echo
echo "  >>> GATE: check BOTH arms before training."
echo "      (a) test split within ~2x of CUAD-whole's 1,222 paragraphs"
echo "      (b) 10-20% bin >= 100 paragraphs"
echo "      Abandon an arm that fails either. Do not train it."
echo "      Press Ctrl-C now if the preprocessing counts above look wrong."
echo
sleep 20

for ARM in S L; do
    for VARIANT in mean v3; do
        for s in $SEEDS_5; do
            run_once "s1b_${VARIANT}_win${ARM}_s$s" \
                $PY scripts/train_ledgar.py \
                    --data_dir "data/processed_cuadw_win$ARM" \
                    --dmp_variant "$VARIANT" \
                    --seed "$s" \
                    --tag "cuadw_win${ARM}_${VARIANT}"
        done
    done
done

for ARM in S L; do
    run_once "s1b_analyse_win$ARM" \
        $PY scripts/analyse_ledgar_variant.py \
            --tag "cuadw_win${ARM}_v3" \
            --variant v3 \
            --mean_tag "cuadw_win${ARM}_mean" \
            --data_dir "data/processed_cuadw_win$ARM" \
            --seeds $SEEDS_5 \
            --bin_lo 0.10 --bin_hi 0.20 \
            --ckpt_dir "$CKPT"
done

cat <<'VERDICT'

=======================================================================
  VERDICT RULE — and one precondition that comes first.

  PRECONDITION: both arms must reach an absolute macro-F1 in the region of
  CUAD-whole's 0.33. If they land near 0.12 again, the arms are still
  starved and the result is INCONCLUSIVE regardless of the deltas. Say so
  and stop; do not read a null off a model that cannot do the task.

  Given that precondition holds:

    in-band advantage in BOTH arms   -> density effect; C8 stands as written
    in arm L only                    -> length effect; the band is a
                                        long-paragraph subset and C8 must be
                                        restated as a pooling-architecture
                                        result
    in NEITHER arm                   -> the regime does not transfer to CUAD
                                        under any segmentation, which sharpens
                                        Chapter 8's LEDGAR-specific reading
                                        from "one preprocessing" to "three"

  Whichever it is, this lands after deposit. Record it for the December
  defence and for the AI & Law revision. Chapter 9 is already hedged for
  every one of these outcomes.
=======================================================================
VERDICT
