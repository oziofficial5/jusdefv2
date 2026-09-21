#!/bin/bash
# =============================================================================
#  JusDef — Ampere, 21 to 30 September 2026.  NINE DAYS.
# =============================================================================
#
#  The GPU window and the thesis deposit END ON THE SAME DAY.
#
#    GPU available   21 Sep -> 30 Sep        (9 days, ~216 h if exclusive)
#    Thesis deposit  30 Sep
#    Defence         December 2026
#    AI & Law revision (Khaliq & Montanelli), under review, no fixed date
#
#  READ THIS BEFORE SUBMITTING ANYTHING.
#
#  A GPU run that finishes on 29 September cannot realistically change the
#  deposited text. Analysing it, deciding what it means, editing four chapters
#  and re-uploading takes longer than the day that would be left. Anything that
#  must reach the thesis has to be done WITHOUT the GPU, and that is Stage 0.
#
#  So plan the GPU for the DEFENCE, not for the text. The single most likely
#  question in December is: "Chapters 9 and 10 name one outstanding experiment.
#  Did you run it?" Nine days of A100 is enough to be able to answer yes.
#
#  ONE RISK, STATED PLAINLY. If Stage 1 comes back saying the operating regime
#  is a length effect, you will know it before you deposit, and you will then be
#  obliged to act on it with days to spare. The thesis is already hedged for
#  this -- Ch. 9 says the regime "should be cited as a property of this pipeline
#  and this corpus" until the run is done -- so the edit is survivable, but it
#  is real. The alternative, not running it and being asked in December why
#  not when you had the machine, is worse. Run it.
#
#  PLAN
#    Day 1        Stage 0, no GPU. The only work that can reach the thesis.
#    Day 1        Submit Stage 1 immediately, in parallel. Do not wait.
#    Days 1-6     Stage 1, ~51 GPU-hours.  The experiment the thesis names.
#    Days 7-9     Stage 2, ~25 GPU-hours.  Depth sweep, low risk.
#    Day 9        Stop. Deposit.
#
#  Five seeds, not ten. This matches the five-seed CUAD panels already in
#  Chapter 8, so the numbers are directly comparable with what is written, and
#  it halves the wall-clock. Ten seeds would need ~94 h and leaves no slack for
#  a failed job.
#
#  DROPPED FOR LACK OF TIME (they were in the month-long plan):
#    - contrastive encoder intervention: needs a day of new code first, then
#      ~30 h. Does not fit, and is the least likely of the three to change any
#      claim. Do it after deposit if the GPU is extended.
#    - a denser corpus: a data-acquisition problem, not a compute one.
#
#  Usage:
#      bash run_ampere_9day.sh 0      # no GPU; run this on a laptop today
#      bash run_ampere_9day.sh 1      # submit to the GPU today, in parallel
#      bash run_ampere_9day.sh 2      # only after Stage 1 has finished
#
#  SLURM (confirm partition and GPU string with `sinfo` first):
#      #SBATCH --partition=gpu
#      #SBATCH --gres=gpu:a100:1
#      #SBATCH --cpus-per-task=8
#      #SBATCH --mem=64G
#      #SBATCH --time=48:00:00
#
#  Every step is sentinel-gated under outputs/sentinels/, so a timeout or a
#  requeue resumes rather than restarting. With nine days that matters.
#
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

LOG=outputs/logs
CKPT=outputs/checkpoints
SENT=outputs/sentinels
mkdir -p "$LOG" "$CKPT" "$SENT" logs

STAGE="${1:-0}"
PY=${PY:-python}
SEEDS_5="42 43 44 45 46"
SEEDS_10="42 43 44 45 46 47 48 49 50 51"

run_once () {
    local name="$1"; shift
    if [[ -f "$SENT/$name.done" ]]; then echo "[skip]  $name"; return 0; fi
    echo "[run ]  $name"
    echo "        $*"
    "$@"
    touch "$SENT/$name.done"
}

want () { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }


# =============================================================================
#  STAGE 0 — DAY 1, NO GPU.  The only work that can reach the deposited thesis.
# =============================================================================
#
#  Three analyses over logs that already exist. None can fail in a way that
#  costs anything, and each removes a sentence in which the thesis currently
#  concedes something is unresolved. Do these before touching the GPU queue.
#
if want 0; then
echo "=============== STAGE 0 — day 1, no GPU ==============="

# ---------------------------------------------------------------------------
# 0.1  Dispersion convention behind the data-efficiency table.
#
# Chapter 8 reports the largest data-efficiency delta as -0.030 +/- 0.026 on
# three seeds and concludes t(2) = 1.16, p = 0.37. That t follows only if the
# +/- is the STANDARD ERROR. The thesis states twice -- Ch. 9 statistical
# validity, and the per-seed appendix caption -- that dispersion is the sample
# standard deviation throughout, under which the same numbers give t(2) = 2.01,
# p = 0.18. Both leave the result non-significant, so the conclusion is safe,
# but the arithmetic as printed does not close.
#
# The t and p have been REMOVED from Ch. 8 pending this. Reinstate them once
# the output says which convention the table follows.
#
# recompute_ledgar_dispersion.py is directly on point: its docstring records
# that two existing logs already disagree because one uses ddof=1 and the other
# ddof=0. The answer may be on disk already.
# ---------------------------------------------------------------------------
run_once s0_dispersion_audit \
    $PY scripts/recompute_ledgar_dispersion.py

run_once s0_dataeff_recheck \
    $PY scripts/analyse_dataeff.py \
        --sizes 500 1500 5000 15000 60000 \
        --seeds 42 43 44 \
        --logdir "$LOG"

# ---------------------------------------------------------------------------
# 0.2  Benjamini-Hochberg over the full ten-seed panel.
#
# Ch. 9 reports the 10-20% bin surviving BH at q = 0.011 across six disjoint
# bins. Confirm that figure comes from the panel the thesis now headlines
# rather than from an earlier one.
# ---------------------------------------------------------------------------
run_once s0_fdr_recheck \
    $PY scripts/analyse_ledgar_allbins_fdr.py --seeds $SEEDS_10

# ---------------------------------------------------------------------------
# 0.3  Per-seed bootstrap on the ten-seed panel.                  [NEEDS-CODE]
#
# Ch. 9 discloses: "per-seed p-values on the 10-20% bin were computed for the
# original three-seed panel only ... so no five- or ten-seed count of
# significant seeds is claimed anywhere in this thesis." The per-paragraph
# predictions for all ten seeds are already on disk, so this costs nothing but
# the wiring.
#
# NEEDS-CODE: bootstrap_v4_twostage.py already does exactly this machinery --
# "10,000 paired resamples on per-paragraph F1 contributions, reported per seed
# and pooled across seeds" -- but is hard-coded to the v4_twostage comparison
# and takes no arguments. Copy it to scripts/bootstrap_v3_vs_mean.py and point
# it at the v3 and mean tags over ten seeds. Half a day.
#
# Lowest priority of the three: if day 1 runs out, drop this one. The existing
# Ch. 9 disclosure is honest as it stands.
# ---------------------------------------------------------------------------
if [[ -f scripts/bootstrap_v3_vs_mean.py ]]; then
    run_once s0_perseed_bootstrap \
        $PY scripts/bootstrap_v3_vs_mean.py
else
    echo "  SKIP 0.3: scripts/bootstrap_v3_vs_mean.py not written yet."
    echo "            Adapt scripts/bootstrap_v4_twostage.py. Optional."
fi

echo
echo "  STAGE 0 done. Make the Ch. 8 and Ch. 9 edits and re-upload those two"
echo "  files. This is the only part of the nine days that changes the thesis."
echo
fi


# =============================================================================
#  STAGE 1 — DAYS 1-6, ~51 GPU-HOURS.  The experiment the thesis names.
# =============================================================================
#
#  SUBMIT THIS ON DAY 1, alongside Stage 0. Do not wait for Stage 0 to finish;
#  they share no inputs.
#
#  Ch. 9 (external validity) and Ch. 10 (future work) both name exactly one
#  outstanding run:
#
#     "What is needed is a corpus whose paragraph-length distribution differs
#      materially from LEDGAR's, so that in-band paragraphs exist at lengths
#      LEDGAR cannot supply and the in-band advantage can be compared at
#      matched length."
#
#  WHY. Because d_p = k/n over small integers, no paragraph shorter than six
#  sentences can enter the 10-20% band. LEDGAR's in-band paragraphs average
#  7.74 sentences against 1.96 for the rest of the test set, so the band is a
#  long-paragraph subset by construction. The within-corpus control already run
#  -- hold n >= 6, vary density, giving +0.0753 in-band against +0.0137 out --
#  shows density does work that length alone does not, but it cannot dissolve
#  the entanglement, because on a fixed corpus every in-band paragraph is long.
#
#  DESIGN. Re-segment CUAD-whole at two sentence windows that place in-band
#  paragraphs at different lengths.
#
#     arm A   --min_sentences 6  --max_sentences 7    in-band at the floor
#     arm B   --min_sentences 10 --max_sentences 20   in-band long, as LEDGAR
#
#  preprocess_cuad_whole.py already takes these flags -- that is how CUAD-whole
#  was built at 2-12 -- so no new preprocessing code is needed.
#
if want 1; then
echo "=============== STAGE 1 — length vs density at matched length ==============="

for ARM in "A 6 7" "B 10 20"; do
    set -- $ARM; NAME=$1; LO=$2; HI=$3
    run_once "s1_preprocess_len$NAME" \
        $PY scripts/preprocess_cuad_whole.py \
            --output_dir "data/processed_cuadw_len$NAME" \
            --min_sentences "$LO" --max_sentences "$HI" \
            --detector_ckpt "$CKPT/operator_detector_neural.pt"
done

# -----------------------------------------------------------------------------
#  GATE — DO NOT SKIP. Costs minutes, protects five days.
#
#  CUAD-spans already taught this lesson: its 10-20% bin held one paragraph and
#  the regime was untestable. Count the band in both arms before training.
#
#  PROCEED ONLY IF the 10-20% bin holds >= 100 paragraphs in that arm.
#  LEDGAR's is 156 and CUAD-whole's is 123. Below roughly 100 the comparison is
#  not powered, and an underpowered null is worse than no run: it looks like
#  evidence against the regime when it is only noise. Abandon the arm instead.
#
#  NEEDS-CODE: analyse_cuad_density_subset.py is hard-coded to the original
#  CUAD directory and takes no arguments. Add --data_dir, or write a ten-line
#  counter over the processed pickles. Ten minutes.
# -----------------------------------------------------------------------------
echo
echo "  >>> GATE: count the 10-20% bin in both arms NOW."
echo "      Abandon any arm below ~100 paragraphs rather than reporting its null."
echo

# Five seeds, matching the five-seed CUAD panels already in Chapter 8.
for ARM in A B; do
    for VARIANT in mean v3; do
        for s in $SEEDS_5; do
            run_once "s1_${VARIANT}_len${ARM}_s$s" \
                $PY scripts/train_ledgar.py \
                    --data_dir "data/processed_cuadw_len$ARM" \
                    --dmp_variant "$VARIANT" \
                    --seed "$s" \
                    --tag "cuadw_len${ARM}_${VARIANT}"
        done
    done
done

for ARM in A B; do
    run_once "s1_analyse_len$ARM" \
        $PY scripts/analyse_ledgar_variant.py \
            --tag "cuadw_len${ARM}_v3" \
            --variant v3 \
            --mean_tag "cuadw_len${ARM}_mean" \
            --data_dir "data/processed_cuadw_len$ARM" \
            --seeds $SEEDS_5 \
            --bin_lo 0.10 --bin_hi 0.20 \
            --ckpt_dir "$CKPT"
done

echo
echo "  VERDICT RULE — decide this before you look at the numbers:"
echo "    in-band advantage in BOTH arms  -> density effect; C8 stands as written"
echo "    in arm B only                   -> length effect; C8 must be restated"
echo "                                       as a pooling-architecture result"
echo "    in neither                      -> regime is LEDGAR-only, which"
echo "                                       Chapter 8 already argues"
echo
echo "  If the answer arrives before 28 Sep it can still reach the thesis."
echo "  After that, record it for the defence and the AI & Law revision and"
echo "  leave the deposited text as it stands -- Ch. 9 is already hedged for"
echo "  exactly this outcome."
echo
fi


# =============================================================================
#  STAGE 2 — DAYS 7-9, ~25 GPU-HOURS.  Depth sweep. Low risk, low stakes.
# =============================================================================
#
#  Only start this once Stage 1 has finished. If Stage 1 overruns, drop Stage 2
#  entirely -- it cannot change a claim, whereas Stage 1 can.
#
#  Ch. 9 and Ch. 10: the two-layer case is already tested and does not help
#  (aggregate +0.0054, inside seed-to-seed variance, dense-bin deficit intact).
#  What remains untested is whether stacks deeper than two behave differently.
#  The thesis claims the dense-regime bottleneck is REPRESENTATIONAL rather
#  than a matter of depth, and that claim currently rests on n = 2.
#
#  Expect it to fail: three and four layers on a graph this shallow should
#  oversmooth, and sign cancellation is depth-driven. A clean negative here
#  strengthens the thesis rather than threatening it, which is why it is the
#  safe thing to run in the last three days.
#
#  --num_layers and --v3_coef_reg_strength are both already supported.
#
if want 2; then
echo "=============== STAGE 2 — deeper V3Layer stacks ==============="

for L in 3 4; do
    for s in $SEEDS_5; do
        run_once "s2_v3_L${L}_s$s" \
            $PY scripts/train_ledgar.py \
                --dmp_variant v3 --num_layers "$L" --seed "$s" \
                --tag "v3_L${L}"
    done
done

# One depth at a stronger drift regulariser: the current strength is tuned for
# a single layer and cancellation compounds with depth.
for s in 42 43 44; do
    run_once "s2_v3_L3_reg_s$s" \
        $PY scripts/train_ledgar.py \
            --dmp_variant v3 --num_layers 3 --v3_coef_reg_strength 0.05 \
            --seed "$s" --tag v3_L3_reg05
done

for T in v3_L3 v3_L4 v3_L3_reg05; do
    run_once "s2_analyse_$T" \
        $PY scripts/analyse_ledgar_variant.py \
            --tag "$T" --variant v3 \
            --seeds $SEEDS_5 \
            --bin_lo 0.10 --bin_hi 0.20 \
            --ckpt_dir "$CKPT"
done
echo
fi


echo "======================================================================="
echo "  Done for stage: $STAGE"
echo "  Sentinels in $SENT — delete one to force that step to re-run."
echo "======================================================================="
