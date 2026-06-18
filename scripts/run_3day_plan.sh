#!/bin/bash
# =============================================================================
# 3-day Ampere plan: kill the three rejection-risk vectors
#
#   Risk 1 (severe):  3 of 4 ablations beat v3 on aggregate. Without per-regime
#                     ablations on the 10-20% bin, the architectural choices
#                     have no empirical justification.
#   Risk 2 (moderate-severe): No canonical baselines on LEDGAR beyond mean.
#                     Reviewers will ask: "what about R-GCN, GAT, BERT alone?"
#   Risk 3 (moderate): Narrow 156-paragraph regime is a cherry-picking target.
#                     Need bin-sensitivity sweep to defuse.
#
# Stages:
#   DAY 1: Per-regime ablations (4 ablations × 5 seeds = 20 runs) — kills Risk 1
#   DAY 2: Multi-layer extension + bin sensitivity + baseline stubs — kills Risk 3
#   DAY 3: Failure-mode analysis (CPU) + figures (CPU) + ECtHR (if pipeline exists)
#
# Estimated GPU wall-clock based on the prior 47-minute run:
#   Each train_ledgar.py run ~3-4 min on A100.
#   Day 1: 20 runs × 4 min = ~80 min
#   Day 2: 10 runs × 4 min + analysis = ~50 min
#   Day 3: mostly CPU + buffer
#
# The 3-day "budget" is for safety, debugging, and to absorb the new training
# code that some Day-2/3 tasks require (see TODOs below).
#
# Sentinels: each block writes outputs/sentinels/<name>.done — re-runs skip.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   bash scripts/run_3day_plan.sh
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints outputs/figures

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

ABLATIONS=("shared_w_revert" "hard_attention" "unit_coefs" "no_drift_reg")
SEEDS=(42 43 44 45 46)

# =============================================================================
# DAY 1 — Per-regime ablations (5 seeds × 4 ablations)
#
# This is the highest-ROI work. After this finishes, the per-regime evaluator
# (Day 1.5) produces a 4-ablation × 7-density-bin table that either:
#   (a) shows at least one ablation HURTS v3 on the 10-20% bin
#       → architectural choices are justified on the regime; thesis defensible
#   (b) shows all ablations match v3 on the 10-20% bin too
#       → architecture has no per-regime justification; reframe thesis
#       (see Chapter 7 honest-framing plan)
# =============================================================================

banner "DAY 1: Per-regime ablations × 5 seeds (20 runs)"

for ABL in "${ABLATIONS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    SENT_NAME="d1_${ABL}_s${SEED}"
    if have $SENT_NAME; then
      echo "[skip] $SENT_NAME"
      continue
    fi
    echo ""
    echo "--- Ablation $ABL seed $SEED ---"

    # Build per-ablation flags
    EXTRA_FLAGS=""
    case "$ABL" in
      shared_w_revert)   EXTRA_FLAGS="--v3_shared_w_revert" ;;
      hard_attention)    EXTRA_FLAGS="--v3_hard_attention" ;;
      unit_coefs)        EXTRA_FLAGS="--v3_init_coefs 1.0,1.0,1.0,1.0" ;;
      no_drift_reg)      EXTRA_FLAGS="--v3_coef_reg_strength 0.0" ;;
    esac

    python -u scripts/train_ledgar.py \
      --seed $SEED \
      --tag v3_abl_${ABL} \
      --dmp_variant v3 \
      $EXTRA_FLAGS \
      2>&1 | tee $LOG/${SENT_NAME}.log
    done_run $SENT_NAME
  done
done

banner "DAY 1 TRAINING COMPLETE"

# -----------------------------------------------------------------------------
# DAY 1.5 — Per-regime ablation evaluation (no training; loads checkpoints)
#
# REQUIRES a new script: scripts/analyse_ledgar_ablations_stratified.py
# Pattern: load all ledgar_v3_abl_{ABL}_s{SEED}.pt + ledgar_baseline_mean_s{SEED}.pt
#          for SEED in {42..46}, ABL in {4 ablations}, run density-stratified eval
#          and report ablation × density-bin × seed → table + per-bin mean ± std.
#
# If the helper script is missing, this block degrades gracefully (the .done
# sentinel is still written so the day-2 stages can run). Re-run after writing
# the helper.
# -----------------------------------------------------------------------------

if ! have d1_per_regime_ablation_eval; then
  banner "DAY 1.5: Per-regime ablation evaluation"
  if [[ -f scripts/analyse_ledgar_ablations_stratified.py ]]; then
    python -u scripts/analyse_ledgar_ablations_stratified.py 2>&1 \
      | tee $LOG/d1_per_regime_ablation_eval.log
  else
    echo "[STUB] scripts/analyse_ledgar_ablations_stratified.py is missing."
    echo "[STUB] Extend analyse_ledgar_density_subset.py to loop over"
    echo "[STUB]   tags = ['v3_pilot', 'v3_abl_shared_w_revert', 'v3_abl_hard_attention',"
    echo "[STUB]           'v3_abl_unit_coefs', 'v3_abl_no_drift_reg']"
    echo "[STUB] and seeds 42..46, producing a tag × density-bin × seed table."
    echo "[STUB] Skipping for now — re-run this block after writing the script."
  fi
  done_run d1_per_regime_ablation_eval
fi

# =============================================================================
# DAY 2 — Multi-layer extension + density-bin sensitivity + baseline stubs
# =============================================================================

banner "DAY 2: Multi-layer V3 extension + bin sensitivity + baseline stubs"

# ---------------------------------------------------------------------------
# DAY 2.1 — Extend multi-layer V3 to 5 seeds (currently 3 seeds in prior run)
# ---------------------------------------------------------------------------

for SEED in 45 46; do
  SENT_NAME="d2_v3_2layer_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- Multi-layer V3 (num_layers=2) seed $SEED ---"
  python -u scripts/train_ledgar.py \
    --seed $SEED \
    --tag v3_2layer \
    --dmp_variant v3 \
    --num_layers 2 \
    2>&1 | tee $LOG/${SENT_NAME}.log
  done_run $SENT_NAME
done

# ---------------------------------------------------------------------------
# DAY 2.2 — Density-bin sensitivity sweep (no training; analysis only)
#
# Defuses the "cherry-picked 10-20% threshold" objection. Re-runs the
# stratified analysis with three alternative bin definitions and shows the
# v3 win persists (or doesn't) across nearby bin choices.
#
# REQUIRES a new script: scripts/analyse_ledgar_bin_sensitivity.py
# (or: pass --bin_centre and --bin_width to the existing analyse script)
# ---------------------------------------------------------------------------

if ! have d2_bin_sensitivity; then
  echo ""
  echo "--- Density-bin sensitivity sweep ---"
  if [[ -f scripts/analyse_ledgar_bin_sensitivity.py ]]; then
    python -u scripts/analyse_ledgar_bin_sensitivity.py 2>&1 \
      | tee $LOG/d2_bin_sensitivity.log
  else
    echo "[STUB] scripts/analyse_ledgar_bin_sensitivity.py is missing."
    echo "[STUB] Should sweep bins: (8,22) (9,18) (10,20) (12,25) (10,15)"
    echo "[STUB] and report 5-seed mean delta on each. Skipping for now."
  fi
  done_run d2_bin_sensitivity
fi

# ---------------------------------------------------------------------------
# DAY 2.3 — Canonical baselines on LEDGAR
#
# CURRENTLY NOT RUNNABLE: train_ledgar.py only supports --dmp_variant {v3,mean}.
# R-GCN and GAT require new model files + training code.
#
# Minimal viable: just-LegalBERT-mean baseline can be approximated by the
# existing "mean" variant (which already is mean of LegalBERT sentence embs
# → linear classifier). So "BERT-only" is effectively the current mean.
# Recording this as a thesis-note rather than a new run.
#
# R-GCN on LEDGAR: stub. Either implement scripts/train_ledgar_rgcn.py
# (treat the paragraph as a single-relation graph over sentences) or skip
# and frame in Chapter 8 as a known limitation.
# ---------------------------------------------------------------------------

if ! have d2_baseline_stub; then
  echo ""
  echo "--- LEDGAR canonical-baseline stubs (NOT RUN) ---"
  echo "  R-GCN-on-LEDGAR:        requires new training code. Stubbed."
  echo "  GAT-on-LEDGAR:          requires new training code. Stubbed."
  echo "  LegalBERT-mean:         equivalent to existing 'mean' variant."
  echo "  Recommendation:         frame Chapter 8 with the existing 'mean' baseline"
  echo "                          + a Section 9.X limitation paragraph naming what"
  echo "                          R-GCN/GAT would add and why it's deferred to"
  echo "                          future work."
  done_run d2_baseline_stub
fi

banner "DAY 2 COMPLETE"

# =============================================================================
# DAY 3 — Robustness + figures + buffer
# =============================================================================

banner "DAY 3: Failure-mode analysis + figures + ECtHR (if available)"

# ---------------------------------------------------------------------------
# DAY 3.1 — Failure-mode analysis on existing v2 checkpoints (CPU)
#
# Task #29 (pending). Reuses existing EUR-Lex v2 checkpoints — no GPU needed.
# Confirms STE bias, sign cancellation, and W-undertraining on existing
# trained models. Produces evidence for Chapter 6's mechanistic claim.
# ---------------------------------------------------------------------------

if ! have d3_failure_mode_analysis; then
  echo ""
  echo "--- Failure-mode analysis on v2 checkpoints ---"
  if [[ -f scripts/analyse_v2_failure_modes.py ]]; then
    python -u scripts/analyse_v2_failure_modes.py 2>&1 \
      | tee $LOG/d3_failure_mode_analysis.log
  else
    echo "[STUB] scripts/analyse_v2_failure_modes.py is missing."
    echo "[STUB] Should: (a) measure gate-saturation rate per operator (STE bias)"
    echo "[STUB]         (b) measure op_coef norm across depth (sign cancellation)"
    echo "[STUB]         (c) measure ||W_op|| imbalance vs message count (undertraining)"
    echo "[STUB] Skipping for now."
  fi
  done_run d3_failure_mode_analysis
fi

# ---------------------------------------------------------------------------
# DAY 3.2 — ECtHR 5-seed run (if pipeline exists)
#
# CURRENTLY UNCERTAIN: ECtHR training pipeline status in this repo.
# If train_ecthr.py exists, run 5 seeds. Otherwise stub.
# ---------------------------------------------------------------------------

if ! have d3_ecthr_5seed; then
  echo ""
  echo "--- ECtHR 5-seed v3 + mean baseline (if pipeline available) ---"
  if [[ -f scripts/train_ecthr.py ]]; then
    for SEED in 42 43 44 45 46; do
      python -u scripts/train_ecthr.py --seed $SEED --tag v3 --dmp_variant v3 \
        2>&1 | tee $LOG/d3_ecthr_v3_s${SEED}.log
      python -u scripts/train_ecthr.py --seed $SEED --tag mean --dmp_variant mean \
        2>&1 | tee $LOG/d3_ecthr_mean_s${SEED}.log
    done
  else
    echo "[STUB] scripts/train_ecthr.py is missing."
    echo "[STUB] ECtHR pipeline not in repo. Either:"
    echo "[STUB]   (a) port the LEDGAR pipeline to ECtHR (preprocess + train + eval), OR"
    echo "[STUB]   (b) frame Chapter 8 with EUR-Lex + LEDGAR only and note ECtHR as"
    echo "[STUB]       single-seed density-measurement context (which is what you have)."
    echo "[STUB] Recommendation: (b) — ECtHR density of 0.58% predicts no architectural"
    echo "[STUB] benefit per the operating-regime hypothesis, so an ECtHR null result"
    echo "[STUB] adds little beyond what EUR-Lex already shows."
  fi
  done_run d3_ecthr_5seed
fi

# ---------------------------------------------------------------------------
# DAY 3.3 — Figure generation (CPU, local)
#
# REQUIRES a new script: scripts/make_thesis_figures.py
# Should produce the 10 figures listed in the 3-day planning chat:
#   1. density_stratified_bars.pdf
#   2. cross_corpus_density_hist.pdf
#   3. bin_sensitivity.pdf
#   4. per_regime_ablation_heatmap.pdf
#   5. op_coef_trajectory.pdf
#   6. attention_entropy_violin.pdf
#   7. confusion_matrix_10_20_bin.pdf
#   8. eurlex_underperformance_ci.pdf
#   9. compute_vs_gain_pareto.pdf
#  10. annotation_pipeline_schematic.pdf (Tikz, separate)
# ---------------------------------------------------------------------------

if ! have d3_figures; then
  echo ""
  echo "--- Thesis figures ---"
  if [[ -f scripts/make_thesis_figures.py ]]; then
    python -u scripts/make_thesis_figures.py 2>&1 | tee $LOG/d3_figures.log
  else
    echo "[STUB] scripts/make_thesis_figures.py is missing."
    echo "[STUB] All figure logic stubs are matplotlib + JSON-log readers."
    echo "[STUB] Can be written locally without GPU."
  fi
  done_run d3_figures
fi

banner "DAY 3 COMPLETE"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "3-DAY PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== Day 1: Per-regime ablation runs (5 seeds × 4 ablations) ==="
for ABL in "${ABLATIONS[@]}"; do
  echo "  $ABL:"
  for SEED in "${SEEDS[@]}"; do
    p="outputs/logs/ledgar_v3_abl_${ABL}_s${SEED}.json"
    if [[ -f "$p" ]]; then
      python -c "
import json
d = json.load(open('$p'))
print(f'    s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "    s$SEED: unparseable"
    else
      echo "    s$SEED: MISSING"
    fi
  done
done

echo ""
echo "=== Day 2: Multi-layer V3 5-seed (extension of prior 3-seed) ==="
for SEED in 42 43 44 45 46; do
  p="outputs/logs/ledgar_v3_2layer_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  v3_2layer s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "  v3_2layer s$SEED: unparseable"
  else
    echo "  v3_2layer s$SEED: MISSING"
  fi
done

echo ""
echo "=== Helper scripts that need to be written (Day-2 and Day-3 stubs) ==="
for HELPER in \
    scripts/analyse_ledgar_ablations_stratified.py \
    scripts/analyse_ledgar_bin_sensitivity.py \
    scripts/analyse_v2_failure_modes.py \
    scripts/make_thesis_figures.py; do
  if [[ -f "$HELPER" ]]; then
    echo "  [present] $HELPER"
  else
    echo "  [missing] $HELPER"
  fi
done

echo ""
echo "All sentinels:"
ls -la outputs/sentinels/d?_*.done 2>/dev/null | head -30
