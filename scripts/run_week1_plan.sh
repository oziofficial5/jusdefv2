#!/bin/bash
# =============================================================================
# Week 1: direct empirical evidence for Chapter 6 + interpretability for Ch. 7
#
# Two analyses, no training. Both run on existing checkpoints.
#
#   STAGE 1 (~10-30 min, CPU+light GPU): failure-mode analysis on v2 EUR-Lex
#   STAGE 2 (~30-60 min, GPU):           counterfactual operator-intervention
#                                        sensitivity on LEDGAR
#
# Transforms two thesis-attack vectors:
#   - "you infer failure modes, don't measure them" -> direct measurement (Ch.6)
#   - "where's evidence v3 actually uses operator info?" -> sensitivity table (Ch.6/Ch.7)
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py src/model/*.py
#   bash scripts/run_week1_plan.sh 2>&1 | tee outputs/logs/run_week1.log
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/figures

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

# =============================================================================
# STAGE 1 -- v2 failure-mode direct measurement on existing EUR-Lex checkpoints
# =============================================================================

banner "STAGE 1: v2 failure-mode direct measurement (Chapter 6 evidence)"

if ! have w1_v2_failure_modes; then
  python -u scripts/analyse_v2_failure_modes.py 2>&1 | tee $LOG/w1_v2_failure_modes.log
  done_run w1_v2_failure_modes
fi

# =============================================================================
# STAGE 2 -- Counterfactual operator-intervention sensitivity on LEDGAR
# =============================================================================

banner "STAGE 2: counterfactual operator sensitivity (Chapter 6 + 7 evidence)"

if ! have w1_counterfactual_sensitivity; then
  python -u scripts/analyse_counterfactual_sensitivity.py 2>&1 \
    | tee $LOG/w1_counterfactual_sensitivity.log
  done_run w1_counterfactual_sensitivity
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "WEEK 1 PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== KEY VERDICTS ==="
echo ""
echo "--- v2 failure-mode verdict ---"
grep -A 20 "FAILURE-MODE VERDICT" $LOG/w1_v2_failure_modes.log 2>/dev/null | tail -20

echo ""
echo "--- Counterfactual sensitivity verdict ---"
grep -A 20 "SENSITIVITY VERDICT" $LOG/w1_counterfactual_sensitivity.log 2>/dev/null | tail -20

echo ""
echo "=== ARTEFACTS ==="
ls -la outputs/logs/v2_failure_mode_analysis.json \
       outputs/logs/ledgar_counterfactual_sensitivity.json \
       2>/dev/null

echo ""
echo "=== FIGURES ==="
ls -la outputs/figures/fig7_*.pdf outputs/figures/fig8_*.pdf \
       outputs/figures/fig9_*.pdf 2>/dev/null

echo ""
echo "Sentinels:"
ls -la outputs/sentinels/w1_*.done 2>/dev/null
