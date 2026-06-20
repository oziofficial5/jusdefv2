#!/bin/bash
# =============================================================================
# Tonight's 24h plan: three thesis-strengthening analyses
#
#   TASK 1 (~30 min):   Bootstrap p-values for v4_twostage on regime + aggregate
#   TASK 2 (~30 min):   Op_coef trajectory: retrain v3 seed 42 with logging + figure
#   TASK 3 (~30 min):   EUR-Lex within-corpus density-stratified regime test
#
# Total: ~90 minutes wall-clock. The remaining 22 hours are buffer for any
# eval reruns and for writing thesis updates.
#
# All three tasks operate on existing checkpoints + minimal retraining.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py src/model/*.py
#   nohup bash scripts/run_tonight_plan.sh > outputs/logs/run_tonight.log 2>&1 &
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

# =============================================================================
# TASK 1 -- bootstrap significance for v4_twostage
# =============================================================================

banner "TASK 1: paired-bootstrap significance for v4_twostage"

if ! have task1_bootstrap; then
  python -u scripts/bootstrap_v4_twostage.py 2>&1 | tee $LOG/task1_bootstrap.log
  done_run task1_bootstrap
fi

# =============================================================================
# TASK 2 -- op_coef trajectory: retrain v3 seed 42 with per-epoch logging
# =============================================================================

banner "TASK 2: v3 retrain with op_coef trajectory logging (seed 42)"

if ! have task2_v3_trajectory; then
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v3_traj \
    --dmp_variant v3 \
    --epochs 30 \
    2>&1 | tee $LOG/task2_v3_traj_s42.log
  done_run task2_v3_trajectory
fi

# =============================================================================
# TASK 3 -- EUR-Lex within-corpus density-stratified regime test
# =============================================================================

banner "TASK 3: EUR-Lex within-corpus density-stratified regime test"

if ! have task3_eurlex_regime; then
  # If the existing eval JSONs don't have per-document predictions populated,
  # the analysis script will print a SKIP message naming the missing variant.
  # In that case run eval_all_jusdef.py first to populate them.
  python -u scripts/analyse_eurlex_density_stratified.py 2>&1 \
    | tee $LOG/task3_eurlex_regime.log
  done_run task3_eurlex_regime
fi

# =============================================================================
# REGENERATE FIGURES (picks up fig17 trajectory + any other newly available data)
# =============================================================================

banner "Regenerate all figures (incl. fig17 op_coef trajectory)"

if ! have task_figures_regen; then
  python -u scripts/make_thesis_figures.py 2>&1 | tee $LOG/task_figures_regen.log
  done_run task_figures_regen
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "TONIGHT PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== Task 1: Bootstrap verdict ==="
grep -A 15 "BOOTSTRAP VERDICT" $LOG/task1_bootstrap.log 2>/dev/null | tail -15

echo ""
echo "=== Task 2: Final v3_traj seed 42 result ==="
grep -E "test_macro_f1|best_val_macro_f1" $LOG/task2_v3_traj_s42.log 2>/dev/null | tail -5

echo ""
echo "=== Task 3: EUR-Lex within-corpus regime verdict ==="
grep -A 20 "EUR-LEX WITHIN-CORPUS" $LOG/task3_eurlex_regime.log 2>/dev/null | tail -20

echo ""
echo "=== Figures regenerated ==="
ls -la outputs/figures/fig17_*.pdf 2>/dev/null
ls -la outputs/figures/fig*.pdf 2>/dev/null | wc -l | xargs -I{} echo "Total PDFs: {}"

echo ""
echo "=== Artefacts ==="
ls -la outputs/logs/bootstrap_v4_twostage.json \
       outputs/logs/eurlex_density_stratified.json \
       outputs/logs/ledgar_v3_traj_s42.json 2>/dev/null

echo ""
echo "All sentinels:"
ls -la outputs/sentinels/task*.done 2>/dev/null
