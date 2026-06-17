#!/bin/bash
# =============================================================================
# 4-day Ampere plan: ablations + multi-layer + extra seeds + Y_exc evaluation
#
# Total budget: ~96 GPU hours (4 days on A100).
# Expected total runtime: ~80-90 hours (with buffer).
#
# Stages:
#   DAY 1 (~26h): 4 LEDGAR v3 ablations (single seed each)
#   DAY 2 (~24h): Multi-layer V3Layer on LEDGAR (3 seeds, 2 layers)
#   DAY 3 (~26h): LEDGAR seeds 45, 46 for v3 and mean baseline (4 runs)
#   DAY 4 (~6h):  EUR-Lex Y_exc evaluation on existing checkpoints + buffer
#
# Sentinels: each ablation/run writes outputs/sentinels/<name>.done
# Re-running the script skips completed runs.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   bash scripts/run_4day_plan.sh
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

# =============================================================================
# DAY 1 — LEDGAR ablations (single seed each, isolates v3 components)
# =============================================================================

banner "DAY 1: LEDGAR v3 ablations"

# Ablation 1: revert single shared W to per-operator W (tests W-undertraining fix)
if ! have d1_a1_shared_w_revert; then
  echo ""
  echo "--- Ablation 1: shared_w_revert (per-operator W instead of shared) ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v3_abl_shared_w_revert \
    --dmp_variant v3 \
    --v3_shared_w_revert \
    2>&1 | tee $LOG/d1_a1_shared_w_revert.log
  done_run d1_a1_shared_w_revert
fi

# Ablation 2: hard attention (argmax) instead of soft softmax
if ! have d1_a2_hard_attention; then
  echo ""
  echo "--- Ablation 2: hard_attention (argmax instead of softmax) ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v3_abl_hard_attention \
    --dmp_variant v3 \
    --v3_hard_attention \
    2>&1 | tee $LOG/d1_a2_hard_attention.log
  done_run d1_a2_hard_attention
fi

# Ablation 3: unit coefs (1,1,1,1) — no signed semantics
if ! have d1_a3_unit_coefs; then
  echo ""
  echo "--- Ablation 3: unit_coefs (all +1, no signed semantics) ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v3_abl_unit_coefs \
    --dmp_variant v3 \
    --v3_init_coefs "1.0,1.0,1.0,1.0" \
    2>&1 | tee $LOG/d1_a3_unit_coefs.log
  done_run d1_a3_unit_coefs
fi

# Ablation 4: no drift regulariser (lambda_reg = 0)
if ! have d1_a4_no_drift_reg; then
  echo ""
  echo "--- Ablation 4: no_drift_reg (lambda_reg = 0) ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v3_abl_no_drift_reg \
    --dmp_variant v3 \
    --v3_coef_reg_strength 0.0 \
    2>&1 | tee $LOG/d1_a4_no_drift_reg.log
  done_run d1_a4_no_drift_reg
fi

banner "DAY 1 COMPLETE"

# =============================================================================
# DAY 2 — Multi-layer V3Layer (3 seeds, num_layers=2)
# =============================================================================

banner "DAY 2: Multi-layer V3Layer on LEDGAR (num_layers=2)"

for SEED in 42 43 44; do
  if have d2_v3_2layer_s${SEED}; then
    echo "[skip] v3 2-layer seed $SEED already done"
    continue
  fi
  echo ""
  echo "--- Multi-layer V3 seed $SEED (num_layers=2) ---"
  python -u scripts/train_ledgar.py \
    --seed $SEED \
    --tag v3_2layer \
    --dmp_variant v3 \
    --num_layers 2 \
    2>&1 | tee $LOG/d2_v3_2layer_s${SEED}.log
  done_run d2_v3_2layer_s${SEED}
done

banner "DAY 2 COMPLETE"

# =============================================================================
# DAY 3 — Extra LEDGAR seeds (45, 46) for v3 + mean baseline
# =============================================================================

banner "DAY 3: LEDGAR seeds 45, 46 for v3 and mean baseline"

for SEED in 45 46; do
  # Mean baseline
  if have d3_baseline_mean_s${SEED}; then
    echo "[skip] mean baseline seed $SEED already done"
  else
    echo ""
    echo "--- Mean baseline seed $SEED ---"
    python -u scripts/train_ledgar.py \
      --seed $SEED \
      --tag baseline_mean \
      --dmp_variant mean \
      2>&1 | tee $LOG/d3_baseline_mean_s${SEED}.log
    done_run d3_baseline_mean_s${SEED}
  fi

  # V3 pilot
  if have d3_v3_pilot_s${SEED}; then
    echo "[skip] v3 pilot seed $SEED already done"
  else
    echo ""
    echo "--- V3 seed $SEED ---"
    python -u scripts/train_ledgar.py \
      --seed $SEED \
      --tag v3_pilot \
      --dmp_variant v3 \
      2>&1 | tee $LOG/d3_v3_pilot_s${SEED}.log
    done_run d3_v3_pilot_s${SEED}
  fi
done

banner "DAY 3 COMPLETE"

# =============================================================================
# DAY 4 — EUR-Lex Y_exc-stratified evaluation + density-stratified re-runs
# =============================================================================

banner "DAY 4: EUR-Lex Y_exc + density-stratified analysis"

# Y_exc evaluation runs through the existing eval pipeline. Uses
# data/annotations/exception_labels.json which lists the 21 exception-dependent
# EuroVoc labels. eval_all_jusdef.py already computes the y_exc bucket if the
# file is present.
if ! have d4_eurlex_yexc; then
  echo ""
  echo "--- EUR-Lex Y_exc evaluation across all v2/v3 checkpoints ---"
  python -u scripts/eval_all_jusdef.py 2>&1 | tee $LOG/d4_eurlex_yexc.log
  done_run d4_eurlex_yexc
fi

# Re-run LEDGAR density-stratified analysis with the new 5-seed pool
if ! have d4_ledgar_5seed_stratified; then
  echo ""
  echo "--- LEDGAR 5-seed density-stratified analysis ---"
  python -u scripts/analyse_ledgar_density_subset.py 2>&1 | tee $LOG/d4_ledgar_5seed_stratified.log || true
  done_run d4_ledgar_5seed_stratified
fi

banner "DAY 4 COMPLETE"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "4-DAY PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== LEDGAR ablation results (seed 42) ==="
for ABL in shared_w_revert hard_attention unit_coefs no_drift_reg; do
  p="outputs/logs/ledgar_v3_abl_${ABL}_s42.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  v3_abl_${ABL}: test_macro_f1={d[\"test_macro_f1\"]:.4f}  test_micro_f1={d[\"test_micro_f1\"]:.4f}')
" 2>/dev/null || echo "  v3_abl_$ABL: result file present but unparseable"
  else
    echo "  v3_abl_$ABL: MISSING"
  fi
done

echo ""
echo "=== LEDGAR v3 multi-layer (num_layers=2) 3-seed results ==="
for SEED in 42 43 44; do
  p="outputs/logs/ledgar_v3_2layer_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  v3_2layer s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}  test_micro_f1={d[\"test_micro_f1\"]:.4f}')
" 2>/dev/null || echo "  v3_2layer s$SEED: result file present but unparseable"
  else
    echo "  v3_2layer s$SEED: MISSING"
  fi
done

echo ""
echo "=== LEDGAR 5-seed (42-46) v3 and baseline ==="
for TAG in baseline_mean v3_pilot; do
  echo "  $TAG:"
  for SEED in 42 43 44 45 46; do
    p="outputs/logs/ledgar_${TAG}_s${SEED}.json"
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
echo "All sentinels:"
ls -la outputs/sentinels/d?_*.done 2>/dev/null | head -20
