#!/bin/bash
# =============================================================================
# CUAD cross-corpus regime test plan (~8-10 hours)
#
#   STAGE 1 (~2-3 hours): preprocess CUAD into JusDefLEDGAR pkl format
#   STAGE 2 (~3-4 hours): train v3 + mean baseline on CUAD, 5 seeds each
#   STAGE 3 (~5 min):     density-stratified analysis on CUAD test
#
# Purpose: test whether the LEDGAR 10-20% operating regime replicates on
# a second contract-clause corpus. The within-corpus EUR-Lex test (run
# earlier today) showed the regime is NOT corpus-density-driven; CUAD
# is the proper cross-corpus contract-clause comparison.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   nohup bash scripts/run_cuad_plan.sh > outputs/logs/run_cuad.log 2>&1 &
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints data/processed_cuad

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

SEEDS=(42 43 44 45 46)

# =============================================================================
# STAGE 1 -- preprocess CUAD into JusDefLEDGAR pkl format
# =============================================================================

banner "STAGE 1: preprocess CUAD"

if ! have cuad_preprocess; then
  python -u scripts/preprocess_cuad.py 2>&1 | tee $LOG/cuad_preprocess.log
  done_run cuad_preprocess
fi

# Detect the number of clause-type classes
NUM_CLASSES=$(wc -l < data/processed_cuad/label_vocab.txt)
echo "CUAD num_classes detected: $NUM_CLASSES"

# =============================================================================
# STAGE 2 -- train v3 and mean baseline on CUAD, 5 seeds each
# =============================================================================

banner "STAGE 2: train v3 + mean baseline on CUAD"

for SEED in "${SEEDS[@]}"; do
  # mean baseline
  SENT_NAME="cuad_baseline_mean_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
  else
    echo ""
    echo "--- CUAD mean baseline seed $SEED ---"
    python -u scripts/train_ledgar.py \
      --seed $SEED \
      --tag cuad_baseline_mean \
      --dmp_variant mean \
      --data_dir data/processed_cuad \
      --num_classes $NUM_CLASSES \
      2>&1 | tee $LOG/cuad_baseline_mean_s${SEED}.log
    done_run $SENT_NAME
  fi

  # v3
  SENT_NAME="cuad_v3_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
  else
    echo ""
    echo "--- CUAD v3 seed $SEED ---"
    python -u scripts/train_ledgar.py \
      --seed $SEED \
      --tag cuad_v3 \
      --dmp_variant v3 \
      --data_dir data/processed_cuad \
      --num_classes $NUM_CLASSES \
      2>&1 | tee $LOG/cuad_v3_s${SEED}.log
    done_run $SENT_NAME
  fi
done

# =============================================================================
# STAGE 3 -- density-stratified evaluation on CUAD test
# =============================================================================

banner "STAGE 3: CUAD density-stratified evaluation"

if ! have cuad_density_eval; then
  python -u scripts/analyse_cuad_density_subset.py 2>&1 \
    | tee $LOG/cuad_density_eval.log
  done_run cuad_density_eval
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "CUAD PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== CUAD per-seed test_macro_f1 ==="
for VARIANT in cuad_baseline_mean cuad_v3; do
  echo "  $VARIANT:"
  for SEED in "${SEEDS[@]}"; do
    p="outputs/logs/ledgar_${VARIANT}_s${SEED}.json"
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
echo "=== CUAD CROSS-CORPUS REGIME VERDICT ==="
grep -A 20 "CUAD CROSS-CORPUS REGIME VERDICT" $LOG/cuad_density_eval.log 2>/dev/null | tail -20

echo ""
echo "All CUAD sentinels:"
ls -la outputs/sentinels/cuad_*.done 2>/dev/null
