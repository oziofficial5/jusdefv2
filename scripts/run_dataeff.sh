#!/bin/bash
# =============================================================================
# Data-efficiency curve — the direct test of the paper's central thesis:
# "defeasibility-aware aggregation is an inductive-bias substitute for encoder
# adaptation." Prediction: the operator head helps MORE when training data is
# scarce (less data to adapt the encoder) and is subsumed when data is plentiful.
#
# For each training-set size we fine-tune the encoder with mean vs operator-aware
# (baseline-anchored) heads, identical recipe, and measure the operator benefit
# on aggregate and on the 10-20% regime. If Delta(ops - mean) rises as size falls,
# the inductive-bias-substitute claim is confirmed with a clean curve.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   SMOKE=1 bash scripts/run_dataeff.sh
#   nohup bash scripts/run_dataeff.sh > outputs/logs/dataeff.log 2>&1 &
#   tail -f outputs/logs/dataeff.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SIZES="${SIZES:-500 1500 5000 15000 60000}"   # 60000 = full LEDGAR train
SEEDS="${SEEDS:-42 43 44}"
AGGS="${AGGS:-mean v4_soft}"
EPOCHS="${EPOCHS:-10}"
CKPT=outputs/checkpoints; LOG=outputs/logs
mkdir -p "$CKPT" "$LOG"

EXTRA=""; SFX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SIZES="500 1500"; EPOCHS=1; SFX="_smoke"
    echo "*** SMOKE MODE (sizes $SIZES, 1 epoch, tag suffix '$SFX') ***"
fi

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "==================================="; echo " $1 -- $(stamp)"; echo "==================================="; }

for size in $SIZES; do
  banner "train size = $size"
  for agg in $AGGS; do
    bias=""; [[ "$agg" == "v4_soft" ]] && bias="--v4_soft_init_bias -5.0"
    for s in $SEEDS; do
      tag="ft_${agg}_n${size}${SFX}"
      if [[ -f "$CKPT/ledgar_${tag}_s$s.pt" ]]; then
        echo "[skip] $tag s$s"
      else
        python -u scripts/train_ledgar_ft.py --agg "$agg" --seed "$s" --tag "$tag" \
          --max_train "$size" --epochs "$EPOCHS" --patience 3 $bias \
          --delete_ckpt_after \
          2>&1 | tee "$LOG/ledgar_${tag}_s$s.log"
      fi
    done
  done
done

if [[ -z "$SFX" ]]; then
  banner "data-efficiency analysis"
  python -u scripts/analyse_dataeff.py --sizes $SIZES --seeds $SEEDS \
    2>&1 | tee "$LOG/analyse_dataeff.log"
fi
