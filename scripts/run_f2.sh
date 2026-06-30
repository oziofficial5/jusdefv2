#!/bin/bash
# =============================================================================
# F2 — fine-tuned encoder + operator-aware head (Days 3-4 of the plan).
#
# Puts LegalBERT in the training loop (no frozen-embedding ceiling) and asks:
# does an operator-aware aggregation head beat FT+mean ON TOP of a fine-tuned
# encoder, especially in the 10-20% regime?
#
# Clean ablation — same encoder, same data, only the aggregation head changes:
#   AGGS="mean v4_soft"   (FT+mean control vs FT+baseline-anchored gate)
# Add v3 with: AGGS="mean v3 v4_soft"
#
# FT is expensive (~1-2 h/run on A100), so default is 3 seeds. Each run also
# logs the 10-20% bin macro-F1 directly.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   SMOKE=1 bash scripts/run_f2.sh           # wiring check (tiny train, 1 epoch)
#   bash scripts/run_f2.sh 2>&1 | tee outputs/logs/run_f2.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEEDS="${SEEDS:-42 43 44}"
AGGS="${AGGS:-mean v4_soft}"
CKPT=outputs/checkpoints
LOG=outputs/logs
mkdir -p "$CKPT" "$LOG"

EXTRA=""; SFX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="--max_train 1500 --epochs 1"; SFX="_smoke"
    echo "*** SMOKE MODE (throwaway, tag suffix '$SFX') ***"
fi

echo "F2 fine-tuned encoder | aggs: $AGGS | seeds: $SEEDS"
for agg in $AGGS; do
    for s in $SEEDS; do
        tag="ft_${agg}${SFX}"
        if [[ -f "$CKPT/ledgar_${tag}_s$s.pt" ]]; then
            echo "[skip] $tag s$s"
        else
            python -u scripts/train_ledgar_ft.py --agg "$agg" --seed "$s" \
                --tag "$tag" $EXTRA 2>&1 | tee "$LOG/ledgar_${tag}_s$s.log"
        fi
    done
done

if [[ -z "$SFX" ]]; then
    echo ""
    echo "=== F2 comparison (FT+ops vs FT+mean, aggregate + 10-20% regime) ==="
    python -u scripts/analyse_ft_compare.py --aggs $AGGS --ref mean --seeds $SEEDS \
        2>&1 | tee "$LOG/analyse_ft_compare.log"
else
    echo "[smoke] skipping comparison"
fi
