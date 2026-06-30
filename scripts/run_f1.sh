#!/bin/bash
# =============================================================================
# F1 — residual / baseline-anchored soft gate.
#
# Mechanism: v4_soft already gates the v3 update against the mean baseline via
# h_para = h_mean + alpha * (h_v3 - h_mean), with alpha a learned per-paragraph
# sigmoid. The stock init_bias=+5.0 starts alpha~1 (full v3). We flip it to
# -5.0 so alpha~0 at init: the model STARTS as the mean baseline and learns to
# switch the operator update ON only where it reduces loss. No model code change.
#
# Hypothesis: this removes v3's aggregate loss (gate -> 0 off-regime) while
# keeping the 10-20% regime win (gate -> 1 in-regime).
#
# Success metric (printed by analyse_ledgar_variant.py):
#   aggregate Δ(variant - mean) >= 0  AND  regime 10-20% Δ > 0
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   SMOKE=1 bash scripts/run_f1.sh        # wiring check
#   bash scripts/run_f1.sh 2>&1 | tee outputs/logs/run_f1.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
TAG="v4_soft_anchor"
INIT_BIAS="-5.0"
CKPT=outputs/checkpoints
LOG=outputs/logs
mkdir -p "$CKPT" "$LOG"

EXTRA=""; SFX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="--max_train 2000 --epochs 3"; SFX="_smoke"
    echo "*** SMOKE MODE (throwaway, tag suffix '$SFX') ***"
fi

echo "F1 baseline-anchored gate: dmp_variant=v4_soft, init_bias=$INIT_BIAS, seeds: $SEEDS"
for s in $SEEDS; do
    if [[ -f "$CKPT/ledgar_${TAG}${SFX}_s$s.pt" ]]; then
        echo "[skip] ${TAG}${SFX} s$s"
    else
        python -u scripts/train_ledgar.py --seed "$s" --tag "${TAG}${SFX}" \
            --dmp_variant v4_soft --v4_soft_init_bias "$INIT_BIAS" $EXTRA \
            2>&1 | tee "$LOG/ledgar_${TAG}${SFX}_s$s.log"
    fi
done

if [[ -z "$SFX" ]]; then
    echo ""
    echo "=== F1 analysis: ${TAG} vs mean (aggregate + 10-20% regime) ==="
    python -u scripts/analyse_ledgar_variant.py --tag "$TAG" --variant v4_soft \
        --init_bias "$INIT_BIAS" --seeds $SEEDS 2>&1 | tee "$LOG/analyse_${TAG}.log"
else
    echo "[smoke] skipping analysis"
fi
