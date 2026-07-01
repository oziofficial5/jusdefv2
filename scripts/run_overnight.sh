#!/bin/bash
# =============================================================================
# Overnight batch — finish the F2 story + F4 + baseline confirmation.
# Run it, go to sleep, read the analyses in the morning. Fully RESUMABLE:
# every train step is skipped if its checkpoint already exists, so if it dies
# or you run out of time, just launch it again and it continues.
#
# Priority order (most important first — if the night is short, the top gets done):
#   P1  FT+v3 (ungated) + extend FT+mean/FT+v4_soft to 5 seeds  -> hardens "FT subsumes operators"
#   P2  F4: FT+v4_soft with auxiliary operator loss (3 seeds)    -> last lever
#   P3  paragraph-level LegalBERT-FT baseline (3 seeds)          -> confirms 0.817 > ~0.80
#   D   analyses (FT comparison + F4 comparison)
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   SMOKE=1 bash scripts/run_overnight.sh          # 10-min wiring check (do this once)
#   nohup bash scripts/run_overnight.sh > outputs/logs/overnight.log 2>&1 &
#   tail -f outputs/logs/overnight.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

FT_SEEDS="${FT_SEEDS:-42 43 44 45 46}"     # for P1 (FT ablation)
AUX_SEEDS="${AUX_SEEDS:-42 43 44}"         # for P2 (F4) and P3 (baseline)
CKPT=outputs/checkpoints
LOG=outputs/logs
mkdir -p "$CKPT" "$LOG"

EXTRA=""; SFX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="--max_train 1500 --epochs 1"; SFX="_smoke"
    echo "*** SMOKE MODE (throwaway, tag suffix '$SFX') ***"
fi

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }

ft() {  # ft <agg> <seed> <tag> [extra args...]
    local agg="$1" seed="$2" tag="$3"; shift 3
    if [[ -f "$CKPT/ledgar_${tag}${SFX}_s$seed.pt" ]]; then
        echo "[skip] ${tag}${SFX} s$seed"
    else
        python -u scripts/train_ledgar_ft.py --agg "$agg" --seed "$seed" \
            --tag "${tag}${SFX}" "$@" $EXTRA 2>&1 | tee "$LOG/ledgar_${tag}${SFX}_s$seed.log"
    fi
}

banner "PRE-FLIGHT"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
test -f data/processed_ledgar/test_processed.pkl || { echo "ERROR: run run_pilot_ledgar.sh first"; exit 1; }

# ---- P1: FT ablation, 5 seeds, three heads --------------------------------
banner "P1  FT+{mean,v3,v4_soft} across seeds: $FT_SEEDS"
for s in $FT_SEEDS; do
    ft mean    "$s" "ft_mean"
    ft v3      "$s" "ft_v3"
    ft v4_soft "$s" "ft_v4_soft" --v4_soft_init_bias -5.0
done

# ---- P2: F4 multi-task (auxiliary operator loss) --------------------------
banner "P2  F4 FT+v4_soft + aux operator loss (0.3) across seeds: $AUX_SEEDS"
for s in $AUX_SEEDS; do
    ft v4_soft "$s" "ft_v4_soft_aux" --v4_soft_init_bias -5.0 --aux_op_weight 0.3
done

# ---- P3: paragraph-level LegalBERT-FT baseline ----------------------------
banner "P3  paragraph-level LegalBERT-FT baseline: $AUX_SEEDS"
for s in $AUX_SEEDS; do
    if [[ -f "$CKPT/legalbert_ledgar_legalbert_ft${SFX}_s$s.pt" ]]; then
        echo "[skip] legalbert_ft s$s"
    elif [[ -n "$SFX" ]]; then
        echo "[smoke] skipping legalbert baseline"
    else
        python -u scripts/train_legalbert_ledgar.py --seed "$s" --tag legalbert_ft \
            2>&1 | tee "$LOG/legalbert_ledgar_ft_s$s.log"
    fi
done

# ---- D: analyses ----------------------------------------------------------
banner "D  analyses"
if [[ -n "$SFX" ]]; then
    echo "[smoke] skipping analyses"
else
    echo "--- FT ablation: FT+ops vs FT+mean (aggregate + 10-20%) ---"
    python -u scripts/analyse_ft_compare.py --aggs mean v3 v4_soft --ref mean \
        --seeds $FT_SEEDS 2>&1 | tee "$LOG/analyse_ft_compare.log" || true
    echo "--- F4: does aux operator loss help? (v4_soft_aux vs v4_soft) ---"
    python -u scripts/analyse_ft_compare.py --aggs v4_soft v4_soft_aux --ref v4_soft \
        --seeds $AUX_SEEDS 2>&1 | tee "$LOG/analyse_f4.log" || true
fi

banner "OVERNIGHT DONE — download before the booking ends"
echo "  scp -r milan:~/jusdefv2/jusdefv2/outputs/logs ./outputs/"
