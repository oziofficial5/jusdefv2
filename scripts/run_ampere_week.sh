#!/bin/bash
# =============================================================================
# Ampere week — harden the LEDGAR operating-regime claim for the viva.
#
# Assumes the LEDGAR corpus is already preprocessed at data/processed_ledgar/
# (run scripts/run_pilot_ledgar.sh first if not). LEDGAR runs are ~1-1.5h/seed,
# so a week buys many seeds + the decisive operator-permutation control.
#
#   P0  build operator-permutation control corpus            (seconds)
#   P2  mean + v3 (real ops) across all seeds                (10-seed regime CI)
#   P1  v3 trained on SHUFFLED operators across all seeds    (the decisive control)
#   P3  v4_twostage across all seeds                         (double-win robustness)
#   D   analyses: density subset, shuffle control, v4 bootstrap
#
# Resumable: each train step is skipped if its checkpoint already exists.
#
# Launch:
#   cd ~/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py src/model/*.py
#   SMOKE=1 bash scripts/run_ampere_week.sh        # fast dry-run first (verify wiring)
#   bash scripts/run_ampere_week.sh 2>&1 | tee outputs/logs/ampere_week.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
LOG=outputs/logs
CKPT=outputs/checkpoints
SENT=outputs/sentinels
mkdir -p "$LOG" "$CKPT" "$SENT"

# Smoke mode: tiny + fast, just to confirm the pipeline runs end to end.
EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="--max_train 2000 --epochs 3"
    echo "*** SMOKE MODE: $EXTRA (results are throwaway) ***"
fi

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }

# ---- pre-flight ------------------------------------------------------------
banner "PRE-FLIGHT"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
test -f data/processed_ledgar/test_processed.pkl || {
    echo "ERROR: data/processed_ledgar not found. Run scripts/run_pilot_ledgar.sh first."; exit 1; }

# ---- P0: build the operator-permutation control corpus ---------------------
banner "P0  build shuffled-operator control corpus"
if [[ -f data/processed_ledgar_shuffleop/test_processed.pkl ]]; then
    echo "[skip] shuffled corpus exists"
else
    python -u scripts/make_shuffled_operators.py 2>&1 | tee "$LOG/make_shuffleop.log"
fi

# ---- P2: mean baseline + v3 (real operators), all seeds --------------------
banner "P2  mean + v3 (real ops) across seeds: $SEEDS"
for s in $SEEDS; do
    if [[ -f "$CKPT/ledgar_baseline_mean_s$s.pt" ]]; then
        echo "[skip] mean s$s"
    else
        python -u scripts/train_ledgar.py --seed "$s" --tag baseline_mean \
            --dmp_variant mean $EXTRA 2>&1 | tee "$LOG/ledgar_baseline_mean_s$s.log"
    fi
    if [[ -f "$CKPT/ledgar_v3_pilot_s$s.pt" ]]; then
        echo "[skip] v3 s$s"
    else
        python -u scripts/train_ledgar.py --seed "$s" --tag v3_pilot \
            --dmp_variant v3 $EXTRA 2>&1 | tee "$LOG/ledgar_v3_pilot_s$s.log"
    fi
done

# ---- P1: v3 on SHUFFLED operators, all seeds (the decisive control) --------
banner "P1  v3 on SHUFFLED operators across seeds: $SEEDS"
for s in $SEEDS; do
    if [[ -f "$CKPT/ledgar_v3_shuffleop_s$s.pt" ]]; then
        echo "[skip] v3_shuffleop s$s"
    else
        python -u scripts/train_ledgar.py --seed "$s" --tag v3_shuffleop \
            --dmp_variant v3 --data_dir data/processed_ledgar_shuffleop $EXTRA \
            2>&1 | tee "$LOG/ledgar_v3_shuffleop_s$s.log"
    fi
done

# ---- P3: v4_twostage, all seeds (needs the real v3 checkpoint) -------------
banner "P3  v4_twostage across seeds: $SEEDS"
for s in $SEEDS; do
    if [[ -f "$SENT/v4_twostage_s$s.done" ]]; then
        echo "[skip] v4_twostage s$s"
    else
        python -u scripts/train_v4_twostage.py --seed "$s" --tag v4_twostage \
            --pretrained_v3 "$CKPT/ledgar_v3_pilot_s$s.pt" \
            2>&1 | tee "$LOG/ledgar_v4_twostage_s$s.log"
        touch "$SENT/v4_twostage_s$s.done"
    fi
done

# ---- D: analyses -----------------------------------------------------------
banner "D  analyses"
echo "--- single-seed density-stratified (seed 42 checkpoints) ---"
python -u scripts/analyse_ledgar_density_subset.py 2>&1 | tee "$LOG/analyse_density_subset.log" || true
echo "--- multi-seed OPERATOR-PERMUTATION CONTROL (the headline result) ---"
python -u scripts/analyse_ledgar_shuffle_control.py --seeds $SEEDS 2>&1 | tee "$LOG/analyse_shuffle_control.log" || true
echo "--- v4_twostage bootstrap ---"
python -u scripts/bootstrap_v4_twostage.py 2>&1 | tee "$LOG/bootstrap_v4.log" || true

banner "DONE — remember to download results before the booking ends"
echo "  scp -r milan:~/jusdefv2/outputs/logs ./outputs/"
echo "  scp -r milan:~/jusdefv2/outputs/checkpoints ./outputs/"
