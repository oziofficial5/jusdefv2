#!/bin/bash
# ============================================================================
# Ampere job: extend the fine-tuned LEDGAR experiments to 10 seeds and
# decompose the 10-20% regime gain by clause type.
#
#   Stage FT  : fine-tuned encoder, seeds 47-51, aggs {mean, v3, v4_soft}
#               (seeds 42-46 assumed already present from the earlier run)
#   Stage AUX : v4_soft + auxiliary operator loss, seeds 47-51 (completes the
#               "aux operator loss" row to 10 seeds; optional but cheap)
#   Stage CMP : aggregate the 10-seed FT comparison (no GPU)
#   Stage PC  : per-class decomposition of the regime gain
#               (needs frozen v3_pilot / baseline_mean checkpoints s42-46)
#
# Each unit writes a sentinel in outputs/sentinels/ and is skipped on rerun.
# Delete a sentinel to force that unit to run again.
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# cd to the repo root (the directory containing this script's parent), so the
# job works regardless of where it is launched from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "repo root: $(pwd)"

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p "$LOG" "$SENT" outputs/checkpoints

have() { [[ -f "$SENT/$1.done" ]]; }
mark() { touch "$SENT/$1.done"; }
stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "== FT-to-10-seeds + per-class :: start $(stamp) =="
python -c "import torch; print('cuda', torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# --- Stage FT: seeds 47-51 x {mean, v3, v4_soft} ---------------------------
for seed in 47 48 49 50 51; do
  for agg in mean v3 v4_soft; do
    key="ft_${agg}_s${seed}"
    if have "$key"; then echo "skip $key"; continue; fi
    echo "== train $key :: $(stamp) =="
    python -u scripts/train_ledgar_ft.py --agg "$agg" --seed "$seed" \
      2>&1 | tee "$LOG/${key}.log"
    mark "$key"
  done
done

# --- Stage AUX: v4_soft + auxiliary operator loss, seeds 47-51 -------------
for seed in 47 48 49 50 51; do
  key="ft_v4soft_aux_s${seed}"
  if have "$key"; then echo "skip $key"; continue; fi
  echo "== train $key :: $(stamp) =="
  python -u scripts/train_ledgar_ft.py --agg v4_soft --seed "$seed" \
    --aux_op_weight 0.1 --tag ft_v4soft_aux 2>&1 | tee "$LOG/${key}.log"
  mark "$key"
done

# --- Stage CMP: 10-seed FT comparison (no GPU) -----------------------------
echo "== FT comparison over 10 seeds :: $(stamp) =="
python -u scripts/analyse_ft_compare.py --aggs mean v3 v4_soft \
  --seeds 42 43 44 45 46 47 48 49 50 51 2>&1 | tee "$LOG/ft_compare_10seed.log"

# --- Stage PC: per-class decomposition of the regime gain ------------------
if [[ ! -f "$LOG/predictions_10_20_bin.json" ]]; then
  echo "== dump 10-20% predictions (frozen models) :: $(stamp) =="
  python -u scripts/dump_predictions_10_20.py 2>&1 | tee "$LOG/dump_pred_10_20.log" \
    || echo "WARN: dump_predictions failed (frozen checkpoints missing?); skipping per-class"
fi
if [[ -f "$LOG/predictions_10_20_bin.json" ]]; then
  echo "== per-class regime-gain decomposition :: $(stamp) =="
  python -u scripts/analyse_perclass_regime.py 2>&1 | tee "$LOG/perclass_regime.log"
fi

echo "== done :: $(stamp) =="
