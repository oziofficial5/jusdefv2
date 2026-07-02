#!/bin/bash
# =============================================================================
# Revision experiment: operators-ZEROED third arm of the permutation control.
# Defuses the reviewer challenge that the shuffled arm's +0.038 is operator
# PRESENCE, not content-free inductive bias. Cheap (frozen embeddings, ~1h/seed).
# Runs fine IN PARALLEL with the FT data-efficiency job (different checkpoints).
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   nohup bash scripts/run_thirdarm.sh > outputs/logs/thirdarm.log 2>&1 &
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
CKPT=outputs/checkpoints; LOG=outputs/logs; mkdir -p "$CKPT" "$LOG"

echo "=== build zeroed-operator corpus ==="
[[ -f data/processed_ledgar_zeroop/test_processed.pkl ]] \
  && echo "[skip] zeroed corpus exists" \
  || python -u scripts/make_zeroed_operators.py 2>&1 | tee "$LOG/make_zeroop.log"

echo "=== train v3 on zeroed operators (10 seeds) ==="
for s in $SEEDS; do
  if [[ -f "$CKPT/ledgar_v3_zeroop_s$s.pt" ]]; then
    echo "[skip] v3_zeroop s$s"
  else
    python -u scripts/train_ledgar.py --seed "$s" --tag v3_zeroop \
      --dmp_variant v3 --data_dir data/processed_ledgar_zeroop \
      2>&1 | tee "$LOG/ledgar_v3_zeroop_s$s.log"
  fi
done

echo "=== three-arm decomposition ==="
python -u scripts/analyse_thirdarm.py --seeds $SEEDS 2>&1 | tee "$LOG/analyse_thirdarm.log"
