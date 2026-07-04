#!/bin/bash
# =============================================================================
# Pivot B: JusDef operator prior on ContractNLI (defeat-as-label task).
# Tests whether operator-aware aggregation helps where exceptions flip the
# Entailment/Contradiction label --- the task the thesis says operators are for.
#
# PREREQUISITE (one-time, manual): download ContractNLI from
#   https://stanfordnlp.github.io/contract-nli/
# and place train.json / dev.json / test.json in data/contract_nli_raw/
#
# Launch:
#   cd ~/jusdefv2/jusdefv2 && git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   python scripts/preprocess_contractnli.py --debug   # smoke: 30 docs/split
#   nohup bash scripts/run_contractnli.sh > outputs/logs/contractnli.log 2>&1 &
#   tail -f outputs/logs/contractnli.log
# =============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SEEDS="${SEEDS:-42 43 44}"
CKPT=outputs/checkpoints; LOG=outputs/logs; mkdir -p "$CKPT" "$LOG"

echo "=== preprocess ContractNLI (embed + detect operators) ==="
if [[ -f data/processed_contractnli/test_processed.pkl ]]; then
  echo "[skip] already preprocessed"
else
  python -u scripts/preprocess_contractnli.py 2>&1 | tee "$LOG/cnli_preprocess.log"
fi

echo "=== train baseline (mean) and operator-aware (v3), seeds $SEEDS ==="
for agg in mean v3; do
  for s in $SEEDS; do
    if [[ -f "$LOG/cnli_${agg}_s$s.json" ]]; then
      echo "[skip] $agg s$s"
    else
      python -u scripts/train_contractnli.py --agg "$agg" --seed "$s" \
        2>&1 | tee "$LOG/cnli_${agg}_s$s.log"
    fi
  done
done

echo "=== comparison ==="
python -u scripts/analyse_contractnli.py --seeds $SEEDS 2>&1 | tee "$LOG/analyse_cnli.log"
