#!/bin/bash
# End-to-end LEDGAR pilot:
#  1. Preprocess LEDGAR train/val/test (embed + detect operators)
#  2. Run baseline (mean aggregation, no operator awareness)
#  3. Run v3 pilot
#  4. Print comparison summary
#
# Expected runtime on A100:
#  - Preprocess: ~3-5 hours (60K train + 10K val + 10K test paragraphs)
#  - Baseline:   ~1 hour (lighter model, single seed)
#  - v3 pilot:   ~1.5 hours
# Total: ~5-7 hours
#
# Decision rule after pilot:
#  - v3 > baseline by >= 1 F1 point (macro)  -> 3-seed evaluation justified
#  - v3 > baseline by 0.3-1 F1                -> marginal; ablations clarify
#  - v3 <= baseline                            -> revisit architecture / hyperparams

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints

echo "=========================================="
echo " LEDGAR PILOT PIPELINE"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo "Density context (from prior measurement):"
echo "  EUR-Lex:  0.71% non-AFF"
echo "  ECtHR:    0.58% non-AFF"
echo "  LEDGAR:  20.03% non-AFF (28x denser)"
echo ""

# Stage 1: preprocess LEDGAR
if [[ -f $SENT/ledgar_preprocess.done ]]; then
    echo "[SKIP] LEDGAR preprocessing already done"
else
    echo "=== Stage 1: preprocess LEDGAR ==="
    python -u scripts/preprocess_ledgar.py 2>&1 | tee $LOG/ledgar_preprocess.log
    touch $SENT/ledgar_preprocess.done
fi

# Stage 2: baseline (mean aggregation, no operator awareness)
if [[ -f $SENT/ledgar_baseline_s42.done ]]; then
    echo "[SKIP] LEDGAR mean baseline already done"
else
    echo ""
    echo "=== Stage 2: baseline (mean aggregation, seed 42) ==="
    python -u scripts/train_ledgar.py \
        --seed 42 \
        --tag baseline_mean \
        --dmp_variant mean \
        2>&1 | tee $LOG/ledgar_baseline_s42.log
    touch $SENT/ledgar_baseline_s42.done
fi

# Stage 3: v3 pilot
if [[ -f $SENT/ledgar_v3_s42.done ]]; then
    echo "[SKIP] LEDGAR v3 pilot already done"
else
    echo ""
    echo "=== Stage 3: V3 pilot (operator-aware, seed 42) ==="
    python -u scripts/train_ledgar.py \
        --seed 42 \
        --tag v3_pilot \
        --dmp_variant v3 \
        2>&1 | tee $LOG/ledgar_v3_s42.log
    touch $SENT/ledgar_v3_s42.done
fi

echo ""
echo "=========================================="
echo " LEDGAR PILOT COMPLETE"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo ""
echo "=== Results comparison ==="
python -c "
import json
import os
for tag in ['baseline_mean', 'v3_pilot']:
    p = f'outputs/logs/ledgar_{tag}_s42.json'
    if os.path.exists(p):
        d = json.load(open(p))
        print(f'  {tag:<16} test_macro={d[\"test_macro_f1\"]:.4f} test_micro={d[\"test_micro_f1\"]:.4f}')
    else:
        print(f'  {tag:<16} MISSING')
"
