#!/bin/bash
# Single-seed v3 pilot to determine whether the v3 architecture closes the
# v2-vs-R-GCN gap on EUR-Lex (keyword graphs).
#
# Usage:
#   bash scripts/run_pilot_v3.sh
#
# This runs ONE seed (42) on the EXISTING keyword-operator graphs. It does
# not touch any existing v2/v3 checkpoints. Pilot writes to:
#   outputs/checkpoints/jusdef_v3_pilot_s42.pt
#   outputs/logs/jusdef_v3_pilot_s42.json

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/checkpoints outputs/logs

echo "================================================="
echo " v3 PILOT — single-seed, EUR-Lex keyword graphs"
echo "================================================="
echo "Expected runtime: ~8-10 hours on A100"
echo "Comparison points (from prior runs):"
echo "  R-GCN h=512 mean (3 seeds):     test_macro = 0.2731"
echo "  JusDef v2 (hard DMP) mean:      test_macro = 0.1822"
echo ""
echo "Decision rule after pilot:"
echo "  v3 test_macro >= 0.25  -> 3-seed v3 evaluation justified"
echo "  v3 test_macro 0.20-0.25 -> v3 helps but ablate to find best variant"
echo "  v3 test_macro < 0.20    -> diagnose v3 (likely sign-cancellation regulariser)"
echo ""

python -u scripts/train_jusdef.py \
    --seed 42 \
    --tag v3_pilot \
    --dmp_variant v3 \
    --graph_dir data/processed/graphs \
    2>&1 | tee outputs/logs/v3_pilot_$(date +%Y%m%d-%H%M).log

echo ""
echo "================================================="
echo " v3 PILOT COMPLETE"
echo "================================================="
ls -la outputs/logs/jusdef_v3_pilot_s42.json 2>/dev/null && \
    cat outputs/logs/jusdef_v3_pilot_s42.json
