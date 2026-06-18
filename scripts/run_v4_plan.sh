#!/bin/bash
# =============================================================================
# 3-day v4 plan: density-routed v3
#
# Builds on the empirical operating regime discovered in Chapter 8:
# v3 wins on the 10-20% non-AFF density bin and loses elsewhere.
# v4 routes per-paragraph to v3 (when density is in [lo, hi)) or to mean
# (otherwise). Two variants:
#
#   v4_hard: inference-time hard gate on observable density
#            (default routing window [0.10, 0.20])
#   v4_soft: learned per-paragraph soft gate from a 3-feature MLP,
#            initialised so initial gate ~ 1 (training starts as v3 and
#            learns to suppress v3 on paragraphs where mean is better)
#
# Expected outcome:
#   aggregate LEDGAR macro-F1:  v3 0.684 -> v4 ~0.71 (mean-baseline level)
#   10-20% bin macro-F1:        v3 0.682 -> v4 ~0.68 (regime gain preserved)
#
# Stages:
#   DAY 1 (~30 min): smoke test seed 42 for v4_hard + v4_soft, then 5-seed
#                    v4_hard runs
#   DAY 2 (~30 min): 5-seed v4_soft runs + threshold sensitivity for v4_hard
#                    at three alternative windows (8-22, 10-20, 12-18)
#   DAY 3 (~10 min): v4 per-regime stratified analysis (analyse_ledgar_v4.py)
#
# Sentinels: outputs/sentinels/v4_*.done -- re-runs skip completed work.
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py
#   nohup bash scripts/run_v4_plan.sh > outputs/logs/run_v4.log 2>&1 &
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

SEEDS=(42 43 44 45 46)

# =============================================================================
# DAY 1.1 -- smoke tests on seed 42 (catch any forward-pass bugs early)
# =============================================================================

banner "DAY 1.1: v4 smoke tests on seed 42"

if ! have v4_smoke_hard_s42; then
  echo ""
  echo "--- Smoke: v4_hard seed 42 ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v4_hard \
    --dmp_variant v4_hard \
    --v4_density_lo 0.10 \
    --v4_density_hi 0.20 \
    2>&1 | tee $LOG/v4_hard_s42.log
  done_run v4_smoke_hard_s42
fi

if ! have v4_smoke_soft_s42; then
  echo ""
  echo "--- Smoke: v4_soft seed 42 ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag v4_soft \
    --dmp_variant v4_soft \
    2>&1 | tee $LOG/v4_soft_s42.log
  done_run v4_smoke_soft_s42
fi

# =============================================================================
# DAY 1.2 -- v4_hard 5-seed (seed 42 reuses the smoke checkpoint)
# =============================================================================

banner "DAY 1.2: v4_hard 5-seed"

for SEED in 43 44 45 46; do
  SENT_NAME="v4_hard_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- v4_hard seed $SEED ---"
  python -u scripts/train_ledgar.py \
    --seed $SEED \
    --tag v4_hard \
    --dmp_variant v4_hard \
    --v4_density_lo 0.10 \
    --v4_density_hi 0.20 \
    2>&1 | tee $LOG/v4_hard_s${SEED}.log
  done_run $SENT_NAME
done

# =============================================================================
# DAY 2.1 -- v4_soft 5-seed (seed 42 reuses the smoke checkpoint)
# =============================================================================

banner "DAY 2.1: v4_soft 5-seed"

for SEED in 43 44 45 46; do
  SENT_NAME="v4_soft_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- v4_soft seed $SEED ---"
  python -u scripts/train_ledgar.py \
    --seed $SEED \
    --tag v4_soft \
    --dmp_variant v4_soft \
    2>&1 | tee $LOG/v4_soft_s${SEED}.log
  done_run $SENT_NAME
done

# =============================================================================
# DAY 2.2 -- v4_hard threshold sensitivity (seed 42 only, three windows)
# =============================================================================

banner "DAY 2.2: v4_hard threshold sensitivity"

# Format: NAME LO HI
for SPEC in \
  "v4_hard_w8_22 0.08 0.22" \
  "v4_hard_w12_18 0.12 0.18"; do
  read -r NAME LO HI <<< "$SPEC"
  SENT_NAME="${NAME}_s42"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- $NAME seed 42 (window $LO - $HI) ---"
  python -u scripts/train_ledgar.py \
    --seed 42 \
    --tag $NAME \
    --dmp_variant v4_hard \
    --v4_density_lo $LO \
    --v4_density_hi $HI \
    2>&1 | tee $LOG/${NAME}_s42.log
  done_run $SENT_NAME
done

# =============================================================================
# DAY 3 -- v4 per-regime stratified analysis
# =============================================================================

banner "DAY 3: v4 per-regime stratified evaluation"

if ! have v4_analysis; then
  python -u scripts/analyse_ledgar_v4.py 2>&1 | tee $LOG/v4_analysis.log
  done_run v4_analysis
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "V4 PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== v4_hard 5-seed test_macro_f1 (full test set) ==="
for SEED in 42 43 44 45 46; do
  p="outputs/logs/ledgar_v4_hard_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "  s$SEED: unparseable"
  else
    echo "  s$SEED: MISSING"
  fi
done

echo ""
echo "=== v4_soft 5-seed test_macro_f1 (full test set) ==="
for SEED in 42 43 44 45 46; do
  p="outputs/logs/ledgar_v4_soft_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "  s$SEED: unparseable"
  else
    echo "  s$SEED: MISSING"
  fi
done

echo ""
echo "=== v4_hard threshold sensitivity (seed 42) ==="
for NAME in v4_hard v4_hard_w8_22 v4_hard_w12_18; do
  p="outputs/logs/ledgar_${NAME}_s42.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  ${NAME}: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "  ${NAME}: unparseable"
  else
    echo "  ${NAME}: MISSING"
  fi
done

echo ""
echo "Critical: read the V4 VERDICT block at outputs/logs/v4_analysis.log"
echo "All sentinels:"
ls -la outputs/sentinels/v4_*.done 2>/dev/null | head -30
