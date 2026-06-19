#!/bin/bash
# =============================================================================
# v4 on EUR-Lex: the potential big win
#
# EUR-Lex non-AFF density is 0.71%. Under v4_hard with window [0.10, 0.20]:
#   ~99% of documents have per-concept density well below 0.10
#   -> almost every concept routes to the operator-agnostic SAGEConv path
#   -> v4_hard aggregate behaviour approximates R-GCN-equivalent
#   -> directly addresses the 9-point gap (v3 0.183 vs R-GCN 0.273)
#
# The training-time distribution-shift problem that broke v4_hard on LEDGAR
# is much milder here: classifier sees mean-like features almost everywhere,
# v3 contribution is a small minority perturbation that doesn't compete
# with the dominant feature distribution.
#
# Stages:
#   v4 LEDGAR diagnostic (5 min)         -- what did v4_soft learn?
#   v4_hard EUR-Lex 5 seeds (~6 hours)   -- the make-or-break experiment
#   v4_soft EUR-Lex 5 seeds (~6 hours)
#   Threshold sensitivity for v4_hard (seed 42, 2 alt windows; ~1.5 hours)
#   v4 EUR-Lex per-Yexc evaluation       -- via eval_all_jusdef.py
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py src/model/*.py src/train/*.py
#   nohup bash scripts/run_v4_eurlex_plan.sh > outputs/logs/run_v4_eurlex.log 2>&1 &
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints outputs/figures

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo ""; echo "============================================="; echo " $1 -- $(stamp)"; echo "============================================="; }
done_run() { touch "$SENT/$1.done"; echo "[done] $1 at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

# =============================================================================
# STAGE 0 -- LEDGAR v4_soft gate diagnostic (no GPU training, ~3 min)
# =============================================================================

banner "STAGE 0: v4_soft LEDGAR gate diagnostic"

if ! have v4_soft_gate_diag; then
  python -u scripts/analyse_v4_soft_gate.py 2>&1 | tee $LOG/v4_soft_gate_diag.log
  done_run v4_soft_gate_diag
fi

# =============================================================================
# STAGE 1 -- v4_hard on EUR-Lex (smoke test seed 42 + 5-seed)
# =============================================================================

banner "STAGE 1.1: v4_hard EUR-Lex smoke (seed 42)"

if ! have v4_hard_eurlex_s42; then
  echo ""
  echo "--- v4_hard EUR-Lex seed 42 (smoke + first multi-seed) ---"
  python -u scripts/train_jusdef.py \
    --seed 42 \
    --tag v4_hard_eurlex \
    --dmp_variant v4_hard \
    --v4_density_lo 0.10 \
    --v4_density_hi 0.20 \
    2>&1 | tee $LOG/v4_hard_eurlex_s42.log
  done_run v4_hard_eurlex_s42
fi

banner "STAGE 1.2: v4_hard EUR-Lex remaining seeds (43, 44, 45, 46)"

for SEED in 43 44 45 46; do
  SENT_NAME="v4_hard_eurlex_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- v4_hard EUR-Lex seed $SEED ---"
  python -u scripts/train_jusdef.py \
    --seed $SEED \
    --tag v4_hard_eurlex \
    --dmp_variant v4_hard \
    --v4_density_lo 0.10 \
    --v4_density_hi 0.20 \
    2>&1 | tee $LOG/v4_hard_eurlex_s${SEED}.log
  done_run $SENT_NAME
done

# =============================================================================
# STAGE 2 -- v4_soft on EUR-Lex (5 seeds)
# =============================================================================

banner "STAGE 2: v4_soft EUR-Lex 5 seeds"

for SEED in 42 43 44 45 46; do
  SENT_NAME="v4_soft_eurlex_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- v4_soft EUR-Lex seed $SEED ---"
  python -u scripts/train_jusdef.py \
    --seed $SEED \
    --tag v4_soft_eurlex \
    --dmp_variant v4_soft \
    2>&1 | tee $LOG/v4_soft_eurlex_s${SEED}.log
  done_run $SENT_NAME
done

# =============================================================================
# STAGE 3 -- v4_hard threshold sensitivity on EUR-Lex (seed 42, 2 windows)
# =============================================================================

banner "STAGE 3: v4_hard EUR-Lex threshold sensitivity"

for SPEC in \
  "v4_hard_eurlex_w5_15 0.05 0.15" \
  "v4_hard_eurlex_w15_30 0.15 0.30"; do
  read -r NAME LO HI <<< "$SPEC"
  SENT_NAME="${NAME}_s42"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""
  echo "--- $NAME seed 42 (window $LO - $HI) ---"
  python -u scripts/train_jusdef.py \
    --seed 42 \
    --tag $NAME \
    --dmp_variant v4_hard \
    --v4_density_lo $LO \
    --v4_density_hi $HI \
    2>&1 | tee $LOG/${NAME}_s42.log
  done_run $SENT_NAME
done

# =============================================================================
# STAGE 4 -- evaluate all EUR-Lex checkpoints (including new v4) on Y_exc
# =============================================================================

banner "STAGE 4: EUR-Lex Y_exc evaluation across all checkpoints"

if ! have v4_eurlex_yexc_eval; then
  python -u scripts/eval_all_jusdef.py 2>&1 | tee $LOG/v4_eurlex_yexc_eval.log
  done_run v4_eurlex_yexc_eval
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "V4 EUR-LEX PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== v4_hard EUR-Lex test_macro_f1 (5 seeds) ==="
for SEED in 42 43 44 45 46; do
  p="outputs/logs/jusdef_v4_hard_eurlex_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
m = d.get('test_macro_f1') or d.get('test_metrics', {}).get('macro_f1')
print(f'  s$SEED: test_macro_f1={m:.4f}' if m else f'  s$SEED: no test_macro_f1 field')
" 2>/dev/null || echo "  s$SEED: unparseable"
  else
    echo "  s$SEED: MISSING"
  fi
done

echo ""
echo "=== v4_soft EUR-Lex test_macro_f1 (5 seeds) ==="
for SEED in 42 43 44 45 46; do
  p="outputs/logs/jusdef_v4_soft_eurlex_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
m = d.get('test_macro_f1') or d.get('test_metrics', {}).get('macro_f1')
print(f'  s$SEED: test_macro_f1={m:.4f}' if m else f'  s$SEED: no test_macro_f1 field')
" 2>/dev/null || echo "  s$SEED: unparseable"
  else
    echo "  s$SEED: MISSING"
  fi
done

echo ""
echo "=== v4_hard EUR-Lex threshold sensitivity (seed 42) ==="
for NAME in v4_hard_eurlex v4_hard_eurlex_w5_15 v4_hard_eurlex_w15_30; do
  p="outputs/logs/jusdef_${NAME}_s42.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
m = d.get('test_macro_f1') or d.get('test_metrics', {}).get('macro_f1')
print(f'  ${NAME}: test_macro_f1={m:.4f}' if m else f'  ${NAME}: no test_macro_f1 field')
" 2>/dev/null || echo "  ${NAME}: unparseable"
  else
    echo "  ${NAME}: MISSING"
  fi
done

echo ""
echo "Critical numbers to read first:"
echo "  - v4_hard EUR-Lex 5-seed mean (should match R-GCN ~0.273 if port works)"
echo "  - Y_exc evaluation: v4 vs v3 vs R-GCN on the exception-dependent labels"
echo "All sentinels:"
ls -la outputs/sentinels/v4_*eurlex*.done outputs/sentinels/v4_soft_gate_diag.done 2>/dev/null | head -30
