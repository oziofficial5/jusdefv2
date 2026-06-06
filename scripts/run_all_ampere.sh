#!/bin/bash
# ============================================================================
# Full end-to-end Ampere pipeline for JusDef v2.
#
# This script runs:
#   STAGE 0: environment check
#   STAGE 1: preprocessing (sections, concepts, operators, authorities)
#   STAGE 2: embeddings (LegalBERT doc + section + label embs)
#   STAGE 3: EuroVoc label adjacency
#   STAGE 4: HeteroData graph build
#   STAGE 5: smoke test
#   STAGE 6: R-GCN baseline (3 seeds)
#   STAGE 7: JusDef main (3 seeds)
#   STAGE 8: JusDef ablations (3 seeds × 3 variants)
#   STAGE 9: final unified evaluation
#
# Each stage prints to stdout via `tee` so progress is visible in slurm logs.
# Set PYTHONUNBUFFERED=1 in your env to avoid block buffering.
#
# Resume policy: each stage writes sentinel files. If a sentinel exists,
# the stage is skipped. Delete a sentinel to force a stage rerun.
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
stage()  { echo ""; echo "=========================================="; echo "STAGE $1 — $2 — $(stamp)"; echo "=========================================="; }
done_stage() { touch "$SENT/$1.done"; echo "STAGE $1 done at $(stamp)"; }
have() { [[ -f "$SENT/$1.done" ]]; }

# ----------------------------------------------------------------------------
stage 0 "environment check"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"

# ----------------------------------------------------------------------------
if have 1; then echo "STAGE 1 already done, skipping"; else
stage 1 "preprocessing (sections, concepts, operators, authorities)"
python -u scripts/preprocess_all.py --device cuda 2>&1 | tee $LOG/stage1_preprocess.log
done_stage 1
fi

# ----------------------------------------------------------------------------
if have 2; then echo "STAGE 2 already done, skipping"; else
stage 2 "extract embeddings (LegalBERT doc + section)"
python -u scripts/extract_embeddings.py --device cuda 2>&1 | tee $LOG/stage2_embs.log
python -u scripts/make_label_embs.py 2>&1 | tee $LOG/stage2_label_embs.log
done_stage 2
fi

# ----------------------------------------------------------------------------
if have 3; then echo "STAGE 3 already done, skipping"; else
stage 3 "EuroVoc label adjacency"
python -u scripts/build_label_adj.py 2>&1 | tee $LOG/stage3_label_adj.log
done_stage 3
fi

# ----------------------------------------------------------------------------
if have 4; then echo "STAGE 4 already done, skipping"; else
stage 4 "build HeteroData graphs"
python -u scripts/build_graphs.py 2>&1 | tee $LOG/stage4_graphs.log
done_stage 4
fi

# ----------------------------------------------------------------------------
stage 5 "smoke test (must pass before training)"
python -u -m pytest tests/test_architecture_fixes.py -v 2>&1 | tee $LOG/stage5_pytest.log
python -u scripts/smoke_test_jusdef.py 2>&1 | tee $LOG/stage5_smoke.log
echo "STAGE 5 smoke passed."

# ----------------------------------------------------------------------------
if have 6; then echo "STAGE 6 already done, skipping"; else
stage 6 "R-GCN baseline (3 seeds)"
for SEED in 42 43 44; do
  echo "-- R-GCN seed $SEED --"
  python -u scripts/train_rgcn.py --seed $SEED 2>&1 | tee $LOG/stage6_rgcn_s${SEED}.log
done
done_stage 6
fi

# ----------------------------------------------------------------------------
if have 7; then echo "STAGE 7 already done, skipping"; else
stage 7 "JusDef main (3 seeds)"
for SEED in 42 43 44; do
  echo "-- JusDef full seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag full 2>&1 | tee $LOG/stage7_full_s${SEED}.log
done
done_stage 7
fi

# ----------------------------------------------------------------------------
if have 8; then echo "STAGE 8 already done, skipping"; else
stage 8 "JusDef ablations (3 seeds × 3 variants)"
for SEED in 42 43 44; do
  echo "-- no_dmp seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_dmp --no_dmp 2>&1 | tee $LOG/stage8_no_dmp_s${SEED}.log
  echo "-- no_auth seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_auth --no_authority 2>&1 | tee $LOG/stage8_no_auth_s${SEED}.log
  echo "-- no_dmp_no_auth seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_dmp_no_auth --no_dmp --no_authority 2>&1 | tee $LOG/stage8_no_dmp_no_auth_s${SEED}.log
done
done_stage 8
fi

# ----------------------------------------------------------------------------
stage 9 "final unified evaluation"
python -u scripts/eval_all_jusdef.py 2>&1 | tee $LOG/stage9_eval.log

echo ""
echo "=========================================="
echo "ALL STAGES COMPLETE — $(stamp)"
echo "=========================================="
ls -la outputs/logs/*.json
