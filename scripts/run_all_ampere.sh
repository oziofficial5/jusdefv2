#!/bin/bash
# ============================================================================
# Full end-to-end Ampere pipeline for JusDef v2.
#
# Stages:
#   0   environment check
#   1   preprocessing (sections, concepts, KEYWORD operators, authorities)
#   2   embeddings (LegalBERT doc + section + label)
#   3   EuroVoc label adjacency
#   4   build HeteroData graphs (keyword-operator graphs)
#   5   smoke test (pytest + 5-epoch sanity)
#   6   R-GCN baseline x 3 seeds
#   6.5 R-GCN capacity-match at hidden_dim=768 (1 seed)
#   7   JusDef main x 3 seeds (full)
#   7.5 Neural operator detector pipeline:
#           train detector -> relabel processed -> rebuild graphs -> JusDef seed 42
#       (set --jusdef_neural_seeds in env to override)
#   8   JusDef ablations x 3 seeds x 3 variants
#   9   final unified evaluation
#
# Sentinels: each stage writes outputs/sentinels/<n>.done. Re-running the
# script skips completed stages. To force a rerun, delete the sentinel.
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p $LOG $SENT outputs/checkpoints

# How many seeds to run the neural-detector JusDef on. Default 1 (seed 42).
# If you have spare compute, export JUSDEF_NEURAL_SEEDS="42 43 44" before submission.
: "${JUSDEF_NEURAL_SEEDS:=42}"

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
stage() { echo ""; echo "=========================================="; echo "STAGE $1 -- $2 -- $(stamp)"; echo "=========================================="; }
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
stage 1 "preprocessing (sections, concepts, KEYWORD operators, authorities)"
python -u scripts/preprocess_all.py --device cuda 2>&1 | tee $LOG/stage1_preprocess.log
done_stage 1
fi

# ----------------------------------------------------------------------------
if have 2; then echo "STAGE 2 already done, skipping"; else
stage 2 "extract embeddings (LegalBERT doc + section + label)"
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
stage 4 "build HeteroData graphs (keyword operators)"
python -u scripts/build_graphs.py \
    --input_dir data/processed \
    --output_dir data/processed/graphs 2>&1 | tee $LOG/stage4_graphs.log
done_stage 4
fi

# ----------------------------------------------------------------------------
stage 5 "smoke test"
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
if have 6_5; then echo "STAGE 6.5 already done, skipping"; else
stage 6.5 "R-GCN capacity match (hidden_dim=768, seed 42)"
python -u scripts/train_rgcn.py --seed 42 --hidden_dim 768 2>&1 | tee $LOG/stage6_5_rgcn_h768.log
# train_rgcn saves to outputs/logs/baseline_rgcn_seed42.json — rename to disambiguate
if [[ -f outputs/logs/baseline_rgcn_seed42.json ]] && [[ ! -f outputs/logs/baseline_rgcn_h768_seed42.json ]]; then
  cp outputs/logs/baseline_rgcn_seed42.json outputs/logs/baseline_rgcn_h768_seed42.json
fi
if [[ -f outputs/checkpoints/best_rgcn_seed42.pt ]] && [[ ! -f outputs/checkpoints/best_rgcn_h768_seed42.pt ]]; then
  cp outputs/checkpoints/best_rgcn_seed42.pt outputs/checkpoints/best_rgcn_h768_seed42.pt
fi
done_stage 6_5
fi

# ----------------------------------------------------------------------------
if have 7; then echo "STAGE 7 already done, skipping"; else
stage 7 "JusDef main (3 seeds, keyword operators)"
for SEED in 42 43 44; do
  echo "-- JusDef full seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag full 2>&1 | tee $LOG/stage7_full_s${SEED}.log
done
done_stage 7
fi

# ----------------------------------------------------------------------------
if have 7_5; then echo "STAGE 7.5 already done, skipping"; else
stage 7.5 "Neural operator detector pipeline"

# 7.5a — train the neural detector on 3000 annotated sentences
if [[ ! -f outputs/checkpoints/operator_detector_neural.pt ]]; then
  echo "-- 7.5a train neural operator detector --"
  python -u scripts/train_operator_detector.py \
      --data data/annotations/operator_labels_3000.jsonl \
      --epochs 6 --batch_size 32 --lr 2e-5 --seed 42 \
      2>&1 | tee $LOG/stage7_5a_detector_train.log
fi

# 7.5b — re-label processed pickles using the neural detector
if [[ ! -f data/processed_neural/test_processed.pkl ]]; then
  echo "-- 7.5b relabel operators (keyword -> neural) --"
  python -u scripts/relabel_operators_neural.py \
      --input_dir data/processed \
      --output_dir data/processed_neural \
      2>&1 | tee $LOG/stage7_5b_relabel.log
fi

# 7.5c — rebuild graphs using the relabeled processed pickles
if [[ ! -f data/processed_neural/graphs/test_graphs.pt ]]; then
  echo "-- 7.5c rebuild graphs with neural operators --"
  python -u scripts/build_graphs.py \
      --input_dir data/processed_neural \
      --output_dir data/processed_neural/graphs \
      --emb_dir data/processed/embeddings \
      --label_adj_path data/processed/label_adj.pt \
      2>&1 | tee $LOG/stage7_5c_graphs.log
fi

# 7.5d — train JusDef on neural-operator graphs
echo "-- 7.5d JusDef on neural-operator graphs, seeds: $JUSDEF_NEURAL_SEEDS --"
for SEED in $JUSDEF_NEURAL_SEEDS; do
  echo "-- JusDef neural seed $SEED --"
  python -u scripts/train_jusdef.py \
      --seed $SEED --tag full_neural \
      --graph_dir data/processed_neural/graphs \
      2>&1 | tee $LOG/stage7_5d_full_neural_s${SEED}.log
done

done_stage 7_5
fi

# ----------------------------------------------------------------------------
if have 8; then echo "STAGE 8 already done, skipping"; else
stage 8 "JusDef ablations (3 seeds x 3 variants, keyword operators)"
for SEED in 42 43 44; do
  echo "-- no_dmp seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_dmp --no_dmp \
      2>&1 | tee $LOG/stage8_no_dmp_s${SEED}.log
  echo "-- no_auth seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_auth --no_authority \
      2>&1 | tee $LOG/stage8_no_auth_s${SEED}.log
  echo "-- no_dmp_no_auth seed $SEED --"
  python -u scripts/train_jusdef.py --seed $SEED --tag no_dmp_no_auth --no_dmp --no_authority \
      2>&1 | tee $LOG/stage8_no_dmp_no_auth_s${SEED}.log
done
done_stage 8
fi

# ----------------------------------------------------------------------------
stage 9 "final unified evaluation"
python -u scripts/eval_all_jusdef.py 2>&1 | tee $LOG/stage9_eval.log

echo ""
echo "=========================================="
echo "ALL STAGES COMPLETE -- $(stamp)"
echo "=========================================="
ls -la outputs/logs/*.json
