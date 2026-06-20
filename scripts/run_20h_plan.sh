#!/bin/bash
# =============================================================================
# 20-hour Ampere plan: three blocks
#
#   BLOCK A (~6h): CUAD whole-contract -> train -> analyze
#   BLOCK B (~8-10h): LegalBERT fine-tuning baseline on LEDGAR (5 seeds)
#   BLOCK C (~2h): per-regime v4_twostage ablation (4 ablations × 5 seeds)
#
# Launch:
#   cd ~/jusdefv2/jusdefv2
#   git pull origin jusdef-ledgar
#   sed -i 's/\r$//' scripts/*.sh scripts/*.py src/model/*.py
#   nohup bash scripts/run_20h_plan.sh > outputs/logs/run_20h.log 2>&1 &
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
# BLOCK A -- CUAD whole-contract preprocessing + train + analyze
# =============================================================================

banner "BLOCK A.1: CUAD whole-contract preprocessing"

if ! have a1_cuad_whole_preprocess; then
  python -u scripts/preprocess_cuad_whole.py 2>&1 | tee $LOG/a1_cuad_whole_preprocess.log
  done_run a1_cuad_whole_preprocess
fi

# Detect classes
if [[ -f data/processed_cuad_whole/label_vocab.txt ]]; then
  NUM_CUAD_CLASSES=$(wc -l < data/processed_cuad_whole/label_vocab.txt)
  echo "CUAD-whole num_classes: $NUM_CUAD_CLASSES"
else
  echo "WARNING: CUAD-whole label_vocab.txt missing; skipping Block A.2/A.3"
  NUM_CUAD_CLASSES=0
fi

if [[ $NUM_CUAD_CLASSES -gt 0 ]]; then
  banner "BLOCK A.2: train CUAD-whole v3 + mean baseline, 5 seeds"
  for SEED in "${SEEDS[@]}"; do
    SENT_NAME="a2_cuad_whole_mean_s${SEED}"
    if ! have $SENT_NAME; then
      python -u scripts/train_ledgar.py \
        --seed $SEED --tag cuad_whole_mean --dmp_variant mean \
        --data_dir data/processed_cuad_whole \
        --num_classes $NUM_CUAD_CLASSES \
        2>&1 | tee $LOG/${SENT_NAME}.log
      done_run $SENT_NAME
    fi
    SENT_NAME="a2_cuad_whole_v3_s${SEED}"
    if ! have $SENT_NAME; then
      python -u scripts/train_ledgar.py \
        --seed $SEED --tag cuad_whole_v3 --dmp_variant v3 \
        --data_dir data/processed_cuad_whole \
        --num_classes $NUM_CUAD_CLASSES \
        2>&1 | tee $LOG/${SENT_NAME}.log
      done_run $SENT_NAME
    fi
  done

  banner "BLOCK A.3: CUAD-whole density-stratified evaluation"
  # Reuse analyse_cuad_density_subset.py but point at the new pkl dir.
  # We swap the data dir + file prefixes via a tiny inline wrapper.
  if ! have a3_cuad_whole_eval; then
    python -u - <<'PYEOF' 2>&1 | tee $LOG/a3_cuad_whole_eval.log
import sys, os, pickle, json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, os.getcwd())
from src.model.jusdef_ledgar import JusDefLEDGAR

SEEDS = [42, 43, 44, 45, 46]
VARIANTS = {
    "mean":     ({"dmp_variant": "mean"}, "ledgar_cuad_whole_mean"),
    "v3_pilot": ({"dmp_variant": "v3"},   "ledgar_cuad_whole_v3"),
}
BINS = [
    ("all",     0.00, 1.01),
    ("0%",      0.00, 0.001),
    ("0-5%",    0.001, 0.05),
    ("5-10%",   0.05, 0.10),
    ("10-20%",  0.10, 0.20),
    ("20-30%",  0.20, 0.30),
    ("30-50%",  0.30, 0.50),
    ("50%+",    0.50, 1.01),
    (">=20%",   0.20, 1.01),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
with open("data/processed_cuad_whole/test_processed.pkl", "rb") as f:
    test_data = pickle.load(f)
print(f"CUAD-whole test paragraphs: {len(test_data)}")
num_classes = int(max(p["label"] for p in test_data)) + 1
print(f"num_classes: {num_classes}")

density = np.array([sum(1 for o in d["operators"] if o != 0)/max(len(d["operators"]),1)
                    for d in test_data])
labels_true = np.array([d["label"] for d in test_data])

print("\nDensity distribution:")
for thresh in [0.05, 0.10, 0.15, 0.20]:
    n = int((density >= thresh).sum())
    print(f"  >= {thresh*100:5.1f}%: {n:4d} ({n/len(test_data)*100:5.2f}%)")
n_10_20 = int(((density>=0.10) & (density<0.20)).sum())
print(f"  10-20% bin: {n_10_20} paragraphs")

@torch.no_grad()
def predict_all(model, data, device):
    model.eval()
    preds = []
    for p in data:
        emb = p["embeddings"].to(device)
        s2p = torch.zeros(emb.size(0), dtype=torch.long, device=device)
        ops = torch.tensor(p["operators"], dtype=torch.long, device=device)
        logits = model(emb, s2p, ops, num_paragraphs=1)
        preds.append(int(logits.argmax(dim=-1).item()))
    return np.array(preds)

results = {}
ckpt_dir = Path("outputs/checkpoints")
for tag, (kw, prefix) in VARIANTS.items():
    print(f"\n=== {tag} ===")
    results[tag] = {}
    for seed in SEEDS:
        ckpt = ckpt_dir / f"{prefix}_s{seed}.pt"
        if not ckpt.is_file():
            print(f"  [skip] s{seed}: missing")
            continue
        full = dict(in_dim=768, hidden_dim=512, num_classes=num_classes, num_layers=1)
        full.update(kw)
        model = JusDefLEDGAR(**full).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        preds = predict_all(model, test_data, device)
        per_bin = {}
        for name, lo, hi in BINS:
            mask = (density >= lo) & (density < hi)
            n = int(mask.sum())
            if n < 5:
                per_bin[name] = {"n": n, "macro_f1": None}; continue
            ls = labels_true[mask]; ps = preds[mask]
            lp = sorted(set(ls.tolist()))
            f1 = f1_score(ls, ps, labels=lp, average="macro", zero_division=0)
            per_bin[name] = {"n": n, "macro_f1": float(f1)}
        results[tag][seed] = per_bin
        print(f"  s{seed}: all={per_bin['all']['macro_f1']:.4f}  "
              f"10-20%={per_bin['10-20%']['macro_f1']} (n={per_bin['10-20%']['n']})  "
              f">=20%={per_bin['>=20%']['macro_f1']} (n={per_bin['>=20%']['n']})")
        del model
        if device == "cuda": torch.cuda.empty_cache()

print("\n" + "="*100)
print(" CUAD-WHOLE PER-REGIME EVALUATION (5-seed mean +- std)")
print("="*100)
header = f"{'variant':<14}" + "".join(f"{n:>14}" for n,_,_ in BINS)
print(header); print("-"*len(header))
agg = {}
for tag in VARIANTS:
    row = f"{tag:<14}"; agg[tag] = {}
    for name,_,_ in BINS:
        vs = [results[tag][s][name]["macro_f1"] for s in SEEDS
              if s in results[tag] and results[tag][s][name]["macro_f1"] is not None]
        if vs:
            m = float(np.mean(vs)); st = float(np.std(vs, ddof=1)) if len(vs)>1 else 0.0
            agg[tag][name] = {"mean": m, "std": st, "n_seeds": len(vs)}
            row += f"  {m:.4f}+-{st:.3f}"
        else:
            agg[tag][name] = {"mean": None, "std": None, "n_seeds": 0}
            row += f"{'---':>14}"
    print(row)

print("\n" + "="*70)
print(" CUAD-WHOLE CROSS-CORPUS REGIME VERDICT")
print("="*70)
for name in ["all", "10-20%", ">=20%"]:
    v3m = agg["v3_pilot"].get(name, {}).get("mean")
    mnm = agg["mean"].get(name, {}).get("mean")
    if v3m is None or mnm is None:
        continue
    d = v3m - mnm
    sign = "+" if d >= 0 else "-"
    print(f"  {name:<8} v3 {v3m:.4f}  mean {mnm:.4f}  delta {sign}{abs(d):.4f}")

regime = agg["v3_pilot"].get("10-20%", {}).get("mean")
mean_r = agg["mean"].get("10-20%", {}).get("mean")
if regime is not None and mean_r is not None:
    delta = regime - mean_r
    if delta > 0.02:
        print(f"\n  -> CROSS-CORPUS REGIME REPLICATED: v3 wins by +{delta:.4f}")
    elif delta < -0.02:
        print(f"\n  -> regime NOT replicated: v3 loses by {delta:.4f}")
    else:
        print(f"\n  -> regime delta {delta:+.4f} (within noise)")

out_path = Path("outputs/logs/cuad_whole_density_stratified.json")
with open(out_path, "w") as f:
    json.dump({"aggregate": agg, "per_seed": results}, f, indent=2)
print(f"\nSaved {out_path}")
PYEOF
    done_run a3_cuad_whole_eval
  fi
fi

# =============================================================================
# BLOCK B -- LegalBERT fine-tuning baseline on LEDGAR, 5 seeds
# =============================================================================

banner "BLOCK B: LegalBERT fine-tuning baseline on LEDGAR (5 seeds)"

for SEED in "${SEEDS[@]}"; do
  SENT_NAME="b_legalbert_ft_s${SEED}"
  if have $SENT_NAME; then
    echo "[skip] $SENT_NAME"
    continue
  fi
  echo ""; echo "--- LegalBERT fine-tune seed $SEED ---"
  python -u scripts/train_legalbert_ledgar.py \
    --seed $SEED --tag ft --epochs 5 --batch_size 16 \
    2>&1 | tee $LOG/${SENT_NAME}.log
  done_run $SENT_NAME
done

# =============================================================================
# BLOCK C -- per-regime v4_twostage ablation (4 ablations x 5 seeds)
# =============================================================================

banner "BLOCK C: per-regime v4_twostage ablation"

# Each ablation: freeze a different v3 component into v4_twostage's frozen-v3 stage.
# Implementation strategy: pre-train v3 with the ablation flag, then run two-stage
# classifier-only training on top of the ablated frozen features.

ABLATIONS=("shared_w_revert" "hard_attention" "unit_coefs" "no_drift_reg")
for ABL in "${ABLATIONS[@]}"; do
  case "$ABL" in
    shared_w_revert)  EXTRA="--v3_shared_w_revert" ;;
    hard_attention)   EXTRA="--v3_hard_attention" ;;
    unit_coefs)       EXTRA="--v3_init_coefs 1.0,1.0,1.0,1.0" ;;
    no_drift_reg)     EXTRA="--v3_coef_reg_strength 0.0" ;;
  esac

  for SEED in "${SEEDS[@]}"; do
    # Train v3 with this ablation (need fresh v3 checkpoint to use as Stage-1)
    V3_ABL_TAG="v3_abl_${ABL}_for_v4ts"
    SENT_NAME="c_v3_abl_${ABL}_s${SEED}"
    if ! have $SENT_NAME; then
      python -u scripts/train_ledgar.py \
        --seed $SEED --tag $V3_ABL_TAG --dmp_variant v3 $EXTRA \
        2>&1 | tee $LOG/${SENT_NAME}.log
      done_run $SENT_NAME
    fi
    # Now run two-stage on top of this ablated v3
    SENT_NAME="c_v4ts_abl_${ABL}_s${SEED}"
    if ! have $SENT_NAME; then
      python -u scripts/train_v4_twostage.py \
        --seed $SEED --tag v4ts_abl_${ABL} \
        --pretrained_v3 outputs/checkpoints/ledgar_${V3_ABL_TAG}_s${SEED}.pt \
        2>&1 | tee $LOG/${SENT_NAME}.log
      done_run $SENT_NAME
    fi
  done
done

# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner "20H PLAN COMPLETE -- $(stamp)"

echo ""
echo "=== CUAD-whole density distribution (test split) ==="
grep -A 8 "Per-paragraph non-AFF density distribution" $LOG/a1_cuad_whole_preprocess.log 2>/dev/null | tail -10

echo ""
echo "=== CUAD-whole regime verdict ==="
grep -A 12 "CUAD-WHOLE CROSS-CORPUS REGIME VERDICT" $LOG/a3_cuad_whole_eval.log 2>/dev/null | tail -12

echo ""
echo "=== LegalBERT fine-tune 5-seed results ==="
for SEED in "${SEEDS[@]}"; do
  p="outputs/logs/legalbert_ledgar_ft_s${SEED}.json"
  if [[ -f "$p" ]]; then
    python -c "
import json
d = json.load(open('$p'))
print(f'  s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}  test_micro_f1={d[\"test_micro_f1\"]:.4f}')
" 2>/dev/null || echo "  s$SEED: unparseable"
  else
    echo "  s$SEED: MISSING"
  fi
done

echo ""
echo "=== v4_twostage ablation results ==="
for ABL in shared_w_revert hard_attention unit_coefs no_drift_reg; do
  echo "  $ABL:"
  for SEED in "${SEEDS[@]}"; do
    p="outputs/logs/ledgar_v4ts_abl_${ABL}_s${SEED}.json"
    if [[ -f "$p" ]]; then
      python -c "
import json
d = json.load(open('$p'))
print(f'    s$SEED: test_macro_f1={d[\"test_macro_f1\"]:.4f}')
" 2>/dev/null || echo "    s$SEED: unparseable"
    else
      echo "    s$SEED: MISSING"
    fi
  done
done

echo ""
echo "Sentinels:"
ls -la outputs/sentinels/{a1,a2,a3,b,c}*.done 2>/dev/null | head -40
