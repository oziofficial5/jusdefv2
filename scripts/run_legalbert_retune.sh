#!/bin/bash
# ============================================================================
# Reviewer fix: properly-tuned LEDGAR baselines for the structure-vs-scale
# section. The published LexGLUE numbers are LegalBERT 0.819 macro / 0.883
# micro and TF-IDF+SVM 0.814 macro; the submitted paper reported LegalBERT at
# only 0.808 (under-tuned: batch 16, 5 epochs), reversing the published order.
#
#   Stage TFIDF : TF-IDF + LinearSVC (CPU, fast) -> confirms the classical number
#   Stage FT    : direct paragraph LegalBERT fine-tuning, seeds 42-46, TUNED
#                 (batch 32, up to 20 epochs, patience 4, max_len 256, lr 2e-5)
#   Stage AGG   : summarise 5-seed LegalBERT vs TF-IDF vs published, honest verdict
#
# Sentinel-resumable (outputs/sentinels/). Delete a sentinel to force a rerun.
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "repo root: $(pwd)"

LOG=outputs/logs
SENT=outputs/sentinels
mkdir -p "$LOG" "$SENT" outputs/checkpoints

have() { [[ -f "$SENT/$1.done" ]]; }
mark() { touch "$SENT/$1.done"; }
stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "== LegalBERT re-tune + TF-IDF :: start $(stamp) =="
python -c "import torch; print('cuda', torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# --- Stage TFIDF: classical baseline (CPU, minutes) ------------------------
key="tfidf_svm"; json="$LOG/ledgar_${key}.json"
if have "$key" || [[ -f "$json" ]]; then
  echo "skip $key (result present)"
else
  echo "== TF-IDF+SVM :: $(stamp) =="
  python -u scripts/run_tfidf_svm_ledgar.py --tag "$key" 2>&1 | tee "$LOG/${key}.log"
  mark "$key"
fi

# --- Stage FT: properly-tuned direct paragraph LegalBERT, seeds 42-46 ------
for seed in 42 43 44 45 46; do
  key="legalbert_tuned_s${seed}"
  json="$LOG/legalbert_ledgar_ft_tuned_s${seed}.json"
  if have "$key" || [[ -f "$json" ]]; then echo "skip $key (result present)"; continue; fi
  echo "== train $key :: $(stamp) =="
  python -u scripts/train_legalbert_ledgar.py --seed "$seed" --tag ft_tuned \
    --epochs 20 --batch_size 32 --patience 4 --max_length 256 --lr 2e-5 \
    2>&1 | tee "$LOG/${key}.log"
  mark "$key"
done

# --- Stage AGG: honest ordering vs TF-IDF and published --------------------
echo "== summary :: $(stamp) =="
python -u scripts/summarise_ledgar_baselines.py 2>&1 | tee "$LOG/ledgar_baselines_summary.log"

echo "== done :: $(stamp) =="
