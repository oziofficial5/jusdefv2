"""
Preprocess CUAD (Contract Understanding Atticus Dataset) for JusDef training.

CUAD is structured as SQuAD-style QA: each row has a contract document and a
question asking for a specific clause type. The answer span is the relevant
clause text. We reinterpret this as paragraph-level classification: each
distinct answer span becomes a paragraph, labelled by its clause type
(question category). This makes CUAD a 41-class single-label paragraph
classification task analogous to LEDGAR.

Output format matches preprocess_ledgar.py:
  data/processed_cuad/{train,validation,test}_processed.pkl
  each entry: {"label": int, "sentences": [...], "operators": [int...],
               "embeddings": tensor(N_sents, 768)}

The CUAD HuggingFace dataset has only train/test; we split off 10% of train
as validation.

Usage:
    python scripts/preprocess_cuad.py
    python scripts/preprocess_cuad.py --debug   # smoke test on 50 instances
"""
import os
import re
import sys
import argparse
import pickle
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


class OperatorClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))


def sentence_split(text):
    """Conservative sentence splitter, same convention as preprocess_ledgar.py."""
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def extract_clause_type_from_question(question):
    """CUAD questions follow patterns like:
        'Highlight the parts (if any) of this contract related to "X" that ...'
    Return X (the clause-type tag).
    """
    m = re.search(r'related to\s+"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    # Fallback: first 40 chars of question
    return question[:40].strip()


@torch.no_grad()
def embed_and_detect(sentences, tokenizer, encoder, detector, device,
                     batch_size=64, max_len=128):
    all_embs = []
    all_ops = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        out = encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls = out.last_hidden_state[:, 0]
        all_embs.append(cls.cpu())
        det_logits = detector(input_ids, attn_mask)
        ops = det_logits.argmax(dim=-1).cpu()
        all_ops.append(ops)
    return torch.cat(all_embs, dim=0), torch.cat(all_ops, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_ckpt",
                        default="outputs/checkpoints/operator_detector_neural.pt")
    parser.add_argument("--output_dir", default="data/processed_cuad")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.10,
                        help="Fraction of train to hold out as validation")
    parser.add_argument("--min_span_chars", type=int, default=20,
                        help="Skip answer spans shorter than this (likely noise)")
    parser.add_argument("--debug", action="store_true",
                        help="Process only first 200 instances per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load operator detector + backbone (same protocol as preprocess_ledgar.py)
    print(f"Loading neural operator detector from {args.detector_ckpt}", flush=True)
    state = torch.load(args.detector_ckpt, map_location=device)
    backbone_name = state["config"]["backbone"]
    print(f"  Backbone: {backbone_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    encoder = AutoModel.from_pretrained(backbone_name).to(device).eval()
    detector = OperatorClassifier(backbone_name, num_classes=4).to(device).eval()
    detector.load_state_dict(state["state_dict"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # PHASE 1 -- build (clause_text -> clause_type) pairs from CUAD
    # --------------------------------------------------------------
    print("\nLoading CUAD (theatticusproject/cuad-qa)...", flush=True)
    cuad_train = load_dataset("theatticusproject/cuad-qa", split="train")
    cuad_test = load_dataset("theatticusproject/cuad-qa", split="test")
    print(f"  CUAD train rows: {len(cuad_train)}; test rows: {len(cuad_test)}",
          flush=True)

    def collect_spans(ds, name):
        spans = []  # list of (clause_text, clause_type)
        for row in tqdm(ds, desc=f"collecting spans ({name})"):
            q = row.get("question", "")
            ans = row.get("answers", {})
            if not isinstance(ans, dict):
                continue
            texts = ans.get("text", [])
            if not texts:
                continue
            for t in texts:
                t = (t or "").strip()
                if len(t) < args.min_span_chars:
                    continue
                ctype = extract_clause_type_from_question(q)
                spans.append((t, ctype))
        return spans

    train_spans = collect_spans(cuad_train, "train")
    test_spans = collect_spans(cuad_test, "test")
    print(f"  After filtering: {len(train_spans)} train spans, "
          f"{len(test_spans)} test spans", flush=True)

    if args.debug:
        train_spans = train_spans[:200]
        test_spans = test_spans[:50]

    # Build clause-type vocabulary from union of train+test types
    all_types = sorted({t for _, t in train_spans} | {t for _, t in test_spans})
    type_to_id = {t: i for i, t in enumerate(all_types)}
    print(f"\n  Number of clause-type labels: {len(all_types)}", flush=True)
    print(f"  Sample labels: {all_types[:5]}", flush=True)

    # Save label vocabulary
    with open(out_dir / "label_vocab.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(all_types):
            f.write(f"{i}\t{name}\n")
    print(f"  Saved label vocab to {out_dir / 'label_vocab.txt'}", flush=True)

    # --------------------------------------------------------------
    # PHASE 2 -- split train into train/val
    # --------------------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(train_spans)
    n_val = int(len(train_spans) * args.val_fraction)
    val_spans = train_spans[:n_val]
    train_spans = train_spans[n_val:]
    print(f"\n  After val split: train={len(train_spans)}, "
          f"val={len(val_spans)}, test={len(test_spans)}", flush=True)

    # --------------------------------------------------------------
    # PHASE 3 -- process each split: sentence-split, embed, detect
    # --------------------------------------------------------------
    for split_name, split_spans in [("train", train_spans),
                                     ("validation", val_spans),
                                     ("test", test_spans)]:
        print(f"\n{'=' * 50}", flush=True)
        print(f"Processing CUAD {split_name}", flush=True)
        print(f"{'=' * 50}", flush=True)

        processed = []
        op_counts = Counter()
        label_counts = Counter()

        for clause_text, clause_type in tqdm(split_spans, desc=split_name):
            sentences = sentence_split(clause_text)
            if not sentences:
                continue
            embs, ops = embed_and_detect(sentences, tokenizer, encoder, detector,
                                          device, batch_size=args.batch_size,
                                          max_len=args.max_len)
            processed.append({
                "label": type_to_id[clause_type],
                "sentences": sentences,
                "operators": ops.tolist(),
                "embeddings": embs,
            })
            label_counts[clause_type] += 1
            for op in ops.tolist():
                op_counts[op] += 1

        out_path = out_dir / f"{split_name}_processed.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)

        total_ops = sum(op_counts.values()) or 1
        print(f"\n  Saved {len(processed)} paragraphs to {out_path}", flush=True)
        print(f"  Operator distribution (sentences):", flush=True)
        for op_idx in range(4):
            op_name = ["AFF", "NEG", "EXC", "OVR"][op_idx]
            count = op_counts.get(op_idx, 0)
            print(f"    {op_name}: {count:6d}  ({100 * count / total_ops:.2f}%)",
                  flush=True)

        # Per-paragraph density distribution (key metric for regime hypothesis)
        densities = []
        for p in processed:
            n = len(p["operators"])
            if n == 0:
                continue
            n_non_aff = sum(1 for o in p["operators"] if o != 0)
            densities.append(n_non_aff / n)
        if densities:
            import numpy as np
            d = np.array(densities)
            print(f"  Per-paragraph non-AFF density:", flush=True)
            print(f"    mean: {d.mean()*100:.2f}%", flush=True)
            for thresh in [0.05, 0.10, 0.15, 0.20]:
                frac = (d >= thresh).mean()
                print(f"    >= {thresh*100:5.1f}%: {(d >= thresh).sum():4d} "
                      f"paragraphs ({frac*100:.2f}%)", flush=True)
            in_regime = ((d >= 0.10) & (d < 0.20)).sum()
            print(f"    in 10-20% operating regime: {in_regime} paragraphs", flush=True)

    print(f"\nCUAD preprocessing complete. Number of classes: {len(all_types)}",
          flush=True)


if __name__ == "__main__":
    main()
