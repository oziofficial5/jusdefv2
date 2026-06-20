"""
Preprocess CUAD by WHOLE-CONTRACT paragraph splitting (not single-clause spans).

The first CUAD preprocessor (preprocess_cuad.py) used each QA answer span as
a paragraph, producing a bimodal per-paragraph density distribution
({0%, >=20%}, with ~1 paragraph in the 10-20% operating-regime window).
This version splits full contract texts into multi-sentence paragraphs --
analogous to LEDGAR's structure -- and labels each paragraph by the most-
overlapping CUAD clause type. The expected result is a smoother density
gradient that makes the 10-20% regime test meaningful on CUAD.

Output format matches preprocess_ledgar.py:
  data/processed_cuad_whole/{train,validation,test}_processed.pkl

Usage:
    python scripts/preprocess_cuad_whole.py
    python scripts/preprocess_cuad_whole.py --debug
"""
import os
import re
import sys
import argparse
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict

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
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def paragraph_split(text, min_sentences=2, max_sentences=12):
    """Split a full contract into paragraphs.

    Heuristics:
      1. First split on double newlines (canonical paragraph breaks).
      2. Within each block, sentence-tokenize.
      3. If a block has > max_sentences, sub-split into windows of ~max_sentences.
      4. Discard blocks with < min_sentences (avoids headings, signature lines).
    """
    paragraphs = []
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    for block in blocks:
        sents = sentence_split(block)
        if len(sents) < min_sentences:
            continue
        # Window-split long blocks
        if len(sents) <= max_sentences:
            paragraphs.append((sents, block))
        else:
            for i in range(0, len(sents), max_sentences):
                window_sents = sents[i:i + max_sentences]
                if len(window_sents) < min_sentences:
                    continue
                paragraphs.append((window_sents, " ".join(window_sents)))
    return paragraphs


def extract_clause_type(question):
    m = re.search(r'related to\s+"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    return question[:40].strip()


@torch.no_grad()
def embed_and_detect(sentences, tokenizer, encoder, detector, device,
                     batch_size=64, max_len=128):
    all_embs, all_ops = [], []
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


def build_contract_annotations(rows):
    """Group CUAD QA rows by contract title and collect annotated clause spans.

    Returns: dict[contract_title] -> {"text": str, "spans": [(span_text, clause_type), ...]}
    """
    contracts = {}
    for row in rows:
        title = row.get("title", "")
        context = row.get("context", "")
        question = row.get("question", "")
        ans = row.get("answers", {})
        if not title or not context:
            continue
        if title not in contracts:
            contracts[title] = {"text": context, "spans": []}
        # ensure consistent context (the same contract title should always have the same context)
        if contracts[title]["text"] != context:
            # rare; keep the first one
            pass
        if not isinstance(ans, dict):
            continue
        texts = ans.get("text", [])
        if not texts:
            continue
        ctype = extract_clause_type(question)
        for t in texts:
            if t and len(t.strip()) > 0:
                contracts[title]["spans"].append((t.strip(), ctype))
    return contracts


def label_paragraph(para_text, spans):
    """For a paragraph, return the most-overlapping clause type or None.

    Overlap = number of characters of the span found as substring of the paragraph.
    A span is considered "in" the paragraph if a substantial fraction (>= 0.6) of
    its characters appears in the paragraph text.
    """
    overlap_chars = Counter()
    para_lower = para_text.lower()
    for span_text, clause_type in spans:
        if len(span_text) < 30:
            continue
        sl = span_text.lower()
        # Two-stage check: substring (fast) then fuzzy fallback
        if sl in para_lower:
            overlap_chars[clause_type] += len(sl)
        else:
            # cheap overlap heuristic: longest common substring fraction
            # to keep this fast, just check first/last 40 chars of the span
            head = sl[:40]
            tail = sl[-40:]
            if head in para_lower or tail in para_lower:
                overlap_chars[clause_type] += min(len(sl), 200)
    if not overlap_chars:
        return None
    return overlap_chars.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_ckpt",
                        default="outputs/checkpoints/operator_detector_neural.pt")
    parser.add_argument("--output_dir", default="data/processed_cuad_whole")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--min_sentences", type=int, default=2)
    parser.add_argument("--max_sentences", type=int, default=12)
    parser.add_argument("--include_unlabelled", action="store_true",
                        help="If set, paragraphs with no overlapping annotated clause "
                             "type are kept with label 'Other'; otherwise dropped")
    parser.add_argument("--debug", action="store_true",
                        help="Process only first 20 contracts per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print(f"Loading detector from {args.detector_ckpt}", flush=True)
    state = torch.load(args.detector_ckpt, map_location=device)
    backbone_name = state["config"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    encoder = AutoModel.from_pretrained(backbone_name).to(device).eval()
    detector = OperatorClassifier(backbone_name, num_classes=4).to(device).eval()
    detector.load_state_dict(state["state_dict"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------- PHASE 1: load CUAD and group by contract ---------------
    print("\nLoading CUAD (theatticusproject/cuad-qa)...", flush=True)
    cuad_train = load_dataset("theatticusproject/cuad-qa", split="train")
    cuad_test = load_dataset("theatticusproject/cuad-qa", split="test")
    print(f"  train rows: {len(cuad_train)}; test rows: {len(cuad_test)}", flush=True)

    print("  Grouping QA rows by contract...", flush=True)
    train_contracts = build_contract_annotations(cuad_train)
    test_contracts = build_contract_annotations(cuad_test)
    print(f"  Unique train contracts: {len(train_contracts)}; "
          f"test contracts: {len(test_contracts)}", flush=True)

    if args.debug:
        train_contracts = dict(list(train_contracts.items())[:20])
        test_contracts = dict(list(test_contracts.items())[:5])

    # --------------- PHASE 2: build label vocabulary ---------------
    all_types = set()
    for c in train_contracts.values():
        for _, ct in c["spans"]:
            all_types.add(ct)
    for c in test_contracts.values():
        for _, ct in c["spans"]:
            all_types.add(ct)
    all_types = sorted(all_types)
    if args.include_unlabelled:
        all_types.append("Other")
    type_to_id = {t: i for i, t in enumerate(all_types)}
    print(f"\n  Number of clause-type labels: {len(all_types)}", flush=True)

    with open(out_dir / "label_vocab.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(all_types):
            f.write(f"{i}\t{name}\n")
    print(f"  Saved label vocab to {out_dir / 'label_vocab.txt'}", flush=True)

    # --------------- PHASE 3: paragraph-split contracts ---------------
    def split_contracts(contracts, name):
        out = []  # list of (paragraph_sentences, paragraph_text, label_str)
        for title, c in tqdm(contracts.items(), desc=f"splitting {name}"):
            paras = paragraph_split(c["text"],
                                     min_sentences=args.min_sentences,
                                     max_sentences=args.max_sentences)
            for sents, ptext in paras:
                lbl = label_paragraph(ptext, c["spans"])
                if lbl is None:
                    if not args.include_unlabelled:
                        continue
                    lbl = "Other"
                out.append((sents, ptext, lbl))
        return out

    train_paras_all = split_contracts(train_contracts, "train")
    test_paras = split_contracts(test_contracts, "test")
    print(f"\n  After paragraph-splitting: "
          f"train={len(train_paras_all)}, test={len(test_paras)}", flush=True)

    # Train -> train/val split
    rng = random.Random(args.seed)
    rng.shuffle(train_paras_all)
    n_val = int(len(train_paras_all) * args.val_fraction)
    val_paras = train_paras_all[:n_val]
    train_paras = train_paras_all[n_val:]
    print(f"  After val split: train={len(train_paras)}, val={len(val_paras)}",
          flush=True)

    # --------------- PHASE 4: embed + detect per split ---------------
    for split_name, split_paras in [("train", train_paras),
                                     ("validation", val_paras),
                                     ("test", test_paras)]:
        print(f"\n{'=' * 50}\nProcessing CUAD-whole {split_name}\n{'=' * 50}",
              flush=True)
        processed = []
        op_counts = Counter()
        label_counts = Counter()

        for sents, _ptext, lbl in tqdm(split_paras, desc=split_name):
            if not sents:
                continue
            embs, ops = embed_and_detect(sents, tokenizer, encoder, detector,
                                          device, batch_size=args.batch_size,
                                          max_len=args.max_len)
            processed.append({
                "label": type_to_id[lbl],
                "sentences": sents,
                "operators": ops.tolist(),
                "embeddings": embs,
            })
            label_counts[lbl] += 1
            for op in ops.tolist():
                op_counts[op] += 1

        out_path = out_dir / f"{split_name}_processed.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)

        total_ops = sum(op_counts.values()) or 1
        print(f"\n  Saved {len(processed)} paragraphs to {out_path}", flush=True)
        for op_idx, op_name in enumerate(["AFF", "NEG", "EXC", "OVR"]):
            count = op_counts.get(op_idx, 0)
            print(f"    {op_name}: {count:6d}  ({100*count/total_ops:.2f}%)",
                  flush=True)

        # Per-paragraph density distribution
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
            print(f"  Per-paragraph non-AFF density distribution:", flush=True)
            print(f"    mean: {d.mean()*100:.2f}%, median: "
                  f"{np.median(d)*100:.2f}%", flush=True)
            for lo, hi, name in [(0.001, 0.05, "0-5%"), (0.05, 0.10, "5-10%"),
                                  (0.10, 0.20, "10-20%"), (0.20, 0.30, "20-30%"),
                                  (0.30, 0.50, "30-50%"), (0.50, 1.01, "50%+")]:
                n_in = int(((d >= lo) & (d < hi)).sum())
                print(f"    {name:<10}: {n_in:4d} paragraphs "
                      f"({100*n_in/len(d):5.2f}%)", flush=True)

    print(f"\nCUAD-whole preprocessing complete. num_classes: {len(all_types)}",
          flush=True)


if __name__ == "__main__":
    main()
