"""
Re-label every concept's operator using the trained neural detector.

Input:  data/processed/{train,validation,test}_processed.pkl
        outputs/checkpoints/operator_detector_neural.pt

Output: data/processed_neural/{train,validation,test}_processed.pkl

For each concept in each section, build a "governing sentence" string
(the same logic as src/preprocess/operator_detector.detect_operators_in_section)
and replace c["operator"] with the neural prediction.

Embeddings and structural metadata are NOT re-computed — only the operator
field on each concept is overwritten. Stage 4 (build_graphs.py) then reads
from data/processed_neural/ via --input_dir.

Approximate runtime on A100: ~30 min over all 55K docs.
"""
import os
import re
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


LABEL_TO_INT = {"AFF": 0, "NEG": 1, "EXC": 2, "OVR": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class OperatorClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls))


def sentences_for_section(section_text):
    """Same splitter as operator_detector.detect_operators_in_section."""
    return re.split(r"(?<=[.;])\s+", section_text)


def governing_sentence(section_text, span_start):
    """Find the sentence containing the concept span_start, like the keyword detector."""
    sentences = sentences_for_section(section_text)
    char_count = 0
    for sent in sentences:
        char_count += len(sent) + 1
        if char_count >= span_start:
            return sent
    return section_text


@torch.no_grad()
def predict_batch(model, tokenizer, sentences, device, max_len, batch_size):
    preds = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", default="outputs/checkpoints/operator_detector_neural.pt"
    )
    parser.add_argument("--input_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/processed_neural")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading detector from {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    backbone = state["config"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = OperatorClassifier(backbone, num_classes=4).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    print(f"Detector val_macro_f1: {state.get('val_macro_f1', 'n/a')}")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy the label_adj path expectation — train_jusdef.py reads from data/processed/label_adj.pt
    # so we re-use the shared one (no copy needed). build_graphs.py looks under input_dir.
    # For the neural pipeline we leave label_adj at its canonical location.

    summary = {}
    for split in ["train", "validation", "test"]:
        pkl_in = in_dir / f"{split}_processed.pkl"
        if not pkl_in.exists():
            print(f"SKIP {split}: {pkl_in} not found")
            continue

        with open(pkl_in, "rb") as f:
            docs = pickle.load(f)

        # Collect all (doc_idx, sec_idx, concept_idx, sentence) tuples
        triples = []
        sentences = []
        for di, doc in enumerate(docs):
            for si, sec in enumerate(doc["sections"]):
                text = sec.get("text", "")
                concepts = sec.get("concepts", [])
                for ci, c in enumerate(concepts):
                    span_start = c.get("span_start", 0)
                    sent = governing_sentence(text, span_start)
                    triples.append((di, si, ci))
                    sentences.append(sent)

        print(f"\n[{split}] {len(docs)} docs, {len(sentences)} concepts to re-label")
        if not sentences:
            with open(out_dir / f"{split}_processed.pkl", "wb") as f:
                pickle.dump(docs, f)
            summary[split] = {"docs": len(docs), "concepts": 0}
            continue

        preds = []
        chunk = 4096  # to avoid one huge tqdm step
        for i in tqdm(range(0, len(sentences), chunk), desc=f"{split} batches"):
            preds.extend(
                predict_batch(
                    model, tokenizer,
                    sentences[i : i + chunk],
                    device, args.max_len, args.batch_size,
                )
            )

        # Write predictions back
        prev_counts = Counter()
        new_counts = Counter()
        for (di, si, ci), p in zip(triples, preds):
            cur = docs[di]["sections"][si]["concepts"][ci].get("operator", "AFF")
            new = INT_TO_LABEL[p]
            prev_counts[cur] += 1
            new_counts[new] += 1
            docs[di]["sections"][si]["concepts"][ci]["operator"] = new

        pkl_out = out_dir / f"{split}_processed.pkl"
        with open(pkl_out, "wb") as f:
            pickle.dump(docs, f)
        print(f"  saved {pkl_out}")
        print(f"  keyword detector dist: {dict(prev_counts)}")
        print(f"  neural detector  dist: {dict(new_counts)}")
        summary[split] = {
            "docs": len(docs),
            "concepts": len(preds),
            "keyword_dist": dict(prev_counts),
            "neural_dist": dict(new_counts),
        }

    with open(out_dir / "relabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_dir / 'relabel_summary.json'}")


if __name__ == "__main__":
    main()
