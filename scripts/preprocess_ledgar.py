"""
Preprocess LEDGAR (LexGLUE) for JusDef-LEDGAR training.

For each LEDGAR instance:
  1. Sentence-segment the paragraph
  2. LegalBERT [CLS] embedding for each sentence
  3. Neural operator detector prediction for each sentence

Output:
  data/processed_ledgar/{train,validation,test}_processed.pkl

Each entry: {
  "label": int (0..99),
  "sentences": list of strings,
  "operators": list of int (0=AFF, 1=NEG, 2=EXC, 3=OVR),
  "embeddings": torch.Tensor (N_sents, 768),
}

This script reuses the trained `operator_detector_neural.pt` from EUR-Lex
work — the architectural claim is that operator detection trained on
EUR-Lex transfers to LEDGAR (English legal text in both).
"""
import os
import re
import sys
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


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
        return self.head(self.dropout(out.last_hidden_state[:, 0]))


def sentence_split(text):
    """Conservative sentence splitter — same as preprocess_all (EUR-Lex)."""
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


@torch.no_grad()
def embed_and_detect(sentences, tokenizer, encoder, detector, device,
                     batch_size=64, max_len=128):
    """
    Run LegalBERT encoder + operator detector in batched fashion.

    Returns:
        embeddings: (N, hidden) tensor
        operators: (N,) tensor of operator ids
    """
    all_embs = []
    all_ops = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        # Embedder: use encoder backbone for sentence embeddings
        encoder_out = encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls = encoder_out.last_hidden_state[:, 0]  # (B, hidden)
        all_embs.append(cls.cpu())

        # Operator detector
        det_logits = detector(input_ids, attn_mask)
        ops = det_logits.argmax(dim=-1).cpu()
        all_ops.append(ops)

    return torch.cat(all_embs, dim=0), torch.cat(all_ops, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector_ckpt", default="outputs/checkpoints/operator_detector_neural.pt",
        help="Path to trained neural operator detector",
    )
    parser.add_argument(
        "--output_dir", default="data/processed_ledgar",
        help="Output directory for processed pickles",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for embedding and detection",
    )
    parser.add_argument(
        "--max_len", type=int, default=128,
        help="Max token length per sentence",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Process only first 50 instances per split (smoke test)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load the trained neural detector and its backbone
    print(f"Loading neural operator detector from {args.detector_ckpt}", flush=True)
    state = torch.load(args.detector_ckpt, map_location=device)
    backbone_name = state["config"]["backbone"]
    print(f"  Backbone: {backbone_name}", flush=True)
    print(f"  Detector val macro F1: {state.get('val_macro_f1', 'n/a')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    # Shared backbone for both embedding extraction and detection
    encoder = AutoModel.from_pretrained(backbone_name).to(device).eval()
    detector = OperatorClassifier(backbone_name, num_classes=4).to(device).eval()
    detector.load_state_dict(state["state_dict"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n{'=' * 50}", flush=True)
        print(f"Processing LEDGAR {split}", flush=True)
        print(f"{'=' * 50}", flush=True)

        ds = load_dataset("lex_glue", "ledgar", split=split)
        if args.debug:
            ds = ds.select(range(min(50, len(ds))))
        print(f"  {len(ds)} instances", flush=True)

        processed = []
        op_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for inst in tqdm(ds, desc=split):
            text = inst.get("text", inst.get("contract", ""))
            label = inst["label"]

            sentences = sentence_split(text)
            if not sentences:
                # Skip empty paragraphs
                continue

            embs, ops = embed_and_detect(
                sentences, tokenizer, encoder, detector, device,
                batch_size=args.batch_size, max_len=args.max_len,
            )

            processed.append({
                "label": int(label),
                "sentences": sentences,
                "operators": ops.tolist(),
                "embeddings": embs,  # (N_sents, 768)
            })

            for op in ops.tolist():
                op_counts[op] = op_counts.get(op, 0) + 1

        # Save
        out_path = out_dir / f"{split}_processed.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)

        total_ops = sum(op_counts.values())
        print(f"\n  Saved {len(processed)} paragraphs to {out_path}", flush=True)
        print(f"  Total sentences: {total_ops}", flush=True)
        for op_id, op_name in INT_TO_LABEL.items():
            n = op_counts.get(op_id, 0)
            pct = n / max(total_ops, 1) * 100
            print(f"    {op_name}: {n} ({pct:.2f}%)", flush=True)
        print(f"  Non-AFF density: "
              f"{(total_ops - op_counts.get(0, 0)) / max(total_ops, 1) * 100:.2f}%",
              flush=True)


if __name__ == "__main__":
    main()
