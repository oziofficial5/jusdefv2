"""
Fine-tune Legal-BERT on the 3000-sentence operator annotation dataset.

Input:  data/annotations/operator_labels_3000.jsonl
        ({"sentence": "...", "label": "AFF"|"NEG"|"EXC"|"OVR", ...})

Output: outputs/checkpoints/operator_detector_neural.pt
        outputs/logs/operator_detector_training.json

The trained model is used by scripts/relabel_operators_neural.py to re-label
the operator tag on every concept across train/validation/test pickles.

Hold-out: 90/10 random split (stratified by label). Reports macro-F1 on
held-out 10% and saves the best-on-held-out checkpoint.

Reference: Chapter 1, contribution C5 / RQ4-extension.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModel


LABEL_TO_INT = {"AFF": 0, "NEG": 1, "EXC": 2, "OVR": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class OperatorDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class OperatorClassifier(nn.Module):
    def __init__(self, backbone_name="nlpaueb/legal-bert-base-uncased", num_classes=4, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls))


def load_annotations(path):
    sents, labels = [], []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("sentence") or d.get("text") or ""
            lab = d.get("label") or d.get("ai_label") or d.get("operator")
            if not text or lab not in LABEL_TO_INT:
                skipped += 1
                continue
            sents.append(text)
            labels.append(LABEL_TO_INT[lab])
    print(f"Loaded {len(sents)} sentences ({skipped} skipped).")
    print(f"Label distribution: {Counter(INT_TO_LABEL[l] for l in labels)}")
    return sents, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/annotations/operator_labels_3000.jsonl")
    parser.add_argument("--backbone", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument(
        "--ckpt", default="outputs/checkpoints/operator_detector_neural.pt"
    )
    parser.add_argument(
        "--log", default="outputs/logs/operator_detector_training.json"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sents, labels = load_annotations(args.data)
    train_s, val_s, train_y, val_y = train_test_split(
        sents, labels, test_size=args.val_frac, random_state=args.seed, stratify=labels
    )
    print(f"Train: {len(train_s)}  Val: {len(val_s)}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = OperatorDataset(train_s, train_y, tokenizer, args.max_len)
    val_ds = OperatorDataset(val_s, val_y, tokenizer, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = OperatorClassifier(args.backbone, num_classes=4).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            optim.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = F.cross_entropy(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_dl))

        model.eval()
        val_preds, val_gold = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                pred = logits.argmax(dim=-1).cpu().tolist()
                val_preds.extend(pred)
                val_gold.extend(batch["label"].tolist())

        macro = f1_score(val_gold, val_preds, average="macro", zero_division=0)
        report = classification_report(
            val_gold, val_preds,
            labels=[0, 1, 2, 3],
            target_names=["AFF", "NEG", "EXC", "OVR"],
            output_dict=True, zero_division=0,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_macro_f1={macro:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": macro, "report": report})

        if macro > best_val_f1:
            best_val_f1 = macro
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "backbone": args.backbone,
                        "num_classes": 4,
                        "max_len": args.max_len,
                    },
                    "val_macro_f1": macro,
                    "epoch": epoch,
                },
                args.ckpt,
            )
            print(f"  saved checkpoint to {args.ckpt}")

    with open(args.log, "w") as f:
        json.dump(
            {
                "best_val_macro_f1": round(float(best_val_f1), 4),
                "history": history,
                "config": vars(args),
            },
            f,
            indent=2,
        )
    print(f"\nBest val macro-F1: {best_val_f1:.4f}")
    print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
