"""
Fine-tune LegalBERT on LEDGAR as the transformer-baseline reference.

The thesis's mean-aggregation, R-GCN, v3, and v4 results all use frozen
LegalBERT sentence embeddings as input features. The single most-asked
reviewer baseline is: what if we fine-tune the full LegalBERT model
end-to-end? This script provides that comparison.

Implementation: AutoModelForSequenceClassification on LegalBERT
(nlpaueb/legal-bert-base-uncased), tokenising each LEDGAR paragraph as a
single sequence (truncated to 512 tokens), standard transformer
fine-tuning recipe. Five seeds for direct multi-seed comparability with
the v3/v4 results.

Usage:
    python scripts/train_legalbert_ledgar.py --seed 42 --tag legalbert_ft
    # ...etc for seeds 43-46.

Output:
    outputs/checkpoints/legalbert_ledgar_ft_s{seed}.pt
    outputs/logs/legalbert_ledgar_ft_s{seed}.json
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset


class LEDGARDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return {
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro",
                                    zero_division=0)),
        "micro_f1": float(f1_score(all_labels, all_preds, average="micro",
                                    zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="legalbert_ft")
    parser.add_argument("--backbone", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seed: {args.seed}", flush=True)

    print(f"Loading LEDGAR via lex_glue...", flush=True)
    ds_train = load_dataset("lex_glue", "ledgar", split="train")
    ds_val = load_dataset("lex_glue", "ledgar", split="validation")
    ds_test = load_dataset("lex_glue", "ledgar", split="test")
    print(f"  train: {len(ds_train)}, val: {len(ds_val)}, test: {len(ds_test)}",
          flush=True)

    print(f"Loading tokenizer + model {args.backbone}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=args.num_classes,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    train_ds = LEDGARDataset(ds_train["text"], ds_train["label"],
                              tokenizer, args.max_length)
    val_ds = LEDGARDataset(ds_val["text"], ds_val["label"],
                            tokenizer, args.max_length)
    test_ds = LEDGARDataset(ds_test["text"], ds_test["label"],
                             tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"legalbert_ledgar_{args.tag}_s{args.seed}.pt"

    best_val_macro = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch} | loss={avg_loss:.4f} | "
              f"val_macro={val_metrics['macro_f1']:.4f} | "
              f"val_micro={val_metrics['micro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    # Test eval
    print(f"\nLoading best checkpoint for test eval...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    results = {
        "best_val_macro_f1": round(best_val_macro, 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_micro_f1": round(test_metrics["micro_f1"], 4),
        "config": {
            "seed": args.seed, "tag": args.tag, "backbone": args.backbone,
            "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
            "max_length": args.max_length, "num_classes": args.num_classes,
        },
    }
    log_path = Path("outputs/logs") / f"legalbert_ledgar_{args.tag}_s{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50, flush=True)
    print(f"RESULTS: LegalBERT-FT LEDGAR (seed={args.seed})", flush=True)
    print("=" * 50, flush=True)
    print(f"  best_val_macro_f1: {results['best_val_macro_f1']:.4f}", flush=True)
    print(f"  test_macro_f1:     {results['test_macro_f1']:.4f}", flush=True)
    print(f"  test_micro_f1:     {results['test_micro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
