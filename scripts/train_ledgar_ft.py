"""
F2 — end-to-end fine-tuned encoder + operator-aware aggregation head.

The frozen-embedding experiments (train_ledgar.py) cap out ~10 macro-F1 below a
fine-tuned transformer. F2 puts LegalBERT IN the training loop and asks the
question that actually matters for the thesis:

    Does an operator-aware aggregation head add benefit ON TOP of a fine-tuned
    encoder — especially in the 10-20% non-AFF density regime?

To make this a clean ablation, the encoder is fine-tuned over SENTENCES and the
ONLY thing that changes between conditions is the aggregation head:

    --agg mean      : fine-tuned encoder + mean pooling          (the FT control)
    --agg v3        : fine-tuned encoder + ungated v3 aggregation
    --agg v4_soft   : fine-tuned encoder + baseline-anchored gate (F1-style)

Same encoder, same data (sentences + operators from data/processed_ledgar), same
recipe — so any gap is attributable to the operator head, not the encoder.

Compares directly against:
  - within-framework FT+mean control (this script, --agg mean)
  - the existing paragraph-level LegalBERT-FT baseline (train_legalbert_ledgar.py)

Output:
  outputs/checkpoints/ledgar_ft_<agg>_s<seed>.pt
  outputs/logs/ledgar_ft_<agg>_s<seed>.json   (aggregate + 10-20% bin macro-F1)

Usage:
  python scripts/train_ledgar_ft.py --agg mean    --seed 42
  python scripts/train_ledgar_ft.py --agg v4_soft --seed 42 --v4_soft_init_bias -5.0
"""
import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from src.model.jusdef_ledgar import JusDefLEDGAR


class F2Model(nn.Module):
    """Fine-tuned sentence encoder + JusDefLEDGAR aggregation head."""

    def __init__(self, backbone, agg, hidden_dim=512, num_classes=100,
                 v4_soft_init_bias=-5.0, freeze_bottom=0, aux_op_weight=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        if freeze_bottom > 0:
            # freeze embeddings + bottom N transformer layers to save compute
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_bottom]:
                for p in layer.parameters():
                    p.requires_grad = False
        self.head = JusDefLEDGAR(
            in_dim=self.encoder.config.hidden_size, hidden_dim=hidden_dim,
            num_classes=num_classes, num_layers=1, dmp_variant=agg,
            v4_soft_init_bias=v4_soft_init_bias,
        )
        # F4: auxiliary per-sentence operator-prediction head (multi-task).
        # Forces the fine-tuned encoder to STAY operator-aware; tests whether
        # that recovers any regime benefit once the encoder is fine-tuned.
        self.aux_op_head = (nn.Linear(self.encoder.config.hidden_size, 4)
                            if aux_op_weight > 0 else None)

    def forward(self, input_ids, attn_mask, sent_to_para, operators, num_paragraphs,
                return_aux=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls = out.last_hidden_state[:, 0]            # (TotalSents, 768)
        logits = self.head(cls, sent_to_para, operators, num_paragraphs)
        if return_aux and self.aux_op_head is not None:
            return logits, self.aux_op_head(cls)
        return logits

    def coef_reg(self):
        return self.head.v3_coef_regulariser()


def make_batches(data, batch_size, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [data[j] for j in idx[i:i + batch_size]]


def collate(batch, tokenizer, max_len, device):
    sents, sent_to_para, ops, labels = [], [], [], []
    for i, p in enumerate(batch):
        s = p["sentences"] if p["sentences"] else [""]
        sents.extend(s)
        sent_to_para.extend([i] * len(s))
        o = list(p["operators"])
        if len(o) != len(s):                         # safety: align lengths
            o = (o + [0] * len(s))[:len(s)]
        ops.extend(o)
        labels.append(p["label"])
    enc = tokenizer(sents, truncation=True, padding=True, max_length=max_len,
                    return_tensors="pt")
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.tensor(sent_to_para, dtype=torch.long, device=device),
            torch.tensor(ops, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long), len(batch))


@torch.no_grad()
def predict_all(model, data, tokenizer, max_len, device, batch_size):
    model.eval()
    preds, labels = [], []
    for batch in make_batches(data, batch_size, False, None):
        ii, am, s2p, op, lab, npar = collate(batch, tokenizer, max_len, device)
        logits = model(ii, am, s2p, op, npar)
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def macro(labels, preds, mask=None):
    if mask is not None:
        labels, preds = labels[mask], preds[mask]
        present = sorted(set(labels.tolist()))
        return f1_score(labels, preds, labels=present, average="macro", zero_division=0)
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg", choices=["mean", "v3", "v4_soft"], default="mean")
    ap.add_argument("--tag", default=None, help="default: ft_<agg>")
    ap.add_argument("--data_dir", default="data/processed_ledgar")
    ap.add_argument("--backbone", default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--v4_soft_init_bias", type=float, default=-5.0)
    ap.add_argument("--aux_op_weight", type=float, default=0.0,
                    help="F4: weight on auxiliary per-sentence operator-prediction "
                         "loss (0 = off). Forces encoder to stay operator-aware.")
    ap.add_argument("--freeze_bottom", type=int, default=0,
                    help="freeze embeddings + bottom N encoder layers (speed)")
    ap.add_argument("--bin_lo", type=float, default=0.10)
    ap.add_argument("--bin_hi", type=float, default=0.20)
    ap.add_argument("--max_train", type=int, default=0, help=">0 limits train (smoke)")
    args = ap.parse_args()
    tag = args.tag or f"ft_{args.agg}"

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | seed {args.seed} | agg {args.agg} | tag {tag}", flush=True)

    dd = Path(args.data_dir)
    train = pickle.load(open(dd / "train_processed.pkl", "rb"))
    val = pickle.load(open(dd / "validation_processed.pkl", "rb"))
    test = pickle.load(open(dd / "test_processed.pkl", "rb"))
    if args.max_train > 0:
        train = train[:args.max_train]
    print(f"  train {len(train)} val {len(val)} test {len(test)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = F2Model(args.backbone, args.agg, args.hidden_dim, args.num_classes,
                    args.v4_soft_init_bias, args.freeze_bottom,
                    aux_op_weight=args.aux_op_weight).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}", flush=True)

    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    optim = torch.optim.AdamW([
        {"params": enc_p, "lr": args.encoder_lr},
        {"params": head_p, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    steps = (len(train) // args.batch_size + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * steps), steps)

    ckpt = Path("outputs/checkpoints"); ckpt.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt / f"ledgar_{tag}_s{args.seed}.pt"
    best_val, patience = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in make_batches(train, args.batch_size, True, rng):
            ii, am, s2p, op, lab, npar = collate(batch, tokenizer, args.max_len, device)
            lab = lab.to(device)
            optim.zero_grad()
            if args.aux_op_weight > 0:
                logits, aux = model(ii, am, s2p, op, npar, return_aux=True)
                loss = (F.cross_entropy(logits, lab) + model.coef_reg()
                        + args.aux_op_weight * F.cross_entropy(aux, op))
            else:
                logits = model(ii, am, s2p, op, npar)
                loss = F.cross_entropy(logits, lab) + model.coef_reg()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            tot += loss.item(); nb += 1
        vp, vl = predict_all(model, val, tokenizer, args.max_len, device, args.batch_size)
        vmac = macro(vl, vp)
        print(f"  epoch {epoch} | loss {tot/max(nb,1):.4f} | val_macro {vmac:.4f}", flush=True)
        if vmac > best_val:
            best_val = vmac; patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch}", flush=True); break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    tp, tl = predict_all(model, test, tokenizer, args.max_len, device, args.batch_size)
    density = np.array([sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
                        for d in test])
    mask = (density >= args.bin_lo) & (density < args.bin_hi)
    res = {
        "best_val_macro_f1": round(float(best_val), 4),
        "test_macro_f1": round(float(macro(tl, tp)), 4),
        "test_micro_f1": round(float(f1_score(tl, tp, average="micro", zero_division=0)), 4),
        "test_macro_f1_10_20": round(float(macro(tl, tp, mask)), 4),
        "n_bin": int(mask.sum()),
        "config": {"seed": args.seed, "agg": args.agg, "tag": tag,
                   "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
                   "epochs": args.epochs, "freeze_bottom": args.freeze_bottom,
                   "v4_soft_init_bias": args.v4_soft_init_bias,
                   "aux_op_weight": args.aux_op_weight},
        "test_preds": tp.tolist(),          # for later bootstrap
    }
    out = Path("outputs/logs") / f"ledgar_{tag}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("\n" + "=" * 52)
    print(f"RESULTS: F2 {tag} seed {args.seed}")
    print(f"  test_macro_f1      : {res['test_macro_f1']:.4f}")
    print(f"  test_micro_f1      : {res['test_micro_f1']:.4f}")
    print(f"  test_macro_f1_10_20: {res['test_macro_f1_10_20']:.4f}  (n={res['n_bin']})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
