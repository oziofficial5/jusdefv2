"""
Two-stage training for v4_hard on LEDGAR.

Motivation: the single-stage v4_hard suffered from training-time distribution
shift. The classifier saw two different feature distributions (mean for ~65%
of paragraphs that route to the mean path; v3 for ~17% that route to v3).
The classifier under-fit the minority v3 distribution and consequently
produced low macro-F1 on the 10-20% bin at inference (0.534 vs v3's 0.682).

The counterfactual sensitivity analysis (analyse_counterfactual_sensitivity.py)
established that v4_hard's v3 sub-architecture is doing the right thing
mechanistically -- it concentrates operator sensitivity on the 10-20% bin
(true_shift 0.25, vs 0.013 for v3_pilot on the same bin). The classifier
failure is a training-protocol issue, not an architectural one.

Two-stage fix:
    Stage 1: train a standard v3 LEDGAR model end-to-end (already done,
             reuse the existing ledgar_v3_pilot_s{seed}.pt checkpoints).
    Stage 2: build a v4_hard model with:
                 - V3Layer weights initialised from the Stage-1 v3 checkpoint
                   and FROZEN (no gradient through V3Layer)
                 - Input projection FROZEN (matches Stage-1 features)
                 - Hard density gate (no learned parameters)
                 - Classifier head fine-tuned from scratch (or warm-started
                   from v3's classifier)
             The classifier now sees a STATIONARY feature distribution at
             training time (gate is observable from input density), so no
             distribution shift.

Expected outcome:
    Aggregate LEDGAR macro-F1: matches mean baseline (~0.70) because most
        paragraphs route to mean
    10-20% bin macro-F1: matches v3 (~0.68) because the v3 sub-architecture
        weights are frozen from a converged v3 model
    Counterfactual sensitivity on 10-20% bin: should match v4_hard's existing
        0.25 true-shift since the architecture is identical there

If this works, v4_twostage is the architecture C9 was supposed to deliver:
    - Regime-routed (operator-awareness concentrated on the regime)
    - Aggregate-competitive (no parameter-overhead penalty outside the regime)
    - Same parameter count as v3 (no additional capacity needed for the win)

Usage:
    python scripts/train_v4_twostage.py --seed 42 --tag v4_twostage \\
        --pretrained_v3 outputs/checkpoints/ledgar_v3_pilot_s42.pt

Output:
    outputs/checkpoints/ledgar_<tag>_s<seed>.pt
    outputs/logs/ledgar_<tag>_s<seed>.json
"""
import os
import sys
import json
import pickle
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


def collate_batch(batch):
    sent_embs_list, sent_to_para_list, ops_list, labels = [], [], [], []
    for i, p in enumerate(batch):
        n = p["embeddings"].size(0)
        sent_embs_list.append(p["embeddings"])
        sent_to_para_list.append(torch.full((n,), i, dtype=torch.long))
        ops_list.append(torch.tensor(p["operators"], dtype=torch.long))
        labels.append(p["label"])
    return (
        torch.cat(sent_embs_list, dim=0),
        torch.cat(sent_to_para_list, dim=0),
        torch.cat(ops_list, dim=0),
        torch.tensor(labels, dtype=torch.long),
        len(batch),
    )


def iterate_minibatches(data, batch_size, shuffle=False, rng=None):
    indices = list(range(len(data)))
    if shuffle:
        (rng or random).shuffle(indices)
    for i in range(0, len(indices), batch_size):
        yield collate_batch([data[j] for j in indices[i:i + batch_size]])


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    all_preds, all_labels = [], []
    for sent_embs, sent_to_para, operators, labels_t, num_paras in \
            iterate_minibatches(data, batch_size):
        sent_embs = sent_embs.to(device)
        sent_to_para = sent_to_para.to(device)
        operators = operators.to(device)
        logits = model(sent_embs, sent_to_para, operators, num_paras)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels_t.tolist())
    return {
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(all_labels, all_preds, average="micro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data/processed_ledgar")
    parser.add_argument("--tag", default="v4_twostage")
    parser.add_argument("--pretrained_v3", required=True,
                        help="Path to converged ledgar_v3_pilot_s<seed>.pt checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--v4_density_lo", type=float, default=0.10)
    parser.add_argument("--v4_density_hi", type=float, default=0.20)
    parser.add_argument("--warm_start_classifier", action="store_true",
                        help="Warm-start the classifier from the pretrained v3's classifier")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seed: {args.seed} | Variant: v4_twostage")
    print(f"Pretrained v3: {args.pretrained_v3}")

    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading data from {data_dir}...")
    with open(data_dir / "train_processed.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir / "validation_processed.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir / "test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Build v4_hard model
    model = JusDefLEDGAR(
        in_dim=768,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dmp_variant="v4_hard",
        v4_density_lo=args.v4_density_lo,
        v4_density_hi=args.v4_density_hi,
    ).to(device)

    # Load Stage-1 v3 weights and copy into the v4 model
    print(f"\nStage 1 -> Stage 2: loading {args.pretrained_v3}")
    v3_state = torch.load(args.pretrained_v3, map_location=device)

    # Map v3 keys onto v4 model. v4 has the same V3Layer + input_proj + classifier
    # as v3 (it's just v3 with a HardDensityRouter wrapped around the update).
    model_state = model.state_dict()
    copied = 0
    skipped = []
    for k, v in v3_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            copied += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    print(f"  Copied {copied} tensors from v3; skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped (first 5): {skipped[:5]}")

    # Freeze input_proj and V3Layer; train only the classifier
    print("\nStage 2: freezing v3 sub-architecture; training classifier head")
    for p in model.input_proj.parameters():
        p.requires_grad = False
    for layer in model.agg_layers:
        for p in layer.parameters():
            p.requires_grad = False
    # router (HardDensityRouter) has no parameters to freeze
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"  Frozen params:    {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    if not args.warm_start_classifier:
        # Re-initialise classifier head from scratch (default)
        nn.init.kaiming_uniform_(model.classifier.weight, a=5**0.5)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print("  Classifier head: re-initialised from scratch")
    else:
        print("  Classifier head: warm-started from pretrained v3")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ledgar_{args.tag}_s{args.seed}.pt"

    best_val_macro = -1.0
    patience_counter = 0

    print("\nTraining (classifier-head only)...")
    for epoch in range(args.epochs):
        model.train()
        # Keep frozen modules in eval mode so dropout/BN behave as at inference
        model.input_proj.eval()
        for layer in model.agg_layers:
            layer.eval()
        total_loss, n_batches = 0.0, 0

        for sent_embs, sent_to_para, operators, labels_t, num_paras in \
                iterate_minibatches(train_data, args.batch_size, shuffle=True, rng=rng):
            sent_embs = sent_embs.to(device)
            sent_to_para = sent_to_para.to(device)
            operators = operators.to(device)
            labels_t = labels_t.to(device)

            optimizer.zero_grad()
            logits = model(sent_embs, sent_to_para, operators, num_paras)
            loss = F.cross_entropy(logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_data, device, args.batch_size)
        print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"val_macro={val_metrics['macro_f1']:.4f} | "
              f"val_micro={val_metrics['micro_f1']:.4f}", flush=True)

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test eval
    print("\nTest evaluation (loading best checkpoint)...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_data, device, args.batch_size)

    results = {
        "best_val_macro_f1": round(float(best_val_macro), 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_micro_f1": round(test_metrics["micro_f1"], 4),
        "config": {
            "seed": args.seed,
            "tag": args.tag,
            "dmp_variant": "v4_twostage",
            "pretrained_v3": str(args.pretrained_v3),
            "v4_density_lo": args.v4_density_lo,
            "v4_density_hi": args.v4_density_hi,
            "warm_start_classifier": args.warm_start_classifier,
            "epochs_planned": args.epochs,
        },
    }

    log_path = Path("outputs/logs") / f"ledgar_{args.tag}_s{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"RESULTS: LEDGAR v4_twostage (seed={args.seed})")
    print("=" * 50)
    print(f"  best_val_macro_f1: {results['best_val_macro_f1']:.4f}")
    print(f"  test_macro_f1:     {results['test_macro_f1']:.4f}")
    print(f"  test_micro_f1:     {results['test_micro_f1']:.4f}")
    print(f"\nSaved to {log_path}")


if __name__ == "__main__":
    main()
