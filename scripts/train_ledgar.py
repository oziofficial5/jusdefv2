"""
Train JusDef-LEDGAR (paragraph-level operator-aware classifier).

Reads `data/processed_ledgar/{train,validation,test}_processed.pkl` produced
by `scripts/preprocess_ledgar.py`.

The architecture is single-label 100-class classification on LEDGAR contract
clauses using V3-layer operator-aware aggregation over sentence embeddings.

Comparison protocol:
- R-GCN-equivalent baseline: use `--dmp_variant mean` (no operator awareness)
- V3 main: `--dmp_variant v3`

Evaluation metrics: macro-F1, micro-F1 (test set).

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
    """
    Pack a list of paragraph dicts into a flat batched tensor structure.

    Returns:
        sent_embs: (TotalSents, 768)
        sent_to_para: (TotalSents,) integer paragraph id per sentence
        operators: (TotalSents,)
        labels: (B,)
        num_paragraphs: B
    """
    sent_embs_list = []
    sent_to_para_list = []
    ops_list = []
    labels = []
    for i, p in enumerate(batch):
        n = p["embeddings"].size(0)
        sent_embs_list.append(p["embeddings"])
        sent_to_para_list.append(torch.full((n,), i, dtype=torch.long))
        ops_list.append(torch.tensor(p["operators"], dtype=torch.long))
        labels.append(p["label"])

    sent_embs = torch.cat(sent_embs_list, dim=0)
    sent_to_para = torch.cat(sent_to_para_list, dim=0)
    operators = torch.cat(ops_list, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return sent_embs, sent_to_para, operators, labels_t, len(batch)


def iterate_minibatches(data, batch_size, shuffle=False, rng=None):
    indices = list(range(len(data)))
    if shuffle:
        if rng is None:
            random.shuffle(indices)
        else:
            rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch = [data[j] for j in indices[i : i + batch_size]]
        yield collate_batch(batch)


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    all_preds = []
    all_labels = []
    for sent_embs, sent_to_para, operators, labels_t, num_paras in iterate_minibatches(
        data, batch_size
    ):
        sent_embs = sent_embs.to(device)
        sent_to_para = sent_to_para.to(device)
        operators = operators.to(device)

        logits = model(sent_embs, sent_to_para, operators, num_paras)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels_t.tolist())

    macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return {"macro_f1": float(macro), "micro_f1": float(micro)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data/processed_ledgar")
    parser.add_argument("--tag", default="v3_pilot")
    parser.add_argument(
        "--dmp_variant", choices=["v3", "mean", "v4_hard", "v4_soft"], default="v3",
        help="'v3' = operator-aware (V3Layer); 'mean' = R-GCN-equivalent baseline; "
             "'v4_hard' = density-gated routing between v3 and mean (no learned router); "
             "'v4_soft' = learned per-paragraph gate between v3 and mean",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of V3 aggregation layers")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--v3_coef_reg_strength", type=float, default=0.01)
    parser.add_argument(
        "--v3_init_coefs", type=str, default="1.0,-1.0,-0.5,1.0",
        help="Comma-separated init values for AFF,NEG,EXC,OVR. Default '1.0,-1.0,-0.5,1.0'; "
             "ablation: '1.0,1.0,1.0,1.0' for unit-coef variant",
    )
    parser.add_argument(
        "--v3_hard_attention", action="store_true",
        help="Ablation: use one-hot argmax attention instead of soft softmax",
    )
    parser.add_argument(
        "--v3_shared_w_revert", action="store_true",
        help="Ablation: use per-operator W matrices instead of single shared W",
    )
    parser.add_argument("--v4_density_lo", type=float, default=0.10,
                        help="v4_hard: lower density bound for routing to v3")
    parser.add_argument("--v4_density_hi", type=float, default=0.20,
                        help="v4_hard: upper density bound for routing to v3")
    parser.add_argument("--v4_router_hidden_dim", type=int, default=32,
                        help="v4_soft: hidden dim of the routing MLP")
    parser.add_argument("--v4_soft_init_bias", type=float, default=5.0,
                        help="v4_soft: initial bias on the routing logit; "
                             "sigmoid(bias) is the initial gate value (~1 at default)")
    parser.add_argument("--max_train", type=int, default=0,
                        help="If > 0, limit training set size (for smoke)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seed: {args.seed} | Variant: {args.dmp_variant}")

    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading data from {data_dir}...")
    with open(data_dir / "train_processed.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir / "validation_processed.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir / "test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)

    if args.max_train > 0:
        train_data = train_data[: args.max_train]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Parse v3 init coefs
    v3_init_coefs = tuple(float(x) for x in args.v3_init_coefs.split(","))
    assert len(v3_init_coefs) == 4, "v3_init_coefs must be 4 comma-separated floats"

    # Model
    model = JusDefLEDGAR(
        in_dim=768,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dmp_variant=args.dmp_variant,
        v3_init_coefs=v3_init_coefs,
        v3_coef_reg_strength=args.v3_coef_reg_strength,
        v3_hard_attention=args.v3_hard_attention,
        v3_shared_w_revert=args.v3_shared_w_revert,
        v4_density_lo=args.v4_density_lo,
        v4_density_hi=args.v4_density_hi,
        v4_router_hidden_dim=args.v4_router_hidden_dim,
        v4_soft_init_bias=args.v4_soft_init_bias,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  V3 config: variant={args.dmp_variant}, layers={args.num_layers}, "
          f"init_coefs={v3_init_coefs}, reg={args.v3_coef_reg_strength}, "
          f"hard_attn={args.v3_hard_attention}, shared_w_revert={args.v3_shared_w_revert}")
    if args.dmp_variant == "v4_hard":
        print(f"  V4 hard router: density_lo={args.v4_density_lo}, "
              f"density_hi={args.v4_density_hi}")
    elif args.dmp_variant == "v4_soft":
        print(f"  V4 soft router: hidden_dim={args.v4_router_hidden_dim}, "
              f"init_bias={args.v4_soft_init_bias}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_macro = -1.0
    patience_counter = 0
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ledgar_{args.tag}_s{args.seed}.pt"

    # Per-epoch op_coef trajectory log (only meaningful for v3-family variants)
    op_coef_trajectory = []

    print("\nTraining...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for sent_embs, sent_to_para, operators, labels_t, num_paras in iterate_minibatches(
            train_data, args.batch_size, shuffle=True, rng=rng
        ):
            sent_embs = sent_embs.to(device)
            sent_to_para = sent_to_para.to(device)
            operators = operators.to(device)
            labels_t = labels_t.to(device)

            optimizer.zero_grad()
            logits = model(sent_embs, sent_to_para, operators, num_paras)
            ce_loss = F.cross_entropy(logits, labels_t)
            reg = model.v3_coef_regulariser()
            loss = ce_loss + reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += ce_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_data, device, args.batch_size)

        print(
            f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
            f"val_macro={val_metrics['macro_f1']:.4f} | "
            f"val_micro={val_metrics['micro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Log per-epoch op_coef trajectory for v3-family variants
        if args.dmp_variant in ("v3", "v4_hard", "v4_soft"):
            entry = {"epoch": epoch}
            for i, layer in enumerate(model.agg_layers):
                if hasattr(layer, "op_coef"):
                    coef = layer.op_coef.detach().cpu().numpy()
                    entry[f"layer{i}_AFF"] = float(coef[0])
                    entry[f"layer{i}_NEG"] = float(coef[1])
                    entry[f"layer{i}_EXC"] = float(coef[2])
                    entry[f"layer{i}_OVR"] = float(coef[3])
            op_coef_trajectory.append(entry)

    # Test eval with best checkpoint
    print("\nTest evaluation (loading best checkpoint)...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_data, device, args.batch_size)

    # Inspect final V3 op_coef values (v3 family includes v4 variants)
    op_coef_log = None
    if args.dmp_variant in ("v3", "v4_hard", "v4_soft"):
        op_coef_log = []
        for i, layer in enumerate(model.agg_layers):
            op_coef_log.append({
                "layer": i,
                "AFF": float(layer.op_coef[0].item()),
                "NEG": float(layer.op_coef[1].item()),
                "EXC": float(layer.op_coef[2].item()),
                "OVR": float(layer.op_coef[3].item()),
            })

    results = {
        "best_val_macro_f1": round(float(best_val_macro), 4),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 4),
        "test_micro_f1": round(float(test_metrics["micro_f1"]), 4),
        "config": {
            "seed": args.seed,
            "tag": args.tag,
            "dmp_variant": args.dmp_variant,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "epochs_planned": args.epochs,
            "v3_coef_reg_strength": args.v3_coef_reg_strength,
        },
        "v3_op_coef": op_coef_log,
        "op_coef_trajectory": op_coef_trajectory,
    }

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ledgar_{args.tag}_s{args.seed}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"RESULTS: LEDGAR JusDef ({args.dmp_variant}, seed={args.seed})")
    print("=" * 50)
    print(f"  best_val_macro_f1: {results['best_val_macro_f1']:.4f}")
    print(f"  test_macro_f1:     {results['test_macro_f1']:.4f}")
    print(f"  test_micro_f1:     {results['test_micro_f1']:.4f}")
    if op_coef_log is not None:
        print(f"  Final V3 coefs (layer 0): "
              f"AFF={op_coef_log[0]['AFF']:.3f} "
              f"NEG={op_coef_log[0]['NEG']:.3f} "
              f"EXC={op_coef_log[0]['EXC']:.3f} "
              f"OVR={op_coef_log[0]['OVR']:.3f}")
    print(f"\nSaved to {log_path}")


if __name__ == "__main__":
    main()
