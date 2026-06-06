"""
Train R-GCN baseline on JusDef heterogeneous legal KG.

Usage:
  On Ampere (full):    python scripts/train_rgcn.py --seed 42
  On laptop (debug):   python scripts/train_rgcn.py --seed 42 --debug
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baselines import RGCN, tune_threshold
from src.preprocess.data_loader import load_eurlex, make_seen_unseen_split


def get_doc_and_label_embs(model, g, device):
    """
    Run forward pass on one graph.
    Doc embedding = mean of section embeddings after GNN.
    """
    x_dict = {nt: g[nt].x for nt in g.node_types}
    edge_index_dict = {}
    for et in g.edge_types:
        ei = g[et].edge_index
        if ei.size(1) > 0:
            edge_index_dict[et] = ei

    h = model(x_dict, edge_index_dict)

    # One doc per graph; pool over sections as document representation
    doc_emb = h["sec"].mean(dim=0, keepdim=True)   # [1, out_dim]
    label_embs = h["label"]                        # [100, out_dim]

    return doc_emb, label_embs


def train_one_epoch(model, graphs, seen_mask, optimizer, device):
    """Train for one epoch over all document graphs."""
    model.train()
    total_loss = 0.0
    n_graphs = 0

    seen_dev = seen_mask.to(device)

    for g in graphs:
        g = g.to(device)
        optimizer.zero_grad()

        doc_emb, label_embs = get_doc_and_label_embs(model, g, device)
        scores = model.score(doc_emb, label_embs).squeeze(0)  # [100]
        targets = g.y.to(device).float()                      # [100], multi-label

        # Only train on seen labels
        loss = F.binary_cross_entropy_with_logits(
            scores[seen_dev], targets[seen_dev]
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_graphs += 1

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, graphs, device, threshold=None):
    """
    If `threshold` is None, tunes a global threshold on the provided graphs.
    If `threshold` is a float, applies it as-is. Test eval MUST receive the
    val-tuned threshold to avoid optimistic bias.
    """
    model.eval()
    all_logits = []
    all_targets = []

    for g in graphs:
        g = g.to(device)
        doc_emb, label_embs = get_doc_and_label_embs(model, g, device)
        logits = model.score(doc_emb, label_embs).squeeze(0)
        all_logits.append(logits.cpu())
        all_targets.append(g.y.cpu())

    logits_np = torch.stack(all_logits).numpy()
    targets_np = torch.stack(all_targets).numpy()

    logits_clipped = np.clip(logits_np, -40, 40)
    probs_np = 1.0 / (1.0 + np.exp(-logits_clipped))

    if threshold is None:
        threshold, _ = tune_threshold(probs_np, targets_np)

    preds = (probs_np >= threshold).astype(int)
    macro = f1_score(targets_np, preds, average="macro", zero_division=0)
    micro = f1_score(targets_np, preds, average="micro", zero_division=0)

    return {
        "macro_f1": round(float(macro), 4),
        "micro_f1": round(float(micro), 4),
        "threshold": round(float(threshold), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train R-GCN baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Seed: {args.seed}")

    # Load graphs
    graph_dir = Path("data/processed/graphs")
    print("Loading graphs...")
    train_graphs = torch.load(graph_dir / "train_graphs.pt", map_location="cpu")
    val_graphs = torch.load(graph_dir / "validation_graphs.pt", map_location="cpu")
    test_graphs = torch.load(graph_dir / "test_graphs.pt", map_location="cpu")

    if args.debug:
        train_graphs = train_graphs[:50]
        val_graphs = val_graphs[:20]
        test_graphs = test_graphs[:20]

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, "
          f"Test: {len(test_graphs)}")

    # Seen/unseen split
    df_train = load_eurlex("train")
    seen, unseen = make_seen_unseen_split(df_train, n_unseen=20)
    seen_mask = torch.tensor([i in set(seen) for i in range(100)],
                             dtype=torch.bool)
    print(f"  Seen: {len(seen)}, Unseen: {len(unseen)}")

    # Model
    model = RGCN(
        in_dim=768,
        hidden_dim=args.hidden_dim,
        out_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training
    print("\nTraining...")
    best_val_f1 = -1.0
    patience_counter = 0
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_rgcn_seed{args.seed}.pt"

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_graphs, seen_mask, optimizer, device)
        val_m = evaluate(model, val_graphs, device)

        print(f"  Epoch {epoch:3d} | loss={loss:.4f} | "
              f"val_macro={val_m['macro_f1']:.4f} | "
              f"val_micro={val_m['micro_f1']:.4f}")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test — threshold is tuned on val and frozen for test
    print("\nTest evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val_m = evaluate(model, val_graphs, device)
    val_threshold = val_m["threshold"]
    test_m = evaluate(model, test_graphs, device, threshold=val_threshold)

    results = {
        "model": "RGCN",
        "seed": args.seed,
        "best_val_macro_f1": round(best_val_f1, 4),
        "val_macro_f1_at_best": val_m["macro_f1"],
        "val_threshold": val_threshold,
        "test_macro_f1": test_m["macro_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "dropout": args.dropout,
        },
    }

    print("\n" + "=" * 50)
    print("RESULTS: R-GCN Baseline")
    print("=" * 50)
    for k, v in results.items():
        if k != "config":
            print(f"  {k}: {v}")

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"baseline_rgcn_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to outputs/logs/baseline_rgcn_seed{args.seed}.json")


if __name__ == "__main__":
    main()