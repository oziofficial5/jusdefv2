"""Train JusDef model."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from src.train.trainer import train_jusdef

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--stage1_end", type=int, default=50)
    parser.add_argument("--stage2_end", type=int, default=100)
    parser.add_argument("--no_dmp", action="store_true")
    parser.add_argument("--no_authority", action="store_true")
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--max_train", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading graphs...")
    graph_dir = Path("data/processed/graphs")
    train_graphs = torch.load(graph_dir / "train_graphs.pt", map_location="cpu")
    val_graphs = torch.load(graph_dir / "validation_graphs.pt", map_location="cpu")
    test_graphs = torch.load(graph_dir / "test_graphs.pt", map_location="cpu")

    if args.max_train > 0:
        train_graphs = train_graphs[:args.max_train]
        print(f"  Limited to {len(train_graphs)} train graphs")

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    adj_path = Path("data/processed/label_adj.pt")
    if not adj_path.exists():
        raise FileNotFoundError(
            f"F3a label_adj missing at {adj_path}. "
            "Run scripts/build_label_adj.py before training. "
            "Falling back to identity would silently disable F3a."
        )
    label_adj = torch.load(adj_path, map_location="cpu")
    nnz_off_diag = ((label_adj != 0).sum() - label_adj.shape[0]).item()
    print(f"  Label adjacency: {label_adj.shape}, off-diag edges: {nnz_off_diag}")
    if nnz_off_diag == 0:
        raise ValueError(
            "label_adj has zero off-diagonal entries — F3a would be inactive."
        )

    tag = args.tag
    ckpt = f"outputs/checkpoints/jusdef_{tag}_s{args.seed}.pt"

    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "patience": args.patience,
        "stage1_end": args.stage1_end,
        "stage2_end": args.stage2_end,
        "use_dmp": not args.no_dmp,
        "use_authority": not args.no_authority,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "seen_labels": list(range(80)),
        "unseen_labels": list(range(80, 100)),
        "label_adj": label_adj,
        "checkpoint_path": ckpt,
    }

    print(f"\n{'='*50}")
    print(f"JusDef: {tag} (seed={args.seed})")
    print(f"  DMP={not args.no_dmp}, Authority={not args.no_authority}")
    print(f"{'='*50}\n")

    results = train_jusdef(config)

    if results:
        results["config"] = {"seed": args.seed, "tag": tag,
                             "use_dmp": not args.no_dmp,
                             "use_authority": not args.no_authority}
        Path("outputs/logs").mkdir(parents=True, exist_ok=True)
        log = f"outputs/logs/jusdef_{tag}_s{args.seed}.json"
        with open(log, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {log}")

if __name__ == "__main__":
    main()