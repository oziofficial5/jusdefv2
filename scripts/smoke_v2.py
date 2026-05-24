"""End-to-end smoke for v2: trains 5 epochs on 10 graphs.

Verifies the full forward+backward+optimizer loop runs with
F1/F1b/F2 fixes and patched authority features.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.trainer import train_jusdef


def main():
    print("Loading graphs for smoke_v2...", flush=True)

    train_graphs = torch.load(
        "data/processed/graphs/train_graphs.pt",
        map_location="cpu",
    )[:10]
    val_graphs = torch.load(
        "data/processed/graphs/validation_graphs.pt",
        map_location="cpu",
    )[:5]
    test_graphs = torch.load(
        "data/processed/graphs/test_graphs.pt",
        map_location="cpu",
    )[:5]

    label_adj = torch.eye(100)
    adj_path = Path("data/processed/label_adj.pt")
    if adj_path.exists():
        label_adj = torch.load(adj_path, map_location="cpu")

    config = {
        "seed": 42,
        "epochs": 5,
        "lr": 5e-4,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "patience": 100,
        "stage1_end": 50,
        "stage2_end": 100,
        "use_dmp": True,
        "use_authority": True,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "seen_labels": list(range(80)),
        "unseen_labels": list(range(80, 100)),
        "label_adj": label_adj,
        "checkpoint_path": "outputs/checkpoints/smoke_v2.pt",
    }

    results = train_jusdef(config)
    print("\nSMOKE RESULT:", results)
    print("If you see this and no crash, v2 forward+backward+optimizer all work.")


if __name__ == "__main__":
    main()