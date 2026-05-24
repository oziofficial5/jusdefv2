"""End-to-end smoke test for JusDef v2 on a tiny subset."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.trainer import train_jusdef  # adjust if your entry point differs


def main():
    graph_dir = Path("data/processed/graphs")

    train_graphs = torch.load(graph_dir / "train_graphs.pt", map_location="cpu")[:10]
    val_graphs = torch.load(graph_dir / "validation_graphs.pt", map_location="cpu")[:5]
    test_graphs = torch.load(graph_dir / "test_graphs.pt", map_location="cpu")[:5]

    label_adj_path = Path("data/processed/label_adj.pt")
    if label_adj_path.exists():
        label_adj = torch.load(label_adj_path, map_location="cpu")
    else:
        label_adj = torch.eye(100)

    config = {
        "seed": 42,
        "epochs": 5,
        "lr": 5e-4,
        "hiddendim": 256,      # smaller for CPU
        "numlayers": 2,
        "dropout": 0.3,
        "temperature": 5.0,
        "lambda1": 0.1,
        "lambda2": 0.1,
        "patience": 100,       # avoid early stop in smoke
        "stage1end": 50,
        "stage2end": 100,
        "usedmp": True,
        "useauthority": True,
        "traingraphs": train_graphs,
        "valgraphs": val_graphs,
        "testgraphs": test_graphs,
        "seenlabels": list(range(80)),
        "unseenlabels": list(range(80, 100)),
        "labeladj": label_adj,
        "checkpointpath": "outputs/checkpoints/smoke_v2.pt",
    }

    results = train_jusdef(config)
    print("\nSmoke results:", results)
    print("\nIf you see this and no crash, v2 trains end-to-end on laptop.")


if __name__ == "__main__":
    main()