"""Add synthetic auth_type/auth_level/auth_recency to r2 edges in cached graphs.

For laptop Phase 0 and smoke_v2 only — real features will be produced by
kg_builder.py on Ampere during full graph construction.
"""
import torch
import numpy as np
from pathlib import Path

PATHS = [
    "data/processed/graphs/test_graphs.pt",
    "data/processed/graphs/validation_graphs.pt",
    "data/processed/graphs/train_graphs.pt",
]

R2 = ("sec", "mentions", "conc")
R2_REV = ("conc", "mentions_rev", "sec")


def patch_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"SKIP {path} (missing)")
        return

    graphs = torch.load(p, map_location="cpu")
    rng = np.random.RandomState(42)

    for g in graphs:
        for et in (R2, R2_REV):
            if et not in g.edge_types:
                continue
            store = g[et]
            n = store.edge_index.size(1)
            if n == 0:
                store.auth_type = torch.zeros(0, dtype=torch.long)
                store.auth_level = torch.zeros(0, dtype=torch.float)
                store.auth_recency = torch.zeros(0, dtype=torch.float)
                continue
            store.auth_type = torch.tensor(
                rng.randint(0, 6, size=n), dtype=torch.long
            )
            store.auth_level = torch.tensor(
                rng.uniform(0.0, 1.0, size=n), dtype=torch.float
            )
            store.auth_recency = torch.tensor(
                rng.uniform(0.0, 1.0, size=n), dtype=torch.float
            )

    torch.save(graphs, p)
    print(f"PATCHED {path} with synthetic auth features ({len(graphs)} graphs)")


if __name__ == "__main__":
    for path in PATHS:
        patch_file(path)