# scripts/patch_test_graphs_reverse_edge.py
import torch
from pathlib import Path

path = Path("data/processed/graphs/test_graphs.pt")
graphs = torch.load(path, map_location="cpu")

r2_key = ("sec", "mentions", "conc")
rev_key = ("conc", "mentions_rev", "sec")

for g in graphs:
    if r2_key in g.edge_types:
        ei = g[r2_key].edge_index
        if ei.size(1) > 0:
            src = ei[0]
            dst = ei[1]
            g[rev_key].edge_index = torch.stack([dst, src], dim=0)
            g[rev_key].operator = g[r2_key].operator.clone()
            g[rev_key].priority = g[r2_key].priority.clone()
        else:
            g[rev_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
            g[rev_key].operator = torch.zeros(0, dtype=torch.long)
            g[rev_key].priority = torch.zeros(0, dtype=torch.float)
    else:
        # ensure key exists anyway
        g[rev_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g[rev_key].operator = torch.zeros(0, dtype=torch.long)
        g[rev_key].priority = torch.zeros(0, dtype=torch.float)

torch.save(graphs, path)
print("Patched test_graphs.pt with mentions_rev edges")