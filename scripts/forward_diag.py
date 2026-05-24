import torch, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.model.jusdef import JusDef

graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
g = graphs[0]
r2_key = ("sec", "mentions", "conc")

torch.manual_seed(42)
model = JusDef(
    in_dim=768,
    hidden_dim=512,
    num_layers=2,
    use_dmp=True,
    use_authority=True,
).eval()

def fwd(graph):
    x_dict = {nt: graph[nt].x for nt in graph.node_types}
    ei_dict = {et: graph[et].edge_index for et in graph.edge_types}
    r2 = graph[r2_key]
    eattr = {
        r2_key: {
            "operator": r2.operator,
            "priority": r2.priority,
            "auth_type": r2.auth_type,
            "auth_level": r2.auth_level,
            "auth_recency": r2.auth_recency,
        }
    }
    with torch.no_grad():
        h, _ = model(x_dict, ei_dict, eattr)
    return h

h_orig = fwd(g)

g2 = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")[0]
rng = np.random.RandomState(7)
g2[r2_key].operator = torch.tensor(
    rng.randint(0, 4, size=g2[r2_key].operator.numel()),
    dtype=torch.long,
)
h_rand = fwd(g2)

for nt in h_orig:
    d = (h_orig[nt] - h_rand[nt]).abs().max().item()
    print(f"  {nt}: max |Δ| = {d:.6e}")