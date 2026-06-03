"""Build EuroVoc-derived label adjacency matrix from official 2-digit
broader-domain grouping.

Each label has a name of the form 'XXXX <descriptor>' where XXXX is
the EuroVoc microthesaurus code. Labels sharing the same first 2 digits
belong to the same broader EuroVoc domain (e.g. domain 20 = trade).

Output: data/processed/label_adj.pt — symmetric [100, 100] float32 tensor.
"""
import json
import re
from collections import Counter
from pathlib import Path

import torch

names_path = Path("data/annotations/eurovoc_label_names.json")
out_path = Path("data/processed/label_adj.pt")

with open(names_path, encoding="utf-8") as f:
    names = json.load(f)

n_labels = len(names)
assert n_labels == 100, f"Expected 100 labels, got {n_labels}"

domain = [None] * n_labels
code4 = [None] * n_labels

for k, entry in names.items():
    idx = int(k)
    name = entry["name"]
    m = re.match(r"^(\d{4})\s+(.+)$", name)
    assert m, f"Could not parse label {k}: {name}"
    code4[idx] = m.group(1)
    domain[idx] = m.group(1)[:2]

assert all(d is not None for d in domain), "Some labels were not assigned a domain"

adj = torch.zeros((n_labels, n_labels), dtype=torch.float32)
for i in range(n_labels):
    for j in range(n_labels):
        if i != j and domain[i] == domain[j]:
            adj[i, j] = 1.0

n_edges = int(adj.sum().item())
print(f"Labels: {n_labels}")
print(f"Total label-to-label edges: {n_edges}")
print(f"Average out-degree per label: {n_edges / n_labels:.2f}")
print(f"Matrix is symmetric: {torch.equal(adj, adj.T)}")
print(f"Diagonal is zero: {torch.equal(torch.diag(adj), torch.zeros(n_labels))}")

domain_counts = Counter(domain)
print("\nDomain block sizes:")
for d, c in sorted(domain_counts.items()):
    expected_edges = c * (c - 1)  # directed edges i->j, j->i
    print(f"  Domain {d}: {c} labels -> {expected_edges} directed edges")

print("\nSample parsed labels:")
for i in range(min(10, n_labels)):
    raw = names[str(i)]["name"]
    print(f"  {i}: code={code4[i]}, domain={domain[i]}, name={raw}")

out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(adj, out_path)
print(f"\nSaved to {out_path}")

print("\nSample 15x15 block:")
print(adj[:15, :15].numpy())