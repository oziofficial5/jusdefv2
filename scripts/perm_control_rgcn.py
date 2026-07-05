"""
Relational demonstration of the permutation control: decompose R-GCN's
relation-type gain on the exact benchmarks R-GCN was validated on
(AIFB, MUTAG; Schlichtkrull et al. 2018).

The structural signal here is the per-edge RELATION TYPE (the closest analog to
JusDef's typed operator edges). Three arms, identical model capacity
(num_relations kept fixed), only the relation signal changes:
  true    : R-GCN with the real edge relation types
  shuffled: R-GCN with globally permuted relation types
            (marginal relation distribution preserved EXACTLY, edge<->relation
             alignment destroyed)
  zeroed  : R-GCN with all edges set to a single relation (relation info removed)

Decomposition of the relation-awareness gain:
  genuine relation semantics = acc(true)     - acc(shuffled)
  relation inductive bias    = acc(shuffled) - acc(zeroed)
  total gain                 = acc(true)     - acc(zeroed)
Diagnostic signature = sign(shuffled - zeroed): negative => relation types are
genuine mechanism; positive => having multiple relations helps even misassigned
(inductive bias). Parallels the Cora/Citeseer/Pubmed and LEDGAR results.

Usage:
    python scripts/perm_control_rgcn.py --datasets AIFB MUTAG --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse, math
import statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden, num_classes, num_relations, num_bases=30):
        super().__init__()
        self.c1 = FastRGCNConv(num_nodes, hidden, num_relations, num_bases=num_bases)
        self.c2 = FastRGCNConv(hidden, num_classes, num_relations, num_bases=num_bases)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.c1(None, edge_index, edge_type))
        return self.c2(x, edge_index, edge_type)


def train_eval(model, data, edge_type, device, epochs=50, lr=0.01, wd=5e-4):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ei = data.edge_index.to(device)
    et = edge_type.to(device)
    tri, trY = data.train_idx.to(device), data.train_y.to(device)
    tei, teY = data.test_idx.to(device), data.test_y.to(device)
    best = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(ei, et)
        F.cross_entropy(out[tri], trY).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(ei, et).argmax(-1)
            acc = (pred[tei] == teY).float().mean().item()
            best = max(best, acc)
    return best


def shuffle_types(et, seed):
    g = torch.Generator().manual_seed(seed)
    return et[torch.randperm(et.size(0), generator=g)]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    return m, se, (m / se if se > 0 else float("nan")), sum(1 for x in d if x > 0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["AIFB", "MUTAG"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--root", default="data/entities")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for name in args.datasets:
        ds = Entities(root=f"{args.root}/{name}", name=name)
        data = ds[0]
        nrel, ncls, nnod = ds.num_relations, ds.num_classes, data.num_nodes
        et_true = data.edge_type
        et_zero = torch.zeros_like(et_true)
        acc_t, acc_s, acc_z = [], [], []
        for s in args.seeds:
            torch.manual_seed(s)
            acc_t.append(train_eval(RGCN(nnod, args.hidden, ncls, nrel, args.bases), data, et_true, device))
            torch.manual_seed(s)
            acc_s.append(train_eval(RGCN(nnod, args.hidden, ncls, nrel, args.bases), data, shuffle_types(et_true, s), device))
            torch.manual_seed(s)
            acc_z.append(train_eval(RGCN(nnod, args.hidden, ncls, nrel, args.bases), data, et_zero, device))
        mt, ms, mz = st.mean(acc_t), st.mean(acc_s), st.mean(acc_z)
        gen_m, gen_se, gen_t, gen_pos, n = paired(acc_t, acc_s)
        tot_m, *_ = paired(acc_t, acc_z)
        bias_m, *_ = paired(acc_s, acc_z)
        print("=" * 66)
        print(f" {name}  ({n} seeds, {nrel} relations, {ncls} classes)")
        print("=" * 66)
        print(f"  R-GCN (true relations)     acc = {mt:.4f}")
        print(f"  R-GCN (shuffled relations) acc = {ms:.4f}")
        print(f"  R-GCN (single relation)    acc = {mz:.4f}")
        print(f"  ---")
        print(f"  total relation gain (true - single)      = {tot_m:+.4f}")
        print(f"  relation inductive bias (shuf - single)  = {bias_m:+.4f}")
        print(f"  genuine relation semantics (true - shuf) = {gen_m:+.4f} "
              f"(SE {gen_se:.4f}, t={gen_t:.2f}, {gen_pos}/{n} seeds)")
        if gen_m < 0.03 and abs(bias_m) < 0.03:
            sig = "NO/weak relation effect (genuine ~0; like LEDGAR/MUTAG)"
        elif bias_m < 0:
            sig = "GENUINE relations (misassigned relations worse than one relation)"
        else:
            sig = "inductive bias (multiple relations help even misassigned)"
        print(f"  diagnostic signature (shuf - single) = {bias_m:+.4f}  ->  {sig}\n")

    print("Interpretation: the same true/shuffled/zeroed control decomposes R-GCN's")
    print("relation-type advantage on its own benchmarks, extending the diagnostic from")
    print("graph edges (Cora/Citeseer/Pubmed) and legal operators (LEDGAR) to typed edges.")


if __name__ == "__main__":
    main()
