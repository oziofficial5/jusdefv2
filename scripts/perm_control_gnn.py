"""
Generalisation demonstration for the capacity-vs-mechanism permutation control.

The thesis introduces an operator-permutation decomposition (true / shuffled /
zeroed) that separates the genuine-mechanism share of an architectural gain from
its inductive-bias share. This script shows the same methodology is a GENERAL
diagnostic tool by applying it to a canonical, published GNN claim --- "graph
structure helps node classification" --- on Cora, Citeseer and Pubmed.

Three arms (structural signal = the graph edges):
  true    : GCN on the real graph                         (structure aligned to content)
  shuffled: GCN on a node-permuted graph (edge (u,v) -> (pi(u), pi(v)));
            topology/degree distribution preserved EXACTLY, but the alignment
            between structure and node features/labels is destroyed
  zeroed  : MLP (no edges)                                 (no graph at all)

Decomposition of the GCN gain over the MLP on each dataset:
  genuine structure-alignment = acc(true)     - acc(shuffled)
  topology / inductive bias   = acc(shuffled) - acc(MLP)
  total GCN gain              = acc(true)     - acc(MLP)

This parallels the LEDGAR three-arm result and demonstrates the method transfers
to foundational GNN architectures on standard benchmarks. Cheap: each run is
seconds; the whole sweep is minutes.

Usage:
    python scripts/perm_control_gnn.py --datasets Cora Citeseer Pubmed --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse, math
import statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, ind, hid, out, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(ind, hid); self.c2 = GCNConv(hid, out); self.dp = dropout

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, self.dp, self.training)
        return self.c2(x, ei)


class MLP(nn.Module):
    def __init__(self, ind, hid, out, dropout=0.5):
        super().__init__()
        self.l1 = nn.Linear(ind, hid); self.l2 = nn.Linear(hid, out); self.dp = dropout

    def forward(self, x, ei=None):
        x = F.relu(self.l1(x)); x = F.dropout(x, self.dp, self.training)
        return self.l2(x)


def train_eval(model, data, ei, device, epochs=200, lr=0.01, wd=5e-4):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    x, y = data.x.to(device), data.y.to(device)
    ei = ei.to(device) if ei is not None else None
    trm, vam, tem = data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device)
    best_val = best_test = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        F.cross_entropy(out[trm], y[trm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(-1)
            va = (pred[vam] == y[vam]).float().mean().item()
            te = (pred[tem] == y[tem]).float().mean().item()
            if va > best_val:
                best_val, best_test = va, te
    return best_test


def perm_edges(ei, n, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[ei]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = m / se if se > 0 else float("nan")
    return m, se, t, sum(1 for x in d if x > 0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer", "Pubmed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--root", default="data/planetoid")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for name in args.datasets:
        ds = Planetoid(root=args.root, name=name)
        data = ds[0]
        ind, out, N = ds.num_features, ds.num_classes, data.num_nodes
        acc_t, acc_s, acc_m = [], [], []
        for s in args.seeds:
            torch.manual_seed(s)
            acc_t.append(train_eval(GCN(ind, args.hidden, out), data, data.edge_index, device))
            torch.manual_seed(s)
            acc_s.append(train_eval(GCN(ind, args.hidden, out), data, perm_edges(data.edge_index, N, s), device))
            torch.manual_seed(s)
            acc_m.append(train_eval(MLP(ind, args.hidden, out), data, None, device))
        mt, ms, mm = st.mean(acc_t), st.mean(acc_s), st.mean(acc_m)
        gen_m, gen_se, gen_t, gen_pos, n = paired(acc_t, acc_s)   # genuine structure
        tot_m, tot_se, _, _, _ = paired(acc_t, acc_m)             # total GCN gain
        top_m, _, _, _, _ = paired(acc_s, acc_m)                  # topology / bias
        frac = gen_m / tot_m if tot_m != 0 else float("nan")
        print("=" * 66)
        print(f" {name}  ({n} seeds)")
        print("=" * 66)
        print(f"  GCN (true edges)     acc = {mt:.4f}")
        print(f"  GCN (shuffled edges) acc = {ms:.4f}")
        print(f"  MLP (no edges)       acc = {mm:.4f}")
        print(f"  ---")
        print(f"  total GCN gain (true - MLP)          = {tot_m:+.4f}")
        print(f"  topology/inductive bias (shuf - MLP) = {top_m:+.4f}")
        print(f"  genuine structure (true - shuf)      = {gen_m:+.4f} "
              f"(SE {gen_se:.4f}, t={gen_t:.2f}, {gen_pos}/{n} seeds)")
        print(f"  genuine-structure share of gain      = {frac*100:.0f}%\n")

    print("Interpretation: the same true/shuffled/zeroed decomposition used for legal")
    print("operators cleanly separates, on canonical benchmarks, how much of a GNN's")
    print("advantage is genuine structure-content alignment vs. topology/inductive bias.")


if __name__ == "__main__":
    main()
