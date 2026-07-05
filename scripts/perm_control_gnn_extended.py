"""
Extended methodology-generalisation sweep: the true/shuffled/zeroed permutation
control across MANY node-classification datasets, spanning homophilous graphs
(where structure is expected to be genuine) and heterophilous graphs (where
structure is known to help little). If the diagnostic is real, it should report
LARGE genuine structure on homophilous graphs and SMALL/near-zero on heterophilous
ones -- placing them on the same spectrum as legal operators (mostly inductive
bias). This hardens the "general diagnostic" claim.

All datasets use a uniform seeded random 60/20/20 node split so the true/shuffled/
MLP comparison is internally consistent across datasets (the genuine effect,
true-shuffled, is split-robust). Each dataset wrapped in try/except so one failure
does not abort the sweep.

Usage:
    python scripts/perm_control_gnn_extended.py --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/perm_control_gnn_extended.py --datasets Cora Chameleon Actor
"""
import argparse, math, traceback
import statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

HOMOPHILOUS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers", "CS", "Physics", "WikiCS"]
HETEROPHILOUS = ["Chameleon", "Squirrel", "Actor", "Cornell", "Texas", "Wisconsin",
                 "Roman-empire", "Amazon-ratings", "Minesweeper"]


def load(name, root):
    from torch_geometric.datasets import (Planetoid, Amazon, Coauthor, WikiCS,
                                          WebKB, WikipediaNetwork, Actor,
                                          HeterophilousGraphDataset)
    n = name.lower()
    if n in ("cora", "citeseer", "pubmed"):
        return Planetoid(f"{root}/planetoid", name)[0]
    if n in ("photo", "computers"):
        return Amazon(f"{root}/amazon", name.capitalize())[0]
    if n in ("cs", "physics"):
        return Coauthor(f"{root}/coauthor", name.upper() if n == "cs" else name.capitalize())[0]
    if n == "wikics":
        return WikiCS(f"{root}/wikics")[0]
    if n in ("cornell", "texas", "wisconsin"):
        return WebKB(f"{root}/webkb", name.capitalize())[0]
    if n in ("chameleon", "squirrel"):
        return WikipediaNetwork(f"{root}/wikipedia", name)[0]
    if n == "actor":
        return Actor(f"{root}/actor")[0]
    if n in ("roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"):
        return HeterophilousGraphDataset(f"{root}/hetero", name)[0]
    raise ValueError(f"unknown dataset {name}")


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


def rand_masks(n, seed, tr=0.6, va=0.2):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    ntr, nva = int(tr * n), int(va * n)
    m = lambda idx: torch.zeros(n, dtype=torch.bool).scatter_(0, idx, True)
    return m(perm[:ntr]), m(perm[ntr:ntr + nva]), m(perm[ntr + nva:])


def train_eval(model, x, y, ei, masks, device, epochs=200, lr=0.01, wd=5e-4):
    trm, vam, tem = masks
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = best_test = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, ei)[trm], y[trm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(-1)
            va = (pred[vam] == y[vam]).float().mean().item()
            te = (pred[tem] == y[tem]).float().mean().item()
            if va > best_val:
                best_val, best_test = va, te
    return best_test


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = st.mean(d)
    se = (st.stdev(d) / math.sqrt(n)) if n > 1 else 0.0
    return m, se, (m / se if se > 0 else float("nan")), sum(1 for x in d if x > 0), n


def run_one(name, seeds, hidden, root, device):
    data = load(name, root)
    x, y, ei = data.x.float(), data.y.long(), data.edge_index
    if y.dim() > 1:
        y = y.squeeze()
    ind, out, N = x.size(1), int(y.max()) + 1, x.size(0)
    x, y, ei = x.to(device), y.to(device), ei.to(device)
    at, as_, am = [], [], []
    for s in seeds:
        masks = tuple(m.to(device) for m in rand_masks(N, s))
        g = torch.Generator().manual_seed(s + 1000)
        ei_shuf = torch.randperm(N, generator=g).to(device)[ei]
        torch.manual_seed(s); at.append(train_eval(GCN(ind, hidden, out), x, y, ei, masks, device))
        torch.manual_seed(s); as_.append(train_eval(GCN(ind, hidden, out), x, y, ei_shuf, masks, device))
        torch.manual_seed(s); am.append(train_eval(MLP(ind, hidden, out), x, y, None, masks, device))
    mt, ms, mm = st.mean(at), st.mean(as_), st.mean(am)
    gen_m, gen_se, gen_t, gen_pos, n = paired(at, as_)
    bias_m, *_ = paired(as_, am)
    return mt, ms, mm, gen_m, gen_se, gen_t, gen_pos, n, bias_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=HOMOPHILOUS + HETEROPHILOUS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--root", default="data/gnn_bench")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    print(f"{'dataset':<16}{'GCN':>7}{'shuf':>7}{'MLP':>7}{'genuine':>10}{'t':>7}{'sig(shuf-MLP)':>15}  group")
    print("-" * 82)
    for name in args.datasets:
        grp = "homoph" if name in HOMOPHILOUS else "hetero"
        try:
            mt, ms, mm, gen_m, gen_se, gen_t, gen_pos, n, bias_m = run_one(
                name, args.seeds, args.hidden, args.root, device)
            print(f"{name:<16}{mt:>7.3f}{ms:>7.3f}{mm:>7.3f}{gen_m:>+10.3f}{gen_t:>7.1f}"
                  f"{bias_m:>+15.3f}  {grp}", flush=True)
        except Exception as e:
            print(f"{name:<16}  FAILED: {e.__class__.__name__}: {e}", flush=True)
            traceback.print_exc()
    print("\nExpected: homophilous graphs show LARGE genuine structure (true-shuf); "
          "heterophilous graphs show SMALL/near-zero genuine effect, placing them on the "
          "same spectrum as legal operators (mostly inductive bias).")


if __name__ == "__main__":
    main()
