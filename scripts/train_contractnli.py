"""
Operator-aware ContractNLI classifier (Pivot B): does the JusDef operator prior help
on a task where defeat determines the label?

Model: hypothesis-conditioned attention over contract sentences, then a 3-way head.
The ONLY difference between the two variants is the aggregation:
  --agg mean : hypothesis-attention over sentences (no operator awareness)   [baseline]
  --agg v3   : same attention + per-operator signed coefficients c_omega      [operator-aware]
So any gap is attributable to the operator prior, on a task (unlike LEDGAR clause
typing) where exceptions/negations flip Entailment<->Contradiction.

Reads data/processed_contractnli/{train,validation,test}_processed.pkl
(from preprocess_contractnli.py). Reports macro-F1 over 3 classes, per-class F1,
and a "mentioned-only" macro-F1 over {Entailment, Contradiction} --- the subset
where defeasibility matters most.

Output: outputs/checkpoints/cnli_<agg>_s<seed>.pt , outputs/logs/cnli_<agg>_s<seed>.json
"""
import os, sys, json, pickle, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class CNLIModel(nn.Module):
    def __init__(self, in_dim=768, hidden=256, agg="v3", num_classes=3, dropout=0.3,
                 v3_init=(1.0, -1.0, -0.5, 1.0), reg=0.01):
        super().__init__()
        self.agg = agg
        self.reg = reg
        self.sent_proj = nn.Linear(in_dim, hidden)
        self.hyp_proj = nn.Linear(in_dim, hidden)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        if agg == "v3":
            self.coef = nn.Parameter(torch.tensor(v3_init, dtype=torch.float))
            self.register_buffer("coef0", torch.tensor(v3_init, dtype=torch.float))
        self.cls = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def coef_reg(self):
        if self.agg == "v3":
            return self.reg * ((self.coef - self.coef0) ** 2).sum()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, sent_embs, ops, hyp_emb):
        # sent_embs [Ns,768], ops LongTensor[Ns], hyp_emb [768]
        hs = F.relu(self.sent_proj(sent_embs))                 # [Ns,H]
        hh = F.relu(self.hyp_proj(hyp_emb)).unsqueeze(0)        # [1,H]
        z = self.attn(torch.cat([hs, hh.expand(hs.size(0), -1)], -1)).squeeze(-1)
        alpha = torch.softmax(z, dim=0)                         # [Ns]
        if self.agg == "v3":
            w = alpha * self.coef[ops]                          # signed by operator
        else:
            w = alpha
        doc = (w.unsqueeze(-1) * hs).sum(0)                     # [H]
        h = hh.squeeze(0)
        feats = torch.cat([doc, h, (doc - h).abs(), doc * h])   # [4H]
        return self.cls(feats)                                  # [num_classes]


def load_split(d, name):
    return pickle.load(open(Path(d) / f"{name}_processed.pkl", "rb"))


def evaluate(model, data, device):
    model.eval()
    docs, hyp, ex = data["docs"], data["hyp_emb"], data["examples"]
    preds, labels = [], []
    with torch.no_grad():
        for e in ex:
            doc = docs[e["doc"]]
            se = doc["embeddings"].to(device)
            op = torch.tensor(doc["operators"], dtype=torch.long, device=device)
            he = hyp[e["hyp"]].to(device)
            preds.append(int(model(se, op, he).argmax(-1).item()))
            labels.append(e["label"])
    preds, labels = np.array(preds), np.array(labels)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    per = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    m = np.isin(labels, [1, 2])   # mentioned-only (Entailment vs Contradiction)
    ment = f1_score(labels[m], preds[m], labels=[1, 2], average="macro", zero_division=0) if m.sum() else 0.0
    return {"macro_f1": float(macro), "f1_NM": float(per[0]), "f1_Ent": float(per[1]),
            "f1_Con": float(per[2]), "mentioned_macro_f1": float(ment)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg", choices=["mean", "v3"], default="v3")
    ap.add_argument("--data_dir", default="data/processed_contractnli")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} | seed {args.seed} | agg {args.agg}", flush=True)

    train = load_split(args.data_dir, "train")
    val = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    print(f"  train ex {len(train['examples'])} | val {len(val['examples'])} | test {len(test['examples'])}", flush=True)

    # class weights for imbalance (NotMentioned dominates)
    from collections import Counter
    lc = Counter(e["label"] for e in train["examples"])
    tot = sum(lc.values())
    cw = torch.tensor([tot / (3 * max(lc[i], 1)) for i in range(3)], dtype=torch.float, device=device)

    model = CNLIModel(hidden=args.hidden, agg=args.agg, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    docs, hyp, ex = train["docs"], train["hyp_emb"], train["examples"]

    ckpt = Path("outputs/checkpoints"); ckpt.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt / f"cnli_{args.agg}_s{args.seed}.pt"
    best_val, patience = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        idx = list(range(len(ex))); rng.shuffle(idx)
        tot_loss = nb = 0
        for i in range(0, len(idx), args.batch_size):
            opt.zero_grad()
            loss = 0.0
            for j in idx[i:i + args.batch_size]:
                e = ex[j]; doc = docs[e["doc"]]
                se = doc["embeddings"].to(device)
                op = torch.tensor(doc["operators"], dtype=torch.long, device=device)
                he = hyp[e["hyp"]].to(device)
                logit = model(se, op, he).unsqueeze(0)
                lab = torch.tensor([e["label"]], device=device)
                loss = loss + F.cross_entropy(logit, lab, weight=cw)
            loss = loss / len(idx[i:i + args.batch_size]) + model.coef_reg()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot_loss += float(loss); nb += 1
        vm = evaluate(model, val, device)
        print(f"  epoch {epoch} loss {tot_loss/max(nb,1):.4f} | val_macro {vm['macro_f1']:.4f} "
              f"| val_mentioned {vm['mentioned_macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ {epoch}", flush=True); break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    res = evaluate(model, test, device)
    res.update({"best_val_macro_f1": round(best_val, 4), "seed": args.seed, "agg": args.agg})
    if args.agg == "v3":
        res["coef"] = model.coef.detach().cpu().tolist()
    out = Path("outputs/logs") / f"cnli_{args.agg}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("\n" + "=" * 50)
    print(f"RESULTS ContractNLI {args.agg} seed {args.seed}")
    for k in ["macro_f1", "mentioned_macro_f1", "f1_NM", "f1_Ent", "f1_Con"]:
        print(f"  {k}: {res[k]:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
