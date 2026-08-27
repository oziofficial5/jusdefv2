"""Train JusDef model."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from src.train.trainer import train_jusdef

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--stage1_end", type=int, default=50)
    parser.add_argument("--stage2_end", type=int, default=100)
    parser.add_argument("--no_dmp", action="store_true")
    parser.add_argument("--no_authority", action="store_true")
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument(
        "--graph_dir", type=str, default="data/processed/graphs",
        help="Directory containing {train,validation,test}_graphs.pt",
    )
    parser.add_argument(
        "--dmp_variant", type=str, default="hard",
        choices=["hard", "v3", "v4_hard", "v4_soft"],
        help="Defeat aggregator variant: 'hard' = v2 DMP, 'v3' = signal-preserving, "
             "'v4_hard' = density-routed (hard gate), 'v4_soft' = density-routed (learned gate)",
    )
    parser.add_argument("--v4_density_lo", type=float, default=0.10,
                        help="v4_hard: lower density bound for routing to v3 path")
    parser.add_argument("--v4_density_hi", type=float, default=0.20,
                        help="v4_hard: upper density bound for routing to v3 path")
    parser.add_argument("--v4_router_hidden_dim", type=int, default=32,
                        help="v4_soft: hidden dim of routing MLP")
    parser.add_argument("--v4_soft_init_bias", type=float, default=5.0,
                        help="v4_soft: initial bias for the routing logit")
    parser.add_argument(
        "--ablate", type=str, default="",
        help="Comma-separated list of v2 architectural corrections to switch OFF, "
             "for the correction-attribution study. Names are the mechanisms, not "
             "the thesis labels, to avoid collision with scripts/run_f1.sh: "
             "'reverse_edges' (thesis fix F1: drops the (conc,mentions_rev,sec) "
             "relation, so operator-aware concept updates no longer reach the "
             "section nodes); 'authority_grad' (thesis fix F2: takes the authority "
             "scorer out of the gradient path, equivalent to --no_authority); "
             "'ontology_edges' (thesis fix F3a: drops the (conc,ontology,conc) "
             "relation and the EuroVoc label adjacency). "
             "'all' switches off all three, which reconstructs the v1 workshop "
             "architecture under the corrected evaluation protocol.",
    )
    args = parser.parse_args()

    ablate = {a.strip() for a in args.ablate.split(",") if a.strip()}
    if "all" in ablate:
        ablate = {"reverse_edges", "authority_grad", "ontology_edges"}
    known = {"reverse_edges", "authority_grad", "ontology_edges"}
    unknown = ablate - known
    if unknown:
        raise ValueError(f"Unknown --ablate targets {sorted(unknown)}; "
                         f"choose from {sorted(known)} or 'all'.")
    if ablate:
        print(f"ABLATION: switching off {sorted(ablate)}")

    print(f"Loading graphs from {args.graph_dir}...")
    graph_dir = Path(args.graph_dir)
    train_graphs = torch.load(graph_dir / "train_graphs.pt", map_location="cpu")
    val_graphs = torch.load(graph_dir / "validation_graphs.pt", map_location="cpu")
    test_graphs = torch.load(graph_dir / "test_graphs.pt", map_location="cpu")

    if args.max_train > 0:
        train_graphs = train_graphs[:args.max_train]
        print(f"  Limited to {len(train_graphs)} train graphs")

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # --- architectural ablations (correction-attribution study) --------------
    # The model already skips any relation whose edge_index is empty (see the
    # `v.size(1) > 0` filter in src/model/jusdef.py), so a relation is removed
    # by emptying it rather than by touching model code.
    def _drop_relation(graphs, rel, what):
        dropped = touched = 0
        for g in graphs:
            if rel in g.edge_types:
                dropped += int(g[rel].edge_index.size(1))
                g[rel].edge_index = torch.zeros((2, 0), dtype=torch.long)
                touched += 1
        print(f"    {what}: emptied {rel} on {touched} graphs ({dropped} edges removed)")

    if "reverse_edges" in ablate:
        rel = ("conc", "mentions_rev", "sec")
        for name, gs in (("train", train_graphs), ("val", val_graphs), ("test", test_graphs)):
            _drop_relation(gs, rel, f"reverse_edges/{name}")
    if "ontology_edges" in ablate:
        rel = ("conc", "ontology", "conc")
        for name, gs in (("train", train_graphs), ("val", val_graphs), ("test", test_graphs)):
            _drop_relation(gs, rel, f"ontology_edges/{name}")

    adj_path = Path("data/processed/label_adj.pt")
    if "ontology_edges" in ablate:
        # Deliberately disabling F3a: identity adjacency, no label smoothing.
        # The guard below exists to stop this happening silently; here it is
        # the point of the run, so it is bypassed explicitly and logged.
        # Shape and dtype are taken from the real adjacency where it exists, so
        # the ablated run differs from the full run in exactly one respect.
        if adj_path.exists():
            _real = torch.load(adj_path, map_location="cpu")
            label_adj = torch.eye(_real.shape[0], dtype=_real.dtype)
        else:
            label_adj = torch.eye(100)
        print(f"  Label adjacency: ABLATED to identity {tuple(label_adj.shape)}, "
              "off-diag edges: 0 (F3a inactive by request)")
    else:
        if not adj_path.exists():
            raise FileNotFoundError(
                f"F3a label_adj missing at {adj_path}. "
                "Run scripts/build_label_adj.py before training. "
                "Falling back to identity would silently disable F3a."
            )
        label_adj = torch.load(adj_path, map_location="cpu")
        nnz_off_diag = ((label_adj != 0).sum() - label_adj.shape[0]).item()
        print(f"  Label adjacency: {label_adj.shape}, off-diag edges: {nnz_off_diag}")
        if nnz_off_diag == 0:
            raise ValueError(
                "label_adj has zero off-diagonal entries — F3a would be inactive."
            )

    tag = args.tag
    ckpt = f"outputs/checkpoints/jusdef_{tag}_s{args.seed}.pt"

    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "patience": args.patience,
        "stage1_end": args.stage1_end,
        "stage2_end": args.stage2_end,
        "use_dmp": not args.no_dmp,
        "use_authority": (not args.no_authority) and ("authority_grad" not in ablate),
        "dmp_variant": args.dmp_variant,
        "v4_density_lo": args.v4_density_lo,
        "v4_density_hi": args.v4_density_hi,
        "v4_router_hidden_dim": args.v4_router_hidden_dim,
        "v4_soft_init_bias": args.v4_soft_init_bias,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "seen_labels": list(range(80)),
        "unseen_labels": list(range(80, 100)),
        "label_adj": label_adj,
        "checkpoint_path": ckpt,
    }

    print(f"\n{'='*50}")
    print(f"JusDef: {tag} (seed={args.seed})")
    print(f"  DMP={config['use_dmp']}, Authority={config['use_authority']}")
    if ablate:
        print(f"  Ablated corrections: {', '.join(sorted(ablate))}")
        if ablate == {"reverse_edges", "authority_grad", "ontology_edges"}:
            print("  -> this is the v1 workshop architecture under the corrected protocol")
    print(f"{'='*50}\n")

    results = train_jusdef(config)

    if results:
        results["config"] = {"seed": args.seed, "tag": tag,
                             "use_dmp": config["use_dmp"],
                             "use_authority": config["use_authority"],
                             "graph_dir": args.graph_dir,
                             "ablated": sorted(ablate)}
        Path("outputs/logs").mkdir(parents=True, exist_ok=True)
        log = f"outputs/logs/jusdef_{tag}_s{args.seed}.json"
        with open(log, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {log}")

if __name__ == "__main__":
    main()