"""
JusDef training loop with staged training.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from src.model.jusdef import JusDef
from src.model.baselines import tune_threshold
from src.train.losses import JusDefLoss


def get_stage(epoch, stage1_end=50, stage2_end=100):
    if epoch < stage1_end:
        return 1
    elif epoch < stage2_end:
        return 2
    else:
        return 3


def forward_one_graph(model, g, device):
    g = g.to(device)
    x_dict = {nt: g[nt].x for nt in g.node_types}
    ei_dict = {et: g[et].edge_index for et in g.edge_types}

    r2_key = ("sec", "mentions", "conc")
    edge_attr_dict = None
    if r2_key in g.edge_types and g[r2_key].edge_index.size(1) > 0:
        store = g[r2_key]
        attrs = {
            "operator": store.operator,
            "priority": store.priority,
        }
        for k in ("auth_type", "auth_level", "auth_recency"):
            if hasattr(store, k):
                attrs[k] = getattr(store, k)
        edge_attr_dict = {r2_key: attrs}

    h, defeat_info = model(x_dict, ei_dict, edge_attr_dict)

    # Document embedding: pool over section embeddings
    doc_emb = model.pool_document(h["sec"])
    scores = model.score(doc_emb, h["label"])  # logits [1, 100]

    conc_embs = h.get("conc", None)
    return scores, defeat_info, conc_embs


def train_one_epoch(
    model,
    graphs,
    seen_mask,
    label_adj,
    criterion,
    optimizer,
    device,
    stage,
):
    model.train()
    total_loss = 0.0
    n_graphs = 0

    seen_mask = seen_mask.to(device)
    label_adj = label_adj.to(device) if label_adj is not None else None

    for g in graphs:
        optimizer.zero_grad()

        scores, defeat_info, conc_embs = forward_one_graph(model, g, device)
        targets = g.y.to(device).unsqueeze(0).float()  # [1, 100]

        active_embs = None
        defeated_embs = None
        if defeat_info is not None:
            active_embs = defeat_info["active_embs"]
            defeated_embs = defeat_info["defeated_embs"]

        loss = criterion(
            scores,          # logits [1, 100]
            targets,         # targets [1, 100]
            seen_mask,
            conc_embs=conc_embs,
            label_adj=label_adj,
            active_embs=active_embs,
            defeated_embs=defeated_embs,
            stage=stage,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_graphs += 1

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, graphs, device):
    model.eval()
    all_scores = []
    all_targets = []

    for g in graphs:
        scores, _, _ = forward_one_graph(model, g, device)  # logits [1, 100]
        all_scores.append(scores.cpu().squeeze(0))          # [100]
        all_targets.append(g.y.cpu())                       # [100]

    logits_np = torch.stack(all_scores).numpy()    # [N_docs, 100]
    targets_np = torch.stack(all_targets).numpy()  # [N_docs, 100]

    # Numerically stable sigmoid on logits
    logits_clipped = np.clip(logits_np, -40, 40)
    probs_np = 1.0 / (1.0 + np.exp(-logits_clipped))

    # Tune threshold on probabilities
    best_thresh, _ = tune_threshold(probs_np, targets_np)
    preds = (probs_np >= best_thresh).astype(int)

    macro = f1_score(targets_np, preds, average="macro", zero_division=0)
    micro = f1_score(targets_np, preds, average="micro", zero_division=0)

    return {
        "macro_f1": round(float(macro), 4),
        "micro_f1": round(float(micro), 4),
        "threshold": round(float(best_thresh), 4),
    }


def train_jusdef(config):
    print("  [DEBUG] train_jusdef entered")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [DEBUG] device={device}")

    model = JusDef(
        in_dim=768,
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        temperature=config.get("temperature", 5.0),
        use_dmp=config.get("use_dmp", True),
        use_authority=config.get("use_authority", True),
    ).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 5e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    criterion = JusDefLoss(
        lambda1=config.get("lambda1", 0.1),
        lambda2=config.get("lambda2", 0.1),
    )

    seen_set = set(config["seen_labels"])
    seen_mask = torch.tensor([i in seen_set for i in range(100)], dtype=torch.bool)
    label_adj = config.get("label_adj", None)

    best_val_f1 = -1.0
    patience_counter = 0
    checkpoint_path = Path(
        config.get("checkpoint_path", "outputs/checkpoints/best_jusdef.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    max_epochs = config.get("epochs", 200)
    print(f"  [DEBUG] starting loop, {max_epochs} epochs")

    for epoch in range(max_epochs):
        stage = get_stage(
            epoch,
            config.get("stage1_end", 50),
            config.get("stage2_end", 100),
        )

        avg_loss = train_one_epoch(
            model,
            config["train_graphs"],
            seen_mask,
            label_adj,
            criterion,
            optimizer,
            device,
            stage,
        )

        val_metrics = evaluate(model, config["val_graphs"], device)

        print(
            f"  Epoch {epoch:3d} [stage {stage}] | "
            f"loss={avg_loss:.4f} | "
            f"val_macro={val_metrics['macro_f1']:.4f} | "
            f"val_micro={val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 20):
                print(f"  Early stopping at epoch {epoch}")
                break

    print("  [DEBUG] loading best checkpoint")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_metrics = evaluate(model, config["test_graphs"], device)

    results = {
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_threshold": test_metrics["threshold"],
    }
    print(f"  [DEBUG] returning results: {results}")
    return results