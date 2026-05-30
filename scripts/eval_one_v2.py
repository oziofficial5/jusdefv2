"""One-off eval of v2 checkpoint to get all 5 metrics."""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, ".")

from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph
from src.model.baselines import tune_threshold


CKPT = "outputs/checkpoints/jusdef_v2_full_hd512_s1only_s42.pt"
TEST = "data/processed/graphs/test_graphs.pt"

print(f"Loading checkpoint: {CKPT}")
model = JusDef(
    in_dim=768,
    hidden_dim=512,
    num_layers=2,
    use_dmp=True,
    use_authority=True,
)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Device: {device}")

print(f"Loading test graphs from {TEST}")
test_graphs = torch.load(TEST, map_location="cpu")

print(f"Running inference on {len(test_graphs)} graphs...")
all_logits, all_targets = [], []
with torch.no_grad():
    for i, g in enumerate(test_graphs):
        scores, _, _ = forward_one_graph(model, g, device)
        all_logits.append(scores.cpu().squeeze(0))
        all_targets.append(g.y.cpu())
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test_graphs)}")

logits = torch.stack(all_logits).numpy()
targets = torch.stack(all_targets).numpy()
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

print("\nTuning threshold...")
best_t, _ = tune_threshold(probs, targets)
preds = (probs >= best_t).astype(int)

# Load Y_exc labels
exc = json.load(open("data/annotations/exception_labels.json"))[
    "exception_override_labels"
]
seen = list(range(80))
unseen = list(range(80, 100))

print("\n=== v2 full hd512 seed 42 — TEST METRICS ===")
print(
    f'Macro-F1:    {f1_score(targets, preds, average="macro", zero_division=0):.4f}'
)
print(
    f'Micro-F1:    {f1_score(targets, preds, average="micro", zero_division=0):.4f}'
)
print(
    f'F1(Y_s):     {f1_score(targets[:, seen], preds[:, seen], average="macro", zero_division=0):.4f}'
)
print(
    f'F1(Y_u):     {f1_score(targets[:, unseen], preds[:, unseen], average="macro", zero_division=0):.4f}'
)
print(
    f'F1(Y_exc):   {f1_score(targets[:, exc], preds[:, exc], average="macro", zero_division=0):.4f}'
)
print(f"Threshold:   {best_t:.4f}")

# Also save to JSON in the standard format
out = {
    "test_macro_f1": float(
        f1_score(targets, preds, average="macro", zero_division=0)
    ),
    "test_micro_f1": float(
        f1_score(targets, preds, average="micro", zero_division=0)
    ),
    "test_f1_seen": float(
        f1_score(targets[:, seen], preds[:, seen], average="macro", zero_division=0)
    ),
    "test_f1_unseen": float(
        f1_score(
            targets[:, unseen], preds[:, unseen], average="macro", zero_division=0
        )
    ),
    "test_f1_exc": float(
        f1_score(targets[:, exc], preds[:, exc], average="macro", zero_division=0)
    ),
    "test_threshold": float(best_t),
}

with open("outputs/logs/jusdef_v2_full_hd512_s42.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nSaved to outputs/logs/jusdef_v2_full_hd512_s42.json")

