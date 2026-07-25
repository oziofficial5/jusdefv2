"""
Dump per-paragraph predictions on the 10-20% non-AFF density bin for
v3_pilot, v4_twostage, and the mean baseline. Used to build the
confusion matrix figure (fig18) and any class-level analyses.

Outputs:
    outputs/logs/predictions_10_20_bin.json
"""
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.jusdef_ledgar import JusDefLEDGAR


SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

VARIANTS = {
    "mean":         ({"dmp_variant": "mean"},          "baseline_mean"),
    "v3_pilot":     ({"dmp_variant": "v3"},            "v3_pilot"),
    "v4_twostage":  ({"dmp_variant": "v4_hard",
                      "v4_density_lo": 0.10,
                      "v4_density_hi": 0.20},          "v4_twostage"),
}


def build_model(tag, device):
    kwargs, _ = VARIANTS[tag]
    full = dict(in_dim=768, hidden_dim=512, num_classes=100, num_layers=1)
    full.update(kwargs)
    return JusDefLEDGAR(**full).to(device)


@torch.no_grad()
def predict_paragraph(model, p, device):
    sent_embs = p["embeddings"].to(device)
    sent_to_para = torch.zeros(sent_embs.size(0), dtype=torch.long, device=device)
    ops = torch.tensor(p["operators"], dtype=torch.long, device=device)
    logits = model(sent_embs, sent_to_para, ops, num_paragraphs=1)
    return int(logits.argmax(dim=-1).item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open("data/processed_ledgar/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)

    density = np.array([
        sum(1 for o in d["operators"] if o != 0) / max(len(d["operators"]), 1)
        for d in test_data
    ])
    bin_idx = np.where((density >= 0.10) & (density < 0.20))[0]
    print(f"  10-20% bin: {len(bin_idx)} paragraphs")

    true_labels = [int(test_data[i]["label"]) for i in bin_idx]

    ckpt_dir = Path("outputs/checkpoints")
    predictions = {"true_labels": true_labels, "n_paragraphs": len(bin_idx)}

    for tag, (_, file_prefix) in VARIANTS.items():
        print(f"\n=== {tag} ===")
        predictions[tag] = {}
        for seed in SEEDS:
            ckpt = ckpt_dir / f"ledgar_{file_prefix}_s{seed}.pt"
            if not ckpt.is_file():
                print(f"  s{seed}: missing")
                continue
            model = build_model(tag, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            preds = [predict_paragraph(model, test_data[i], device) for i in bin_idx]
            n_correct = sum(1 for p, y in zip(preds, true_labels) if p == y)
            predictions[tag][seed] = preds
            print(f"  s{seed}: {n_correct}/{len(bin_idx)} correct")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    out_path = Path("outputs/logs/predictions_10_20_bin.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
