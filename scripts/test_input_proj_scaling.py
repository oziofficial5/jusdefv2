import torch

src = "outputs/checkpoints/jusdef_v2_full_hd512_s42.pt"
dst = "outputs/checkpoints/jusdef_v2_full_hd512_s42_inputscaled.pt"

state = torch.load(src, map_location="cpu")

# If checkpoint is wrapped in a dict with "model", unwrap it
if isinstance(state, dict) and "model" in state:
    sd = state["model"]
else:
    sd = state

v2_scaled = {k: v.clone() for k, v in sd.items()}

# These ratios come from the v1 vs v2 norm comparison
# v1: input_proj.conc.weight ||W|| ≈ 13.057; v2: ≈ 0.021 → ratio ≈ 622
# v1: input_proj.sec.weight  ||W|| ≈ 11.569; v2: ≈ 0.281 → ratio ≈ 41
# v1: out_proj.weight        ||W|| ≈ 23.817; v2: ≈ 9.711 → ratio ≈ 2.45

scaling = {
    "input_proj.conc.weight": 622.0,
    "input_proj.sec.weight": 41.0,
    "out_proj.weight": 2.45,
}

for key, factor in scaling.items():
    if key in v2_scaled:
        v2_scaled[key] = v2_scaled[key] * factor
        print(f"Scaled {key} by {factor}: new norm = {v2_scaled[key].norm().item():.3f}")
    else:
        print(f"[WARN] {key} not found in checkpoint")

# Re-wrap if original state had a "model" key
if isinstance(state, dict) and "model" in state:
    new_state = dict(state)
    new_state["model"] = v2_scaled
else:
    new_state = v2_scaled

torch.save(new_state, dst)
print(f"Saved scaled checkpoint to {dst}")
