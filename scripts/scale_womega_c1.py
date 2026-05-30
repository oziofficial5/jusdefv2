import torch
from pathlib import Path

CKPT_IN = Path("outputs/checkpoints/jusdef_v2_full_hd512_s42.pt")
CKPT_OUT = Path("outputs/checkpoints/jusdef_v2_full_hd512_s42_scaled.pt")

def main():
    state = torch.load(CKPT_IN, map_location="cpu")
    scaled = {}
    for k, v in state.items():
        if k.startswith("dmp_layers.") and ".W_op." in k and k.endswith(".weight"):
            # 6x scaling to roughly match v1 norms
            scaled[k] = v * 6.0
        else:
            scaled[k] = v

    torch.save(scaled, CKPT_OUT)
    print(f"Saved scaled checkpoint to {CKPT_OUT}")

if __name__ == "__main__":
    main()
