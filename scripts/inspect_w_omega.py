import torch

v1 = torch.load("outputs/checkpoints/jusdef_full_s42.pt", map_location="cpu")
v2 = torch.load("outputs/checkpoints/jusdef_v2_full_hd512_s42.pt", map_location="cpu")

ops = ["AFF", "NEG", "EXC", "OVR"]

def get_w_omega(state_dict, layer_idx=0):
    Ws = []
    for op in range(4):
        key = f"dmp_layers.{layer_idx}.W_op.{op}.weight"
        if key in state_dict:
            Ws.append(state_dict[key])
        else:
            print(f"Missing key in state_dict: {key}")
    return Ws

v1_sd = v1 if isinstance(v1, dict) else v1
v2_sd = v2 if isinstance(v2, dict) else v2

v1_Ws = get_w_omega(v1_sd)
v2_Ws = get_w_omega(v2_sd)

print("v1 W_omega divergence (max ||W_i - W_j||):")
for i in range(len(v1_Ws)):
    for j in range(i + 1, len(v1_Ws)):
        d = (v1_Ws[i] - v1_Ws[j]).abs().max().item()
        print(f"  W_{ops[i]} - W_{ops[j]}: {d:.4f}")

print("\nv2 W_omega divergence:")
for i in range(len(v2_Ws)):
    for j in range(i + 1, len(v2_Ws)):
        d = (v2_Ws[i] - v2_Ws[j]).abs().max().item()
        print(f"  W_{ops[i]} - W_{ops[j]}: {d:.4f}")