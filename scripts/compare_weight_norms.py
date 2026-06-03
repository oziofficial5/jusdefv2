import torch

v1 = torch.load("/home/francesco/Jusdef/outputs/checkpoints/jusdef_full_s42.pt", map_location="cpu")
v2 = torch.load("/home/francesco/jusdefv2/jusdefv2/outputs/checkpoints/jusdef_v2_full_hd512_s42.pt", map_location="cpu")

# Filter to common keys
common = set(v1.keys()) & set(v2.keys())
v1_only = set(v1.keys()) - set(v2.keys())
v2_only = set(v2.keys()) - set(v1.keys())

if v1_only:
    print(f"In v1 only ({len(v1_only)}):")
    for k in sorted(v1_only)[:20]:
        shape = v1[k].shape if torch.is_tensor(v1[k]) else type(v1[k]).__name__
        print(f"  {k}: {shape}")

if v2_only:
    print(f"\nIn v2 only ({len(v2_only)}):")
    for k in sorted(v2_only)[:20]:
        shape = v2[k].shape if torch.is_tensor(v2[k]) else type(v2[k]).__name__
        print(f"  {k}: {shape}")

print("\n=== Common module weights, sorted by |Δ ||W|| | ===")
rows = []
for k in common:
    if not (torch.is_tensor(v1[k]) and torch.is_tensor(v2[k])):
        continue
    if v1[k].shape != v2[k].shape:
        continue
    n1 = v1[k].norm().item()
    n2 = v2[k].norm().item()
    delta = abs(n1 - n2)
    rows.append((delta, k, n1, n2))

rows.sort(reverse=True)
print(f"{'Module':<60} {'v1 ||W||':>10} {'v2 ||W||':>10} {'Δ':>10}")
print("-" * 95)
for delta, k, n1, n2 in rows[:30]:
    short = k if len(k) <= 58 else "..." + k[-55:]
    print(f"{short:<60} {n1:>10.3f} {n2:>10.3f} {delta:>10.3f}")
