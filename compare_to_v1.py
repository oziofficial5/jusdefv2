"""Quick comparison of v2 seed-42 vs v1 numbers."""
import json
import sys
import os


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


v1 = {
    "macro": 0.2519,
    "micro": 0.3879,
    "y_seen": 0.3085,
    "y_unseen": 0.0255,
    "y_exc": 0.3562,
    "threshold": 0.10,
}

p = "outputs/logs/jusdef_v2_full_hd512_s42.json"
v2 = load(p)

if v2 is None:
    print(f"{p} not found yet.")
    sys.exit()

print(f"{'Metric':<14} {'v1':>10} {'v2':>10} {'Delta':>10}")
print("-" * 46)

for k_v2, k_v1, label in [
    ("test_macro_f1", "macro", "Macro-F1"),
    ("test_micro_f1", "micro", "Micro-F1"),
    ("test_f1_seen", "y_seen", "F1(Y_s)"),
    ("test_f1_unseen", "y_unseen", "F1(Y_u)"),
    ("test_f1_exc", "y_exc", "F1(Y_exc)"),
    ("test_threshold", "threshold", "Threshold"),
]:
    a = v2.get(k_v2)
    b = v1[k_v1]
    if a is None:
        print(f"{label:<14} {b:>10.4f} {'N/A':>10}")
        continue
    d = a - b
    sign = "+" if d > 0 else ""
    print(f"{label:<14} {b:>10.4f} {a:>10.4f} {sign}{d:>9.4f}")
