"""scripts/run_self_review.py — interactive self-review of 100 AI labels."""

import json
from pathlib import Path
import sys

# Adjust this to your environment as needed
PATH = Path.home() / "OneDrive" / "Documents" / "papers" / "annotated" / "self_review_100.jsonl"
TMP = PATH.with_suffix(".jsonl.tmp")

def load_items(path: Path):
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_items(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def main():
    items = load_items(PATH)
    labels = ["AFF", "NEG", "EXC", "OVR"]

    # Resume support: skip items already labelled
    done = sum(1 for x in items if x.get("my_label"))
    total = len(items)
    print(f"Already done: {done}/{total}. Remaining: {total - done}")
    print("Type 0/1/2/3 for AFF/NEG/EXC/OVR, q to quit, ? to peek AI label\n")

    for i, item in enumerate(items):
        if item.get("my_label"):
            continue
        text = item.get("text", "")
        ai_label = item.get("ai_label")

        print(f"\n[{i+1}/{total}]")
        print(f"  {text[:400]}")

        while True:
            try:
                choice = input("  Your label > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Graceful exit on Ctrl+C / Ctrl+Z
                print("\nInterrupted. Saving progress...")
                save_items(TMP, items)
                TMP.replace(PATH)
                sys.exit(0)

            if choice == "q":
                save_items(TMP, items)
                TMP.replace(PATH)
                print("Saved. Re-run to resume.")
                sys.exit(0)

            if choice == "?":
                print(f"  (AI thought: {ai_label})")
                continue

            if choice in {"0", "1", "2", "3"}:
                my = labels[int(choice)]
                item["my_label"] = my
                item["agree"] = "yes" if my == ai_label else "no"
                break

            print("  Invalid. Try 0/1/2/3, ?, or q.")

    save_items(PATH, items)
    agreements = sum(1 for x in items if x.get("agree") == "yes")
    print(f"\nDone. {agreements}/{total} agreements.")

if __name__ == "__main__":
    main()