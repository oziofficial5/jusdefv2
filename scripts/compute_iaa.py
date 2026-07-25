"""scripts/compute_iaa.py -- Cohen's kappa between the AI labels and the
independent annotator's labels (300-sentence blind pass), plus the primary
author's 100-sentence self-review. Reads the released files under
data/annotations/iaa/."""

import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent.parent / "data" / "annotations" / "iaa"
LABEL_ORDER = ["AFF", "NEG", "EXC", "OVR"]


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ann_path = BASE / "Annotation_300_returned.jsonl"
    ai_path = BASE / "Annotation_300_with_ai.jsonl"
    self_path = BASE / "self_review_100.jsonl"

    # Load the independent annotator's labels (blind 300-sentence pass).
    annotator = {}
    for d in load_jsonl(ann_path):
        label = d.get("label_name")
        if label:
            annotator[d["text"]] = label

    # Load AI labels
    ai = {}
    for d in load_jsonl(ai_path):
        label = d.get("ai_label", d.get("label_name"))
        if label:
            ai[d["text"]] = label

    # Align by identical text
    texts = sorted(annotator.keys() & ai.keys())
    ann_labels = [annotator[t] for t in texts]
    ai_labels = [ai[t] for t in texts]

    print(f"Matched pairs: {len(texts)}/300")

    if not texts:
        print("No matched pairs found. Check text formatting or file paths.")
        return

    print("\n=== Cohen's kappa (AI vs. independent annotator) ===")
    print(f"  kappa = {cohen_kappa_score(ann_labels, ai_labels):.4f}")

    print("\n=== Agreement rate ===")
    agree = sum(1 for s, a in zip(ann_labels, ai_labels) if s == a)
    print(f"  {agree}/{len(texts)} = {100 * agree / len(texts):.1f}%")

    print("\n=== Classification report (AI vs. annotator) ===")
    print(
        classification_report(
            ann_labels,
            ai_labels,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    print("\n=== Confusion matrix (rows: annotator, cols: AI) ===")
    cm = confusion_matrix(ann_labels, ai_labels, labels=LABEL_ORDER)
    print("        AFF  NEG  EXC  OVR")
    for i, lab in enumerate(LABEL_ORDER):
        print(f"  {lab}:  {cm[i,0]:4d} {cm[i,1]:4d} {cm[i,2]:4d} {cm[i,3]:4d}")

    # Self-review (primary author vs AI)
    print("\n=== Self-review (primary author vs. AI) ===")
    me_match = []
    ai_match = []

    for d in load_jsonl(self_path):
        my_label = d.get("my_label")
        ai_label = d.get("ai_label")
        if my_label and ai_label:
            me_match.append(my_label)
            ai_match.append(ai_label)

    print(f"  N = {len(me_match)}")
    if me_match:
        print(f"  Self kappa = {cohen_kappa_score(me_match, ai_match):.4f}")
        agree = sum(1 for s, a in zip(me_match, ai_match) if s == a)
        print(f"  Self agreement: {agree}/{len(me_match)} = {100 * agree / len(me_match):.1f}%")
    else:
        print("  No self-review labels filled in yet.")


if __name__ == "__main__":
    main()
