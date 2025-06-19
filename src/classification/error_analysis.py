#!/usr/bin/env python3
"""
Usage:
    python stroke_error_analysis.py path/to/predictions.json
"""

import json
import sys
from collections import Counter, defaultdict

def load_records(path):
    with open(path, "r", encoding="utf‑8") as f:
        return json.load(f)

def build_confusion(records):
    """Returns nested dict: confusion[true][pred] = count"""
    confusion = defaultdict(Counter)
    for r in records:
        confusion[r["true_class"]][r["predicted_class"]] += 1
    return confusion

def print_confusion(confusion):
    classes = sorted(confusion.keys())
    # header
    hdr = ["true \\ pred"] + classes
    print("\nConfusion matrix (counts)")
    print("\t".join(hdr))
    for t in classes:
        row = [t] + [str(confusion[t][p]) for p in classes]
        print("\t".join(row))

def most_confused_with(confusion, top_n=3):
    result = {}
    for true_cls, preds in confusion.items():
        # Remove correct predictions
        wrong = [(p, c) for p, c in preds.items() if p != true_cls]
        # Sort by count descending
        wrong_sorted = sorted(wrong, key=lambda kv: kv[1], reverse=True)
        # Pad with ("‑", 0) if fewer than top_n items
        result[true_cls] = wrong_sorted[:top_n] + [("‑", 0)] * (top_n - len(wrong_sorted))
    return result

def topk_accuracies(records, k_vals=(1, 2, 3)):
    hits = Counter({k: 0 for k in k_vals})
    for r in records:
        true_cls = r["true_class"]
        # sort probabilities biggest→smallest
        ranked = sorted(
            r["probabilities"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        labels_in_order = [lbl for lbl, _ in ranked]
        for k in k_vals:
            if true_cls in labels_in_order[:k]:
                hits[k] += 1
    total = len(records)
    return {k: hits[k] * 100.0 / total for k in k_vals}

def main(path):
    records = load_records(path)
    confusion = build_confusion(records)

    # 1) confusion matrix
    print_confusion(confusion)

    # 2) most‑confused‑with list
    print("\nMost common confusions per true class")
    print(f"{'True Class':<15} ➜  {'1st':<18}  {'2nd':<18}  {'3rd':<18}")
    for t, confusions in most_confused_with(confusion).items():
        formatted = "  ".join([f"{p:<15}({cnt})" for p, cnt in confusions])
        print(f"{t:<15} ➜  {formatted}")

    # 3) top‑k accuracies
    accs = topk_accuracies(records)
    print("\nOverall accuracy")
    for k in sorted(accs):
        print(f"top‑{k}: {accs[k]:.2f}%")

main("../../results/replace/20250614_181201/04_raw_keypoints/prediction_log/04_raw_keypoints_iteration1.json")
