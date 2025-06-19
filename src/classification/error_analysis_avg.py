#!/usr/bin/env python3
"""
Usage:
    python stroke_error_analysis.py fold1.json fold2.json fold3.json
"""

import json
import sys
from collections import Counter, defaultdict

def load_records(path):
    with open(path, "r", encoding="utf‑8") as f:
        return json.load(f)

def build_confusion(records, confusion):
    """Updates a shared confusion matrix dict"""
    for r in records:
        confusion[r["true_class"]][r["predicted_class"]] += 1

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
        result[true_cls] = wrong_sorted[:top_n] + [("‑", 0)] * (top_n - len(wrong_sorted))
    return result

def update_topk_accuracies(records, k_vals, hit_counter):
    for r in records:
        true_cls = r["true_class"]
        ranked = sorted(r["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        labels_in_order = [lbl for lbl, _ in ranked]
        for k in k_vals:
            if true_cls in labels_in_order[:k]:
                hit_counter[k] += 1
    return len(records)

def main(paths):
    confusion = defaultdict(Counter)
    k_vals = (1, 2, 3)
    hit_counter = Counter({k: 0 for k in k_vals})
    total_records = 0

    for path in paths:
        records = load_records(path)
        build_confusion(records, confusion)
        total_records += update_topk_accuracies(records, k_vals, hit_counter)

    # 1) confusion matrix
    print_confusion(confusion)

    # 2) most‑confused‑with list
    print("\nMost common confusions per true class")
    print(f"{'True Class':<15} ➜  {'1st':<18}  {'2nd':<18}  {'3rd':<18}")
    for t, confusions in most_confused_with(confusion).items():
        formatted = "  ".join([f"{p:<15}({cnt})" for p, cnt in confusions])
        print(f"{t:<15} ➜  {formatted}")

    # 3) top‑k accuracies
    print("\nAverage overall accuracy across folds")
    for k in sorted(hit_counter):
        acc = hit_counter[k] * 100.0 / total_records
        print(f"top‑{k}: {acc:.2f}%")

paths = [
    "../../results/replace/20250615_002222/35_keypoints_midpoints_table_time/prediction_log/35_keypoints_midpoints_table_time_iteration1.json",
    "../../results/replace/20250615_002222/35_keypoints_midpoints_table_time/prediction_log/35_keypoints_midpoints_table_time_iteration2.json",
    "../../results/replace/20250615_002222/35_keypoints_midpoints_table_time/prediction_log/35_keypoints_midpoints_table_time_iteration3.json"
]

main(paths)
