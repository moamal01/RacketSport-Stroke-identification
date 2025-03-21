from collections import Counter

class_counts = Counter(y)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y)

print(f"Baseline Accuracy: {baseline_acc:.2f}")
