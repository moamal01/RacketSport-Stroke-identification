import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(split="train"):
    file_path = f"data/splits/oversampled/{split}.npz"
    data = np.load(file_path)
    return data["X"], data["y"]

def plot_label_distribution(y_data, title):
    plt.figure(figsize=(10, 5))
    sns.countplot(y=y_data, order=np.unique(y_data))
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Label")
    plt.show()

# Example usage
X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("val")
X_test, y_test = load_dataset("test")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)

# Baseline accuracy
class_counts = Counter(y_train)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y_train)

print(f"Baseline Accuracy: {baseline_acc:.2f}")

# Evaluate
y_train_pred = clf.predict(X_train)  # Get predictions on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate training accuracy
print(f"Training Accuracy: {train_accuracy:.2f}")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))

print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Confusion matrix
labels = np.unique(np.concatenate([y_test, y_pred]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()