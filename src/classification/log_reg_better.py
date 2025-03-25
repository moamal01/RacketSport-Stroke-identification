import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from utility_functions import plot_label_distribution, plot_confusion_matrix
from src.data_utils.dataloader import EmbeddingDataset
from torch.utils.data import DataLoader

train_loader = DataLoader(EmbeddingDataset("train"), batch_size=32, shuffle=True)
val_loader = DataLoader(EmbeddingDataset("val"), batch_size=32, shuffle=False)
test_loader = DataLoader(EmbeddingDataset("test"), batch_size=32, shuffle=False)

def load_dataset(split="train"):
    file_path = f"data/splits/oversampled/{split}.npz"
    data = np.load(file_path)
    return data["X"], data["y"]

X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("val")
X_test, y_test = load_dataset("test")

plot_label_distribution(y_train, "Train Set Label Distribution")
plot_label_distribution(y_test, "Test Set Label Distribution")

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
plot_confusion_matrix(y_test, y_pred, True)