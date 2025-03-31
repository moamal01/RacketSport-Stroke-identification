import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.append(os.path.abspath('../../'))

from utility_functions import (
    plot_label_distribution,
    plot_confusion_matrix,
    load_json_with_dicts,
    get_embeddings_and_labels,
    get_embeddings_and_labels_special
)

data1 = load_json_with_dicts(f"data/events/events_markup1.json")
data2 = load_json_with_dicts(f"data/events/events_markup2.json")

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

embeddings = []
labels = []

video1_embeddings, video1_labels = get_embeddings_and_labels_special(stroke_frames_1)
embeddings.extend(video1_embeddings)
labels.extend(video1_labels)
        
video1m_embeddings, video1m_labels = get_embeddings_and_labels_special(stroke_frames_1, True)
embeddings.extend(video1m_embeddings)
labels.extend(video1m_labels)

video2_embeddings, video2_labels = get_embeddings_and_labels(2, stroke_frames_2)
embeddings.extend(video2_embeddings)
labels.extend(video2_labels)

video2m_embeddings, video2m_labels = get_embeddings_and_labels(2, stroke_frames_2, True)
embeddings.extend(video2m_embeddings)
labels.extend(video2m_labels)

label_counts = Counter(labels)
min_label_threshold = 6
valid_labels = [label for label, count in label_counts.items() if count >= min_label_threshold]

# Filter embeddings and labels based on valid labels
filtered_embeddings = []
filtered_labels = []

for embedding, label in zip(embeddings, labels):
    if label in valid_labels:
        filtered_embeddings.append(embedding)
        filtered_labels.append(label)

# Encode labels as numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(filtered_labels)

# Stack embeddings into a 2D array (X)
X = np.vstack(filtered_embeddings)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Length of training set {len(X_train)}")
print(f"Length of test set {len(X_test)}")
plot_label_distribution(filtered_labels, "Original Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='saga', multi_class='auto', penalty='l2')
clf.fit(X_train, y_train)

# Baseline accuracy
class_counts = Counter(y)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y)
print(f"Baseline Accuracy: {baseline_acc:.2f}")

# Evaluate
y_train_pred = clf.predict(X_train)  # Get predictions on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate training accuracy
print(f"Training Accuracy: {train_accuracy:.2f}")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

weights = np.abs(clf.coef_).sum(axis=0)
important_features = np.argsort(weights)[::-1]
print(f"Top 10 important dimensions: {important_features[:10]}")
print(f"Number of nonzero dimensions: {np.sum(weights > 0)} / 512")

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Confusion matrix
y_test = label_encoder.inverse_transform(y_test)
y_pred = label_encoder.inverse_transform(y_pred)
plot_confusion_matrix(y_test, y_pred, True)