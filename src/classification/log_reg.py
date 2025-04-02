import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os
import sys

sys.path.append(os.path.abspath('../../'))

from utility_functions import (
    plot_label_distribution,
    plot_confusion_matrix,
    load_json_with_dicts,
    get_embeddings_and_labels
)

# Load data for each video
data1 = load_json_with_dicts(f"data/events/events_markup1.json")
data2 = load_json_with_dicts(f"data/events/events_markup2.json")
data3 = load_json_with_dicts(f"data/events/events_markup3.json")

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}
stroke_frames_3 = {k: v for k, v in data3.items() if v not in excluded_values}

# Get embeddings and labels
video1_embeddings, video1_labels = get_embeddings_and_labels(1, stroke_frames_1)
video2_embeddings, video2_labels = get_embeddings_and_labels(2, stroke_frames_2)
video3_embeddings, video3_labels = get_embeddings_and_labels(3, stroke_frames_3)

# Combine video 1 & 2 for training
train_embeddings = video1_embeddings + video2_embeddings
train_labels = video1_labels + video2_labels

# Ensure test labels exist in training set
train_label_counts = Counter(train_labels)
valid_labels = set(train_label_counts.keys())
filtered_test_embeddings = []
filtered_test_labels = []

for emb, label in zip(video3_embeddings, video3_labels):
    if label in valid_labels:
        filtered_test_embeddings.append(emb)
        filtered_test_labels.append(label)

# Encode labels
label_encoder = LabelEncoder()
y_all_train = label_encoder.fit_transform(train_labels)
y_test = label_encoder.transform(filtered_test_labels)  # Use same encoder

# Convert embeddings into arrays
X_all_train = np.vstack(train_embeddings)
X_test = np.vstack(filtered_test_embeddings)

# Split Video 1 & 2 into train (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_all_train, y_all_train, test_size=0.2, random_state=42)

print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_val), "Validation Set Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='saga', multi_class='auto', penalty='l2')
clf.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Final Evaluation on Test Set (Video 3)
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy on Video 3: {test_accuracy:.2f}")

# Train Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
rf_val_acc = accuracy_score(y_val, clf_rf.predict(X_val))
rf_test_acc = accuracy_score(y_test, clf_rf.predict(X_test))

print(f"Random Forest Validation Accuracy: {rf_val_acc:.2f}")
print(f"Random Forest Test Accuracy on Video 3: {rf_test_acc:.2f}")

# Confusion matrix for test set
y_test_decoded = label_encoder.inverse_transform(y_test)
y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, True)
