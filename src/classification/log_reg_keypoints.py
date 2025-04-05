import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
import pandas as pd
import json
from utility_functions import (
    plot_label_distribution,
    plot_confusion_matrix,
    get_keypoints_and_labels,
    plot_coefficients
)

simplify = True
mirror = False

# Load keypoint data
file_paths = [
    ("midpoints_video1.csv", "mirrored_midpoints_video1.csv"),
    ("midpoints_video2.csv", "mirrored_midpoints_video2.csv"),
    ("midpoints_video3.csv", "mirrored_midpoints_video3.csv")
]

df_videos = [(pd.read_csv(fp1), pd.read_csv(fp2)) for fp1, fp2 in file_paths]

# Load event data
json_paths = [
    "data/events/events_markup1.json",
    "data/events/events_markup2.json",
    "data/events/events_markup3.json"
]

data_videos = [json.load(open(jp, "r")) for jp in json_paths]

# Filter timestamps
excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = [{k: v for k, v in data.items() if v not in excluded_values} for data in data_videos]

# Prepare keypoints and labels
keypoints_train, label_train = [], []

# Training (Video 1 & 2)
for i in range(2):
    keypoints, label = get_keypoints_and_labels(stroke_frames[i], df_videos[i][0],simplify=simplify)
    keypoints_train.extend(keypoints)
    label_train.extend(label)
    
    if mirror:
        keypoints_m, label_m = get_keypoints_and_labels(stroke_frames[i], df_videos[i][1], simplify=simplify)
        keypoints_train.extend(keypoints_m)
        label_train.extend(label_m)

# Testing (Video 3)
keypoints_test, label_test = get_keypoints_and_labels(stroke_frames[2], df_videos[2][0], simplify=simplify)

# Filter labels with sufficient samples
label_counts = Counter(label_train)
min_label_threshold = 6
valid_labels = {label for label, count in label_counts.items() if count >= min_label_threshold}

# Apply filtering
filtered_keypoint_list = [kp for kp, lbl in zip(keypoints_train, label_train) if lbl in valid_labels]
filtered_labels = [lbl for lbl in label_train if lbl in valid_labels]
filtered_keypoints_test = [kp for kp, lbl in zip(keypoints_test, label_test) if lbl in valid_labels]
filtered_labels_test = [lbl for lbl in label_test if lbl in valid_labels]

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(filtered_labels)
y_test = label_encoder.transform(filtered_labels_test)

# Stack keypoints into arrays
X_train = np.vstack(filtered_keypoint_list)
X_test = np.vstack(filtered_keypoints_test)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
plot_label_distribution(filtered_labels, "Training Set Label Distribution")
plot_label_distribution(filtered_labels_test, "Test Set Label Distribution")

# Baseline accuracy
class_counts = Counter(y_train)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y_train)
print(f"Baseline Accuracy: {baseline_acc:.2f}")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2')
clf.fit(X_train, y_train)

# Evaluate logistic regression
y_pred_train = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Logistic Regression training Accuracy: {train_accuracy:.2f}")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test Accuracy: {accuracy:.2f}")

# Train Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=69)
clf_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))
print(f"Random Forest Test Accuracy: {rf_acc:.2f}")

# XGBoost Classifier
# clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
# clf_xgb.fit(X_train, y_train)
# xgb_acc = accuracy_score(y_test, clf_xgb.predict(X_test))
# print(f"XGBoost Accuracy: {xgb_acc:.2f}")

# Confusion matrix
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
plot_confusion_matrix(y_test_labels, y_pred_labels, True)

# Coefficients heatmap
plot_coefficients(clf.coef_, label_encoder.classes_)
