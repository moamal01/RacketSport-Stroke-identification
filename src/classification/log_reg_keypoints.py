import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import json
import ast
from utility_functions import plot_label_distribution, plot_confusion_matrix

file_path1 = "midpoints.csv"
file_path2 = "mirrored_midpoints_video1.csv"
file_path3 = "midpoints_video2.csv"
file_path4 = "mirrored_midpoints_video2.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)

with open(f"data/events/events_markup2.json", "r") as file:
    data3 = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data3.items() if v not in excluded_values}

keypoint_list = []
labels = []

def mirror_string(input_str):
    mirrored = input_str.replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')
    return mirrored

for frame, value in stroke_frames_1.items():
    if value == "other" or value == "otherotherother":
        continue
    
    player = value.split(" ")[0]
    label = value.replace(" ", "_")
    
    path = f"embeddings/video_1/{frame}/0/{player}.npy"
    if os.path.exists(path):
        event_row = df1.loc[df1['Event frame'] == int(frame)]
        keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
        keypoints = np.array(keypoints)[:, :2]
        keypoint_list.append(keypoints.flatten())
        labels.append(label)
        
for frame, value in stroke_frames_1.items():
    if value == "other" or value == "otherotherother":
        continue
    
    value = mirror_string(value)
    
    player = value.split(" ")[0]
    label = value.replace(" ", "_")
    
    path = f"embeddings/video_1/{frame}/0/{player}.npy"
    if os.path.exists(path):
        event_row = df2.loc[df2['Event frame'] == int(frame)]
        keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
        keypoints = np.array(keypoints)[:, :2]
        keypoint_list.append(keypoints.flatten())
        labels.append(label)

for frame, value in stroke_frames_2.items():
    if value in {"other", "otherotherother"}:
        continue

    label = value.split(" ")[0]
    player = label.split("_")[0]
    value3 = label.split("_")[2]

    path = f"embeddings/video_2/{frame}/0/{player}.npy"
    if os.path.exists(path):
        event_row = df3.loc[df3['Event frame'] == int(frame)]
        keypoint = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
        keypoint = np.array(keypoint)[:, :2]
        keypoint_list.append(keypoint.flatten())
        labels.append(label)

for frame, value in stroke_frames_2.items():
    if value in {"other", "otherotherother"}:
        continue
    
    value = mirror_string(value)

    label = value.split(" ")[0]
    player = label.split("_")[0]
    value3 = label.split("_")[2]

    path = f"embeddings/video_2/{frame}/0/{player}.npy"
    if os.path.exists(path):
        event_row = df4.loc[df4['Event frame'] == int(frame)]
        keypoint = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
        keypoint = np.array(keypoint)[:, :2]
        keypoint_list.append(keypoint.flatten())
        labels.append(label)

label_counts = Counter(labels)
min_label_threshold = 6
valid_labels = [label for label, count in label_counts.items() if count >= min_label_threshold]

# Filter embeddings and labels based on valid labels
filtered_keypoint_list = []
filtered_labels = []

for keypoints, label in zip(keypoint_list, labels):
    if label in valid_labels:
        filtered_keypoint_list.append(keypoints)
        filtered_labels.append(label)

# Encode labels as numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(filtered_labels)

# Stack embeddings into a 2D array (X)
X = np.vstack(filtered_keypoint_list)

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

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Confusion matrix
y_test = label_encoder.inverse_transform(y_test)
y_pred = label_encoder.inverse_transform(y_pred)
plot_confusion_matrix(y_test, y_pred, True)