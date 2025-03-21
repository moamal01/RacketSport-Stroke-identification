import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from collections import Counter

player = "right"
video_number = 2

with open(f"data/events/events_markup{video_number}.json", "r") as file:
    data = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames.items():
    if value == "other" or value == "otherotherother":
        continue

    value1 = value.split(" ")[0]
    value2 = value1.split("_")[2]
    if player in value1: # and labels.count(value) < 10:
        if os.path.exists(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"):
            embeddings.append(np.load(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"))  
            labels.append(value1.split('_', 1)[1])


X = np.vstack(embeddings)  # Shape: (num_samples, embedding_dim)

# Encode labels as numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)

# Baseline
class_counts = Counter(y)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y)

print(f"Baseline Accuracy: {baseline_acc:.2f}")

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
