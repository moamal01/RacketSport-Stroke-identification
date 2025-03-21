import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Player and video settings
player = "right"
video_number = 2

# Load the data
with open(f"data/events/events_markup{video_number}.json", "r") as file:
    data = json.load(file)

# Exclude specific event types
excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

# Initialize lists to hold embeddings and labels
embeddings = []
labels = []

# Process the frames to extract embeddings and corresponding labels
for frame, value in stroke_frames.items():
    if value == "other" or value == "otherotherother":
        continue

    value1 = value.split(" ")[0]
    value2 = value1.split("_")[2]
    if player in value1:
        if os.path.exists(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"):
            embeddings.append(np.load(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"))  
            labels.append(value1.split('_', 1)[1])


label_counts = Counter(labels)
min_label_threshold = 5
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

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)

# Baseline accuracy
class_counts = Counter(y)
most_common_class = max(class_counts, key=class_counts.get)
baseline_acc = class_counts[most_common_class] / len(y)

print(f"Baseline Accuracy: {baseline_acc:.2f}")

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))

print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()