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
import seaborn as sns

def plot_label_distribution(y_data, title):
    plt.figure(figsize=(12, 7))
    sns.countplot(y=y_data, order=np.unique(y_data))
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Label")
    plt.show()

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)
    
with open(f"data/events/events_markup2.json", "r") as file:
    data2 = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames_1.items():
    if value == "other" or value == "otherotherother":
        continue
    
    player = value.split(" ")[0]
    label = value.replace(" ", "_")
    
    file_path = f"imbeddings/video_1/{frame}/0/{player}.npy"
    if os.path.exists(file_path):
        embeddings.append(np.load(f"imbeddings/video_1/{frame}/0/{player}.npy"))  
        labels.append(label)

for frame, value in stroke_frames_2.items():
    if value in {"other", "otherotherother"}:
        continue

    label = value.split(" ")[0]
    value2 = label.split("_")[0]
    value3 = label.split("_")[2]

    file_path = f"imbeddings/video_2/{frame}/0/{value2}.npy"
    if os.path.exists(file_path):
        embeddings.append(np.load(file_path))
        labels.append(label)  # Extract the class label


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

plot_label_distribution(filtered_labels, "Original Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution")
plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
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
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(label_encoder.inverse_transform(y_test)))
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="Blues", ax=ax)

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()