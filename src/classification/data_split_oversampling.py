import numpy as np
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from utility_functions import load_json_with_dicts

data1 = load_json_with_dicts(f"data/events/events_markup1.json")
data2 = load_json_with_dicts(f"data/events/events_markup2.json")

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
    
    file_path = f"embeddings/video_1/{frame}/0/{player}.npy"
    if os.path.exists(file_path):
        embeddings.append(np.load(file_path).squeeze())  
        labels.append(label)

for frame, value in stroke_frames_2.items():
    if value in {"other", "otherotherother"}:
        continue

    label = value.split(" ")[0]
    value2 = label.split("_")[0]
    value3 = label.split("_")[2]

    file_path = f"embeddings/video_2/{frame}/0/{value2}.npy"
    if os.path.exists(file_path):
        embeddings.append(np.load(file_path).squeeze())
        labels.append(label)  # Extract the class label
            
# Ensure there are classes have sufficient populations
label_counts = Counter(labels)
min_label_threshold = 2
valid_labels = [label for label, count in label_counts.items() if count >= min_label_threshold]

# Filter embeddings and labels based on valid labels
filtered_embeddings = []
filtered_labels = []

for embedding, label in zip(embeddings, labels):
    if label in valid_labels:
        filtered_embeddings.append(embedding)
        filtered_labels.append(label)

# Convert to NumPy arrays
embeddings = np.array(filtered_embeddings)
labels = np.array(filtered_labels)

# Stratified Split: Train (80%), Validation (10%), Test (10%)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(sss.split(embeddings, labels))

X_train, y_train = embeddings[train_idx], labels[train_idx]
X_temp, y_temp = embeddings[temp_idx], labels[temp_idx]

# Oversample small classes
label_counts = Counter(y_train)
X_train_balanced = []
y_train_balanced = []

max_count = max(label_counts.values())
average_count = int(sum(label_counts.values()) / len(label_counts))

for label in label_counts:
    samples_needed = average_count - label_counts[label]
    indices = np.where(y_train == label)[0]
    
    X_train_balanced.extend(X_train[indices])
    y_train_balanced.extend(y_train[indices])
    
    if samples_needed < 5:
        continue
    
    num_samples_to_duplicate = len(indices)
    extra_samples = random.choices(indices.tolist(), k=num_samples_to_duplicate)
    
    while len(extra_samples) < samples_needed:
        extra_samples.extend(random.choices(indices.tolist(), k=num_samples_to_duplicate))
    
    X_train_balanced.extend(X_train[extra_samples])
    y_train_balanced.extend(y_train[extra_samples])

# Convert back to numpy arrays
X_train_balanced = np.array(X_train_balanced)
y_train_balanced = np.array(y_train_balanced)

# Filter y_temp to ensure no class has fewer than 2 samples
temp_label_counts = Counter(y_temp)
valid_temp_labels = [label for label, count in temp_label_counts.items() if count >= 2]
filtered_X_temp = []
filtered_y_temp = []

for x, y in zip(X_temp, y_temp):
    if y in valid_temp_labels:
        filtered_X_temp.append(x)
        filtered_y_temp.append(y)

X_temp = np.array(filtered_X_temp)
y_temp = np.array(filtered_y_temp)

# Further stratified split for validation and test (each 10%)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss.split(X_temp, y_temp))

X_val, y_val = X_temp[val_idx], y_temp[val_idx]
X_test, y_test = X_temp[test_idx], y_temp[test_idx]

# Save the datasets
output_dir = "data/splits/oversampled"
os.makedirs(output_dir, exist_ok=True)

np.savez_compressed(f"{output_dir}/train.npz", X=X_train_balanced, y=y_train_balanced)
np.savez_compressed(f"{output_dir}/val.npz", X=X_val, y=y_val)
np.savez_compressed(f"{output_dir}/test.npz", X=X_test, y=y_test)

print("Stratified Data successfully split and saved!")
