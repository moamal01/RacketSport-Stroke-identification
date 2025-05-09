import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd


sys.path.append(os.path.abspath('../../'))

from utility_functions import (plot_label_distribution, plot_confusion_matrix, get_features, plot_probabilities)

per_player_classifiers = False
test_on_one = True
simplify = True
mirrored_only = False
add_mirrored = False
videos = [1, 2, 3]
train_videos = videos[:-1]
test_videos = [videos[-1]]

# Generic processing function
def process_videos(videos, sequence, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify, long_edition=False):
    results = []
    labels = []

    for video in videos:
        data, video_labels = get_features(video_number=video, sequence_frames=sequence, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_table=add_table, add_embeddings=add_embeddings, mirror=mirrored_only, simplify=simplify, long_edition=long_edition)
        results.extend(data)
        labels.extend(video_labels)

        if add_mirrored and len(videos) > 1:
            data, video_labels = get_features(video, sequence, raw, add_keypoints, add_midpoints, add_table, add_embeddings, mirror=True, simplify=simplify, long_edition=long_edition)
            results.extend(data)
            labels.extend(video_labels)

    return results, labels


def get_specified_features(videos, sequence_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify=simplify, long_edition=False):
    return process_videos(videos, sequence_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify=simplify, long_edition=long_edition)

# Combine all data
def get_splits(long_sequence=False, raw=False, add_keypoints=True, add_midpoints=False, add_table=False, add_embeddings=False, process_both_players=False):
    if long_sequence:
        sequence_frames = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    else:
        sequence_frames = [0]
    
    all_data, all_labels = get_specified_features(videos, sequence_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify, process_both_players)

    # Encode all labels
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)
    all_embeddings_np = np.vstack(all_data)

    def can_stratify(labels):
        label_counts = Counter(labels)
        return all(count >= 2 for count in label_counts.values())

    if test_on_one:
        train_embeddings, train_labels = get_specified_features(train_videos, sequence_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify, process_both_players)

        # Filter test samples from video 3 that have seen labels
        train_label_set = set(train_labels)
        filtered_test_embeddings = []
        filtered_test_labels = []

        video3_embeddings, video3_labels = get_specified_features(test_videos, sequence_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify, process_both_players)

        for emb, label in zip(video3_embeddings, video3_labels):
            if label in train_label_set:
                filtered_test_embeddings.append(emb)
                filtered_test_labels.append(label)

        X_train = np.vstack(train_embeddings)
        y_train = label_encoder.transform(train_labels)

        X_val, y_val = None, None  # No validation set in this case
        X_test = np.vstack(filtered_test_embeddings)
        y_test = label_encoder.transform(filtered_test_labels)

    else:
        # Random 80/10/10 split
        if can_stratify(all_labels_encoded):
            strat = all_labels_encoded
        else:
            strat = None
            print("⚠️ Not all classes have ≥2 samples. Falling back to non-stratified split.")

        # First: 90% temp, 10% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_embeddings_np, all_labels_encoded, test_size=0.1, random_state=42, stratify=strat
        )

        if can_stratify(y_temp):
            strat_temp = y_temp
        else:
            strat_temp = None
            print("⚠️ Not all classes in temp split have ≥2 samples. Validation will be non-stratified.")

        # Then: 10% of 90% (i.e. 10% overall) for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=1/9, random_state=42, stratify=strat_temp
        )

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder

def classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    probabilities = []
    # Print stats
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}, Test samples: {len(X_test)}")

    #plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution", simplify=simplify)
    #if y_val is not None:
    #    plot_label_distribution(label_encoder.inverse_transform(y_val), "Validation Set Label Distribution", simplify=simplify)
    #plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution", simplify=simplify)

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, solver='saga', penalty='l2')
    clf.fit(X_train, y_train)
    
    class_counts = Counter(y_train)
    most_common_class = max(class_counts, key=class_counts.get)
    baseline_acc = class_counts[most_common_class] / len(y_train)
    print(f"Baseline Accuracy: {baseline_acc:.2f}")

    # Validation accuracy
    if X_val is not None:
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Test accuracy
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    # --- Softmax outputs ---
    y_test_probs = clf.predict_proba(X_test)  # shape: (num_samples, num_classes)
    class_names = label_encoder.classes_

    for i in range(len(X_test)):
        for j in range(len(y_test_probs[i])):
            # probs, true_label
            probabilities.append({
                "predicted_class": class_names[j],
                "probability": y_test_probs[i][j],
                "probabilities": y_test_probs[i],
                "true_class": class_names[y_test[i]]
            })

    # Train Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)

    if X_val is not None:
        rf_val_acc = accuracy_score(y_val, clf_rf.predict(X_val))
        print(f"Random Forest Validation Accuracy: {rf_val_acc:.2f}")

    rf_test_acc = accuracy_score(y_test, clf_rf.predict(X_test))
    print(f"Random Forest Test Accuracy: {rf_test_acc:.2f}")

    # Confusion Matrix
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
    #plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, True)
    
    return probabilities

print("Raw keypoints")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(raw=True, add_keypoints=True, process_both_players=True)
probs = classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
plot_probabilities(probs, len(X_test), label_encoder)
print("-----------")

# print("Raw keypoints over time")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, raw=True, add_keypoints=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

# print("Embeddings")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(add_embeddings=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

# print("Embeddings over time")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_embeddings=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

# print("Normalized keypoints")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(add_keypoints=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

# print("Normalized keypoints over time")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

# print("Normalized keypoints and player midpoints over time")
# X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True, process_both_players=True)
# classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
# print("-----------")

print("Normalized keypoints, player midpoints and table position over time")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True, add_table=True, process_both_players=True)
probs = classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
plot_probabilities(probs, len(X_test), label_encoder)
print("-----------")

print("Normalized keypoints, player midpoints, table position and embeddings over time")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True, add_table=True, add_embeddings=True, process_both_players=True)
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")


if per_player_classifiers:
    print("Classification on embeddings")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(add_embeddings=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints raw")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(raw=True, add_keypoints=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on embeddings and raw keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(raw=True, add_keypoints=True, add_embeddings=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(add_keypoints=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on embeddings and keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(add_keypoints=True, add_embeddings=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time with midpoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time with midpoints along with table midpoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True, add_table=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on all features")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits(long_sequence=True, add_keypoints=True, add_midpoints=True, add_table=True, add_embeddings=True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")