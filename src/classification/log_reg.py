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
    get_embeddings_and_labels,
    get_keypoints_and_labels,
    get_concat_and_labels
)

test_on_one = False
simplify = False
mirrored_only = False
add_mirrored = False
videos = [1, 2]
train_videos = videos[:-1]
test_videos = [videos[-1]]

# Get embeddings and labels
def get_embeddings(videos, simplify):
    embeddings = []
    labels = []
    
    for video in videos:
        video_embeddings, video_labels = get_embeddings_and_labels(video, mirror=mirrored_only, simplify=simplify)
        embeddings.extend(video_embeddings)
        labels.extend(video_labels)
        
        if add_mirrored and len(videos) > 1:
            video_embeddings, video_labels = get_embeddings_and_labels(video, mirror=True, simplify=simplify)
            embeddings.extend(video_embeddings)
            labels.extend(video_labels)
            
    return embeddings, labels

def get_keypoints(videos, simplify):
    keypoints = []
    labels = []
    
    for video in videos:
        video_keypoints, video_labels = get_keypoints_and_labels(video, mirror=mirrored_only, simplify=simplify)
        keypoints.extend(video_keypoints)
        labels.extend(video_labels)
    
        if add_mirrored and len(videos) > 1:
            video_keypoints, video_labels = get_keypoints_and_labels(video, mirror=True, simplify=simplify)
            keypoints.extend(video_keypoints)
            labels.extend(video_labels)
    
    return keypoints, labels

def get_concatenated(videos, simplify):
    concatenated = []
    labels = []
    
    for video in videos:
        video_concatenated, video_labels = get_concat_and_labels(video, mirror=mirrored_only, simplify=simplify)
        concatenated.extend(video_concatenated)
        labels.extend(video_labels)
        
        if add_mirrored and len(videos) > 1:
            video_concatenated, video_labels = get_concat_and_labels(video, mirror=True, simplify=simplify)
            concatenated.extend(video_concatenated)
            labels.extend(video_labels)
    
    return concatenated, labels

# Combine all data
def get_splits(type="embeddings"):
    if type == "embeddings":
        all_data, all_labels = get_embeddings(videos, simplify)
    elif type == "keypoints":
        all_data, all_labels = get_keypoints(videos, simplify)
    else:
        all_data, all_labels = get_concatenated(videos, simplify)

    # Encode all labels
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)
    all_embeddings_np = np.vstack(all_data)

    def can_stratify(labels):
        label_counts = Counter(labels)
        return all(count >= 2 for count in label_counts.values())

    if test_on_one:
        if type == "embeddings":
            train_embeddings, train_labels = get_embeddings(train_videos, simplify)
        elif type == "keypoints":
            train_embeddings, train_labels = get_keypoints(train_videos, simplify)
        else:
            train_embeddings, train_labels = get_concatenated(train_videos, simplify)

        # Filter test samples from video 3 that have seen labels
        train_label_set = set(train_labels)
        filtered_test_embeddings = []
        filtered_test_labels = []

        if type == "embeddings":
            video3_embeddings, video3_labels = get_embeddings(test_videos, simplify)
        elif type == "keypoints":
            video3_embeddings, video3_labels = get_keypoints(test_videos, simplify)
        else:
            video3_embeddings, video3_labels = get_concatenated(test_videos, simplify)

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
    # Print stats
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}, Test samples: {len(X_test)}")

    plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution")
    if y_val is not None:
        plot_label_distribution(label_encoder.inverse_transform(y_val), "Validation Set Label Distribution")
    plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution")

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
    plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, True)
    
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("embeddings")
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoints")
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("concat")
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
