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

from utility_functions import (plot_label_distribution, plot_confusion_matrix, get_keypoints_and_labels)

per_player_classifiers = False
test_on_one = True
simplify = True
mirrored_only = False
add_mirrored = False
videos = [1, 2, 3]
train_videos = videos[:-1]
test_videos = [videos[-1]]

# Generic processing function

def process_videos2(videos, sequence, raw, add_keypoints, add_midpoints, add_table, add_embeddings, simplify, long_edition=False):
    results = []
    labels = []

    for video in videos:
        data, video_labels = get_keypoints_and_labels(video, sequence, raw, add_keypoints, add_midpoints, add_table, add_embeddings, mirror=mirrored_only, simplify=simplify, long_edition=long_edition)
        results.extend(data)
        labels.extend(video_labels)

        if add_mirrored and len(videos) > 1:
            data, video_labels = get_keypoints_and_labels(video, sequence, raw, add_keypoints, add_midpoints, add_table, add_embeddings, mirror=True, simplify=simplify, long_edition=long_edition)
            results.extend(data)
            labels.extend(video_labels)

    return results, labels

# Specific functions using the generic one
def get_embeddings(videos, simplify):
    return process_videos2(videos, [0], False, False, False, False, True, simplify)

def get_keypoints_raw(videos, simplify):
    return process_videos2(videos, [0], True, True, False, False, False, simplify)

def get_concatenated_raw(videos, simplify):
    return process_videos2(videos, [0], False, True, False, False, True, simplify)

def get_keypoints(videos, simplify):
    return process_videos2(videos, [0], False, True, False, False, False, simplify)

def get_concatenated(videos, simplify):
    return process_videos2(videos, [0], False, True, False, False, True, simplify)

def get_keypoints_time(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, False, False, False, simplify)

def get_keypoints_time_and_mid(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, False, False, simplify)

def get_keypoints_time_and_mid_and_tab(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, True, False, simplify)

def get_every_feature(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, True, True, simplify)

def get_every_feature(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, True, True, simplify)

def jaja(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, False, False, False, simplify, True)

def jaja2(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, False, False, simplify, True)

def jaja3(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, True, False, simplify, True)

def jaja4(videos, simplify):
    return process_videos2(videos, [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], False, True, True, True, True, simplify, True)

def jaja5(videos, sequences_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, mirror=True, simplify=simplify, long_edition=False):
    return process_videos2(videos, sequences_frames, raw, add_keypoints, add_midpoints, add_table, add_embeddings, mirror=True, simplify=simplify, long_edition=False)

def get_specified_features(videos, sequence_frames, add_keypoints=False, add_midpoints=False, add_table=False, add_embeddings=False, simplify=False, mirror=False):
    results = []
    labels = []
    
    for video in videos:
        data, video_labels = get_keypoints_and_labels(video, sequence_frames=sequence_frames, raw=False, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_table=add_table, add_embeddings=add_embeddings, simplify=simplify, mirror=mirrored_only)
        results.extend(data)
        labels.extend(video_labels)
        
    return results, labels

# Combine all data
def get_splits(type, sequence_frames, add_keypoints=True, add_midpoints=False, add_table=False, add_embeddings=False, mirror=False):
    if type == "embeddings":
        all_data, all_labels = get_embeddings(videos, simplify)
    elif type == "keypoints_raw":
        all_data, all_labels = get_keypoints_raw(videos, simplify)
    elif type == "keypoints":
        all_data, all_labels = get_keypoints(videos, simplify)
    elif type == "concat_raw":
        all_data, all_labels = get_concatenated_raw(videos, simplify)
    elif type == "keypoints_time":
        all_data, all_labels = get_keypoints_time(videos, simplify)
    elif type == "keypoint_time_and_mid":
        all_data, all_labels = get_keypoints_time_and_mid(videos, simplify)
    elif type == "keypoint_time_and_mid_and_tab":
        all_data, all_labels = get_keypoints_time_and_mid_and_tab(videos, simplify)
    elif type == "everything":
        all_data, all_labels = get_every_feature(videos, simplify)
    elif type == "specify":
        all_data, all_labels = jaja5(videos )
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
        elif type == "keypoints_raw":
            train_embeddings, train_labels = get_keypoints_raw(train_videos, simplify)
        elif type == "keypoints":
            train_embeddings, train_labels = get_keypoints(train_videos, simplify)
        elif type == "concat_raw":
            train_embeddings, train_labels = get_concatenated_raw(train_videos, simplify)
        elif type == "keypoints_time":
            train_embeddings, train_labels = get_keypoints_time(train_videos, simplify)
        elif type == "keypoint_time_and_mid":
            train_embeddings, train_labels = get_keypoints_time_and_mid(train_videos, simplify)
        elif type == "keypoint_time_and_mid_and_tab":
            train_embeddings, train_labels = get_keypoints_time_and_mid_and_tab(videos, simplify)
        elif type == "everything":
            train_embeddings, train_labels = get_every_feature(videos, simplify)
        elif type == "specify":
            train_embeddings, train_labels = jaja(videos, simplify)
        else:
            train_embeddings, train_labels = get_concatenated(train_videos, simplify)

        # Filter test samples from video 3 that have seen labels
        train_label_set = set(train_labels)
        filtered_test_embeddings = []
        filtered_test_labels = []

        if type == "embeddings":
            video3_embeddings, video3_labels = get_embeddings(test_videos, simplify)
        elif type == "keypoints_raw":
            video3_embeddings, video3_labels = get_keypoints_raw(test_videos, simplify)
        elif type == "keypoints":
            video3_embeddings, video3_labels = get_keypoints(test_videos, simplify)
        elif type == "concat_raw":
            video3_embeddings, video3_labels = get_concatenated_raw(test_videos, simplify)
        elif type == "keypoints_time":
            video3_embeddings, video3_labels = get_keypoints_time(test_videos, simplify)
        elif type == "keypoint_time_and_mid":
            video3_embeddings, video3_labels = get_keypoints_time_and_mid(test_videos, simplify)
        elif type == "keypoint_time_and_mid_and_tab":
            video3_embeddings, video3_labels = get_keypoints_time_and_mid_and_tab(test_videos, simplify)
        elif type == "everything":
            video3_embeddings, video3_labels = get_every_feature(test_videos, simplify)
        elif type == "specify":
            video3_embeddings, video3_labels = jaja(videos, simplify)
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
    
if per_player_classifiers:
    print("Classification on embeddings")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("embeddings", False)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints raw")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoints_raw")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on embeddings and raw keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("concat_raw")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoints")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on embeddings and keypoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("concat")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoints_time")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time with midpoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoint_time_and_mid")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on keypoints over time with midpoints along with table midpoints")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("keypoint_time_and_mid_and_tab")
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")
    print("Classification on all features")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("everything", True)
    classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    print("-----------")

print("Specify keypoints")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("specify", True)
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
print("Specify keypoints, midpoints")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("specify", True, True)
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
print("Specify keypoints, midpoints and table midpoints")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("specify", True, True, True)
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")
print("Specify keypoints, midpoints, table midpoints, and embeddings")
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = get_splits("specify", True, True, True, True)
classify(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
print("-----------")