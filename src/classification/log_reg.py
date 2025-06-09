import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os
import sys
import json
import joblib
import time
import statistics
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../'))

from utility_functions import (plot_label_distribution, plot_confusion_matrix, get_features, plot_probabilities, plot_accuracies)

cross_validation = True
per_player_classifiers = False
test_on_one = True
simplify = True
mirrored_only = False
add_mirrored = False
test_on_no_stroke = False
timestamp = time.strftime("%Y%m%d_%H%M%S")
frame_range = 90

# Generic processing function
def process_videos(videos, sequence, raw, add_keypoints, add_midpoints, add_rackets, add_table, add_ball, add_scores, add_k_score, add_embeddings, simplify, missing_strat, long_edition=False):
    results = []
    labels = []

    for video in videos:
        data, video_labels, frames, skipped_frames = get_features(video_number=video, sequence_range=sequence, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_table=add_table, add_ball=add_ball, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=mirrored_only, simplify=simplify, long_edition=long_edition)
        results.extend(data)
        labels.extend(video_labels)

        if add_mirrored and len(videos) > 1:
            data, video_labels, frames, skipped_frames = get_features(video_number=video, sequence_range=sequence, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_table=add_table, add_ball=add_ball, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=True, simplify=simplify, long_edition=long_edition)
            results.extend(data)
            labels.extend(video_labels)

    return results, labels, frames, skipped_frames

# Combine all data
def get_splits(train_videos, test_videos, long_sequence=False, raw=False, add_keypoints=False, add_midpoints=False, add_rackets=False, add_table=False, add_ball=False, add_scores=False, add_k_score=False, add_embeddings=False, missing_strat="default", process_both_players=False):
    if long_sequence:
        sequence_frames = frame_range
    else:
        sequence_frames = 0
    
    all_data, all_labels, _, _ = process_videos(train_videos, sequence_frames, raw, add_keypoints, add_midpoints, add_rackets, add_table, add_ball, add_scores, add_k_score, add_embeddings, simplify, missing_strat, process_both_players)

    if len(all_data) == 0:
        print("⚠️ No samples for this experiment – skipping.")
        return None, None, None, None, None, None, None, None, None

    # Encode all labels
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)
    all_embeddings_np = np.vstack(all_data)

    def can_stratify(labels):
        label_counts = Counter(labels)
        return all(count >= 2 for count in label_counts.values())

    if test_on_one:
        train_embeddings, train_labels, _, _ = process_videos(train_videos, sequence_frames, raw, add_keypoints, add_midpoints, add_rackets, add_table, add_ball, add_scores, add_k_score, add_embeddings, simplify, missing_strat, process_both_players)

        # Filter test samples from video 3 that have seen labels
        train_label_set = set(train_labels)
        filtered_test_embeddings = []
        filtered_test_labels = []

        video3_embeddings, video3_labels, frames, skipped_frames = process_videos(test_videos, sequence_frames, raw, add_keypoints, add_midpoints, add_rackets, add_table, add_ball, add_scores, add_k_score, add_embeddings, simplify, missing_strat, process_both_players)

        for emb, label in zip(video3_embeddings, video3_labels):
            if label in train_label_set or label in 'left_forhand': # Change this
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
        X_temp, X_test, y_temp, y_test = train_test_split(all_embeddings_np, all_labels_encoded, test_size=0.1, random_state=42, stratify=strat)

        if can_stratify(y_temp):
            strat_temp = y_temp
        else:
            strat_temp = None
            print("⚠️ Not all classes in temp split have ≥2 samples. Validation will be non-stratified.")

        # Then: 10% of 90% (i.e. 10% overall) for validation
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42, stratify=strat_temp)

    return X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder

def classify(X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder):
    probabilities = []
    # Filter out skipped frames from the original `frames` list
    if frames is None:
        print("⚠️ No samples for this experiment – skipping.")
        return None, None, None, None, None
    
    filtered_frames = [f for f in frames if f not in skipped_frames]

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
    print(f"Baseline Accuracy:                      {baseline_acc:.2f}")
    
    # Train accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Logistic Regression Train Accuracy:     {train_accuracy:.2f}")

    # Validation accuracy
    if X_val is not None:
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy:                {val_accuracy:.2f}")

    # Test accuracy
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Logistic Regression Test Accuracy:      {test_accuracy:.2f}")
    
    # --- Softmax outputs ---
    y_test_probs = clf.predict_proba(X_test)
    class_names = label_encoder.classes_
    
    # Find the index of "no_stroke" in class_names
    if test_on_no_stroke:
        no_stroke_index = class_names.tolist().index('no_stroke')

        for i in range(len(X_test)):
            for j in range(len(y_test_probs[i])):
                # probs, true_label
                probabilities.append({
                    "predicted_class": class_names[j],
                    "probability": y_test_probs[i][j],
                    "probabilities": [prob for k, prob in enumerate(y_test_probs[i]) if k != no_stroke_index],
                    "true_class": class_names[y_test[i]]
                })
    else:
        for i in range(len(X_test)):
            most_probable_index = np.argmax(y_test_probs[i])
            probabilities.append({
                "true_class": class_names[y_test[i]],
                "frame": filtered_frames[i],
                "predicted_class": class_names[most_probable_index],
                "probabilities": dict(zip(class_names, y_test_probs[i].tolist()))
            })

    # Train Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    
    # Random forest train accuracy
    rf_train_acc = accuracy_score(y_train, clf_rf.predict(X_train))
    print(f"Random Forest Train Accuracy:            {rf_train_acc:.2f}")

    if X_val is not None:
        rf_val_acc = accuracy_score(y_val, clf_rf.predict(X_val))
        print(f"Random Forest Validation Accuracy:  {rf_val_acc:.2f}")

    # Random forest test accuracy
    rf_test_acc = accuracy_score(y_test, clf_rf.predict(X_test))
    print(f"Random Forest Test Accuracy:            {rf_test_acc:.2f}")
    
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)

    return probabilities, y_test_decoded, y_test_pred_decoded, test_accuracy, train_accuracy, rf_test_acc, rf_train_acc, clf, clf_rf

def save_predictions(data, filename, output_dir):
    """Saves the prediction data as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(data, f, indent=2)

    print(f"Predictions saved to {os.path.join(output_dir, filename)}")


experiments = [
    {"desc": "01_embeddings", "kwargs":                                     {"add_embeddings": True}},
    #{"desc": "02_raw_keypoints", "kwargs":                                  {"raw": True, "add_keypoints": True}},
    # {"desc": "02_raw_keypoints_add_scores", "kwargs":                       {"raw": True, "add_keypoints": True, "add_scores": True}},
    # {"desc": "03_raw_keypoints_rackets", "kwargs":                          {"raw": True, "add_keypoints": True, "add_rackets": True}},
    # {"desc": "03_raw_keypoints_rackets_add_scores", "kwargs":               {"raw": True, "add_keypoints": True, "add_rackets": True, "add_scores": True}},
    # {"desc": "04_raw_keypoints_ball", "kwargs":                             {"raw": True, "add_keypoints": True, "add_ball": True}},
    # {"desc": "04_raw_keypoints_ball_add_scores", "kwargs":                  {"raw": True, "add_keypoints": True, "add_ball": True, "add_scores": True}},
    # {"desc": "05_raw_keypoints_embeddings", "kwargs":                       {"raw": True, "add_keypoints": True, "add_embeddings": True}},
    # {"desc": "05_raw_keypoints_embeddings_add_scores", "kwargs":            {"raw": True, "add_keypoints": True, "add_embeddings": True, "add_scores": True}},
    # {"desc": "06_raw_keypoints_time", "kwargs":                             {"long_sequence": True, "raw": True, "add_keypoints": True}},
    # {"desc": "06_raw_keypoints_time_add_scores", "kwargs":                  {"long_sequence": True, "raw": True, "add_keypoints": True, "add_scores": True}},
    # {"desc": "07_raw_keypoints_rackets_ball_time", "kwargs":                {"long_sequence": True, "raw": True, "add_keypoints": True, "add_rackets": True, "add_ball": True}},
    # {"desc": "07_raw_keypoints_rackets_ball_time_add_scores", "kwargs":     {"long_sequence": True, "raw": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True}},
    # {"desc": "08_norm_keypoints", "kwargs":                                 {"add_keypoints": True}},
    # {"desc": "08_norm_keypoints_add_scores", "kwargs":                      {"add_keypoints": True, "add_scores": True}},
    # {"desc": "09_norm_keypoints_rackets", "kwargs":                         {"add_keypoints": True, "add_rackets": True}},
    # {"desc": "09_norm_keypoints_rackets_add_scores", "kwargs":              {"add_keypoints": True, "add_rackets": True, "add_scores": True}},
    # {"desc": "10_norm_keypoints_ball", "kwargs":                            {"add_keypoints": True, "add_ball": True}},
    # {"desc": "10_norm_keypoints_ball_add_scores", "kwargs":                 {"add_keypoints": True, "add_ball": True, "add_scores": True}},
    {"desc": "11_norm_keypoints_embeddings", "kwargs":                      {"add_keypoints": True, "add_embeddings": True}},
    #{"desc": "11_norm_keypoints_embeddings_add_scores", "kwargs":           {"add_keypoints": True, "add_embeddings": True, "add_scores": True}},
    # {"desc": "12_norm_keypoints_time", "kwargs":                            {"long_sequence": True, "add_keypoints": True}},
    # {"desc": "12_norm_keypoints_time_k_score", "kwargs":                    {"long_sequence": True, "add_keypoints": True}, "add_k_score": True},
    # {"desc": "12_norm_keypoints_time_add_scores", "kwargs":                 {"long_sequence": True, "add_keypoints": True, "add_scores": True}},
    # {"desc": "13_norm_keypoints_midpoints_time", "kwargs":                  {"long_sequence": True, "add_keypoints": True, "add_midpoints": True}},
    # {"desc": "13_norm_keypoints_midpoints_time_add_scores", "kwargs":       {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_scores": True}},
    # {"desc": "13_norm_keypoints_midpoints_time_add_scores_k_score", "kwargs": {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_scores": True, "add_k_score": True}},
    {"desc": "14_norm_keypoints_midpoints_table_time", "kwargs":            {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True}},
    #{"desc": "14_norm_keypoints_midpoints_table_time_add_scores", "kwargs": {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True}},
    {"desc": "14_norm_keypoints_midpoints_table_time_ball", "kwargs":       {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_ball": True}},
    {"desc": "15_norm_keypoints_rackets_ball_time", "kwargs":               {"long_sequence": True, "add_keypoints": True, "add_rackets": True, "add_ball": True}},
    {"desc": "15_norm_keypoints_rackets_ball_time_add_scores", "kwargs":    {"long_sequence": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True}},
    {"desc": "16_norm_keypoints_all_time", "kwargs":                        {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_rackets": True, "add_ball": True}},
    {"desc": "16_norm_keypoints_all_time_add_scores", "kwargs":             {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_rackets": True, "add_ball": True, "add_scores": True}},
    #{"desc": "17_norm_keypoints_all_embeddings_time", "kwargs":             {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_rackets": True, "add_ball": True, "add_embeddings": True}},
    #{"desc": "17_norm_keypoints_all_embeddings_time_add_scores", "kwargs":  {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_rackets": True, "add_ball": True, "add_embeddings": True, "add_scores": True}},
]

for exp in experiments:
    print(f"Running experiment: {exp['desc']}")
    print(f'Frame range: {frame_range}')
    
    # Prepare filenames and directories
    strat = "replace"
    filename = exp["desc"].replace(" ", "_").replace(",", "").lower()
    save_dir = f"results/{strat}/{timestamp}/{filename}"
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(f"results/{strat}/{timestamp}", "log.txt")
    
    # Open the log file in append mode ("a") to avoid overwriting
    with open(log_path, "a") as log_file, redirect_stdout(log_file):
        print(f"+++++++++++++++ Running experiment: {exp['desc']} +++++++++++++++")
        
        train_accuracies = []
        accuracies = []
        
        train_accuracies_rf = []
        accuracies_rf = []
        

        splits = [([1,2], [3]), ([2,3], [1]), ([1,3], [2]), ([1,2,3], [4])]
        
        for idx, split in enumerate(splits, start=1):
            train_videos, test_videos = split
            
            if len(train_videos) < 3:
                if cross_validation:
                    print(f"========== Iteration {idx} ==========")
                else:
                    continue
            else:
                print(f"========== Full model ==========")
                
            train_videos, test_videos = split

            X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder = get_splits(
                **exp["kwargs"],
                train_videos=train_videos,
                test_videos=test_videos,
                missing_strat=strat,
                process_both_players=True
            )

            probs, y_test_decoded, y_test_pred_decoded, test_accuracy, train_accuracy, test_accuracy_rf, train_accuracy_rf, log_clf, rf_clf = classify(
                X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder
            )
            
            plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, save_dir, concatenate=True, iteration=str(idx))
            
            train_accuracies.append(train_accuracy)
            accuracies.append(test_accuracy)
            train_accuracies_rf.append(train_accuracy_rf)
            accuracies_rf.append(test_accuracy_rf)
            
            if probs is None:
                print("Not enough samples for this experiment")
                print("\nxxxxxxxxxxxxxx")
                continue

            if len(train_videos) < 3:
                save_predictions(probs, os.path.join(save_dir, f"{filename}.json"), ".")
                joblib.dump(log_clf, os.path.join(save_dir, "logistic_regression_model.joblib"))
                joblib.dump(rf_clf, os.path.join(save_dir, "random_forest_model.joblib"))
                joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))

            if "probs" in locals():
                pass
                #plot_probabilities(probs, len(X_test))
                
        plot_accuracies(train_accuracies, accuracies, f"{save_dir}_log")
        plot_accuracies(train_accuracies_rf, accuracies_rf, f"{save_dir}_rf")

        if cross_validation:
            print("-----------")
            print(f"Logistic regression cross-validation train accuracy:    {statistics.mean(train_accuracies)}")
            print(f"Logistic regression cross-validation test accuracy:     {statistics.mean(accuracies)}")
            print(f"Random Forest cross-validation train accuracy:          {statistics.mean(train_accuracies_rf)}")
            print(f"Random Forest cross-validation test accuracy:           {statistics.mean(accuracies_rf)}")
        print("\nxxxxxxxxxxxxxx")

    print(f"Finished: {exp['desc']}, log saved to: {log_path}")



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
