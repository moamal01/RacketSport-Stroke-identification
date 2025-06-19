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
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath('../../'))

from utility_functions import (
    plot_label_distribution,
    plot_confusion_matrix,
    get_features,
    plot_probabilities,
    plot_accuracies,
    plot_umap2
)


debug = False

if debug:
    prefix = ""
else:
    prefix = "../../"

logistic_regression = True
max_iterations = 10000

random_forest = False
max_depth = 10

cross_validation = True
per_player_classifiers = False
test_on_one = True
simplify = True
mirrored_only = False
add_mirrored = False
test_on_no_stroke = False
timestamp = time.strftime("%Y%m%d_%H%M%S")
frame_range = 90

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
    stacked_data = np.vstack(all_data)

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
        X_temp, X_test, y_temp, y_test = train_test_split(stacked_data, all_labels_encoded, test_size=0.1, random_state=42, stratify=strat)

        if can_stratify(y_temp):
            strat_temp = y_temp
        else:
            strat_temp = None
            print("⚠️ Not all classes in temp split have ≥2 samples. Validation will be non-stratified.")

        # Then: 10% of 90% (i.e. 10% overall) for validation
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42, stratify=strat_temp)

    return all_labels, all_data, X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder

def classify(X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder, log=True, rf=True):
    probabilities = []
    probabilities_rf = []
    test_accuracy = 0
    train_accuracy = 0
    rf_test_acc = 0
    rf_train_acc = 0
    clf = None
    clf_rf = None
    
    y_test_pred_decoded = None
    y_test_pred_decoded_rf = None
    
    f1_test_scores_log = []
    f1_train_scores_log = []
    f1_train_scores_rf = []
    f1_test_scores_rf = []
    
    class_names = label_encoder.classes_
    
    # Filter out skipped frames from the original `frames` list
    if frames is None:
        print("⚠️ No samples for this experiment – skipping.")
        return None, None, None, None, None, None, None, None, None, None
    
    filtered_frames = [f for f in frames if f not in skipped_frames]

    # Print stats
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}, Test samples: {len(X_test)}")

    #plot_label_distribution(label_encoder.inverse_transform(y_train), "Train Set Label Distribution", simplify=simplify)
    #if y_val is not None:
    #    plot_label_distribution(label_encoder.inverse_transform(y_val), "Validation Set Label Distribution", simplify=simplify)
    #plot_label_distribution(label_encoder.inverse_transform(y_test), "Test Set Label Distribution", simplify=simplify)

    # Train Logistic Regression
    if log:
        clf = LogisticRegression(max_iter=max_iterations, solver='saga', penalty='l2')
        clf.fit(X_train, y_train)
        
        # Find most common class (already done)
        class_counts = Counter(y_train)
        most_common_class = max(class_counts, key=class_counts.get)
        baseline_acc = class_counts[most_common_class] / len(y_train)

        # Baseline predictions (predict the most common class for all samples)
        y_train_baseline_pred = np.full_like(y_train, fill_value=most_common_class)

        # Compute F1 scores for baseline classifier
        f1_macro_baseline = f1_score(y_train, y_train_baseline_pred, average='macro')
        f1_weighted_baseline = f1_score(y_train, y_train_baseline_pred, average='weighted')
        f1_micro_baseline = f1_score(y_train, y_train_baseline_pred, average='micro')

        # Print them
        print("# Baseline Classifier ----")
        print(f"Baseline Accuracy:                      {baseline_acc:.2f}")
        print(f"Baseline F1 Score:                       Macro {f1_macro_baseline:.2f}")
        print(f"Baseline F1 Score:                       Weighted {f1_weighted_baseline:.2f}")
        print(f"Baseline F1 Score:                       Micro {f1_micro_baseline:.2f}")
        
        # Train accuracy
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        f1_macro_train = f1_score(y_train, y_train_pred, average='macro')
        f1_weighted_train = f1_score(y_train, y_train_pred, average='weighted')
        f1_micro_train = f1_score(y_train, y_train_pred, average='micro')
        print("# Logistic Regression training ----")
        print(f"Logistic Regression Train Accuracy:     {train_accuracy:.2f}")
        print(f"F1 Score:                               Macro {f1_macro_train:.2f}")
        print(f"F1 Score:                               Weighted {f1_weighted_train:.2f}")
        print(f"F1 Score:                               Micro {f1_micro_train:.2f}")
        f1_train_scores_log.append(f1_macro_train)
        f1_train_scores_log.append(f1_weighted_train)
        f1_train_scores_log.append(f1_micro_train)

        # Validation accuracy
        if X_val is not None:
            y_val_pred = clf.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy:                {val_accuracy:.2f}")

        # Test accuracy
        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        f1_macro_test = f1_score(y_test, y_test_pred, average='macro')
        f1_weighted_test = f1_score(y_test, y_test_pred, average='weighted')
        f1_micro_test = f1_score(y_test, y_test_pred, average='micro')
        print("# Logistic Regression test --------")
        print(f"Logistic Regression Test Accuracy:          +{test_accuracy:.2f}+")
        print(f"F1 Score:                                   Macro +{f1_macro_test:.2f}+")
        print(f"F1 Score:                                   Weighted +{f1_weighted_test:.2f}+")
        print(f"F1 Score:                                   Micro +{f1_micro_test:.2f}+")
        f1_test_scores_log.append(f1_macro_test)
        f1_test_scores_log.append(f1_weighted_test)
        f1_test_scores_log.append(f1_micro_test)
        
        y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
        
        # --- Softmax outputs ---
        y_test_probs = clf.predict_proba(X_test)
        
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
    if rf:
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=max_depth)
        clf_rf.fit(X_train, y_train)
        
        # Random forest train accuracy
        y_train_pred_rf = clf_rf.predict(X_train)
        
        rf_train_acc = accuracy_score(y_train, y_train_pred_rf)
        f1_macro_rf_train = f1_score(y_train, y_train_pred_rf, average='macro')
        f1_weighted_rf_train = f1_score(y_train, y_train_pred_rf, average='weighted')
        f1_micro_rf_train = f1_score(y_train, y_train_pred_rf, average='micro')

        print("# Random Forest training ----------")
        print(f"Random Forest Train Accuracy:               {rf_train_acc:.2f}")
        print(f"Random Forest Test Accuracy:                    +{rf_test_acc:.2f}+")
        print(f"F1 Score:                                       Macro +{f1_macro_rf_train:.2f}+")
        print(f"F1 Score:                                       weighted +{f1_weighted_rf_train:.2f}+")
        print(f"F1 Score:                                       Micro +{f1_micro_rf_train:.2f}+")
        f1_train_scores_rf.append(f1_macro_rf_train)
        f1_train_scores_rf.append(f1_weighted_rf_train)
        f1_train_scores_rf.append(f1_micro_rf_train)

        if X_val is not None:
            rf_val_acc = accuracy_score(y_val, clf_rf.predict(X_val))
            print(f"Random Forest Validation Accuracy:  {rf_val_acc:.2f}")

        # Random forest test accuracy
        y_test_pred_rf = clf_rf.predict(X_test)
        rf_test_acc = accuracy_score(y_test, y_test_pred_rf)

        f1_macro_rf = f1_score(y_test, y_test_pred_rf, average='macro')
        f1_weighted_rf = f1_score(y_test, y_test_pred_rf, average='weighted')
        f1_micro_rf = f1_score(y_test, y_test_pred_rf, average='micro')
        
        print("# Random Forest test --------------")
        print(f"Random Forest Test Accuracy:                +{rf_test_acc:.2f}+")
        print(f"F1 Score:                                       Macro +{f1_macro_rf:.2f}+")
        print(f"F1 Score:                                       weighted +{f1_weighted_rf:.2f}+")
        print(f"F1 Score:                                       Micro +{f1_micro_rf:.2f}+")
        f1_test_scores_rf.append(f1_macro_rf)
        f1_test_scores_rf.append(f1_weighted_rf)
        f1_test_scores_rf.append(f1_micro_rf)
        
        y_test_pred_decoded_rf = label_encoder.inverse_transform(y_test_pred_rf)
        
        # --- Softmax outputs ---
        y_test_probs_rf = clf_rf.predict_proba(X_test)
        
        if test_on_no_stroke:
            no_stroke_index = class_names.tolist().index('no_stroke')

            for i in range(len(X_test)):
                for j in range(len(y_test_probs_rf[i])):
                    # probs, true_label
                    probabilities_rf.append({
                        "predicted_class": class_names[j],
                        "probability": y_test_probs_rf[i][j],
                        "probabilities": [prob for k, prob in enumerate(y_test_probs_rf[i]) if k != no_stroke_index],
                        "true_class": class_names[y_test[i]]
                    })
        else:
            for i in range(len(X_test)):
                most_probable_index = np.argmax(y_test_probs_rf[i])
                probabilities_rf.append({
                    "true_class": class_names[y_test[i]],
                    "frame": filtered_frames[i],
                    "predicted_class": class_names[most_probable_index],
                    "probabilities": dict(zip(class_names, y_test_probs_rf[i].tolist()))
                })
    
    y_test_decoded = label_encoder.inverse_transform(y_test)

    return probabilities, probabilities_rf, y_test_decoded, y_test_pred_decoded, y_test_pred_decoded_rf, test_accuracy, train_accuracy, f1_train_scores_log, f1_test_scores_log, rf_test_acc, rf_train_acc, f1_train_scores_rf, f1_test_scores_rf, clf, clf_rf

def save_predictions(data, filename, output_dir):
    """Saves the prediction data as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(data, f, indent=2)

experiments = [
    {"desc": "01_clip", "kwargs":                                                       {"add_embeddings": True}},
    {"desc": "02_raw_keypoints_clip", "kwargs":                                         {"raw": True, "add_keypoints": True, "add_embeddings": True}},
    {"desc": "03_raw_keypoints_clip_fullscores", "kwargs":                              {"raw": True, "add_keypoints": True, "add_embeddings": True, "add_scores": True, "add_k_score": True}},

    {"desc": "04_raw_keypoints", "kwargs":                                              {"raw": True, "add_keypoints": True}},
    {"desc": "05_raw_keypoints_scores", "kwargs":                                       {"raw": True, "add_keypoints": True, "add_scores": True}},
    {"desc": "06_raw_keypoints_fullscores", "kwargs":                                   {"raw": True, "add_keypoints": True, "add_scores": True, "add_k_score": True}},
    {"desc": "07_raw_keypoints_ball", "kwargs":                                         {"raw": True, "add_keypoints": True, "add_ball": True}},
    {"desc": "08_raw_keypoints_ball_scores", "kwargs":                                  {"raw": True, "add_keypoints": True, "add_ball": True, "add_scores": True}},
    {"desc": "09_raw_keypoints_ball_fullscores", "kwargs":                              {"raw": True, "add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True}},
    {"desc": "10_raw_keypoints_ball_racket_fullscores", "kwargs":                       {"raw": True, "add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True, "add_rackets": True}},
    {"desc": "11_raw_keypoints_ball_racket_clip_fullscores", "kwargs":                  {"raw": True, "add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True, "add_rackets": True, "add_embeddings": True}},

    {"desc": "12_raw_keypoints_time", "kwargs":                                         {"long_sequence": True, "raw": True, "add_keypoints": True}},
    {"desc": "13_raw_keypoints_time_scores", "kwargs":                                  {"long_sequence": True, "raw": True, "add_keypoints": True, "add_scores": True}},
    {"desc": "14_raw_keypoints_time_fullscores", "kwargs":                              {"long_sequence": True, "raw": True, "add_keypoints": True, "add_scores": True, "add_k_score": True}},
    {"desc": "15_raw_keypoints_time_ball", "kwargs":                                    {"long_sequence": True, "raw": True, "add_keypoints": True, "add_ball": True}},
    {"desc": "16_raw_keypoints_time_ball_racket", "kwargs":                             {"long_sequence": True, "raw": True, "add_keypoints": True, "add_ball": True, "add_rackets": True}},
    {"desc": "17_raw_keypoints_ball_racket_time_scores", "kwargs":                      {"long_sequence": True, "raw": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True}},
    {"desc": "18_raw_keypoints_ball_racket_time_fullscores", "kwargs":                  {"long_sequence": True, "raw": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True, "add_k_score": True}},

    {"desc": "19_keypoints", "kwargs":                                                  {"add_keypoints": True}},
    {"desc": "20_keypoints_scores", "kwargs":                                           {"add_keypoints": True, "add_scores": True}},
    {"desc": "21_keypoints_fullscores", "kwargs":                                       {"add_keypoints": True, "add_scores": True, "add_k_score": True}},
    {"desc": "22_keypoints_ball", "kwargs":                                             {"add_keypoints": True, "add_ball": True}},
    {"desc": "23_keypoints_ball_scores", "kwargs":                                      {"add_keypoints": True, "add_ball": True, "add_scores": True}},
    {"desc": "24_keypoints_ball_fullscores", "kwargs":                                  {"add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True}},
    {"desc": "25_keypoints_ball_racket_fullscores", "kwargs":                           {"add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True, "add_rackets": True}},
    {"desc": "26_keypoints_ball_racket_clip_fullscores", "kwargs":                      {"add_keypoints": True, "add_ball": True, "add_scores": True, "add_k_score": True, "add_rackets": True, "add_embeddings": True}},
    
    {"desc": "27_keypoints_time", "kwargs":                                             {"long_sequence": True, "add_keypoints": True}},
    {"desc": "28_keypoints_time_scores", "kwargs":                                      {"long_sequence": True, "add_keypoints": True, "add_scores": True}},
    {"desc": "29_keypoints_time_fullscores", "kwargs":                                  {"long_sequence": True, "add_keypoints": True, "add_scores": True, "add_k_score": True}},
    {"desc": "30_keypoints_time_ball", "kwargs":                                        {"long_sequence": True, "add_keypoints": True, "add_ball": True}},
    {"desc": "31_keypoints_time_ball_racket", "kwargs":                                 {"long_sequence": True, "add_keypoints": True, "add_ball": True, "add_rackets": True}},
    {"desc": "32_keypoints_ball_racket_time_scores", "kwargs":                          {"long_sequence": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True}},
    {"desc": "33_keypoints_ball_racket_time_fullscores", "kwargs":                      {"long_sequence": True, "add_keypoints": True, "add_rackets": True, "add_ball": True, "add_scores": True, "add_k_score": True}},

    {"desc": "34_keypoints_midpoints_time", "kwargs":                                   {"long_sequence": True, "add_keypoints": True, "add_midpoints": True}},
    {"desc": "35_keypoints_midpoints_table_time", "kwargs":                             {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True}},
    {"desc": "36_keypoints_midpoints_table_time_scores", "kwargs":                      {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True}},
    {"desc": "37_keypoints_midpoints_table_time_fullscores", "kwargs":                  {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_k_score": True}},
    {"desc": "38_keypoints_midpoints_table_ball_time_fullscores", "kwargs":             {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_k_score": True, "add_ball": True}},
    {"desc": "39_keypoints_midpoints_table_ball_racket_time_fullscores", "kwargs":      {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_k_score": True, "add_ball": True, "add_rackets": True}},
    {"desc": "40_keypoints_midpoints_table_ball_racket_time_fullscores_clip", "kwargs": {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_k_score": True, "add_ball": True, "add_rackets": True, "add_embeddings": True}},

    {"desc": "41_keypoints_midpoints_table_ball_time_scores", "kwargs":             {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_ball": True}},
    {"desc": "42_keypoints_midpoints_table_ball_racket_time_scores", "kwargs":      {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_ball": True, "add_rackets": True}},
    {"desc": "43_keypoints_midpoints_table_ball_racket_time_scores_clip", "kwargs": {"long_sequence": True, "add_keypoints": True, "add_midpoints": True, "add_table": True, "add_scores": True, "add_ball": True, "add_rackets": True, "add_embeddings": True}}
]

for exp in experiments:
    print(f"Running experiment: {exp['desc']}")
    print(f'Frame range: {frame_range}')
    
    # Prepare filenames and directories
    strat = "replace"
    filename = exp["desc"].replace(" ", "_").replace(",", "").lower()
    save_dir = prefix + f"results/{strat}/{timestamp}/{filename}"
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(prefix + f"results/{strat}/{timestamp}", "log.txt")
    
    # Open the log file in append mode ("a") to avoid overwriting
    with open(log_path, "a") as log_file, redirect_stdout(log_file):
        print(f"+++++++++++++++ Running experiment: {exp['desc']} max depth {max_depth} +++++++++++++++")
        
        # Logistic regression lists
        train_accuracies = []
        f1_train_accuracies_macro = []
        f1_train_accuracies_weighted = []
        f1_train_accuracies_micro = []

        accuracies = []
        f1_accuracies_macro = []
        f1_accuracies_weighted = []
        f1_accuracies_micro = []
        
        # Random forest lists
        train_accuracies_rf = []
        f1_train_accuracies_macro_rf = []
        f1_train_accuracies_weighted_rf = []
        f1_train_accuracies_micro_rf = []

        accuracies_rf = []
        f1_accuracies_macro_rf = []
        f1_accuracies_weighted_rf = []
        f1_accuracies_micro_rf = []

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

            all_labels, all_data, X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder = get_splits(
                **exp["kwargs"],
                train_videos=train_videos,
                test_videos=test_videos,
                missing_strat=strat,
                process_both_players=True
            )

            probs, probs_rf, y_test_decoded, y_test_pred_decoded, y_test_pred_decoded_rf, test_accuracy, train_accuracy, f1_train_log, f1_test_log, test_accuracy_rf, train_accuracy_rf, f1_train_rf, f1_test_rf, log_clf, rf_clf = classify(
                X_train, y_train, X_val, y_val, X_test, y_test, frames, skipped_frames, label_encoder, logistic_regression, random_forest
            )
            
            if len(train_videos) < 3:
                if logistic_regression:
                    train_accuracies.append(train_accuracy)
                    accuracies.append(test_accuracy)
                    f1_train_accuracies_macro.append(f1_train_log[0])
                    f1_train_accuracies_weighted.append(f1_train_log[1])
                    f1_train_accuracies_micro.append(f1_train_log[2])

                    f1_accuracies_macro.append(f1_test_log[0])
                    f1_accuracies_weighted.append(f1_test_log[1])
                    f1_accuracies_micro.append(f1_test_log[2])
                
                if random_forest:
                    train_accuracies_rf.append(train_accuracy_rf)
                    accuracies_rf.append(test_accuracy_rf)
                    f1_train_accuracies_macro_rf.append(f1_train_rf[0])
                    f1_train_accuracies_weighted_rf.append(f1_train_rf[1])
                    f1_train_accuracies_micro_rf.append(f1_train_rf[2])

                    f1_accuracies_macro_rf.append(f1_test_rf[0])
                    f1_accuracies_weighted_rf.append(f1_test_rf[1])
                    f1_accuracies_micro_rf.append(f1_test_rf[2])

            if len(train_videos) > 2:
                plot_umap2(all_labels, all_data, 15, save_dir=save_dir)
                joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))
                
                if logistic_regression:
                    save_predictions(probs, f"{filename}_full.json", save_dir + "/prediction_log")
                    joblib.dump(log_clf, os.path.join(save_dir, "logistic_regression_model.joblib"))
                    
                if random_forest:
                    save_predictions(probs_rf, f"{filename}_full.json", save_dir + "/prediction_rf")
                    joblib.dump(rf_clf, os.path.join(save_dir, "random_forest_model.joblib"))

                
            else:
                if logistic_regression:
                    save_predictions(probs, f"{filename}_iteration{idx}.json", save_dir + "/prediction_log")
                    plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, save_dir + "/confusion_matrices/log", concatenate=True, iteration=f"{str(idx)}")
                    
                if random_forest:
                    save_predictions(probs_rf, f"{filename}_iteration{idx}.json", save_dir + "/prediction_rf")
                    plot_confusion_matrix(y_test_decoded, y_test_pred_decoded_rf, save_dir + "/confusion_matrices/rf", concatenate=True, iteration=f"{str(idx)}")
                    
                

            if "probs" in locals():
                pass
                #plot_probabilities(probs, len(X_test))

        if logistic_regression:
            # Means
            mean_train_acc = statistics.mean(train_accuracies)
            mean_f1_macro_train_acc = statistics.mean(f1_train_accuracies_macro)
            mean_f1_weighted_train_acc = statistics.mean(f1_train_accuracies_weighted)
            mean_f1_micro_train_acc = statistics.mean(f1_train_accuracies_micro)
            
            mean_acc = statistics.mean(accuracies)
            mean_f1_macro_acc = statistics.mean(f1_accuracies_macro)
            mean_f1_weighted_acc = statistics.mean(f1_accuracies_weighted)
            mean_f1_micro_acc = statistics.mean(f1_accuracies_micro)

            # Appending
            train_accuracies.append(mean_train_acc)
            f1_train_accuracies_macro.append(mean_f1_macro_train_acc)
            f1_train_accuracies_weighted.append(mean_f1_weighted_train_acc)
            f1_train_accuracies_micro.append(mean_f1_micro_train_acc)
            
            accuracies.append(mean_acc)
            f1_accuracies_macro.append(mean_f1_macro_acc)
            f1_accuracies_weighted.append(mean_f1_weighted_acc)
            f1_accuracies_micro.append(mean_f1_micro_acc)
            
            # Plotting
            plot_accuracies(train_accuracies, accuracies, f"{save_dir}/accuracies_log.png")
            plot_accuracies(f1_train_accuracies_macro, f1_accuracies_macro, f"{save_dir}/f1_plots_log/f1_macro_scores.png")
            plot_accuracies(f1_train_accuracies_weighted, f1_accuracies_weighted, f"{save_dir}/f1_plots_log/f1_weighted_scores.png")
            plot_accuracies(f1_train_accuracies_micro, f1_accuracies_micro, f"{save_dir}/f1_plots_log/f1_micro_scores.png")
            
        if random_forest:
            # Means
            mean_train_acc_rf = statistics.mean(train_accuracies_rf)
            mean_f1_macro_train_acc_rf = statistics.mean(f1_train_accuracies_macro_rf)
            mean_f1_weighted_train_acc_rf = statistics.mean(f1_train_accuracies_weighted_rf)
            mean_f1_micro_train_acc_rf = statistics.mean(f1_train_accuracies_micro_rf)
            
            mean_acc_rf = statistics.mean(accuracies_rf)
            mean_f1_macro_acc_rf = statistics.mean(f1_accuracies_macro_rf)
            mean_f1_weighted_acc_rf = statistics.mean(f1_accuracies_weighted_rf)
            mean_f1_micro_acc_rf = statistics.mean(f1_accuracies_micro_rf)

            # Appending
            train_accuracies_rf.append(mean_train_acc_rf)
            f1_train_accuracies_macro_rf.append(mean_f1_macro_train_acc_rf)
            f1_train_accuracies_weighted_rf.append(mean_f1_weighted_train_acc_rf)
            f1_train_accuracies_micro_rf.append(mean_f1_micro_train_acc_rf)
            
            accuracies_rf.append(mean_acc_rf)
            f1_accuracies_macro_rf.append(mean_f1_macro_acc_rf)
            f1_accuracies_weighted_rf.append(mean_f1_weighted_acc_rf)
            f1_accuracies_micro_rf.append(mean_f1_micro_acc_rf)
            
            # Plotting
            plot_accuracies(train_accuracies_rf, accuracies_rf, f"{save_dir}/accuracies_rf.png")
            plot_accuracies(f1_train_accuracies_macro_rf, f1_accuracies_macro_rf, f"{save_dir}/f1_plots_rf/f1_macro_scores.png")
            plot_accuracies(f1_train_accuracies_weighted_rf, f1_accuracies_weighted_rf, f"{save_dir}/f1_plots_rf/f1_weighted_scores.png")
            plot_accuracies(f1_train_accuracies_micro_rf, f1_accuracies_micro_rf, f"{save_dir}/f1_plots_rf/f1_micro_scores.png")

        if cross_validation:
            print("-----------")
            if logistic_regression:
                print(f"Logistic regression cross-validation train accuracy:    {mean_train_acc}")
                print(f"Logistic regression cross-validation test accuracy:     {mean_acc}")
                print(f"Logistic regression cross-validation f1 macro score:    {mean_f1_macro_acc}")
                print(f"Logistic regression cross-validation f1 weighted score: {mean_f1_weighted_acc}")
                print(f"Logistic regression cross-validation f1 micro score:    {mean_f1_micro_acc}\n")
                
            if random_forest:
                print(f"Random forest cross-validation train accuracy:          {mean_train_acc_rf}")
                print(f"Random forest cross-validation test accuracy:           {mean_acc_rf}")
                print(f"Random forest cross-validation f1 macro score:          {mean_f1_macro_acc_rf}")
                print(f"Random forest cross-validation f1 weighted score:       {mean_f1_weighted_acc_rf}")
                print(f"Random forest cross-validation f1 micro score:          {mean_f1_micro_acc_rf}")

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
