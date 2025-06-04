import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

joint_list = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

feature_labels = [f"{kp}_{coord}" for kp in joint_list for coord in ['x', 'y']]

def mirror_string(input_str):
    """Returns the a string with left and right swapped.

    Args:
        path (str): The string that should be mirrored.

    Returns:
        str: The mirrored string.
    """
    mirrored = input_str.replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')
    return mirrored


def load_json_with_dicts(path: str) -> dict:
    """Returns the contents of a json file.

    Args:
        path (str): The path to the json file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path, "r") as file:
        data: dict = json.load(file)

    return data


def get_timestamps(video_number):
    with open(f"data/extended_events/events_markup{video_number}.json", "r") as file:
        data = json.load(file)
        
    excluded_values = {"empty_event", "bounce", "net"}

    return {k: v for k, v in data.items() if v not in excluded_values}

def get_timestamps2(video_number):
    with open(f"data/extended_events/events_markup{video_number}.json", "r") as file:
        data = json.load(file)

    return {k: v for k, v in data.items() if v in "no_stroke"}

def get_player_and_label(value, player_to_get, simplify, mirror=False):
    if mirror:
            value = mirror_string(value)

    if value == "right_forehand_loop right_leaning left_foot_lifted":
        pass

    if "pause" not in value or "play" not in value:
        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0]
    else:
        label = value

    if player != player_to_get and player_to_get != "both":
        return

    if simplify:
        if "serve" in label:
            label = f"{player}_{label_parts[2]}"
        elif "play" in label or "pause" in label:
            pass
        else:
            label = f"{player}_{label_parts[1]}"
    
    return player, label


def get_player_features(df, frame, sequence_frame, raw, player, add_midpoints, add_rackets, add_scores, add_k_score, missing_strat="default"):
    numbers = 2
    if add_k_score:
        numbers = 3

    Threshold = 0.9
        
    event_row = df[df['Event frame'] == int(frame)]
    idx = event_row.index[0]
    pos = df.index.get_loc(idx)
    event_row = df.iloc[pos + sequence_frame]
    pos = df.index.get_loc(idx) + sequence_frame # Cleanup

    score1 = event_row["Left score"]
    score2 = event_row["Right score"]

    column = f"Keypoints {player}" if raw else f"{player.capitalize()} mid-normalized"
    score = event_row[f"{player.capitalize()} score"]

    # Missing strategies
    if missing_strat == "default":
        if score1 < Threshold or score2 < Threshold:
            return None
        
        features = np.array(ast.literal_eval(event_row[column]))[:, :numbers].flatten()

        if add_midpoints:
            midpoint = ast.literal_eval(event_row[f"{player.capitalize()} player midpoint"])
            features = np.concatenate((features, midpoint))

        score = event_row[f"{player.capitalize()} score"]
        
        if add_scores:
            features = np.concatenate((features, np.array([score])))

        if add_rackets:
            racket1 = ast.literal_eval(event_row["Left racket"])
            racket2 = ast.literal_eval(event_row["Right racket"])
            if not racket1 or not racket2:
                return None
            else:
                racket = ast.literal_eval(event_row[f"{player.capitalize()} racket"])
                features = np.concatenate((features, racket))
                
                if add_scores:
                    racket_score = event_row[f"{player.capitalize()} racket score"]
                    features = np.concatenate((features, np.array([racket_score])))

    elif missing_strat == "replace":
        if score < Threshold:
            features = np.array([[-1, -1] for _ in range(17)])[:, :numbers].flatten()

            if add_midpoints:
                midpoint = np.array([-1, -1])
                features = np.concatenate((features, midpoint))
                
            if add_scores:
                score = np.array([-1])
                features = np.concatenate((features, score))

        if add_rackets:
            racket = ast.literal_eval(event_row[f"{player.capitalize()} racket"])
            if not racket:
                racket = np.array([-1, -1])
            features = np.concatenate((features, racket))
            
            if add_scores:
                racket_score = np.array([-1])
                features = np.concatenate((features, racket_score))
             
        return features

    elif missing_strat == "fall_back":
        if score < Threshold:
            not_found = True
            rows_back = 1

            while not_found:
                if pos - rows_back < 0:
                    return None

                prev_row = df.iloc[pos - rows_back]
                prev_score = prev_row[f"{player.capitalize()} score"]

                if prev_score > Threshold:
                    features = np.array(ast.literal_eval(prev_row[column]))[:, :numbers].flatten()
                    not_found = False

                    if add_midpoints:
                        midpoint = ast.literal_eval(prev_row[f"{player.capitalize()} player midpoint"])
                        features = np.concatenate((features, midpoint))
                        
                    if add_scores:
                        features = np.concatenate((features, np.array([prev_score])))
                else:
                    rows_back += 1

            if add_rackets:
                not_found = True
                rows_back = 0

                while not_found:
                    if pos - rows_back < 0:
                        racket = np.array([-1, -1])
                        prev_score = np.array([-1])
                        not_found = False

                    prev_row = df.iloc[pos - rows_back]
                    racket = ast.literal_eval(prev_row[f"{player.capitalize()} racket"])
                    prev_score = prev_row[f"{player.capitalize()} racket score"]
                    
                    if racket:
                        not_found = False
                    else:
                        rows_back += 1
                
                features = np.concatenate((features, racket))
                features = np.concatenate((features, np.array([prev_score])))
        else:
            features = np.array(ast.literal_eval(event_row[column])).flatten()

            if add_midpoints:
                midpoint = ast.literal_eval(event_row[f"{player.capitalize()} player midpoint"])
                features = np.concatenate((features, midpoint))

            score = event_row[f"{player.capitalize()} score"]
            
            if add_scores:
                features = np.concatenate((features, np.array([score])))
                
            if add_rackets:
                not_found = True
                rows_back = 0

                while not_found:
                    if pos - rows_back < 0: # Move check up
                        racket = np.array([-1, -1])
                        prev_score = np.array([-1])
                        not_found = False

                    prev_row = df.iloc[pos - rows_back]
                    racket = ast.literal_eval(prev_row[f"{player.capitalize()} racket"])
                    prev_score = prev_row[f"{player.capitalize()} racket score"]
                    
                    if racket:
                        not_found = False
                    else:
                        rows_back += 1
                
                features = np.concatenate((features, racket))
                features = np.concatenate((features, np.array([prev_score])))
            
            
    
    return features

def get_ball(df, frame, sequence_frame, features, add_scores):
    event_row = df[df['Event frame'] == int(frame)]
    idx = event_row.index[0]
    pos = df.index.get_loc(idx)
    event_row = df.iloc[pos + sequence_frame]
    pos = df.index.get_loc(idx) + sequence_frame # Cleanup

    not_found = True
    rows_back = 0
        
    while not_found:
        prev_row = df.iloc[pos - rows_back]
        ball = ast.literal_eval(prev_row['Ball midpoint'])
        score =prev_row['Ball score']
        
        if ball:
            not_found = False
        else:
            rows_back += 1

        if pos - rows_back < 0:
            ball = np.array([-1, -1])
            not_found = False

    features = concatenate_features(features, ball)
    
    if add_scores:
        features = np.concatenate((features, np.array([score])))

    return features


def get_table(df, frame, sequence_frame, features):
    event_row = df[df['Event frame'] == int(frame)]
    idx = event_row.index[0]
    pos = df.index.get_loc(idx)
    event_row = df.iloc[pos + sequence_frame]

    table_midpoint = ast.literal_eval(event_row[f"Table midpoint"])
    features = concatenate_features(features, table_midpoint) 
    
    return features


def get_embeddings(video_number, frame, player=None, single_player=False, mirror=False):
    mirrored = "m" if mirror else ""

    file_path_of_interest = f"embeddings/video_{video_number}{mirrored}/{frame}/0/{player}.npy"
    if single_player:
        if not os.path.exists(file_path_of_interest):
            return None
    else:
        file_path = f"embeddings/video_{video_number}{mirrored}/{frame}/0/left.npy"
        file_path2 = f"embeddings/video_{video_number}{mirrored}/{frame}/0/right.npy" 
        if not os.path.exists(file_path) or not os.path.exists(file_path2):
            return None
    
    return np.load(file_path_of_interest).squeeze()


def concatenate_features(features, new_features):
    if features is None:
        features = new_features
    else:
        features = np.concatenate((features, new_features))

    return features


def compose_features(df, frame, sequence_frame, video_number, player, features, raw=False, add_keypoints=False, add_midpoints=False, add_rackets=False, add_scores=False, add_k_score=False, add_embeddings=False, missing_strat="default", mirror=False):    
    event_row = df[df['Event frame'] == int(frame)]
    if event_row.empty:
        return

    if add_keypoints:
        keypoints = get_player_features(
            df=df,
            frame=frame,
            sequence_frame=sequence_frame, raw=raw,
            player=player,
            add_midpoints=add_midpoints,
            add_rackets=add_rackets,
            add_scores=add_scores,
            add_k_score=add_k_score,
            missing_strat=missing_strat,
        )
        
        if keypoints is None:
            return None

        features = concatenate_features(features, keypoints)
        
    if add_embeddings:
        embeddings = get_embeddings(video_number, frame, player)
        if embeddings is None:
            return None
        
        features = concatenate_features(features, embeddings) 
        
    return features


def get_features(video_number, sequence_range, sequence_gap=2, raw=False, add_keypoints=False, add_midpoints=False,
                        add_rackets=False, add_table=False, add_ball=False, add_scores=False, add_k_score=False,
                        add_embeddings=False, missing_strat="default", mirror=False, simplify=False, long_edition=False,
                        player_to_get="both"):
    feature_list = []
    labels = []


    if video_number == 1:
        timestamps = get_timestamps(video_number)
    else:
        timestamps = get_timestamps(video_number)

    keypoints_table = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)
    frames = []
    skipped_frames = []
    

    for frame, value in timestamps.items():
        frames.append(int(frame))

        if value in {"other", "otherotherother"}:
            continue

        player, label = get_player_and_label(value, player_to_get, simplify, mirror)
        features = None
        
        if long_edition:
            # Left player features
            for i in range(-sequence_range, sequence_range + sequence_gap):
                frame_feature = compose_features(df=df, frame=frame, sequence_frame=i, video_number=video_number, player="left", features=features, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=mirror)
                if frame_feature is None:
                    features = None
                    break
                features = frame_feature
            # Ball
            if add_ball:
                for i in range(-sequence_range, sequence_range + sequence_gap):
                    frame_feature = get_ball(df=df, frame=frame, sequence_frame=i, features=features, add_scores=add_scores)
                    if frame is None:
                        features = None
                        break
                    features = frame_feature
            # Table
            if add_table:
                for i in range(-sequence_range, sequence_range + sequence_gap):
                    frame_feature = get_table(df=df, frame=frame, sequence_frame=i, features=features)
                    if frame_feature is None:
                        features = None
                        break
                    features = frame_feature
            # Right player features
            for i in range(-sequence_range, sequence_range + sequence_gap):
                frame_feature = compose_features(df=df, frame=frame, sequence_frame=i, video_number=video_number, player="right", features=features, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=mirror)
                if frame_feature is None:
                    features = None
                    break
                features = frame_feature
        else:
            for i in range(-sequence_range, sequence_range + sequence_gap):
                features = compose_features(df, frame, i, video_number, player, features, raw, add_keypoints, add_midpoints, add_table, add_scores, add_k_score, add_embeddings, missing_strat)
        
        if features is not None:
            feature_list.append(features)
            labels.append(label)
        else:
            skipped_frames.append(int(frame))
            
    return feature_list, labels, frames, skipped_frames

def get_feature(video_number, frames, sequence_range, sequence_gap=1, raw=False, add_keypoints=False, add_midpoints=False,
                        add_rackets=False, add_table=False, add_ball=False, add_scores=False, add_k_score=False, add_embeddings=False,
                        missing_strat="default", mirror=False, simplify=False, long_edition=False, player_to_get="both"):

    keypoints_table = f"../data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)

    for frame in frames:
        player, _ = get_player_and_label("Unknown_unknown_unknown unknown unknown", player_to_get, simplify, mirror)
        features = None
        
        if long_edition:
            # Left player features
            for i in range(-sequence_range, sequence_range + sequence_gap):
                frame_feature = compose_features(df=df, frame=frame, sequence_frame=i, video_number=video_number, player="left", features=features, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=mirror)
                if frame_feature is None:
                    features = None
                    break
                features = frame_feature
            # Ball
            if add_ball:
                for i in range(-sequence_range, sequence_range + sequence_gap):
                    frame_feature = get_ball(df=df, frame=frame, sequence_frame=i, features=features, add_scores=add_scores)
                    if frame is None:
                        features = None
                        break
                    features = frame_feature
            # Table
            if add_table:
                for i in range(-sequence_range, sequence_range + sequence_gap):
                    frame_feature = get_table(df=df, frame=frame, sequence_frame=i, features=features)
                    if frame_feature is None:
                        features = None
                        break
                    features = frame_feature
            # Right player features
            for i in range(-sequence_range, sequence_range + sequence_gap):
                frame_feature = compose_features(df=df, frame=frame, sequence_frame=i, video_number=video_number, player="right", features=features, raw=raw, add_keypoints=add_keypoints, add_midpoints=add_midpoints, add_rackets=add_rackets, add_scores=add_scores, add_k_score=add_k_score, add_embeddings=add_embeddings, missing_strat=missing_strat, mirror=mirror)
                if frame_feature is None:
                    features = None
                    break
                features = frame_feature
        else:
            for i in range(-sequence_range, sequence_range + sequence_gap):
                features = compose_features(df, frame, i, video_number, player, features, raw, add_keypoints, add_midpoints, add_table, add_scores, add_embeddings, missing_strat)
        
        if features is not None:
            return features


def plot_label_distribution(y_data: list, title: str, simplify=False) -> None:
    """Plots the distribution of labels as a bar plot, showing the frequency of each label in the list.

    Args:
        y_data (list): A list of labels
        title (str): The title for the plot
    """
    plt.rcParams.update({'font.size': 22})
    sns.set_theme()
    plot_height = 3 if simplify else 8
    
    plt.figure(figsize=(16, plot_height))
    sns.countplot(y=y_data, order=np.unique(y_data))
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Label")
    plt.show()



def plot_confusion_matrix(test_labels: list, pred_labels: list, save_dir="", concatenate: bool=False) -> None:
    """Plots a confusion matrix comparing predicted labels to test labels.

    Args:
        test_labels (list): The true test labels.
        pred_labels (list): The predicted labels.
        concatenate (bool): Gather the unique labels of test and prediction labels into one list of unique labels.
    """
    plt.style.use('default')
    cm = confusion_matrix(test_labels, pred_labels)
    
    if concatenate:
        test_labels = np.unique(np.concatenate([test_labels, pred_labels]))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(test_labels)
    )
    _, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        #print(f"Confusion matrix saved to {save_path}")
    
    #plt.show()



def plot_coefficients(coefs, classes):
    plt.figure(figsize=(12, 7))
    sns.heatmap(coefs, cmap="coolwarm", annot=False, xticklabels=feature_labels, yticklabels=classes)
    plt.xlabel("Keypoint Coordinates")
    plt.ylabel("Class")
    plt.title("Logistic Regression Coefficients")
    plt.xticks(rotation=45, ha="right")
    plt.show()
    

def plot_umap(labels, cm, data, text_embeddings, player, video_number, neighbors, type):
    unique_labels = list(set(labels))
    cmap = cm.get_cmap("tab20", len(unique_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

    markers = ['o', 's', 'D', 'P', '*', 'X', '^', 'v', '<', '>', 'p', 'h']
    marker_dict = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(data[mask, 0], data[mask, 1], 
                    s=20, label=label, color=color_dict[label], marker=marker_dict[label], 
                    edgecolors='black', linewidth=0.5, alpha=0.8)
        
    #plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], 
    #            s=2, c='black', label="Text Embeddings", marker='o')

    # Add captions to text embeddings
    # for i, caption in enumerate(text_labels):
    #     plt.text(text_embeddings[i, 0] + 1.15, text_embeddings[i, 1], caption, 
    #              fontsize=8, color='black', ha='center', va='center', alpha=0.7)

    plt.title(f"UMAP Projection of {type} for {player} player in video_{video_number}. Neighbors = {neighbors}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(markerscale=1, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)
    plt.tight_layout()

    #plt.savefig(f"figures/umaps/cleaned/LALALALAred_umap_video_{video_number}_player{player}_neighbors{neighbors}.png", dpi=300)

    plt.show()
    
    
def plot_probabilities(probs, num_samples):
    sample_probs = [list(map(float, entry['probabilities'].values())) for entry in probs]
    highest_probs = [max(p) for p in sample_probs]
    lowest_probs = [min(p) for p in sample_probs]
    diff_probs = [high - low for high, low in zip(highest_probs, lowest_probs)]

    x = np.arange(num_samples)

    plt.rcParams.update({'font.size': 22})
    sns.set_theme()
    
    plt.figure(figsize=(16, 5))
    plt.bar(x, lowest_probs, label='Lowest Probability', color='orange')
    plt.bar(x, diff_probs, bottom=lowest_probs, label='Difference to Highest', color='skyblue')

    plt.xlabel("Test samples")
    plt.ylabel("Probability")
    plt.title("Highest vs. Lowest Class Probabilities per Prediction")
    plt.xticks(ticks=np.arange(0, len(x), step=max(1, len(x)//20)))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    #plt.show()
