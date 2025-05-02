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
    with open(f"data/events/events_markup{video_number}.json", "r") as file:
        data = json.load(file)
        
    excluded_values = {"empty_event", "bounce", "net"}

    return {k: v for k, v in data.items() if v not in excluded_values}

def get_player_and_label(value, player_to_get, simplify, mirror=False):
    if mirror:
            value = mirror_string(value)

    label = value.split(" ")[0]
    label_parts = label.split("_")
    player = label_parts[0]

    if player != player_to_get and player_to_get != "both":
        return

    if simplify:
        if "serve" in label:
            label = f"{player}_{label_parts[2]}"
        else:
            label = f"{player}_{label_parts[1]}"
    
    return player, label


def get_embeddings_and_labels(video_number, mirror=False, simplify=False, player_to_get="both") -> list | list:
    """

    Args:
        timestamps (_type_): The dictionary containing stroke timestamps and labels
        mirror (bool, optional): Set to true to get mirrored data. Defaults to False.

    Returns:
        features (list): A list of numpy arrays, where each array is the embedding corresponding to a label.
        labels (list): A list of strings, where each string is the label corresponding to the respective embedding.
    """
    features = []
    labels = []
    timestamps = get_timestamps(video_number)

    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue

        player, label = get_player_and_label(value, player_to_get, simplify)
        embeddings = get_embeddings(video_number, frame, player, True, mirror)

        if embeddings is not None:
            features.append(embeddings)
            labels.append(label)

    return features, labels


def get_keypoints_and_labels(video_number, sequence_frames, raw=False, add_midpoints=False,
                             add_table=False, add_embeddings=False, mirror=False,
                             simplify=False, player_to_get="both") -> list | list:
    """
    Args:
        timestamps (_type_): The dictionary containing stroke timestamps and labels
        mirror (bool, optional): Set to true to get mirrored data. Defaults to False.

    Returns:
        features (list): A list of numpy arrays, where each array is the embedding corresponding to a label.
        labels (list): A list of strings, where each string is the label corresponding to the respective embedding.
    """
    keypoint_list = []
    labels = []

    timestamps = get_timestamps(video_number)
    keypoints_table = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)

    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue

        player, label = get_player_and_label(value, player_to_get, simplify, mirror)
        features = None
        
        for sequence_frame in sequence_frames:
            features = compose_features(df, frame, sequence_frame, video_number, player, features, raw, add_midpoints, add_table, add_embeddings)
            
        if features is not None:
            keypoint_list.append(features)
            labels.append(label)
            
    return keypoint_list, labels


def get_embeddings(video_number, frame, player=None, single_player=False, mirror=False):
    mirrored = "m" if mirror else ""

    file_path_of_interest = f"embeddings/video_{video_number}{mirrored}/{frame}/0/{player}.npy"
    if single_player:
        if not os.path.exists(file_path_of_interest):
            return None # Is this the right thing to do?
    else:
        file_path = f"embeddings/video_{video_number}{mirrored}/{frame}/0/left.npy"
        file_path2 = f"embeddings/video_{video_number}{mirrored}/{frame}/0/right.npy" 
        if not os.path.exists(file_path) or not os.path.exists(file_path2):
            return None
    
    return np.load(file_path_of_interest).squeeze()


def compose_features(df, frame, sequence_frame, video_number, player, features, raw=False, add_midpoints=False, add_table=False, add_embeddings=False, mirror=False): # Should have add_keypoints as well       
    event_row = df[(df['Event frame'] == int(frame)) & (df['Sequence frame'] == sequence_frame)]
    if event_row.empty:
        return
    
    column = f"Keypoints {player}" if raw else f"{player.capitalize()} distances"

    keypoints = ast.literal_eval(event_row.iloc[0][column])
    keypoints = np.array(keypoints)[:, :2].flatten()
    
    if features is None:
        features = keypoints
    else:
        features = np.concatenate((features, keypoints)) 
        
    if add_table:
        table_midpoint = ast.literal_eval(event_row.iloc[0][f"Table midpoint"])
        features = np.concatenate((features, table_midpoint))  
    
    if add_midpoints:
        player_midpoint = ast.literal_eval(event_row.iloc[0][f"{player.capitalize()} player midpoint"])
        features = np.concatenate((features, player_midpoint))

    if add_embeddings:
        embeddings = get_embeddings(video_number, frame, player)
        if embeddings is None:
            return None
        
        features = np.concatenate((features, embeddings))
    
    return features


def get_features(video_number, simplify=False, add_midpoints=False, add_table=False, add_embeddings=False, mirror= False):
    features_list = []
    labels = []

    timestamps = get_timestamps(video_number)
    data_path = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(data_path)

    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue

        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")

        # Special simplification for serve label
        if simplify:
            player = label_parts[0]
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        sequence_frames = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        features = None

        # Left player features
        for sequence_frame in sequence_frames:            
            features = compose_features(df, frame, sequence_frame, video_number, "left", features, add_midpoints, add_table=False, add_embeddings=add_embeddings, mirror=mirror)
        
        # Right player features
        for sequence_frame in sequence_frames:
            features = compose_features(df, frame, sequence_frame, video_number, "right", features, add_midpoints, add_table=add_table, add_embeddings=add_embeddings, mirror=mirror)
        
        if features is not None:
            labels.append(label)
            features_list.append(features)

    return features_list, labels


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



def plot_confusion_matrix(test_labels: list, pred_labels: list, concatenate: bool=False) -> None:
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
    plt.show()



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