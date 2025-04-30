import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rcParams.update({'font.size': 22})
sns.set_theme()

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
    mirrored = ""
    
    timestamps = get_timestamps(video_number)
    
    for frame, value in timestamps.items():
        #if not (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0]
        
        if player != player_to_get and player_to_get != "both":
            continue
        
        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        file_path = f"embeddings/video_{video_number}{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(file_path):
            features.append(np.load(file_path))
            labels.append(label)
    
    return features, labels


def get_keypoints_and_labels(video_number, mirror=False, simplify=False, player_to_get="both") -> list | list:
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
        
        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0].capitalize()
        
        if player != player_to_get and player_to_get != "both":
            continue
        
        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        event_row = df.loc[df['Event frame'] == int(frame)]
        keypoint = ast.literal_eval(event_row.iloc[0][f"{player} distances"])
        keypoint = np.array(keypoint)[:, :2]
        keypoint_list.append(keypoint.flatten())
        labels.append(label)
            
    return keypoint_list, labels

def get_keypoints_and_labels_time(video_number, mirror=False, simplify=False, player_to_get="both"):
    keypoint_list = []
    labels = []

    timestamps = get_timestamps(video_number)
    keypoints_table = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)

    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue

        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0].capitalize()

        if player != player_to_get and player_to_get != "both":
            continue

        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        sequence_frames = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        event_keypoints = None

        for sequence_frame in sequence_frames:
            event_row = df[(df['Event frame'] == int(frame)) & (df['Sequence frame'] == sequence_frame)]
            if event_row.empty:
                continue  # skip missing data

            sequence_keypoints = ast.literal_eval(event_row.iloc[0][f"{player} distances"])
            sequence_keypoints = np.array(sequence_keypoints)[:, :2].flatten()

            if event_keypoints is None:
                event_keypoints = sequence_keypoints
            else:
                event_keypoints = np.concatenate((event_keypoints, sequence_keypoints))

        if event_keypoints is not None:
            keypoint_list.append(event_keypoints)
            labels.append(label)

    return keypoint_list, labels

def get_keypoints_and_labels_time_and_midpoints(video_number, mirror=False, simplify=False, player_to_get="both"):
    keypoint_list = []
    labels = []

    timestamps = get_timestamps(video_number)
    keypoints_table = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)

    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue

        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0].capitalize()

        if player != player_to_get and player_to_get != "both":
            continue

        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        sequence_frames = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        event_keypoints = None

        for sequence_frame in sequence_frames:
            event_row = df[(df['Event frame'] == int(frame)) & (df['Sequence frame'] == sequence_frame)]
            if event_row.empty:
                continue  # skip missing data

            sequence_keypoints = ast.literal_eval(event_row.iloc[0][f"{player} distances"])
            sequence_keypoints = np.array(sequence_keypoints)[:, :2].flatten()
            sequence_midpoint = ast.literal_eval(event_row.iloc[0][f"{player} player midpoint"]) # No need for flattening
            
            key_and_mid_points = np.concatenate((sequence_keypoints, sequence_midpoint))

            if event_keypoints is None:
                event_keypoints = key_and_mid_points
            else:
                event_keypoints = np.concatenate((event_keypoints, key_and_mid_points))

        if event_keypoints is not None:
            keypoint_list.append(event_keypoints)
            labels.append(label)

    return keypoint_list, labels


def get_keypoints_and_labels_raw(video_number, mirror=False, simplify=False, player_to_get="both") -> list | list:
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
        
        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0]
        
        if player != player_to_get and player_to_get != "both":
            continue
        
        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        event_row = df.loc[df['Event frame'] == int(frame)]
        keypoint = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
        keypoint = np.array(keypoint)[:, :2]
        keypoint_list.append(keypoint.flatten())
        labels.append(label)
            
    return keypoint_list, labels


def get_concat_and_labels(video_number, mirror=False, simplify=False, player_to_get="both") -> list | list:
    """

    Args:
        timestamps (_type_): The dictionary containing stroke timestamps and labels
        mirror (bool, optional): Set to true to get mirrored data. Defaults to False.

    Returns:
        features (list): A list of numpy arrays, where each array is the embedding corresponding to a label.
        labels (list): A list of strings, where each string is the label corresponding to the respective embedding.
    """
    concat_list = []
    labels = []
    mirrored = ""
    
    timestamps = get_timestamps(video_number)
    keypoints_table = f"data/video_{video_number}/midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)
    
    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0].capitalize()
        
        if player != player_to_get and player_to_get != "both":
            continue
        
        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        file_path = f"embeddings/video_{video_number}{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(file_path):
            embedding = np.load(file_path)
            event_row = df.loc[df['Event frame'] == int(frame)]
            keypoints = ast.literal_eval(event_row.iloc[0][f"{player} distances"])
            keypoints = np.array(keypoints)[:, :2]
            concat_list.append(np.concatenate([embedding.squeeze(), keypoints.flatten()]))
            labels.append(label)
            
    return concat_list, labels


def get_concat_and_labels_raw(video_number, mirror=False, simplify=False, player_to_get="both") -> list | list:
    """

    Args:
        timestamps (_type_): The dictionary containing stroke timestamps and labels
        mirror (bool, optional): Set to true to get mirrored data. Defaults to False.

    Returns:
        features (list): A list of numpy arrays, where each array is the embedding corresponding to a label.
        labels (list): A list of strings, where each string is the label corresponding to the respective embedding.
    """
    concat_list = []
    labels = []
    mirrored = ""
    
    timestamps = get_timestamps(video_number)
    keypoints_table = f"midpoints_video{video_number}.csv"
    df = pd.read_csv(keypoints_table)
    
    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0]
        
        if player != player_to_get and player_to_get != "both":
            continue
        
        if simplify:
            if "serve" in label:
                label = f"{player}_{label_parts[2]}"
            else:
                label = f"{player}_{label_parts[1]}"

        file_path = f"embeddings/video_{video_number}{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(file_path):
            embedding = np.load(file_path)
            event_row = df.loc[df['Event frame'] == int(frame)]
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            keypoints = np.array(keypoints)[:, :2]
            concat_list.append(np.concatenate([embedding.squeeze(), keypoints.flatten()]))
            labels.append(label)
            
    return concat_list, labels


def plot_label_distribution(y_data: list, title: str, simplify=False) -> None:
    """Plots the distribution of labels as a bar plot, showing the frequency of each label in the list.

    Args:
        y_data (list): A list of labels
        title (str): The title for the plot
    """
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