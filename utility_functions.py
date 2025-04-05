import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
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


def get_keypoints_and_labels(timestamps, df, mirror=False, simplify=False) -> list | list:
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
    
    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)

        label = value.split(" ")[0]
        label_parts = label.split("_")
        player = label_parts[0]
        
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


def plot_label_distribution(y_data: list, title: str) -> None:
    """Plots the distribution of labels as a bar plot, showing the frequency of each label in the list.

    Args:
        y_data (list): A list of labels
        title (str): The title for the plot
    """
    plt.figure(figsize=(12, 7))
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