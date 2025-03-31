import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

keypoints = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

feature_labels = [f"{kp}_{coord}" for kp in keypoints for coord in ['x', 'y']]

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


def get_embeddings_and_labels_special(timestamps, mirror=False) -> list | list:
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

    for frame, value in timestamps.items():
        if value == "other" or value == "otherotherother":
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"
        
        player = value.split(" ")[0]
        label = value.replace(" ", "_")
        
        file_path = f"embeddings/video_1{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(file_path):
            features.append(np.load(file_path))  
            labels.append(label)

    return features, labels

def get_embeddings_and_labels(video, timestamps, mirror=False) -> list | list:
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
    
    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"

        label = value.split(" ")[0]
        value2 = label.split("_")[0]
        value3 = label.split("_")[2]

        file_path = f"embeddings/video_{video}{mirrored}/{frame}/0/{value2}.npy"
        if os.path.exists(file_path):
            features.append(np.load(file_path))
            labels.append(label)
    
    return features, labels

def get_keypoints_and_labels_special(video, timestamps, df, mirror=False) -> list | list:
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
    mirrored = ""
    
    for frame, value in timestamps.items():
        if value == "other" or value == "otherotherother":
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"
        
        player = value.split(" ")[0]
        label = value.replace(" ", "_")
        
        path = f"embeddings/video_{video}{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(path):
            event_row = df.loc[df['Event frame'] == int(frame)]
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            keypoints = np.array(keypoints)[:, :2]
            keypoint_list.append(keypoints.flatten())
            labels.append(label)
            
    return keypoint_list, labels


def get_keypoints_and_labels(video, timestamps, df, mirror=False) -> list | list:
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
    mirrored = ""
    
    for frame, value in timestamps.items():
        if value in {"other", "otherotherother"}:
            continue
        
        if mirror:
            value = mirror_string(value)
            mirrored = "m"

        label = value.split(" ")[0]
        player = label.split("_")[0]
        value3 = label.split("_")[2]

        path = f"embeddings/video_{video}{mirrored}/{frame}/0/{player}.npy"
        if os.path.exists(path):
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

    # Rotate the x-axis labels to avoid overlap
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