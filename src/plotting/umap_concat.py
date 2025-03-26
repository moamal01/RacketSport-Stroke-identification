import json
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import ast

video_number = 2
neighbors = 15

player = "right"

file_path1 = "midpoints.csv" # Make a cleaned midpoints file!
file_path2 = "midpoints_video2.csv" # Make a cleaned midpoints file!

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)
    
with open(f"data/events/events_markup2.json", "r") as file:
    data2 = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

features = []
labels = []

for frame, value in stroke_frames_1.items():
    if value == "other" or value == "otherotherother":
        continue
    
    event_row = df1.loc[df1['Event frame'] == int(frame)]

    if player in value:
        if os.path.exists(f"embeddings/video_1/{frame}/0/{player}.npy"):
            embedding = np.load(f"embeddings/video_1/{frame}/0/{player}.npy")
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            
            # Remove confidence scores (keep only x and y)
            keypoints = np.array(keypoints)
            keypoints_xy = keypoints[:, :2]
            
            features.append(np.concatenate([embedding.squeeze(), keypoints_xy.flatten()]))  
            labels.append(value.replace(" ", "_"))

for frame, value in stroke_frames_2.items():
    if value == "other" or value == "otherotherother":
        continue
    
    event_row = df2.loc[df2['Event frame'] == int(frame)]
    
    value1 = value.split(" ")[0]
    value2 = value1.split("_")[2]
    
    if player in value1:
        if os.path.exists(f"embeddings/video_2/{frame}/0/{player}.npy"):
            embedding = np.load(f"embeddings/video_2/{frame}/0/{player}.npy")
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            
            # Remove confidence scores (keep only x and y)
            keypoints = np.array(keypoints)
            keypoints_xy = keypoints[:, :2]
            
            features.append(np.concatenate([embedding.squeeze(), keypoints_xy.flatten()]))  
            labels.append(value1)
        
# Stack embeddings into a single arrays
features = np.vstack(features)

# Create Umap
umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(features)

unique_labels = list(set(labels))
cmap = cm.get_cmap("tab10", len(unique_labels))
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

# Create scatter plot with colors and different markers
plt.figure(figsize=(10, 6))
for label in unique_labels:
    mask = np.array(labels) == label  # Convert mask to NumPy array
    marker = 'v' if 'forehand' in label else 'x'
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=4, label=label, color=color_dict[label], marker=marker)

plt.title(f"UMAP Projection of Image Embeddings for {player} player in both videos. Neighbors = {neighbors}.")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(f"figures/umaps/cleaned/concatenated_{player}_neighbors{neighbors}.png", dpi=300)

plt.show()
