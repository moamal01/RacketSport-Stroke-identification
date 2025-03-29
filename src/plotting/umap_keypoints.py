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
key_of_interest = "question_based"

file_path1 = "midpoints.csv"
file_path2 = "midpoints_video2.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)

with open(f"data/events/events_markup2.json", "r") as file:
    data2 = json.load(file)
    
with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

features = []
labels = []
text_embeddings = []
text_labels = []

for frame, value in stroke_frames_1.items():
    if value in {"other", "otherotherother"}:
        continue

    event_row = df1.loc[df1['Event frame'] == int(frame)]

    if player in value:
        path = f"embeddings/video_1/{frame}/0/{player}.npy"
        if os.path.exists(path):
            embedding = np.load(path)
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            
            keypoints = np.array(keypoints)[:, :2]  # Remove confidence scores
            features.append(keypoints.flatten())
            labels.append(value.replace(" ", "_"))

for frame, value in stroke_frames_2.items():
    if value in {"other", "otherotherother"}:
        continue

    event_row = df2.loc[df2['Event frame'] == int(frame)]
    
    value1 = value.split(" ")[0]

    if player in value1:
        path = f"embeddings/video_2/{frame}/0/{player}.npy"
        if os.path.exists(path):
            embedding = np.load(path)
            keypoints = ast.literal_eval(event_row.iloc[0][f"Keypoints {player}"])
            
            keypoints = np.array(keypoints)[:, :2]  # Remove confidence scores
            features.append(keypoints.flatten())
            labels.append(value1)
            
for key, grouping in clip_captions.items():
    for caption in grouping:
        file_path = f"embeddings/text/{key}/{caption}//embedding.npy"
        text_embeddings.append(np.load(file_path))
        text_labels.append(caption)

# Convert data into NumPy arrays
features = np.vstack(features)
text_embeddings = np.vstack(text_embeddings)

# UMAP Dimensionality Reduction
umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(features)
text_embeddings_2d = umap_model.transform(text_embeddings)

# Unique labels and colors
unique_labels = sorted(set(labels))  # Sorted for consistent color/marker assignment
cmap = cm.get_cmap("tab20", len(unique_labels))  # More colors than tab10
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

# Unique markers
markers = ['o', 's', 'D', 'P', '*', 'X', '^', 'v', '<', '>', 'p', 'h']
marker_dict = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

# Create scatter plot
plt.figure(figsize=(12, 7))
for label in unique_labels:
    mask = np.array(labels) == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=20, label=label, color=color_dict[label], marker=marker_dict[label], 
                edgecolors='black', linewidth=0.5, alpha=0.8)
    
plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
          s=4, c='black', label="Text Embeddings", marker='o')

# Add captions to text embeddings
# for i, caption in enumerate(text_labels):
#     offset = 0.02 * len(caption)
#     plt.text(text_embeddings_2d[i, 0] + offset, text_embeddings_2d[i, 1], caption, 
#              fontsize=8, color='black', ha='center', va='center', alpha=0.7)

plt.title(f"UMAP Projection of keypoints for {player} player\nNeighbors = {neighbors}")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the figure
os.makedirs("figures/umaps/keypoints_only", exist_ok=True)
plt.savefig(f"figures/umaps/keypoints_only/caption_{player}_neighbors{neighbors}.png", dpi=300)
plt.show()
