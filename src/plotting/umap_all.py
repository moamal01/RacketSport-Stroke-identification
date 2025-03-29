import json
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

player = "right"
video_number = 2
neighbors = 15
key_of_interest = "question_based"

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)
    
with open(f"data/events/events_markup2.json", "r") as file:
    data2 = json.load(file)
    
with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

image_embeddings = []
image_labels = []
text_embeddings = []
text_labels = []

for frame, value in stroke_frames_1.items():
    #if (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
    
        if value == "other" or value == "otherotherother":
            continue

        if player in value: # and labels.count(value) < 10:
            if os.path.exists(f"embeddings/video_1/{frame}/0/{player}.npy"):
                image_embeddings.append(np.load(f"embeddings/video_1/{frame}/0/{player}.npy"))  
                image_labels.append(value.replace(" ", "_"))

for frame, value in stroke_frames_2.items():
    #if (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
    
        if value == "other" or value == "otherotherother":
            continue

        value1 = value.split(" ")[0]
        value2 = value1.split("_")[2]
        if player in value1: # and labels.count(value) < 10:
            if os.path.exists(f"embeddings/video_2/{frame}/0/{player}.npy"):
                image_embeddings.append(np.load(f"embeddings/video_2/{frame}/0/{player}.npy"))  
                image_labels.append(value1)

for key, grouping in clip_captions.items():
    for caption in grouping:
        #if key == key_of_interest:
            file_path = f"embeddings/text/{key}/{caption}//embedding.npy"
            text_embeddings.append(np.load(file_path))
            text_labels.append(caption)

# Stack embeddings into a single arrays
image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)

# Create Umap
umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(image_embeddings)
text_embeddings_2d = umap_model.transform(text_embeddings)

unique_labels = list(set(image_labels))
cmap = cm.get_cmap("tab10", len(unique_labels))
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

# Create scatter plot with colors and different markers
markers = ['o', 's', 'D', 'P', '*', 'X', '^', 'v', '<', '>', 'p', 'h']
marker_dict = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

# Create scatter plot
plt.figure(figsize=(12, 7))
for label in unique_labels:
    mask = np.array(image_labels) == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=20, label=label, color=color_dict[label], marker=marker_dict[label], 
                edgecolors='black', linewidth=0.5, alpha=0.8)
    
plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
            s=2, c='black', label="Text Embeddings", marker='o')

# Add captions to text embeddings
# for i, caption in enumerate(text_labels):
#     plt.text(text_embeddings_2d[i, 0] + 1.15, text_embeddings_2d[i, 1], caption, 
#              fontsize=8, color='black', ha='center', va='center', alpha=0.7)

plt.title(f"UMAP Projection of Image Embeddings for {player} player in both videos. Neighbors = {neighbors}.")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(f"figures/umaps/cleaned/captions_{player}_neighbors{neighbors}.png", dpi=300)

plt.show()
