import json
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

player = "left"
video_number = 2
neighbors = 15

with open(f"data/events/events_markup1.json", "r") as file:
    data1 = json.load(file)
    
with open(f"data/events/events_markup2.json", "r") as file:
    data2 = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames_1 = {k: v for k, v in data1.items() if v not in excluded_values}
stroke_frames_2 = {k: v for k, v in data2.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames_1.items():
    #if (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
    
        if value == "other" or value == "otherotherother":
            continue

        if player in value: # and labels.count(value) < 10:
            if os.path.exists(f"embeddings/video_1/{frame}/0/{player}.npy"):
                embeddings.append(np.load(f"embeddings/video_1/{frame}/0/{player}.npy"))  
                labels.append(value.replace(" ", "_"))

for frame, value in stroke_frames_2.items():
    #if (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
    
        if value == "other" or value == "otherotherother":
            continue

        value1 = value.split(" ")[0]
        value2 = value1.split("_")[2]
        if player in value1: # and labels.count(value) < 10:
            if os.path.exists(f"embeddings/video_2/{frame}/0/{player}.npy"):
                embeddings.append(np.load(f"embeddings/video_2/{frame}/0/{player}.npy"))  
                labels.append(value1)

print(len(labels))
print(len(embeddings))

embeddings = np.vstack(embeddings)  # Stack embeddings into a single array

umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

unique_labels = list(set(labels))
cmap = cm.get_cmap("tab10", len(unique_labels))  # Get a colormap
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}  # Assign colors

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

#plt.savefig(f"figures/umaps/umap_both_videos_player{player}_neighbors{neighbors}.png", dpi=300)

plt.show()
