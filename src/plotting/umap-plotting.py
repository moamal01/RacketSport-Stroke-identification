import json
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

player = "left"

with open("data/events/events_markup2.json", "r") as file:
    data = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames.items():
    value = value.split(" ")[0]
    if player in value:
        if os.path.exists(f"imbeddings/video_2/{frame}/0/{player}.npy"):
            embeddings.append(np.load(f"imbeddings/video_2/{frame}/0/{player}.npy"))  
            labels.append(value)

print(len(labels))
print(len(embeddings))

embeddings = np.vstack(embeddings)  # Stack embeddings into a single array

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

unique_labels = list(set(labels))
cmap = cm.get_cmap("tab10", len(unique_labels))  # Get a colormap
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}  # Assign colors

# Create scatter plot with colors
plt.figure(figsize=(10, 6))
for label in unique_labels:
    mask = np.array(labels) == label  # Convert mask to NumPy array
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=5, alpha=0.7, label=label, color=color_dict[label])

plt.title(f"UMAP Projection of Image Embeddings for {player} player")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# Move legend outside the plot
plt.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to fit everything
plt.show()
