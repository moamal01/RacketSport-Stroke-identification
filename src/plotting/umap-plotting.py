import json
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

player = "right"
video_number = 2
neighbors = 5

with open(f"data/events/events_markup{video_number}.json", "r") as file:
    data = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames.items():
    if not (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:
    
        if value == "other" or value == "otherotherother":
            continue

        value1 = value.split(" ")[0]
        value2 = value1.split("_")[2]
        if player in value1: # and labels.count(value) < 10:
            if os.path.exists(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"):
                embeddings.append(np.load(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"))  
                labels.append(value1.split('_', 1)[1])

print(len(labels))
print(len(embeddings))

embeddings = np.vstack(embeddings)  # Stack embeddings into a single array

umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

unique_labels = list(set(labels))
cmap = cm.get_cmap("tab20", len(unique_labels))  # Get a colormap
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}  # Assign colors

# Create scatter plot with colors and different markers
plt.figure(figsize=(10, 6))
for label in unique_labels:
    mask = np.array(labels) == label  # Convert mask to NumPy array
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=4, label=label, color=color_dict[label])

plt.title(f"UMAP Projection of Image Embeddings for {player} player in video_{video_number}. Neighbors = {neighbors}")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=1, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)
plt.tight_layout()

plt.savefig(f"figures/umaps/for_report/red_umap_video_{video_number}_player{player}_neighbors{neighbors}.png", dpi=300)

plt.show()
