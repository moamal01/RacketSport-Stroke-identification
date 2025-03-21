import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

player = "left"
video_number = 2

with open(f"data/events/events_markup{video_number}.json", "r") as file:
    data = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

embeddings = []
labels = []

for frame, value in stroke_frames.items():
    if value == "other" or value == "otherotherother":
        continue
    
    value1 = value.split(" ")[0]
    value2 = value1.split("_")[2]
    if player in value1: # and labels.count(value) < 10:
        if os.path.exists(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"):
            embeddings.append(np.load(f"imbeddings/video_{video_number}/{frame}/0/{player}.npy"))  
            labels.append(value1)

print(len(labels))
print(len(embeddings))

embeddings = np.vstack(embeddings)  # Stack embeddings into a single array

tsne_model = TSNE(n_components=2, perplexity=5, learning_rate=200, random_state=42)
embeddings_2d = tsne_model.fit_transform(embeddings)

unique_labels = list(set(labels))
cmap = cm.get_cmap("tab10", len(unique_labels))  # Get a colormap
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}  # Assign colors

# Create scatter plot with colors
plt.figure(figsize=(10, 6))
for label in unique_labels:
    mask = np.array(labels) == label  # Convert mask to NumPy array
    marker = 'v' if 'forehand' in label else 'x'
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=4, label=label, color=color_dict[label], marker=marker)

plt.title(f"t-sne Projection of Image Embeddings for {player} player")
plt.xlabel("t-sne Dimension 1")
plt.ylabel("t-sne Dimension 2")

# Move legend outside the plot
plt.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to fit everything
plt.show()
