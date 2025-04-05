import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from utility_functions import (get_embeddings_and_labels)
plt.rcParams.update({'font.size': 22})
sns.set_theme()

# Get Data
player = "both"
video_numbers = [1, 2]
neighbors = 15

key_of_interest = "question_based"
    
with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

all_embeddings = []
all_labels = []
text_embeddings = []
text_labels = []

for video_number in video_numbers:
    embeddings, labels = get_embeddings_and_labels(video_number, player_to_get=player, simplify=True)
    all_embeddings.extend(embeddings)
    all_labels.extend(labels)

for key, grouping in clip_captions.items():
    for caption in grouping:
        #if key == key_of_interest:
            file_path = f"embeddings/text/{key}/{caption}//embedding.npy"
            text_embeddings.append(np.load(file_path))
            text_labels.append(caption)

# UMAP
umap_model = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
all_embeddings = np.vstack(all_embeddings)
text_embeddings = np.vstack(text_embeddings)

embeddings_2d = umap_model.fit_transform(all_embeddings)
text_embeddings_2d = umap_model.transform(text_embeddings)

unique_labels = list(set(all_labels))
cmap = cm.get_cmap("tab20", len(unique_labels))
color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

# Plotting
markers = ['o', 's', 'D', 'P', '*', 'X', '^', 'v', '<', '>', 'p', 'h']
marker_dict = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

plt.figure(figsize=(10, 6))
for label in unique_labels:
    mask = np.array(all_labels) == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                s=20, label=label, color=color_dict[label], marker=marker_dict[label], 
                edgecolors='black', linewidth=0.5, alpha=0.8)
    
#plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
#            s=2, c='black', label="Text Embeddings", marker='o')

# Add captions to text embeddings
# for i, caption in enumerate(text_labels):
#     plt.text(text_embeddings_2d[i, 0] + 1.15, text_embeddings_2d[i, 1], caption, 
#              fontsize=8, color='black', ha='center', va='center', alpha=0.7)

plt.title(f"UMAP Projection of Image Embeddings for {player} player in video_{video_number}. Neighbors = {neighbors}")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=1, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)
plt.tight_layout()

plt.savefig(f"figures/umaps/cleaned/LALALALAred_umap_video_{video_number}_player{player}_neighbors{neighbors}.png", dpi=300)

plt.show()
