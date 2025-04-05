import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from utility_functions import get_embeddings_and_labels, get_keypoints_and_labels, plot_umap
plt.rcParams.update({'font.size': 22})
sns.set_theme()

# Get Data
player = "right"
video_numbers = [1, 2]
neighbors = 15

key_of_interest = "question_based"
    
with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

all_embeddings = []
all_embedding_labels = []
all_keypoints = []
all_keypoints_labels = []
text_embeddings = []
text_labels = []

for video_number in video_numbers:
    embeddings, labels = get_embeddings_and_labels(video_number, player_to_get=player, simplify=True)
    all_embeddings.extend(embeddings)
    all_embedding_labels.extend(labels)
    
for video_number in video_numbers:
    keypoints, labels = get_keypoints_and_labels(video_number, player_to_get=player, simplify=True)
    all_keypoints.extend(keypoints)
    all_keypoints_labels.extend(labels)

for key, grouping in clip_captions.items():
    for caption in grouping:
        #if key == key_of_interest:
            file_path = f"embeddings/text/{key}/{caption}/embedding.npy"
            text_embeddings.append(np.load(file_path))
            text_labels.append(caption)

# UMAP embeddings
umap_model_embeddings = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
all_embeddings = np.vstack(all_embeddings)
text_embeddings = np.vstack(text_embeddings)

embeddings_2d = umap_model_embeddings.fit_transform(all_embeddings)
text_embeddings_embeddings_2d = umap_model_embeddings.transform(text_embeddings)

# UMAP keypoints
umap_model_keypoints = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
all_keypoints = np.vstack(all_keypoints)

keypoints_2d = umap_model_keypoints.fit_transform(all_keypoints)
text_embeddings_keypoints_2d = umap_model_keypoints.transform(text_embeddings)

# Plot embeddings UMAP
plot_umap(all_embedding_labels, cm, embeddings_2d, text_embeddings_embeddings_2d, player, video_numbers[0], neighbors)
plot_umap(all_keypoints_labels, cm, keypoints_2d, text_embeddings_embeddings_2d, player, video_numbers[0], neighbors)
