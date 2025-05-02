import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from utility_functions import get_embeddings_and_labels, get_features, get_concat_and_labels, plot_umap
plt.rcParams.update({'font.size': 22})
sns.set_theme()

player = "both"
video_numbers = [1, 2, 3]
neighbors = 15
mirror = False
add_mirror = False
simplify = True

key_of_interest = "question_based"
    
with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

all_embeddings = []
all_embedding_labels = []
all_keypoints = []
all_keypoints_labels = []
text_embeddings = []
text_labels = []
concatenated_featues = []
concatenated_labels = []

def process_video_data(video_numbers, get_data_function, all_data_list, all_labels_list, player, mirror=False, simplify=False):
    for video_number in video_numbers:
        data, labels = get_data_function(video_number, player_to_get=player, mirror=mirror, simplify=simplify)
        all_data_list.extend(data)
        all_labels_list.extend(labels)

process_video_data(video_numbers, get_embeddings_and_labels, all_embeddings, all_embedding_labels, player, mirror=mirror, simplify=simplify)
process_video_data(video_numbers, get_features, all_keypoints, all_keypoints_labels, player, mirror=mirror, simplify=simplify)
process_video_data(video_numbers, get_concat_and_labels, concatenated_featues, concatenated_labels, player, mirror=mirror, simplify=simplify)

if add_mirror:
    process_video_data(video_numbers, get_embeddings_and_labels, all_embeddings, all_embedding_labels, player, mirror=True, simplify=simplify)
    process_video_data(video_numbers, get_features, all_keypoints, all_keypoints_labels, player, mirror=True, simplify=simplify)
    process_video_data(video_numbers, get_concat_and_labels, concatenated_featues, concatenated_labels, player, mirror=True, simplify=simplify)


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

# UMAP concatenated
umap_model_concat = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean', random_state=42)
concatenated_featues = np.vstack(concatenated_featues)

concat_2d = umap_model_concat.fit_transform(concatenated_featues)
text_embeddings_concat_2d = umap_model_concat.transform(text_embeddings)

# Plot embeddings UMAP
plot_umap(all_embedding_labels, cm, embeddings_2d, text_embeddings_embeddings_2d, player, video_numbers[0], neighbors, "image embeddings")
plot_umap(all_keypoints_labels, cm, keypoints_2d, text_embeddings_embeddings_2d, player, video_numbers[0], neighbors, "keypoints")
plot_umap(concatenated_labels, cm, concat_2d, text_embeddings_concat_2d, player, video_numbers[0], neighbors, "image ebeddings and keypoints")