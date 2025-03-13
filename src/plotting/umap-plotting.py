import json
import numpy as np
import umap
import matplotlib.pyplot as plt

with open("data/events/events_markup1.json", "r") as file:
    data = json.load(file)

excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = [k for k, v in data.items() if v not in excluded_values]

embeddings = []
valid_frames = []

for frame in stroke_frames:
    embeddings.append(np.load(f"imbeddings/video_1/{frame}/0/image_embeddings.npy"))

embeddings = np.vstack(embeddings) 

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)
plt.title("UMAP Projection of Image Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()