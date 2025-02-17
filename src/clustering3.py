import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Sample DataFrame (Replace with actual DataFrame)
df = pd.read_csv("keypoints3_saved.csv")

# Convert string representation to actual list
keypoints = df["Player_1 keypoints"].apply(ast.literal_eval)

# Extract X and Y coordinates
X_coords = np.array([[kp[i][0] for i in range(len(kp))] for kp in keypoints])
Y_coords = np.array([[kp[i][1] for i in range(len(kp))] for kp in keypoints])

# Flatten into a feature vector per player
features = np.hstack((X_coords, Y_coords))

# Ensure perplexity < number of samples
perplexity_value = min(5, len(features) - 1)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], marker='o', color='blue', edgecolors='k')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Player Keypoints")
plt.show()
