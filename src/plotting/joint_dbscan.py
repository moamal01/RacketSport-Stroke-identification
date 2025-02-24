import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the dataset
file_path = "keypoints4_saved.csv"
df = pd.read_csv(file_path)

# Convert "Player_1 keypoints" column from string to list of coordinates
df["Player_1 keypoints"] = df["Player_1 keypoints"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# Extract only the first keypoint from each frame
keypoints_list = [sublist[16] for sublist in df["Player_1 keypoints"] if sublist]
keypoints_array = np.array(keypoints_list)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=20, min_samples=10)
clusters = dbscan.fit_predict(keypoints_array)

# Plot the clustered keypoints
plt.figure(figsize=(8, 6))
plt.scatter(keypoints_array[:, 0], keypoints_array[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.6)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("DBSCAN Clustering of 17th Joint in Player 1 Keypoints")
plt.colorbar(label="Cluster Label")
plt.show()