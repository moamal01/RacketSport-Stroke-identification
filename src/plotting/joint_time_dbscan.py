import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the dataset
file_path = "keypoints4_saved.csv"
df = pd.read_csv(file_path)

# Ensure there's a Frame column (or create one if missing)
if "Event frame" not in df.columns:
    df["Event frame"] = range(len(df))

# Convert "Player_1 keypoints" column from string to list of coordinates
df["Player_1 keypoints"] = df["Player_1 keypoints"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# Extract the 17th joint (index 16) from each frame
df["Keypoint_1"] = df["Player_1 keypoints"].apply(lambda x: x[0] if x else None)
df = df.dropna(subset=["Keypoint_1"])  # Remove empty keypoints

# Convert to NumPy array
keypoints_array = np.array(df["Keypoint_1"].tolist())
time_array = df["Event frame"].values.reshape(-1, 1)  # Reshape for clustering

# Apply DBSCAN clustering on the time axis
dbscan = DBSCAN(eps=20, min_samples=10)  # Adjust eps to define time-cluster separation
clusters = dbscan.fit_predict(time_array)

# Assign cluster labels back to the DataFrame
df["Cluster"] = clusters

# Plot the clustered keypoints
plt.figure(figsize=(8, 6))
plt.scatter(keypoints_array[:, 0], keypoints_array[:, 1], c=df["Event frame"], cmap='rainbow', s=5, alpha=0.6)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Clustering by Time: first Joint in Player 1 Keypoints")
plt.colorbar(label="Frame Number")
plt.show()
