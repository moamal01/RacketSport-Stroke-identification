import numpy as np
import pandas as pd
import ast
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("keypoints3_saved.csv")

# Parse keypoints from strings
def parse_keypoints(kp_str):
    return np.array(ast.literal_eval(kp_str)).flatten()  # Flatten 2D keypoints into 1D

df["Player_1_features"] = df["Player_1 keypoints"].apply(parse_keypoints)
#df["Player_2_features"] = df["Player_2 keypoints"].apply(parse_keypoints)

# Combine features from both players
X = np.vstack(df["Player_1_features"].values)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust cluster count
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Scatter plot of clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="viridis", alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Stroke Clustering")
plt.colorbar(label="Cluster")
plt.show()
