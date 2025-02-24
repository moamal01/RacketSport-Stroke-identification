import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "keypoints.csv"  # Change this to your file path
df = pd.read_csv(file_path)

# Convert keypoints from string to list
df["Player_2 keypoints"] = df["Player_2 keypoints"].apply(ast.literal_eval)

# Extract first joint positions
joint_positions = np.array([kp[10] for kp in df["Player_2 keypoints"]])  # First joint (x, y) positions
print("Positions")
print(joint_positions)

# Compute Euclidean distances between consecutive frames
distances = np.linalg.norm(np.diff(joint_positions, axis=0), axis=1)
print("Distances")
print(distances)

frames = np.arange(1, len(distances) + 1)

# Plot the Euclidean distances
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(distances) + 1), distances, marker="o", linestyle="-", color="b")
plt.xlabel("Frame")
plt.ylabel("Euclidean Distance")
plt.title("Euclidean Distance of First Joint Across Frames")
plt.grid(True)
plt.xticks(frames)
plt.show()
