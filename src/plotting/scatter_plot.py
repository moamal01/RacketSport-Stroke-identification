import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Load the keypoints
df = pd.read_csv("keypoints4_saved.csv")
keypoints = df["Player_1 keypoints"].apply(ast.literal_eval)

# Define colors for each joint
num_joints = len(keypoints.iloc[0])  # Assuming all keypoint lists are the same length
colors = plt.cm.get_cmap('tab10', num_joints)  # Get a colormap with unique colors

plt.figure(figsize=(8, 6))

# Plot each joint with a unique color
for i in range(num_joints):
    x_coords = [keypoint_list[i][0] for keypoint_list in keypoints]
    y_coords = [keypoint_list[i][1] for keypoint_list in keypoints]
    plt.scatter(x_coords, y_coords, color=colors(i), label=f'Joint {i}')

plt.xlabel('X Coordinates', fontsize=14)
plt.ylabel('Y Coordinates', fontsize=14)
plt.title('Scatterplot of Player_1 Keypoints', fontsize=16)
plt.legend()
plt.show()
