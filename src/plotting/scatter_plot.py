import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Plot style settings
plt.rcParams.update({'font.size': 22})
sns.set_theme()

# Load keypoints
df = pd.read_csv("../../data/video_1/keypoints_video1.csv")
keypoints = []

for idx, row in df.iterrows():
    if not idx % 64 == 0:
        continue

    keypoints_row = ast.literal_eval(row["Keypoints"])
    keypoints.append(keypoints_row[0])

# Joint names and color mapping
joint_list = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

num_joints = len(joint_list)
colors = plt.cm.get_cmap('tab10', num_joints)

# Prepare plot
plt.figure(figsize=(8, 10))

# Plot each joint over all frames
for i in tqdm(range(num_joints), desc="Plotting joints", unit="joint"):
    x_coords = []
    y_coords = []
    for keypoint_list in keypoints:
        keypoint = keypoint_list[i]
        x_coords.append(keypoint[0])
        y_coords.append(keypoint[1])

    plt.scatter(x_coords, y_coords, color=colors(i), label=joint_list[i], s=3)

# Axis labels
plt.xlabel('X Coordinates', fontsize=14)
plt.ylabel('Y Coordinates', fontsize=14)

# Move x-axis label to the top
ax = plt.gca()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.yaxis.tick_left()

# Add title below the plot
plt.suptitle('Scatterplot of Keypoints with Highest Scores per Frame', fontsize=16, y=1.0)

# Invert y-axis to match image coordinate system
plt.gca().invert_yaxis()

# Remove duplicate labels in legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

# Save and show plot
plt.savefig("keypoint_scatter.png", dpi=300, bbox_inches='tight')
plt.show()
