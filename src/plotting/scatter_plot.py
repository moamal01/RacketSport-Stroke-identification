import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
plt.rcParams.update({'font.size': 22})
sns.set_theme()

df = pd.read_csv("keypoints_saved.csv")
keypoints = df["Keypoints"].apply(ast.literal_eval)

joint_list = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

num_joints = len(joint_list)
colors = plt.cm.get_cmap('tab10', num_joints)

plt.figure(figsize=(8, 12))

half_keypoints = keypoints[:len(keypoints)//8]

for keypoint_list in tqdm(half_keypoints, desc="Plotting joints", unit="keypoint list"):
    for i in range(num_joints):
        x_coords = [keypoint[i][0] for keypoint in keypoint_list]
        y_coords = [keypoint[i][1] for keypoint in keypoint_list]
        plt.scatter(x_coords, y_coords, color=colors(i), label=joint_list[i], s=5)


plt.xlabel('X Coordinates', fontsize=14)
plt.ylabel('Y Coordinates', fontsize=14)
plt.title('Scatterplot of Player_1 Keypoints', fontsize=16)
plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.show()
