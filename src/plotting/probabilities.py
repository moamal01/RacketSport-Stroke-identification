import sys
import os
import joblib
import numpy as np
sys.path.append(os.path.abspath('../..'))
frames = 5369
from utility_functions import get_feature
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 22})
sns.set_theme()

model = joblib.load('results/default/20250521_144847/14_norm_keypoints_midpoints_table_time/logistic_regression_model.joblib')
label_encoder = joblib.load('results/default/20250521_144847/14_norm_keypoints_midpoints_table_time/label_encoder.joblib')

start_frame = 5
end_frame = 785
frames_kept = []

probabilities = []

for i in tqdm(range(start_frame, end_frame), desc="Processing Frames"):
    x = get_feature(4, [i], 5, 1, False, True, True, False, True,
                    False, False, False, False, True, True)
    if x is None:
        continue

    frames_kept.append(i)
    x = x.reshape(1, -1)
    prob = model.predict_proba(x)[0]
    probabilities.append(prob)

probabilities = np.vstack(probabilities) # (n_frames_kept, n_classes)

# Plot
plt.figure(figsize=(10, 5))
for cls_idx, cls_name in enumerate(label_encoder.classes_):
    plt.plot(frames_kept, probabilities[:, cls_idx], label=cls_name)

plt.xlabel("Frame index")
plt.ylabel("Predicted probability")
plt.title("Per-frame stroke probabilities")
plt.legend()
plt.tight_layout()
plt.show()