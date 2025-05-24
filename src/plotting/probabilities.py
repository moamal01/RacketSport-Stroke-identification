import sys
import os
import joblib
import numpy as np
sys.path.append(os.path.abspath('../..'))
frames = 5369
from utility_functions import get_feature
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 22})
sns.set_theme()

# OBS: Needs to run through testvid again

model = joblib.load('../../results/default/20250522_002633/14_norm_keypoints_midpoints_table_time_ball/logistic_regression_model.joblib')
label_encoder = joblib.load('../../results/default/20250522_002633/14_norm_keypoints_midpoints_table_time_ball/label_encoder.joblib')

start_frame = 5
end_frame = 1020
frames_kept = []

left_serve = 926

probabilities = []

for i in tqdm(range(start_frame, end_frame), desc="Processing Frames"):
    x = get_feature(4, [i], 5, 1, False, True, True, False, True,
                    True, False, False, "fall_back", False, True, True)
    if x is None:
        continue

    frames_kept.append(i)
    x = x.reshape(1, -1)
    prob = model.predict_proba(x)[0]
    probabilities.append(prob)

probabilities = np.vstack(probabilities)

# Plot
plt.figure(figsize=(10, 5))
for cls_idx, cls_name in enumerate(label_encoder.classes_):
    plt.plot(frames_kept, probabilities[:, cls_idx], label=cls_name)

# Add serve timestamp
plt.axvline(x=left_serve, color='red', linestyle='--', linewidth=2, label='Left Serve')
plt.text(left_serve + 5, 0.9, 'Left Serve', color='red')

plt.xlabel("Frame index")
plt.ylabel("Predicted probability")
plt.title("Per-frame stroke probabilities")
plt.legend()
plt.tight_layout()
plt.show()