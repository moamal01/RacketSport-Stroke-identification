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

model = joblib.load('../../results/default/20250603_145008/14_norm_keypoints_midpoints_table_time_ball/logistic_regression_model.joblib')
label_encoder = joblib.load('../../results/default/20250603_145008/14_norm_keypoints_midpoints_table_time_ball/label_encoder.joblib')

start_frame = 5
end_frame = 541
frames_kept = []

right_serve = 424
left_backhand = 512

probabilities = []

for i in tqdm(range(start_frame, end_frame), desc="Processing Frames"):
    x = get_feature(
                video_number=4,
                frames=[i],
                sequence_range=90,
                sequence_gap=2,
                raw=False,
                add_keypoints=True,
                add_midpoints=True,
                add_rackets=False,
                add_table=True,
                add_ball=True,
                add_scores=False,
                add_embeddings=False,
                missing_strat="fall_back",
                mirror=False,
                simplify=True,
                long_edition=True
            )
    
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
plt.axvline(x=right_serve, color='red', linestyle='--', linewidth=1, label='Right Serve')
plt.text(right_serve + 5, 0.9, 'Right Serve', color='red')
plt.axvline(x=left_backhand, color='red', linestyle='--', linewidth=1, label='Left Backhand')
plt.text(left_backhand + 5, 0.9, 'Left backhand', color='red')

plt.xlabel("Frame index")
plt.ylabel("Predicted probability")
plt.title("Per-frame stroke probabilities")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("stroke_probabilities_plot_nophases.png", dpi=300, bbox_inches='tight') 
plt.show()