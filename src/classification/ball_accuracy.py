import pandas as pd
import sys
import os
import math
import ast
import cv2
from numpy import random

sys.path.append(os.path.abspath('../..'))
from utility_functions import load_json_with_dicts

# Paths and settings
midpoints_table = "../../data/video_4/midpoints_video4.csv"
video_path = "../../videos/game_4.mp4"
df = pd.read_csv(midpoints_table)
true_balls = load_json_with_dicts(f"../../data/video_4/ball_markup.json")

width = 1920
height = 1080
radius = 30
confidence_threshold = 0.3

# Functions
def unnormalize_pred_balls(ball):
    norm_x = ball[0] * width
    norm_y = ball[1] * height
    return [norm_x, norm_y]

def is_it_inside(true_ball, detected_ball, radius=radius):
    dx = true_ball[0] - detected_ball[0]
    dy = true_ball[1] - detected_ball[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance <= radius

# Metrics
TP = 0
FP = 0
FN = 0
detections = []
trues = []
scores = []
frames = []

# Loop over all ground-truth balls
for key, coords in true_balls.items():
    frame_no = int(key)
    row = df[df['Event frame'] == frame_no]

    true_ball = [int(coords['x']), int(coords['y'])]

    if row.empty:
        FN += 1
        continue

    detected_raw = ast.literal_eval(row["Ball midpoint"].values[0])
    ball_score = row["Ball score"].values[0]

    if detected_raw == [] or ball_score < confidence_threshold:
        FN += 1
        continue

    detected_ball = unnormalize_pred_balls(detected_raw)
    trues.append(true_ball)
    detections.append(detected_ball)
    scores.append(ball_score)
    frames.append(frame_no)

    if is_it_inside(true_ball, detected_ball):
        TP += 1
    else:
        FP += 1
        FN += 1

# Metrics summary
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("===== Evaluation Metrics =====")
print(f"True Positives:  {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision:       {precision:.3f}")
print(f"Recall:          {recall:.3f}")
print(f"F1 Score:        {f1:.3f}")

# Show a random frame with true and predicted ball
if frames:
    idx = random.randint(len(frames))
    frame_no = frames[idx]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Draw ground truth
        cv2.circle(
            frame,
            (trues[idx][0], trues[idx][1]),
            radius=radius,
            color=(0, 0, 255),  # Red
            thickness=1
        )

        # Draw predicted ball (small green dot)
        cv2.circle(
            frame,
            (int(detections[idx][0]), int(detections[idx][1])),
            radius=3,
            color=(0, 255, 0),  # Green
            thickness=-1
        )

        cv2.imshow(f"Event Preview (Frame {frame_no})", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Failed to load the frame for visualization.")
else:
    print("⚠️ No valid detections to visualize.")
