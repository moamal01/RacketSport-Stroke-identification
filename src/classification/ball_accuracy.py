import pandas as pd
import sys
import os
import math
import ast
import cv2
from numpy import random

sys.path.append(os.path.abspath('../..'))
from utility_functions import load_json_with_dicts

midpoints_table = "data/video_4/midpoints_video4.csv"
video_path = "videos/game_4.mp4"
df = pd.read_csv(midpoints_table)
true_balls = load_json_with_dicts(f"data/video_4/ball_markup.json")

width = 1920
height = 1080
radius = 30

def unnormalize_pred_balls(ball):
    norm_x = ball[0] * width
    norm_y = ball[1] * height
    
    return [norm_x, norm_y]

def is_it_inside(true_ball, detected_ball, radius=radius):
    dx = true_ball[0] - detected_ball[0]
    dy = true_ball[1] - detected_ball[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance <= radius

insides = 0
outsides = 0
detections = []
trues = []
scores = []
frames = []

for key, coords in true_balls.items():
    row = df[(df['Event frame'] == int(key))]
    
    if row.empty:
        continue
    
    detected_ball = ast.literal_eval(row["Ball midpoint"].values[0])
    ball_score = row["Ball score"].values[0]
    
    if ball_score < 0.9:
        continue
    
    if detected_ball == []:
        outsides += 1
        continue

    detected_ball = unnormalize_pred_balls(detected_ball)
    true_ball = [int(coords['x']), int(coords['y'])]
    
    detections.append(detected_ball)
    scores.append(ball_score)
    trues.append(true_ball)
    frames.append(int(key))
    
    if is_it_inside(true_ball, detected_ball):
        insides += 1
    else:
        outsides += 1

idx = random.randint(len(frames))
frame_no = frames[idx]  

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

cv2.circle(
    frame,
    (trues[idx][0], trues[idx][1]),
    radius=radius,
    color=(0,   0, 255),
    thickness=-1
)
cv2.circle(
    frame,
    (int(detections[idx][0]), int(detections[idx][1])),
    radius=3,
    color=(0, 255, 0),
    thickness=-1
)

cv2.imshow(f"Event Preview (Frame {frame_no})", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(insides)
print(outsides)
