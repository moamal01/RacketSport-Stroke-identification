import sys
import os
import joblib
import numpy as np
sys.path.append(os.path.abspath('../'))
from utility_functions import get_feature
import cv2
from tqdm import tqdm
from collections import deque

debug = False

if debug:
    prefix = ""
else:
    prefix = "../"

# Models
model = joblib.load(prefix + 'results/replace/20250610_104540/39_keypoints_midpoints_table_ball_racket_time_fullscores/logistic_regression_model.joblib')
label_encoder = joblib.load(prefix + 'results/replace/20250610_104540/39_keypoints_midpoints_table_ball_racket_time_fullscores/label_encoder.joblib')
class_names = label_encoder.classes_

# Video properties
cap = cv2.VideoCapture(prefix + "videos/testvidf.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(prefix + "videos/full_model-cheat_ball.mp4", fourcc, fps, (width, height))

# Variables
last_event = ""
last_event_frame = 0
last_stroke = ""
pause = True
points = 0
frame_range = 90
frame_gap = 2
sma_window_size = 40

prob_history = deque(maxlen=sma_window_size)
    
for frame_number in tqdm(range(frame_count - (frame_range * frame_gap)), desc="Processing Frames"):
    ret, frame = cap.read()

    x = get_feature(
        video_number=4,
        frames=[frame_number],
        sequence_range=frame_range,
        sequence_gap=frame_gap,
        raw=False,
        add_keypoints=True,
        add_midpoints=True,
        add_rackets=True,
        add_table=True,
        add_ball=True,
        cheat_ball=True,
        add_scores=True,
        add_k_score=True,
        add_embeddings=False,
        missing_strat="replace",
        mirror=False,
        simplify=True,
        long_edition=True
    )

    if x is not None:
        x = x.reshape(1, -1)
        probabilities = model.predict_proba(x)[0]
        prob_history.append(probabilities)
        smoothed_probabilities = np.mean(prob_history, axis=0)
        best_i = np.argmax(smoothed_probabilities)

        new_event = label_encoder.inverse_transform([best_i])[0]
        last_event_frame = frame_number
        
        # --- EVENT HANDLING ---
        if "pause" in new_event:
            pause = True
            point_pending = True
        elif "serve" in new_event and point_pending:
            points += 1
            point_pending = False
            pause = False
        elif "pause" not in new_event and "play" not in new_event: # Event which is not a phase is detected.
            pause = False
            point_pending = False


        # Track last stroke (for display only)
        if "play" not in new_event and "pause" not in new_event:
            last_stroke = new_event

        last_event = new_event  # Update last_event after all logic                

    # Show last predicted stroke (top right)
    cv2.putText(
        frame,
        f"Last stroke: {last_stroke}",
        (width - 420, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    # Show phase label centered
    phase_text = f"Phase: {'Pause' if pause else 'Play'}"
    text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(
        frame,
        phase_text,
        ((width - text_size[0]) // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(
        frame,
        f"Points: {points}",
        ((width - text_size[0]) // 2, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Show probabilities vertically (left side)
    class_labels = label_encoder.classes_
    for i, label in enumerate(class_labels):
        prob = probabilities[i]
        text = f"{label}: {prob:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 50 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )
    
    out.write(frame)
    
cap.release()
out.release()