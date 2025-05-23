import sys
import os
import joblib
import numpy as np
sys.path.append(os.path.abspath('../'))
frames = 5369
from utility_functions import get_feature
import cv2
from tqdm import tqdm

# Models
model = joblib.load('results/default/20250522_002633/14_norm_keypoints_midpoints_table_time_ball/logistic_regression_model.joblib')
label_encoder = joblib.load('results/default/20250522_002633/14_norm_keypoints_midpoints_table_time_ball/label_encoder.joblib')
class_names = label_encoder.classes_

# Video properties
cap = cv2.VideoCapture("videos/testvidf.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Add the last frames in video 4!
frame_count = 5369 #int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter("videos/result4.mp4", fourcc, fps, (width, height))

# Variables
last_stroke = ""
last_stroke_frame = 0
pause = True
points = 0
frame_range = 5
frame_gap = 1

old_probabilities = [0, 0, 0, 0, 0, 0]
    
for frame_number in tqdm(range(frame_count - (frame_range * frame_gap)), desc="Processing Frames"):
    ret, frame = cap.read()
    
    if frame_number % 1 == 0:
        x = get_feature(4, [frame_number], frame_range, frame_gap, False, True, True, False, True, True, False, False, "fall_back", False, True, True)
        if x is not None:
            if frame_number - last_stroke_frame > 150:
                pause = True
            
            x = x.reshape(1, -1)  
            probabilities = model.predict_proba(x)[0]
            old_probabilities = probabilities

            best_pred = np.max(probabilities)
            best_i = np.argmax(probabilities)
            
            if best_pred > 0.8:
                new_stroke = label_encoder.inverse_transform([best_i])[0]
                last_stroke = new_stroke
                last_stroke_frame = frame_number
                
                if pause and "serve" in last_stroke:
                    points += 1
                    
                pause = False

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
        prob = old_probabilities[i]
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