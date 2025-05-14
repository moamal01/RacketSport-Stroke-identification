import json
import cv2
from tqdm import tqdm

# Video properties
cap = cv2.VideoCapture("../videos/game_3.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter("Result.mp4", fourcc, fps, (width, height))

with open(f"../raw_keypoints.json", "r") as file:
        data = json.load(file)

# Prepare mappings    
frame_to_probs = {entry["frame"]: entry["probabilities"] for entry in data}
frame_to_stroke = {entry["frame"]: entry["predicted_class"] for entry in data}

last_stroke = ""
last_stroke_frame = 0
pause = True
points = 0
all_classes = ["left_backhand", "left_forehand", "left_serve", "right_backhand", "right_forehand", "right_serve"]
stroke_probabilities = {cls: 0.0 for cls in all_classes}

for frame_number in tqdm(range(frame_count), desc="Processing Frames", ncols=100, unit="frame"):
    ret, frame = cap.read()
    
    if frame_number - last_stroke_frame > 150:
        pause = True
    
    if frame_number in frame_to_stroke:
        new_stroke = frame_to_stroke[frame_number]
        last_stroke = new_stroke
        last_stroke_frame = frame_number
        stroke_probabilities = frame_to_probs[frame_number]
        
        if pause and "serve" in last_stroke:
            points += 1
            
        pause = False

    # Show last predicted stroke (top right)
    cv2.putText(
        frame,
        f"Last stroke: {last_stroke}",
        (width - 420, 35),
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
    for i, (stroke_type, prob) in enumerate(stroke_probabilities.items()):
        text = f"{stroke_type}: {prob:.2f}"
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
