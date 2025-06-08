import pandas as pd
import ast
import cv2
import os
import json
from tqdm import tqdm

# Load CSV file
video = 4
mirrored = False
start_at = 0
full_video = False

if mirrored:
    m = "m"
    file_path = f"../../data/video_{video}/mirrored_midpoints_video{video}.csv"
else:
    m = ""
    file_path = f"../../data/video_{video}/midpoints_video{video}.csv"

df = pd.read_csv(file_path)

video_path = f"../../videos/game_{video}.mp4"
cap = cv2.VideoCapture(video_path)

def get_player_boxes(df):
    left_player_boxes = []
    right_player_boxes = []
    frames = []
    left_scores = []
    right_scores = []
    
    with open(f"../../data/extended_events/events_markup{video}.json", "r") as keypoint_file:
        data = json.load(keypoint_file)
    
    excluded_values = {"empty_event", "bounce", "net"}
    loaded_keys = {k: v for k, v in data.items() if v not in excluded_values}
    
    
    for _, row in df.iterrows():
        event_frame = row["Event frame"]
        if event_frame < start_at:
            continue
        
        if not full_video:
            if str(event_frame) not in loaded_keys:
                continue
        
        left_bboxes = ast.literal_eval(row["Left bbox"])
        right_bboxes = ast.literal_eval(row["Right bbox"])
        left_score = row["Left score"]
        right_score = row["Right score"]
        
        
        left_player_boxes.append(left_bboxes)
        right_player_boxes.append(right_bboxes)
        frames.append(event_frame)
        left_scores.append(left_score)
        right_scores.append(right_score)
    
    return left_player_boxes, right_player_boxes, frames, left_scores, right_scores


def process_player(bbox, frame, index, player, score):
    if not bbox or score <= 0.9:
        return
    
    frame_directory = f"../../cropped/video_{video}{m}/{frames[index]}/0"
    image_path = f"{frame_directory}/{player}.png"
    
    if os.path.exists(image_path):
        return
    
    os.makedirs(frame_directory, exist_ok=True)  # Safe even if it exists

    x1, y1, x2, y2 = bbox
    x1, x2 = x1 * 1920, x2 * 1920
    y1, y2 = y1 * 1080, y2 * 1080

    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite(image_path, cropped_img)



def save_crops(left_bboxes, right_bboxes, frames, left_scores, right_scores):
    for i in tqdm(range(len(left_bboxes)), desc="Processing Frames"):           
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        _, frame = cap.read()
        
        if mirrored:
            frame = cv2.flip(frame, 1)
            
        process_player(left_bboxes[i], frame, i, "left", left_scores[i])
        process_player(right_bboxes[i], frame, i, "right", right_scores[i])
        

    
left_bboxes, right_bboxes, frames, ls, rs = get_player_boxes(df)
save_crops(left_bboxes, right_bboxes, frames, ls, rs)