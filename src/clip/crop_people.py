import pandas as pd
import ast
import cv2
import os

# Load CSV file
file_path = "midpoints.csv"
df = pd.read_csv(file_path)

video_path = "videos/game_1.mp4"
cap = cv2.VideoCapture(video_path)

def get_player_boxes(df):
    left_player_boxes = []
    right_player_boxes = []
    frames = []
    left_scores = []
    right_scores = []
    
    
    for _, row in df.iterrows():
        left_bboxes = ast.literal_eval(row["Left bbox"])
        right_bboxes = ast.literal_eval(row["Right bbox"])
        left_score = row["Left score"]
        right_score = row["Right score"]
        event_frames = row["Event frame"]
        sequence_frames = row["Sequence frame"]
        
        left_player_boxes.append(left_bboxes)
        right_player_boxes.append(right_bboxes)
        frames.append(event_frames + sequence_frames)
        left_scores.append(left_score)
        right_scores.append(right_score)
    
    return left_player_boxes, right_player_boxes, frames, left_scores, right_scores
    

def save_crops(left_bboxes, right_bboxes, frames, left_scores, right_scores):
    for i in range(len(left_bboxes)):        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        _, frame = cap.read()
        
        process_player(left_bboxes[i], frame, i, "left", left_scores[i])
        process_player(right_bboxes[i], frame, i, "right", right_scores[i])
        
        
def process_player(bbox, frame, index, player, score):
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * 1920, x2 * 1920
        y1, y2 = y1 * 1080, y2 * 1080
        
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if score > 0.9:
            frame_directory = f"cropped/video_1/{frames[index]}/0"
            if not os.path.exists(frame_directory):
                os.makedirs(frame_directory)
            
            cv2.imwrite(f"{frame_directory}/{player}.png", cropped_img)

    
left_bboxes, right_bboxes, frames, ls, rs = get_player_boxes(df)
save_crops(left_bboxes, right_bboxes, frames, ls, rs)