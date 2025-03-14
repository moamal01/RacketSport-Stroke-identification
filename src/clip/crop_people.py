import pandas as pd
import ast
import cv2
import os

# Load CSV file
file_path = "midpoints_video2.csv"
df = pd.read_csv(file_path)

video_path = "videos/game_2.mp4"
cap = cv2.VideoCapture(video_path)

def get_player_boxes(df):
    left_player_boxes = []
    right_player_boxes = []
    frames = []
    
    for _, row in df.iterrows():
        left_bboxes = ast.literal_eval(row["Left bbox"])
        right_bboxes = ast.literal_eval(row["Right bbox"])
        event_frames = row["Event frame"]
        sequence_frames = row["Sequence frame"]
        
        left_player_boxes.append(left_bboxes)
        right_player_boxes.append(right_bboxes)
        frames.append(event_frames + sequence_frames)
    
    return left_player_boxes, right_player_boxes, frames
    

def save_crops(left_bboxes, right_bboxes, frames):
    for i in range(len(left_bboxes)):        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        _, frame = cap.read()
        
        process_player(left_bboxes[i], frame, i, "left")
        process_player(right_bboxes[i], frame, i, "right")
        
        
def process_player(bbox, frame, index, player):
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * 1920, x2 * 1920
        y1, y2 = y1 * 1080, y2 * 1080
        
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
        frame_directory = f"cropped/video_2/{frames[index]}/0"
        if not os.path.exists(frame_directory):
            os.makedirs(frame_directory)
        
        cv2.imwrite(f"{frame_directory}/{player}.png", cropped_img)

    
left_bboxes, right_bboxes, frames = get_player_boxes(df)
save_crops(left_bboxes, right_bboxes, frames)