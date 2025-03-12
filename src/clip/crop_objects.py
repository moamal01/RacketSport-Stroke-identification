import pandas as pd
import ast
import cv2
import os

# Load CSV file
file_path = "normalized_data.csv"
df = pd.read_csv(file_path)

video_path = "videos/game_1.mp4"
cap = cv2.VideoCapture(video_path)

def get_player_boxes(df):
    table_boxes = []
    racket_boxes = []
    ball_boxes = []
    frames = []
    
    for _, row in df.iterrows():
        table_box = ast.literal_eval(row["Table boxes"])
        racket_box = ast.literal_eval(row["Racket boxes"])
        ball_box = ast.literal_eval(row["Ball boxes"])
        event_frames = row["Event frame"]
        sequence_frames = row["Sequence frame"]
        
        table_boxes.append(table_box)
        racket_boxes.append(racket_box)
        ball_boxes.append(ball_box)
        frames.append(event_frames + sequence_frames)
    
    return table_boxes, racket_boxes, ball_boxes, frames
    

def save_crops(list, frames, object):
    for i in range(len(list)):  
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        _, frame = cap.read()

        if list[i]:
            process_object(list[i][0], frame, i, object)
        
def process_object(bbox, frame, index, object):
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * 1920, x2 * 1920
        y1, y2 = y1 * 1080, y2 * 1080
        
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
        frame_directory = f"cropped/video_1/{frames[index]}/{object}"
        if not os.path.exists(frame_directory):
            os.makedirs(frame_directory)
        
        cv2.imwrite(f"{frame_directory}/object.png", cropped_img)

    
table_boxes, racket_boxes, ball_boxes, frames = get_player_boxes(df)
save_crops(table_boxes, frames, 32)
save_crops(racket_boxes, frames, 38)
save_crops(ball_boxes, frames, 60)