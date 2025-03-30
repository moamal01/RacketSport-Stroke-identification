import pandas as pd
import ast

# Load CSV file
file_path = "normalized_data_video2.csv"
df = pd.read_csv(file_path)

def mirror_string(input_str):
    mirrored = input_str.replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')
    return mirrored

def mirror_keypoints(keypoints):
    mirrored_keypoints = []
    for keypoint in keypoints:
        x, y, confidence = keypoint
        x = 1 - x
        
        mirrored_keypoints.append([x, y, confidence])
        
    return mirrored_keypoints

def mirror_bboxes(bbox):
    mirrored_bboxes = []
    x_min, y_min, x_max, y_max = bbox
    x_min = 1 - x_min
    x_max = 1 - x_max
    
    mirrored_bboxes.append([x_min, y_min, x_max, y_max])
        
    return mirrored_bboxes
    
for idx, row in df.iterrows():
    type = row['Type']
    df.at[idx, 'Type'] = mirror_string(type)
    
    # Keypoints
    keypoints_list = ast.literal_eval(row["Keypoints"])
    mirrored_keypoints = []
    for keypoints in keypoints_list:
        mirrored_keypoints.append(mirror_keypoints(keypoints))
    
    # Boxes
    people_boxes_list = ast.literal_eval(row["People boxes"])
    mirrored_people_boxes = []
    for boxes in people_boxes_list:
        mirrored_people_boxes.append(mirror_bboxes(boxes))
        
    ball_boxes_list = ast.literal_eval(row["Ball boxes"])
    mirrored_ball_boxes = []
    for boxes in ball_boxes_list:
        mirrored_ball_boxes.append(mirror_bboxes(boxes))
        
    racket_boxes_list = ast.literal_eval(row["Racket boxes"])
    mirrored_racket_boxes = []
    for boxes in racket_boxes_list:
        mirrored_racket_boxes.append(mirror_bboxes(boxes))
        
    table_boxes_list = ast.literal_eval(row["Table boxes"])
    mirrored_table_boxes = []
    for boxes in racket_boxes_list:
        mirrored_table_boxes.append(mirror_bboxes(boxes))
        
    df.at[idx, 'Keypoints'] = mirrored_keypoints
    df.at[idx, 'People boxes'] = people_boxes_list
    df.at[idx, 'Ball boxes'] = ball_boxes_list
    df.at[idx, 'Racket boxes'] = racket_boxes_list
    df.at[idx, 'Table boxes'] = table_boxes_list
    

new_file_path = "mirrored_midpoints_video2.csv"
df.to_csv(new_file_path, index=False)