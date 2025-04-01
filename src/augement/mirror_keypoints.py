import pandas as pd
import ast
from utility_functions import mirror_string

# Load CSV file
video = 3
file_path = f"normalized_data_video{3}.csv"
df = pd.read_csv(file_path)

def mirror_keypoints(keypoints):
    mirrored_keypoints = []
    for keypoint in keypoints:
        x, y, confidence = keypoint
        x = 1 - x
        
        mirrored_keypoints.append([x, y, confidence])
        
    return mirrored_keypoints

def mirror_bboxes(bbox):
    x_min, y_min, x_max, y_max = bbox
    mirrored_x_min = 1 - x_max
    mirrored_x_max = 1 - x_min
    
    return [mirrored_x_min, y_min, mirrored_x_max, y_max]
    
    
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
    df.at[idx, 'People boxes'] = mirrored_people_boxes
    df.at[idx, 'Ball boxes'] = mirrored_ball_boxes
    df.at[idx, 'Racket boxes'] = mirrored_racket_boxes
    df.at[idx, 'Table boxes'] = mirrored_table_boxes
    

output_file = f"mirrored_normalized_video{3}.csv"
df.to_csv(output_file, index=False)
print(f"Mirroring completed to {output_file}")