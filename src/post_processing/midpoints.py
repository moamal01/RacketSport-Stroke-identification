import pandas as pd
import ast

# Load CSV file
video = 1
file_path = f"../../data/video_{video}/normalized_data_video{video}.csv"
df = pd.read_csv(file_path)
mirrored = True

def get_table_midpoints(df):
    table_midpoints = []
    for _, row in df.iterrows():
        table_boxes_row = ast.literal_eval(row["Table boxes"])
        table_midpoints.append([(table_boxes_row[0][0] + table_boxes_row[0][2]) / 2, table_boxes_row[0][1]])

    return table_midpoints

def compute_player_midpoints(df, table_midpoints):
    paths = []
    event_frames = []
    keypoints_left = []
    keypoints_right = []
    left_score = []
    right_score = []
    left_bboxes = []
    right_bboxes = []
    left_racket_boxes = []
    right_racket_boxes = []
    left_racket_scores = []
    right_racket_scores = []
    ball_boxes = []
    ball_scores = []
    
    
    for idx, row in df.iterrows():
        keypoints_row = ast.literal_eval(row["Keypoints"])
        scores_row = ast.literal_eval(row["People scores"])
        people_boxes_row = ast.literal_eval(row["People boxes"])
        racket_boxes_row = ast.literal_eval(row["Racket boxes"])
        racket_scores_row = ast.literal_eval(row["Racket scores"])
        ball_boxes_row = ast.literal_eval(row["Ball boxes"])
        ball_scores_row = ast.literal_eval(row["Ball scores"])
        
        TABLE_MIDPOINT = table_midpoints[idx]

        path = row["Path"]
        event_frame = row["Event frame"]
        
        added_left = False 
        added_right = False
            
        # Keypoints
        for i, keypoints in enumerate(keypoints_row):
            if keypoints[11][0] < TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_left) <= idx:
                keypoints_left.append(keypoints)
                left_score.append(scores_row[i])
                left_bboxes.append(people_boxes_row[i])
                added_left = True
            elif keypoints[11][0] > TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_right) <= idx:
                keypoints_right.append(keypoints)
                right_score.append(scores_row[i])
                right_bboxes.append(people_boxes_row[i])
                added_right = True
               
        # Rackets 
        if not added_left:
            keypoints_left.append([])
            left_score.append(0)
            left_bboxes.append([])

        if not added_right:
            keypoints_right.append([])
            right_score.append(0)
            right_bboxes.append([])
            
            
        for i, box in enumerate(racket_boxes_row):
            if box[0] < TABLE_MIDPOINT[0] and len(left_racket_boxes) <= idx:
                left_racket_boxes.append(box)
                left_racket_scores.append(racket_scores_row[i])
            elif box[0] > TABLE_MIDPOINT[0] and len(right_racket_boxes) <= idx:
                right_racket_boxes.append(box)
                right_racket_scores.append(racket_scores_row[i])
                
        if len(left_racket_boxes) <= idx:
            left_racket_boxes.append([])
        if len(left_racket_scores) <= idx:
            left_racket_scores.append(0)
        if len(right_racket_boxes) <= idx:
            right_racket_boxes.append([])
        if len(right_racket_scores) <= idx:
            right_racket_scores.append(0)
            
        # Ball
        for i, ball in enumerate(ball_boxes_row):
            if len(ball_boxes) <= idx:
                ball_boxes.append(ball)
                ball_scores.append(ball_scores_row[i])
        
        if len(ball_boxes) <= idx:
            ball_boxes.append([])
        if len(ball_scores) <= idx:
            ball_scores.append(0)

        paths.append(path)
        event_frames.append(event_frame)

    
    return paths, event_frames, keypoints_left, left_score, left_bboxes, keypoints_right, right_score, right_bboxes, left_racket_boxes, left_racket_scores, right_racket_boxes, right_racket_scores, ball_boxes, ball_scores

def get_hips(keypoints_left, keypoints_right):
    ll_hip = []     # Left player, left hip
    lr_hip = []     # Left player, right hip
    rl_hip = []     # Right player, left hip
    rr_hip = []     # Right player, right hip
    
    for i in range(len(keypoints_left)):
        if keypoints_left[i] == []:
            ll_hip.append([0, 0, 0])
            lr_hip.append([0, 0, 0])
        else:      
            ll_hip.append(keypoints_left[i][11])
            lr_hip.append(keypoints_left[i][12])
            
        if keypoints_right[i] == []:
            rl_hip.append([0, 0, 0])
            rr_hip.append([0, 0, 0])
        else:
            rl_hip.append(keypoints_right[i][11])
            rr_hip.append(keypoints_right[i][12])
    
    return ll_hip, lr_hip, rl_hip, rr_hip


def get_midpoint(left_hips, right_hips):
    midpoints = []
    for i in range(len(left_hips)):
        midpoint_x = (left_hips[i][0] + right_hips[i][0]) / 2
        midpoint_y = (left_hips[i][1] + right_hips[i][1]) / 2
        
        midpoints.append([midpoint_x, midpoint_y])
    
    return midpoints

def get_box_midpoint(boxes):
    midpoints = []
    
    for box in boxes:
        if box == []:
            midpoints.append([])
        else:
            x_min, y_min, x_max, y_max = box
            midpoint_x = (x_min + x_max) / 2
            midpoint_y = (y_min + y_max) / 2
            midpoints.append([midpoint_x, midpoint_y])
            

    return midpoints
    

def normalize(left_hip, right_hip, keypoint):
    keypoint_x = keypoint[0]
    keypoint_y = keypoint[1]
    
    midpoint_x = (left_hip[0] + right_hip[0]) / 2
    midpoint_y = (left_hip[1] + right_hip[1]) / 2
    
    return [keypoint_x - midpoint_x, keypoint_y - midpoint_y]

def normalize_midpoints(midpoints, table_midpoints):
    normalized_midpoints_list = []
    for i in range(len(midpoints)):
        if midpoints[i] == []:
            normalized_midpoints_list.append([])
        else:
            normalized_midpoints_list.append([midpoints[i][0] - table_midpoints[i][0], midpoints[i][1] - table_midpoints[i][1]])
        
    return normalized_midpoints_list
    

def get_normalized(ll_hips, lr_hips, keypoints_left, rl_hips, rr_hips, keypoints_right):
    left_player_distance_to_midpoint = []
    right_player_distance_to_midpoint = []
    
    for i in range(len(keypoints_left)):
        left_player_left_hip = ll_hips[i]
        left_player_right_hip = lr_hips[i]
        right_player_left_hip = rl_hips[i]
        right_player_right_hip = rr_hips[i]
        
        left_dist_frame_list = []
        right_dist_frame_list = []

        for j in range(len(keypoints_left[i])):
            if keypoints_left[i] == []:
                left_norm = []
            else:
                left_keypoint = keypoints_left[i][j]
                left_norm = normalize(left_player_left_hip, left_player_right_hip, left_keypoint)
            
            if keypoints_right[i] == []:
                right_norm = []
            else:
                right_keypoint = keypoints_right[i][j]
                right_norm = normalize(right_player_left_hip, right_player_right_hip, right_keypoint)
    
            left_dist_frame_list.append(left_norm)
            right_dist_frame_list.append(right_norm)
        
        # Populate top level lists
        left_player_distance_to_midpoint.append(left_dist_frame_list)
        right_player_distance_to_midpoint.append(right_dist_frame_list)
        
    
    return left_player_distance_to_midpoint, right_player_distance_to_midpoint

table_midpoints = get_table_midpoints(df)
paths, event_frames, keypoints_left, left_score, left_bboxes, keypoints_right, right_score, right_bboxes, left_racket_boxes, left_racket_scores, right_racket_boxes, right_racket_scores, ball_boxes, ball_scores = compute_player_midpoints(df, table_midpoints)
ll_hip, lr_hip, rl_hip, rr_hip = get_hips(keypoints_left, keypoints_right)
# midpoints
left_midpoints, right_midpoints = get_midpoint(ll_hip, lr_hip), get_midpoint(rl_hip, rr_hip)
left_racket_midpoints, right_racket_midpoints = get_box_midpoint(left_racket_boxes), get_box_midpoint(right_racket_boxes)
ball_midpoints = get_box_midpoint(ball_boxes)

# Normalized midpoints
left_mid_normalized, right_mid_normalized = get_normalized(ll_hip, lr_hip, keypoints_left, rl_hip, rr_hip, keypoints_right)
left_normalized_midpoints, right_normalized_midpoints = normalize_midpoints(left_midpoints, table_midpoints), normalize_midpoints(right_midpoints, table_midpoints)
normalized_left_racket_midpoint, normalized_right_left_racket_midpoint = normalize_midpoints(left_racket_midpoints, left_midpoints), normalize_midpoints(right_racket_midpoints, right_midpoints)
normalized_ball_midpoints = normalize_midpoints(ball_midpoints, table_midpoints)

# Prepare data for saving to CSV
output_file = f"../../data/video_{video}/midpoints_video{video}.csv"
if mirrored:
    output_file = f"../../data/video_{video}/mirrored_midpoints_video{video}.csv"

data = {
    'Path': paths,
    'Event frame': event_frames,
    'Table midpoint': table_midpoints, 
    'Ball midpoint': ball_midpoints,
    'Normalized ball midpoints': normalized_ball_midpoints,
    'Ball score': ball_scores,
    'Keypoints left': keypoints_left,
    'Left score': left_score,
    'Left bbox': left_bboxes,
    'Left player midpoint': left_midpoints,
    'Left player normalized midpoint': left_normalized_midpoints,
    'Left mid-normalized': left_mid_normalized,
    'Left racket': left_racket_midpoints,
    'Left normalized racket': left_normalized_midpoints,
    'Left racket score': left_racket_scores,
    'Keypoints right': keypoints_right,
    'Right score': right_score,
    'Right bbox': right_bboxes,
    'Right player midpoint': right_midpoints,
    'Right player normalized midpoint': right_normalized_midpoints,
    'Right mid-normalized': right_mid_normalized,
    'Right racket': right_racket_midpoints,
    'Right normalized racket': right_racket_midpoints,
    'Right racket score': right_racket_scores
}

# Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")