import pandas as pd
import ast
import math

# Load CSV file
video = 3
file_path = f"normalized_data_video{video}.csv"
df = pd.read_csv(file_path)
mirrored = True

TABLE_MIDPOINT = (0.5, 0.5)

def compute_player_midpoints(df):
    paths = []
    event_frames = []
    sequence_frames = []
    keypoints_left = []
    keypoints_right = []
    left_score = []
    right_score = []
    left_bboxes = []
    right_bboxes = []
    
    for idx, row in df.iterrows():
        if isinstance(row["Keypoints"], str):
            keypoints_row = ast.literal_eval(row["Keypoints"])

        if isinstance(row["People scores"], str):
            scores_row = ast.literal_eval(row["People scores"])
            
        if isinstance(row["People boxes"], str):
            bboxes_row = ast.literal_eval(row["People boxes"])

        path = row["Path"]
        event_frame = row["Event frame"]
        sequence_frame = row["Sequence frame"]
        
        added = False
            
        for i, keypoints in enumerate(keypoints_row):
            if keypoints[11][0] < TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_left) <= idx:
                keypoints_left.append(keypoints)
                left_score.append(scores_row[i])
                left_bboxes.append(bboxes_row[i])
                added = True
            elif keypoints[11][0] > TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_right) <= idx:
                keypoints_right.append(keypoints)
                right_score.append(scores_row[i])
                right_bboxes.append(bboxes_row[i])
                added = True
        
        if added:
            paths.append(path)
            event_frames.append(event_frame)
            sequence_frames.append(sequence_frame)
    
    return paths, event_frames, sequence_frames, keypoints_left, left_score, left_bboxes, keypoints_right, right_score, right_bboxes

def get_hips(keypoints_left, keypoints_right):
    ll_hip = []     # Left player, left hip
    lr_hip = []     # Left player, right hip
    rl_hip = []     # Right player, left hip
    rr_hip = []     # Right player, right hip
    
    for i in range(len(keypoints_left)):
        ll_hip.append(keypoints_left[i][11])
        lr_hip.append(keypoints_left[i][12])
        rl_hip.append(keypoints_right[i][11])
        rr_hip.append(keypoints_right[i][12])
    
    return ll_hip, lr_hip, rl_hip, rr_hip

def calculate_distance(midpoint, keypoint):    
    keypoint_x = keypoint[0]
    keypoint_y = keypoint[1]
    
    midpoint_x = midpoint[0]
    midpoint_y = midpoint[1]
    
    d = math.sqrt((midpoint_x - keypoint_x) ** 2 + (midpoint_y - keypoint_y) ** 2)
    
    return d

def get_distance(ll_hips, keypoints_left, rl_hips, keypoints_right):
    left_player_distance_to_midpoint = []
    right_player_distance_to_midpoint = []
    
    for i in range(len(keypoints_left)):
        left_hip = ll_hips[i]
        right_hip = rl_hips[i]
        
        left_frame_list = []
        right_frame_list = []

        for j in range(len(keypoints_left[i])):
            left_keypoint = keypoints_left[i][j]
            right_keypoint = keypoints_right[i][j]
            
            left_d = calculate_distance(left_hip, left_keypoint)
            right_d = calculate_distance(right_hip, right_keypoint)
                        
            left_frame_list.append(left_d)
            right_frame_list.append(right_d)
        
        left_player_distance_to_midpoint.append(left_frame_list)
        right_player_distance_to_midpoint.append(right_frame_list)
        
    
    return left_player_distance_to_midpoint, right_player_distance_to_midpoint
    
paths, event_frames, sequence_frames, keypoints_left, left_score, left_bboxes, keypoints_right, right_score, right_bboxes = compute_player_midpoints(df)
ll_hip, lr_hip, rl_hip, rr_hip = get_hips(keypoints_left, keypoints_right)
left_distances, right_distances = get_distance(ll_hip, keypoints_left, rl_hip, keypoints_right)

# Prepare data for saving to CSV
output_file = f"midpoints_video{video}.csv"
if mirrored:
    output_file = f"mirrored_midpoints_video{video}.csv"

data = {
    'Path': paths,
    'Event frame': event_frames,
    'Sequence frame': sequence_frames,
    'Keypoints left': keypoints_left,
    'Left score': left_score,
    'Left bbox': left_bboxes,
    'Left player left hip': ll_hip,
    'Left player right hip': lr_hip,
    'Left distances': left_distances,
    'Keypoints right': keypoints_right,
    'Right score': right_score,
    'Right bbox': right_bboxes,
    'Right player left hip': rl_hip,
    'Right player right hip': rr_hip,
    'Right distances': right_distances
}

# Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")