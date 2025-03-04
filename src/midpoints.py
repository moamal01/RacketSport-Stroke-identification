import pandas as pd
import ast

# Load CSV file
file_path = "normalized_data.csv"
df = pd.read_csv(file_path)

TABLE_MIDPOINT = (0.5, 0.5)

# Function to compute player's body midpoint (e.g., average of hips)
def compute_player_midpoints(df):
    paths = []
    event_frames = []
    sequence_frames = []
    keypoints_left = []
    keypoints_right = []
    left_score = []
    right_score = []
    
    for idx, row in df.iterrows():
        if isinstance(row["Keypoints"], str):
            keypoints_row = ast.literal_eval(row["Keypoints"])
            scores_row = ast.literal_eval(row["People scores"])
            path = row["Path"]
            event_frame = row["Event frame"]
            sequence_frame = row["Sequence frame"]
            
        for i, keypoints in enumerate(keypoints_row):
            if keypoints[0][0] < TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_left) <= idx:
                keypoints_left.append(keypoints)
                left_score.append(scores_row[i])
            elif keypoints[0][0] > TABLE_MIDPOINT[0] and abs(keypoints[11][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_right) <= idx:
                paths.append(path)
                event_frames.append(event_frame)
                sequence_frames.append(sequence_frame)
                keypoints_right.append(keypoints)
                right_score.append(scores_row[i])
    
    return paths, event_frames, sequence_frames, keypoints_left, left_score, keypoints_right, right_score

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
    
    
paths, event_frames, sequence_frames, keypoints_left, left_score, keypoints_right, right_score = compute_player_midpoints(df)
ll_hip, lr_hip, rl_hip, rr_hip = get_hips(keypoints_left, keypoints_right)

# Prepare data for saving to CSV
data = {
    'Path': paths,
    'Event frame': event_frames,
    'Sequence frame': sequence_frames,
    'Keypoints left': keypoints_left,
    'Left score': left_score,
    'Left player left hip': ll_hip,
    'Left player right hip': lr_hip,
    'Keypoints right': keypoints_right,
    'Right score': right_score,
    'Right player left hip': rl_hip,
    'Right player right hip': rr_hip,
}

# Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
output_file = "midpoints1.csv"
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")