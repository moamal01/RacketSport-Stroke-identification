import pandas as pd
import ast

# Load CSV file
file_path = "normalized_data.csv"
df = pd.read_csv(file_path)

TABLE_MIDPOINT = (0.5, 0.5)

# Function to compute player's body midpoint (e.g., average of hips)
def compute_player_midpoints(df):
    keypoints_left_left_hip = []
    keypoints_right_left_hip = []
    left_score = []
    right_score = []
    
    for idx, row in df.iterrows():
        if isinstance(row["Keypoints"], str):
            keypoints_row = ast.literal_eval(row["Keypoints"])
            scores_row = ast.literal_eval(row["People scores"])
            
        for i, keypoints in enumerate(keypoints_row):
            if keypoints[0][0] < TABLE_MIDPOINT[0] and abs(keypoints[0][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_left_left_hip) <= idx:
                keypoints_left_left_hip.append(keypoints)
                left_score.append(scores_row[i])
            elif keypoints[0][0] > TABLE_MIDPOINT[0] and abs(keypoints[0][0] - TABLE_MIDPOINT[0]) > 0.1 and len(keypoints_right_left_hip) <= idx:
                keypoints_right_left_hip.append(keypoints)
                right_score.append(scores_row[i])
    
    return keypoints_left_left_hip, left_score, keypoints_right_left_hip, right_score

    
keypoints_left, left_score, keypoints_right, right_score = compute_player_midpoints(df)

# Prepare data for saving to CSV
data = {
    'Keypoints Left': keypoints_left,
    'Left Score': left_score,
    'Keypoints Right': keypoints_right,
    'Right Score': right_score
}

# Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
output_file = "midpoints.csv"
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")