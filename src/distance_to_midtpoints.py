import pandas as pd
import ast
import math

# Load CSV file
file_path = "normalized_data.csv"
df = pd.read_csv(file_path)

TABLE_MIDPOINT = (0.5, 0.5)

def calculate_distance(midpoint, keypoint):    
    keypoint_x = keypoint[0]
    keypoint_y = keypoint[1]
    
    midpoint_x = midpoint[0]
    midpoint_y = midpoint[1]
    
    d = math.sqrt((midpoint_x - keypoint_x) ** 2 + (midpoint_y - keypoint_y) ** 2)
    
    return d


def compute_player_midpoints(df):
    distances = []
    midtpoints = []

    for idx, row in df.iterrows():
        mm = []
        dd = []

        if isinstance(row["Keypoints"], str):
            keypoints_row = ast.literal_eval(row["Keypoints"])

        for i in range(len(keypoints_row)):
            ddd = []
            left_hip = keypoints_row[i][11]
            mm.append(left_hip)

            for j in range(len(keypoints_row[i])):
                keypoint = keypoints_row[i][j]
                
                d = calculate_distance(left_hip, keypoint)
                ddd.append(d)
            
            dd.append(ddd)
          
        midtpoints.append(mm)  
        distances.append(dd)
                
    return midtpoints, distances

    
midpoints, distances = compute_player_midpoints(df)

# Prepare data for saving to CSV
data = {
    'Path': df["Path"],
    'Event frame': df["Event frame"],
    'Sequence frame': df["Sequence frame"],
    'Keypoints': df["Keypoints"],
    'Midpoints': midpoints,
    'Distances to midpoint': distances
}

#Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
output_file = "midpoints_all.csv"
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")