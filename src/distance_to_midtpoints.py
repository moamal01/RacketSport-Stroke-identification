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

def compute_table_midpoints(df):
    t_midtpoints = []

    for _, row in df.iterrows():
        mm = []

        if isinstance(row["Table boxes"], str):
            table_row = ast.literal_eval(row["Table boxes"])

        for _ in range(len(table_row)):
            for j in range(len(table_row)):
                table_coords = table_row[j]
                
                x1 = table_coords[0]
                x2 = table_coords[2]
                
                mm.append([(x1 + x2) / 2, table_coords[3]])
            
            mm.append(mm)
        t_midtpoints.append(mm)  
                
    return t_midtpoints


def center_to_midpoint_distances(midpoints):
    distances = []
    for row in midpoints:
        distances_within_row = []
        for midpoint in row:
            d = math.sqrt((midpoint[0] - 0.5) ** 2 + (midpoint[1] - 0.5) ** 2)
            distances_within_row.append(d)
        distances.append(distances_within_row)
    return distances

# def midpoint_distances(p_midpoints, t_midpoints):
#     distances = []
    
#     for i in range(len(p_midpoints)):
#         for j in range(len(t_midpoints)):
#             for midpoint in p_midpoints[j]:
#                 d_for_each_tablemidpoint = []
#                 for k in range(len(t_midpoints[i])):
#                     print(t_midpoints[i][k])
#                     tx = t_midpoints[i][k][0]
#                     ty = t_midpoints[i][k][0]
                    
#                     mx = midpoint[0]
#                     my = midpoint[0]
                    
#                     d = math.sqrt((tx - mx) ** 2 + (ty - my) ** 2)
                    
#                     d_for_each_tablemidpoint.append(d)
                    
#             break
#         break
  
    

midpoints, distances = compute_player_midpoints(df)
table_midpoints = compute_table_midpoints(df)
tm_distances = center_to_midpoint_distances(midpoints)

# Prepare data for saving to CSV
data = {
    'Path': df["Path"],
    'Event frame': df["Event frame"],
    'Sequence frame': df["Sequence frame"],
    'Keypoints': df["Keypoints"],
    'Midpoints': midpoints,
    'Distances to midpoint': distances,
    'Distances to table midpoints': tm_distances,
    'Table boxes': df['Table boxes'],
    'Table midpoints': table_midpoints
}

#Create DataFrame from the data
result_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
output_file = "midpoints_all.csv"
result_df.to_csv(output_file, index=False)

print(f"Keypoints and scores have been saved to {output_file}")