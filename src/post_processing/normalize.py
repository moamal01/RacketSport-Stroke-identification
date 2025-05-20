import pandas as pd
import ast

# Load CSV
video = 4
input_file = f"../../data/video_{video}/merged_output_video{video}.csv"
output_file = f"../../data/video_{video}/normalized_data_video{video}.csv"

# Read CSV
df = pd.read_csv(input_file)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def normalize_keypoints(row, width, height):
    try:
        # Convert string to list if needed
        if isinstance(row, str):
            row = ast.literal_eval(row)
        
        if isinstance(row, list):
            for person in row:
                for joint in person:
                    joint[0] = joint[0] / width
                    joint[1] = joint[1] / height
        return row
    except Exception as e:
        print(f"Error processing row: {row}, Error: {e}")
        return row
    
def normalize_boxes(row, width, height):
    try:
        # Convert string to list if needed
        if isinstance(row, str):
            row = ast.literal_eval(row)
        
        if isinstance(row, list):
            for box in row:
                box[0] = box[0] / width     # X1
                box[1] = box[1] / height    # Y1
                box[2] = box[2] / width     # X2
                box[3] = box[3] / height    # Y2
        return row
    except Exception as e:
        print(f"Error processing row: {row}, Error: {e}")
        return row

# Normalize all specified columns
keypoint_columns = ["Keypoints"]
bbox_columns = ["People boxes", "Ball boxes", "Racket boxes", "Table boxes"]

for column in keypoint_columns:
    if column in df.columns:
        df[column] = df[column].apply(lambda x: normalize_keypoints(x, IMAGE_WIDTH, IMAGE_HEIGHT))
        

for column in bbox_columns:
    if column in df.columns:
        df[column] = df[column].apply(lambda x: normalize_boxes(x, IMAGE_WIDTH, IMAGE_HEIGHT))

# Save the result to a new CSV
df.to_csv(output_file, index=False)

print(f"Normalized data saved to {output_file}")