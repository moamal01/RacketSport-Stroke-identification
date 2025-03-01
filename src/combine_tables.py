import pandas as pd
import ast

# File paths
keypoints_file = "keypoints5.csv"
bbox_file = "bbox5.csv"

# Load the CSV files
keypoints_df = pd.read_csv(keypoints_file)
bbox_df = pd.read_csv(bbox_file)

# Function to safely convert bbox strings to lists of numbers
def parse_bbox(value):
    try:
        return [float(num) for num in ast.literal_eval(value)]
    except (ValueError, SyntaxError, TypeError):
        return []

# Initialize empty columns for categorized bounding boxes and scores
bbox_df["Ball boxes"] = None
bbox_df["Ball scores"] = None
bbox_df["Racket boxes"] = None
bbox_df["Racket scores"] = None
bbox_df["Table boxes"] = None
bbox_df["Table scores"] = None

# Assign bounding boxes and scores to respective categories
for index, row in bbox_df.iterrows():
    bbox = parse_bbox(row["Bboxes"])
    
    if row["Class ID"] == 32:  # Ball
        bbox_df.at[index, "Ball boxes"] = bbox
        bbox_df.at[index, "Ball scores"] = row["Score"]
    elif row["Class ID"] == 38:  # Racket
        bbox_df.at[index, "Racket boxes"] = bbox
        bbox_df.at[index, "Racket scores"] = row["Score"]
    elif row["Class ID"] == 60:  # Table
        bbox_df.at[index, "Table boxes"] = bbox
        bbox_df.at[index, "Table scores"] = row["Score"]

# Drop unneeded columns
bbox_df = bbox_df.drop(columns=["Class ID", "Score", "Bboxes"])

# Group by Event frame and Sequence frame to aggregate multiple detections
def aggregate_lists(series):
    return [item for sublist in series.dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]

bbox_agg = bbox_df.groupby(["Event frame", "Sequence frame"]).agg(aggregate_lists).reset_index()

# Merge with keypoints dataset
merged_df = pd.merge(keypoints_df, bbox_agg, on=["Event frame", "Sequence frame"], how="left")

# Fill NaN values with empty lists where necessary
for col in ["Ball boxes", "Ball scores", "Racket boxes", "Racket scores", "Table boxes", "Table scores"]:
    merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, list) else [])

# Save to CSV
merged_df.to_csv("merged_output.csv", index=False)

print("Merged dataset saved as 'merged_output.csv'")
