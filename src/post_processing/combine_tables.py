import pandas as pd
import ast

# File paths
video = 3
keypoints_file = f"keypoints_video{video}.csv"
bbox_file = f"bbox_video{video}.csv"

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

def aggregate_lists(series):
    return [sublist for sublist in series.dropna() if isinstance(sublist, list)]

def flatten_lists(series):
    return [item for item in series.dropna()]

# Apply different aggregation functions
bbox_agg = bbox_df.groupby(["Event frame", "Sequence frame"]).agg({
    "Ball boxes": aggregate_lists,
    "Ball scores": flatten_lists,
    "Racket boxes": aggregate_lists,
    "Racket scores": flatten_lists,
    "Table boxes": aggregate_lists,
    "Table scores": flatten_lists
}).reset_index()

# Merge with keypoints dataset
merged_df = pd.merge(keypoints_df, bbox_agg, on=["Event frame", "Sequence frame"], how="left")

# Fill NaN values with empty lists where necessary
for col in ["Ball boxes", "Ball scores", "Racket boxes", "Racket scores", "Table boxes", "Table scores"]:
    merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, list) else [])

# Save to CSV
merged_df.to_csv(f"merged_output_video{video}.csv", index=False)

print(f"Merged dataset saved as 'merged_output{video}.csv'")