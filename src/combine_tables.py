import pandas as pd
from collections import defaultdict

def load_data(bbox_file, keypoints_file, output_file):
    # Load CSV files
    df_bbox = pd.read_csv(bbox_file)
    df_keypoints = pd.read_csv(keypoints_file)
    
    # Create a dictionary to group bounding boxes by (Event frame, Sequence frame)
    grouped_boxes = defaultdict(lambda: {"people": [], "racket": [], "table": [], "ball": []})
    
    # Map class IDs to categories (adjust these mappings if necessary)
    class_mapping = {0: "people", 60: "racket", 1: "table", 2: "ball"}
    
    for _, row in df_bbox.iterrows():
        key = (row["Event frame"], row["Sequence frame"])
        box = ([row["X1"], row["Y1"], row["X2"], row["Y2"]], row["Score"])
        category = class_mapping.get(row["Class ID"], None)
        if category:
            grouped_boxes[key][category].append(box)
    
    # Merge with keypoints data
    merged_data = []
    for _, row in df_keypoints.iterrows():
        key = (row["Event frame"], row["Sequence frame"])
        boxes = grouped_boxes[key]
        
        merged_data.append([
            row["Path"], row["Type"], row["Event frame"], row["Sequence frame"],
            row["Player_1 keypoints"], row["Player_2 keypoints"],
            boxes["people"], boxes["racket"], boxes["table"], boxes["ball"]
        ])
    
    # Convert to DataFrame and save
    columns = ["Path", "Type", "Event frame", "Sequence frame", "Player_1 keypoints", "Player_2 keypoints", 
               "people boxes", "racket boxes", "table boxes", "ball boxes"]
    df_merged = pd.DataFrame(merged_data, columns=columns)
    df_merged.to_csv(output_file, index=False)
    print(f"Merged CSV saved as {output_file}")
    
load_data("bbox5.csv", "keypoints5.csv", "merged_output.csv")