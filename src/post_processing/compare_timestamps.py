import pandas as pd
 
video = 2
bbox_file = f'../../data/video_{video}/bbox_video{video}.csv'
keypoint_file = f'../../data/video_{video}/keypoints_video{video}.csv'

def compare_event_frames(file1, file2):
    # Load CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Extract unique event frame values
    event_frames_1 = set(df1['Event frame'].unique())
    event_frames_2 = set(df2['Event frame'].unique())

    # Compare
    if event_frames_1 == event_frames_2:
        print("✅ Event frames match in both files.")
    else:
        print("❌ Event frames do NOT match.")
        only_in_file1 = event_frames_1 - event_frames_2
        only_in_file2 = event_frames_2 - event_frames_1

        if only_in_file1:
            print(f"Event frames only in {file1}: {sorted(only_in_file1)}")
        if only_in_file2:
            print(f"Event frames only in {file2}: {sorted(only_in_file2)}")


compare_event_frames(bbox_file, keypoint_file)
