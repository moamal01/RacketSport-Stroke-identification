import pandas as pd

video = 2
bbox = False
# Read the two CSV files
if bbox:
    df1 = pd.read_csv(f'../../data/video_{video}/bbox_video{video}q3.csv')
    df2 = pd.read_csv(f'../../data/video_{video}/bbox_video{video}full4.csv')
else:
    df1 = pd.read_csv(f'../../data/video_{video}/keypoints_video{video}q3.csv')
    df2 = pd.read_csv(f'../../data/video_{video}/keypoints_video{video}full4.csv')

# Combine the two dataframes
combined_df = pd.concat([df1, df2])

# Sort by the 'frame' column
sorted_df = combined_df.sort_values(by=['Event frame'])

# Optionally reset the index
sorted_df.reset_index(drop=True, inplace=True)

# Save to a new CSV file
if bbox:
    sorted_df.to_csv(f'../../data/video_{video}/bbox_video{video}.csv', index=False)
else:
    sorted_df.to_csv(f'../../data/video_{video}/keypoints_video{video}.csv', index=False)
