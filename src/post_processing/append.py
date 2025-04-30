import pandas as pd

new_csv = "../../data/video_2/bbox_video2"
df1 = pd.read_csv(f'{new_csv}_p1.csv')
df2 = pd.read_csv(f'{new_csv}_p2.csv')

# Ensure headers match
if list(df1.columns) != list(df2.columns):
    raise ValueError("CSV headers do not match!")

# Combine and export
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv(f"{new_csv}.csv", index=False)

print(f"Combined CSV created as {new_csv}.csv")
