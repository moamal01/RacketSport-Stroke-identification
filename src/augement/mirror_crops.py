import pandas as pd
import cv2
import json

video_number = 2
player = "left"

file_path = "../midpoints.csv"
df = pd.read_csv(file_path)

video_path = "../videos/game_1.mp4"
cap = cv2.VideoCapture(video_path)

with open(f"../data/events/events_markup{video_number}.json", "r") as file:
    data = json.load(file)
    
excluded_values = {"empty_event", "bounce", "net"}
stroke_frames = {k: v for k, v in data.items() if v not in excluded_values}

for frame, value in stroke_frames.items():
    if value == "other" or value == "otherotherother":
        continue
    
    value1 = value.split(" ")[0]
    value2 = value1.split("_")[2]
    if player in value1: # and labels.count(value) < 10:
        file_path = f"../cropped/video_{video_number}/{frame}/0/{player}.png"
        if os.path.exists(file_path):
            if not (29000 < int(frame) and int(frame) < 66000 ) or (94000 < int(frame) and int(frame) < 135000) or int(frame) > 150000:

            
                image_data = mpimg.imread(file_path)
                
                # Display the image using matplotlib
                plt.imshow(image_data)
                plt.title(f"Frame: {frame} - Stroke: {value1}")
                plt.axis('off')  # Hide the axis
                plt.show()  # Show the image