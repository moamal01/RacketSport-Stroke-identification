import pandas as pd
import sys
import os
import math
import ast

sys.path.append(os.path.abspath('../..'))
from utility_functions import load_json_with_dicts

midpoints_table = "../../data/video_4/midpoints_video4.csv"
df = pd.read_csv(midpoints_table)
true_balls = load_json_with_dicts(f"../../data/video_4/ball_markup.json")

def normalize_true_balls(x, y):
    norm_x = x / 1920
    norm_y = y / 1080
    
    return [norm_x, norm_y]

def is_it_inside(true_ball, detected_ball, radius=1.0):
    dx = true_ball[0] - detected_ball[0]
    dy = true_ball[1] - detected_ball[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance <= radius

insides = 0
outsides = 0

for key, coords in true_balls.items():
    row = df[(df['Event frame'] == int(key))]
    
    if row.empty:
        continue
    
    detected_ball = ast.literal_eval(row["Ball midpoint"].values[0])
    if detected_ball == []:
        outsides += 1

    true_ball = normalize_true_balls(int(coords['x']), int(coords['y']))
    
    if is_it_inside(true_ball, detected_ball):
        insides += 1
    else:
        outsides += 1
        

print(insides)
print(outsides)
