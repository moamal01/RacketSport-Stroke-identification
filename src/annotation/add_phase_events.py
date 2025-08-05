import json


video = 4

with open(f"data/events/events_markup{video}.json", "r") as f:
    data = json.load(f)

excluded_values = {"net", "bounce", "empty_event", "other"}
data = {k: v for k, v in data.items() if v not in excluded_values}

play_timestamps = []
pause_timestamps = []

last_stroke = None
last_frame = 0

for key, value in data.items():
    midpoint = round((int(key) + last_frame) / 2)
    
    if "serve" in value:
        pause_timestamps.append(midpoint)
        if int(key) - last_frame > 1000:
            pause_timestamps.append(midpoint + 400)
            pause_timestamps.append(midpoint - 400)
        
    elif "serve" not in value:
        play_timestamps.append(midpoint)
    else:
        pause_timestamps.append(midpoint)

    last_frame = int(key)

for ts in pause_timestamps:
    data[str(ts)] = "pause"

for ts in play_timestamps:
    data[str(ts)] = "play"

sorted_data = dict(sorted(data.items(), key=lambda x: int(x[0])))

with open(f"data/extended_events/events_markup{video}.json", "w") as f:
    json.dump(sorted_data, f, indent=4)

print(f"Added phase timestamps to data")
