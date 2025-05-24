import json

video = 2
input_file = f"../../data/extended_events/events_markup{video}p1.json"
output_file = f"../../data/extended_events/events_markup{video}p2.json"
offset = 75306

with open(input_file, "r") as f:
    data = json.load(f)

adjusted_data = {
    str(int(key) - offset): value
    for key, value in data.items()
}

with open(output_file, "w") as f:
    json.dump(adjusted_data, f, indent=4)

print("Adjusted JSON saved to", output_file)
