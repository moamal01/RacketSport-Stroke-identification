import cv2
import json

# Load the event frames from JSON
json_path = "notebooks/empty_event_keys3.json"  # Update this if needed
video_path = "videos/game_1f.mp4"  # Update this with your actual video file

with open(json_path, "r") as f:
    event_frames = json.load(f)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Process each event frame
for frame_num in sorted(map(int, event_frames.keys())):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Jump to frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow(f"Frame {frame_num}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"Frame {frame_num}: Current label = {event_frames[str(frame_num)]}")

    # Get user input for new label
    new_label = input("Enter new label (or press Enter to keep 'empty_event'): ").strip()
    if new_label:
        event_frames[str(frame_num)] = new_label  # Update the label

# Save updated labels
updated_json_path = "updated_event_keys.json"
with open(updated_json_path, "w") as f:
    json.dump(event_frames, f, indent=4)

print(f"Updated labels saved to {updated_json_path}.")
cap.release()
cv2.destroyAllWindows()
