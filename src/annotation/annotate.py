import cv2
import json

# File paths
json_path = "events_markup.json"
video_path = "videos/game_1f.mp4"

# Load the event frames from JSON
with open(json_path, "r") as f:
    event_frames = json.load(f)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Process each event frame
for frame_num, label in sorted(event_frames.items(), key=lambda x: int(x[0])):
    if label != "empty_event":
        continue  # Skip non-empty events

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))  # Jump to frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_num}. Skipping...")
        continue

    # Display the frame
    cv2.imshow(f"Frame {frame_num}", frame)
    print(f"Frame {frame_num}: Current label = {label}")

    # Wait for user input to show the frame
    key = cv2.waitKey(0)  # Wait indefinitely for user input
    if key == 27:  # Press 'ESC' to exit early
        break

    # Close frame window before asking for input
    cv2.destroyAllWindows()

    # Get user input for new label
    new_label = input("Enter new label (or press Enter to keep 'empty_event'): ").strip()
    if new_label:
        event_frames[frame_num] = new_label  # Update the label

        # Save immediately to prevent data loss
        with open(json_path, "w") as f:
            json.dump(event_frames, f, indent=4)

        print(f"Label updated and saved for frame {frame_num}.")

cap.release()
cv2.destroyAllWindows()
print("Annotation process completed.")
