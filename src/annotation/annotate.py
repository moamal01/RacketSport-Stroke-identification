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

    start_frame = max(0, int(frame_num) - 20)  # Ensure start frame is non-negative
    end_frame = min(int(frame_num) + 20, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to the starting frame
        print(f"Showing frames {start_frame} to {end_frame} (Event frame: {frame_num})")
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if we can't read a frame

            cv2.imshow(f"Event Preview (Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))})", frame)

            key = cv2.waitKey(50)  # Play at ~20 FPS (adjust as needed)
            if key == 27:  # Press 'ESC' to exit early
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord('r'):  # Press 'r' to restart the segment
                break  # Break out of the inner loop to restart
        else:
            break  # Break out of the outer loop if no restart requested

    cv2.destroyAllWindows()  # Close the video window before labeling

    # Get user input for the new label
    new_label = input("Enter new label (or press Enter to keep 'empty_event'): ").strip()

    if new_label:
        label_map = {
            "e": "empty_event",
            # Serve
            "lfsv": "left forehand serve",
            "rfsv": "right forehand serve",
            "lbsv": "left backhand serve",
            "rbsv": "right backhand serve",
            # Loop
            "lfl": "left forehand loop",
            "rfl": "right forehand loop",
            "lbl": "left backhand loop",
            "rbl": "right backhand loop",
            # Short
            "lfsh": "left forehand short",
            "rfsh": "right forehand short",
            "lbsh": "left backhand short",
            "rbsh": "right backhand short",
            # Block
            "lfb": "left forehand block",
            "rfb": "right forehand block",
            "lbb": "left backhand block",
            "rbb": "right backhand block",
            # Push
            "lfp": "left forehand push",
            "rfp": "right forehand push",
            "lbp": "left backhand push",
            "rbp": "right backhand push",
            # Flick
            "lff": "left forehand flick",
            "rff": "right forehand flick",
            "lbf": "left backhand flick",
            "rbf": "right backhand flick",
            # Smash
            "lfs": "left forehand smash",
            "rfs": "right forehand smash",
            "lbs": "left backhand smash",
            "rbs": "right backhand smash",
            # Lob
            "lflo": "left forehand lob",
            "rflo": "right forehand lob",
            "lblo": "left backhand lob",
            "rblo": "right backhand lob",
        }
        new_label = label_map.get(new_label, new_label)

        event_frames[frame_num] = new_label  # Update the label

        # Save immediately to prevent data loss
        with open(json_path, "w") as f:
            json.dump(event_frames, f, indent=4)

        print(f"Label updated and saved for frame {frame_num}.")
        print(f"New label for {frame_num} is {new_label}.")
        print("------")

cap.release()
cv2.destroyAllWindows()
print("Annotation process completed.")
