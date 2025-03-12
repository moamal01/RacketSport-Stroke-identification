import cv2
import json

# File paths
json_path = "data/events/events_markup2.json"
video_path = "videos/game_2f.mp4"

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
    if "left_backhand_smash" not in label :
        continue  # Skip other events

    start_frame = max(0, int(frame_num) - 20)  # Ensure start frame is non-negative
    end_frame = min(int(frame_num) + 10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to the starting frame
        print(f"Showing frames {start_frame} to {end_frame} (Event frame: {frame_num})")
        print(label)
        
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
                replay = True
                break  # Break out of the inner loop to restart
        else:
            break  # Break out of the outer loop if no restart requested

    cv2.destroyAllWindows()  # Close the video window before labeling

    # Get user input for the new label
    new_label = input("Enter new label (or press Enter to keep current event): ").strip()

    if new_label:
        label_map = {
            "e": "empty_event",
            # Serve
            "lfsv": "left_forehand_serve",
            "rfsv": "right_forehand_serve",
            "lbsv": "left_backhand_serve",
            "rbsv": "right_backhand_serve",
            # Loop
            "lfl": "left_forehand_loop",
            "rfl": "right_forehand_loop",
            "lbl": "left_backhand_loop",
            "rbl": "right_backhand_loop",
            # Short
            "lfsh": "left_forehand_short",
            "rfsh": "right_forehand_short",
            "lbsh": "left_backhand_short",
            "rbsh": "right_backhand_short",
            # Block
            "lfb": "left_forehand_block",
            "rfb": "right_forehand_block",
            "lbb": "left_backhand_block",
            "rbb": "right_backhand_block",
            # Push
            "lfp": "left_forehand_push",
            "rfp": "right_forehand_push",
            "lbp": "left_backhand_push",
            "rbp": "right_backhand_push",
            # Flick
            "lff": "left_forehand_flick",
            "rff": "right_forehand_flick",
            "lbf": "left_backhand_flick",
            "rbf": "right_backhand_flick",
            # Smash
            "lfs": "left_forehand_smash",
            "rfs": "right_forehand_smash",
            "lbs": "left_backhand_smash",
            "rbs": "right_backhand_smash",
            # Lob
            "lflo": "left_forehand_lob",
            "rflo": "right_forehand_lob",
            "lblo": "left_backhand_lob",
            "rblo": "right_backhand_lob",
        }

        new_label = label_map.get(new_label, new_label)
        
        body_pos_label = input("Enter lean information (or press Enter to keep current event): ").strip()
        if body_pos_label:
            label_map = {
                "b": " back_heavy",
                "f": " front_heavy",
                "r": " right_leaning",
                "l": " left_leaning",
                "n": " neutral",
                "u": " unknown"
            }
        
        body_pos_label = label_map.get(body_pos_label, body_pos_label)
        
        leg_label = input("Enter leg information (or press Enter to keep current event): ").strip()
        if leg_label:
            label_map = {
                "b": " both_feet_planted",
                "bl": " both_feet_lifted",
                "r": " right_foot_lifted",
                "l": " left_foot_lifted",
                "u": " unknown"
            }
            
        leg_label = label_map.get(leg_label, leg_label)
        
        new_label = new_label + body_pos_label + leg_label

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
