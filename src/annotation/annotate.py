import cv2
import json
import os

# File paths
video = 4
json_path = f"../../data/events/events_markup{video}.json"
video_path = f"../../videos/game_{video}f.mp4"
label_of_interest = "point"

# Load the event frames from JSON
with open(json_path, "r") as f:
    event_frames = json.load(f)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0  # Start at first frame

# Optionally precompute indices for special events (e.g. serves)
label_indices = [int(f) for f, lbl in event_frames.items() if label_of_interest in lbl]
current_idx = -1

while True:
    # Clamp to valid range
    current_frame = max(0, min(current_frame, frame_total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Show the frame
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            event_data = json.load(f)
    else:
        event_data = {}

    event = event_data.get(str(current_frame), None)
    label = event if event else ""
    cv2.putText(frame, f"Frame: {current_frame + 1}  Event: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Frame Navigator", frame)

    key = cv2.waitKey(0) & 0xFF  # Wait until a key is pressed

    if key == ord('p'):  # Spacebar → prompt for labeling
        print(f"Labeling frame {current_frame}...")

        point_input = input("Enter point label (shortcut or full name, Enter to skip): ").strip()
        if not point_input:
            pass  # skip if nothing entered
        else:
            point_map = {
                "e": "empty_event",
                "b": "bounce",
                # Net
                "ln": "left_net",
                "rn": "right_net",
                # Not hitting ball
                "lnb": "left_not_hitting_ball",
                "rnb": "right_not_hitting_ball",
                # Winner
                "lw": "left_winner",
                "rw": "right_winner",
                # Double bounce (Type of winner)
                "ld": "left_double_bounce",
                "rd": "right_double_bounce",
                # Out (over the net but missing the table on the other side)
                "lo": "left_out",
                "ro": "right_out",
                # Miss on own side
                "lm": "left_miss_on_own_side",
                "rm": "right_miss_on_own_side"
            }

            point_label = point_map.get(point_input, point_input)
            event_frames[str(current_frame)] = point_label

            with open(json_path, "w") as f:
                json.dump(event_frames, f, indent=4)

            print(f"✅ Label saved for frame {current_frame}: {point_label}")
            print("------")

    elif key == ord(' '):  # Stroke labeling
        print(f"Labeling frame {current_frame}...")

        stroke_input = input("Enter shot label (shortcut or full name, Enter to skip): ").strip()
        stroke_map = {
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
        stroke_label = stroke_map.get(stroke_input, stroke_input) if stroke_input else ""

        body_pos_input = input("Enter lean information (or press Enter to skip): ").strip()
        body_map = {
            "b": "back_heavy",
            "f": "front_heavy",
            "r": "right_leaning",
            "l": "left_leaning",
            "n": "neutral",
            "u": "unknown"
        }
        body_pos_label = body_map.get(body_pos_input, body_pos_input) if body_pos_input else ""

        leg_info_input = input("Enter leg information (or press Enter to skip): ").strip()
        leg_map = {
            "b": "both_feet_planted",
            "bl": "both_feet_lifted",
            "r": "right_foot_lifted",
            "l": "left_foot_lifted",
            "u": "unknown"
        }
        leg_info_label = leg_map.get(leg_info_input, leg_info_input) if leg_info_input else ""

        # Build final label parts (only non-empty)
        parts = [label for label in [stroke_label, body_pos_label, leg_info_label] if label]
        final_label = " ".join(parts)

        if final_label:  # Only save if something entered
            event_frames[str(current_frame)] = final_label
            with open(json_path, "w") as f:
                json.dump(event_frames, f, indent=4)
            print(f"✅ Label saved for frame {current_frame}: {final_label}")
        else:
            print("⚠️ No label entered, skipping.")

        print("------")


    elif key == ord('d'):  # Next frame
        current_frame += 1

    elif key == ord('a'):  # Previous frame
        current_frame -= 1

    elif key == ord('s'):  # Skip forward 10 frames
        current_frame = min(current_frame + 10, frame_total - 1)

    elif key == ord('w'):  # Skip back 10 frames
        current_frame = max(current_frame - 10, 0)

    elif key == ord('f'):  # Jump to frame before next serve
        current_idx += 1
        if current_idx < len(label_indices):
            current_frame = max(label_indices[current_idx], 0)
            print(f"Jumped to frame {current_frame} before label #{current_idx+1}")
        else:
            print("No more serve events.")

    elif key == ord('q') or key == 27:  # Quit
        break

cap.release()
cv2.destroyAllWindows()
