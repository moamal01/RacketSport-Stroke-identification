import cv2
import json
import os

# File paths
video = 5
json_path = f"../../data/events/events_markup{video}.json"
video_path = f"../../videos/game_{video}f.mp4"
start_at = 0

if os.path.exists(json_path):
    with open(json_path, "r") as f:
        event_data = json.load(f)
else:
    event_data = {}

cap = cv2.VideoCapture(video_path)
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

current_frame = start_at

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    event = event_data.get(str(current_frame), None)

    # Draw event info if exists
    label = event if event else ""
    cv2.putText(frame, f"Frame: {current_frame}  Event: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(0)

    if key == ord(' '):  # Spacebar → add "empty_event"
        print(f"Added empty_event at frame {current_frame}")
        event_data[str(current_frame)] = "empty_event"
        with open(json_path, "w") as f:
            json.dump(event_data, f, indent=2)
        current_frame += 1
    elif key == ord('d'):  # Next frame
        current_frame += 1
    elif key == ord('a'):  # Previous frame
        current_frame = max(current_frame - 1, 0)
    elif key == ord('s'):  # Skip forward 10 frames
        current_frame = min(current_frame + 10, frame_total - 1)
    elif key == ord('q') or key == 27:  # Quit
        break

cap.release()
cv2.destroyAllWindows()
