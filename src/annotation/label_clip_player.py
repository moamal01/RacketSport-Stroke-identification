import cv2
import json

video_number = 1
LABEL_TO_SHOW = "left_net"      # Label to view
PRECEEDING_FRAMES = 30          # Number of frames before the label
EXCEEDING_FRAMES = 5            # Number of frames after the label

# === Load data ===
with open(f"../../data/events/events_markup{video_number}.json", "r") as f:
    frame_labels = json.load(f)

frame_labels = {int(k): v for k, v in frame_labels.items()}
frames_with_label = sorted([frame for frame, label in frame_labels.items() if label == LABEL_TO_SHOW])

cap = cv2.VideoCapture(f"../../videos/game_{video_number}f.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

current_index = 0

# === Show clips ===
def play_clip(frame_number):
    start_frame = max(0, frame_number - PRECEEDING_FRAMES)
    end_frame = min(total_frames - 1, frame_number + EXCEEDING_FRAMES)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Player", frame)
        cv2.waitKey(int(1000 / fps))


def show_clip(index):
    current_index = index
    frame_number = frames_with_label[current_index]
    print(f"Showing clip around frame {frame_number}")

    play_clip(frame_number)

    key = cv2.waitKey(0)

    # d or right arrow
    if key == ord('d'):
        if current_index < len(frames_with_label) - 1:
            show_clip(current_index + 1)
        else:
            close_viewer()
            print("No more clips.")

    # a or left arrow
    elif key == ord('a'):
        if current_index > 0:
            show_clip(current_index - 1)
        else:
            close_viewer()
            print("No more clips.")

    # ESC
    elif key == 27:
        close_viewer()
        print("Viewer closed.")


def close_viewer():
    cap.release()
    cv2.destroyAllWindows()


if frames_with_label:
    show_clip(0)
else:
    print(f"No frames found with label '{LABEL_TO_SHOW}'")
    close_viewer()