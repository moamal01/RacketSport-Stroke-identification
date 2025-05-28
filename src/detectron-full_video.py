from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from tqdm.auto import tqdm
import torch
import cv2
import csv
import sys
import os
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")

sys.path.append(os.path.abspath('../'))

from utility_functions import load_json_with_dicts

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load the object detection model
cfg_det = get_cfg()
cfg_det.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_det.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
cfg_det.MODEL.DEVICE = device
detector = DefaultPredictor(cfg_det)

# Load the keypoint detection model
cfg_kp = get_cfg()
cfg_kp.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg_kp.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
cfg_kp.MODEL.DEVICE = device
keypoint_detector = DefaultPredictor(cfg_kp)

video = 2
start_at = 49765
video_path = f"../videos/game_{video}.mp4"
cap = cv2.VideoCapture(video_path)
efficient = True
threshold = 180

print(device)

# Visualize
write_video = False
    
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Width: {width}, Height: {height}, FPS: {fps}, Frame count: {frame_count}")

if width == 0 or height == 0 or fps == 0:
    print("Error: Could not read video properties.")
    cap.release()
    exit()

if write_video:
    output_path = f"../videos/game{video}-detections.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# CSV output files
keypoint_filename = f"../data/video_{video}/keypoints_video{video}p1p2.csv"
bbox_filename = f"../data/video_{video}/bbox_video{video}p1p2.csv"

if efficient:
    # Timestamps
    data = load_json_with_dicts(f"../data/extended_events/events_markup{video}.json")
    key_frames = sorted(int(k) for k in data.keys())

    intervals = []
    for k in key_frames:
        lo = max(0, k - threshold)
        hi = k + threshold
        if not intervals or lo > intervals[-1][1]:
            intervals.append([lo, hi])
        else:
            intervals[-1][1] = max(intervals[-1][1], hi)

    print(intervals)
    interval_idx = 0
    num_intervals = len(intervals)

    while interval_idx < num_intervals and start_at > intervals[interval_idx][1]:
        interval_idx += 1

# Main loop
with open(keypoint_filename, mode="w", newline="") as keypoint_file, open(bbox_filename, mode="w", newline="") as bbox_file:
    keypoint_writer = csv.writer(keypoint_file)
    bbox_writer = csv.writer(bbox_file)

    keypoint_writer.writerow(["Path", "Event frame", "Keypoints", "People boxes", "People scores"])
    bbox_writer.writerow(["Event frame", "Class ID", "Score", "Bboxes"])

    for frame_number in tqdm(range(frame_count), desc="Processing frames"):
        if int(frame_number) < start_at:
            continue
        
        if efficient:
            if interval_idx >= num_intervals:
                break
            
            current_start, current_end = intervals[interval_idx]
            if frame_number < current_start:
                cap.grab()
                continue
            
            if frame_number > current_end:
                interval_idx += 1
                continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"End of video or error at frame {frame_number}")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            keypoint_outputs = keypoint_detector(frame_rgb)
            outputs = detector(frame_rgb)

            instances = outputs["instances"].to("cpu")
            keypoint_instances = keypoint_outputs["instances"].to("cpu")
    
            # Save keypoints
            keypoint_writer.writerow([video_path, frame_number, keypoint_instances.pred_keypoints.tolist(), keypoint_instances.pred_boxes.tensor.tolist(), keypoint_instances.scores.tolist()])

            class_filter = torch.tensor([32, 38, 60])
            
            mask = torch.isin(instances.pred_classes, class_filter)  # Create a boolean mask
            correct_instances = instances[mask]
        
            # Save objects
            for i in range(len(correct_instances)):
                box = correct_instances.pred_boxes.tensor[i]
                score = float(correct_instances.scores[i])

                class_id = correct_instances.pred_classes[i].item()
                bbox_writer.writerow([frame_number, class_id, score, box.tolist()])
            
            # Visualize results
            if write_video:
                v_det = Visualizer(frame_rgb, MetadataCatalog.get(cfg_det.DATASETS.TRAIN[0]), scale=1.2)
                vis_det = v_det.draw_instance_predictions(correct_instances)
                
                v_kp = Visualizer(frame_rgb, MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]), scale=1.2)
                vis_kp = v_kp.draw_instance_predictions(keypoint_outputs["instances"].to("cpu"))

                # Convert back to BGR for OpenCV
                image_det = vis_det.get_image()
                image_kp = vis_kp.get_image()

                # Blend the keypoints visualization with the detection visualization
                alpha = 0.2  # Adjust transparency as needed
                blended_image = cv2.addWeighted(image_det, 1 - alpha, image_kp, alpha, 0)

                # Convert back to BGR for OpenCV
                result_frame = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)

                # Ensure correct dtype
                result_frame = result_frame.astype("uint8")

                # Resize if necessary
                if result_frame.shape[:2] != (height, width):
                    result_frame = cv2.resize(result_frame, (width, height))

                # Write frame to output video
                out.write(result_frame)

                # Display the frame
                cv2.imshow("Keypoints Detection", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("User quit.")
                    break

        except Exception as e:
            print(f"Error during processing: {e}")


cap.release()
if write_video:
    out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
print(f"Keypoints saved to {keypoint_filename}")