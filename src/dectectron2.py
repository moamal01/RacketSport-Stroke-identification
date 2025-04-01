from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import cv2
import csv
from utility_functions import load_json_with_dicts

# Load the object detection model
cfg_det = get_cfg()
cfg_det.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_det.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
cfg_det.MODEL.DEVICE = "cpu"
detector = DefaultPredictor(cfg_det)

# Load the keypoint detection model
cfg_kp = get_cfg()
cfg_kp.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg_kp.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
cfg_kp.MODEL.DEVICE = "cpu"
keypoint_detector = DefaultPredictor(cfg_kp)

video = 3
video_path = f"videos/game_{video}.mp4"
cap = cv2.VideoCapture(video_path)

# Thesholds
min_person_area = 30000
min_table_area = 400000

data = load_json_with_dicts(f"data/events/events_markup{video}.json")
    
excluded_values = {"empty_event", "bounce", "net"}
loaded_keys = {k: v for k, v in data.items() if v not in excluded_values}
    
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

output_path = "test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# CSV
keypoint_filename = f"keypoints_video{video}.csv"
bbox_filename = f"bbox_video{video}.csv"

# Main loop
with open(keypoint_filename, mode="w", newline="") as keypoint_file, open(bbox_filename, mode="w", newline="") as bbox_file:
    keypoint_writer = csv.writer(keypoint_file)
    bbox_writer = csv.writer(bbox_file)

    keypoint_writer.writerow(["Path", "Type", "Event frame", "Sequence frame", "Keypoints", "People boxes", "People scores"])
    bbox_writer.writerow(["Event frame", "Sequence frame", "Class ID", "Score", "Bboxes"])
    
    for key_frame, value_frame in loaded_keys.items():
        print(key_frame)
        start_frame = int(key_frame) - 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = 0

        try:
            while cap.isOpened() and frame_number <= 0:
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
                
                player_1_keypoints = []
                player_2_keypoints = []

                player_1_boxes = []
                player_2_boxes = []

                player_1_scores = []
                player_2_scores = []

                for kp, box, score in zip(
                    keypoint_instances.pred_keypoints, 
                    keypoint_instances.pred_boxes, 
                    keypoint_instances.scores
                ):
                    if kp[0, 0] < 700:
                        player_1_keypoints.append(kp)
                        player_1_boxes.append(box)
                        player_1_scores.append(score)
                    elif kp[0, 0] > 1100:
                        player_2_keypoints.append(kp)
                        player_2_boxes.append(box)
                        player_2_scores.append(score)

                # Ensure that lists are not empty
                default_kp_shape = (1, 17, 3)
                default_box_shape = (1, 4)
                default_score_shape = (1,)

                player_1_keypoints = torch.stack(player_1_keypoints) if player_1_keypoints else torch.zeros(default_kp_shape)
                player_1_boxes = torch.stack(player_1_boxes) if player_1_boxes else torch.zeros(default_box_shape)
                player_1_scores = torch.tensor(player_1_scores) if player_1_scores else torch.zeros(default_score_shape)

                player_2_keypoints = torch.stack(player_2_keypoints) if player_2_keypoints else torch.zeros(default_kp_shape)
                player_2_boxes = torch.stack(player_2_boxes) if player_2_boxes else torch.zeros(default_box_shape)
                player_2_scores = torch.tensor(player_2_scores) if player_2_scores else torch.zeros(default_score_shape)
        
                # Save keypoints
                keypoint_writer.writerow([video_path, value_frame, key_frame, frame_number, keypoint_instances.pred_keypoints.tolist(), keypoint_instances.pred_boxes.tensor.tolist(), keypoint_instances.scores.tolist()])

                class_filter = torch.tensor([32, 38, 60])
                
                mask = torch.isin(instances.pred_classes, class_filter)  # Create a boolean mask
                correct_instances = instances[mask]
            
                # Save objects
                for i in range(len(correct_instances)):
                    box = correct_instances.pred_boxes.tensor[i]
                    score = float(correct_instances.scores[i])

                    class_id = correct_instances.pred_classes[i].item()
                    bbox_writer.writerow([key_frame, frame_number, class_id, score, box.tolist()])
                
                # Table filter
                box_areas = (instances.pred_boxes.tensor[:, 2] - instances.pred_boxes.tensor[:, 0]) * (instances.pred_boxes.tensor[:, 3] - instances.pred_boxes.tensor[:, 1])
                table_mask = (instances.pred_classes == 60) & (box_areas < min_table_area)
                mask = mask & ~table_mask

                # Apply the mask to filter instances
                filtered_instances = instances[mask]
                
                # Visualize results
                v_det = Visualizer(frame_rgb, MetadataCatalog.get(cfg_det.DATASETS.TRAIN[0]), scale=1.2)
                vis_det = v_det.draw_instance_predictions(filtered_instances)
                
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

                frame_number += 1
        except Exception as e:
            print(f"Error during processing: {e}")


cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
print(f"Keypoints saved to {keypoint_filename}")