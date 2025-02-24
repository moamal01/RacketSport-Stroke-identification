from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import cv2
import csv
import json

# Load the object detection model
cfg_det = get_cfg()
cfg_det.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_det.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg_det.MODEL.DEVICE = "cpu"
detector = DefaultPredictor(cfg_det)

# Load the keypoint detection model
cfg_kp = get_cfg()
cfg_kp.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg_kp.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg_kp.MODEL.DEVICE = "cpu"
keypoint_detector = DefaultPredictor(cfg_kp)

video_path = "videos/game_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get first frame
with open("notebooks/empty_event_keys2.json", "r") as keypoint_file:
    loaded_keys = json.load(keypoint_file)

first_pair = list(loaded_keys.items())[0]
key_frame = first_pair[0]
value_frame = first_pair[1]

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

output_path = "test6.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# CSV
keypoint_filename = "keypoints1.csv"
bbox_filename = "bbox1.csv"

start_frame = 2295
end_frame = 2315
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Thesholds
min_person_area = 30000
min_table_area = 400000

frame_number = start_frame

with open(keypoint_filename, mode="w", newline="") as keypoint_file, open(bbox_filename, mode="w", newline="") as bbox_file:
    keypoint_writer = csv.writer(keypoint_file)
    bbox_writer = csv.writer(bbox_file)
    
    keypoint_writer.writerow(["Path", "Stroke", "Event frame", "Sequence frame", "Player_1 keypoints", "Player_2 keypoints"])
    bbox_writer.writerow(["Frame", "Class ID", "X1", "Y1", "X2", "Y2"])

    try:
        while cap.isOpened() and frame_number <= end_frame:
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

            for kp in keypoint_instances.pred_keypoints:
                if kp[0, 0] < 700:
                    player_1_keypoints.append(kp)
                elif kp[0, 0] > 1100:
                    player_2_keypoints.append(kp)

            # Ensure that they are not empty
            default_shape = (1, 17, 3)

            if player_1_keypoints:
                player_1_keypoints = torch.stack(player_1_keypoints)
            else:
                player_1_keypoints = torch.zeros(default_shape)

            if player_2_keypoints:
                player_2_keypoints = torch.stack(player_2_keypoints)
            else:
                player_2_keypoints = torch.zeros(default_shape)

            # Save keypoints
            keypoint_writer.writerow([video_path, value_frame, key_frame, frame_number, player_1_keypoints[0].tolist(), player_2_keypoints[0].tolist()])

            class_filter = torch.tensor([32, 38, 60])  # Allowed class IDs
            
            box_areas = (instances.pred_boxes.tensor[:, 2] - instances.pred_boxes.tensor[:, 0]) * \
                (instances.pred_boxes.tensor[:, 3] - instances.pred_boxes.tensor[:, 1])
            
            mask = torch.isin(instances.pred_classes, class_filter)
            correct_instances = instances[mask]
            
            # Save objects
            for i in range(len(correct_instances)):
                box = correct_instances.pred_boxes.tensor[i]
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = correct_instances.pred_classes[i].item()
                bbox_writer.writerow([frame_number, class_id, x1, y1, x2, y2])
            
            # Person filter
            spectator_mask = (instances.pred_classes == 0) & (box_areas < min_person_area)
            mask = mask & ~spectator_mask
            
            # Table filter
            table_mask = (instances.pred_classes == 60) & (box_areas < min_table_area)
            mask = mask & ~table_mask

            # Apply the mask to filter instances
            filtered_instances = instances[mask]
            
            # Save crop of people
            for i in range(len(filtered_instances)):
                if filtered_instances.pred_classes[i].item() == 0:
                    box = filtered_instances.pred_boxes.tensor[i]
                    x1, y1, x2, y2 = map(int, box.tolist())  # Convert tensor to list and cast to int
                    
                    # Ensure bounding box is within frame dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # Crop the region from the frame
                    cropped_img = frame[y1:y2, x1:x2]

                    # Save the cropped image
                    if cropped_img.size > 0:  # Ensure it's not empty
                        filename = f"cropped/frame_{frame_number}_box_{i}.jpg"
                        cv2.imwrite(filename, cropped_img)

            
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
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video processing complete.")
        print(f"Keypoints saved to {keypoint_filename}")