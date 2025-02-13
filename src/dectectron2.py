from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch

cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

video_path = "videos/game_1.mp4"
cap = cv2.VideoCapture(video_path)

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

start_frame = 2220
end_frame = 2222
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Thesholds
min_person_area = 30000
min_table_area = 400000

frame_number = start_frame

try:
    while cap.isOpened() and frame_number <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"End of video or error at frame {frame_number}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        outputs = predictor(frame_rgb)

        instances = outputs["instances"].to("cpu")
        class_filter = torch.tensor([0, 32, 38, 60])  # Allowed class IDs
        
        box_areas = (instances.pred_boxes.tensor[:, 2] - instances.pred_boxes.tensor[:, 0]) * \
            (instances.pred_boxes.tensor[:, 3] - instances.pred_boxes.tensor[:, 1])
        
        mask = torch.isin(instances.pred_classes, class_filter)  # Create a boolean mask
        
        # Person filter
        spectator_mask = (instances.pred_classes == 0) & (box_areas < min_person_area)
        mask = mask & ~spectator_mask
        
        # Table filter
        table_mask = (instances.pred_classes == 60) & (box_areas < min_table_area)
        mask = mask & ~table_mask

        # Apply the mask to filter instances
        filtered_instances = instances[mask]
        
        # Save crop of people
        for i, box in enumerate(filtered_instances.pred_boxes.tensor):
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
        v = Visualizer(frame_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        vis = v.draw_instance_predictions(filtered_instances)

        # Convert back to BGR for OpenCV
        result_frame = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)

        # Ensure correct dtype
        result_frame = result_frame.astype("uint8")

        # Resize if necessary
        if result_frame.shape[:2] != (height, width):
            result_frame = cv2.resize(result_frame, (width, height))

        # Write frame to output video
        out.write(result_frame)

        # Display the frame (optional)
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