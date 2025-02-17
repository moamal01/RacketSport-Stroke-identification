from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json
import cv2

# Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Input video
input_video_path = "/videos/game_1.mp4"
cap = cv2.VideoCapture(input_video_path)

# Output video
output_video_path = "test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter(output_video_path, fourcc)

# Event frames
with open("notebooks/empty_event_keys2.json", "r") as file:
    loaded_keys = json.load(file)

third_pair = list(loaded_keys.items())[2]
print(third_pair[0]) 

for event_frame in loaded_keys:
    current_frame = event_frame - 10
    end_frame = event_frame + 10
    
    while current_frame <= end_frame:
        current_frame += 1