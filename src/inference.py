import joblib
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import statistics

# Load classifier
clf = joblib.load("logistic_regression_model.joblib") # Change model

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

# Video properties
cap = cv2.VideoCapture("testvid.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("Result.mp4", fourcc, fps, (width, height))

# Variables
keypoints = []
tables = []

# Process video

# Post processing
table_midpoints = statistics.median(tables)



#y_pred = clf.predict(X_new)
