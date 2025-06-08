import cv2

# File paths
video_path = "../../videos/game_4.mp4"

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5335)
ret, frame = cap.read()

HEIGHT = 1080
WIDTH = 1920
ball_midpoint = [0.8476895014444987, 0.6694981892903646]
actual_ball = [0.299, 0.582]
    
cv2.circle(frame, (int(ball_midpoint[0] * WIDTH), int(ball_midpoint[1] * HEIGHT)), radius=2, color=(0, 0, 255), thickness=-1)
cv2.circle(frame, (int(actual_ball[0] * WIDTH), int(actual_ball[1] * HEIGHT)), radius=10, color=(0, 255, 0), thickness=2)
cv2.imshow(f"Event Preview (Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))})", frame)
cv2.waitKey(0)
