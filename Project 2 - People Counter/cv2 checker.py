# import cv2
#
# print("OpenCV version:", cv2.__version__)
# print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
# # cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D CUDA_ARCH_BIN=<Your_CUDA_Architecture> ..
import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

# RTSP Stream URL
rtsp_url = "rtsp://192.168.137.204:8080/h264_ulaw.sdp"
cap = cv2.VideoCapture(rtsp_url)



# cap = cv2.VideoCapture("../Videos/drill.mp4")  # For Video
# cap = cv2.VideoCapture(0)  # For Video

# cap = cv2.VideoCapture(rtsp_url)  # Capture from RTSP stream
# cap = cv2.VideoCapture(rtsp_url) if not use_cuda else cv2.cuda.VideoCapture(rtsp_url)
model = YOLO("../Yolo-Weights/yolov8n.pt")

# FPS Calculation
fps_counter = cv2.getTickCount()

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("square.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Vertical Line Position
vertical_line_x = 140  # Adjust this value based on your video resolution
# limitsUp = [700, 520, 800, 1080]

# # limitsUp = [103, 161, 296, 161]
# limitsDown = [0, 0, 0, 0]

# Initialization
totalCountUp = 0
totalCountDown = 0
lastIdUp = -1
lastIdDown = -1
while True:
    success, img = cap.read()

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count Logic
        # if cx < vertical_line_x and totalCountUp < id:
        #     totalCountUp = id
        #     cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), (0, 255, 0), 5)
        #
        # if cx > vertical_line_x and totalCountDown < id:
        #     totalCountDown = id
        #     cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), (0, 255, 0), 5)

    if cx < vertical_line_x and lastIdUp != id:
        lastIdUp = id
        totalCountUp += 1
        cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), (0, 255, 0), 5)

    if cx > vertical_line_x and lastIdDown != id:
        lastIdDown = id
        totalCountDown += 1
        cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), (0, 255, 0), 5)

    # cv2.putText(img, f'Count Up: {totalCountUp}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # cv2.putText(img, f'Count Down: {totalCountDown}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.putText(img, f'Count Up: {totalCountUp}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, f'Count Down: {totalCountDown}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    current_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_time - fps_counter)
    fps_counter = current_time

    # Display FPS
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()