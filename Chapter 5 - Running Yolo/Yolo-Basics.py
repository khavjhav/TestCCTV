# from ultralytics import YOLO
# import cv2
#
#  model = YOLO('../Yolo-Weights/yolov8n.pt')
# results = model("Images/3.png", show=True)
# cv2.waitKey(0)
#
# # video_path = 'path/to/video/file.mp4'
# # cap = cv2.VideoCapture(video_path)


from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Open video file
video_path = 'cctv/cctv.mp4'
cap = cv2.VideoCapture(video_path)

# Process each frame of the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect humans in the frame
    results = model(frame)

    # Filter out only human detections
    human_detections = results.xyxy[0][results.xyxy[0][:, -1] == 0]

    # Draw bounding boxes around humans
    for detection in human_detections:
        x1, y1, x2, y2, _ = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the frame with human detections
    cv2.imshow('Human Detections', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
