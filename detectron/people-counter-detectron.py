import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up video capture
cap = cv2.VideoCapture("../Videos/drill.mp4")

# Set up output video writer
output_file = "output_video.avi"
output_width, output_height = 800, 600
output_fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

# Load the Detectron2 model
config_file = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model_weights = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model = DefaultPredictor(config_file=config_file, model=model_weights)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Perform object detection
    outputs = model(frame)

    # Visualize the predictions
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get("coco_2017_train"), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_frame = v.get_image()[:, :, ::-1]

    # Count the number of people detected
    num_people = len(outputs["instances"])

    # Draw the count on the frame
    cv2.putText(vis_frame, f"People Count: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the frame for output
    vis_frame = cv2.resize(vis_frame, (output_width, output_height))

    # Write the frame to the output video file
    out.write(vis_frame)

    # Display the frame
    cv2.imshow("Frame", vis_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
