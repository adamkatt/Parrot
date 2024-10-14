import torch
import cv2

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection using YOLOv5
    results = model(frame)
    
    # Render the detection results on the frame
    frame_with_detections = results.render()[0]

    # Display the frame with detections
    cv2.imshow('YOLOv5 Object Detection', frame_with_detections)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
