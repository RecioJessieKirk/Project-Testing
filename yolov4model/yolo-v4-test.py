import cv2
import numpy as np
import time

# Load YOLOv4-Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Optimize for embedded systems (CPU/GPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_CUDA if using GPU

# Load class labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start webcam capture
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if the wrong camera is selected
cap.set(3, 320)  # Set width to 320px (for performance)
cap.set(4, 240)  # Set height to 240px

while cap.isOpened():
    start_time = cv2.getTickCount()  # Start timing
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is read

    height, width = frame.shape[:2]
    
    # Convert frame to blob (for YOLO)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Lower confidence threshold to detect more objects
                center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # Ensure indices is not empty before processing
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():  # Fix: Use .flatten() to prevent IndexError
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            detected_objects.append(label)  # Store detected objects for debugging
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate inference time (ms)
    end_time = cv2.getTickCount()
    inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # Convert to milliseconds
    fps = 1000 / inference_time  # Calculate FPS

    # Debugging: Print FPS, inference time, and detected objects
    print(f"FPS: {fps:.2f} | Inference Time: {inference_time:.2f} ms | Objects: {', '.join(detected_objects) if detected_objects else 'None'}")

    # Show frame
    cv2.imshow("YOLOv4-Tiny Webcam Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
