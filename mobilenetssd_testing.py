import cv2
import numpy as np
import pyttsx3
import time

# ✅ Load MobileNet-SSD Model
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ✅ Initialize Object Labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# ✅ Initialize Text-to-Speech (TTS)
engine = pyttsx3.init()
last_announced_object = None

# ✅ Open Webcam (Optimized for Orange Pi Zero 2 W)
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Reduce width
cap.set(4, 240)  # Reduce height
cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce CPU usage

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Convert Frame to Blob (Optimized for MobileNet SSD)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # ✅ Find the Closest Object
    h, w = frame.shape[:2]
    closest_object = None
    min_distance = float("inf")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            obj_name = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Compute distance to center
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            distance = np.sqrt((obj_center_x - w // 2) ** 2 + (obj_center_y - h // 2) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_object = obj_name

    # ✅ Announce Closest Object
    if closest_object and closest_object != last_announced_object:
        print(f"Announcing: {closest_object}")
        engine.say(closest_object)
        engine.runAndWait()
        last_announced_object = closest_object

    # ✅ Measure Inference Time
    elapsed_time = time.time() - start_time
    print(f"Inference Time: {elapsed_time:.3f} sec ({1 / elapsed_time:.1f} FPS)")

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
