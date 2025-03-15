import torch
import cv2
import numpy as np
import pyttsx3
import time

# ✅ Load YOLOv5-Nano Model (FP16 for Speed)
device = "cpu"  # Orange Pi lacks CUDA
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5n.pt").to(device)
model.half()  # Use FP16 for performance

# ✅ Initialize Text-to-Speech (TTS)
engine = pyttsx3.init()

# ✅ Open Webcam (Lower FPS & Resolution)
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Reduce width
cap.set(4, 240)  # Reduce height
cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS

last_announced_object = None
y_pressed_time = None
y_hold_duration = 2  # Reduce hold time

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Convert Frame to Torch Tensor
    results = model(frame, size=320)  # Resize input to 320x320

    # ✅ Extract Closest Object
    detected_objects = []
    min_distance = float("inf")
    closest_object = None
    h, w = frame.shape[:2]

    for *xyxy, conf, cls in results.xyxy[0]:  # Extract boxes
        obj_name = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, xyxy)

        # Compute Distance to Frame Center
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2
        distance = np.sqrt((obj_center_x - 160) ** 2 + (obj_center_y - 120) ** 2)

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
