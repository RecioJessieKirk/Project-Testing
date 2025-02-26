from ultralytics import YOLO
from PIL import Image
import cv2
model = YOLO('yolov5nu.pt')
try:
    results = model(source='/dev/video1', show=True) # This runs the detection loop internally
    while True:
        # Check if the window has been closed
        if cv2.getWindowProperty('Ultralytics YOLO', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by the user. Stopping detection.")
            break
except KeyboardInterrupt:
    print("Program interrupted manually.")
finally:
    # Ensure resources are released
    cv2.destroyAllWindows()
    print("Detection stopped, and resources cleaned up.")