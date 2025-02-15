import cv2
import numpy as np

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Read the first frame
_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Thresholds
THRESHOLD = 10  # Motion level threshold for optical flow
MOTION_THRESHOLD = 5000  # Pixel change threshold for frame differences
VERTICAL_THRESHOLD = 0.2  # Threshold for vertical downward motion

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Calculate motion in the vertical direction
    vertical_movement = np.mean(np.sin(ang))  # Vertical component of flow

    # Check for downward motion
    if vertical_movement < -VERTICAL_THRESHOLD:
        print("Potential fall detected based on downward motion!")

    # Detect abrupt motion (potential fall)
    motion_level = np.mean(mag)
    if motion_level > THRESHOLD:
        print("Potential fall detected based on motion level!")

    # Calculate Frame Differences
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    motion_pixels = cv2.countNonZero(thresh)

    # Detect significant scene change
    if motion_pixels > MOTION_THRESHOLD:
        print("Potential fall detected based on scene change!")

    # Update previous frame
    prev_gray = gray

    # Show the frame (for debugging or visualization)
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
