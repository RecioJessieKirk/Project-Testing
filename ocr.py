import cv2
import pytesseract
import time

# Set Tesseract path (Windows only, skip for Linux/macOS)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'U' to capture and analyze text. Press 'Q' to quit.")

last_capture_time = 0  # Prevent spamming

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Display live video feed
    cv2.imshow("Live Feed", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('u'):  # Press 'U' to capture and analyze
        current_time = time.time()

        # Prevent multiple captures within 2 seconds (Anti-spam)
        if current_time - last_capture_time > 2:
            last_capture_time = current_time
            print("\n🔍 Capturing image... Please wait.")

            time.sleep(1.5)  # Small delay for better OCR accuracy

            # Perform OCR on the original colored image
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(frame, config=custom_config).strip()

            # Display extracted text in terminal
            print(f"📝 Detected Text:\n{text}\n")

    elif key == ord('q'):  # Press 'Q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
