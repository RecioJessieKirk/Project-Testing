import cv2
import easyocr
import pyttsx3

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], model_storage_directory='ocr_model')
engine = pyttsx3.init()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Press 'C' to Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("📸 Image Captured!")
        cv2.imwrite("captured_image.jpg", frame)
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Process captured image
results = reader.readtext("captured_image.jpg")
extracted_text = " ".join([text for (_, text, _) in results])

print("📝 Detected Text:")
print(extracted_text)

# Ask user if they want it read aloud
response = input("I have found a text, would you like me to read it? (Yes/No): ")
if response.lower() in ["yes", "y"]:
    engine.say(extracted_text)
    engine.runAndWait()
