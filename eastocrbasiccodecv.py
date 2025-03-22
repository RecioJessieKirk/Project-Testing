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
    cv2.imshow("Press 'C' to Capture | 'Q' to Quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Image Captured! Processing text...")
        cv2.waitKey(1)  # Ensure the frame updates

        # Process the current frame
        results = reader.readtext(frame)
        extracted_text = " ".join([text for (_, text, _) in results])

        if extracted_text.strip():  # Only if text is detected
            print("Detected Text:")
            print(extracted_text)

            # Ask via TTS if the user wants the text read aloud
            engine.say("I have found a text. Press Y for Yes or N for No.")
            engine.runAndWait()

            while True:  # Wait for user input (Y/N)
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
                if key == ord('y'):
                    print("Reading text...")
                    engine.say(extracted_text)
                    engine.runAndWait()
                    break
                elif key == ord('n'):
                    print("Skipping text reading.")
                    break

        else:
            print("No text detected.")
            engine.say("No text detected. Try again.")
            engine.runAndWait()

        print("Ready for another capture. Press 'C' to capture again or 'Q' to quit.")

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
