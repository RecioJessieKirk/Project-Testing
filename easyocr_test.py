import cv2
import easyocr
import numpy as np
import re

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], model_storage_directory='ocr_model')  # English language model

# Function to preprocess image (denoise, enhance contrast)
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image at '{image_path}'.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply adaptive thresholding for better text detection
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Denoise to remove small artifacts
    processed = cv2.fastNlMeansDenoising(processed, h=30)

    return processed

# Image file path
image_path = "testimages/test_image_6.jpg"  # Change this to your image file

try:
    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Perform OCR with optimized settings
    text_results = reader.readtext(processed_image, detail=0)

    if not text_results:
        print("⚠ No text detected. Try adjusting preprocessing settings.")
    else:
        # Print extracted text without modification
        raw_text = " ".join(text_results)
        print(f"\n📝 Detected Text:\n{raw_text}\n")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")