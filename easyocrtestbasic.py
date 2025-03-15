import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], model_storage_directory='ocr_model')  # 'en' for English, add more languages if needed

# Load and analyze image
image_path = "testimages/test_image_6.jpg"  # Replace with your image file

results = reader.readtext(image_path)

# Print detected text
print("📝 Detected Text:")
for (bbox, text, prob) in results:
    print(f" - {text} (Confidence: {prob:.2f})")
