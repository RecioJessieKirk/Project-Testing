import pytesseract
from PIL import Image

img = Image.open("test_image_3.jpg")
text = pytesseract.image_to_string(img)
print(text)
