import easyocr
import cv2

# 1. Initialize OCR reader (English plates)
reader = easyocr.Reader(['en'])  # first run may download models

# 2. Read image (full car or just plate)
img = cv2.imread("downloads/tripwire_03df51fe-6194-47ac-ba86-ae64db78cbaa.jpeg")

# 3. Run OCR
results = reader.readtext(img, detail=1)  # detail=1 => (bbox, text, conf)

for bbox, text, conf in results:
    print("OCR:", text, "conf:", conf)
