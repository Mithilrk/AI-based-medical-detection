import cv2
import numpy as np
import easyocr
import pyttsx3
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\Anindita_documents\Echo Sight Model\ECHOSPOTv4 - Copy\runs\detect\train\weights\best.pt"
IMAGE_PATH = r"D:\Anindita_documents\Echo Sight Model\ECHOSPOTv4 - Copy\test\images\IMG-20251115-WA0039_jpg.rf.cb1d6714c2295c6525a9a565d13d4034.jpg"
OUTPUT_FOLDER = r"D:\Anindita_documents\Echo Sight Model\ECHOSPOTv4 - Copy\PostModel\output"
SAVE_OUTPUT = True

# Keywords / class mapping
MEDICINE_KEYWORDS = [
    'medicine', 'paracetamol', 'cipcal', 'dolo-650', 'livogen', 'amoxycilin-625', 'ranitidine', 'eye-drops',
    'neomycin', 'oseltamvir', 'cofsils', 'metrogyl-DG-Gel', 'Mefenamic-Acid-HCL-Tablets', 'Sitagliptin-Phosphate',
    # Add all other medicine classes here
]
CURRENCY_KEYWORDS = ['10','20','100','200','500']
CHOCOLATE_KEYWORDS = ['chupa-chups','dairy-Milk']

# Initialize OCR and TTS
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # slower for clarity
engine.setProperty('volume', 1.0)

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(img):
    # Resize if needed
    img = cv2.resize(img, (768, 768))
    # Convert to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
input_img = preprocess_image(img)

# ---------------- DETECTION ----------------
results = model.predict(input_img)

output_img = img.copy()
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    for box, cls in zip(boxes, classes):
        class_name = model.names[cls]
        x1, y1, x2, y2 = box.astype(int)
        roi = img[y1:y2, x1:x2]

        # OCR only on detected region
        ocr_results = reader.readtext(roi)
        text = " ".join([res[1] for res in ocr_results])

        # Determine category for TTS
        if class_name in MEDICINE_KEYWORDS:
            category = "medicine"
        elif class_name in CURRENCY_KEYWORDS:
            category = "currency"
        elif class_name in CHOCOLATE_KEYWORDS:
            category = "chocolate"
        else:
            category = "object"

        print(f"{category} detected: {class_name}, text reads: {text}")

        # Speak detected object naturally
        if text.strip():
            speak(f"{category} detected: {class_name}. Text reads: {text}")
        else:
            speak(f"{category} detected: {class_name}")

        # Draw box + label on image
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}"
        cv2.putText(output_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ---------------- SAVE OUTPUT ----------------
if SAVE_OUTPUT:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "output_image.jpg")
    cv2.imwrite(output_path, output_img)
    print("Output image saved at:", output_path)

# ---------------- DISPLAY IMAGE ----------------
output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(output_img_rgb)
plt.axis('off')
plt.show()
