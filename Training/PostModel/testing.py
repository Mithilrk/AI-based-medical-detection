# EchoSight Prototype: YOLOv12 + OCR + TTS
# Works on PC; can be adapted for Raspberry Pi hardware later

import cv2
from ultralytics import YOLO
import easyocr
import pyttsx3
import time
import os

# ---------------- CONFIG ----------------
YOLO_MODEL_PATH = "D:/Anindita_documents/Echo Sight Model/ECHOSPOTv4 - Copy/runs/detect/train/weights/best.pt"
TEST_IMAGES_FOLDER = "D:/Anindita_documents/Echo Sight Model/ECHOSPOTv4 - Copy/test/images"
SAVE_OUTPUT = True
OUTPUT_FOLDER = "D:/Anindita_documents/Echo Sight Model/ECHOSPOTv4 - Copy/runs/detect/predict_with_ocr"

# ---------------- INIT ----------------
model = YOLO(YOLO_MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])
tts_engine = pyttsx3.init()

if SAVE_OUTPUT and not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ---------------- HELPER FUNCTION ----------------
def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def process_image(img_path):
    """Run YOLO, OCR, and TTS on a single image."""
    img = cv2.imread(img_path)
    results = model.predict(source=img_path)

    # Loop through detected boxes
    for r in results:
        if len(r.boxes) == 0:
            continue
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            
            # Crop the detected object
            roi = img[y1:y2, x1:x2]

            # OCR
            ocr_result = ocr_reader.readtext(roi)
            detected_text = " ".join([text for (_, text, _) in ocr_result]) or label

            print(f"Detected {label}: {detected_text}")
            
            # Draw bounding box + label on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, detected_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak detected text
            speak(detected_text)

    # Save output image
    if SAVE_OUTPUT:
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, base_name), img)

# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    # Simulate button press for each test image
    for img_file in os.listdir(TEST_IMAGES_FOLDER):
        img_path = os.path.join(TEST_IMAGES_FOLDER, img_file)
        print(f"\nProcessing {img_file}...")
        process_image(img_path)
        time.sleep(1)  # Optional delay between images
