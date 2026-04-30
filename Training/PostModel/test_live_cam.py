import cv2
import numpy as np
import easyocr
import pyttsx3
from ultralytics import YOLO
import os
import tkinter as tk
from PIL import Image, ImageTk
from difflib import get_close_matches
import threading

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\Anindita_documents\Echo Sight Model\ECHOSPOTv4 - Copy\runs\detect\train\weights\best.pt"
SAVE_OUTPUT = False
OUTPUT_FOLDER = r"D:\Anindita_documents\Echo Sight Model\ECHOSPOTv4 - Copy\PostModel\output"

# Keywords / class mapping
MEDICINE_KEYWORDS = [
    'medicine', 'paracetamol', 'cipcal', 'dolo-650', 'livogen', 'amoxycilin-625', 'ranitidine', 'eye-drops',
    'neomycin', 'oseltamvir', 'cofsils', 'metrogyl-dg-gel', 'mefenamic-acid-hcl-tablets', 'sitagliptin-phosphate',
]
CHOCOLATE_KEYWORDS = ['chupa-chups','dairy-milk']
CURRENCY_VALUES = ['r10', 'r20', 'r100', 'r200', 'r500']

# ---------------- INIT OCR + TTS ----------------
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(img):
    img = cv2.resize(img, (768, 768))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_medicine(ocr_text):
    """Return closest matching medicine from OCR text"""
    ocr_text = ocr_text.lower()
    matches = get_close_matches(ocr_text, MEDICINE_KEYWORDS, n=1, cutoff=0.6)
    return matches[0] if matches else None

def detect_category(roi, class_name):
    """Determine category and value/class from ROI"""
    rotations = [0, 90, 180, 270]
    currency_value = None
    ocr_text_full = ""

    for angle in rotations:
        if angle != 0:
            roi_rot = cv2.rotate(roi, {90:cv2.ROTATE_90_CLOCKWISE,
                                       180:cv2.ROTATE_180,
                                       270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        else:
            roi_rot = roi

        ocr_results = reader.readtext(roi_rot)
        ocr_text = " ".join([res[1] for res in ocr_results]).lower()
        ocr_text_full += " " + ocr_text

        # Check for currency keywords first
        for val in CURRENCY_VALUES:
            if val in ocr_text:
                currency_value = val.upper()
                break
        if "reserve bank" in ocr_text or "rbi" in ocr_text:
            currency_value = "UNKNOWN VALUE"
        if currency_value:
            return "currency", currency_value, ocr_text_full.strip()

    # Medicine detection: YOLO class first
    if class_name.lower() in MEDICINE_KEYWORDS:
        return "medicine", class_name, ocr_text_full.strip()
    # Fuzzy OCR medicine detection
    med_match = detect_medicine(ocr_text_full)
    if med_match:
        return "medicine", med_match, ocr_text_full.strip()

    # Chocolate detection
    if any(k in ocr_text_full for k in CHOCOLATE_KEYWORDS) or class_name.lower() in CHOCOLATE_KEYWORDS:
        return "chocolate", class_name, ocr_text_full.strip()

    # Default
    return "object", class_name, ocr_text_full.strip()

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- GUI ----------------
window = tk.Tk()
window.title("Echo Sight - Live Detection")
canvas_width, canvas_height = 640, 480
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()
current_frame = None

# ---------------- DETECTION FUNCTION ----------------
def detect_frame():
    global current_frame
    if current_frame is None:
        return

    input_img = preprocess_image(current_frame)
    results = model.predict(input_img, verbose=False)

    output_img = current_frame.copy()
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, classes):
            class_name = model.names[cls]
            x1, y1, x2, y2 = box.astype(int)
            roi = current_frame[y1:y2, x1:x2]

            category, value_or_class, ocr_text = detect_category(roi, class_name)

            # Speak
            if category == "currency":
                speak(f"Currency detected: {value_or_class}")
            else:
                speak(f"{category} detected: {value_or_class}. OCR reads: {ocr_text}" if ocr_text.strip() else f"{category} detected: {value_or_class}")

            print(f"{category} detected: {class_name}, OCR reads: {ocr_text}")

            # Draw box + label
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if SAVE_OUTPUT:
        output_path = os.path.join(OUTPUT_FOLDER, "live_output.jpg")
        cv2.imwrite(output_path, output_img)

    # Show frame
    img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

# ---------------- THREADING ----------------
def detect_frame_thread():
    btn_detect.config(state=tk.DISABLED)
    try:
        detect_frame()
    finally:
        btn_detect.config(state=tk.NORMAL)

# ---------------- VIDEO LOOP ----------------
def update_frame():
    global current_frame
    ret, frame = cap.read()
    if ret:
        current_frame = frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        canvas.img_tk = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    window.after(10, update_frame)

# ---------------- BUTTON ----------------
btn_detect = tk.Button(window, text="Detect Objects", command=lambda: threading.Thread(target=detect_frame_thread).start())
btn_detect.pack()

# ---------------- CLOSE HANDLER ----------------
def on_closing():
    cap.release()
    engine.stop()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
window.mainloop()
