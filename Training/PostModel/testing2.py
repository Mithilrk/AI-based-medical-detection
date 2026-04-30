import cv2
from ultralytics import YOLO
import easyocr
import pyttsx3
import RPi.GPIO as GPIO
import time
import pytesseract


# ---------------- CONFIG ----------------
MODEL_PATH = r"/home/gomma/Downloads/Echo Sight Model/ECHOSPOTv4 - Copy/runs/detect/train/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
CAMERA_DEVICE = 0  # USB camera index

BUTTON_PIN = 17  # GPIO pin where your push button is connected

MEDICINE_KEYWORDS = [...]
CURRENCY_KEYWORDS = ['10', '20', '100', '200', '500']
CHOCOLATE_KEYWORDS = ['chupa-chups', 'dairy-Milk', 'milky-bar', 'munch']

# ---------------- TTS INIT ----------------
tts = pyttsx3.init()
tts.setProperty('rate', 150)

def speak(text):
    print("[AUDIO] >", text)
    tts.say(text)
    tts.runAndWait()

# ---------------- LOAD MODEL ----------------
try:
    model = YOLO(MODEL_PATH)
    speak("Model loaded successfully.")
except:
    speak("Error. Could not load the detection model.")
    raise

reader = easyocr.Reader(['en'])
speak("OCR system ready.")

# ---------------- CAMERA INIT ----------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
cap.set(3, 640)
cap.set(4, 480)


if not cap.isOpened():
    speak("Error. Camera not found. Check USB connection.")
    raise SystemExit

# Setup button
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

speak("System ready. Press the button to detect objects.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera frame error.")
            continue
        
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            speak("Button pressed. Capturing image.")
            output_frame = frame.copy()

            speak("Detecting objects. Please wait.")
            results = model(frame)
            
            results = model(frame)

            # --- NEW: speak raw detections ---
            for r in results:
                for cls_id in r.boxes.cls.cpu().numpy():
                    class_name = r.names[int(cls_id)]
                    speak(f"{class_name} detected.")


            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                names = result.names

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    class_name = names[int(cls_id)]

                    crop = frame[y1:y2, x1:x2]
                    speak("Reading text from detected object.")
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    text = pytesseract.image_to_string(crop_rgb).strip()

                    # Categorize object
                    if class_name in MEDICINE_KEYWORDS:
                        category = "medicine"
                    elif class_name in CURRENCY_KEYWORDS:
                        category = "currency"
                    elif class_name in CHOCOLATE_KEYWORDS:
                        category = "chocolate"
                    else:
                        category = "object"

                    # Prepare speech
                    speech_text = f"{category} detected. It is {class_name}."
                    if text.strip():
                        speech_text += f" Text reads: {text}"

                    speak(speech_text)

            # Save output
            timestamp = int(time.time())
            out_path = f"/home/gomma/Downloads/Echo Sight Model/ECHOSPOTv4 - Copy/PostModel/output/capture_{timestamp}.jpg"
            cv2.imwrite(out_path, output_frame)
            speak("Image saved successfully.")

            # Wait for button release
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)

except KeyboardInterrupt:
    speak("Shutting down.")

finally:
    cap.release()
    GPIO.cleanup()
    speak("System stopped.")
