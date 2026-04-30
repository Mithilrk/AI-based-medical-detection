#!/usr/bin/env python3
import cv2
import time
import threading

# import model and IO libs
from ultralytics import YOLO
import easyocr
import pyttsx3

# GPIO - import may fail if run inside venv; run with system python if so
import RPi.GPIO as GPIO

# ---------------- CONFIG ----------------
MODEL_PATH = r"/home/gomma/Downloads/Echo Sight Model/ECHOSPOTv4 - Copy/runs/detect/train/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.2
CAMERA_DEVICE = 0  # USB camera index

BUTTON_PIN = 17   # physical BCM pin for the push button
LED_PIN = 27      # optional LED for visual feedback (set to None to disable)

# keywords (same as your list; adjust as needed)
MEDICINE_KEYWORDS = ['Amantrel','B-9 plus','Livogen','Mefenamic-Acid-HCL-Tablets',
                     'Sitagliptin-Phosphate','Zinc-Sulphate-Dispersible-Tablets',
                     'Zytee-L','alkem','amoxycilin-625','amoxycillin','calpol-500',
                     'calpol-625','cilnidipine','cipcal','cipcal-500','cipla ROKO',
                     'cofsils','coldact','digene','disprin','dolo-65','dolo-650',
                     'evfon','gemer sita ir 50-500-1','hifenac-p','iron-tonic',
                     'livogen-tablet','medicine','mefarc spas','metrogyl-DG-Gel',
                     'neomycin','orafast','orofer xt plus','oseltamvir','paracetamol',
                     'ranitidine','rapiclav','rifaximin','rifaximin tablets-400','syrup']

CURRENCY_KEYWORDS = ['10','20','100','200','500']
CHOCOLATE_KEYWORDS = ['chupa-chups','dairy-Milk','milky-bar','munch']

# ---------------- INIT ----------------
print("Loading model and OCR (this can take a while)...")
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])
tts = pyttsx3.init(driverName='espeak')
tts.setProperty('rate', 150)

cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
if LED_PIN is not None:
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)

print("Ready. Camera preview open. Press the button to capture an image and detect objects.")
print("Press 'q' in the preview window to quit.")

# A small helper to avoid blocking the main preview while doing model inference+TTS
def handle_capture(frame_copy):
    # flash LED quickly to show we captured (if LED configured)
    if LED_PIN is not None:
        GPIO.output(LED_PIN, GPIO.HIGH)
    try:
        # run model on the captured frame
        results = model(frame_copy)  # model inference
        output_frame = frame_copy.copy()

        any_speech = []

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
                crop = frame_copy[y1:y2, x1:x2]
                ocr_result = reader.readtext(crop)
                text = " ".join([res[1] for res in ocr_result])
                

                # Draw bounding box for saved image feedback
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, class_name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow("output", output_frame)
                cv2.waitKey(1)

                if class_name in MEDICINE_KEYWORDS:
                    category = "medicine"
                elif class_name in CURRENCY_KEYWORDS:
                    category = "currency"
                elif class_name in CHOCOLATE_KEYWORDS:
                    category = "chocolate"
                else:
                    category = "object"

                speech_text = f"{category} detected: {class_name}"
                if text.strip():
                    speech_text += f", text reads: {text}"

                print(speech_text)
                any_speech.append(speech_text)

        # Save image with boxes (if any results)
        timestamp = int(time.time())
        out_path = f"./output/capture_{timestamp}.jpg"
        cv2.imwrite(out_path, output_frame)
        print(f"Saved capture to {out_path}")

        # speak found texts (joined)
        if any_speech:
            tts_text = " . ".join(any_speech)
        else:
            tts_text = "No confident objects detected."
            print(tts_text)

        tts.say(tts_text)
        tts.runAndWait()

    except Exception as e:
        print("Error in capture handler:", e)

    finally:
        if LED_PIN is not None:
            GPIO.output(LED_PIN, GPIO.LOW)


# main loop with camera preview
try:
    last_pressed = 0
    DEBOUNCE_SECS = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        # show preview
        cv2.imshow("Camera Preview - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # check button press (active low)
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            now = time.time()
            if now - last_pressed > DEBOUNCE_SECS:
                print("Button pressed — capturing...")
                last_pressed = now

                # copy frame and start background thread for inference + TTS
                frame_copy = frame.copy()
                t = threading.Thread(target=handle_capture, args=(frame_copy,), daemon=True)
                t.start()

            # wait for release briefly to avoid flood
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.05)

except KeyboardInterrupt:
    print("User requested exit")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Clean exit.")
