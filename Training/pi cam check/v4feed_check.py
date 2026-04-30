import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(r'C:\YOLO\ECHOSPOT_YOLO_MODELS\ECHOSPOTv4\runs\detect\train\weights\best.pt')

# Start webcam feed (use 0 or your camera index)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference (returns results object)
    results = model.predict(source=frame, conf=0.25, show=False)

    # Display results (draw boxes)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv12 webcam detection', annotated_frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
