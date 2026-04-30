from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs/detect/weights/best.pt")

# Run prediction on a folder of images
results = model.predict(source=r"D:/Anindita_documents/Echo Sight Model/ECHOSPOTv4 - Copy/test/images", save=True)

# results are saved automatically in runs/detect/predict/
