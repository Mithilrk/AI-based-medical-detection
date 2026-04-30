from ultralytics import YOLO
import cv2
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo12m.pt")
    model.to("cuda")
    results = model.train(
        data=r"C:\YOLO\ECHOSPOT_YOLO_MODELS\ECHOSPOTv4\data.yaml",
        epochs=80,          
        imgsz=768,          
        batch=-1,            
        workers=2,
        optimizer="AdamW",
        patience=20,         
        warmup_epochs=3.0,   
        lr0=0.001,
        weight_decay=0.0005,
        close_mosaic=30,
        dropout=0.1,
        verbose=True
    )

if __name__ == "__main__":
    main()