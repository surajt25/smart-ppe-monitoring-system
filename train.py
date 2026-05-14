import os
from ultralytics import YOLO

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

def main():

    # loading pre-trained model
    model = YOLO("yolov8n.pt")

    # training model 
    model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        name="ppe_detection_model"
    )

if __name__ == "__main__":
    main()