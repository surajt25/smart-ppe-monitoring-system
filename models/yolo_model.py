from ultralytics import YOLO
from pathlib import Path

def load_model():
    #model = YOLO("yolov8n.pt")
    
    BASE_DIR = Path(__file__).resolve().parent

    model_path = BASE_DIR / "trained" / "best.pt"
    
    if not model_path.exists():
       raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    
    return model

