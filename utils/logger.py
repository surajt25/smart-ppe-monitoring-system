import cv2
import os
from datetime import datetime

def save_violation(frame):

    os.makedirs("snapshots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"snapshots/{timestamp}.jpg"

    cv2.imwrite(filename, frame)