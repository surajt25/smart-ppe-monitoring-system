import cv2

from models.yolo_model import load_model
from utils.detector import run_detection

model = load_model()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output_frame = run_detection(
        model,
        frame,
        conf_threshold=0.7
    )

    cv2.imshow("PPE Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()