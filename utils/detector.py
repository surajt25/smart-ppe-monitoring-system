import cv2

def run_detection(model, frame):
    results = model(frame)

    annotated_frame = results[0].plot()

    return annotated_frame, results