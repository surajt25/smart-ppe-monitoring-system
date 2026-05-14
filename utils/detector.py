import cv2

def run_detection(model, frame, conf_threshold):

    results = model(frame)

    annotated_frame = frame.copy()

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return annotated_frame