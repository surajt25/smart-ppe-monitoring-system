import cv2

def run_detection(model, frame, conf_threshold):

    frame = cv2.resize(frame, (640, 480))
    
    results = model(frame)

    annotated_frame = frame.copy()
    
    detection_counts = {}

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]
            
            if class_name not in detection_counts:
                detection_counts[class_name] = 0

            detection_counts[class_name] += 1

            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
    return annotated_frame, detection_counts