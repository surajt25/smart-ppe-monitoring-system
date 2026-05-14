def check_violations(results):
    detected_classes = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            detected_classes.append(class_id)

    return detected_classes