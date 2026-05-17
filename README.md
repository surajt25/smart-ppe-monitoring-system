# Smart PPE Monitoring System

A real-time Personal Protective Equipment (PPE) detection system built using YOLOv8, OpenCV, and Streamlit.

The application can detect safety helmets, vests, and masks from images, videos, and live webcam streams through an interactive dashboard interface.

---

## Live Demo

https://smart-ppe-monitoring-system.streamlit.app/

---

## GitHub Repository

https://github.com/surajt25/smart-ppe-monitoring-system

---

## Features

- PPE detection using a custom-trained YOLOv8 model
- Image upload detection
- Video upload detection
- Real-time webcam monitoring (local mode)
- Live detection statistics dashboard
- Adjustable confidence threshold
- Streamlit-based interactive UI
- Modular project structure
- Cloud deployment using Streamlit Community Cloud
- MLflow integration for training experiments

---

## Detection Classes

The model currently detects:

- Helmet
- Safety Vest
- Mask

---

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Pillow
- MLflow
- Git & GitHub

---

## Project Structure

```text
smart-ppe-monitoring-system/

├── .streamlit/
├── models/
│   ├── trained/
│   └── yolo_model.py
│
├── utils/
│   ├── detector.py
│   ├── logger.py
│   └── violation_checker.py
│
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
└── packages.txt
```

---

## How It Works

1. The user uploads an image or video, or starts webcam mode.
2. Frames are passed to the YOLOv8 detection model.
3. The model predicts PPE objects with confidence scores.
4. OpenCV draws bounding boxes and labels on detected objects.
5. Streamlit displays the processed output along with live detection statistics.

---

## Model Training

The PPE detection model was trained on a custom dataset downloaded from Roboflow using YOLOv8.

### Training Configuration

```python
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0
)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/surajt25/smart-ppe-monitoring-system.git

cd smart-ppe-monitoring-system
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

Planned future upgrades for the project:

1. PPE violation detection system
2. Snapshot logging for violations
3. Advanced frontend UI
4. Person tracking and analytics
5. Database integration
6. Authentication system
7. FastAPI backend integration

---

## Deployment

The application is deployed using Streamlit Community Cloud.
