import tempfile

import streamlit as st
import cv2
import numpy as np

from PIL import Image

from models.yolo_model import load_model
from utils.detector import run_detection

st.set_page_config(
    page_title="Smart PPE Monitoring System",
    layout="wide"
)

st.title(" Smart PPE Monitoring System")

st.markdown(
    """
    Real-time PPE Detection using YOLOv8 and OpenCV
    """
)

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

mode = st.sidebar.selectbox(
    "Select Input Type",
    ["Image Upload", "Video Upload"]
)


model = load_model()

if mode == "Image Upload":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        detected_image = run_detection(
            model,
            image_np,
            confidence
        )

        st.subheader("Detection Result")

        st.image(
            detected_image,
            channels="RGB",
            use_container_width=True
        )   

elif mode == "Video Upload":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            detected_frame = run_detection(
                model,
                frame,
                confidence
            )

            stframe.image(
                detected_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()