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

st.title("Smart PPE Monitoring System")

st.markdown(
    """
     Detect helmets, safety vests, and masks using a custom-trained YOLOv8 model.
    """
)

st.divider()
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown(
    """
    Configure detection parameters.
    """
)

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

        with st.spinner("Running PPE detection..."):

            detected_image, detection_counts = run_detection(
                model,
                image_np,
                confidence
            )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Detection Result")

            st.image(
                detected_image,
                channels="RGB",
                use_container_width=True
            )

        st.success("Detection completed successfully.")
        st.subheader("Detection Statistics")

        stats_cols = st.columns(3)

        for idx, (label, count) in enumerate(detection_counts.items()):

            with stats_cols[idx]:

                st.metric(
                    label=label.capitalize(),
                    value=count
                )

elif mode == "Video Upload":
    st.subheader("📹 PPE Video Detection")

    uploaded_video = st.file_uploader(
        "Upload PPE Monitoring Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_video.read())

        status_message = st.empty()

        status_message.info("Initializing video processing...")

        cap = cv2.VideoCapture(tfile.name)

        status_message.empty()

        stframe = st.empty()
        
        video_info = st.empty()
        video_info.info(
            "Video processing may take a few moments depending on video size."
        )
        

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            detected_frame, detection_counts = run_detection(
                model,
                frame,
                confidence
            )

            stframe.image(
                detected_frame,
                channels="BGR",
                width="stretch"
            )

        cap.release()
        video_info.empty()  
        st.success("Video processing completed successfully.")