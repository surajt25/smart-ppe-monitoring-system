import streamlit as st

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

st.write("Selected Mode :", mode)
st.write("Confidence Threshold :", confidence)