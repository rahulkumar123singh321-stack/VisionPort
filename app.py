
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# Page configuration
st.set_page_config(
    page_title="VisionPort",
    page_icon="🚜",
    layout="wide",
)

# Application title
st.title("🚦 VisionPort v0.1")

st.write("GPU-Accelerated Traffic Object Detection")

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on: **{device.upper()}**")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# Upload traffic image
uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")

    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )

    # Perform object detection
    results = model.predict(
        source=image,
        device=device
    )

    result_image = results[0].plot()

    st.subheader("Detection Results")

    st.image(
        result_image,
        caption="Detected Objects",
        channels="BGR",
        use_container_width=True
    )

    # Display detection count
    st.success(
        f"Total Objects Detected: {len(results[0].boxes)}"
    )
