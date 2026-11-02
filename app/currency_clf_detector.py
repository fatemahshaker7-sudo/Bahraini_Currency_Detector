import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps

model_path = os.path.join(os.path.dirname(__file__), "best.pt")

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

st.title("Bahraini Currency Detector")

camera_photo = st.camera_input("Take a photo")
uploaded_photo = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

file = camera_photo or uploaded_photo

if file:
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")

    results = model(image)
    annotated_frame = results[0].plot()

    st.image(annotated_frame[:, :, ::-1])  # BGR -> RGB
