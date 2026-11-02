import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import cv2

model = YOLO(r"d:\F47\GA_DSB\EXCERSIZES\Week9\bahrain_currency\best.pt")

st.title("Bahraini Currency Detector")

mode = st.radio("Choose input", ["Live camera", "Upload photo"], horizontal=True)

if mode == "Live camera":
    run = st.checkbox("Start camera")
    frame_box = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from the camera.")
            break

        results = model(frame)                 # prediction
        annotated_frame = results[0].plot()    # draw boxes

        frame_box.image(annotated_frame, channels="BGR")  # show in the app

    cap.release()

else:
    uploaded_photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    if uploaded_photo:
        image = ImageOps.exif_transpose(Image.open(uploaded_photo)).convert("RGB")
        results = model(image)
        annotated_frame = results[0].plot()
        st.image(annotated_frame[:, :, ::-1])  # BGR -> RGB
