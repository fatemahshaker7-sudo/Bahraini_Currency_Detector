import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
from PIL import Image, ImageOps

model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))
st.title("Bahraini Currency Detector")
mode = st.radio("Choose input", ["Live camera", "Upload photo"], horizontal=True)

if mode == "Live camera":
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)    # prediction
        annotated_frame = results[0].plot()    # draw boxes
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    webrtc_streamer(key="currency", video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},)
else:
    uploaded_photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded_photo:
        image = ImageOps.exif_transpose(Image.open(uploaded_photo)).convert("RGB")
        results = model(image)
        st.image(results[0].plot()[:, :, ::-1])  # BGR -> RGB
