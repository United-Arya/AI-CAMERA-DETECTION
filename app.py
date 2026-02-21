import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("AI Object Detection System")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)

    res_plotted = results[0].plot()

    st.image(res_plotted, caption="Detected Image", use_column_width=True)
#paste command
# streamlit run app.py
