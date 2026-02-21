import streamlit as st
import cv2
from ultralytics import YOLO

st.title("AI Object Detection System")

model = YOLO("yolov8n.pt")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(annotated_frame)

cap.release()
#paste command
# streamlit run app.py