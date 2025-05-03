# yolov8s.py
from ultralytics import YOLO
import streamlit as st
import cv2
import time
from datetime import datetime, timedelta
from utils import save_alert_screenshot, play_warning_sound

###################
# Image detection #
###################
def detect_people_image(image_path, model_path, people_limit):
    model = YOLO(model_path)
    image = cv2.imread(image_path)

    start_time = time.time()

    results = model.predict(image, classes=[0], conf=0.35)

    end_time = time.time()
    time_taken = end_time - start_time
    count = len(results[0].boxes.cls)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    alert_text = f"People Count: {count}"
    alert_color = (0, 0, 255) if count > people_limit else (0, 255, 0)
    cv2.putText(image, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
    
    if count > people_limit:
        cv2.putText(image, "ALERT: Too many people!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(image, f"Time: {time_taken:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return image, count, time_taken

###################
# Video detection #
###################
def detect_people_video(video_path, model_path, output_path, people_limit):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    max_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, classes=[0], conf=0.35)
        count = len(results[0].boxes.cls)
        max_count = max(max_count, count)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(frame, f"People: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

    total_time = time.time() - start_time
    return output_path, total_time, max_count

#########################
# Live camera detection #
#########################
def detect_people_camera(model_path, people_limit):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    alert_placeholder = st.empty()
    stop_button = st.button("Stop Camera", key="stop")

    MIN_BOX_HEIGHT = 100

    # Track last alert time
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = datetime.min

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture from camera.")
            break

        results = model.predict(frame, classes=[0])
        count = len(results[0].boxes.cls)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        alert_color = (0, 0, 255) if count > people_limit else (0, 255, 0)
        cv2.putText(frame, f"People: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)

        now = datetime.now()

        # Handle alert and sound
        if count > people_limit:
            cv2.putText(frame, "ALERT: Too many people!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            alert_placeholder.warning(f"🚨 ALERT: {count} people detected! Exceeds limit of {people_limit}.")
            save_alert_screenshot(frame)
        else:
            alert_placeholder.empty()

        # Sound control (runs regardless of alert showing or not)
        play_warning_sound(current_value=count, threshold=people_limit)

        stframe.image(frame, channels="BGR")

        if stop_button:
            break

    cap.release()