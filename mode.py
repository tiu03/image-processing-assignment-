# yolov8s.py
from ultralytics import YOLO
import streamlit as st
import cv2
import time
from datetime import datetime
from sort.sort import Sort
import numpy as np
from utils import save_alert_screenshot, play_warning_sound
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_path = "models/best.pt"

###################
# Image detection #
###################
def detect_people_image(image_path, model_path, people_limit, enable_alert=True):
    model = YOLO(model_path)
    image = cv2.imread(image_path)

    start_time = time.time()

    results = model.predict(image, classes=[0], conf=0.4)

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

    if enable_alert:
        play_warning_sound(current_value=count, threshold=people_limit, duration=3)

    return image, count, time_taken


########################
# Zone image detection #
########################
# Utility to check if a point is in polygon
def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

# Detection function
def detect_people_in_zone(image_path, model_path, zone_points, people_limit, enable_alert=True):
    image = cv2.imread(image_path)
    model = YOLO(model_path)

    start_time = time.time()
    results = model.predict(image, classes=[0], conf=0.7)
    end_time = time.time()
    time_taken = end_time - start_time

    cv2.polylines(image, [np.array(zone_points, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
    people_in_zone = 0

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if is_point_in_polygon((center_x, center_y), zone_points):
            people_in_zone += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 1)

    alert_text = f"People in Zone: {people_in_zone}"
    alert_color = (0, 0, 255) if people_in_zone > people_limit else (0, 255, 0)
    cv2.putText(image, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)

    if people_in_zone > people_limit:
        cv2.putText(image, "ALERT: Too many people in the zone!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(image, f"Time: {time_taken:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    if enable_alert:
        play_warning_sound(current_value=people_in_zone, threshold=people_limit, duration=3)

    
    return image, people_in_zone, time_taken


###################
# Video detection #
###################
def detect_people_video(video_path, model_path, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize SORT tracker
    tracker = Sort(max_age=20,         # frames to keep 'lost' track alive
                   min_hits=5,         #frames before it's a valid track
                   iou_threshold=0.35) # minimum overlap for matching

    max_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, classes=[0], conf=0.4)
        detections = []

        # Convert detections to SORT format: [x1, y1, x2, y2, confidence]
        for box, score in zip(results[0].boxes.xyxy, results[0].boxes.conf):
            x1, y1, x2, y2 = box
            detections.append([float(x1), float(y1), float(x2), float(y2), float(score)])

        detections = np.array(detections)
        tracked_objects = tracker.update(detections)

        # Count and draw tracking results
        current_ids = set()
        for x1, y1, x2, y2, track_id in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
            current_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        count = len(current_ids)
        max_count = max(max_count, count)

        # Display people count
        cv2.putText(frame, f"People: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

    total_time = time.time() - start_time
    return output_path, total_time, max_count


#########################
# Live camera detection #
#########################
def detect_people_camera(model_path, people_limit, enable_alert=True):
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
            
            if enable_alert:
                play_warning_sound(current_value=count, threshold=people_limit)
                
        else:
            alert_placeholder.empty()

        # Play warning sound only if enabled and condition is met
        if enable_alert:
            play_warning_sound(current_value=count, threshold=people_limit)

        stframe.image(frame, channels="BGR")

        if stop_button:
            break

    cap.release()


######################
# Geometry Utilities #
######################
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersects(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_direction(pt_old, pt_new, line_start, line_end):
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]
    move_dx = pt_new[0] - pt_old[0]
    move_dy = pt_new[1] - pt_old[1]
    cross = line_dx * move_dy - line_dy * move_dx
    return "out" if cross > 0 else "in"     # return "out" if cross > 0 else "in"; in->out, out->in

#######################
# Line Crossing Video #
#######################
def detect_live_stream(video_path, model_path, line, people_limit,
                       enable_alert=True, output_path="output.mp4"):
    st_frame = st.empty()
    alert_placeholder = st.empty()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    line_pt1, line_pt2 = line
    total_crossings = 0
    in_count = 0
    out_count = 0
    inside_people = 0

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    memory = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, classes=[0], conf=0.3)
        detections = []

        for box in results[0].boxes.xyxy.cpu().numpy():
            if len(box) >= 4:
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([x1, y1, x2, y2])

        dets = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets)

        current_memory = {}
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            current_memory[track_id] = (cx, cy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            if track_id in memory:
                prev_cx, prev_cy = memory[track_id]
                if intersects((prev_cx, prev_cy), (cx, cy), line_pt1, line_pt2):
                    direction = get_direction((prev_cx, prev_cy), (cx, cy), line_pt1, line_pt2)
                    total_crossings += 1
                    if direction == "in":
                        in_count += 1
                    else:
                        out_count += 1

        memory = current_memory
        inside_people = in_count - out_count

        # Draw line and counts
        cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 2)

        base_x, base_y = 20, 40
        line_spacing = 50

        cv2.putText(frame, f"Total Crossing: {total_crossings}", (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        cv2.putText(frame, f"In: {in_count}", (base_x, base_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Out: {out_count}", (base_x, base_y + 2 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Inside: {inside_people}", (base_x, base_y + 3 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

        if inside_people > people_limit:
            cv2.putText(frame, f"ALERT: {inside_people} inside! Limit: {people_limit}",
                        (base_x, base_y + 4 * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            alert_placeholder.warning(f"🚨 ALERT: {inside_people} inside! Limit: {people_limit}.")
            
            if enable_alert:
                play_warning_sound(current_value=inside_people, threshold=people_limit, duration=3)
                
        else:
            alert_placeholder.empty()

        # Play warning sound only if enabled and condition is met
        if enable_alert:
            play_warning_sound(current_value=inside_people, threshold=people_limit, duration=3)
        
        st_frame.image(frame, channels="BGR")
        out.write(frame)

    cap.release()
    out.release()

    return output_path
