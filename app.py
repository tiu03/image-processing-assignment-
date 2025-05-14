# app.py
import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import glob
import random
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from mode import detect_people_image, detect_people_video, detect_people_camera, detect_live_stream, detect_people_in_zone
model_path = "models/best.pt"

# Streamlit title
st.set_page_config(page_title="People Counting System", page_icon="📸")
st.title("📸 People Counting System")
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar: choose input mode
mode = st.sidebar.selectbox("Choose Input Mode", ["Image", "Zone Image", "Video", "Line Crossing Video", "Camera"])

people_limit = st.sidebar.number_input("⚠️ Set People Limit for Alert", min_value=0, max_value=100, value=5, step=1)

# Only show people limit and alert checkbox if mode supports alerts
if mode != "Video":
    enable_alert = st.sidebar.checkbox("🔔 Enable Alert When People Exceed Limit", value=True)
else:
    # Assign default values or disable alerts in video mode
    enable_alert = False
    
# Sidebar: choose model
#model_choice = st.sidebar.selectbox("Choose Detection Model", ["YOLOv8s", "Best"])

# Only show "Clear Alert Images" button in Camera mode
if mode == "Camera":
    if st.sidebar.button("🧹 Clear All Saved Alert Images"):
        for file in glob.glob("assets/alerts/*.jpg"):
            os.remove(file)
        st.sidebar.success("All alert images deleted.")
        st.rerun()

# Fun fact display
fun_facts = [
    "Did you know? YOLO stands for 'You Only Look Once'.",
    "Security cameras help reduce crime by up to 51%."
]
st.sidebar.info("💡 " + random.choice(fun_facts))

# Model path and import
#if model_choice == "YOLOv8s":
#    from yolov8s import detect_people_image, detect_people_video, detect_people_camera, detect_live_stream, detect_people_in_zone
#    model_path = "models/yolov8s.pt"
#elif model_choice == "Best":
#    from yolov8s import detect_people_image, detect_people_video, detect_people_camera, detect_live_stream, detect_people_in_zone
#    model_path = "models/best.pt"
#else:    
#    st.error("Unsupported model")
#    st.stop()

# Ensure asset folders exist
os.makedirs('assets/test', exist_ok=True)
model_result_dir = os.path.join('assets/result')
os.makedirs(model_result_dir, exist_ok=True)
os.makedirs('assets/alerts', exist_ok=True)

 
###################
# Image detection #
###################
if mode == "Image":
    st.subheader("🖼️ Upload an Image for People Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        save_path = os.path.join('assets/test', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.is_alert_playing = False
        processing_text = st.empty()
        processing_text.write("Processing the image... Please wait.")
        progress_bar = st.progress(0)
        
        for i in range(1, 101):
            progress_bar.progress(i)
            if i == 100:
                result_img, count, time_taken = detect_people_image(save_path, model_path, people_limit, enable_alert)
                
                result_path = os.path.join(model_result_dir, uploaded_file.name)
                cv2.imwrite(result_path, result_img)
                
                processing_text.write("Image processed successfully!")
                st.image(result_img, channels="BGR")
                st.success(f"✅ Total People Detected: {count}")
                st.success(f"✅ Time Taken: {time_taken:.2f}s")
                if count > people_limit:
                    st.error(f"🚨 ALERT: Detected {count} people! This exceeds the limit of {people_limit}.")
                st.download_button("Download Processed Image", open(result_path, "rb").read(), uploaded_file.name)


########################
# Zone image detection #
########################
elif mode == "Zone Image":
    st.subheader("🖼️ Upload an Image and Select ROI (Rectangle) for People Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        save_path = os.path.join('assets/test', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Reset alert and ROI state
        st.session_state.is_alert_playing = False
        if "roi_points" not in st.session_state:
            st.session_state.roi_points = []

        # Load and display image
        image = cv2.imread(save_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        st.markdown("### 🖱️ Click on two points (Top-left and Bottom-right) to select ROI:")
        coords = streamlit_image_coordinates(pil_image, key="roi")

        if coords is not None and len(st.session_state.roi_points) < 2:
            st.session_state.roi_points.append((int(coords["x"]), int(coords["y"])))

        if len(st.session_state.roi_points) == 2:
            pt1 = st.session_state.roi_points[0]
            pt2 = st.session_state.roi_points[1]

            # Create rectangular zone points (clockwise)
            zone_points = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]

            st.success("✅ ROI selected. Ready to process!")
            if st.button("🚀 Run Zone Detection"):
                result_img, zone_count, time_taken = detect_people_in_zone(
                    save_path, model_path, zone_points, people_limit, enable_alert
                )
                result_path = os.path.join(model_result_dir, f"zone_{uploaded_file.name}")
                cv2.imwrite(result_path, result_img)

                st.image(result_img, channels="BGR")
                st.success(f"✅ People in Zone: {zone_count}")
                st.success(f"✅ Time Taken: {time_taken:.2f}s")
                if zone_count > people_limit:
                    st.error(f"🚨 ALERT: {zone_count} people detected in zone! Limit is {people_limit}.")
                st.download_button("Download Result Image", open(result_path, "rb").read(), file_name=f"zone_{uploaded_file.name}")

                # Reset ROI points for next run
                st.session_state.roi_points = []

        elif len(st.session_state.roi_points) > 2:
            st.session_state.roi_points = []
            st.warning("⚠️ Only 2 points allowed. ROI selection reset.")
            
            
###################
# Video detection #
###################
elif mode == "Video":
    st.subheader("Upload a Video for People Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        save_path = os.path.join('assets/test', uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.success("✅ Video uploaded successfully.")

        if st.button("Run Detection"):
            with st.spinner("Processing the video... This may take a while."):
                result_path = os.path.join(model_result_dir, uploaded_video.name)
                output_path, total_time, max_count = detect_people_video(save_path, model_path, output_path=result_path)

            if max_count > people_limit:
                st.error(f"🚨 ALERT: Maximum {max_count} people detected! Limit is {people_limit}.")

            st.success("✅ Detection Completed and Video Ready!")
            st.video(output_path)
            st.success(f"🕒 Time Taken: {total_time:.2f} seconds")

            with open(output_path, "rb") as video_file:
                st.download_button("Download Processed Video", video_file.read(), file_name=uploaded_video.name)


#######################
# Line Crossing Video #
#######################
elif mode == "Line Crossing Video":
    st.subheader("🎥 Upload a Video and Click Two Points to Define the Line")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    # Initialize detection state
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False

    if uploaded_video:
        video_path = os.path.join("assets/test", uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.write("👆 Use the sliders below to define your crossing line:")

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Display image for user to visually reference
            st.image(frame_pil, caption="Preview of the video frame")

            st.write("🔲 **Point A** and **Point B** sliders will define the crossing line.")
            st.write("📝 Adjust the **X** and **Y** sliders for both points to set the line.")

            # Sliders to define points
            col1, col2 = st.columns(2)
            with col1:
                st.write("Point A:")
                point_a_x = st.slider("Point A - X", 0, frame.shape[1], frame.shape[1] // 4)
                point_a_y = st.slider("Point A - Y", 0, frame.shape[0], frame.shape[0] // 4)
            with col2:
                st.write("Point B:")
                point_b_x = st.slider("Point B - X", 0, frame.shape[1], 3 * frame.shape[1] // 4)
                point_b_y = st.slider("Point B - Y", 0, frame.shape[0], 3 * frame.shape[0] // 4)

            pt1 = (point_a_x, point_a_y)
            pt2 = (point_b_x, point_b_y)

            # Draw line on preview
            frame_with_line = frame.copy()
            cv2.line(frame_with_line, pt1, pt2, (255, 0, 0), 2)
            st.image(frame_with_line, channels="BGR", caption="✅ Line preview")

            # Handle start button
            if st.button("▶ Start Detection") and not st.session_state.detection_running:
                st.session_state.detection_running = True

                # Start actual detection
                with st.spinner("🔍 Detecting... Please wait."):
                    result_path = os.path.join(model_result_dir, f"results_{uploaded_video.name}")
                    output_path = detect_live_stream(
                        video_path, model_path, line=(pt1, pt2),
                        people_limit=people_limit, enable_alert=enable_alert,
                        output_path=result_path
                    )
                    st.session_state.detection_running = False

                    st.success("✅ Detection finished!")
                    st.video(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Result Video", f, file_name=output_path)


#########################
# Live camera detection #
#########################
elif mode == "Camera":
    st.subheader("📷 Real-Time Camera People Detection")

    start = st.button("▶️ Start Camera")
    if start:
        detect_people_camera(model_path, people_limit, enable_alert)
    
    reset_button = st.button("⏹️ Stop Camera")
    if reset_button:
        st.rerun()

    st.markdown("### 📸 Saved Alert Screenshots")
    alert_images = sorted(glob.glob("assets/alerts/*.jpg"), reverse=True)

    if alert_images:
        for img_path in alert_images[:5]:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(img_path, width=300, caption=os.path.basename(img_path))
                filename = os.path.basename(img_path)
                try:
                    timestamp_str = filename.replace("alert_", "").replace(".jpg", "")
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    st.caption(f"🕒 Captured at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                except ValueError:
                    st.caption("🕒 Timestamp: Unknown")

            with col2:
                with open(img_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=os.path.basename(img_path),
                        mime="image/jpeg"
                    )
    else:
        st.info("No alert screenshots saved yet.")
