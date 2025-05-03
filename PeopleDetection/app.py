# app.py
import streamlit as st
import os
import cv2
import glob
from datetime import datetime, timedelta






# Streamlit title
st.title("📸 People Counting System")
st.markdown("<br>", unsafe_allow_html=True)

people_limit = st.sidebar.number_input("⚠️ Set People Limit for Alert", min_value=1, max_value=100, value=5, step=1)

# Sidebar: choose input mode
mode = st.sidebar.selectbox("Choose Input Mode", ["Image", "Video", "Camera", "zonevideo", "zoneimg","ZoneCamera"])

# Sidebar: choose model
model_choice = st.sidebar.selectbox("Choose Detection Model", ["YOLOv8s", "YOLO11n"])

# Model path and import
if model_choice == "YOLOv8s":
    from yolov8s import detect_people_image, detect_people_video, detect_people_camera
    model_path = "models/yolov8s.pt"
elif model_choice == "YOLO11n":
    from yolo11 import detect_ppl_video_zone,image_detect_zone,cam_detect_zone
    from yolov8s import detect_people_image, detect_people_video, detect_people_camera
    model_path = "models/yolo11n.pt"
else:    
    st.error("Unsupported model")
    st.stop()

# Ensure asset folders exist
os.makedirs('assets/test', exist_ok=True)
model_result_dir = os.path.join('assets/result', model_choice.replace('.pt', ''))
os.makedirs(model_result_dir, exist_ok=True)
os.makedirs('assets/alerts', exist_ok=True)


###################
# Image detection #
###################
if mode == "Image":
    st.subheader("Upload an Image for People Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        save_path = os.path.join('assets/test', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        processing_text = st.empty()
        processing_text.write("Processing the image... Please wait.")
        progress_bar = st.progress(0)
        
        for i in range(1, 101):
            progress_bar.progress(i)
            if i == 100:
                result_img, count, time_taken = detect_people_image(save_path, model_path, people_limit)
                
                result_path = os.path.join(model_result_dir, uploaded_file.name)
                cv2.imwrite(result_path, result_img)
                
                processing_text.write("Image processed successfully!")
                st.image(result_img, channels="BGR")
                st.success(f"✅ Total People Detected: {count}")
                st.success(f"✅ Time Taken: {time_taken:.2f}s")
                if count > people_limit:
                    st.error(f"🚨 ALERT: Detected {count} people! This exceeds the limit of {people_limit}.")
                st.download_button("Download Processed Image", open(result_path, "rb").read(), uploaded_file.name)

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

        processing_text = st.empty()
        processing_text.write("Processing the video... Please wait.")
        progress_bar = st.progress(0)

        for i in range(1, 101):
            progress_bar.progress(i)
            if i == 100:
                result_path = os.path.join(model_result_dir, uploaded_video.name)
                output_path, total_time, max_count = detect_people_video(save_path, model_path, output_path=result_path, people_limit=people_limit)
                if max_count > people_limit:
                    st.error(f"🚨 ALERT: Maximum {max_count} people detected during video! Limit is {people_limit}.")

                processing_text.write("Video processed successfully!")
                st.video(output_path)
                st.success("✅ Detection Completed and Video Ready!")
                st.success(f"✅ Time Taken: {total_time:.2f} seconds")
                st.download_button("Download Processed Video", open(output_path, "rb").read(), uploaded_video.name)

#########################
# Live camera detection #
#########################
elif mode == "Camera":
    st.subheader("📷 Real-Time Camera People Detection")

    start = st.button("Start Camera")
    if start:
        detect_people_camera(model_path, people_limit)
    
    reset_button = st.button("Stop Camera")
    if reset_button:
        st.rerun()

    st.markdown("### 📸 Saved Alert Screenshots")
    alert_images = sorted(glob.glob("assets/alerts/*.jpg"), reverse=True)

    if alert_images:
        for img_path in alert_images[:5]:  # Show latest 5
            col1, col2 = st.columns([2, 1])  # Layout: image and download side-by-side

            with col1:
                st.image(img_path, width=300, caption=os.path.basename(img_path))

                # Extract timestamp from filename, assuming format like: alert_20250503_123456.jpg
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
        
elif mode == "zonevideo":
    st.subheader("Upload a Video for People Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        save_path = os.path.join('assets/test', uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        processing_text = st.empty()
        processing_text.write("Processing the video... Please wait.")
        detect_ppl_video_zone(save_path, model_path)
          ### to-do ah li ka to !!!!!!!!!!
elif mode == "zoneimg":
    st.subheader("Upload an Image for People Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        save_path = os.path.join('assets/test', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        

        image_detect_zone(save_path, model_path)
        ### to-do ah li ka to !!!!!!!!!!
  
elif mode == "ZoneCamera":
    st.subheader("📷 Real-Time Camera People Detection")

    start = st.button("Start Camera")
    if start:
        cam_detect_zone(model_path)
    else:
        st.info("No alert screenshots saved yet.")
                ### to-do ah li ka to !!!!!!!!!!
        

    
    
        
# streamlit run PeopleDetection/app.py  