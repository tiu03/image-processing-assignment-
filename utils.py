# utils.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from datetime import datetime
import streamlit as st
import pygame
import threading
import time
from streamlit_image_coordinates import streamlit_image_coordinates

# Save alert frame
def save_alert_screenshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_filename = f"assets/alerts/alert_{timestamp}.jpg"
    cv2.imwrite(alert_filename, frame)
    return alert_filename
     
# Initialize pygame mixer only once
if 'pygame_initialized' not in st.session_state:
    pygame.mixer.init()
    st.session_state.pygame_initialized = True

# Define play/stop logic with optional duration
def play_warning_sound(current_value, threshold, duration=None):
    # Initialize session state variables
    if 'is_alert_playing' not in st.session_state:
        st.session_state.is_alert_playing = False

    audio_path = "assets/sound/alert-109578.mp3"

    # Play sound if count is above threshold and not already playing
    if current_value > threshold:
        if not st.session_state.is_alert_playing:
            if os.path.exists(audio_path):
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play(-1)  # Loop

                    st.session_state.is_alert_playing = True

                    # If duration is given (image case), stop after duration
                    if duration:
                        def stop_after_delay():
                            time.sleep(duration)
                            pygame.mixer.music.stop()
                            pygame.mixer.quit()
                            st.session_state.is_alert_playing = False

                        threading.Thread(target=stop_after_delay, daemon=True).start()

                except Exception as e:
                    st.error(f"❌ Failed to play sound: {e}")
            else:
                st.warning(f"⚠️ Sound file not found: {audio_path}")

    # For real-time mode (e.g. camera): stop sound if under threshold
    elif st.session_state.is_alert_playing and duration is None:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        st.session_state.is_alert_playing = False
        

def select_roi_with_clicks(image, key_prefix="roi"):
    """
    Allows user to click two points to select a rectangular ROI.
    Returns the list of 4 polygon points if 2 clicks are made.
    """
    st.markdown("### 🖱️ Click on two points to select ROI (Top-Left & Bottom-Right):")
    coords = streamlit_image_coordinates(image, key=key_prefix)

    if "roi_points" not in st.session_state:
        st.session_state.roi_points = []

    if coords is not None and len(st.session_state.roi_points) < 2:
        x, y = int(coords["x"]), int(coords["y"])
        st.session_state.roi_points.append((x, y))

    if len(st.session_state.roi_points) == 2:
        pt1 = st.session_state.roi_points[0]
        pt2 = st.session_state.roi_points[1]

        # Return the 4 corner points of the rectangle
        zone_points = [
            pt1,
            (pt2[0], pt1[1]),
            pt2,
            (pt1[0], pt2[1])
        ]
        return zone_points

    return None