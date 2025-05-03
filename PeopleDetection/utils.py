# utils.py
import os
import cv2
from datetime import datetime, timedelta
import streamlit as st
from playsound import playsound
import pygame

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

# Define play/stop logic
def play_warning_sound(current_value, threshold):
    # Initialize session state variables
    if 'is_alert_playing' not in st.session_state:
        st.session_state.is_alert_playing = False

    audio_path = "alert-109578.mp3"

    # Value exceeds threshold: play sound if not already playing
    if current_value > threshold:
        if not st.session_state.is_alert_playing:
            if os.path.exists(audio_path):
                try:
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play(-1)  # Loop until stopped
                    st.session_state.is_alert_playing = True
                except Exception as e:
                    st.error(f"❌ Failed to play sound: {e}")
            else:
                st.warning(f"⚠️ Sound file not found: {audio_path}")
    
    # Value is safe: stop the sound if it's playing
    elif st.session_state.is_alert_playing:
        pygame.mixer.music.stop()
        st.session_state.is_alert_playing = False