import cv2
from ultralytics import YOLO


def detect_people_camera():


    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture from camera.")
            break

        # Show frame in OpenCV window
        cv2.imshow('Real-time People Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function
detect_people_camera()