import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os


# --- Global Variables ---
roi_points = []
polygon_zone = None

tracker = sv.ByteTrack()
zone_annotator = None
frame_copy_for_drawing = None

# --- Mouse Callback for Polygon Drawing ---
def draw_polygon(event, x, y, flags, param):
    global roi_points, frame_copy_for_drawing, polygon_zone

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        cv2.circle(frame_copy_for_drawing, (x, y), 5, (0, 255, 0), -1)
        if len(roi_points) > 1:
            cv2.line(frame_copy_for_drawing, roi_points[-2], roi_points[-1], (255, 0, 0), 2)
        cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)

# --- Main Program ---
def detect_ppl_video_zone(video_path,model_path):
    global roi_points, polygon_zone, frame_copy_for_drawing
    global zone_annotator
    model = YOLO(model_path)

    # --- Video File Configuration ---
    # *** Change this path to your video file ***
    VIDEO_PATH = video_path # e.g., "my_footage/traffic.avi"

    # Check if the video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
        print("Please update the VIDEO_PATH variable with the correct path to your video file.")
        return

    # Initialize video capture from the file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'")
        return

    # --- Draw ROI first ---
    print(f"Drawing zone on the first frame of '{os.path.basename(VIDEO_PATH)}'.")
    print("Draw your polygonal zone by clicking on the video window. Press 'c' when done.")

    ret, frame = cap.read() # Read the first frame to get dimensions and draw on
    if not ret:
        print(f"Error: Could not read the first frame from '{VIDEO_PATH}'.")
        cap.release()
        return

    frame_copy_for_drawing = frame.copy() # Create a copy to draw on

    cv2.namedWindow("Draw Zone - Press 'c' to Confirm")
    cv2.setMouseCallback("Draw Zone - Press 'c' to Confirm", draw_polygon)

    # Display the initial frame and wait for drawing
    cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(roi_points) >= 3:
                cv2.line(frame_copy_for_drawing, roi_points[-1], roi_points[0], (255, 0, 0), 2)
                cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)
                cv2.waitKey(500)
                break
            else:
                print("Please draw at least 3 points for the polygon.")
        elif key == ord('q'):
            print("Drawing cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Draw Zone - Press 'c' to Confirm")

    polygon = np.array(roi_points, dtype=np.int32)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    polygon_zone = sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(frame_width, frame_height)
    )

    box_annotator_outside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=255, g=0, r=0) # Blue
    )

    box_annotator_inside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=0, g=255, r=0) # Green
    )

    zone_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color(b=255, g=255, r=0), # Yellow/Cyan
        thickness=2,
        text_thickness=2,
        text_scale=1,
    )

    print("Zone confirmed. Starting detection...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video

    # --- Detection loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        results = model.predict(
            frame,
            classes=[0], # Only 'person'
            conf=0.4,
            device='0', # Use 'cpu' if no GPU
            stream=False
        )

        result = results[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = tracker.update_with_detections(detections)

        is_in_zone = polygon_zone.trigger(detections)

        detections_in_zone = detections[is_in_zone]
        detections_outside_zone = detections[~is_in_zone]

        # --- Generate labels based on tracker_id ---
        # Create label list for detections outside the zone
        labels_outside = []
        if detections_outside_zone.tracker_id is not None:
             # Use int(tracker_id) to ensure it's a simple integer string
            labels_outside = [f"human: {int(tracker_id)}" for tracker_id in detections_outside_zone.tracker_id]

        # Create label list for detections inside the zone
        labels_inside = []
        if detections_in_zone.tracker_id is not None:
             # Use int(tracker_id) to ensure it's a simple integer string
            labels_inside = [f"human: {int(tracker_id)}" for tracker_id in detections_in_zone.tracker_id]
        # --- End of label generation ---



        # Annotate frame with bounding boxes and labels
        # Draw outside boxes first
        annotated_frame = box_annotator_outside.annotate(
            scene=frame.copy(),
            detections=detections_outside_zone,
            labels=labels_outside # Pass the generated labels
        )
        # Then draw inside boxes on the same frame
        annotated_frame = box_annotator_inside.annotate(
            scene=annotated_frame,
            detections=detections_in_zone,
            labels=labels_inside, # Pass the generated labels
        )

        # Annotate frame with the zone polygon and count
        annotated_frame = zone_annotator.annotate(
            scene=annotated_frame
        )

        cv2.imshow("People Counter", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")
    
    
    
# def detect_people_camera(image_path,model_path):
#     # Load YOLO model
#     model = YOLO(model_path)

#     # Supervision annotator for bounding boxes
#     box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

#     # Open webcam
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture from camera.")
#             break

#         # Run person detection
#         results = model.predict(frame, classes=[0], conf=0.35, device='0')

#         for result in results:
#             detections = sv.Detections.from_ultralytics(result)

#             # Count people
#             person_count = len(detections)

#             # Prepare labels
#             labels = [
#                 f"{model.model.names[class_id]} {confidence:.2f}"
#                 for class_id, confidence in zip(detections.class_id, detections.confidence)
#             ]

#             # Annotate frame
#             frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

#             # Draw count text
#             cv2.putText(frame, f'People Count: {person_count}', (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Show frame
#         cv2.imshow('Real-time People Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()




# # --- Global Variables ---
# roi_points = []
# polygon_zone = None

# # tracker is not needed for a single image
# zone_annotator = None
# image_copy_for_drawing = None # Use a copy for drawing polygon


# --- Main Program ---
def image_detect_zone(image_path,model_path):
    global roi_points, polygon_zone, image_copy_for_drawing
    global zone_annotator
    model = YOLO(model_path) # Detection model only
    # --- Image File Configuration ---
    # *** Change this path to your image file ***
    IMAGE_PATH = image_path # e.g., "photos/crowd.png"

    # Check if the image file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        print("Please update the IMAGE_PATH variable with the correct path to your image file.")
        return

    # Load the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from '{IMAGE_PATH}'")
        return

    # --- Draw ROI first ---
    print(f"Drawing zone on image '{os.path.basename(IMAGE_PATH)}'.")
    print("Draw your polygonal zone by clicking on the image window. Press 'c' when done.")

    image_copy_for_drawing = image.copy() # Create a copy to draw on

    cv2.namedWindow("Draw Zone - Press 'c' to Confirm")
    cv2.setMouseCallback("Draw Zone - Press 'c' to Confirm", draw_polygon)

    # Display the initial image and wait for drawing
    cv2.imshow("Draw Zone - Press 'c' to Confirm", image_copy_for_drawing)

    while True:
        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for a key press
        if key == ord('c'):
            if len(roi_points) >= 3:
                # Close the loop for the polygon visualization on the drawing image
                cv2.line(image_copy_for_drawing, roi_points[-1], roi_points[0], (255, 0, 0), 2)
                cv2.imshow("Draw Zone - Press 'c' to Confirm", image_copy_for_drawing)
                cv2.waitKey(500) # Show the closed polygon briefly
                break
            else:
                print("Please draw at least 3 points for the polygon.")
        elif key == ord('q'):
            print("Drawing cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Draw Zone - Press 'c' to Confirm") # Close the drawing window

    # Convert points to numpy array (integers are needed)
    polygon = np.array(roi_points, dtype=np.int32)

    # Initialize Supervision zone and annotators
    # Use the image's dimensions for frame_resolution_wh
    image_height, image_width, _ = image.shape

    polygon_zone = sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(image_width, image_height)
    )

    # --- Initialize TWO Box Annotators with different colors using sv.Color ---
    # Annotator for bounding boxes OUTSIDE the zone (e.g., Blue)
    box_annotator_outside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=255, g=0, r=0) # Blue
    )

    # Annotator for bounding boxes INSIDE the zone (e.g., Green)
    box_annotator_inside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=0, g=255, r=0) # Green
    )
    # --- End of Box Annotator Initialization ---

    # Annotator for the zone and count
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color(b=255, g=255, r=0), # Yellow/Cyan
        thickness=2,
        text_thickness=2,
        text_scale=1,
        # text_color=(0, 0, 0), # Optional: Black text for count - This one *can* be a tuple or sv.Color
        # filled=True # Optional: Fill the zone polygon slightly
    )

    print("Zone confirmed. Starting detection...")

    # --- Perform Detection and Annotation on the Image ---

    # Perform prediction on the image (no stream=False needed for single image)
    # Only predict for class 'person' (class_id=0)
    results = model.predict(
        image,
        classes=[0], # Only 'person'
        conf=0.4,
        device='0' # Use 'cpu' if no GPU
    )

    # Get the results for the image
    result = results[0]

    # Convert prediction results to supervision Detections format
    detections = sv.Detections.from_ultralytics(result)

    # Tracker update is not needed for a single image

    # Trigger zone and get boolean mask indicating detections inside the zone
    # This updates polygon_zone.current_count internally
    is_in_zone = polygon_zone.trigger(detections)

    # Filter detections based on whether they are inside the zone
    detections_in_zone = detections[is_in_zone]
    detections_outside_zone = detections[~is_in_zone] # ~ is the boolean NOT operator

    # --- Generate labels (e.g., "Person", "Conf: 0.95") ---
    # Since there's no tracker_id for a single image, we can use class name and confidence
    class_names = model.model.names # Get class names from YOLO model

    # Create label list for detections outside the zone
    labels_outside = []
    if detections_outside_zone.confidence is not None and detections_outside_zone.class_id is not None:
        labels_outside = [
            f"{class_names[class_id]} Conf: {confidence:.2f}"
            for confidence, class_id in zip(detections_outside_zone.confidence, detections_outside_zone.class_id)
        ]

    # Create label list for detections inside the zone
    labels_inside = []
    if detections_in_zone.confidence is not None and detections_in_zone.class_id is not None:
        labels_inside = [
            f"{class_names[class_id]} Conf: {confidence:.2f}"
            for confidence, class_id in zip(detections_in_zone.confidence, detections_in_zone.class_id)
        ]
    # --- End of label generation ---

    # Annotate the image with bounding boxes and labels
    # Draw outside boxes first
    annotated_image = box_annotator_outside.annotate(
        scene=image.copy(), # Annotate on a copy of the original image
        detections=detections_outside_zone,
        labels=labels_outside # Pass the generated labels
    )
    # Then draw inside boxes on the same image
    annotated_image = box_annotator_inside.annotate(
        scene=annotated_image,
        detections=detections_in_zone,
        labels=labels_inside # Pass the generated labels
    )

    # Annotate the image with the zone polygon and count
    # The zone_annotator gets the count from polygon_zone.current_count
    annotated_image = zone_annotator.annotate(
        scene=annotated_image # Pass the image to draw on
    )

    # Display the annotated image and wait for a key press to close
    cv2.imshow("People Counter - Image", annotated_image)

    print("Processing complete. Press any key to close the window.")
    cv2.waitKey(0) # Wait indefinitely

    # --- Cleanup ---
    cv2.destroyAllWindows()
    print("Program finished.")

#####################
#####################


# # --- Mouse Callback for Polygon Drawing ---
# def draw_polygon(event, x, y, flags, param):
#     global roi_points, frame_copy_for_drawing, polygon_zone

#     if event == cv2.EVENT_LBUTTONDOWN:
#         roi_points.append((x, y))
#         # Draw points and lines on the copy
#         cv2.circle(frame_copy_for_drawing, (x, y), 5, (0, 255, 0), -1)
#         if len(roi_points) > 1:
#             cv2.line(frame_copy_for_drawing, roi_points[-2], roi_points[-1], (255, 0, 0), 2)
#         cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)

#     # Re-draw the current points and lines if window is refreshed (less common but good practice)
#     # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
#     #     pass # Optional: Can add drawing as mouse moves, but button down is sufficient here

# --- Main Program ---
def cam_detect_zone(model_path):
    global roi_points, polygon_zone, frame_copy_for_drawing
    global zone_annotator
    model = YOLO(model_path) # Detection model
    # --- Camera Configuration ---
    # Use 0 for the default camera. Change if you have multiple cameras (e.g., 1, 2)
    CAMERA_INDEX = 0

    # Initialize video capture from the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        print("Please check if the camera is connected and available.")
        return

    # --- Draw ROI first ---
    print(f"Drawing zone on the first frame from Camera {CAMERA_INDEX}.")
    print("Draw your polygonal zone by clicking on the video window. Press 'c' when done.")

    ret, frame = cap.read() # Read the first frame to get dimensions and draw on
    if not ret:
        print(f"Error: Could not read the first frame from Camera {CAMERA_INDEX}.")
        cap.release()
        return

    frame_copy_for_drawing = frame.copy() # Create a copy to draw on

    cv2.namedWindow("Draw Zone - Press 'c' to Confirm")
    cv2.setMouseCallback("Draw Zone - Press 'c' to Confirm", draw_polygon)

    # Display the initial frame and wait for drawing
    cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(roi_points) >= 3:
                # Close the loop for the polygon visualization on the drawing frame
                cv2.line(frame_copy_for_drawing, roi_points[-1], roi_points[0], (255, 0, 0), 2)
                cv2.imshow("Draw Zone - Press 'c' to Confirm", frame_copy_for_drawing)
                cv2.waitKey(500) # Show the closed polygon briefly
                break
            else:
                print("Please draw at least 3 points for the polygon.")
        elif key == ord('q'):
            print("Drawing cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Draw Zone - Press 'c' to Confirm") # Close the drawing window

    # Convert points to numpy array (integers are needed)
    polygon = np.array(roi_points, dtype=np.int32)

    # Initialize Supervision zone and annotators
    # Use the frame's actual dimensions for frame_resolution_wh
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    polygon_zone = sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(frame_width, frame_height)
    )

    # --- Initialize TWO Box Annotators with different colors using sv.Color ---
    # Annotator for bounding boxes OUTSIDE the zone (e.g., Blue)
    box_annotator_outside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=255, g=0, r=0) # Blue
    )

    # Annotator for bounding boxes INSIDE the zone (e.g., Green)
    box_annotator_inside = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color(b=0, g=255, r=0) # Green
    )
    # --- End of Box Annotator Initialization ---

    # Annotator for the zone and count
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color(b=255, g=255, r=0), # Yellow/Cyan
        thickness=2,
        text_thickness=2,
        text_scale=1,
    )

    print("Zone confirmed. Starting detection...")

    # --- Detection loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream end or camera error.")
            # For camera, a read error might mean the camera was disconnected
            # You might want to break or add specific error handling here
            break

        # Perform prediction on the frame
        # Only predict for class 'person' (class_id=0)
        results = model.predict(
            frame,
            classes=[0], # Only 'person'
            conf=0.4,
            device='cpu', # Use 'cpu' if no GPU
            stream=False # Get results directly
        )

        # Get the results for the first source (our frame)
        result = results[0]

        # Convert prediction results to supervision Detections format
        detections = sv.Detections.from_ultralytics(result)

        # Update tracker with current detections
        detections = tracker.update_with_detections(detections)

        # Trigger zone and get boolean mask indicating detections inside the zone
        # This updates polygon_zone.current_count internally
        is_in_zone = polygon_zone.trigger(detections)

        # # --- Add these print statements ---
        # print(f"Number of tracked detections: {len(detections)}")
        # print(f"Is in zone mask: {is_in_zone}") # This shows True/False for each detection
        # print(f"Current zone count: {polygon_zone.current_count}") # This is the value used by the annotator
        # # --- End of print statements ---
        
        # Filter detections based on whether they are inside the zone
        detections_in_zone = detections[is_in_zone]
        detections_outside_zone = detections[~is_in_zone] # ~ is the boolean NOT operator

        # --- Generate labels based on tracker_id ---
        # Create label list for detections outside the zone
        labels_outside = []
        # Check if tracker_id is available and not None for individual detections
        if detections_outside_zone.tracker_id is not None:
             labels_outside = [
                 f"person: {int(tracker_id)}" for tracker_id in detections_outside_zone.tracker_id
                 if tracker_id is not None # Ensure tracker_id is not None
             ]


        # Create label list for detections inside the zone
        labels_inside = []
        # Check if tracker_id is available and not None for individual detections
        if detections_in_zone.tracker_id is not None:
             labels_inside = [
                 f"ID: {int(tracker_id)}" for tracker_id in detections_in_zone.tracker_id
                 if tracker_id is not None # Ensure tracker_id is not None
             ]
        # --- End of label generation ---


        # Annotate frame with bounding boxes and labels
        # Draw outside boxes first
        annotated_frame = box_annotator_outside.annotate(
            scene=frame.copy(), # Annotate on a copy
            detections=detections_outside_zone,
            labels=labels_outside # Pass the generated labels
        )
        # Then draw inside boxes on the same frame
        annotated_frame = box_annotator_inside.annotate(
            scene=annotated_frame,
            detections=detections_in_zone,
            labels=labels_inside # Pass the generated labels
        )

        # Annotate frame with the zone polygon and count
        annotated_frame = zone_annotator.annotate(
            scene=annotated_frame # Pass the frame
        )

        cv2.imshow("People Counter - Camera", annotated_frame)

        # Use waitKey(1) for camera feed to process frames in near real-time
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # --- Cleanup ---
    cap.release() # Release camera resource
    cv2.destroyAllWindows()
    print("Program finished.")

# if __name__ == "__main__":
#     detect_ppl_video_zone("video.mp4", "yolo11n.pt")
#     detect_people_camera("yolo11n.pt")
#     image_detect_zone("people1.jpg","yolo11n.pt")
#     cam_detect_zone("yolo11n.pt")
