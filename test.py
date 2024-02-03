from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import uuid
import sqlite3
import geocoder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Create SQLite database
db_file = "tracking_info.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Clear the 'tracking' table if it already exists
#conn.commit()

# Create the 'tracking' table with additional columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_name TEXT,
        track_id INTEGER,
        datetime TEXT,
        file_path TEXT,
        latitude REAL,
        longitude REAL,
        address TEXT
    )
''')
conn.commit()

# Load the YOLOv8 model
model = YOLO('Model1\\PH_V1.onnx')

# Open the video file
video_path = "Artifacts\\stock-footage-destroyed-road-difficult-traffic-area-threat-of-traffic-accident-countryside-ukraine.webm"
cap = cv2.VideoCapture(video_path)

# Store the track history and count of saved images for each track ID
track_history = defaultdict(lambda: {'track_points': [], 'image_count': 0, 'latitude': None, 'longitude': None,
                                     'last_frame_time': None, 'address': None})

# Create a folder for saving tracked information with current date and time
output_folder_root = "tracked_info"
output_folder_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_folder_root, exist_ok=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_names = results[0].names

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks and save cropped images with random filenames
        for box, track_id, class_name in zip(boxes, track_ids, class_names):
            x, y, w, h = box

            # Initialize track list, image count, latitude, and longitude if not present
            if track_id not in track_history:
                track_history[track_id] = {'track_points': [], 'image_count': 0, 'latitude': None, 'longitude': None,
                                           'last_frame_time': None, 'address': None}

            track = track_history[track_id]['track_points']
            track.append((float(x), float(y)))  # x, y center point

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Crop the region of interest
            roi = frame[y:h, x:w]

            # Save the cropped image with a random unique filename (limit to 2 images)
            if track_history[track_id]['image_count'] < 2:
                unique_filename = str(uuid.uuid4())[:8]
                tracker_folder = os.path.join(output_folder_root, output_folder_datetime, f"tracker_{track_id}")
                os.makedirs(tracker_folder, exist_ok=True)
                filename = f"{unique_filename}.jpg"
                output_path = os.path.join(tracker_folder, filename)
                cv2.imwrite(output_path, roi)

                # Update image count for the current track ID
                track_history[track_id]['image_count'] += 1

                # Get GPS coordinates and address for the location
                latitude, longitude = track_history[track_id]['latitude'], track_history[track_id]['longitude']
                if latitude is None or longitude is None:
                    location = geocoder.ip('me')
                    if location:
                        latitude, longitude = location.latlng
                        track_history[track_id]['latitude'] = latitude
                        track_history[track_id]['longitude'] = longitude
                        # Get address based on latitude and longitude
                        address = location.address
                        track_history[track_id]['address'] = address

                # Store tracking information in the database
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                object_name = class_name
                cursor.execute('''
                    INSERT INTO tracking (object_name, track_id, datetime, file_path, latitude, longitude, address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (object_name, track_id, current_datetime, output_path, latitude, longitude, address))
                conn.commit()

                # Update last frame time
                track_history[track_id]['last_frame_time'] = datetime.now()

        # Display the annotated frame
        cv2.imshow("Tracking", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Close the OpenCV window
cv2.destroyAllWindows()

# Close the SQLite connection
conn.close()
