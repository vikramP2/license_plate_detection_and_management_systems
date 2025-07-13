import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime, timedelta
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a Video Capture Object
cap = cv2.VideoCapture(0)

# Initialize the YOLO Model
model = YOLO("weights/best.pt")

# Initialize the frame count
count = 0

# Class Names
className = ["License"]

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Create SQLite Table
def create_table():
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS LicensePlates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                license_plate TEXT,
                image_path TEXT
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while creating the table: {e}")
    finally:
        conn.close()

# Paddle OCR with Indian number plate filtering
def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    max_score = 0

    if result:
        for r in result:
            scores = r[0][1]
            scores = int(scores * 100) if not np.isnan(scores) else 0
            if scores > max_score:
                max_score = scores
                text = r[0][0]

    pattern = re.compile('[\W]')
    text = pattern.sub('', text).replace("???", "").replace("O", "0").replace("粤", "")

    indian_plate_pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'
    if re.match(indian_plate_pattern, text):
        return str(text)
    return None

# Save JSON data
def save_json(plate, timestamp, image_path):
    interval_data = {
        "Timestamp": timestamp.isoformat(),
        "License Plate": plate,
        "Image Path": image_path
    }

    cummulative_file_path = "license_plate_reader/assets/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

# Save detected image with relative paths
def save_detected_image(frame, plate_number):
    try:
        relative_image_folder = "assets/images"
        absolute_image_folder = os.path.join(os.getcwd(), "license_plate_reader", relative_image_folder)

        if not os.path.exists(absolute_image_folder):
            os.makedirs(absolute_image_folder)

        image_filename = f"{plate_number}.jpg"
        absolute_image_path = os.path.join(absolute_image_folder, image_filename)
        cv2.imwrite(absolute_image_path, frame)

        relative_image_path = os.path.join("license_plate_reader", relative_image_folder, image_filename)
        print(f"Image saved: {relative_image_path}")
        return relative_image_path
    except Exception as e:
        print(f"Error saving image for plate {plate_number}: {e}")
        return None

# Save to SQLite database
def save_to_database(plate, timestamp, image_path):
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute(''' 
            INSERT INTO LicensePlates (timestamp, license_plate, image_path)
            VALUES (?, ?, ?)
        ''', (timestamp.isoformat(), plate, image_path))
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while saving to the database: {e}")
    finally:
        conn.close()

# Initialize database table
create_table()

startTime = datetime.now()
saved_plates = set()
plate_detection_times = {}
time_threshold = 30

while True:
    ret, frame = cap.read()
    if ret:
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    currentTime = datetime.now()
                    if label in plate_detection_times and (currentTime - plate_detection_times[label]).total_seconds() < time_threshold:
                        continue

                    image_path = save_detected_image(frame, label)
                    save_to_database(label, currentTime, image_path)
                    save_json(label, currentTime, image_path)
                    saved_plates.add(label)
                    plate_detection_times[label] = currentTime

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()