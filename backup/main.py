import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Simple Setup
# -------------------------------
dataset_path = "dataset"
attendance_file = "attendance.xlsx"

# -------------------------------
# Load known faces
# -------------------------------
print("="*50)
print("LOADING DATASET")
print("="*50)

known_face_encodings = []
known_face_names = []

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Created '{dataset_path}' folder. Add person folders with images.")
    exit()

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        count = 0
        for image_name in os.listdir(person_folder):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(person_folder, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
                    count += 1
        if count > 0:
            print(f"  ✅ Loaded {count} images for {person_name}")

print(f"\n📊 Total: {len(known_face_encodings)} faces loaded")
print("="*50)

# -------------------------------
# CREATE FRESH EXCEL FILE EVERY TIME
# -------------------------------
print("\n📋 CREATING FRESH ATTENDANCE FILE")

# Create a brand new DataFrame (clears old data)
df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Save the empty DataFrame to create a fresh file
df.to_excel(attendance_file, index=False)
print(f"  ✅ Created fresh attendance file: {attendance_file}")

# Track today's markings (in memory only)
today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()  # Empty set for today's markings

print(f"  📝 Ready to mark attendance for {today}")
print("="*50)

# -------------------------------
# Start camera
# -------------------------------
print("\n🎥 STARTING CAMERA")
video_capture = cv2.VideoCapture(0)

# Set lower resolution for speed
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera started successfully")
print("ℹ️ Controls: 'q' to quit | 's' to save current attendance")
print("="*50)

frame_count = 0
last_face_locations = []
last_face_names = []

while True:
    # Read frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    # Process every 2nd frame for better speed
    if frame_count % 2 == 0:
        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each face
        last_face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match = np.argmin(face_distances)
                    
                    if matches[best_match]:
                        name = known_face_names[best_match]
            
            last_face_names.append(name)
        
        last_face_locations = face_locations
    
    # Draw boxes using last known locations
    for i, face_location in enumerate(last_face_locations):
        if i < len(last_face_names):
            name = last_face_names[i]
        else:
            name = "Unknown"
        
        # Scale back face location
        top, right, bottom, left = [x * 4 for x in face_location]
        
        # Draw box
        if name != "Unknown":
            color = (0, 255, 0)  # Green
            
            # Mark attendance if not already marked today
            if name not in marked_today:
                now = datetime.now()
                time_string = now.strftime("%H:%M:%S")
                date_string = now.strftime("%Y-%m-%d")
                
                # Add to DataFrame
                new_row = pd.DataFrame({
                    "Name": [name],
                    "Date": [date_string],
                    "Time": [time_string]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                
                # Save to Excel immediately
                df.to_excel(attendance_file, index=False)
                
                marked_today.add(name)
                print(f"✅ {name} marked at {time_string}")
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show info on frame
    cv2.putText(frame, f"Marked: {len(marked_today)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Press 'q' to quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Face Attendance System', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        df.to_excel(attendance_file, index=False)
        print(f"💾 Manual save completed")

# Clean up
video_capture.release()
cv2.destroyAllWindows()

# Final save
df.to_excel(attendance_file, index=False)
print(f"\n📊 SESSION SUMMARY")
print("="*50)
print(f"  Total marked today: {len(marked_today)} people")
print(f"  File saved to: {attendance_file}")
print("="*50)

# Show the attendance that was just recorded
if len(df) > 0:
    print("\n📋 Today's Attendance:")
    for idx, row in df.iterrows():
        print(f"  {row['Time']} - {row['Name']}")
else:
    print("\n📋 No attendance recorded today")