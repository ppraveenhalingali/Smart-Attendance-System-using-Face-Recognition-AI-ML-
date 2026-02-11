import cv2
import face_recognition
import pickle
from datetime import datetime
import os
import pandas as pd

# Load encoded faces
with open("encoded_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Handle both dictionary and tuple formats
if isinstance(known_faces, dict):
    known_encodings = known_faces.get("encodings", [])
    known_names = known_faces.get("names", [])
elif isinstance(known_faces, tuple):
    known_encodings = known_faces[0]
    known_names = known_faces[1]
else:
    raise TypeError("encoded_faces.pkl format not recognized!")

# Create attendance file if it doesn't exist
# Create or check attendance file
if not os.path.exists("attendance.csv") or os.path.getsize("attendance.csv") == 0:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv("attendance.csv", index=False)


# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Mark attendance
            df = pd.read_csv("attendance.csv")
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y")
            time_string = now.strftime("%H:%M:%S")

            # Only mark once per day
            if not ((df["Name"] == name) & (df["Date"] == dt_string)).any():
                df = pd.concat([df, pd.DataFrame([[name, dt_string, time_string]], columns=df.columns)], ignore_index=True)
                df.to_csv("attendance.csv", index=False)
                print(f"Attendance marked for {name}")

        # Draw rectangle and name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
