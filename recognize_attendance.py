import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

# Load encoded faces
with open("encoded_faces.pkl", "rb") as f:
    encoded_faces, classNames = pickle.load(f)

attendance = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces_cur_frame = face_recognition.face_locations(rgb_frame)
    encodes_cur_frame = face_recognition.face_encodings(rgb_frame, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encoded_faces, encode_face)
        face_dist = face_recognition.face_distance(encoded_faces, encode_face)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = classNames[match_index].upper()
            if name not in attendance:
                attendance.append(name)
                now = datetime.now()
                df = pd.DataFrame([[name, now.strftime("%Y-%m-%d %H:%M:%S")]], columns=["Name", "Time"])
                df.to_csv("Attendance.csv", mode='a', header=not bool(pd.read_csv("Attendance.csv") if os.path.exists("Attendance.csv") else None), index=False)
            
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance System - Press 'q' to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance session ended.")
