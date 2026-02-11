📸 Smart Attendance System using Face Recognition (AI + ML)

An automated attendance management system that uses face recognition to identify registered users in real time and mark attendance digitally using a webcam. This project reduces manual effort, prevents proxy attendance, and provides an easy-to-use web interface for viewing and downloading attendance records.

🚀 Features

1. Real-time face detection and recognition

2. Automatic attendance marking with time

3. Web-based interface using Streamlit

4. CSV export of attendance records

5. Prevents duplicate entries per session/day

6. Simple and user-friendly UI

🛠️ Technologies Used

1. Python

2. OpenCV

3. face_recognition (dlib-based ML library)

4. Streamlit

5. Pandas, NumPy

5. Pickle

📂 Project Structure
Smart-Attendance-System/
│
├── dataset/                 # Collected student images
├── capture_faces.py         # Capture images for each student
├── train_faces.py           # Train and save face encodings
├── encoded_faces.pkl        # Trained face data
├── app.py                   # Streamlit web app
├── attendance.csv           # Attendance records
├── mark_attendance.py       # (Optional) CLI-based attendance
├── recognize_attendance.py # (Optional) Alternative recognition
└── README.md

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system


Install required libraries

pip install opencv-python face_recognition streamlit pandas numpy


⚠️ Note: face_recognition requires dlib. On Windows, install precompiled wheels if needed.

▶️ How to Run
Step 1: Capture Faces
python capture_faces.py

Step 2: Train the Model
python train_faces.py

Step 3: Run the Web App
streamlit run app.py

📊 Output

Live camera feed with bounding box and name

Automatic attendance marking

Attendance stored in attendance.csv

Download option from web interface

📈 Results

The system accurately recognizes registered users in real time under normal lighting conditions and marks attendance automatically. Minor errors may occur under low light or unclear face angles.

🚀 Future Scope

Cloud database integration

Mobile app support

Multi-camera support

Advanced deep learning models

Analytics and reports
