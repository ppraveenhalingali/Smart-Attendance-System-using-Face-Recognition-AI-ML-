
# 📸 Smart Attendance System (AI + ML)

An automated attendance management system leveraging **Computer Vision** and **Deep Learning**. This system captures student faces, trains a recognition model, and logs attendance into a CSV file in real-time via a Streamlit web interface or a standalone Python script.

## ✨ Features

* **Real-time Face Recognition**: Utilizes `face_recognition` (dlib-based) to identify students with high accuracy.
* **Dual Interface**: Supports both a **Streamlit Web App** (`app.py`) and a **CLI/OpenCV window** (`mark_attendance.py`)
* **Automated Data Collection**: Easy-to-use script to capture 20 images per student to build a robust dataset.
* **Intelligent Logging**: Prevents duplicate entries by checking if a student has already been marked for the current day.
* **Data Export**: Attendance is saved in `attendance.csv` for easy integration with Excel or HR tools.

## 📂 Project Structure

* `capture_faces.py`: Captures and saves student images to the `dataset/` folder.
* `train_faces.py`: Generates 128-d facial encodings and saves them to `encoded_faces.pkl`.
* `app.py`: The main Streamlit dashboard for monitoring and running the system.
* `mark_attendance.py`: A lightweight script for running recognition without the web UI.
* `attendance.csv`: The local database storing Name, Date, and Time of arrival.

## 🚀 How to Run

### 1. Setup Environment

Ensure you have Python installed, then install the required libraries:

```bash
pip install streamlit opencv-python face_recognition numpy pandas

```

### 2. Capture Student Data

Run the capture script and enter the student's name when prompted. It will automatically take 20 photos using your webcam.

```bash
python capture_faces.py

```

### 3. Train the Model

Process the images in the `dataset` folder to create the facial encoding file:

```bash
python train_faces.py

```

### 4. Launch Attendance System

You can run the web-based version:

```bash
streamlit run app.py

```

Or the standard version:

```bash
python mark_attendance.py

```

## 📊 Technical Workflow

1. **Detection**: The system identifies the location of faces in a video frame using HOG (Histogram of Oriented Gradients).
2. **Encoding**: It transforms facial features into a 128-dimensional vector.
3. **Matching**: It calculates the Euclidean distance between the live face and the known database. A match is confirmed if the distance is below a threshold (e.g., 0.45).


4. **Logging**: If a match is found, the system verifies the date and appends a new record to the CSV.

---
