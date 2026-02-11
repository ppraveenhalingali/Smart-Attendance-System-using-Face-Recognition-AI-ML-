import streamlit as st
import cv2
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    📸 Smart Attendance System (AI + ML)
    </h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
with open("encoded_faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# ---------------- ATTENDANCE FILE ----------------
ATT_FILE = "attendance.csv"
if not os.path.exists(ATT_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATT_FILE, index=False)

# ---------------- CONTROLS ----------------
left, right = st.columns([1, 2])

with left:
    st.subheader("🎛 Controls")
    start = st.button("▶ Start Attendance")
    stop = st.button("⏹ Stop Attendance")

    st.markdown(
        """
        **Instructions:**
        - Click *Start Attendance*
        - Face camera clearly
        - Attendance will be marked automatically
        - Click *Stop Attendance* to end
        """
    )

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

# ---------------- CAMERA ----------------
with right:
    st.subheader("📷 Live Camera")
    frame_window = st.image([])

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match = np.argmin(distances)

            name = "Unknown"
            if distances[best_match] < 0.45:
                name = known_names[best_match]

                df = pd.read_csv(ATT_FILE)
                if name not in df["Name"].values:
                    time_now = datetime.now().strftime("%H:%M:%S")
                    df.loc[len(df)] = [name, time_now]
                    df.to_csv(ATT_FILE, index=False)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_window.image(frame, channels="BGR")

    cap.release()

# ---------------- ATTENDANCE TABLE ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📄 Attendance Records")

df = pd.read_csv(ATT_FILE)

if df.empty:
    st.info("No attendance recorded yet.")
else:
    st.dataframe(df, width="stretch")


# ---------------- DOWNLOAD BUTTON ----------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download Attendance CSV",
    data=csv,
    file_name="attendance.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
    Developed by Nitish | Smart Attendance System using AI & ML
    </p>
    """,
    unsafe_allow_html=True
)

