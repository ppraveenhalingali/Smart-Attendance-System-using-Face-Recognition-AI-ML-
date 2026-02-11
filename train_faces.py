import face_recognition
import os
import cv2
import pickle

dataset_path = "dataset"
images = []
classNames = []

# Load all student images
for student in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student)
    if os.path.isdir(student_path):
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = face_recognition.load_image_file(img_path)
            images.append(img)
            classNames.append(student)

# Encode faces
def encode_faces(images):
    encoded_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:  # Only add if a face is detected
            encoded_list.append(encodings[0])
    return encoded_list

print("Encoding faces...")
encoded_faces = encode_faces(images)

# Save encodings and names
with open("encoded_faces.pkl", "wb") as f:
    pickle.dump((encoded_faces, classNames), f)

print("Encoding completed and saved to encoded_faces.pkl")

