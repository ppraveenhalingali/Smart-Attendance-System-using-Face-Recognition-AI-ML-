import cv2
import os
import time

# Ask for student name
student_name = input("Enter student name: ").strip()
path = f"dataset/{student_name}"

# Create folder if not exists
if not os.path.exists(path):
    os.makedirs(path)

# Count existing images
existing_images = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
count = existing_images
MAX_IMAGES = 20

# Open webcam with DirectShow backend for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot access camera. Close other apps and try again.")
    exit()

print(f"Capturing {MAX_IMAGES} images for {student_name}...")

while count < existing_images + MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for training
    img_resized = cv2.resize(frame, (224, 224))

    # Save image
    file_name = f"{path}/img{count}.jpg"
    cv2.imwrite(file_name, img_resized)
    print(f"Captured {file_name}")
    count += 1

    # Show live capture
    cv2.imshow(f"Capturing {student_name}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Dataset ready for {student_name}. Total images: {count}")

