import streamlit as st
import cv2
import face_recognition
import os

# Load known faces
known_faces_folder = "D:\\images_for_pythonProject\\"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            face_image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(filename.split(".")[0])
        except Exception as e:
            st.error(f"Error loading image {filename}: {e}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Main Streamlit app
st.title("Face Recognition")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        st.error("Error accessing webcam")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if a match has been found
    match_found = False
    for face_encoding in face_encodings:
        results = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        if any(results):
            match_found = True
            break

    if match_found:
        st.write("Match found!")
        # Add code to display match info
    else:
        st.write("No match found!")

    # Display the frame
    st.image(frame, channels="BGR")

# Close webcam and cleanup
video_capture.release()
cv2.destroyAllWindows()
