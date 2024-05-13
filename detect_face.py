from flask import Flask, Response
import cv2
import face_recognition
import os

app = Flask(__name__)

# Known face folder path (consider a more secure storage approach in production)
if not os.path.exists("/images/"):
    os.makedirs("/images")
if not os.path.exists("/new/"):
    os.makedirs("/new/")

#known_faces_folder = "D:\\images_for_pythonProject\\"
known_face_encodings = []
known_face_names = []

# Load known faces with error handling
for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            face_image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
            face_encodings = face_recognition.face_encodings(face_image)  # Might return empty list if no faces
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(filename.split(".")[0])  # Remove the file extension
        except Exception as e:
            print(f"Error loading image {filename}: {e}")

# Initialize webcam with error handling (consider adding a message if webcam access fails)
video_capture = None
try:
    video_capture = cv2.VideoCapture(0)
except Exception as e:
    print(f"Error accessing webcam: {e}")
    exit()

def gen_frames():
    while True:
        if video_capture is not None:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Check if a match has been found
            match_found = False
            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]

                # Compare the encodings
                for i, known_face_encoding in enumerate(known_face_encodings):
                    results = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.6)
                    if results[0]:
                        face_distances = face_recognition.face_distance([known_face_encoding], face_encoding)
                        match_percentage = (1 - face_distances[0]) * 100
                        print(f"Match found with {known_face_names[i]}! Similarity: {match_percentage:.2f}%")
                        match_found = True
                        break

                if match_found:
                    break

            if not match_found:
                # Save the unknown face image without the red box
                cv2.imwrite(os.path.join("/new/", "unknown_face.jpg"), frame)
                print("New face found. Image saved.")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frames for video stream

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition</title>
    </head>
    <body>
        <h1>Face Recognition</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
