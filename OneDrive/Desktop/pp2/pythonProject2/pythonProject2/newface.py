import face_recognition
import numpy as np
import csv

# Load known faces and names
def load_known_faces(folder='known_faces'):
    known_encodings = []
    known_names = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

# Recognize faces in an image
def recognize_faces(frame, known_encodings, known_names):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        recognized_names.append((name, face_location))
    return recognized_names

