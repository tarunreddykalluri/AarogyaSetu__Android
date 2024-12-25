import cv2
import numpy as np
import face_recognition


def process_and_compare_faces(path1, path2):
    # Load and convert first image
    imgModi = face_recognition.load_image_file(path1)
    imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)

    # Load and convert second image
    imgTest = face_recognition.load_image_file(path2)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    # Find faces in first image
    face_locations_modi = face_recognition.face_locations(imgModi)
    if not face_locations_modi:
        raise ValueError("No face detected in the first image")

    # Get encoding for first face
    faceloc = face_locations_modi[0]
    encodeModi = face_recognition.face_encodings(imgModi)[0]

    # Draw rectangle around first face
    cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

    # Find faces in second image
    face_locations_test = face_recognition.face_locations(imgTest)
    if not face_locations_test:
        raise ValueError("No face detected in the second image")

    # Get encoding for first face in test image (using [0] instead of [1])
    facelocTest = face_locations_test[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]  # Changed from [1] to [0]

    # Draw rectangle around test face
    cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]),
                  (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

    # Compare faces
    results = face_recognition.compare_faces([encodeModi], encodeTest)
    faceDis = face_recognition.face_distance([encodeModi], encodeTest)

    # Print results
    print(f"Match: {results[0]}, Distance: {faceDis[0]:.2f}")

    # Add text to image
    cv2.putText(imgTest, f'{results[0]} {faceDis[0]:.2f}',
                (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    # Display images
    cv2.imshow('First Image', imgModi)
    cv2.imshow('Second Image', imgTest)
    cv2.waitKey(0)  # Fixed from waitKeys to waitKey
    cv2.destroyAllWindows()


# Usage
try:
    process_and_compare_faces('Images_Attendance/modi-image-for-InUth.png',
                              'Images_Attendance/narendra-modi.png')
except Exception as e:
    print(f"Error: {str(e)}")