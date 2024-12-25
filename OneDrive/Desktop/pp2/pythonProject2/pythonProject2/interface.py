
import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox


# Load DNN model if files are available
def load_dnn_model():
    if os.path.exists('models/deploy.prototxt') and os.path.exists('models/res10_300x300_ssd_iter_140000.caffemodel'):
        net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
        return net
    return None


# DNN face detection function
def detect_faces_dnn(image, net):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            faces.append(box.astype("int"))
    return faces


# Capture an image, detect faces, and save it
def capture_and_detect():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image.")
        return

    net = load_dnn_model()
    faces = detect_faces_dnn(frame, net) if net else []

    # Draw rectangles around faces
    for (x, y, x2, y2) in faces:
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    # Save the image
    folder = "captured_faces"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(filename, frame)
    messagebox.showinfo("Saved", f"Image saved as {filename}")


# Create the Tkinter GUI
def create_app():
    root = tk.Tk()
    root.title("Face Detection App")
    root.geometry("300x200")
    button = tk.Button(root, text="Capture & Detect Faces", command=capture_and_detect, font=("Arial", 14))
    button.pack(pady=40)
    root.mainloop()


if __name__ == "__main__":
    create_app()
