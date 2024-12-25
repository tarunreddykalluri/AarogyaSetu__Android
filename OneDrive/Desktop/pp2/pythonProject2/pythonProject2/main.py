import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd


class AttendanceSystem:
    def __init__(self, registered_images_path='registered_faces'):
        self.registered_images_path = registered_images_path
        self.known_face_encodings = []
        self.known_face_names = []

        # Create registered_images directory if it doesn't exist
        if not os.path.exists(registered_images_path):
            os.makedirs(registered_images_path)
            print(f"Created {registered_images_path} directory")
            print("Please add face images to this directory and run the program again")
            return

        self.load_registered_faces()

    def load_registered_faces(self):
        """Load and encode all registered face images"""
        print("Loading registered faces...")

        # Check if directory is empty
        image_files = [f for f in os.listdir(self.registered_images_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not image_files:
            print(f"No images found in {self.registered_images_path}")
            print("Please add face images to this directory and run the program again")
            return

        for filename in image_files:
            try:
                image_path = os.path.join(self.registered_images_path, filename)
                print(f"Processing {filename}...")

                # Load image
                image = face_recognition.load_image_file(image_path)

                # Detect faces
                face_locations = face_recognition.face_locations(image)

                if not face_locations:
                    print(f"No face detected in {filename}. Skipping...")
                    continue

                if len(face_locations) > 1:
                    print(f"Multiple faces detected in {filename}. Using the first face...")

                # Get face encoding
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]

                self.known_face_encodings.append(face_encoding)
                # Use filename without extension as the person's name
                self.known_face_names.append(os.path.splitext(filename)[0])
                print(f"Successfully registered {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        if not self.known_face_encodings:
            print("No faces could be registered. Please check your images and try again.")
        else:
            print(f"\nSuccessfully loaded {len(self.known_face_names)} faces:")
            for name in self.known_face_names:
                print(f"- {name}")

    def mark_attendance(self, name):
        """Record attendance in a CSV file"""
        try:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            attendance_file = "attendance.csv"
            if not os.path.exists(attendance_file):
                df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
                df.to_csv(attendance_file, index=False)

            df = pd.read_csv(attendance_file)

            # Check if attendance already marked for today
            if not ((df['Name'] == name) & (df['Date'] == date)).any():
                new_row = {'Name': name, 'Date': date, 'Time': time}
                df = df._append(new_row, ignore_index=True)
                df.to_csv(attendance_file, index=False)
                return f"Attendance marked for {name}"
            return f"{name} already marked present today"

        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return "Error marking attendance"

    def take_attendance(self):
        """Capture video and process faces for attendance"""
        if not self.known_face_encodings:
            print("No registered faces found. Please add face images and run the program again.")
            return

        print("Starting attendance system... Press 'q' to quit")

        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find faces in current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    # Scale back face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Check if face matches any known face
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        message = self.mark_attendance(name)
                        print(message)

                    # Draw box around face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw name label
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow('Attendance System', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error during attendance: {str(e)}")

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def generate_attendance_report(self, date=None):
        """Generate attendance report for a specific date or all dates"""
        try:
            if not os.path.exists("attendance.csv"):
                return "No attendance records found"

            df = pd.read_csv("attendance.csv")
            if date:
                df = df[df['Date'] == date]

            return df

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    attendance_system.take_attendance()