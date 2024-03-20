import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
from PIL import Image, ImageTk
from train_classifier import FaceRecognitionModel, train_and_save
from utils import load_dataset_names, encode_dataset
import mysql.connector

class FaceVerificationApp:
    def __init__(self, root_dir):
        self.root = root_dir
        self.root.title("Ujian Skripsi")

        # Create labels
        self.label = tk.Label(root_dir, text="[Camera Feed]")
        self.label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.report_label = tk.Label(root_dir, text="Exam Report:", anchor="w", justify="left")
        self.report_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        # Create buttons
        self.record_button = tk.Button(root_dir, text="Record Face", command=self.record_face)
        self.record_button.grid(row=3, column=0, padx=10, pady=10)

        self.train_button = tk.Button(root_dir, text="Train", command=self.train)
        self.train_button.grid(row=3, column=1, padx=10, pady=10)

        self.exam_button = tk.Button(root_dir, text="Run Exam", command=self.toggle_exam)
        self.exam_button.grid(row=3, column=2, padx=10, pady=10)

        self.submit_button = tk.Button(root_dir, text="Submit Data", command=self.submit_data)
        self.submit_button.grid(row=4, column=1, padx=10, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load facemodel & dataset names
        self.face_model = FaceRecognitionModel()
        self.dataset_names = load_dataset_names()
        self.dataset_encodings = encode_dataset(self.dataset_names)
        self.camera = cv2.VideoCapture(0)
        self.student_names = []
        self.face_images = []
        self.is_running_exam = False
        self.start_time = None
        self.face_detection_timer = None
        self.face_count = 0
        self.face_detected_count = 0
        self.warning_count = 0
        self.no_face_detected_count = 0
        self.start_exam_time = None
        self.update()
        self.end_time = None
        self.duration_exam = None
        self.final_grade = None


    def record_face(self):
        _, frame = self.camera.read()
        face_locations = face_recognition.face_locations(frame)

        if not face_locations:
            messagebox.showwarning("Error", "No face detected.")
            return

        # Prompt user to input student name
        student_name = simpledialog.askstring("Input Name", "Enter student name:")
        if not student_name:
            messagebox.showwarning("Warning", "Student name not provided. Skipping.")
            return

        student_dir = os.path.join("dataset", student_name)
        os.makedirs(student_dir, exist_ok=True)

        face_count = 0
        while face_count < 15:
            _, frame = self.camera.read()
            face_locations = face_recognition.face_locations(frame)

            if face_locations:
                face_count += 1
                face_image = Image.fromarray(frame)
                face_image.save(os.path.join(student_dir, f"{face_count}.jpg"))
                self.face_images.append(face_image)
                self.student_names.append(student_name)
                print(f"Captured face {face_count}/15 for {student_name}")
            else:
                retry = messagebox.askretrycancel("Error", f"No face detected for {student_name}. Retry?")
                if not retry:
                    break

        if face_count == 15:
            messagebox.showinfo("Done", f"Face recording for {student_name} completed.")

    def train(self):
        if not self.face_images or not self.student_names:
            messagebox.showwarning("Error", "No face images or student names recorded.")
            return

        try:
            train_and_save()
            messagebox.showinfo("Success", "Dataset has been created.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {e}")

    def toggle_exam(self):
        if self.is_running_exam:
            self.stop_exam()
        else:
            self.start_exam()

    def start_exam(self):
        if not os.path.exists("dataset"):
            messagebox.showwarning("Error", "Dataset not found. Please train the classifier first.")
            return

        self.is_running_exam = True
        self.face_detected_count = 0
        self.start_time = datetime.now()
        self.start_exam_time = self.start_time
        self.face_count = 0
        self.warning_count = 0
        self.no_face_detected_count = 0

        self.exam_button.config(text="Stop Exam")
        # Start the face detection timer
        self.face_detection_timer = self.root.after(3000, self.face_detection_timeout)

    def stop_exam(self):
        self.end_time = datetime.now()
        self.duration_exam = self.end_time - self.start_exam_time
        self.is_running_exam = False
        if self.face_detection_timer is not None:
            self.root.after_cancel(self.face_detection_timer)  # Cancel the face detection timer

        # Generate and display the exam report
        self.generate_report()
        self.final_grade = self.generate_final_grade()
        self.exam_button.config(text="Run Exam")


    def face_detection_timeout(self):
        # This method is called when face detection times out
        self.is_running_exam = False

        if self.warning_count < 3:  # Check if the maximum number of warnings has not been reached
            self.warning_count += 1
            messagebox.showwarning("Warning",
                                      (f"No face detected within 3 seconds. "
                                       f"Please ensure your face is visible. "
                                       f"Warning {self.warning_count}/3"))
            # Restart the face detection timer
            self.face_detection_timer = self.root.after(3000, self.face_detection_timeout)
        else:
            # Maximum number of warnings reached, stop the exam
            self.stop_exam()

    def generate_report(self):
        total_frames = self.face_count
        detected_same_face_frames = self.face_detected_count
        different_face_or_no_face_frames = self.no_face_detected_count
        warning = self.warning_count

        report_text = (
            f"Total: {total_frames} Frame(s)\n"
            f"Detected same face as exam taker: {detected_same_face_frames} Frame(s)\n"
            f"Different face or no face detected: {different_face_or_no_face_frames} Frame(s)\n"
            f"Warning: {warning}\n"
        )

        if (different_face_or_no_face_frames + warning) / total_frames > 0.2:
            report_text += "Cheating detected."
        else:
            report_text += "No cheating detected."

        self.report_label.config(text="Exam Report:\n" + report_text)

    def update(self):
        # Read frame from the camera
        ret, frame = self.camera.read()

        # If exam is running, perform face detection and recognition
        if self.is_running_exam:
            self.face_count += 1
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                self.face_detected_count += len(face_locations)
                # Reset the face detection timer whenever a face is detected
                if self.face_detection_timer is not None:
                    self.root.after_cancel(self.face_detection_timer)
                    self.face_detection_timer = self.root.after(3000, self.face_detection_timeout)
            else:
                self.no_face_detected_count += 1

            # Perform face recognition
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]

                # Resize the face image to the size expected by the face recognition model
                face_image = cv2.resize(face_image, (128, 128))

                # Encode the face
                face_encodings = face_recognition.face_encodings(face_image)
                if face_encodings:
                    # Compare the encoded face with the dataset
                    for i, face_encoding in enumerate(face_encodings):
                        matches = face_recognition.compare_faces(self.dataset_encodings, face_encoding)
                        if True in matches:
                            match_index = matches.index(True)
                            if match_index < len(self.dataset_names):
                                matched_name = self.dataset_names[match_index]

                                # Draw rectangle around the face
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                                # Add label indicating the name of the recognized individual
                                cv2.putText(frame, matched_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)
                            else:
                                print("Match index out of range:", match_index)
                        else:
                            print("No match found in dataset")
                else:
                    print("No face encodings found")

        if ret:  # Check if the frame was successfully captured
            # Resize the frame to have a width of 400 pixels and adjust height to maintain aspect ratio
            width = 400
            height = int(frame.shape[0] * (width / frame.shape[1]))
            frame = cv2.resize(frame, (width, height))

            # Convert the frame to an ImageTk object and update the label
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(img)
            self.label.config(image=img)
            self.label.image = img  # type: ignore

        # Schedule the update method to be called again after 100 milliseconds
        self.root.after(100, self.update)

    def generate_final_grade(self):
        total_frames = self.face_count
        detected_same_face_frames = self.face_detected_count
        different_face_or_no_face_frames = self.no_face_detected_count
        warning = self.warning_count

        rate = detected_same_face_frames / total_frames
        threshold = 0.6

        if warning == 3:
            return "High"
        elif warning < 3 and rate < threshold:
            return "Medium"
        else:
            return "Low"
 
    def on_close(self):
        try:
            # Release the camera resource
            self.camera.release()
        except Exception as e:
            print("An error occurred while releasing the camera resource:", e)

        # Destroy the root window
        self.root.destroy()


    
    def submit_data(self):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root",
                database="report_db")

            cursor = conn.cursor()

    # Insert data into the table
            cursor.execute("""
                INSERT INTO exam_records (
                    no,
                    student_name,
                    start_time,
                    total_frames,
                    detected_same_face_frames,
                    different_face_or_no_face_frames,
                    warning_count,
                    end_time,
                    duration,
                    grade) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (None,
                self.student_names[1],
                self.start_exam_time, 
                self.face_count, 
                self.face_detected_count, 
                self.no_face_detected_count, 
                self.warning_count, 
                self.end_time, 
                self.duration_exam,
                self.final_grade))

            conn.commit()  # Commit the changes to the database
            messagebox.showinfo("Success", "Data submitted successfully.")
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVerificationApp(root)
    root.mainloop()
