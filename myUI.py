import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

class VideoClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Classifier")

        # Load the trained model
        self.model = load_model('/path/to/save/model.h5')

        # Create UI components
        self.label = tk.Label(root, text="Select a video file:")
        self.label.pack()

        self.btn_browse = tk.Button(root, text="Browse", command=self.browse_video)
        self.btn_browse.pack()

        self.btn_classify = tk.Button(root, text="Classify", command=self.classify_video)
        self.btn_classify.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        self.video_path = file_path
        self.label.config(text=f"Selected video: {os.path.basename(self.video_path)}")

    def classify_video(self):
        if hasattr(self, 'video_path'):
            cap = cv2.VideoCapture(self.video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to target size
                frame = cv2.resize(frame, (64, 64))

                # Convert frame to array and normalize pixel values
                frame_array = img_to_array(array_to_img(frame))
                frame_array /= 255.0

                frames.append(frame_array)

            cap.release()

            # Reshape frames to match model input shape
            video_input = np.array(frames).reshape(1, -1, 64, 64, 3)

            # Perform classification
            prediction = self.model.predict(video_input)

            # Display result
            result_text = "Class: Positive" if prediction[0][0] > 0.5 else "Class: Negative"
            self.label.config(text=f"Selected video: {os.path.basename(self.video_path)}\n{result_text}")

            # Display a frame from the video
            self.display_frame(frames[0])
        else:
            self.label.config(text="No video selected. Please browse a video first.")

    def display_frame(self, frame):
        # Convert frame to ImageTk format
        image = Image.fromarray((frame * 255).astype(np.uint8))
        photo = ImageTk.PhotoImage(image=image)

        # Update image label
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClassifierApp(root)
    root.mainloop()
