from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

app = Flask(__name__)

# Load the trained model
model = load_model('/path/to/save/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    video_path = request.form['video_path']

    cap = cv2.VideoCapture(video_path)
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
    prediction = model.predict(video_input)

    # Display result
    result_text = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return render_template('result.html', result=result_text, video_path=os.path.basename(video_path))

if __name__ == '__main__':
    app.run(debug=True)
