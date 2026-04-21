from tensorflow.keras.models import load_model
from uuid import uuid4 as UUID
import numpy as np
import cv2
import os

# Load Haarcascade (FIXED PATH)
haar_path = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(haar_path)

def extract_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for _, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        results.append((face, (x, y, w, h)))

    return results


model = load_model("../results/model/final_emotion_model.keras")

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

OUTPUT_DIR = "../results/preprocessing_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_video(video_path):
    i = 0
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    frame_count = 0

    print("Reading video stream ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:

            faces = extract_faces(frame)

            for (face, (x, y, w, h)) in faces:
                face = face / 255.0
                face = np.expand_dims(face, axis=(0, -1))

                pred = model.predict(face, verbose=0)
                emotion = labels[np.argmax(pred)]
                conf = np.max(pred)

                filename = f"{"Image"}_{i}.png"
                i+=1;
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), face[0, :, :, 0] * 255)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

read_video("../results/preprocessing_test/input.mp4")