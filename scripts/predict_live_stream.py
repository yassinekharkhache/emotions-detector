import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

# Load model
model = load_model("../results/model/final_emotion_model.keras")

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load Haarcascade (FIXED PATH)
haar_path = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(haar_path)

# Open camera
cap = cv2.VideoCapture(0)

print("Reading video stream ...")

while True:
    ret, frame = cap.read()
    if not ret:
        print(":x: Camera not working")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Extract face
        face = gray[y:y+h, x:x+w]

        # Resize to model input
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face / 255.0

        # Reshape for model (1, 48, 48, 1)
        face = np.expand_dims(face, axis=(0, -1))

        # Predict
        pred = model.predict(face, verbose=0)
        emotion = labels[np.argmax(pred)]
        conf = np.max(pred)

        # Print result
        print(f"{time.strftime('%H:%M:%S')} : {emotion} , {int(conf*100)}%")

        # Draw on screen (optional but useful)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show video
    cv2.imshow("Emotion Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()