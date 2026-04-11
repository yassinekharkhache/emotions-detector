import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import extract_faces
import time

model = load_model("../results/model/final_emotion_model.keras")

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

cap = cv2.VideoCapture(0)

print("Reading video stream ...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = extract_faces(frame)

    for face in faces:
        face = face / 255.0
        face = np.expand_dims(face, axis=(0,-1))

        pred = model.predict(face, verbose=0)
        emotion = labels[np.argmax(pred)]
        conf = np.max(pred)

        print(f"{time.strftime('%H:%M:%S')} : {emotion} , {int(conf*100)}%")
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()