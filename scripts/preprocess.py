import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        results.append(face)

    return results