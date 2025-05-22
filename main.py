import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
cap = cv.VideoCapture(0)

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_eye.xml")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame

    face = face_classifier.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

    eye = eye_classifier.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for (x, y, w, h) in eye:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()