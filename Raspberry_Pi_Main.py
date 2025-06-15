import time
import cv2 as cv
import numpy as np
import bluetooth
from picamera import PiCamera
from picamera.array import PiRGBArray
from cvzone.FaceMeshModule import FaceMeshDetector
import constants

# Bluetooth server setup
server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", 1))
server_sock.listen(1)
print("Waiting for Bluetooth connection...")
client_sock, addr = server_sock.accept()
print("Connected to", addr)

# PiCamera setup
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=camera.resolution)
time.sleep(0.1)

detector = FaceMeshDetector(maxFaces=1)

# helper to compute ratio
def eye_ratio(face, out_id, in_id, up_id, lo_id):
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver, _ = detector.findDistance(p_up, p_lo)
    hor, _ = detector.findDistance(p_out, p_in)
    return ver / hor if hor else 0.0

try:
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        img = frame.array
        img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            ratio_L = eye_ratio(
                face,
                constants.Image_Constants.LEFT_EYE_OUT_ID,
                constants.Image_Constants.LEFT_EYE_INSIDE_ID,
                constants.Image_Constants.LEFT_EYE_UP_ID,
                constants.Image_Constants.LEFT_EYE_LOW_ID,
            )
            ratio_R = eye_ratio(
                face,
                constants.Image_Constants.RIGHT_EYE_OUT_ID,
                constants.Image_Constants.RIGHT_EYE_INSIDE_ID,
                constants.Image_Constants.RIGHT_EYE_UP_ID,
                constants.Image_Constants.RIGHT_EYE_LOW_ID,
            )
            msg = f"{ratio_L:.4f},{ratio_R:.4f}\n"
            client_sock.send(msg.encode())
        raw_capture.truncate(0)
except KeyboardInterrupt:
    pass
finally:
    client_sock.close()
    server_sock.close()
