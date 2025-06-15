import time
import cv2 as cv
import bluetooth
from picamera import PiCamera
from picamera.array import PiRGBArray
from cvzone.FaceMeshModule import FaceMeshDetector
import constants

BT_ADDR = "B8:27:EB:C2:A5:45"  # adjust for your receiver

L_IDS = (
    constants.Image_Constants.LEFT_EYE_OUT_ID,
    constants.Image_Constants.LEFT_EYE_INSIDE_ID,
    constants.Image_Constants.LEFT_EYE_UP_ID,
    constants.Image_Constants.LEFT_EYE_LOW_ID,
)
R_IDS = (
    constants.Image_Constants.RIGHT_EYE_OUT_ID,
    constants.Image_Constants.RIGHT_EYE_INSIDE_ID,
    constants.Image_Constants.RIGHT_EYE_UP_ID,
    constants.Image_Constants.RIGHT_EYE_LOW_ID,
)

def eye_ratio(face, ids, det):
    p_out, p_in, p_up, p_lo = [face[i] for i in ids]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    return ver / (hor + 1e-6)

# --- setup camera ---
camera = PiCamera()
# use half resolution to avoid warped capture
width, height = camera.resolution
camera.resolution = (width // 2, height // 2)
raw = PiRGBArray(camera, size=camera.resolution)

# --- bluetooth ---
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((BT_ADDR, 1))

# --- face mesh ---
detector = FaceMeshDetector(maxFaces=1)

time.sleep(2.0)

try:
    for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
        img = frame.array
        raw.truncate(0)
        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            ratio_L = eye_ratio(face, L_IDS, detector)
            ratio_R = eye_ratio(face, R_IDS, detector)
            msg = f"{ratio_L:.4f},{ratio_R:.4f}\n"
            sock.send(msg.encode())
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    sock.close()
    camera.close()

