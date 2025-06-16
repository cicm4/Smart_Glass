import time
import cv2 as cv
import bluetooth
from picamera import PiCamera
from picamera.array import PiRGBArray
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

from constants import Image_Constants

BT_ADDR = "B8:27:EB:C2:A5:45"  # adjust for your receiver

L_OUT = Image_Constants.LEFT_EYE_OUT_ID
L_IN  = Image_Constants.LEFT_EYE_INSIDE_ID
L_UP  = Image_Constants.LEFT_EYE_UP_ID
L_LO  = Image_Constants.LEFT_EYE_LOW_ID

R_OUT = Image_Constants.RIGHT_EYE_OUT_ID
R_IN  = Image_Constants.RIGHT_EYE_INSIDE_ID
R_UP  = Image_Constants.RIGHT_EYE_UP_ID
R_LO  = Image_Constants.RIGHT_EYE_LOW_ID

L_PAIRS = Image_Constants.LEFT_EYE_PAIR_IDS
R_PAIRS = Image_Constants.RIGHT_EYE_PAIR_IDS

def eye_metrics(face, out_id, in_id, up_id, lo_id, det):
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    ratio = ver / (hor + 1e-6)
    return ratio, ver, hor

def vertical_ratios(face, pairs, out_id, in_id, det):
    width, _ = det.findDistance(face[out_id], face[in_id])
    feats = []
    for up_id, lo_id in pairs:
        h, _ = det.findDistance(face[up_id], face[lo_id])
        feats.append(h / (width + 1e-6))
    return feats, width

# ─── setup camera ──────────────────────────────────────────────────────
camera = PiCamera()
width, height = camera.resolution
camera.resolution = (width // 2, height // 2)
raw = PiRGBArray(camera, size=camera.resolution)

# ─── bluetooth & face mesh ─────────────────────────────────────────────
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((BT_ADDR, 1))

detector = FaceMeshDetector(maxFaces=1)

# give camera some time to warm up
time.sleep(2.0)

try:
    for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
        img = frame.array
        raw.truncate(0)
        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            ratio_L, _, _ = eye_metrics(face, L_OUT, L_IN, L_UP, L_LO, detector)
            ratio_R, _, _ = eye_metrics(face, R_OUT, R_IN, R_UP, R_LO, detector)
            verts_L, width_L = vertical_ratios(face, L_PAIRS, L_OUT, L_IN, detector)
            verts_R, width_R = vertical_ratios(face, R_PAIRS, R_OUT, R_IN, detector)
            num_feats = [
                ratio_L,
                ratio_R,
                *verts_L,
                *verts_R,
                width_L,
                width_R,
            ]

            msg = ",".join(f"{f:.4f}" for f in num_feats) + "\n"
            sock.send(msg.encode())
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    sock.close()
    camera.close()
