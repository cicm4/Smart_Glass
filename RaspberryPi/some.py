import time
import cv2 as cv
import numpy as np
from picamera2 import Picamera2  # Raspberry Pi Camera v2 / libcamera API
from cvzone.FaceMeshModule import FaceMeshDetector

from smart_bt import open_rfcomm          # helper that blocks until a paired host connects
from constants import Image_Constants     # landmark indices & pairs

# ─── Bluetooth: wait for the first trusted host ─────────────────────────
sock = open_rfcomm()                      # no MAC hard‑coding – pair button handles bonding

# ─── Camera set‑up (Picamera2) ──────────────────────────────────────────
picam2 = Picamera2()

# Use a modest 640×480 BGR stream – tweak if you need faster FPS
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    controls={"FrameRate": 30},
)
picam2.configure(config)
picam2.start()

# Give sensor & AGC some time to settle
time.sleep(2)

# ─── FaceMesh detector ─────────────────────────────────────────────────
detector = FaceMeshDetector(maxFaces=1)

# Landmark shortcuts
L_OUT, L_IN, L_UP, L_LO = (
    Image_Constants.LEFT_EYE_OUT_ID,
    Image_Constants.LEFT_EYE_INSIDE_ID,
    Image_Constants.LEFT_EYE_UP_ID,
    Image_Constants.LEFT_EYE_LOW_ID,
)
R_OUT, R_IN, R_UP, R_LO = (
    Image_Constants.RIGHT_EYE_OUT_ID,
    Image_Constants.RIGHT_EYE_INSIDE_ID,
    Image_Constants.RIGHT_EYE_UP_ID,
    Image_Constants.RIGHT_EYE_LOW_ID,
)
L_PAIRS = Image_Constants.LEFT_EYE_PAIR_IDS
R_PAIRS = Image_Constants.RIGHT_EYE_PAIR_IDS

# ─── Feature helpers ───────────────────────────────────────────────────

def eye_metrics(face, out_id, in_id, up_id, lo_id):
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver = detector.findDistance(p_up, p_lo)[0]
    hor = detector.findDistance(p_out, p_in)[0]
    return ver / (hor + 1e-6), ver, hor


def vertical_ratios(face, pairs, out_id, in_id):
    width = detector.findDistance(face[out_id], face[in_id])[0]
    feats = []
    for up_id, lo_id in pairs:
        h = detector.findDistance(face[up_id], face[lo_id])[0]
        feats.append(h / (width + 1e-6))
    return feats, width

# ─── Main loop ─────────────────────────────────────────────────────────
try:
    while True:
        # Capture a single BGR frame as NumPy array
        img = picam2.capture_array()

        # Optionally down‑sample to speed up FaceMesh
        img_small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        img_small, faces = detector.findFaceMesh(img_small, draw=False)

        if faces:
            # Because we resized, scale landmark coordinates back
            face = (faces[0] * 2).astype(int)

            ratio_L, _, _ = eye_metrics(face, L_OUT, L_IN, L_UP, L_LO)
            ratio_R, _, _ = eye_metrics(face, R_OUT, R_IN, R_UP, R_LO)
            verts_L, width_L = vertical_ratios(face, L_PAIRS, L_OUT, L_IN)
            verts_R, width_R = vertical_ratios(face, R_PAIRS, R_OUT, R_IN)

            features = [
                ratio_L,
                ratio_R,
                *verts_L,
                *verts_R,
                width_L,
                width_R,
            ]
            sock.send((",".join(f"{f:.4f}" for f in features) + "\n").encode())

except KeyboardInterrupt:
    pass
finally:
    sock.close()
    picam2.stop()
