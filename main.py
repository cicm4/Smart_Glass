# Video Blink Detection Test – NEW INPUT PIPELINE for BlinkDetector
# ---------------------------------------------------------------
import time
import cv2 as cv
import cvzone as cvz
import numpy as np
import torch
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

from constants import (
    MODEL_WEIGHTS,
    STATS_NPZ,
    BLINKING_THREASHOLD,
    Image_Constants,
    Training_Constnats,
    Model_Constants,
)
from Model.model import BlinkDetectorXS as model  # numeric-only model

# ───────── configuration ──────────────────────────────────────
SEQ_LEN = Training_Constnats.SEQUENCE_LENGTH
NUM_FEATURES = Model_Constants.NUM_FEATURES

THRESH = BLINKING_THREASHOLD
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MediaPipe landmark IDs pulled from Image_Constants
L_OUT = Image_Constants.LEFT_EYE_OUT_ID
L_IN = Image_Constants.LEFT_EYE_INSIDE_ID
L_UP = Image_Constants.LEFT_EYE_UP_ID
L_LO = Image_Constants.LEFT_EYE_LOW_ID

R_OUT = Image_Constants.RIGHT_EYE_OUT_ID
R_IN = Image_Constants.RIGHT_EYE_INSIDE_ID
R_UP = Image_Constants.RIGHT_EYE_UP_ID
R_LO = Image_Constants.RIGHT_EYE_LOW_ID

POINTS_USED = Image_Constants.ID_ARRAYS
# ──────────────────────────────────────────────────────────────

# ---------- helper functions (copied from collector) ----------

def ear(face, out_id, in_id, up_id, lo_id, det):
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    return ver / hor, ver, hor


# --------------------------------------------------------------

# ---------- load model & feature stats ------------------------
model = model().to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

stats = np.load(STATS_NPZ)
MEAN, STD = stats["mean"], stats["std"]  # (NUM_FEATURES,) each
# --------------------------------------------------------------

# ---------- runtime buffers -----------------------------------
num_buf = []  # hold SEQ_LEN frames
blink_count = 0
prev_pred = 0
# --------------------------------------------------------------

# ---------- OpenCV / MediaPipe init ---------------------------
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

detector = FaceMeshDetector(maxFaces=1)
plot_y = LivePlot(640, 360, [1, 4])
t0 = time.time()
# --------------------------------------------------------------

while True:
    ok, img = cap.read()
    if not ok:
        break

    img, faces = detector.findFaceMesh(img, draw=False)
    timestamp = time.time() - t0

    if faces:
        face = faces[0]

        # draw landmarks (optional)
        for pid in POINTS_USED:
            cv.circle(img, face[pid], 3, (255, 0, 255), cv.FILLED)

        # ── numeric features
        ratio_L, _, _ = ear(face, L_OUT, L_IN, L_UP, L_LO, detector)
        ratio_R, _, _ = ear(face, R_OUT, R_IN, R_UP, R_LO, detector)
        ratio_avg = (ratio_L + ratio_R) / 2
        num_feats = np.array([ratio_L, ratio_R], dtype=np.float32)

        # add to buffers
        num_buf.append(num_feats)
        if len(num_buf) > SEQ_LEN:
            num_buf.pop(0)

        # run model when window full
        if len(num_buf) == SEQ_LEN:
            num_arr = (np.stack(num_buf) - MEAN) / STD  # z‑score

            num_t = torch.from_numpy(num_arr)[None].to(DEVICE)  # (1,SEQ_LEN,NUM_FEATURES)

            with torch.no_grad():
                p = torch.sigmoid(model(num_t)).item()

            pred = int(p > THRESH)
            if pred == 1 and prev_pred == 0:  # count rising edge
                blink_count += 1
            prev_pred = pred

            cv.putText(
                img,
                f"Model: {'Blink' if pred else 'No blink'}  p={p:.2f}",
                (20, 95),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if pred else (0, 255, 0),
                2,
            )

        # HUD
        cv.putText(
            img,
            f"Blink #: {blink_count}",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv.putText(
            img,
            f"EAR L/R: {ratio_L:.2f}/{ratio_R:.2f}",
            (20, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # plot & show
        img_plot = plot_y.update(ratio_avg)
        img = cv.resize(img, (640, 360))
        img_stack = cvz.stackImages([img, img_plot], 2, 1)
        cv.imshow("BlinkDetector", img_stack)
    else:
        cv.imshow("BlinkDetector", cv.resize(img, (640, 360)))

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
