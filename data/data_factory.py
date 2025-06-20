import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2 as cv
import cvzone as cvz
import numpy as np
import pandas as pd
import keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import constants

CSV_NAME = f"blink_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

cap = cv.VideoCapture(0)
# fix camera warping by halving default resolution
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) / 2)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

BLINK_PRE = constants.Data_Gathering_Constants.BLINK_PRE_FRAMES
BLINK_POST = constants.Data_Gathering_Constants.BLINK_POST_FRAMES

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
L_PAIRS = constants.Image_Constants.LEFT_EYE_PAIR_IDS
R_PAIRS = constants.Image_Constants.RIGHT_EYE_PAIR_IDS

def vertical_ratios(face, pairs, out_id, in_id, det):
    """Return list of vertical/width ratios and eye width."""
    p_out, p_in = face[out_id], face[in_id]
    width, _ = det.findDistance(p_out, p_in)
    feats = []
    for up_id, lo_id in pairs:
        h, _ = det.findDistance(face[up_id], face[lo_id])
        feats.append(h / (width + 1e-6))
    return feats, width

def eye_metrics(face, ids, det):
    """Return EAR ratio along with vertical and horizontal distances."""
    p_out, p_in, p_up, p_lo = [face[i] for i in ids]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    ratio = ver / (hor + 1e-6)
    return ratio, ver, hor

detector = FaceMeshDetector(maxFaces=1)
plot = LivePlot(640, 360, [0, 0.5])

data_rows = []
blink_count = 0
post_ctr = 0
prev_key = False
t0 = time.time()

try:
    while True:
        ok, img = cap.read()
        if not ok:
            break

        img, faces = detector.findFaceMesh(img, draw=False)
        key_now = keyboard.is_pressed("space")

        if key_now and not prev_key:
            blink_count += 1
            post_ctr = BLINK_POST
            cv.putText(
              img,
              "Blink detected!",
              (50, 100),
              cv.FONT_HERSHEY_SIMPLEX,
              2,
              (0, 0, 255),
              4,
              cv.LINE_AA,
            )
            for i in range(1, BLINK_PRE + 1):
                if len(data_rows) >= i:
                    data_rows[-i]["manual_blink"] = 1

        manual_blink = int(key_now or post_ctr > 0)
        if post_ctr:
            post_ctr -= 1

        timestamp = time.time() - t0

        if faces:
            face = faces[0]
            for pid in constants.Image_Constants.ID_ARRAYS:
                cv.circle(img, face[pid], 3, (255, 0, 255), cv.FILLED)
            ratio_L, _, _ = eye_metrics(face, L_IDS, detector)
            ratio_R, _, _ = eye_metrics(face, R_IDS, detector)
            verts_L, width_L = vertical_ratios(
                face, L_PAIRS, constants.Image_Constants.LEFT_EYE_OUT_ID,
                constants.Image_Constants.LEFT_EYE_INSIDE_ID, detector
            )
            verts_R, width_R = vertical_ratios(
                face, R_PAIRS, constants.Image_Constants.RIGHT_EYE_OUT_ID,
                constants.Image_Constants.RIGHT_EYE_INSIDE_ID, detector
            )
            row = {
                "timestamp": timestamp,
                "ratio_left": ratio_L,
                "ratio_right": ratio_R,
                **{f"v{i+1}_left": v for i, v in enumerate(verts_L)},
                **{f"v{i+1}_right": v for i, v in enumerate(verts_R)},
                "width_left": width_L,
                "width_right": width_R,
                "blink_count": blink_count,
                "manual_blink": manual_blink,
            }
            data_rows.append(row)
            plot_img = plot.update((ratio_L + ratio_R) / 2)
            stack = cvz.stackImages([img, plot_img], 2, 1)
            cv.imshow("Blink Recorder", stack)
        else:
            none_row = {
                "timestamp": timestamp,
                "ratio_left": None,
                "ratio_right": None,
                **{f"v{i+1}_left": None for i in range(len(L_PAIRS))},
                **{f"v{i+1}_right": None for i in range(len(R_PAIRS))},
                "width_left": None,
                "width_right": None,
                "blink_count": blink_count,
                "manual_blink": manual_blink,
            }
            data_rows.append(none_row)
            cv.imshow("Blink Recorder", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        prev_key = key_now
finally:
    if data_rows:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(data_rows).to_csv(os.path.join("data", CSV_NAME), index=False)
        print(f"Saved {len(data_rows)} rows → {CSV_NAME}")
    cap.release()
    cv.destroyAllWindows()

