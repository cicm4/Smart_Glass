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

def eye_ratio(face, ids, det):
    p_out, p_in, p_up, p_lo = [face[i] for i in ids]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    return ver / (hor + 1e-6)

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
            ratio_L = eye_ratio(face, L_IDS, detector)
            ratio_R = eye_ratio(face, R_IDS, detector)
            row = dict(
                timestamp=timestamp,
                ratio_left=ratio_L,
                ratio_right=ratio_R,
                blink_count=blink_count,
                manual_blink=manual_blink,
            )
            data_rows.append(row)
            plot_img = plot.update((ratio_L + ratio_R) / 2)
            stack = cvz.stackImages([cv.resize(img, (640, 360)), plot_img], 2, 1)
            cv.imshow("Blink Recorder", stack)
        else:
            data_rows.append(
                dict(
                    timestamp=timestamp,
                    ratio_left=None,
                    ratio_right=None,
                    blink_count=blink_count,
                    manual_blink=manual_blink,
                )
            )
            cv.imshow("Blink Recorder", cv.resize(img, (640, 360)))

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

